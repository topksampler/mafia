"""Qwen 4B LoRA REINFORCE for Detective reveal/hold decisions.

The language model acts on every eligible Detective speech turn. Its input is
the acting player's observation; all other actions use the scripted controls.
No training seed is reused for held-out evaluation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from env.rules import GameConfig, MafiaEnv, RULESET_VERSION
from src.training.reveal_choice import (
    RevealChoice,
    RevealOpportunity,
    RevealPolicy,
    choose_reveal_action,
    eligible_opportunity,
)
from src.training.scripted_reveal import CLAIM_PATTERN, ScriptedPolicy
from src.training.vote import MODEL_ID


TRAIN_ARMS = ("counterclaim", "bluff")
SYSTEM = (
    "You are the Detective in Mafia. Town wins by eliminating the hidden Mafia. "
    "At this speaking turn, decide whether announcing your confirmed Mafia "
    "investigation now or staying silent gives Town the best chance to win. "
    "Both actions use this turn. You may decide again on a later turn. "
    "Use only your own private result and the public conversation. "
    "Answer with exactly Yes or No, using the action mapping in the next message."
)


@dataclass(frozen=True)
class PromptDecision:
    yes_reveal_ids: tuple[int, ...]
    yes_hold_ids: tuple[int, ...]
    choice: RevealChoice


@dataclass(frozen=True)
class Episode:
    seed: int
    arm: str
    winner: str | None
    reward: float
    attempts: int
    eligible_choices: int
    reveals: int
    choice_rows: tuple[dict[str, Any], ...]
    model_decisions: tuple[PromptDecision, ...]
    frames: tuple[dict[str, Any], ...]
    players: tuple[str, ...]
    roles: dict[str, str]


def reveal_messages(
    observation: RevealOpportunity,
    public_events: list[dict[str, Any]],
    *,
    yes_means_reveal: bool,
) -> list[dict[str, str]]:
    """Two swapped action mappings cancel simple Yes/No token preference."""
    state = {
        "you": observation.player,
        "your_role": "detective",
        "day": observation.day,
        "discussion_round": observation.discussion_round,
        "alive": observation.alive,
        "your_private_events": observation.private_events,
        "public_events": public_events,
        "confirmed_mafia_target": observation.confirmed_target,
        "actions": {
            "Yes": "REVEAL" if yes_means_reveal else "HOLD",
            "No": "HOLD" if yes_means_reveal else "REVEAL",
        },
    }
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": json.dumps(state, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def encode_reveal(
    observation: RevealOpportunity, tokenizer: Any, max_prompt_tokens: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    events = list(observation.public_events)
    while True:
        pairs = tuple(
            tuple(tokenizer.apply_chat_template(
                reveal_messages(observation, events, yes_means_reveal=mapping),
                tokenize=True,
                add_generation_prompt=True,
            ))
            for mapping in (True, False)
        )
        if max(map(len, pairs)) <= max_prompt_tokens:
            return pairs[0], pairs[1]
        if not events:
            raise ValueError("Reveal prompt exceeds token limit without public events")
        events = events[1:]


class QwenRevealPolicy:
    def __init__(self, model: Any, tokenizer: Any, max_prompt_tokens: int = 1536) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.device = next(model.parameters()).device
        self.torch = torch
        self.answer_tokens: dict[str, int] = {}
        for answer in ("Yes", "No"):
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Expected one token for {answer}, got {ids}")
            self.answer_tokens[answer] = ids[0]
        self.samples: list[PromptDecision] = []

    def start_episode(self) -> None:
        self.samples.clear()

    def _answer_margin(self, ids: tuple[int, ...]) -> Any:
        inputs = self.torch.tensor([ids], device=self.device)
        logits = self.model(input_ids=inputs).logits[0, -1].float()
        return logits[self.answer_tokens["Yes"]] - logits[self.answer_tokens["No"]]

    def _distribution(
        self, yes_reveal_ids: tuple[int, ...], yes_hold_ids: tuple[int, ...]
    ) -> Any:
        # The second mapping reverses which answer token means Reveal.
        reveal_margin = (
            self._answer_margin(yes_reveal_ids)
            - self._answer_margin(yes_hold_ids)
        ) / 2
        logits = self.torch.stack((self.torch.zeros_like(reveal_margin), reveal_margin))
        return self.torch.distributions.Categorical(logits=logits)

    def choose(self, observation: RevealOpportunity, *, sample: bool) -> RevealChoice:
        yes_reveal_ids, yes_hold_ids = encode_reveal(
            observation, self.tokenizer, self.max_prompt_tokens
        )
        with self.torch.inference_mode():
            distribution = self._distribution(yes_reveal_ids, yes_hold_ids)
            self.last_prob_reveal = float(distribution.probs[1].item())
            index = int(
                distribution.sample().item() if sample
                else distribution.probs.argmax().item()
            )
        choice = RevealChoice.REVEAL if index == 1 else RevealChoice.HOLD
        if sample:
            self.samples.append(PromptDecision(yes_reveal_ids, yes_hold_ids, choice))
        return choice

    def log_prob_and_entropy(self, decision: PromptDecision) -> tuple[Any, Any]:
        distribution = self._distribution(decision.yes_reveal_ids, decision.yes_hold_ids)
        index = 1 if decision.choice == RevealChoice.REVEAL else 0
        return (
            distribution.log_prob(self.torch.tensor(index, device=self.device)),
            distribution.entropy(),
        )


class FixedRevealPolicy:
    """A named evaluation control, not a training policy."""

    def __init__(self, choice: RevealChoice) -> None:
        self.choice = choice
        self.last_prob_reveal = 1.0 if choice == RevealChoice.REVEAL else 0.0

    def choose(self, observation: RevealOpportunity, *, sample: bool) -> RevealChoice:
        return self.choice


def play_game(
    seed: int,
    arm: str,
    policy: RevealPolicy,
    *,
    sample: bool,
    record_trace: bool = False,
    config: GameConfig | None = None,
) -> Episode:
    """Route only eligible Detective speech decisions to the supplied policy."""
    if arm not in TRAIN_ARMS:
        raise ValueError(f"Unsupported opponent arm: {arm}")
    if hasattr(policy, "start_episode"):
        policy.start_episode()
    environment = MafiaEnv(config)
    environment.reset(seed=seed)
    opponents = ScriptedPolicy(arm, seed)
    frames: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    attempts = 0
    while environment.current_player is not None:
        attempts += 1
        if attempts > 300:
            raise RuntimeError(f"Game {seed} ({arm}) exceeded 300 actions")
        view = environment.observe(environment.current_player)
        opportunity = eligible_opportunity(view)
        p_reveal: float | None = None
        if opportunity is not None:
            decision = choose_reveal_action(view, policy, sample=sample)
            action = decision.action
            reason = f"qwen_{decision.choice.value}"
            p_reveal = getattr(policy, "last_prob_reveal", None)
            prior_claims = sum(
                event.get("type") == "speech"
                and CLAIM_PATTERN.fullmatch(event.get("message", "")) is not None
                for event in view["public_events"]
            )
            choices.append({
                "day": view["day"],
                "round": view["discussion_round"],
                "choice": decision.choice.value,
                "p_reveal": p_reveal,
                "p_hold": (1.0 - p_reveal) if p_reveal is not None else None,
                "target": opportunity.confirmed_target,
                "alive_count": len(view["alive"]),
                "prior_public_claims": prior_claims,
            })
        else:
            action, reason, _ = opponents.act(view)
        before = len(view["public_events"])
        result = environment.step(action)
        if not result.accepted:
            raise RuntimeError(f"Controller made an illegal action: {seed} {arm} {reason}")
        if record_trace:
            frames.append({
                "step": attempts,
                "day": view["day"],
                "phase": view["phase"],
                "round": view["discussion_round"],
                "actor": view["player"],
                "actor_role": view["role"],
                "alive": view["alive"],
                "action": action["type"],
                "target": action.get("target"),
                "message": action.get("message", ""),
                "accepted": True,
                "forfeited": False,
                "referee_message": None,
                "private": view["private_events"],
                "public_delta": environment.public_events[before:],
                "reason": reason,
                "p_reveal": p_reveal,
                "p_hold": (1.0 - p_reveal) if p_reveal is not None else None,
            })
    winner = environment.winner
    samples = tuple(getattr(policy, "samples", ()))
    if sample and len(samples) != len(choices):
        raise RuntimeError("A sampled Qwen decision was not saved for each eligible turn")
    return Episode(
        seed=seed,
        arm=arm,
        winner=winner,
        reward=1.0 if winner == "town" else -1.0 if winner == "mafia" else 0.0,
        attempts=attempts,
        eligible_choices=len(choices),
        reveals=sum(row["choice"] == RevealChoice.REVEAL.value for row in choices),
        choice_rows=tuple(choices),
        model_decisions=samples,
        frames=tuple(frames),
        players=environment.players,
        roles={player: role.value for player, role in environment.roles.items()},
    )


def episode_row(episode: Episode) -> dict[str, Any]:
    return {
        "seed": episode.seed,
        "arm": episode.arm,
        "winner": episode.winner,
        "reward": episode.reward,
        "eligible_choices": episode.eligible_choices,
        "reveals": episode.reveals,
        "choices": list(episode.choice_rows),
    }


def evaluate(
    policy: RevealPolicy,
    seeds: range,
    *,
    record_trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    traces = []
    for seed in seeds:
        for arm in TRAIN_ARMS:
            episode = play_game(seed, arm, policy, sample=False, record_trace=record_trace)
            rows.append(episode_row(episode))
            if record_trace:
                traces.append({
                    "seed": seed,
                    "arm": arm,
                    "model": "qwen_reveal_hold",
                    "winner": episode.winner,
                    "attempts": episode.attempts,
                    "players": episode.players,
                    "roles": episode.roles,
                    "frames": episode.frames,
                })
    by_arm = {}
    for arm in TRAIN_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "games": len(arm_rows),
            "town_wins": sum(row["winner"] == "town" for row in arm_rows),
            "town_win_rate": sum(row["winner"] == "town" for row in arm_rows) / len(arm_rows),
            "eligible_games": sum(row["eligible_choices"] > 0 for row in arm_rows),
            "eligible_choices": sum(row["eligible_choices"] for row in arm_rows),
            "reveals": sum(row["reveals"] for row in arm_rows),
        }
    return {
        "games": len(rows),
        "town_wins": sum(row["winner"] == "town" for row in rows),
        "town_win_rate": sum(row["winner"] == "town" for row in rows) / len(rows),
        "by_arm": by_arm,
        "per_game": rows,
    }, traces


def train(
    *,
    output_dir: str,
    dashboard_url: str,
    updates: int = 4,
    eligible_games_per_update: int = 4,
    max_sampled_games_per_update: int = 60,
    eval_games: int = 24,
    eval_interval: int = 2,
    train_start_seed: int = 80_000,
    eval_start_seed: int = 90_000,
    learning_rate: float = 1e-4,
    entropy_coef: float = 0.01,
    publish: Callable[[str, dict[str, Any]], None] | None = None,
    flush_metrics: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Train a LoRA reveal policy with complete-game Town rewards."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if min(
        updates, eligible_games_per_update, max_sampled_games_per_update,
        eval_games, eval_interval,
    ) < 1:
        raise ValueError("Training and evaluation counts must be positive")
    if train_start_seed <= eval_start_seed < train_start_seed + updates * max_sampled_games_per_update:
        raise ValueError("Training seed range could overlap held-out evaluation")
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    started = monotonic()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    policy = QwenRevealPolicy(model, tokenizer)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found")
    tracked_weight = next(
        parameter for name, parameter in model.named_parameters() if "lora_B" in name
    )
    initial_weight = tracked_weight.detach().float().clone()
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    eval_seeds = range(eval_start_seed, eval_start_seed + eval_games)
    fixed_hold, _ = evaluate(FixedRevealPolicy(RevealChoice.HOLD), eval_seeds)
    fixed_reveal, _ = evaluate(FixedRevealPolicy(RevealChoice.REVEAL), eval_seeds)
    run_name = "detective-reveal-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    config = {
        "model": MODEL_ID,
        "ruleset": RULESET_VERSION,
        "role": "detective",
        "team": "town",
        "action_family": "reveal_or_hold_on_every_eligible_speech",
        "algorithm": "episodic_reinforce_with_ema_baseline",
        "opponent_arms": TRAIN_ARMS,
        "updates": updates,
        "eligible_games_per_update": eligible_games_per_update,
        "max_sampled_games_per_update": max_sampled_games_per_update,
        "eval_interval": eval_interval,
        "train_start_seed": train_start_seed,
        "eval_seeds": [eval_seeds.start, eval_seeds.stop - 1],
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "max_prompt_tokens": policy.max_prompt_tokens,
        "fixed_hold_town_win_rate": fixed_hold["town_win_rate"],
        "fixed_reveal_town_win_rate": fixed_reveal["town_win_rate"],
        "torch_version": str(torch.__version__),
    }
    writer = SummaryWriter(log_dir=str(output / "tb" / run_name), flush_secs=5)
    writer.add_text("run/config", json.dumps(config, indent=2), global_step=0)
    print(f"Dashboard: {dashboard_url}; run: {run_name}", flush=True)

    def log_scalars(metrics: dict[str, float | int], step: int) -> None:
        for name, value in metrics.items():
            writer.add_scalar(name, value, global_step=step)
        writer.flush()
        if flush_metrics is not None:
            flush_metrics()

    def eval_metrics(result: dict[str, Any]) -> dict[str, float | int]:
        return {
            "eval/town_win_rate": result["town_win_rate"],
            "eval/counterclaim_town_win_rate": result["by_arm"]["counterclaim"]["town_win_rate"],
            "eval/bluff_town_win_rate": result["by_arm"]["bluff"]["town_win_rate"],
            "eval/eligible_games": sum(arm["eligible_games"] for arm in result["by_arm"].values()),
            "eval/reveals": sum(arm["reveals"] for arm in result["by_arm"].values()),
            "eval/fixed_hold_town_win_rate": fixed_hold["town_win_rate"],
            "eval/fixed_reveal_town_win_rate": fixed_reveal["town_win_rate"],
        }

    sampled_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    recent_rewards: deque[float] = deque(maxlen=20)
    games_seen = 0
    next_seed = train_start_seed
    baseline = 0.0
    final_eval: dict[str, Any] | None = None
    final_traces: list[dict[str, Any]] = []
    try:
        model.eval()
        initial_eval, initial_traces = evaluate(policy, eval_seeds, record_trace=True)
        initial_metrics = {
            **eval_metrics(initial_eval),
            "elapsed_minutes": (monotonic() - started) / 60,
        }
        log_scalars(initial_metrics, step=0)
        if publish is not None:
            publish(run_name, {
                "kind": "initial", "step": 0, "config": config, "metrics": initial_metrics,
            })
        print(
            f"Before training: Town {initial_eval['town_wins']}/{initial_eval['games']} "
            f"across two opponent arms",
            flush=True,
        )

        for update in range(1, updates + 1):
            model.eval()
            eligible_episodes: list[Episode] = []
            sampled_in_update: list[Episode] = []
            while (
                len(eligible_episodes) < eligible_games_per_update
                and len(sampled_in_update) < max_sampled_games_per_update
            ):
                seed = next_seed
                next_seed += 1
                arm = TRAIN_ARMS[(seed - train_start_seed) % len(TRAIN_ARMS)]
                episode = play_game(seed, arm, policy, sample=True)
                sampled_in_update.append(episode)
                sampled_rows.append(episode_row(episode))
                games_seen += 1
                recent_rewards.append(episode.reward)
                if episode.eligible_choices:
                    eligible_episodes.append(episode)
                game_metrics = {
                    "train/game_reward": episode.reward,
                    "train/reward_rolling_20_games": sum(recent_rewards) / len(recent_rewards),
                    "train/game_had_eligible_choice": int(episode.eligible_choices > 0),
                    "train/game_reveal_choices": episode.reveals,
                }
                for name, value in game_metrics.items():
                    writer.add_scalar(name, value, global_step=games_seen)
                writer.flush()
                if publish is not None:
                    publish(run_name, {
                        "kind": "game", "step": games_seen, "metrics": game_metrics,
                    })
            if len(eligible_episodes) < eligible_games_per_update:
                raise RuntimeError(
                    f"Update {update}: only {len(eligible_episodes)} eligible games "
                    f"among {len(sampled_in_update)} sampled; all seeds recorded"
                )
            advantages = [episode.reward - baseline for episode in eligible_episodes]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = entropy_total = 0.0
            choice_count = sum(episode.eligible_choices for episode in eligible_episodes)
            for episode, advantage in zip(eligible_episodes, advantages):
                scale = 1 / (len(eligible_episodes) * episode.eligible_choices)
                for decision in episode.model_decisions:
                    log_prob, entropy = policy.log_prob_and_entropy(decision)
                    loss = -advantage * scale * log_prob - entropy_coef * scale * entropy
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite reveal loss at update {update}")
                    loss.backward()
                    loss_total += float(loss.detach())
                    entropy_total += float(entropy.detach())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite reveal gradient at update {update}")
            optimizer.step()
            mean_reward = sum(episode.reward for episode in eligible_episodes) / len(eligible_episodes)
            baseline = 0.9 * baseline + 0.1 * mean_reward
            changed = float((tracked_weight.detach().float() - initial_weight).abs().max())
            metrics: dict[str, float | int] = {
                "train/sample_count": len(sampled_in_update),
                "train/eligible_games": len(eligible_episodes),
                "train/eligible_choices": choice_count,
                "train/reveal_choices": sum(episode.reveals for episode in eligible_episodes),
                "train/eligible_town_win_rate": sum(
                    episode.winner == "town" for episode in eligible_episodes
                ) / len(eligible_episodes),
                "train/all_sampled_town_win_rate": sum(
                    episode.winner == "town" for episode in sampled_in_update
                ) / len(sampled_in_update),
                "train/mean_reward": mean_reward,
                "train/policy_loss": loss_total,
                "train/mean_action_entropy": entropy_total / choice_count,
                "train/gradient_norm": grad_norm,
                "train/weight_change_max": changed,
                "train/learning_rate": learning_rate,
                "elapsed_minutes": (monotonic() - started) / 60,
            }
            if update % eval_interval == 0 or update == updates:
                model.eval()
                result, traces = evaluate(
                    policy, eval_seeds, record_trace=(update == updates)
                )
                metrics.update(eval_metrics(result))
                if update == updates:
                    final_eval, final_traces = result, traces
            log_scalars(metrics, step=games_seen)
            if publish is not None:
                publish(run_name, {
                    "kind": "update", "step": games_seen, "metrics": metrics,
                })
            history.append({"update": update, "games_seen": games_seen, **metrics})
            print(json.dumps({"update": update, **metrics}), flush=True)

        if changed <= 0:
            raise RuntimeError("LoRA reveal weights did not change")
        if final_eval is None:
            raise RuntimeError("Final held-out evaluation was not recorded")
        run_dir = output / "runs" / run_name
        adapter_path = run_dir / "adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        with (run_dir / "eval_games.jsonl").open("w", encoding="utf-8") as stream:
            for stage, traces in (("before", initial_traces), ("after", final_traces)):
                for trace in traces:
                    trace["stage"] = stage
                    stream.write(json.dumps(trace, ensure_ascii=False, default=list) + "\n")
        report = {
            "config": config,
            "run_name": run_name,
            "dashboard": dashboard_url,
            "adapter_path": str(adapter_path),
            "initial_eval": initial_eval,
            "final_eval": final_eval,
            "fixed_hold_eval": fixed_hold,
            "fixed_reveal_eval": fixed_reveal,
            "history": history,
            "sampled_games": sampled_rows,
            "total_sampled_games": games_seen,
            "total_eligible_training_games": sum(row["eligible_choices"] > 0 for row in sampled_rows),
            "eval_traces_path": str(run_dir / "eval_games.jsonl"),
        }
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return report
    finally:
        writer.close()
