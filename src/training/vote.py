"""First RL experiment: Qwen learns Town vote targets from complete Mafia games.

Only Town votes use the trainable model. All other decisions use seeded random
policies. A legal-target mask makes this an experiment about *which player* to
vote for; action syntax, speech, and overpowering are later experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
from random import Random
from statistics import stdev
from time import monotonic
from typing import Any, Callable

from env.rules import GameConfig, MafiaEnv, RULESET_VERSION


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM = (
    "You are a Town player in Mafia. Town wins by eliminating the hidden Mafia. "
    "Use only your own role, private investigation results, and public events. "
    "Choose the living player most likely to be Mafia. "
    "Reply with only the single digit after P in a legal target ID."
)


@dataclass(frozen=True)
class VoteObservation:
    """The same player view returned by the rules engine, without transport types."""

    current_player: str
    role: str
    day: int
    phase: str
    alive: list[str]
    public_events: list[dict[str, Any]]
    private_events: list[dict[str, Any]]
    legal_actions: dict[str, Any]


@dataclass(frozen=True)
class VoteDecision:
    input_ids: tuple[int, ...]
    legal_targets: tuple[str, ...]
    chosen_target: str
    candidate_input_ids: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class GameResult:
    seed: int
    winner: str | None
    town_reward: float
    decisions: tuple[VoteDecision, ...]
    vote_attempts: int
    warnings: int
    forfeits: int


def vote_messages(observation: VoteObservation, public_events: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Use the active player's view only; omit referee internals and other roles."""
    if observation.phase != "vote" or observation.role == "mafia":
        raise ValueError("This policy only handles Town votes")
    legal = observation.legal_actions
    if observation.current_player is None or legal is None:
        raise ValueError("A live voter and legal targets are required")
    targets = legal["targets"]
    state = {
        "you": observation.current_player,
        "your_role": observation.role,
        "day": observation.day,
        "vote_round": legal["round"],
        "alive": observation.alive,
        "your_private_events": observation.private_events,
        "public_events": public_events,
        "legal_vote_targets": targets,
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))},
    ]


def encode_vote(observation: VoteObservation, tokenizer: Any, max_prompt_tokens: int) -> tuple[int, ...]:
    """Keep recent public history while always retaining legal targets and private facts."""
    events = observation.public_events
    while True:
        messages = vote_messages(observation, events)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) <= max_prompt_tokens:
            return tuple(ids)
        if not events:
            raise ValueError("Vote prompt exceeds token limit even without public events")
        events = events[1:]


class VotePolicy:
    def __init__(self, model: Any, tokenizer: Any, max_prompt_tokens: int = 1536) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.device = next(model.parameters()).device
        self.target_tokens: dict[str, int] = {}
        self.torch = torch

    def _target_token(self, target: str) -> int:
        if target not in self.target_tokens:
            match = re.fullmatch(r"P(\d+)", target)
            if match is None:
                raise ValueError(f"Unexpected player ID: {target}")
            ids = self.tokenizer.encode(match.group(1), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Expected a single token for {target}, got {ids}")
            self.target_tokens[target] = ids[0]
        return self.target_tokens[target]

    def _distribution(self, ids: tuple[int, ...], targets: tuple[str, ...]) -> Any:
        torch = self.torch
        inputs = torch.tensor([ids], device=self.device)
        logits = self.model(input_ids=inputs).logits[0, -1].float()
        candidate_ids = torch.tensor(
            [self._target_token(target) for target in targets], device=self.device
        )
        return torch.distributions.Categorical(logits=logits[candidate_ids])

    def choose(self, observation: VoteObservation, *, sample: bool) -> VoteDecision:
        legal = observation.legal_actions
        if legal is None:
            raise ValueError("Missing legal actions")
        targets = tuple(legal["targets"])
        ids = encode_vote(observation, self.tokenizer, self.max_prompt_tokens)
        with self.torch.inference_mode():
            distribution = self._distribution(ids, targets)
            index = int(distribution.sample().item() if sample else distribution.probs.argmax().item())
        return VoteDecision(ids, targets, targets[index])

    def log_prob_and_entropy(self, decision: VoteDecision) -> tuple[Any, Any]:
        distribution = self._distribution(decision.input_ids, decision.legal_targets)
        index = decision.legal_targets.index(decision.chosen_target)
        index_tensor = self.torch.tensor(index, device=self.device)
        return distribution.log_prob(index_tensor), distribution.entropy()


def random_action(observation: VoteObservation, rng: Random) -> dict[str, Any]:
    """Match the fixed random baseline without importing OpenEnv transport."""
    legal = observation.legal_actions
    if observation.phase == "speak":
        target = rng.choice([p for p in observation.alive if p != observation.current_player])
        message = "" if rng.random() < 0.5 else f"I suspect {target}."
        overpower_targets = legal["overpower_targets"]
        if overpower_targets and rng.random() < 0.25:
            return {"type": "overpower", "target": rng.choice(overpower_targets), "message": message}
        return {"type": "speak", "message": message}
    return {"type": legal["type"], "target": rng.choice(legal["targets"])}


def play_game(
    seed: int,
    policy: VotePolicy | None,
    *,
    sample: bool,
    on_town_vote: Callable[[VoteObservation, VoteDecision, str], None] | None = None,
    config: GameConfig | None = None,
) -> GameResult:
    environment = MafiaEnv(config)
    environment.reset(seed=seed)
    random_policies = {"town": Random(seed * 2 + 1), "mafia": Random(seed * 2 + 2)}
    decisions: list[VoteDecision] = []
    vote_attempts = warnings = forfeits = attempts = 0

    while environment.current_player is not None:
        attempts += 1
        if attempts > 200:
            raise RuntimeError(f"Game seed {seed} exceeded 200 action attempts")
        actor = environment.current_player
        view = environment.observe(actor)
        observation = VoteObservation(
            current_player=actor,
            role=view["role"],
            day=view["day"],
            phase=view["phase"],
            alive=view["alive"],
            public_events=view["public_events"],
            private_events=view["private_events"],
            legal_actions=view["legal_actions"],
        )
        team = "mafia" if observation.role == "mafia" else "town"
        if team == "town" and observation.phase == "vote" and policy is not None:
            decision = policy.choose(observation, sample=sample)
            decisions.append(decision)
            if on_town_vote is not None:
                mafia_player = next(p for p, role in environment.roles.items() if role.value == "mafia")
                on_town_vote(observation, decision, mafia_player)
            action = {"type": "vote", "target": decision.chosen_target}
        else:
            action = random_action(observation, random_policies[team])
        if observation.phase == "vote":
            vote_attempts += 1
        result = environment.step(action)
        warnings += int(not result.accepted and not result.forfeited)
        forfeits += int(result.forfeited)

    reward = 1.0 if environment.winner == "town" else -1.0 if environment.winner == "mafia" else 0.0
    return GameResult(seed, environment.winner, reward, tuple(decisions), vote_attempts, warnings, forfeits)


def evaluate(policy: VotePolicy, seeds: range) -> dict[str, Any]:
    games = [play_game(seed, policy, sample=False) for seed in seeds]
    return {
        "games": len(games),
        "town_win_outcomes": [int(game.winner == "town") for game in games],
        "town_win_rate": sum(game.winner == "town" for game in games) / len(games),
        "town_mean_reward": sum(game.town_reward for game in games) / len(games),
        "vote_valid_rate": 1 - (sum(game.warnings + game.forfeits for game in games) / max(1, sum(game.vote_attempts for game in games))),
        "warnings": sum(game.warnings for game in games),
        "forfeits": sum(game.forfeits for game in games),
    }


def train(
    *,
    output_dir: str,
    dashboard_url: str,
    updates: int = 6,
    games_per_update: int = 8,
    eval_games: int = 24,
    eval_interval: int = 2,
    learning_rate: float = 1e-4,
    entropy_coef: float = 0.01,
    policy_kind: str = "digit",
    eval_start_seed: int = 50_000,
    train_start_seed: int = 1_000,
    flush_metrics: Callable[[], None] | None = None,
    publish: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """On-policy REINFORCE over complete games, updating Qwen LoRA weights."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if min(updates, games_per_update, eval_games, eval_interval) < 1:
        raise ValueError("Update, game, and evaluation counts must be positive")
    if policy_kind not in ("digit", "centered_candidate"):
        raise ValueError(f"Unknown vote policy kind: {policy_kind}")
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
    if policy_kind == "centered_candidate":
        from src.training.candidate_vote import CandidateVotePolicy

        model.gradient_checkpointing_enable()
        policy = CandidateVotePolicy(model, tokenizer, centered_labels=True)
    else:
        policy = VotePolicy(model, tokenizer)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found")
    tracked_weight = next(
        parameter for name, parameter in model.named_parameters() if "lora_B" in name
    )
    initial_weight = tracked_weight.detach().float().clone()
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    eval_seeds = range(eval_start_seed, eval_start_seed + eval_games)
    baseline_games = [play_game(seed, None, sample=False) for seed in eval_seeds]
    baseline_outcomes = [int(game.winner == "town") for game in baseline_games]
    baseline_win_rate = sum(baseline_outcomes) / eval_games

    def paired_difference(outcomes: list[int]) -> tuple[float, float]:
        differences = [candidate - baseline for candidate, baseline in zip(outcomes, baseline_outcomes)]
        mean = sum(differences) / len(differences)
        half_width = 1.96 * stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
        return mean, half_width
    config = {
        "model": MODEL_ID,
        "ruleset": RULESET_VERSION,
        "team": "town",
        "action_family": (
            "centered_candidate_vote_target" if policy_kind == "centered_candidate"
            else "digit_vote_target"
        ),
        "algorithm": "episodic_reinforce_with_ema_baseline",
        "updates": updates,
        "games_per_update": games_per_update,
        "eval_seeds": [eval_seeds.start, eval_seeds.stop - 1],
        "train_seeds": [train_start_seed, train_start_seed + updates * games_per_update - 1],
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "max_prompt_tokens": policy.max_prompt_tokens,
        "baseline_town_win_rate": baseline_win_rate,
        "torch_version": str(torch.__version__),
    }
    print(f"Dashboard: {dashboard_url}", flush=True)
    run_prefix = "town-centered-vote-" if policy_kind == "centered_candidate" else "town-vote-"
    run_name = run_prefix + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(output / "tb" / run_name), flush_secs=5)
    writer.add_text("run/config", json.dumps(config, indent=2), global_step=0)

    def log_scalars(metrics: dict[str, float | int], step: int) -> None:
        for name, value in metrics.items():
            writer.add_scalar(name, value, global_step=step)
        writer.flush()
        if flush_metrics is not None:
            flush_metrics()

    history: list[dict[str, Any]] = []
    recent_rewards: deque[float] = deque(maxlen=20)
    games_seen = 0
    baseline = 0.0
    try:
        model.eval()
        initial_eval = evaluate(policy, eval_seeds)
        initial_diff, initial_half_width = paired_difference(initial_eval["town_win_outcomes"])
        initial_metrics: dict[str, float | int] = {
            "eval/town_win_rate": initial_eval["town_win_rate"],
            "eval/random_baseline_win_rate": baseline_win_rate,
            "eval/mean_reward": initial_eval["town_mean_reward"],
            "eval/vote_valid_rate": initial_eval["vote_valid_rate"],
            "eval/warnings": initial_eval["warnings"],
            "eval/forfeits": initial_eval["forfeits"],
            "eval/paired_difference": initial_diff,
            "eval/paired_ci_low": initial_diff - initial_half_width,
            "eval/paired_ci_high": initial_diff + initial_half_width,
            "elapsed_minutes": (monotonic() - started) / 60,
        }
        log_scalars(initial_metrics, step=0)
        if publish is not None:
            publish(run_name, {"kind": "initial", "step": 0, "config": config, "metrics": initial_metrics})
        print(f"Before training: {initial_eval}; random Town baseline={baseline_win_rate:.3f}", flush=True)

        for update in range(1, updates + 1):
            model.eval()
            games = []
            for i in range(games_per_update):
                game = play_game(train_start_seed + (update - 1) * games_per_update + i, policy, sample=True)
                games.append(game)
                games_seen += 1
                recent_rewards.append(game.town_reward)
                writer.add_scalar("train/game_reward", game.town_reward, global_step=games_seen)
                writer.add_scalar("train/reward_rolling_20_games", sum(recent_rewards) / len(recent_rewards), global_step=games_seen)
                if publish is not None:
                    publish(run_name, {"kind": "game", "step": games_seen, "metrics": {
                        "train/game_reward": game.town_reward,
                        "train/reward_rolling_20_games": sum(recent_rewards) / len(recent_rewards),
                    }})
            mean_reward = sum(game.town_reward for game in games) / len(games)
            advantages = [game.town_reward - baseline for game in games]
            vote_count = sum(len(game.decisions) for game in games)
            if vote_count == 0:
                raise RuntimeError("No Town votes occurred in the training batch")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = entropy_total = 0.0
            for game, advantage in zip(games, advantages):
                if not game.decisions:
                    continue
                scale = 1 / (len(games) * len(game.decisions))
                for decision in game.decisions:
                    log_prob, entropy = policy.log_prob_and_entropy(decision)
                    loss = -advantage * scale * log_prob - entropy_coef * scale * entropy
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite policy loss at update {update}")
                    loss.backward()
                    loss_total += float(loss.detach())
                    entropy_total += float(entropy.detach())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient at update {update}")
            optimizer.step()
            baseline = 0.9 * baseline + 0.1 * mean_reward
            changed = float((tracked_weight.detach().float() - initial_weight).abs().max())
            metrics: dict[str, float | int] = {
                "train/mean_reward": mean_reward,
                "train/town_win_rate": sum(game.winner == "town" for game in games) / len(games),
                "train/policy_loss": loss_total,
                "train/mean_vote_entropy": entropy_total / vote_count,
                "train/votes": vote_count,
                "train/gradient_norm": grad_norm,
                "train/weight_change_max": changed,
                "train/learning_rate": learning_rate,
                "elapsed_minutes": (monotonic() - started) / 60,
            }
            if update % eval_interval == 0 or update == updates:
                model.eval()
                result = evaluate(policy, eval_seeds)
                paired_diff, paired_half_width = paired_difference(result["town_win_outcomes"])
                metrics.update({
                    "eval/town_win_rate": result["town_win_rate"],
                    "eval/random_baseline_win_rate": baseline_win_rate,
                    "eval/mean_reward": result["town_mean_reward"],
                    "eval/vote_valid_rate": result["vote_valid_rate"],
                    "eval/warnings": result["warnings"],
                    "eval/forfeits": result["forfeits"],
                    "eval/paired_difference": paired_diff,
                    "eval/paired_ci_low": paired_diff - paired_half_width,
                    "eval/paired_ci_high": paired_diff + paired_half_width,
                })
            log_scalars(metrics, step=games_seen)
            if publish is not None:
                publish(run_name, {"kind": "update", "step": games_seen, "metrics": metrics})
            history.append({"update": update, "games_seen": games_seen, **metrics})
            print(json.dumps({"update": update, **metrics}), flush=True)

        if changed <= 0:
            raise RuntimeError("LoRA weights did not change")
        adapter_path = output / "runs" / run_name / "adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        report = {"config": config, "run_name": run_name, "initial_eval": initial_eval, "history": history, "adapter_path": str(adapter_path), "dashboard": dashboard_url}
        (adapter_path.parent / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        writer.close()
