"""Frozen policy experiment: score each legal target with one shared answer."""

import json
import re
from typing import Any

from src.training.audit import summarize_votes, swap_player_labels, vote_record
from src.training.vote import MODEL_ID, VoteDecision, VoteObservation, play_game


CANDIDATE_SYSTEM = (
    "You are a Town player in Mafia. Town wins by eliminating the hidden Mafia. "
    "Use only your role, your private investigation results, and public events. "
    "Judge whether the candidate named in the next message is Mafia. "
    "Answer with exactly Yes or No."
)
CENTERED_SYSTEM = (
    "You are YOU, a Town player in Mafia. Town wins by eliminating the hidden Mafia. "
    "SUSPECT is the player you are considering voting out. OTHER players are "
    "other living players; FORMER players are no longer alive. Use only your "
    "role, your private investigation results, and public events. "
    "Is SUSPECT Mafia? Answer with exactly Yes or No."
)


def candidate_messages(
    observation: VoteObservation, public_events: list[dict[str, Any]], candidate: str
) -> list[dict[str, str]]:
    if candidate not in observation.legal_actions["targets"]:
        raise ValueError("Candidate must be a legal vote target")
    state = {
        "you": observation.current_player,
        "your_role": observation.role,
        "day": observation.day,
        "vote_round": observation.legal_actions["round"],
        "alive": observation.alive,
        "your_private_events": observation.private_events,
        "public_events": public_events,
        "candidate": candidate,
    }
    return [
        {"role": "system", "content": CANDIDATE_SYSTEM},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))},
    ]


def centered_candidate_messages(
    observation: VoteObservation, public_events: list[dict[str, Any]], candidate: str
) -> list[dict[str, str]]:
    """Give every suspect the same name and position in the prompt."""
    if candidate not in observation.legal_actions["targets"]:
        raise ValueError("Candidate must be a legal vote target")
    aliases = {observation.current_player: "YOU", candidate: "SUSPECT"}
    others = 0
    for player in observation.alive:
        if player not in aliases:
            others += 1
            aliases[player] = f"OTHER{others}"

    # Dead players can still appear in old events and claims.
    history = json.dumps([public_events, observation.private_events], ensure_ascii=False)
    former = 0
    for match in re.finditer(r"\bP\d+\b", history):
        player = match.group()
        if player not in aliases:
            former += 1
            aliases[player] = f"FORMER{former}"

    pattern = re.compile(r"\bP\d+\b")

    def rename(value: Any) -> Any:
        if isinstance(value, str):
            return pattern.sub(lambda match: aliases.get(match.group(), match.group()), value)
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value

    state = {
        "you": "YOU",
        "your_role": observation.role,
        "day": observation.day,
        "vote_round": observation.legal_actions["round"],
        "alive": ["YOU", "SUSPECT", *(f"OTHER{i}" for i in range(1, others + 1))],
        "your_private_events": rename(observation.private_events),
        "public_events": rename(public_events),
        "candidate": "SUSPECT",
    }
    return [
        {"role": "system", "content": CENTERED_SYSTEM},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))},
    ]


class CandidateVotePolicy:
    """Rank targets by their Yes-versus-No next-token score."""

    def __init__(
        self, model: Any, tokenizer: Any, max_prompt_tokens: int = 1536,
        centered_labels: bool = False,
    ) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.centered_labels = centered_labels
        self.device = next(model.parameters()).device
        self.torch = torch
        self.answer_tokens = {}
        for answer in ("Yes", "No"):
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Expected one token for {answer}, got {ids}")
            self.answer_tokens[answer] = ids[0]

    def _encode(self, observation: VoteObservation, candidate: str) -> tuple[int, ...]:
        events = observation.public_events
        while True:
            builder = centered_candidate_messages if self.centered_labels else candidate_messages
            messages = builder(observation, events, candidate)
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if len(ids) <= self.max_prompt_tokens:
                return tuple(ids)
            if not events:
                raise ValueError("Candidate prompt exceeds token limit without public events")
            events = events[1:]

    def _distribution(self, candidate_prompts: tuple[tuple[int, ...], ...]) -> Any:
        scores = []
        for ids in candidate_prompts:
            inputs = self.torch.tensor([ids], device=self.device)
            logits = self.model(input_ids=inputs).logits[0, -1].float()
            scores.append(logits[self.answer_tokens["Yes"]] - logits[self.answer_tokens["No"]])
        return self.torch.distributions.Categorical(logits=self.torch.stack(scores))

    def choose(self, observation: VoteObservation, *, sample: bool) -> VoteDecision:
        targets = tuple(observation.legal_actions["targets"])
        candidate_prompts = tuple(self._encode(observation, target) for target in targets)
        with self.torch.inference_mode():
            distribution = self._distribution(candidate_prompts)
            index = int(distribution.sample().item() if sample else distribution.probs.argmax().item())
        return VoteDecision((), targets, targets[index], candidate_prompts)

    def log_prob_and_entropy(self, decision: VoteDecision) -> tuple[Any, Any]:
        if len(decision.candidate_input_ids) != len(decision.legal_targets):
            raise ValueError("Candidate prompts must match legal targets")
        distribution = self._distribution(decision.candidate_input_ids)
        index = decision.legal_targets.index(decision.chosen_target)
        return distribution.log_prob(
            self.torch.tensor(index, device=self.device)
        ), distribution.entropy()


def evaluate_candidate_policy(
    start_seed: int = 50_000, games: int = 24, centered_labels: bool = False
) -> dict[str, Any]:
    """Test the new action representation with frozen Qwen, without training."""
    if games < 1:
        raise ValueError("games must be positive")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
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
    model.eval()
    policy = CandidateVotePolicy(model, tokenizer, centered_labels=centered_labels)

    votes: list[dict[str, Any]] = []
    views: list[tuple[int, VoteObservation, VoteDecision]] = []
    winners: dict[int, str | None] = {}
    for seed in range(start_seed, start_seed + games):
        def capture(observation: VoteObservation, decision: VoteDecision, mafia_player: str) -> None:
            votes.append(vote_record(seed, observation, decision, mafia_player))
            views.append((seed, observation, decision))

        winners[seed] = play_game(seed, policy, sample=False, on_town_vote=capture).winner

    label_probe = {}
    for other in ("P3", "P5"):
        matching = [
            (seed, observation, decision)
            for seed, observation, decision in views
            if "P0" in decision.legal_targets and other in decision.legal_targets
        ]
        changed = 0
        chose_real_p0 = 0
        for _, observation, decision in matching:
            swapped = swap_player_labels(observation, "P0", other)
            chosen_alias = policy.choose(swapped, sample=False).chosen_target
            chosen_real = {"P0": other, other: "P0"}.get(chosen_alias, chosen_alias)
            changed += int(chosen_real != decision.chosen_target)
            chose_real_p0 += int(chosen_real == "P0")
        label_probe[f"P0_swap_{other}"] = {
            "same_views": len(matching),
            "changed_real_target": changed,
            "original_chose_p0": sum(decision.chosen_target == "P0" for _, _, decision in matching),
            "after_swap_chose_real_p0": chose_real_p0,
        }

    return {
        "policy": "candidate_centered_yes_no" if centered_labels else "candidate_yes_no",
        "model": MODEL_ID,
        "seeds": [start_seed, start_seed + games - 1],
        "town_wins": sum(winner == "town" for winner in winners.values()),
        "per_seed_winner": winners,
        "vote_summary": summarize_votes(votes),
        "label_probe": label_probe,
        "vote_trace": votes,
    }
