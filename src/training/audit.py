"""Compare frozen and trained Town votes without changing the game policy."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import re
from typing import Any

from src.training.vote import MODEL_ID, VoteDecision, VoteObservation, VotePolicy, play_game


def vote_record(
    seed: int, observation: VoteObservation, decision: VoteDecision, mafia_player: str
) -> dict[str, Any]:
    """Keep referee truth in the audit record, never in the policy observation."""
    known_mafia = any(
        event.get("type") == "investigation"
        and event.get("target") == mafia_player
        and event.get("is_mafia") is True
        for event in observation.private_events
    )
    return {
        "seed": seed,
        "day": observation.day,
        "round": observation.legal_actions["round"],
        "voter": observation.current_player,
        "role": observation.role,
        "target": decision.chosen_target,
        "mafia_player": mafia_player,
        "hit_mafia": decision.chosen_target == mafia_player,
        "knows_mafia": known_mafia,
        "legal_targets": list(decision.legal_targets),
    }


def summarize_votes(votes: list[dict[str, Any]]) -> dict[str, Any]:
    selected = Counter(vote["target"] for vote in votes)
    available = Counter(target for vote in votes for target in vote["legal_targets"])
    known = [vote for vote in votes if vote["knows_mafia"]]
    return {
        "votes": len(votes),
        "mafia_target_rate": sum(vote["hit_mafia"] for vote in votes) / len(votes) if votes else 0.0,
        "detective_votes_with_known_mafia": len(known),
        "known_mafia_vote_rate": sum(vote["hit_mafia"] for vote in known) / len(known) if known else None,
        "target_by_seat": {
            seat: {"chosen": selected[seat], "available": available[seat]}
            for seat in sorted(available)
        },
    }


def swap_player_labels(observation: VoteObservation, first: str, second: str) -> VoteObservation:
    """Rename two seats consistently in every field the voter can see."""
    swap = {first: second, second: first}
    pattern = re.compile(r"\bP[0-5]\b")

    def rename(value: Any) -> Any:
        if isinstance(value, str):
            return pattern.sub(lambda match: swap.get(match.group(), match.group()), value)
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value

    return replace(
        observation,
        current_player=rename(observation.current_player),
        alive=rename(observation.alive),
        public_events=rename(observation.public_events),
        private_events=rename(observation.private_events),
        legal_actions=rename(observation.legal_actions),
    )


def audit_adapter(run_name: str, start_seed: int = 50_000, games: int = 24) -> dict[str, Any]:
    """Run both policies on paired seeds, then compare votes on identical views."""
    if games < 1:
        raise ValueError("games must be positive")
    if "/" in run_name or ".." in run_name:
        raise ValueError("run_name must be one checkpoint directory name")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path("/data/mafia_vote/runs") / run_name / "adapter"
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter not found in Modal Volume: {adapter}")

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
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    seeds = range(start_seed, start_seed + games)
    base_votes: list[dict[str, Any]] = []
    base_views: list[tuple[int, VoteObservation, VoteDecision, str]] = []
    base_winners: dict[int, str | None] = {}
    base_policy = VotePolicy(model, tokenizer)
    label_probe: dict[str, Any] = {}
    with model.disable_adapter():
        for seed in seeds:
            def capture_base(observation: VoteObservation, decision: VoteDecision, mafia_player: str) -> None:
                base_votes.append(vote_record(seed, observation, decision, mafia_player))
                base_views.append((seed, observation, decision, mafia_player))

            base_winners[seed] = play_game(seed, base_policy, sample=False, on_town_vote=capture_base).winner

        for other in ("P3", "P5"):
            matching = [
                (seed, observation, decision)
                for seed, observation, decision, _ in base_views
                if "P0" in decision.legal_targets and other in decision.legal_targets
            ]
            changed_real_target = 0
            chosen_real_p0_after_swap = 0
            chosen_alias_other_after_swap = 0
            examples = []
            for seed, observation, decision in matching:
                swapped = swap_player_labels(observation, "P0", other)
                chosen_alias = base_policy.choose(swapped, sample=False).chosen_target
                chosen_real = {"P0": other, other: "P0"}.get(chosen_alias, chosen_alias)
                changed_real_target += int(chosen_real != decision.chosen_target)
                chosen_real_p0_after_swap += int(chosen_real == "P0")
                chosen_alias_other_after_swap += int(chosen_alias == other)
                if len(examples) < 8 and chosen_real != decision.chosen_target:
                    examples.append({
                        "seed": seed,
                        "voter": observation.current_player,
                        "original_target": decision.chosen_target,
                        "swapped_target_alias": chosen_alias,
                        "swapped_target_real": chosen_real,
                    })
            label_probe[f"P0_swap_{other}"] = {
                "same_views": len(matching),
                "changed_real_target": changed_real_target,
                "original_chose_p0": sum(decision.chosen_target == "P0" for _, _, decision in matching),
                "original_chose_other": sum(decision.chosen_target == other for _, _, decision in matching),
                "after_swap_chose_real_p0": chosen_real_p0_after_swap,
                "after_swap_chose_alias_other": chosen_alias_other_after_swap,
                "examples": examples,
            }

    adapted_policy = VotePolicy(model, tokenizer)
    adapted_votes: list[dict[str, Any]] = []
    adapted_winners: dict[int, str | None] = {}
    for seed in seeds:
        def capture_adapted(observation: VoteObservation, decision: VoteDecision, mafia_player: str) -> None:
            adapted_votes.append(vote_record(seed, observation, decision, mafia_player))

        adapted_winners[seed] = play_game(seed, adapted_policy, sample=False, on_town_vote=capture_adapted).winner

    changed_same_view: list[dict[str, Any]] = []
    for seed, observation, base_decision, mafia_player in base_views:
        adapted_decision = adapted_policy.choose(observation, sample=False)
        if adapted_decision.chosen_target != base_decision.chosen_target:
            record = vote_record(seed, observation, base_decision, mafia_player)
            record["base_target"] = base_decision.chosen_target
            record["adapted_target"] = adapted_decision.chosen_target
            record["adapted_hit_mafia"] = adapted_decision.chosen_target == mafia_player
            changed_same_view.append(record)

    return {
        "run_name": run_name,
        "model": MODEL_ID,
        "seeds": [seeds.start, seeds.stop - 1],
        "base": {
            "town_wins": sum(winner == "town" for winner in base_winners.values()),
            "votes": summarize_votes(base_votes),
            "per_seed_winner": base_winners,
        },
        "adapted": {
            "town_wins": sum(winner == "town" for winner in adapted_winners.values()),
            "votes": summarize_votes(adapted_votes),
            "per_seed_winner": adapted_winners,
        },
        "same_view": {
            "compared_votes": len(base_views),
            "changed_votes": len(changed_same_view),
            "changed": changed_same_view,
        },
        "changed_outcomes": [
            {"seed": seed, "base": base_winners[seed], "adapted": adapted_winners[seed]}
            for seed in seeds if base_winners[seed] != adapted_winners[seed]
        ],
        "label_probe": label_probe,
        "base_vote_trace": base_votes,
        "adapted_vote_trace": adapted_votes,
    }
