"""Audit-only truth must stay separate from the Town player's observation."""

from src.training.audit import swap_player_labels, vote_record
from src.training.vote import VoteDecision, VoteObservation, play_game


class FirstLegalTownVote:
    def choose(self, observation: VoteObservation, *, sample: bool) -> VoteDecision:
        targets = tuple(observation.legal_actions["targets"])
        return VoteDecision((), targets, targets[0])


def test_audit_observer_receives_truth_after_policy_chooses() -> None:
    records = []

    def record(observation: VoteObservation, decision: VoteDecision, mafia_player: str) -> None:
        assert observation.role in {"detective", "villager"}
        assert not hasattr(observation, "roles")
        assert mafia_player in observation.alive
        records.append(vote_record(3, observation, decision, mafia_player))

    play_game(3, FirstLegalTownVote(), sample=False, on_town_vote=record)
    assert records
    assert all(record["target"] in record["legal_targets"] for record in records)


def test_audit_identifies_a_detective_vote_with_known_mafia() -> None:
    observation = VoteObservation(
        current_player="P2",
        role="detective",
        day=2,
        phase="vote",
        alive=["P0", "P2", "P3"],
        public_events=[],
        private_events=[{"type": "investigation", "target": "P3", "is_mafia": True}],
        legal_actions={"type": "vote", "round": "initial", "targets": ["P0", "P3"]},
    )
    decision = VoteDecision((), ("P0", "P3"), "P0")
    record = vote_record(5, observation, decision, "P3")
    assert record["knows_mafia"] is True
    assert record["hit_mafia"] is False


def test_label_probe_swaps_every_visible_player_reference() -> None:
    observation = VoteObservation(
        current_player="P2",
        role="detective",
        day=2,
        phase="vote",
        alive=["P0", "P2", "P5"],
        public_events=[{"message": "I suspect P0, not P5."}],
        private_events=[{"type": "investigation", "target": "P0", "is_mafia": True}],
        legal_actions={"type": "vote", "round": "initial", "targets": ["P0", "P5"]},
    )
    swapped = swap_player_labels(observation, "P0", "P5")
    assert swapped.current_player == "P2"
    assert swapped.alive == ["P5", "P2", "P0"]
    assert swapped.public_events[0]["message"] == "I suspect P5, not P0."
    assert swapped.private_events[0]["target"] == "P5"
    assert swapped.legal_actions["targets"] == ["P5", "P0"]
