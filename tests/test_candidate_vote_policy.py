"""The candidate question uses only the active Town player's information."""

import pytest

from src.training.audit import swap_player_labels
from src.training.candidate_vote import candidate_messages, centered_candidate_messages
from src.training.vote import VoteObservation


def test_candidate_question_keeps_private_investigation_and_names_one_legal_target() -> None:
    observation = VoteObservation(
        current_player="P2",
        role="detective",
        day=2,
        phase="vote",
        alive=["P0", "P2", "P3"],
        public_events=[{"type": "speech", "player": "P0", "message": "P3 is suspicious"}],
        private_events=[{"type": "investigation", "target": "P3", "is_mafia": True}],
        legal_actions={"type": "vote", "round": "initial", "targets": ["P0", "P3"]},
    )
    messages = candidate_messages(observation, observation.public_events, "P3")
    assert '"candidate":"P3"' in messages[1]["content"]
    assert '"is_mafia":true' in messages[1]["content"]
    assert '"roles"' not in messages[1]["content"]

    with pytest.raises(ValueError, match="legal vote target"):
        candidate_messages(observation, observation.public_events, "P2")


def test_centered_question_is_identical_after_player_labels_are_swapped() -> None:
    observation = VoteObservation(
        current_player="P2",
        role="detective",
        day=2,
        phase="vote",
        alive=["P0", "P2", "P5"],
        public_events=[{"type": "speech", "player": "P5", "message": "I suspect P0"}],
        private_events=[{"type": "investigation", "target": "P0", "is_mafia": True}],
        legal_actions={"type": "vote", "round": "initial", "targets": ["P0", "P5"]},
    )
    original = centered_candidate_messages(observation, observation.public_events, "P0")
    swapped = swap_player_labels(observation, "P0", "P5")
    relabeled = centered_candidate_messages(swapped, swapped.public_events, "P5")
    assert original == relabeled
    assert "P0" not in original[1]["content"]
    assert "P5" not in original[1]["content"]
    assert '"candidate":"SUSPECT"' in original[1]["content"]


def test_centered_question_renames_player_ids_beyond_six_seats() -> None:
    observation = VoteObservation(
        current_player="P7",
        role="detective",
        day=1,
        phase="vote",
        alive=[f"P{i}" for i in range(8)],
        public_events=[{"type": "speech", "player": "P6", "message": "I suspect P5"}],
        private_events=[{"type": "investigation", "target": "P5", "is_mafia": True}],
        legal_actions={"type": "vote", "round": "initial", "targets": [f"P{i}" for i in range(7)]},
    )
    messages = centered_candidate_messages(observation, observation.public_events, "P5")
    content = messages[1]["content"]
    assert "P7" not in content
    assert "P6" not in content
    assert "P5" not in content
    assert '"target":"SUSPECT"' in content
