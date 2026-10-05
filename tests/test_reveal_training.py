"""Local controller and prompt checks before the Modal GPU pilot."""

import json

from src.training.reveal_choice import RevealChoice, eligible_opportunity
from src.training.reveal_train import (
    FixedRevealPolicy,
    PromptDecision,
    evaluate,
    play_game,
    reveal_messages,
)


class SequencePolicy:
    def __init__(self, choices):
        self.choices = choices
        self.calls = []
        self.samples = []

    def start_episode(self):
        self.calls.clear()
        self.samples.clear()

    def choose(self, observation, *, sample):
        choice = self.choices[len(self.calls)]
        self.calls.append(observation)
        if sample:
            self.samples.append(PromptDecision((), (), choice))
        return choice


def test_reveal_prompt_uses_actor_view_and_swaps_yes_no_mapping() -> None:
    view = {
        "player": "P7", "role": "detective", "day": 2,
        "phase": "speak", "discussion_round": 2,
        "alive": ["P1", "P7", "P9"],
        "public_events": [
            {"day": 2, "type": "speech", "player": "P1", "message": "Vote P7"}
        ],
        "private_events": [
            {"night": 2, "type": "investigation", "target": "P9", "is_mafia": True}
        ],
        "legal_actions": {"action_types": ["speak", "overpower"]},
    }
    observation = eligible_opportunity(view)
    assert observation is not None
    yes_reveal = json.loads(reveal_messages(
        observation, list(observation.public_events), yes_means_reveal=True
    )[1]["content"])
    yes_hold = json.loads(reveal_messages(
        observation, list(observation.public_events), yes_means_reveal=False
    )[1]["content"])
    assert yes_reveal["actions"] == {"Yes": "REVEAL", "No": "HOLD"}
    assert yes_hold["actions"] == {"Yes": "HOLD", "No": "REVEAL"}
    assert yes_reveal["confirmed_mafia_target"] == "P9"
    assert yes_reveal["public_events"] == view["public_events"]
    assert "roles" not in yes_reveal
    assert "other_private_events" not in yes_reveal


def test_sampled_policy_is_called_on_all_three_actual_eligible_turns() -> None:
    policy = SequencePolicy([
        RevealChoice.HOLD, RevealChoice.REVEAL, RevealChoice.HOLD
    ])
    episode = play_game(
        70_017, "counterclaim", policy, sample=True, record_trace=True
    )
    assert episode.eligible_choices == 3
    assert episode.reveals == 1
    assert len(episode.model_decisions) == 3
    assert [row["round"] for row in episode.choice_rows] == [1, 2, 3]
    assert [row["choice"] for row in episode.choice_rows] == [
        "hold", "reveal", "hold"
    ]
    assert any(
        frame["reason"] == "qwen_reveal" and frame["message"]
        for frame in episode.frames
    )
    assert all(frame["accepted"] for frame in episode.frames)


def test_heldout_evaluation_keeps_natural_games_without_eligible_turns() -> None:
    result, traces = evaluate(
        FixedRevealPolicy(RevealChoice.HOLD),
        range(70_000, 70_010),
        record_trace=True,
    )
    assert result["games"] == len(traces) == 20
    assert {row["arm"] for row in result["per_game"]} == {
        "counterclaim", "bluff"
    }
    assert any(row["eligible_choices"] == 0 for row in result["per_game"])
    assert all(row["reveals"] == 0 for row in result["per_game"])
    assert all(
        choice["p_reveal"] == 0.0 and choice["p_hold"] == 1.0
        for row in result["per_game"] for choice in row["choices"]
    )
    assert all(len(trace["frames"]) == trace["attempts"] for trace in traces)
