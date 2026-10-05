"""The future learned policy must choose on every eligible speaking turn."""

from env.rules import MafiaEnv, Phase, Role
from src.training.reveal_choice import (
    RevealChoice,
    choose_reveal_action,
    eligible_opportunity,
)
from src.training.reveal_timing_probe import run_timing_probe


class SequencePolicy:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.observations = []

    def choose(self, opportunity, *, sample):
        self.observations.append(opportunity)
        return next(self.choices)


def test_policy_is_consulted_at_every_eligible_detective_speech_turn() -> None:
    environment = MafiaEnv()
    environment.reset(seed=8)
    mafia = next(player for player, role in environment.roles.items() if role == Role.MAFIA)
    detective = next(player for player, role in environment.roles.items() if role == Role.DETECTIVE)
    victim = next(player for player, role in environment.roles.items() if role == Role.VILLAGER)
    environment.step({"type": "kill", "target": victim})
    environment.step({"type": "investigate", "target": mafia})
    assert environment.phase == Phase.SPEAK

    policy = SequencePolicy([
        RevealChoice.HOLD, RevealChoice.REVEAL, RevealChoice.HOLD
    ])
    choices = []
    while environment.phase == Phase.SPEAK:
        view = environment.observe(environment.current_player)
        if view["player"] == detective:
            decision = choose_reveal_action(view, policy, sample=True)
            choices.append(decision.choice)
            action = decision.action
        else:
            action = {"type": "speak", "message": ""}
        assert environment.step(action).accepted

    assert choices == [RevealChoice.HOLD, RevealChoice.REVEAL, RevealChoice.HOLD]
    assert [observation.discussion_round for observation in policy.observations] == [1, 2, 3]
    assert any(
        event["type"] == "speech" and event["player"] == detective
        and "is Mafia" in event["message"]
        for event in policy.observations[2].public_events
    )
    detective_speeches = [
        event["message"] for event in environment.public_events
        if event["type"] == "speech" and event["player"] == detective
    ]
    assert detective_speeches[0] == detective_speeches[2] == ""
    assert mafia in detective_speeches[1]


def test_eligibility_uses_only_private_positive_result_on_a_living_target() -> None:
    view = {
        "player": "P7", "role": "detective", "day": 2,
        "phase": "speak", "discussion_round": 2,
        "alive": ["P1", "P7", "P9"],
        "public_events": [],
        "private_events": [
            {"type": "investigation", "target": "P9", "is_mafia": True}
        ],
        "legal_actions": {"action_types": ["speak", "overpower"]},
    }
    opportunity = eligible_opportunity(view)
    assert opportunity is not None
    assert opportunity.confirmed_target == "P9"
    assert not hasattr(opportunity, "roles")

    view["alive"].remove("P9")
    assert eligible_opportunity(view) is None
    view["alive"].append("P9")
    view["private_events"][0]["is_mafia"] = False
    assert eligible_opportunity(view) is None
    view["private_events"][0]["is_mafia"] = True
    view["role"] = "villager"
    assert eligible_opportunity(view) is None


def test_timing_probe_is_paired_and_records_each_control() -> None:
    report, traces = run_timing_probe(70_014, 1)
    assert report["seeds"] == [70_014, 70_014]
    assert set(report["summary"]) == {"first", "last"}
    assert len(traces) == 2
    assert traces[0]["roles"] == traces[1]["roles"]
    assert [frame["target"] for frame in traces[0]["frames"][:2]] == [
        frame["target"] for frame in traces[1]["frames"][:2]
    ]
