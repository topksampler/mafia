"""Behavioral checks for the scripted Detective reveal controls."""

from env.rules import GameConfig
from src.training.scripted_reveal import (
    ScriptedPolicy,
    claim_message,
    latest_public_claim,
    play_scripted_game,
    positive_private_target,
)


def vote_view(*, public_events, legal_targets):
    return {
        "player": "P0",
        "role": "villager",
        "day": 1,
        "phase": "vote",
        "discussion_round": None,
        "alive": ["P0", "P1", "P2", "P3", "P4"],
        "public_events": public_events,
        "private_events": [],
        "legal_actions": {"type": "vote", "round": "initial", "targets": legal_targets},
    }


def test_villager_follows_latest_public_claim_without_referee_role_access() -> None:
    events = [
        {"day": 1, "type": "speech", "player": "P1", "message": claim_message("P2")},
        {"day": 1, "type": "speech", "player": "P3", "message": claim_message("P4")},
    ]
    view = vote_view(public_events=events, legal_targets=["P1", "P2", "P3", "P4"])
    assert latest_public_claim(view) == ("P3", "P4")
    action, reason, context = ScriptedPolicy("bluff", 7).act(view)
    assert action == {"type": "vote", "target": "P4"}
    assert reason == "villager_follows_public_claim"
    assert context["claimant"] == "P3"

    view["legal_actions"]["targets"] = ["P1", "P2"]
    assert latest_public_claim(view) == ("P1", "P2")


def test_detective_uses_private_positive_result_only_for_living_legal_target() -> None:
    view = vote_view(public_events=[], legal_targets=["P1", "P2", "P3", "P4"])
    view["role"] = "detective"
    view["private_events"] = [
        {"night": 1, "type": "investigation", "target": "P2", "is_mafia": True}
    ]
    assert positive_private_target(view) == "P2"
    action, reason, _ = ScriptedPolicy("private_vote", 9).act(view)
    assert action == {"type": "vote", "target": "P2"}
    assert reason == "detective_private_investigation"

    view["alive"].remove("P2")
    assert positive_private_target(view) is None


def test_hold_and_reveal_use_the_same_speech_action_with_different_messages() -> None:
    view = vote_view(public_events=[], legal_targets=["P1", "P2", "P3", "P4"])
    view.update({
        "player": "P1",
        "role": "detective",
        "phase": "speak",
        "discussion_round": 1,
        "private_events": [
            {"night": 1, "type": "investigation", "target": "P2", "is_mafia": True}
        ],
        "legal_actions": {
            "action_types": ["speak", "overpower"],
            "overpower_targets": ["P0", "P2"],
            "max_chars": 500,
        },
    })
    hold, hold_reason, _ = ScriptedPolicy("private_vote", 10).act(view)
    reveal, reveal_reason, _ = ScriptedPolicy("reveal", 10).act(view)
    assert hold == {"type": "speak", "message": ""}
    assert reveal == {"type": "speak", "message": claim_message("P2")}
    assert hold_reason == "detective_holds_result"
    assert reveal_reason == "detective_truthful_reveal"


def test_paired_games_share_initial_night_and_eight_player_traces_are_dynamic() -> None:
    config = GameConfig(players=8)
    random_game = play_scripted_game(70_001, "random", config)
    reveal_game = play_scripted_game(70_001, "reveal", config)
    assert len(random_game["players"]) == len(reveal_game["players"]) == 8
    assert random_game["roles"] == reveal_game["roles"]
    assert [
        (frame["actor"], frame["action"], frame["target"])
        for frame in random_game["frames"][:2]
    ] == [
        (frame["actor"], frame["action"], frame["target"])
        for frame in reveal_game["frames"][:2]
    ]
    assert all(frame["accepted"] for frame in reveal_game["frames"])


def test_actual_reveal_is_true_and_mafia_counterclaim_is_false() -> None:
    eligible_seed = next(
        seed for seed in range(70_000, 70_100)
        if play_scripted_game(seed, "random")["metrics"]["first_night_positive_survivor"]
    )
    reveal = play_scripted_game(eligible_seed, "reveal")
    bluff = play_scripted_game(eligible_seed, "bluff")
    assert any(claim["truthful"] for claim in reveal["metrics"]["claims"])
    assert all(
        claim["truthful"] for claim in reveal["metrics"]["claims"]
        if claim["kind"] == "detective_truthful_reveal"
    )
    assert all(
        not claim["truthful"] for claim in bluff["metrics"]["claims"]
        if claim["kind"] == "mafia_false_claim"
    )


def test_mafia_counterclaim_occurs_only_after_a_public_claim() -> None:
    sample = next(
        play_scripted_game(seed, "counterclaim")
        for seed in range(70_000, 70_100)
        if play_scripted_game(seed, "counterclaim")["metrics"]["mafia_counterclaims"]
    )
    for index, frame in enumerate(sample["frames"]):
        if frame["reason"] != "mafia_false_counterclaim":
            continue
        assert any(
            earlier["day"] == frame["day"]
            and earlier["reason"] == "detective_truthful_reveal"
            for earlier in sample["frames"][:index]
        )
        assert frame["claimed_target"] != sample["metrics"]["mafia"]
