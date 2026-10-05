"""The model adapter must preserve private views and referee feedback."""

import json

from env.environment import MafiaEnvironment
from policy import build_messages, parse_action


def test_prompt_contains_only_the_active_players_view() -> None:
    environment = MafiaEnvironment()
    observation = environment.reset(seed=1)
    prompt = json.loads(build_messages(observation)[1]["content"])

    assert prompt["you"] == observation.current_player
    assert prompt["your_role"] == observation.role
    assert prompt["your_private_events"] == observation.private_events
    assert prompt["public_events"] == observation.public_events
    assert "roles" not in prompt
    assert "rewards" not in prompt


def test_malformed_reply_gets_warning_then_forfeits_same_turn() -> None:
    environment = MafiaEnvironment()
    observation = environment.reset(seed=1)
    player = observation.current_player
    assert player is not None

    warning = environment.step(parse_action("not JSON", player))
    assert warning.current_player == player
    assert warning.action_accepted is False
    assert warning.turn_forfeited is False

    dismissal = environment.step(parse_action("still not JSON", player))
    assert dismissal.current_player != player
    assert dismissal.action_accepted is False
    assert dismissal.turn_forfeited is True
