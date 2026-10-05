"""Check OpenEnv's wire contract and the game's private-information boundary."""

from fastapi.testclient import TestClient

from env.client import MafiaClient
from env.models import MafiaAction
from env.server import app
from env.environment import MafiaEnvironment


def test_openenv_turn_routing_and_state_privacy() -> None:
    env = MafiaEnvironment()
    first = env.reset(seed=42)
    assert first.phase == "night"
    assert first.role == "mafia"
    assert first.legal_actions["type"] == "kill"
    assert set(env.state.model_dump()) == {"episode_id", "step_count"}

    rejected = env.step(MafiaAction(player="wrong", type="kill", target="P0"))
    assert rejected.action_accepted is False
    assert rejected.current_player == first.current_player
    assert rejected.public_events[-1]["type"] == "referee_message"
    assert "player" not in rejected.public_events[-1]
    assert rejected.referee_message == rejected.public_events[-1]["message"]
    assert env.state.step_count == 0

    target = first.legal_actions["targets"][0]
    detective = env.step(MafiaAction(player=first.current_player, type="kill", target=target))
    assert detective.role == "detective"
    assert detective.public_events == rejected.public_events
    assert detective.private_events == []
    assert env.state.step_count == 1

    day = env.step(
        MafiaAction(
            player=detective.current_player,
            type="investigate",
            target=first.current_player,
        )
    )
    assert day.phase == "speak"
    assert any(event["type"] == "dawn" for event in day.public_events)
    assert not any(event["type"] == "investigation" for event in day.public_events)
    if day.role != "detective":
        assert day.private_events == []

    observation = day
    while observation.phase == "speak":
        observation = env.step(
            MafiaAction(player=observation.current_player, type="speak", message="")
        )
    while observation.phase == "vote":
        voter = observation.current_player
        vote = first.current_player if voter != first.current_player else next(
            player for player in observation.alive if player != voter
        )
        observation = env.step(MafiaAction(player=voter, type="vote", target=vote))
        final_actor = voter
    assert observation.done and observation.terminated
    assert observation.winner == "town"
    assert observation.current_player is None
    assert observation.role is None
    assert observation.private_events == []
    assert observation.reward == observation.rewards[final_actor]
    assert observation.rewards[first.current_player] == -1.0


def test_openenv_websocket_contract_and_client_parsing() -> None:
    with TestClient(app) as server, server.websocket_connect("/ws") as socket:
        socket.send_json({"type": "reset", "data": {"seed": 42}})
        reset = socket.receive_json()
        assert reset["type"] == "observation"
        client = MafiaClient(base_url="http://localhost:8000")
        first = client._parse_result(reset["data"])
        assert first.observation.role == "mafia"
        assert first.observation.reward == 0.0
        assert first.observation.done is False

        socket.send_json({"type": "state"})
        state = socket.receive_json()["data"]
        assert set(state) == {"episode_id", "step_count"}

        socket.send_json({
            "type": "step",
            "data": {"player": first.observation.current_player, "type": "vote", "target": "P0"},
        })
        rejected = client._parse_result(socket.receive_json()["data"])
        assert rejected.observation.action_accepted is False
        assert rejected.observation.current_player == first.observation.current_player
        assert rejected.observation.public_events[-1]["type"] == "referee_message"
        socket.send_json({"type": "state"})
        assert socket.receive_json()["data"]["step_count"] == 0

        action = MafiaAction(
            player=first.observation.current_player,
            type="kill",
            target=first.observation.legal_actions["targets"][0],
        )
        socket.send_json({"type": "step", "data": client._step_payload(action)})
        response = socket.receive_json()
        assert response["type"] == "observation"
        next_turn = client._parse_result(response["data"])
        assert next_turn.observation.role == "detective"
        assert next_turn.observation.public_events == rejected.observation.public_events
        socket.send_json({"type": "state"})
        assert socket.receive_json()["data"]["step_count"] == 1


def test_openenv_second_invalid_action_forfeits_turn() -> None:
    environment = MafiaEnvironment()
    first = environment.reset(seed=42)
    invalid = MafiaAction(player=first.current_player, type="vote", target="P0")
    warning = environment.step(invalid)
    assert not warning.action_accepted and not warning.turn_forfeited
    assert environment.state.step_count == 0
    dismissal = environment.step(invalid)
    assert not dismissal.action_accepted and dismissal.turn_forfeited
    assert dismissal.current_player != first.current_player
    assert dismissal.phase == "night"
    assert environment.state.step_count == 1
    assert [event["severity"] for event in dismissal.public_events] == ["warning", "forfeit"]
