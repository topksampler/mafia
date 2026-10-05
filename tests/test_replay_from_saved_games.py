"""The replay uses the roster and events in each trace, not a fixed table size."""

import json

from src.visualization.build_replay import read_game


def test_replay_roster_and_public_events_follow_an_eight_player_trace(tmp_path) -> None:
    players = [f"P{index}" for index in range(8)]
    view = {
        "you": "P3",
        "your_role": "mafia",
        "day": 1,
        "phase": "night",
        "alive": players,
        "public_events": [],
        "your_private_events": [],
    }
    decision = {
        "kind": "decision",
        "attempt": 1,
        "player": "P3",
        "role": "mafia",
        "day": 1,
        "phase": "night",
        "prompt": [{"role": "user", "content": json.dumps(view)}],
        "action": {"type": "kill", "target": "P6"},
        "accepted": True,
        "turn_forfeited": False,
    }
    dawn = {"day": 1, "type": "dawn", "eliminated": "P6", "revealed_role": "villager"}
    summary = {
        "kind": "summary",
        "seed": 99,
        "model": "test-model",
        "winner": "town",
        "attempts": 1,
        "public_events": [dawn],
    }
    trace = tmp_path / "eight-players.jsonl"
    trace.write_text("\n".join(json.dumps(item) for item in (decision, summary)))

    replay = read_game(trace)

    assert replay["players"] == players
    assert replay["frames"][0]["public_delta"] == [dawn]
    assert replay["roles"]["P6"] == "villager"
