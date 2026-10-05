from collections import Counter

import pytest

from env import GameConfig, MafiaEnv, Phase, Role


def enter_day(env: MafiaEnv, *, kill: str | None = None, inspect: str | None = None) -> None:
    assert env.phase == Phase.NIGHT
    mafia = next(p for p in env.players if env.roles[p] == Role.MAFIA)
    detective = next(p for p in env.players if p in env.alive and env.roles[p] == Role.DETECTIVE)
    victim = kill or next(p for p in env.players if p in env.alive and env.roles[p] == Role.VILLAGER)
    assert env.current_player == mafia
    env.step({"type": "kill", "target": victim})
    assert env.current_player == detective
    env.step({"type": "investigate", "target": inspect or mafia})


def advance_speeches(env: MafiaEnv) -> None:
    while env.phase == Phase.SPEAK:
        env.step({"type": "speak", "message": "hello"})


def test_seeded_roles_and_night_first_private_observations() -> None:
    env = MafiaEnv()
    first = env.reset(42)
    roles = dict(env.roles)
    assert env.reset(42) == first
    assert env.roles == roles
    assert Counter(env.roles.values()) == {Role.MAFIA: 1, Role.DETECTIVE: 1, Role.VILLAGER: 4}
    mafia = next(p for p, role in roles.items() if role == Role.MAFIA)
    town = next(p for p, role in roles.items() if role == Role.VILLAGER)
    assert env.phase == Phase.NIGHT
    assert env.current_player == mafia
    assert env.observe(town)["role"] == "villager"
    assert env.observe(town)["private_events"] == []
    assert env.observe(town)["legal_actions"] is None
    assert env.observe(mafia)["legal_actions"]["type"] == "kill"


def test_first_illegal_action_warns_without_advancing_turn() -> None:
    env = MafiaEnv()
    env.reset(1)
    player = env.current_player
    result = env.step({"type": "vote", "target": "P1"})
    assert not result.accepted
    assert env.current_player == player
    assert env.phase == Phase.NIGHT
    assert env.public_events[-1]["type"] == "referee_message"
    assert "player" not in env.public_events[-1]
    assert env.observe(player)["public_events"][-1] == env.public_events[-1]


def test_second_illegal_night_move_forfeits_action_without_revealing_actor() -> None:
    env = MafiaEnv()
    env.reset(1)
    mafia = env.current_player
    detective = next(p for p in env.players if env.roles[p] == Role.DETECTIVE)
    warning = env.step({"type": "vote", "target": detective})
    assert not warning.accepted and not warning.forfeited
    assert env.current_player == mafia
    dismissal = env.step({"type": "vote", "target": detective})
    assert not dismissal.accepted and dismissal.forfeited
    assert env.current_player == detective
    assert all("player" not in event for event in env.public_events)
    assert [event["severity"] for event in env.public_events] == ["warning", "forfeit"]
    env.step({"type": "investigate", "target": mafia})
    assert env.phase == Phase.SPEAK
    assert env.public_events[-1]["type"] == "dawn"
    assert env.public_events[-1]["eliminated"] is None


def test_second_illegal_speech_uses_one_speaking_slot() -> None:
    env = MafiaEnv()
    env.reset(8)
    enter_day(env)
    actor = env.current_player
    for attempt in range(2):
        result = env.step({"type": "vote", "target": actor})
        assert not result.accepted
        assert result.forfeited == (attempt == 1)
    assert env.phase == Phase.SPEAK
    assert env.current_player != actor
    while env.phase == Phase.SPEAK:
        env.step({"type": "speak", "message": ""})
    assert sum(event["type"] == "speech" and event["player"] == actor for event in env.public_events) == 2


def test_forfeited_votes_still_end_with_an_elimination() -> None:
    env = MafiaEnv(GameConfig(max_days=1))
    env.reset(7)
    enter_day(env)
    advance_speeches(env)
    result = None
    while env.phase == Phase.VOTE:
        player = env.current_player
        warning = env.step({"type": "speak", "message": ""})
        assert not warning.forfeited and env.current_player == player
        result = env.step({"type": "speak", "message": ""})
        assert result.forfeited
    assert result is not None and (result.terminated or result.truncated)
    assert any(event["type"] == "vote_tie" for event in env.public_events)
    ballot = env.public_events[-1]
    assert ballot["type"] == "vote_result"
    assert ballot["votes"] == {}
    assert ballot["tie_break"] == "seeded_draw"
    assert ballot["eliminated"] not in env.alive


def test_night_kill_then_investigation_and_dead_detective() -> None:
    env = MafiaEnv()
    env.reset(5)
    mafia = next(p for p in env.players if env.roles[p] == Role.MAFIA)
    detective = next(p for p in env.players if env.roles[p] == Role.DETECTIVE)
    env.step({"type": "kill", "target": detective})
    assert env.public_events == []
    assert env.current_player == detective  # deaths resolve at dawn
    env.step({"type": "investigate", "target": mafia})
    assert detective not in env.alive
    assert env.phase == Phase.SPEAK
    assert env.private_events[detective][-1]["is_mafia"]
    assert env.public_events[-1]["type"] == "dawn"
    assert env.public_events[-1]["eliminated"] == detective
    assert all(event["type"] != "investigation" for event in env.public_events)
    assert detective not in env._turns


def test_three_equal_speaking_turns_and_isolated_observation() -> None:
    env = MafiaEnv()
    env.reset(8)
    enter_day(env)
    survivors = set(env.alive)
    assert len(survivors) == 5
    rounds_seen = []
    while env.phase == Phase.SPEAK:
        rounds_seen.append(env.observe(env.current_player)["discussion_round"])
        env.step({"type": "speak", "message": "hello"})
    speeches = [e for e in env.public_events if e["type"] == "speech"]
    assert len(speeches) == 15
    assert Counter(e["player"] for e in speeches) == {p: 3 for p in survivors}
    assert Counter(rounds_seen) == {1: 5, 2: 5, 3: 5}
    observed = env.observe(env.current_player)
    observed["public_events"][0]["eliminated"] = "tampered"
    assert env.public_events[0]["eliminated"] != "tampered"


@pytest.mark.parametrize("message", ["", "Listen to me instead."])
def test_overpower_replaces_target_slot_and_costs_own_turn(message: str) -> None:
    env = MafiaEnv()
    env.reset(12)
    enter_day(env)
    actor = env.current_player
    target = env.legal_actions(actor)["overpower_targets"][0]
    used = Counter()
    while env.phase == Phase.SPEAK:
        player = env.current_player
        used[player] += 1
        if player == actor and used[player] == 1:
            env.step({"type": "overpower", "target": target, "message": message})
        else:
            env.step({"type": "speak", "message": ""})
    assert used[actor] == 3
    assert used[target] == 2
    assert sum(used.values()) == 14
    claims = [e for e in env.public_events if e["type"] == "floor_claim"]
    replacements = [e for e in env.public_events if e["type"] == "overpower"]
    assert len(claims) == len(replacements) == 1
    assert replacements[0]["player"] == actor
    assert replacements[0]["target"] == target
    assert replacements[0]["message"] == message


def test_vote_is_secret_until_reveal_and_eliminating_mafia_wins() -> None:
    env = MafiaEnv()
    env.reset(3)
    enter_day(env)
    mafia = next(p for p in env.alive if env.roles[p] == Role.MAFIA)
    advance_speeches(env)
    first_voter = env.current_player
    first_target = mafia if first_voter != mafia else next(p for p in env.alive if p != mafia)
    env.step({"type": "vote", "target": first_target})
    assert not any(e["type"] == "vote_result" for e in env.observe(env.current_player)["public_events"])
    while env.phase == Phase.VOTE:
        player = env.current_player
        target = mafia if player != mafia else next(p for p in env.alive if p != mafia)
        result = env.step({"type": "vote", "target": target})
    assert result.terminated
    assert result.winner == "town"
    assert result.rewards[mafia] == -1
    assert all(result.rewards[p] == 1 for p in env.players if p != mafia)
    assert env.public_events[-1]["revealed_role"] == "mafia"


def test_tied_vote_holds_runoff_among_leaders() -> None:
    env = MafiaEnv(GameConfig(max_days=1))
    env.reset(7)
    enter_day(env)
    advance_speeches(env)
    living = [p for p in env.players if p in env.alive]
    first, second, third = living[:3]
    initial_votes = {
        first: second,
        second: first,
        third: first,
        living[3]: second,
        living[4]: third,
    }
    while env.phase == Phase.VOTE:
        player = env.current_player
        if env.legal_actions(player)["round"] == "runoff":
            break
        env.step({"type": "vote", "target": initial_votes[player]})
    assert env.public_events[-1]["type"] == "vote_tie"
    assert env.public_events[-1]["tied"] == [first, second]
    assert env.legal_actions(env.current_player)["targets"] == [second]
    rejected = env.step({"type": "vote", "target": third})
    assert not rejected.accepted
    assert env.current_player == first
    assert env.public_events[-1]["type"] == "referee_message"
    assert "Illegal vote target" in env.public_events[-1]["message"]

    result = None
    while env.phase == Phase.VOTE:
        player = env.current_player
        target = second if player == first else first
        result = env.step({"type": "vote", "target": target})
    assert result is not None and (result.terminated or result.truncated)
    event = env.public_events[-1]
    assert event["type"] == "vote_result"
    assert event["round"] == "runoff"
    assert event["tie_break"] is None
    assert event["eliminated"] == first
    assert first not in env.alive


def test_tied_runoff_uses_seeded_draw_and_still_eliminates() -> None:
    def run(seed: int) -> tuple[str, dict]:
        env = MafiaEnv(GameConfig(max_days=1))
        env.reset(seed)
        enter_day(env)
        advance_speeches(env)
        living = [p for p in env.players if p in env.alive]
        result = None
        while env.phase == Phase.VOTE:
            player = env.current_player
            target = living[(living.index(player) + 1) % len(living)]
            result = env.step({"type": "vote", "target": target})
        assert result is not None and (result.terminated or result.truncated)
        event = env.public_events[-1]
        assert event["type"] == "vote_result"
        assert event["tie_break"] == "seeded_draw"
        assert event["eliminated"] not in env.alive
        assert len(env.alive) == 4
        return event["eliminated"], event

    assert run(7) == run(7)
