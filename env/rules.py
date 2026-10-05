"""A deterministic, partially observable, turn-based Mafia game.

The environment owns all hidden state. Policies only receive ``observe(player)``.
Callers must use ``current_player`` to choose whose action to submit next.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Any, Mapping

RULESET_VERSION = "v0.5"
LEGACY_RULESET_VERSION = "v0.4"


class Role(str, Enum):
    MAFIA = "mafia"
    DETECTIVE = "detective"
    VILLAGER = "villager"


class Phase(str, Enum):
    SPEAK = "speak"
    VOTE = "vote"
    NIGHT = "night"
    FINISHED = "finished"


@dataclass(frozen=True)
class GameConfig:
    players: int = 6
    mafia: int = 1
    detective: int = 1
    discussion_rounds: int = 3
    discussion_mode: str = "open_floor"
    max_speeches_per_player: int = 3
    max_days: int = 5
    max_statement_chars: int = 500

    def __post_init__(self) -> None:
        if self.players < 5 or self.mafia != 1 or self.detective != 1:
            raise ValueError("This ruleset requires at least 5 players, exactly 1 mafia, and 1 detective")
        if min(self.max_days, self.max_statement_chars, self.discussion_rounds, self.max_speeches_per_player) < 1:
            raise ValueError("day, speech, and discussion limits must be positive")
        if self.discussion_mode not in ("open_floor", "scheduled"):
            raise ValueError("discussion_mode must be open_floor or scheduled")


@dataclass(frozen=True)
class StepResult:
    rewards: dict[str, float]
    terminated: bool
    truncated: bool
    winner: str | None
    accepted: bool = True
    feedback: str | None = None
    forfeited: bool = False


class MafiaEnv:
    """The Godfather/referee: keeps hidden roles and advances the night/day cycle."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.players = tuple(f"P{i}" for i in range(self.config.players))
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._rng = Random(seed)
        shuffled = list(self.players)
        self._rng.shuffle(shuffled)
        roles = (
            [Role.MAFIA]
            + [Role.DETECTIVE]
            + [Role.VILLAGER]
            * (self.config.players - self.config.mafia - self.config.detective)
        )
        self.roles = dict(zip(shuffled, roles))
        self.alive = set(self.players)
        self.day = 1
        self.phase = Phase.NIGHT
        self.current_player: str | None = None
        self.winner: str | None = None
        self.truncated = False
        self.public_events: list[dict[str, Any]] = []
        self.private_events: dict[str, list[dict[str, Any]]] = {p: [] for p in self.players}
        self._votes: dict[str, str] = {}
        self._invalid_attempts = 0
        self._vote_round = 0
        self._vote_candidates: tuple[str, ...] | None = None
        self._night_choices: dict[str, str] = {}
        self._overpowers: dict[int, tuple[str, str, str]] = {}
        self._turns: list[str] = []
        self._cursor = 0
        self._speakers_per_round = 0
        self._floor_stage: str | None = None
        self._floor_window = 0
        self._speech_credits: dict[str, int] = {}
        self._floor_requests: dict[str, str] = {}
        self._floor_grant: tuple[str, str] | None = None
        self._interrupt_bids: dict[str, str] = {}
        self._start_night()
        return self.observe(self.current_player)

    def _start_night(self) -> None:
        self.phase = Phase.NIGHT
        order = [
            p for role in (Role.MAFIA, Role.DETECTIVE)
            for p in self.players if p in self.alive and self.roles[p] == role
        ]
        self._set_turns(order)

    def _start_discussion(self) -> None:
        self.phase = Phase.SPEAK
        self._overpowers.clear()
        if self.config.discussion_mode == "open_floor":
            self._floor_window = 0
            self._speech_credits = {
                player: self.config.max_speeches_per_player
                for player in self.players if player in self.alive
            }
            self._begin_floor_requests()
            return
        order = [p for p in self.players if p in self.alive]
        self._rng.shuffle(order)
        self._speakers_per_round = len(order)
        turns = [
            player
            for round_index in range(self.config.discussion_rounds)
            for player in order[round_index:] + order[:round_index]
        ]
        self._set_turns(turns)

    def _begin_floor_requests(self) -> None:
        """Collect sealed requests; no participant sees another's draft."""
        self._floor_stage = "request"
        self._floor_grant = None
        self._floor_requests.clear()
        self._interrupt_bids.clear()
        eligible = [
            player for player in self.players
            if player in self.alive and self._speech_credits[player] > 0
        ]
        if not eligible:
            self._begin_vote()
            return
        self._rng.shuffle(eligible)
        self._set_turns(eligible)

    def _resolve_floor_requests(self) -> None:
        if not self._floor_requests:
            self._begin_vote()
            return
        speaker = self._rng.choice(list(self._floor_requests))
        self._floor_grant = (speaker, self._floor_requests[speaker])
        self._speech_credits[speaker] -= 1
        self.public_events.append({
            "day": self.day, "type": "floor_grant", "window": self._floor_window,
            "player": speaker,
        })
        interrupters = [
            player for player in self.players
            if player in self.alive and player != speaker and self._speech_credits[player] > 0
        ]
        if not interrupters:
            self._resolve_interrupts()
            return
        self._floor_stage = "interrupt"
        self._rng.shuffle(interrupters)
        self._set_turns(interrupters)

    def _resolve_interrupts(self) -> None:
        assert self._floor_grant is not None
        speaker, message = self._floor_grant
        for actor in self._interrupt_bids:
            self._speech_credits[actor] -= 1
        if self._interrupt_bids:
            actor = self._rng.choice(list(self._interrupt_bids))
            self.public_events.append({
                "day": self.day, "type": "overpower", "window": self._floor_window,
                "player": actor, "target": speaker,
                "message": self._interrupt_bids[actor],
            })
        else:
            self.public_events.append({
                "day": self.day, "type": "speech", "window": self._floor_window,
                "player": speaker, "message": message,
            })
        self._floor_window += 1
        self._begin_floor_requests()

    def _set_turns(self, players: list[str]) -> None:
        self._turns = players
        self._cursor = 0
        self.current_player = players[0] if players else None
        self._invalid_attempts = 0

    def _advance(self) -> bool:
        self._cursor += 1
        if self._cursor < len(self._turns):
            self.current_player = self._turns[self._cursor]
            return True
        self.current_player = None
        return False

    def _advance_discussion(self) -> bool:
        self._cursor += 1
        while self._cursor < len(self._turns):
            if self._cursor in self._overpowers:
                actor, target, message = self._overpowers.pop(self._cursor)
                self.public_events.append({
                    "day": self.day,
                    "round": self._cursor // self._speakers_per_round + 1,
                    "type": "overpower",
                    "player": actor,
                    "target": target,
                    "message": message,
                })
                self._cursor += 1
                continue
            self.current_player = self._turns[self._cursor]
            return True
        self.current_player = None
        return False

    def _next_open_slot(self, target: str) -> int | None:
        return next(
            (i for i in range(self._cursor + 1, len(self._turns))
             if self._turns[i] == target and i not in self._overpowers),
            None,
        )

    def legal_actions(self, player: str) -> dict[str, Any]:
        if player != self.current_player or self.phase == Phase.FINISHED:
            raise ValueError("It is not this player's turn")
        if self.phase == Phase.SPEAK:
            if self.config.discussion_mode == "open_floor":
                common = {
                    "stage": self._floor_stage,
                    "max_chars": self.config.max_statement_chars,
                    "speech_credits": self._speech_credits[player],
                }
                if self._floor_stage == "request":
                    return {**common, "action_types": ["wait", "request_floor"]}
                assert self._floor_grant is not None
                return {
                    **common,
                    "action_types": ["allow", "overpower"],
                    "floor_target": self._floor_grant[0],
                    "overpower_targets": [self._floor_grant[0]],
                }
            return {
                "action_types": ["speak", "overpower"],
                "max_chars": self.config.max_statement_chars,
                "overpower_targets": [
                    p for p in self.players
                    if p in self.alive and p != player and self._next_open_slot(p) is not None
                ],
            }
        if self.phase == Phase.VOTE:
            candidates = self._vote_candidates or tuple(p for p in self.players if p in self.alive)
            return {
                "type": "vote",
                "round": "runoff" if self._vote_round else "initial",
                "targets": [p for p in candidates if p != player],
            }
        role = self.roles[player]
        return {
            "type": {Role.MAFIA: "kill", Role.DETECTIVE: "investigate"}[role],
            "targets": [p for p in self.players if p in self.alive and p != player],
        }

    def observe(self, player: str | None) -> dict[str, Any]:
        if player is None or player not in self.roles:
            raise ValueError("Unknown player")
        return {
            "player": player,
            "role": self.roles[player].value,
            "day": self.day,
            "phase": self.phase.value,
            "discussion_round": (self._cursor // self._speakers_per_round + 1) if self.phase == Phase.SPEAK else None,
            "floor_stage": self._floor_stage if self.phase == Phase.SPEAK and self.config.discussion_mode == "open_floor" else None,
            "floor_window": self._floor_window if self.phase == Phase.SPEAK and self.config.discussion_mode == "open_floor" else None,
            "speech_credits": deepcopy(self._speech_credits) if self.phase == Phase.SPEAK and self.config.discussion_mode == "open_floor" else None,
            "alive": [p for p in self.players if p in self.alive],
            "public_events": deepcopy(self.public_events),
            "private_events": deepcopy(self.private_events[player]),
            "legal_actions": self.legal_actions(player) if player == self.current_player else None,
        }

    def reject(self, reason: str) -> StepResult:
        """Warn once, then forfeit the same turn after another invalid move.

        Night announcements omit the actor and attempted target so private
        roles and choices stay hidden.
        """
        player = self.current_player
        if player is None:
            raise ValueError("Game is finished")
        self._invalid_attempts += 1
        forfeited = self._invalid_attempts >= 2
        if self.phase == Phase.NIGHT:
            message = (
                "Godfather: The night turn is forfeited after another illegal move. No action was taken."
                if forfeited else
                "Godfather's final warning: An illegal night move was rejected. "
                "Another illegal move will forfeit this turn."
            )
            event = {"day": self.day, "type": "referee_message", "severity": "forfeit" if forfeited else "warning", "message": message}
        else:
            message = (
                f"Godfather: {player}'s turn is forfeited after another illegal move. No action was taken."
                if forfeited else
                f"Godfather's final warning to {player}: {reason}. "
                "Another illegal move will forfeit this turn."
            )
            event = {"day": self.day, "type": "referee_message", "severity": "forfeit" if forfeited else "warning", "player": player, "message": message}
        self.public_events.append(event)
        if forfeited:
            self._invalid_attempts = 0
            self._forfeit_turn()
        return self._result(accepted=False, feedback=message, forfeited=forfeited)

    def _forfeit_turn(self) -> None:
        if self.phase == Phase.SPEAK:
            if not self._advance_discussion():
                self._begin_vote()
        elif self.phase == Phase.VOTE:
            if not self._advance():
                self._resolve_votes()
        else:
            if not self._advance():
                self._resolve_night()

    def _begin_vote(self) -> None:
        self.phase = Phase.VOTE
        self._vote_round = 0
        self._vote_candidates = None
        self._set_turns([p for p in self.players if p in self.alive])

    def step(self, action: Mapping[str, Any]) -> StepResult:
        try:
            result = self._step_validated(action)
        except ValueError as error:
            return self.reject(str(error))
        self._invalid_attempts = 0
        return result

    def _step_validated(self, action: Mapping[str, Any]) -> StepResult:
        player = self.current_player
        if player is None:
            raise ValueError("Game is finished")
        legal = self.legal_actions(player)
        if not isinstance(action, Mapping):
            raise ValueError("Action must be a mapping")

        if self.phase == Phase.SPEAK:
            if action.get("type") not in legal["action_types"]:
                raise ValueError("Expected a speak or overpower action")
            message = action.get("message")
            if not isinstance(message, str) or len(message) > self.config.max_statement_chars:
                raise ValueError("Speech must be a string within max_chars")
            if action["type"] == "speak":
                self.public_events.append({
                    "day": self.day,
                    "round": self._cursor // self._speakers_per_round + 1,
                    "type": "speech",
                    "player": player,
                    "message": message,
                })
            else:
                target = action.get("target")
                if target not in legal["overpower_targets"]:
                    raise ValueError("Illegal overpower target")
                slot = self._next_open_slot(target)
                assert slot is not None
                self._overpowers[slot] = (player, target, message)
                self.public_events.append({
                    "day": self.day,
                    "round": self._cursor // self._speakers_per_round + 1,
                    "type": "floor_claim",
                    "player": player,
                    "target": target,
                })
            if not self._advance_discussion():
                self._begin_vote()

        elif self.phase == Phase.VOTE:
            if action.get("type") != legal["type"]:
                raise ValueError("Expected vote action")
            target = action.get("target")
            if target not in legal["targets"]:
                raise ValueError("Illegal vote target")
            self._votes[player] = target
            if not self._advance():
                self._resolve_votes()

        else:
            if action.get("type") != legal["type"]:
                raise ValueError(f"Expected {legal['type']} action")
            target = action.get("target")
            if target not in legal["targets"]:
                raise ValueError("Illegal night target")
            self._night_choices[legal["type"]] = target
            if legal["type"] == "investigate":
                self.private_events[player].append({
                    "night": self.day,
                    "type": "investigation",
                    "target": target,
                    "is_mafia": self.roles[target] == Role.MAFIA,
                })
            if not self._advance():
                self._resolve_night()

        return self._result()

    def _result(
        self, *, accepted: bool = True, feedback: str | None = None, forfeited: bool = False
    ) -> StepResult:
        done = self.phase == Phase.FINISHED
        rewards = {p: (1.0 if self._team(p) == self.winner else -1.0) if self.winner else 0.0 for p in self.players}
        return StepResult(rewards, done and not self.truncated, self.truncated, self.winner, accepted, feedback, forfeited)

    def _team(self, player: str) -> str:
        return "mafia" if self.roles[player] == Role.MAFIA else "town"

    def _check_winner(self) -> bool:
        mafia = sum(self.roles[p] == Role.MAFIA for p in self.alive)
        town = len(self.alive) - mafia
        if mafia == 0:
            self.winner = "town"
        elif mafia >= town:
            self.winner = "mafia"
        if self.winner:
            self.phase = Phase.FINISHED
            self.current_player = None
            return True
        return False

    def _resolve_votes(self) -> None:
        counts = Counter(self._votes.values())
        max_votes = max(counts.values(), default=0)
        candidates = self._vote_candidates or tuple(p for p in self.players if p in self.alive)
        leaders = [p for p in candidates if counts[p] == max_votes] if max_votes else list(candidates)
        if not leaders:
            raise RuntimeError("A completed vote must have at least one candidate")
        ballots = dict(self._votes)
        self._votes.clear()
        if len(leaders) > 1 and self._vote_round == 0:
            self.public_events.append({
                "day": self.day,
                "type": "vote_tie",
                "votes": ballots,
                "tied": leaders,
            })
            self._vote_round = 1
            self._vote_candidates = tuple(leaders)
            self._set_turns([p for p in self.players if p in self.alive])
            return

        tie_break = "seeded_draw" if len(leaders) > 1 else None
        eliminated = self._rng.choice(leaders) if tie_break else leaders[0]
        event: dict[str, Any] = {
            "day": self.day,
            "type": "vote_result",
            "round": "runoff" if self._vote_round else "initial",
            "votes": ballots,
            "tied": leaders if tie_break else [],
            "tie_break": tie_break,
            "eliminated": eliminated,
            "revealed_role": self.roles[eliminated].value,
        }
        self.public_events.append(event)
        self._vote_round = 0
        self._vote_candidates = None
        self.alive.remove(eliminated)
        if self._check_winner():
            return
        if self.day >= self.config.max_days:
            self.truncated = True
            self.phase = Phase.FINISHED
            self.current_player = None
            return
        self.day += 1
        self._start_night()

    def _resolve_night(self) -> None:
        eliminated = self._night_choices.get("kill")
        self.public_events.append({
            "day": self.day,
            "type": "dawn",
            "eliminated": eliminated,
            "revealed_role": self.roles[eliminated].value if eliminated else None,
        })
        self._night_choices.clear()
        if eliminated:
            self.alive.remove(eliminated)
        if self._check_winner():
            return
        self._start_discussion()
