"""Expose the Godfather rules engine through OpenEnv's Environment API."""

from copy import deepcopy
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .models import MafiaAction, MafiaObservation
from .rules import MafiaEnv as RulesEngine
from .rules import Phase, StepResult as GameStepResult


class MafiaEnvironment(Environment[MafiaAction, MafiaObservation, State]):
    """A session-local game. Only the controller receives private observations."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._game = RulesEngine()
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs: object
    ) -> MafiaObservation:
        self._game.reset(seed=seed)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return self._observation()

    def step(
        self, action: MafiaAction, timeout_s: float | None = None, **kwargs: object
    ) -> MafiaObservation:
        actor = self._game.current_player
        if actor is None:
            raise ValueError("Game is finished")
        if action.player != actor:
            result = self._game.reject("Action submitted for the wrong player")
        else:
            result = self._game.step(
                action.model_dump(exclude={"player", "metadata"}, exclude_none=True)
            )
        if result.accepted or result.forfeited:
            self._state.step_count += 1
        return self._observation(result, actor)

    @property
    def state(self) -> State:
        # OpenEnv's /state endpoint must not disclose roles or night actors.
        return self._state

    def _observation(
        self, result: GameStepResult | None = None, actor: str | None = None
    ) -> MafiaObservation:
        current = self._game.current_player
        view = self._game.observe(current) if current is not None else None
        rewards = result.rewards if result is not None else {p: 0.0 for p in self._game.players}
        terminated = result.terminated if result is not None else False
        truncated = result.truncated if result is not None else False
        return MafiaObservation(
            current_player=current,
            role=view["role"] if view else None,
            day=self._game.day,
            phase=self._game.phase.value,
            discussion_round=view["discussion_round"] if view else None,
            alive=view["alive"] if view else [p for p in self._game.players if p in self._game.alive],
            public_events=view["public_events"] if view else deepcopy(self._game.public_events),
            private_events=view["private_events"] if view else [],
            legal_actions=view["legal_actions"] if view else None,
            rewards=rewards,
            winner=self._game.winner,
            terminated=terminated,
            truncated=truncated,
            done=self._game.phase == Phase.FINISHED,
            reward=rewards[actor] if actor is not None else 0.0,
            action_accepted=result.accepted if result is not None else None,
            turn_forfeited=result.forfeited if result is not None else False,
            referee_message=result.feedback if result is not None else None,
        )
