"""Typed WebSocket client for the OpenEnv Mafia server."""

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from .models import MafiaAction, MafiaObservation


class MafiaClient(EnvClient[MafiaAction, MafiaObservation, State]):
    """One client controls one game and routes views to the active player."""

    def _step_payload(self, action: MafiaAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[MafiaObservation]:
        observation_data = dict(payload["observation"])
        # OpenEnv puts these base Observation fields on the response envelope.
        observation_data["done"] = payload.get("done", observation_data.get("done", False))
        observation_data["reward"] = payload.get("reward", observation_data.get("reward"))
        observation = MafiaObservation.model_validate(observation_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State.model_validate(payload)
