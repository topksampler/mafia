"""Typed OpenEnv contract for a single Mafia episode."""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MafiaAction(Action):
    """An action by the player named in the preceding observation."""

    player: str
    type: str = Field(description="One of kill, investigate, speak, overpower, vote")
    target: str | None = None
    message: str | None = None


class MafiaObservation(Observation):
    """The next actor's view, or a public-only view after the game ends.

    A trusted controller owns the OpenEnv client and passes this view only to
    ``current_player``. It must never broadcast private_events or role.
    """

    current_player: str | None
    role: str | None
    day: int
    phase: str
    discussion_round: int | None = None
    alive: list[str] = Field(default_factory=list)
    public_events: list[dict[str, Any]] = Field(default_factory=list)
    private_events: list[dict[str, Any]] = Field(default_factory=list)
    legal_actions: dict[str, Any] | None = None
    rewards: dict[str, float] = Field(default_factory=dict)
    winner: str | None = None
    terminated: bool = False
    truncated: bool = False
    action_accepted: bool | None = None
    turn_forfeited: bool = False
    referee_message: str | None = None
