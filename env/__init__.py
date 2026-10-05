"""Mafia rules, with OpenEnv transport loaded only when requested."""

from .rules import GameConfig, MafiaEnv, Phase, Role


def __getattr__(name: str):
    if name == "MafiaClient":
        from .client import MafiaClient

        return MafiaClient
    if name in {"MafiaAction", "MafiaObservation"}:
        from .models import MafiaAction, MafiaObservation

        return {"MafiaAction": MafiaAction, "MafiaObservation": MafiaObservation}[name]
    raise AttributeError(name)

__all__ = [
    "GameConfig",
    "MafiaAction",
    "MafiaClient",
    "MafiaEnv",
    "MafiaObservation",
    "Phase",
    "Role",
]
