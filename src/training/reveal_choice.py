"""Action boundary for a future learned Detective reveal/hold policy.

This module does not decide *when* to reveal. The controller calls the policy
at every eligible Detective speech turn, including after earlier claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from src.training.scripted_reveal import claim_message


class RevealChoice(str, Enum):
    REVEAL = "reveal"
    HOLD = "hold"


@dataclass(frozen=True)
class RevealOpportunity:
    player: str
    day: int
    discussion_round: int
    alive: tuple[str, ...]
    public_events: tuple[dict[str, Any], ...]
    private_events: tuple[dict[str, Any], ...]
    confirmed_target: str


@dataclass(frozen=True)
class RevealDecision:
    opportunity: RevealOpportunity
    choice: RevealChoice
    action: dict[str, str]


class RevealPolicy(Protocol):
    def choose(self, opportunity: RevealOpportunity, *, sample: bool) -> RevealChoice: ...


def eligible_opportunity(view: dict[str, Any]) -> RevealOpportunity | None:
    """Derive eligibility solely from the active player's observation."""
    if view.get("role") != "detective" or view.get("phase") != "speak":
        return None
    if "speak" not in (view.get("legal_actions") or {}).get("action_types", []):
        return None
    player = view["player"]
    alive = tuple(view["alive"])
    if player not in alive:
        return None
    target = next(
        (
            event["target"]
            for event in reversed(view["private_events"])
            if event.get("type") == "investigation"
            and event.get("is_mafia") is True
            and event.get("target") in alive
        ),
        None,
    )
    if target is None:
        return None
    return RevealOpportunity(
        player=player,
        day=view["day"],
        discussion_round=view["discussion_round"],
        alive=alive,
        public_events=tuple(view["public_events"]),
        private_events=tuple(view["private_events"]),
        confirmed_target=target,
    )


def choose_reveal_action(
    view: dict[str, Any], policy: RevealPolicy, *, sample: bool
) -> RevealDecision:
    """Ask the policy on this turn; both answers spend exactly one speech slot."""
    opportunity = eligible_opportunity(view)
    if opportunity is None:
        raise ValueError("No eligible Detective reveal choice on this turn")
    choice = RevealChoice(policy.choose(opportunity, sample=sample))
    message = claim_message(opportunity.confirmed_target) if choice == RevealChoice.REVEAL else ""
    return RevealDecision(
        opportunity=opportunity,
        choice=choice,
        action={"type": "speak", "message": message},
    )
