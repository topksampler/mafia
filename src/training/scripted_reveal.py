"""Paired scripted controls for public Detective information sharing.

No model is called or trained here. Each policy receives only the active
player's observation; the referee's roles are used after actions for metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
import json
from pathlib import Path
from random import Random
import re
from typing import Any

from env.rules import GameConfig, MafiaEnv, RULESET_VERSION, Role
from src.training.vote import VoteObservation, random_action


ARMS = ("random", "private_vote", "reveal", "counterclaim", "bluff")
CLAIM_PATTERN = re.compile(
    r"^I am the Detective\. I investigated (P\d+); \1 is Mafia\. Vote \1\.$"
)


def _random_for(seed: int, view: dict[str, Any]) -> Random:
    """Separate random streams keep matching decisions paired across arms."""
    key = (
        seed, view["player"], view["day"], view["phase"],
        view["discussion_round"], view["legal_actions"].get("round"),
    )
    digest = blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    return Random(int.from_bytes(digest, "big"))


def _observation(view: dict[str, Any]) -> VoteObservation:
    return VoteObservation(
        current_player=view["player"],
        role=view["role"],
        day=view["day"],
        phase=view["phase"],
        alive=view["alive"],
        public_events=view["public_events"],
        private_events=view["private_events"],
        legal_actions=view["legal_actions"],
    )


def positive_private_target(view: dict[str, Any]) -> str | None:
    """Detective sees only its own private answers."""
    if view["role"] != Role.DETECTIVE.value:
        return None
    return next(
        (
            event["target"]
            for event in reversed(view["private_events"])
            if event.get("type") == "investigation"
            and event.get("is_mafia") is True
            and event["target"] in view["alive"]
        ),
        None,
    )


def latest_public_claim(view: dict[str, Any]) -> tuple[str, str] | None:
    """A Villager recognizes the exact scripted claim, without role access."""
    legal_targets = set(view["legal_actions"]["targets"])
    for event in reversed(view["public_events"]):
        if event.get("day") != view["day"] or event.get("type") != "speech":
            continue
        speaker = event.get("player")
        if speaker not in view["alive"]:
            continue
        match = CLAIM_PATTERN.fullmatch(event.get("message", ""))
        if match and match.group(1) in legal_targets:
            return speaker, match.group(1)
    return None


def claim_message(target: str) -> str:
    return f"I am the Detective. I investigated {target}; {target} is Mafia. Vote {target}."


@dataclass
class ScriptedPolicy:
    arm: str
    seed: int
    reveal_round: int | None = None

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"Unknown experiment arm: {self.arm}")
        if self.reveal_round is not None and self.reveal_round < 1:
            raise ValueError("reveal_round must be positive")
        self.revealed: set[tuple[int, str]] = set()
        self.bluffed: set[int] = set()

    def act(self, view: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
        role, phase = view["role"], view["phase"]
        legal = view["legal_actions"]
        private_target = positive_private_target(view)

        if phase == "speak" and self.arm in ("private_vote", "reveal", "counterclaim", "bluff"):
            if role == Role.DETECTIVE.value and private_target is not None:
                key = (view["day"], private_target)
                if self.reveal_round is not None:
                    if key not in self.revealed and view["discussion_round"] >= self.reveal_round:
                        self.revealed.add(key)
                        return (
                            {"type": "speak", "message": claim_message(private_target)},
                            "detective_truthful_reveal",
                            {"claimed_target": private_target},
                        )
                    return (
                        {"type": "speak", "message": ""},
                        "detective_timing_hold",
                        {"known_target": private_target},
                    )
                if key not in self.revealed:
                    self.revealed.add(key)
                    if self.arm == "private_vote":
                        return (
                            {"type": "speak", "message": ""},
                            "detective_holds_result",
                            {"known_target": private_target},
                        )
                    return (
                        {"type": "speak", "message": claim_message(private_target)},
                        "detective_truthful_reveal",
                        {"claimed_target": private_target},
                    )
            can_counterclaim = any(
                event.get("day") == view["day"]
                and event.get("type") == "speech"
                and CLAIM_PATTERN.fullmatch(event.get("message", ""))
                for event in view["public_events"]
            )
            if (
                self.arm in ("counterclaim", "bluff")
                and role == Role.MAFIA.value
                and view["day"] not in self.bluffed
                and (self.arm == "bluff" or can_counterclaim)
            ):
                self.bluffed.add(view["day"])
                target = _random_for(self.seed, view).choice(
                    [player for player in view["alive"] if player != view["player"]]
                )
                return (
                    {"type": "speak", "message": claim_message(target)},
                    "mafia_false_counterclaim" if self.arm == "counterclaim" else "mafia_false_claim",
                    {"claimed_target": target},
                )

        if phase == "vote" and self.arm != "random":
            if role == Role.DETECTIVE.value and private_target in legal["targets"]:
                return (
                    {"type": "vote", "target": private_target},
                    "detective_private_investigation",
                    {"known_target": private_target},
                )
            if role == Role.VILLAGER.value and self.arm in ("reveal", "counterclaim", "bluff"):
                claim = latest_public_claim(view)
                if claim is not None:
                    speaker, target = claim
                    return (
                        {"type": "vote", "target": target},
                        "villager_follows_public_claim",
                        {"claimant": speaker, "claimed_target": target},
                    )

        return (
            random_action(_observation(view), _random_for(self.seed, view)),
            "seeded_random",
            {},
        )


def play_scripted_game(
    seed: int,
    arm: str,
    config: GameConfig | None = None,
    *,
    reveal_round: int | None = None,
) -> dict[str, Any]:
    """Return a replay-ready complete game and evaluator-only metrics."""
    environment = MafiaEnv(config)
    environment.reset(seed=seed)
    policy = ScriptedPolicy(arm, seed, reveal_round)
    frames: list[dict[str, Any]] = []
    attempts = 0
    while environment.current_player is not None:
        attempts += 1
        if attempts > 300:
            raise RuntimeError(f"Game {seed} ({arm}) exceeded 300 actions")
        actor = environment.current_player
        view = environment.observe(actor)
        before = len(view["public_events"])
        action, reason, context = policy.act(view)
        result = environment.step(action)
        if not result.accepted:
            raise RuntimeError(f"Scripted policy made an illegal move: {seed} {arm} {reason}")
        frames.append({
            "step": attempts,
            "day": view["day"],
            "phase": view["phase"],
            "round": view["discussion_round"],
            "actor": actor,
            "actor_role": view["role"],
            "alive": view["alive"],
            "action": action["type"],
            "target": action.get("target"),
            "message": action.get("message", ""),
            "accepted": True,
            "forfeited": False,
            "referee_message": None,
            "private": view["private_events"],
            "public_delta": environment.public_events[before:],
            "reason": reason,
            **context,
        })

    roles = {player: role.value for player, role in environment.roles.items()}
    mafia = next(player for player, role in environment.roles.items() if role == Role.MAFIA)
    detective = next(player for player, role in environment.roles.items() if role == Role.DETECTIVE)
    claims = [
        {
            "day": frame["day"],
            "speaker": frame["actor"],
            "target": frame["claimed_target"],
            "truthful": frame["claimed_target"] == mafia,
            "kind": frame["reason"],
        }
        for frame in frames
        if frame["reason"] in ("detective_truthful_reveal", "mafia_false_claim", "mafia_false_counterclaim")
    ]
    town_votes = [
        frame for frame in frames
        if frame["action"] == "vote" and frame["actor_role"] != Role.MAFIA.value
    ]
    villager_votes = [
        frame for frame in town_votes if frame["actor_role"] == Role.VILLAGER.value
    ]
    first_night_positive_survivor = any(
        frame["day"] == 1
        and frame["phase"] == "night"
        and frame["actor"] == detective
        and frame["target"] == mafia
        and any(
            event.get("type") == "dawn" and event.get("eliminated") != detective
            for event in frame["public_delta"]
        )
        for frame in frames
    )
    metrics = {
        "seed": seed,
        "arm": arm,
        "reveal_round": reveal_round,
        "winner": environment.winner,
        "town_win": int(environment.winner == "town"),
        "mafia": mafia,
        "detective": detective,
        "first_night_positive_survivor": first_night_positive_survivor,
        "truthful_reveals": sum(claim["kind"] == "detective_truthful_reveal" for claim in claims),
        "mafia_bluffs": sum(claim["kind"] in ("mafia_false_claim", "mafia_false_counterclaim") for claim in claims),
        "mafia_counterclaims": sum(claim["kind"] == "mafia_false_counterclaim" for claim in claims),
        "town_votes": len(town_votes),
        "town_votes_for_mafia": sum(frame["target"] == mafia for frame in town_votes),
        "villager_votes": len(villager_votes),
        "villager_votes_for_mafia": sum(frame["target"] == mafia for frame in villager_votes),
        "villager_claim_follow_votes": sum(
            frame["reason"] == "villager_follows_public_claim" for frame in villager_votes
        ),
        "villager_claim_follow_correct": sum(
            frame["reason"] == "villager_follows_public_claim" and frame["target"] == mafia
            for frame in villager_votes
        ),
        "claims": claims,
    }
    return {
        "seed": seed,
        "arm": arm,
        "reveal_round": reveal_round,
        "model": f"scripted_{arm}" + (f"_round{reveal_round}" if reveal_round is not None else ""),
        "ruleset_version": RULESET_VERSION,
        "winner": environment.winner,
        "attempts": attempts,
        "players": list(environment.players),
        "roles": roles,
        "frames": frames,
        "metrics": metrics,
    }


def run_experiment(start_seed: int, games: int, players: int = 6) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if games < 1:
        raise ValueError("games must be positive")
    config = GameConfig(players=players)
    traces = [
        play_scripted_game(seed, arm, config)
        for seed in range(start_seed, start_seed + games)
        for arm in ARMS
    ]
    per_seed = {
        str(seed): {arm: next(
            trace["metrics"] for trace in traces
            if trace["seed"] == seed and trace["arm"] == arm
        ) for arm in ARMS}
        for seed in range(start_seed, start_seed + games)
    }
    summaries = {}
    for arm in ARMS:
        rows = [per_seed[str(seed)][arm] for seed in range(start_seed, start_seed + games)]
        totals = Counter()
        for row in rows:
            totals.update({key: row[key] for key in (
                "town_win", "truthful_reveals", "mafia_bluffs", "mafia_counterclaims", "town_votes",
                "town_votes_for_mafia", "villager_votes", "villager_votes_for_mafia",
                "villager_claim_follow_votes", "villager_claim_follow_correct",
            )})
        summaries[arm] = dict(totals)
    paired = {}
    for left, right in (
        ("random", "private_vote"),
        ("private_vote", "reveal"),
        ("reveal", "counterclaim"),
        ("reveal", "bluff"),
    ):
        changes = {
            "town_win_to_loss": [],
            "town_loss_to_win": [],
            "same_outcome": [],
        }
        for seed in range(start_seed, start_seed + games):
            before, after = per_seed[str(seed)][left], per_seed[str(seed)][right]
            key = (
                "town_win_to_loss" if before["town_win"] and not after["town_win"]
                else "town_loss_to_win" if not before["town_win"] and after["town_win"]
                else "same_outcome"
            )
            changes[key].append(seed)
        paired[f"{left}_to_{right}"] = changes
    eligible = [
        seed for seed in range(start_seed, start_seed + games)
        if per_seed[str(seed)]["random"]["first_night_positive_survivor"]
    ]
    report = {
        "experiment": "scripted_detective_reveal",
        "description": {
            "random": "Seeded random actions for all roles, including speech and votes.",
            "private_vote": "Detective votes for a privately confirmed living Mafia but holds the result in an empty first speech slot; other speech and votes stay random.",
            "reveal": "Detective also publicly reveals a positive result once per day; Villagers follow the latest living claimant's current-day positive claim when legal.",
            "counterclaim": "The reveal arm plus one Mafia false claim per day, only after a public Detective-style claim appears; Villagers use the same rule.",
            "bluff": "The reveal arm plus one false Detective claim per day from Mafia; Villagers use the same claim-following rule.",
        },
        "model_calls": 0,
        "training_updates": 0,
        "ruleset_version": RULESET_VERSION,
        "players": players,
        "seeds": [start_seed, start_seed + games - 1],
        "games_per_arm": games,
        "first_night_positive_survivor_seeds": eligible,
        "summary": summaries,
        "paired_outcomes": paired,
        "per_seed": per_seed,
    }
    return report, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired scripted Detective reveal controls")
    parser.add_argument("--start-seed", type=int, default=70_000)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("runs") / (
        f"scripted-reveal-seeds-{args.start_seed}-{args.start_seed + args.games - 1}"
    )
    report, traces = run_experiment(args.start_seed, args.games, args.players)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output / "games.jsonl").open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "report": str(output / "report.json"),
        "traces": str(output / "games.jsonl"),
        "games_per_arm": args.games,
        "summary": report["summary"],
        "eligible_first_night_reveal_seeds": len(report["first_night_positive_survivor_seeds"]),
        "paired_outcome_counts": {
            name: {kind: len(seeds) for kind, seeds in changes.items()}
            for name, changes in report["paired_outcomes"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
