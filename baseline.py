"""Seeded OpenEnv rollouts and paired evaluation for simple player policies.

The controller sees each private observation only long enough to route it to
the active policy. Policies never receive the referee's full game state.
"""

from dataclasses import dataclass
import json
from math import sqrt
from random import Random
from typing import Callable, Protocol

from env.environment import MafiaEnvironment
from env.models import MafiaAction, MafiaObservation


class Policy(Protocol):
    def act(self, observation: MafiaObservation) -> MafiaAction: ...


PolicyFactory = Callable[[int], Policy]


class RandomPolicy:
    """Legal random actions, including occasional public claims and floor grabs."""

    def __init__(self, seed: int) -> None:
        self.rng = Random(seed)

    def act(self, observation: MafiaObservation) -> MafiaAction:
        player = observation.current_player
        legal = observation.legal_actions
        if player is None or legal is None:
            raise ValueError("A policy can act only on its own live turn")
        if observation.phase == "speak":
            target = self.rng.choice([p for p in observation.alive if p != player])
            message = "" if self.rng.random() < 0.5 else f"I suspect {target}."
            overpower_targets = legal["overpower_targets"]
            if overpower_targets and self.rng.random() < 0.25:
                return MafiaAction(
                    player=player,
                    type="overpower",
                    target=self.rng.choice(overpower_targets),
                    message=message,
                )
            return MafiaAction(player=player, type="speak", message=message)
        return MafiaAction(
            player=player,
            type=legal["type"],
            target=self.rng.choice(legal["targets"]),
        )


@dataclass(frozen=True)
class Decision:
    team: str
    observation: MafiaObservation
    action: MafiaAction


@dataclass(frozen=True)
class Episode:
    seed: int
    winner: str | None
    rewards: dict[str, float]
    decisions: tuple[Decision, ...]
    public_events: tuple[dict, ...]

    @property
    def days(self) -> int:
        return sum(event["type"] == "vote_result" for event in self.public_events)

    @property
    def overpowers(self) -> int:
        return sum(event["type"] == "overpower" for event in self.public_events)


def run_episode(
    seed: int,
    town_factory: PolicyFactory = RandomPolicy,
    mafia_factory: PolicyFactory = RandomPolicy,
) -> Episode:
    """Run one game using only OpenEnv observations and actions."""

    environment = MafiaEnvironment()
    policies = {
        "town": town_factory(seed * 2 + 1),
        "mafia": mafia_factory(seed * 2 + 2),
    }
    observation = environment.reset(seed=seed)
    decisions: list[Decision] = []
    while not observation.done:
        team = "mafia" if observation.role == "mafia" else "town"
        action = policies[team].act(observation)
        decisions.append(Decision(team, observation, action))
        observation = environment.step(action)
    return Episode(
        seed=seed,
        winner=observation.winner,
        rewards=observation.rewards,
        decisions=tuple(decisions),
        public_events=tuple(observation.public_events),
    )


def evaluate(
    seeds: list[int],
    town_factory: PolicyFactory = RandomPolicy,
    mafia_factory: PolicyFactory = RandomPolicy,
) -> dict[str, float | int]:
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    episodes = [run_episode(seed, town_factory, mafia_factory) for seed in seeds]
    return {
        "games": len(episodes),
        "town_wins": sum(episode.winner == "town" for episode in episodes),
        "mafia_wins": sum(episode.winner == "mafia" for episode in episodes),
        "draws": sum(episode.winner is None for episode in episodes),
        "town_win_rate": sum(episode.winner == "town" for episode in episodes) / len(episodes),
        "mafia_win_rate": sum(episode.winner == "mafia" for episode in episodes) / len(episodes),
        "mean_days": sum(episode.days for episode in episodes) / len(episodes),
        "mean_overpowers": sum(episode.overpowers for episode in episodes) / len(episodes),
    }


def evaluate_paired(
    *,
    team: str,
    candidate_factory: PolicyFactory,
    baseline_factory: PolicyFactory,
    opponent_factory: PolicyFactory,
    seeds: list[int],
) -> dict[str, float | int | list[float]]:
    """Compare policies on the same held-out seeds and fixed opponent."""

    if team not in {"town", "mafia"}:
        raise ValueError("team must be town or mafia")
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    differences = []
    candidate_wins = 0
    baseline_wins = 0
    for seed in seeds:
        if team == "town":
            candidate = run_episode(seed, candidate_factory, opponent_factory)
            baseline = run_episode(seed, baseline_factory, opponent_factory)
        else:
            candidate = run_episode(seed, opponent_factory, candidate_factory)
            baseline = run_episode(seed, opponent_factory, baseline_factory)
        candidate_win = int(candidate.winner == team)
        baseline_win = int(baseline.winner == team)
        candidate_wins += candidate_win
        baseline_wins += baseline_win
        differences.append(candidate_win - baseline_win)
    mean_difference = sum(differences) / len(differences)
    variance = (
        sum((value - mean_difference) ** 2 for value in differences) / (len(differences) - 1)
        if len(differences) > 1 else 0.0
    )
    half_width = 1.96 * sqrt(variance / len(differences))
    return {
        "games": len(seeds),
        "candidate_win_rate": candidate_wins / len(seeds),
        "baseline_win_rate": baseline_wins / len(seeds),
        "paired_difference": mean_difference,
        "approx_95pct_interval": [mean_difference - half_width, mean_difference + half_width],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the random Mafia baseline")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate(list(range(args.seed, args.seed + args.games))), indent=2))


if __name__ == "__main__":
    main()
