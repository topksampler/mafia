"""Scripted timing controls for the future learned reveal/hold action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from env.rules import GameConfig
from src.training.scripted_reveal import play_scripted_game


TIMINGS = {"first": 1, "last": 3}


def run_timing_probe(start_seed: int, games: int, players: int = 6) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if games < 1:
        raise ValueError("games must be positive")
    config = GameConfig(players=players, discussion_rounds=3)
    traces: list[dict[str, Any]] = []
    per_seed = {}
    for seed in range(start_seed, start_seed + games):
        variants = {}
        for timing, round_number in TIMINGS.items():
            game = play_scripted_game(
                seed, "counterclaim", config, reveal_round=round_number
            )
            game["timing"] = timing
            traces.append(game)
            variants[timing] = game["metrics"]
        per_seed[str(seed)] = variants

    changes = {"first_win_last_loss": [], "first_loss_last_win": [], "same_outcome": []}
    for seed_text, variants in per_seed.items():
        first, last = variants["first"], variants["last"]
        key = (
            "first_win_last_loss" if first["town_win"] and not last["town_win"]
            else "first_loss_last_win" if not first["town_win"] and last["town_win"]
            else "same_outcome"
        )
        changes[key].append(int(seed_text))

    summary = {}
    for timing in TIMINGS:
        rows = [variants[timing] for variants in per_seed.values()]
        eligible_counts = [
            sum(
                frame["reason"] in ("detective_truthful_reveal", "detective_timing_hold")
                for frame in game["frames"]
            )
            for game in traces if game["timing"] == timing
        ]
        summary[timing] = {
            "town_wins": sum(row["town_win"] for row in rows),
            "games_with_eligible_choice": sum(count > 0 for count in eligible_counts),
            "eligible_choices": sum(eligible_counts),
            "truthful_reveals": sum(row["truthful_reveals"] for row in rows),
            "mafia_counterclaims": sum(row["mafia_counterclaims"] for row in rows),
            "villager_claim_follow_votes": sum(row["villager_claim_follow_votes"] for row in rows),
            "villager_claim_follow_correct": sum(row["villager_claim_follow_correct"] for row in rows),
        }
    report = {
        "experiment": "scripted_reveal_timing_probe",
        "model_calls": 0,
        "training_updates": 0,
        "opponent": "reactive_mafia_counterclaim",
        "policy_choice_frequency": "every eligible Detective speech turn",
        "first_control": "Reveal on first available eligible Detective slot; hold on remaining eligible slots.",
        "last_control": "Hold on earlier eligible Detective slots; reveal on discussion round 3 if that slot remains available.",
        "seeds": [start_seed, start_seed + games - 1],
        "games_per_timing": games,
        "players": players,
        "summary": summary,
        "paired_outcomes": changes,
        "per_seed": per_seed,
    }
    return report, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scripted first/last Detective reveal timing")
    parser.add_argument("--start-seed", type=int, default=70_000)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("runs") / (
        f"reveal-timing-seeds-{args.start_seed}-{args.start_seed + args.games - 1}"
    )
    report, traces = run_timing_probe(args.start_seed, args.games, args.players)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output / "games.jsonl").open("w", encoding="utf-8") as stream:
        for game in traces:
            stream.write(json.dumps(game, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "report": str(output / "report.json"),
        "traces": str(output / "games.jsonl"),
        "summary": report["summary"],
        "paired_outcome_counts": {
            name: len(seeds) for name, seeds in report["paired_outcomes"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
