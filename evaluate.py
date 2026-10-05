"""Measure the frozen Qwen policy over a small set of seeded games."""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from policy import MODEL, SYSTEM_PROMPT
from rollout import run


def load_or_run(seed: int, directory: Path, model: str) -> list[dict[str, Any]]:
    name = model.rsplit("/", 1)[-1].lower()
    path = directory / f"{name}-seed{seed}.jsonl"
    if path.exists():
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if (
                records
                and records[-1].get("kind") == "summary"
                and records[-1].get("seed") == seed
                and records[-1].get("model") == model
                and records[0]["prompt"][0]["content"] == SYSTEM_PROMPT
            ):
                print(f"seed={seed}: reused complete trace ({records[-1]['winner']} won)", flush=True)
                return records
        except (KeyError, ValueError, TypeError):
            pass
    summary = run(seed, path, model=model, verbose=False)
    print(f"seed={seed}: played {summary['attempts']} attempts ({summary['winner']} won)", flush=True)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def evaluate(start_seed: int, games: int, directory: Path, model: str = MODEL) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be positive")
    traces = [load_or_run(seed, directory, model) for seed in range(start_seed, start_seed + games)]
    summaries = [trace[-1] for trace in traces]
    decisions = [record for trace in traces for record in trace[:-1]]
    actions = Counter(record["action"]["type"] for record in decisions)
    winners = Counter(summary["winner"] for summary in summaries)
    result = {
        "model": model,
        "seeds": list(range(start_seed, start_seed + games)),
        "games": games,
        "town_wins": winners["town"],
        "mafia_wins": winners["mafia"],
        "draws": winners[None],
        "town_win_rate": winners["town"] / games,
        "mafia_win_rate": winners["mafia"] / games,
        "action_attempts": len(decisions),
        "accepted_actions": sum(record["accepted"] is True for record in decisions),
        "warnings": sum(record["accepted"] is False and not record["turn_forfeited"] for record in decisions),
        "forfeits": sum(record["turn_forfeited"] for record in decisions),
        "action_types": dict(actions),
        "prompt_tokens": sum(summary["prompt_tokens"] for summary in summaries),
        "completion_tokens": sum(summary["completion_tokens"] for summary in summaries),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.rsplit('/', 1)[-1].lower()}-seeds{start_seed}-{start_seed + games - 1}-eval.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Qwen over seeded Mafia games")
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--directory", type=Path, default=Path("rollouts"))
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    evaluate(args.start_seed, args.games, args.directory, args.model)


if __name__ == "__main__":
    main()
