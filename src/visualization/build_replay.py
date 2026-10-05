"""Compile saved Qwen JSONL rollouts into compact data for the local replay UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRACE_PATTERN = re.compile(r"qwen3-4b-instruct-2507-seed(\d+)\.jsonl$")


def _view(record: dict[str, Any]) -> dict[str, Any]:
    messages = record["prompt"]
    user_message = next(message["content"] for message in messages if message["role"] == "user")
    return json.loads(user_message)


def read_game(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    decisions = [record for record in records if record.get("kind") == "decision"]
    summary = next(record for record in records if record.get("kind") == "summary")
    if not decisions:
        raise ValueError(f"No completed decisions in {path}")

    views = [_view(record) for record in decisions]
    roles = {record["player"]: record["role"] for record in decisions}
    for event in summary["public_events"]:
        eliminated = event.get("eliminated")
        revealed_role = event.get("revealed_role")
        if eliminated and revealed_role:
            roles[eliminated] = revealed_role

    frames = []
    for index, (record, view) in enumerate(zip(decisions, views)):
        after_events = (
            views[index + 1]["public_events"]
            if index + 1 < len(views)
            else summary["public_events"]
        )
        before_events = view["public_events"]
        if after_events[:len(before_events)] != before_events:
            raise ValueError(f"Public event history changed unexpectedly in {path} step {index + 1}")
        action = record["action"]
        frames.append({
            "step": record["attempt"],
            "day": record["day"],
            "phase": record["phase"],
            "round": view.get("discussion_round"),
            "actor": record["player"],
            "actor_role": record["role"],
            "alive": view["alive"],
            "action": action["type"],
            "target": action.get("target"),
            "message": action.get("message") or "",
            "accepted": record["accepted"],
            "forfeited": record["turn_forfeited"],
            "referee_message": record.get("referee_message"),
            "private": view.get("your_private_events", []),
            "public_delta": after_events[len(before_events):],
        })

    return {
        "seed": summary["seed"],
        "model": summary["model"],
        "winner": summary["winner"],
        "attempts": summary["attempts"],
        "players": views[0]["alive"],
        "roles": roles,
        "frames": frames,
    }


def build_data(
    rollout_dir: Path = ROOT / "rollouts",
    scripted_traces: tuple[Path, ...] = (),
    learned_traces: tuple[Path, ...] = (),
) -> dict[str, Any]:
    paths = sorted(
        (path for path in rollout_dir.iterdir() if TRACE_PATTERN.fullmatch(path.name)),
        key=lambda path: int(TRACE_PATTERN.fullmatch(path.name).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"No complete Qwen rollouts in {rollout_dir}")
    games = [read_game(path) | {"label": "Frozen Qwen"} for path in paths]
    for trace_path in scripted_traces:
        with trace_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                game = json.loads(line)
                game["label"] = f"Scripted · {game['arm'].replace('_', ' ')}"
                games.append(game)
    for trace_path in learned_traces:
        report = json.loads(trace_path.with_name("report.json").read_text(encoding="utf-8"))
        choices_by_game = {
            (stage, row["seed"], row["arm"]): row["choices"]
            for stage, evaluation in (("before", report["initial_eval"]), ("after", report["final_eval"]))
            for row in evaluation["per_game"]
        }
        with trace_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                game = json.loads(line)
                key = (game["stage"], game["seed"], game["arm"])
                choices = iter(choices_by_game[key])
                for frame in game["frames"]:
                    if frame.get("reason") in ("qwen_reveal", "qwen_hold"):
                        choice = next(choices)
                        frame["p_reveal"] = choice["p_reveal"]
                if next(choices, None) is not None:
                    raise ValueError(f"Unmatched reveal choices in {trace_path}: {key}")
                game["label"] = f"Qwen reveal · {game['stage']} · {game['arm']}"
                games.append(game)
    return {"source": "saved_mafia_games", "games": games}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Mafia replay data from JSONL rollouts")
    parser.add_argument("--rollouts", type=Path, default=ROOT / "rollouts")
    parser.add_argument("--scripted-traces", type=Path, action="append", default=[],
                        help="Add complete scripted games from a games.jsonl file; may repeat")
    parser.add_argument("--learned-traces", type=Path, action="append", default=[],
                        help="Add before/after Qwen reveal evaluation games; matching report.json is required")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("data.js"))
    args = parser.parse_args()
    payload = json.dumps(
        build_data(args.rollouts, tuple(args.scripted_traces), tuple(args.learned_traces)),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"window.MAFIA_REPLAY_DATA={payload};\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(payload):,} bytes)")


if __name__ == "__main__":
    main()
