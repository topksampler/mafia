"""Inspect how the trained vote adapter differs from frozen Qwen.

Usage: uv run modal run src/training/audit_job.py --run-name town-vote-20261004-094946
"""

import json
from pathlib import Path

import modal

from src.training.modal_job import _hf_secret, image, volume


app = modal.App("mafia-town-vote-audit")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=32768,
    timeout=3600,
    volumes={"/data": volume},
    secrets=[_hf_secret()],
)
def audit_remote(run_name: str, start_seed: int, games: int) -> dict:
    from src.training.audit import audit_adapter

    return audit_adapter(run_name, start_seed, games)


@app.local_entrypoint()
def main(run_name: str, start_seed: int = 50_000, games: int = 24) -> None:
    report = audit_remote.remote(run_name, start_seed, games)
    output = Path("runs") / run_name / "vote_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Vote audit saved to {output}")
    print(json.dumps({
        "base_town_wins": report["base"]["town_wins"],
        "adapted_town_wins": report["adapted"]["town_wins"],
        "same_view_votes": report["same_view"]["compared_votes"],
        "changed_same_view_votes": report["same_view"]["changed_votes"],
        "changed_outcomes": report["changed_outcomes"],
        "label_probe": report["label_probe"],
    }, indent=2))
