"""Run the frozen candidate-wise vote experiment on Modal.

Usage: uv run modal run src/training/candidate_eval_job.py --games 24
"""

import json
from pathlib import Path

import modal

from src.training.modal_job import _hf_secret, image, volume


app = modal.App("mafia-candidate-vote-eval")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=32768,
    timeout=3600,
    volumes={"/data": volume},
    secrets=[_hf_secret()],
)
def evaluate_remote(start_seed: int, games: int, centered_labels: bool) -> dict:
    from src.training.candidate_vote import evaluate_candidate_policy

    return evaluate_candidate_policy(start_seed, games, centered_labels)


@app.local_entrypoint()
def main(start_seed: int = 50_000, games: int = 24, centered_labels: bool = False) -> None:
    report = evaluate_remote.remote(start_seed, games, centered_labels)
    prefix = "candidate-centered" if centered_labels else "candidate-vote"
    output = Path("runs") / f"{prefix}-seeds-{start_seed}-{start_seed + games - 1}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate vote evaluation saved to {output}")
    print(json.dumps({
        "town_wins": report["town_wins"],
        "vote_summary": report["vote_summary"],
        "label_probe": report["label_probe"],
    }, indent=2))
