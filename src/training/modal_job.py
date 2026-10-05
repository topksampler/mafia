"""Run the Town-vote Qwen RL experiment on one Modal A100 GPU.

Usage: uv run modal run src/training/modal_job.py --updates 6 --games-per-update 8
The Hugging Face token stays in a Modal Secret, never in the mounted code.
"""

import json
from pathlib import Path

import modal


SOURCE = Path(__file__).resolve()
# Modal stages entry modules at /root/<filename>.py in the container.
ROOT = SOURCE.parents[2] if SOURCE.parent.name == "training" else Path("/root/mafia")
DATA_DIR = "/data/mafia_vote"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "transformers>=4.51,<5",
        "peft>=0.17,<1",
        "tensorboard",
    )
    .env({"PYTHONPATH": "/root/mafia", "HF_HOME": "/data/huggingface"})
    .add_local_dir(ROOT / "env", "/root/mafia/env")
    .add_local_dir(ROOT / "src", "/root/mafia/src")
)
app = modal.App("mafia-town-vote-rl")
volume = modal.Volume.from_name("mafia-rl-checkpoints", create_if_missing=True)
metrics_store = modal.Dict.from_name("mafia-rl-metrics", create_if_missing=True)


def _hf_secret() -> modal.Secret:
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face login is required: hf auth login")
    return modal.Secret.from_dict({"HF_TOKEN": token})


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=32768,
    timeout=7200,
    volumes={"/data": volume},
    secrets=[_hf_secret()],
)
def train_remote(
    updates: int,
    games_per_update: int,
    eval_games: int,
    dashboard_url: str,
    policy_kind: str,
    eval_start_seed: int,
    train_start_seed: int,
) -> dict:
    from src.training.vote import train

    def publish(run_name: str, event: dict) -> None:
        record = metrics_store.get(run_name)
        if isinstance(record, str):
            record = json.loads(record)
        if record is None:
            record = {"name": run_name, "config": event.get("config", {}), "events": []}
            names = metrics_store.get("__runs__", [])
            metrics_store["__runs__"] = [*names, run_name]
        record["events"].append(event)
        metrics_store[run_name] = json.dumps(record, default=str)

    report = train(
        output_dir=DATA_DIR,
        dashboard_url=dashboard_url,
        updates=updates,
        games_per_update=games_per_update,
        eval_games=eval_games,
        policy_kind=policy_kind,
        eval_start_seed=eval_start_seed,
        train_start_seed=train_start_seed,
        flush_metrics=volume.commit,
        publish=publish,
    )
    volume.commit()
    return {
        "run_name": report["run_name"],
        "dashboard": report["dashboard"],
        "adapter_path": report["adapter_path"],
        "initial_town_win_rate": report["initial_eval"]["town_win_rate"],
        "final_town_win_rate": report["history"][-1]["eval/town_win_rate"],
        "max_weight_change": report["history"][-1]["train/weight_change_max"],
    }


@app.local_entrypoint()
def main(
    updates: int = 6,
    games_per_update: int = 8,
    eval_games: int = 24,
    policy_kind: str = "digit",
    eval_start_seed: int = 50_000,
    train_start_seed: int = 1_000,
) -> None:
    dashboard_url = modal.Function.from_name("mafia-trackio-dashboard", "dashboard").get_web_url()
    if not dashboard_url:
        raise RuntimeError("Deploy the dashboard first: uv run modal deploy src/training/dashboard/modal_app.py")
    print(f"Live dashboard: {dashboard_url}", flush=True)
    result = train_remote.remote(
        updates, games_per_update, eval_games, dashboard_url,
        policy_kind, eval_start_seed, train_start_seed,
    )
    print(json.dumps(result, indent=2), flush=True)
