"""Run the learned Detective reveal/hold pilot on one Modal A100.

Usage:
  uv run modal run src/training/reveal_modal_job.py \
    --updates 2 --eligible-games-per-update 4 --eval-games 32
"""

import json

import modal

from src.training.modal_job import (
    DATA_DIR,
    _hf_secret,
    image,
    metrics_store,
    volume,
)


app = modal.App("mafia-detective-reveal-rl")


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
    eligible_games_per_update: int,
    max_sampled_games_per_update: int,
    eval_games: int,
    eval_interval: int,
    train_start_seed: int,
    eval_start_seed: int,
    dashboard_url: str,
) -> dict:
    from src.training.reveal_train import train

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
        eligible_games_per_update=eligible_games_per_update,
        max_sampled_games_per_update=max_sampled_games_per_update,
        eval_games=eval_games,
        eval_interval=eval_interval,
        train_start_seed=train_start_seed,
        eval_start_seed=eval_start_seed,
        publish=publish,
        flush_metrics=volume.commit,
    )
    volume.commit()
    return {
        "run_name": report["run_name"],
        "dashboard": report["dashboard"],
        "adapter_path": report["adapter_path"],
        "report_path": f"{DATA_DIR}/runs/{report['run_name']}/report.json",
        "eval_traces_path": report["eval_traces_path"],
        "total_sampled_games": report["total_sampled_games"],
        "total_eligible_training_games": report["total_eligible_training_games"],
        "initial_town_wins": report["initial_eval"]["town_wins"],
        "final_town_wins": report["final_eval"]["town_wins"],
        "eval_games": report["final_eval"]["games"],
    }


@app.local_entrypoint()
def main(
    updates: int = 2,
    eligible_games_per_update: int = 4,
    max_sampled_games_per_update: int = 60,
    eval_games: int = 32,
    eval_interval: int = 1,
    train_start_seed: int = 80_000,
    eval_start_seed: int = 90_000,
) -> None:
    dashboard_url = modal.Function.from_name(
        "mafia-trackio-dashboard", "dashboard"
    ).get_web_url()
    if not dashboard_url:
        raise RuntimeError("Deploy the dashboard first")
    print(f"Live dashboard: {dashboard_url}", flush=True)
    result = train_remote.remote(
        updates,
        eligible_games_per_update,
        max_sampled_games_per_update,
        eval_games,
        eval_interval,
        train_start_seed,
        eval_start_seed,
        dashboard_url,
    )
    print(json.dumps(result, indent=2), flush=True)
