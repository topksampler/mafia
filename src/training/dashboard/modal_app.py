"""Read-only live charts for Mafia RL metrics stored in a Modal Dict."""

import json
from pathlib import Path

import modal


app = modal.App("mafia-trackio-dashboard")
ROOT = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("fastapi")
    .add_local_file(ROOT / "index.html", "/root/dashboard.html")
)
metrics_store = modal.Dict.from_name("mafia-rl-metrics", create_if_missing=True)


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    max_containers=1,
    scaledown_window=600,
)
@modal.asgi_app()
def dashboard():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    web = FastAPI()

    @web.get("/", response_class=HTMLResponse)
    def index():
        return Path("/root/dashboard.html").read_text(encoding="utf-8")

    @web.get("/api/runs")
    def runs():
        names = metrics_store.get("__runs__", [])
        data = {}
        for name in names:
            try:
                value = metrics_store.get(name, {})
                data[name] = json.loads(value) if isinstance(value, str) else value
            except Exception:
                # Earlier experiments may have stored objects requiring packages
                # unavailable in this lightweight dashboard image.
                continue
        return {"runs": list(data), "data": data}

    return web
