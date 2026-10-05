"""OpenEnv FastAPI/WebSocket app. Each connection gets its own game."""

from openenv.core.env_server import create_app

from .environment import MafiaEnvironment
from .models import MafiaAction, MafiaObservation

app = create_app(
    MafiaEnvironment,
    MafiaAction,
    MafiaObservation,
    env_name="mafia",
    max_concurrent_envs=8,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
