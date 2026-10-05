"""One stateless Qwen policy shared by every Mafia player.

Each call receives only the active player's OpenEnv observation. The model
never sees the referee's role map or another player's private events.
"""

from dataclasses import dataclass
import json
from typing import Any

from env.models import MafiaAction, MafiaObservation


MODEL = "huggingface/nscale/Qwen/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = """You are one player in a six-player game of Mafia.
There is one Mafia, one Detective, and four Villagers. The Detective and
Villagers are one Town team. Town wins by eliminating Mafia. Mafia wins when
living Mafia count reaches living Town count. The Godfather is the referee,
not a player. Your objective is to help your team win.

At night, Mafia chooses a kill and Detective chooses a private investigation.
During the day, every survivor has three discussion opportunities and then
votes. A discussion action can be speak or overpower. Overpower spends your
current speaking opportunity and uses another player's next open slot for
your message, which may be empty. Speech and role claims may be deceptive.
Only the listed legal actions and targets are available. An illegal action
gets a public warning; another illegal action forfeits that turn.

Return exactly one JSON object, with no Markdown or explanation:
- Night: {"type":"kill","target":"P1"} or
  {"type":"investigate","target":"P1"}
- Discussion: {"type":"speak","message":"..."} or
  {"type":"overpower","target":"P1","message":"..."}
- Vote: {"type":"vote","target":"P1"}
Replace example targets with a legal target. Keep messages within the
max_chars limit. Use the public conversation to judge where votes are moving."""


@dataclass(frozen=True)
class PolicyDecision:
    action: MafiaAction
    messages: list[dict[str, str]]
    raw_response: str
    prompt_tokens: int | None
    completion_tokens: int | None


def build_messages(observation: MafiaObservation) -> list[dict[str, str]]:
    """Serialize only information available to the active player."""

    if observation.current_player is None or observation.legal_actions is None:
        raise ValueError("The policy needs an active player's observation")
    view: dict[str, Any] = {
        "you": observation.current_player,
        "your_role": observation.role,
        "day": observation.day,
        "phase": observation.phase,
        "discussion_round": observation.discussion_round,
        "alive": observation.alive,
        "public_events": observation.public_events,
        "your_private_events": observation.private_events,
        "legal_actions": observation.legal_actions,
    }
    if observation.referee_message and not observation.turn_forfeited:
        view["referee_warning"] = observation.referee_message
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(view, ensure_ascii=False)},
    ]


def parse_action(raw_response: str, player: str) -> MafiaAction:
    """Pass malformed replies to the referee as illegal actions.

    We do not silently replace an illegal model choice with a legal one; the
    warning/forfeit rule is part of the environment's learning signal.
    """

    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        return MafiaAction(player=player, type="invalid", message=raw_response)
    target = payload.get("target")
    message = payload.get("message")
    return MafiaAction(
        player=player,
        type=payload["type"],
        target=target if isinstance(target, str) else None,
        message=message if isinstance(message, str) else None,
    )


class QwenPolicy:
    """A frozen, hosted Qwen policy; no weights are updated here."""

    def __init__(self, model: str = MODEL, api_key: str | None = None) -> None:
        if api_key is None:
            from huggingface_hub import get_token

            api_key = get_token()
        if not api_key:
            raise RuntimeError("Set HF_TOKEN or log in with `hf auth login` first")
        self.model = model
        self.api_key = api_key

    def decide(self, observation: MafiaObservation) -> PolicyDecision:
        from litellm import completion

        player = observation.current_player
        if player is None:
            raise ValueError("Game is finished")
        messages = build_messages(observation)
        options: dict[str, Any] = {}
        if "Qwen3.5" in self.model:
            options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = completion(
            model=self.model,
            api_key=self.api_key,
            messages=messages,
            max_tokens=256,
            temperature=0.7,
            top_p=0.8,
            **options,
        )
        raw = response.choices[0].message.content or ""
        usage = response.usage
        return PolicyDecision(
            action=parse_action(raw, player),
            messages=messages,
            raw_response=raw,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
