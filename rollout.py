"""Record one inspectable frozen-Qwen Mafia game as JSON Lines."""

import argparse
import json
from pathlib import Path

from env.environment import MafiaEnvironment
from env.rules import RULESET_VERSION
from policy import MODEL, QwenPolicy


def run(
    seed: int,
    output: Path,
    max_attempts: int = 200,
    model: str = MODEL,
    verbose: bool = True,
) -> dict[str, object]:
    environment = MafiaEnvironment()
    policy = QwenPolicy(model=model)
    observation = environment.reset(seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    prompt_tokens = 0
    completion_tokens = 0

    with output.open("w", encoding="utf-8") as trace:
        while not observation.done:
            if attempts >= max_attempts:
                raise RuntimeError(f"Stopped after {max_attempts} attempts; partial trace: {output}")
            actor_view = observation
            decision = policy.decide(actor_view)
            observation = environment.step(decision.action)
            attempts += 1
            prompt_tokens += decision.prompt_tokens or 0
            completion_tokens += decision.completion_tokens or 0
            record = {
                "kind": "decision",
                "attempt": attempts,
                "seed": seed,
                "player": actor_view.current_player,
                "role": actor_view.role,
                "day": actor_view.day,
                "phase": actor_view.phase,
                "prompt": decision.messages,
                "raw_response": decision.raw_response,
                "action": decision.action.model_dump(exclude_none=True),
                "accepted": observation.action_accepted,
                "turn_forfeited": observation.turn_forfeited,
                "referee_message": observation.referee_message,
                "prompt_tokens": decision.prompt_tokens,
                "completion_tokens": decision.completion_tokens,
            }
            trace.write(json.dumps(record, ensure_ascii=False) + "\n")
            trace.flush()
            if verbose:
                print(
                    f"{attempts:02d} day={actor_view.day} {actor_view.phase:5s} "
                    f"{actor_view.current_player} {decision.action.type:11s} "
                    f"accepted={observation.action_accepted} "
                    f"forfeited={observation.turn_forfeited}",
                    flush=True,
                )

        summary: dict[str, object] = {
            "kind": "summary",
            "seed": seed,
            "model": model,
            "ruleset_version": RULESET_VERSION,
            "attempts": attempts,
            "winner": observation.winner,
            "rewards": observation.rewards,
            "public_events": observation.public_events,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        trace.write(json.dumps(summary, ensure_ascii=False) + "\n")
        if verbose:
            print(json.dumps({k: v for k, v in summary.items() if k != "public_events"}, indent=2))
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen-Qwen Mafia rollout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    model_name = args.model.rsplit("/", 1)[-1].lower()
    output = args.output or Path(f"rollouts/{model_name}-seed{args.seed}.jsonl")
    run(args.seed, output, args.max_attempts, args.model)


if __name__ == "__main__":
    main()
