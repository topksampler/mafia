# Mafia environment

This project uses [Hugging Face OpenEnv](https://huggingface.co/docs/openenv/main/guides/concepts)
for the environment interface. The [v0 game rules](docs/game_rules.md)
defines the default game: six players, one Mafia, one Detective, four Villagers,
no Doctor, and three discussion turns per survivor each day. `GameConfig.players`
can change the table size while keeping one Mafia and one Detective.

## Environment structure

- [env/rules.py](env/rules.py) defines the Mafia, Detective, and Villager roles.
  It is also the Godfather: it owns roles, turn order,
  public events, private investigations, voting, and victory conditions.
- [env/models.py](env/models.py) defines the typed OpenEnv
  `MafiaAction` and `MafiaObservation` contract.
- [env/environment.py](env/environment.py)
  implements OpenEnv `reset`, `step`, and `state` around the rules engine.
- [env/server.py](env/server.py) exposes that
  environment through OpenEnv's FastAPI and WebSocket server.
- [env/client.py](env/client.py) is the typed OpenEnv client.

One OpenEnv session represents one entire game. A trusted game controller owns
that session and sends the returned observation only to `current_player`.
During play, each `step` returns the **next** actor's view. The `role` and
`private_events` fields are private to that actor. OpenEnv's `state` endpoint
contains only an episode ID and step count, so it does not reveal night roles.
At game end, the observation is public-only.

OpenEnv has one scalar `reward` per step. We set it to the player who just
acted's reward; the observation also carries a `rewards` map for all players.
These are zero until the game ends, then `+1` for winners, `-1` for losers,
and zero for a draw. The controller can use the final map to credit each
player's full trajectory later.

## Local development

Requires Python 3.11+ and `uv`.

```bash
uv sync
uv run mafia-env-server
```

The server listens on `http://127.0.0.1:8000`. The OpenEnv client uses its
persistent WebSocket connection for a game session. The manifest is
[openenv.yaml](openenv.yaml).

```bash
uv run python -m pytest -q
uv run openenv validate --level static --skip-build
```

## Frozen Qwen rollout

The first language policy is [policy.py](policy.py). It uses LiteLLM to call
`Qwen/Qwen3-4B-Instruct-2507` through Hugging Face's Nscale Inference Provider.
The policy receives only the active player's observation and returns one
structured action. No model weights are updated during this rollout.

Log in to Hugging Face once with `hf auth login` (or set `HF_TOKEN`), then run:

```bash
uv run python rollout.py --seed 1
```

[rollout.py](rollout.py) records every prompt, raw model reply, parsed action,
and referee response in `rollouts/qwen3-4b-instruct-2507-seed1.jsonl`. The
trace contains each acting player's private view for analysis; never give the
whole trace to a player policy. The file is ignored by Git.
The [first rollout summary](rollouts/qwen3-4b-instruct-2507-seed1-summary.md)
shows the main game events without reproducing every prompt.

To summarize a range of frozen-model games, use [evaluate.py](evaluate.py):

```bash
uv run python evaluate.py --start-seed 1 --games 10
```

Complete traces with the same model and policy prompt are reused. Missing
seeds make new Hugging Face inference calls. The resulting win rate is a
starting observation, not a measure of training improvement.
The [ten-game report](rollouts/qwen3-4b-instruct-2507-seeds1-10-report.md)
records the observed action errors and limits of this small sample.

## Replay the saved games

The local [game replay](src/visualization/index.html) places players around a
table from each trace's roster, so the number of seats follows the game rather
than a fixed six. Step or play through night actions, discussion, and voting;
the scene changes from night to day. The inspector shows public speech,
private clues, role claims, mention counts, and the vote outcome. It is an
evaluator view: private facts and true roles must never be passed to a player
policy.

```bash
python -m src.visualization.build_replay \
  --scripted-traces runs/scripted-reveal-seeds-70000-70099/games.jsonl \
  --learned-traces runs/detective-reveal-20261004-120608/eval_games.jsonl
python -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/src/visualization/index.html`. The generated
`src/visualization/data.js` is a local, static copy of the ten saved frozen
Qwen games, 500 scripted control games, and 128 before/after reveal-policy
evaluation games. Choose a dataset, enter a seed, and switch policies to compare
the five scripted arms or Qwen before and after training. The arrows
move to the previous or next seed; Step and Play continue across games. Seed
70014 shows a truthful reveal
followed by a Mafia counterclaim. Rebuild the data after recording more games.
The viewer makes no model calls and does not affect training.

## First RL training run

Training code lives in `src/training/`:

- [vote.py](src/training/vote.py) builds the Town vote policy and RL update.
- [modal_job.py](src/training/modal_job.py) launches the GPU job.
- [dashboard/](src/training/dashboard/) contains the Modal dashboard app and its HTML.

The first trainable policy uses open weights from
`Qwen/Qwen3-4B-Instruct-2507` on Modal. Qwen chooses **Town vote
targets** from the active player's observation; seeded random policies handle
all other decisions. A legal-target mask keeps this first experiment focused
on whom to vote for. Each full game's Town win/loss gives a `+1`/`-1` reward.
Episodic REINFORCE updates a LoRA adapter; the 4B base weights stay frozen.

The trainer calls the same rules engine used by OpenEnv in process. OpenEnv
0.7 and the Qwen training libraries currently require incompatible Hugging
Face Hub versions in one Python image, so the GPU image leaves out the OpenEnv
transport package. The game rules and private observations are shared.

With Modal authentication and a Hugging Face token for downloading Qwen,
start the live dashboard and a run:

```bash
uv run modal deploy src/training/dashboard/modal_app.py
uv run modal run src/training/modal_job.py --updates 6 --games-per-update 8 --eval-games 24
```

The [live charts](https://lalith1403--mafia-trackio-dashboard-dashboard.modal.run)
show per-game reward, a rolling reward average, held-out Town win rate, policy
loss, and gradient norm. The page is public-read and contains only aggregate
numeric metrics and run settings. Reports, TensorBoard logs, and adapters are
saved in the `mafia-rl-checkpoints` Modal Volume. The completed
[48-game report](runs/town-vote-20261004-094946/report.json) is also in this
workspace; its downloaded adapter is in the same run folder and ignored by
Git because the weight file is large.

This first run made six updates and changed the adapter weights. On 24 fixed
held-out games, Town won 16 before training and 15 after training. The random
Town baseline won 12 on those seeds. We have a working RL loop, but no measured
improvement yet. The audit below identifies a player-seat bias in the vote
action representation; that is the next policy design choice before training
speech and overpowering.

The [vote policy audit](docs/vote_policy_audit.md) records individual votes
from the frozen model and saved adapter. Reproduce it with:

```bash
uv run modal run src/training/audit_job.py --run-name town-vote-20261004-094946 --games 24
```

A [candidate-wise frozen policy probe](src/training/candidate_vote.py) removed
the direct digit comparison but did not improve results; its findings are in
the same audit document. Run it with
`uv run modal run src/training/candidate_eval_job.py --games 24`.
Adding `--centered-labels` makes the candidate prompt invariant to player-name
swaps, but the frozen policy still did not improve the 24 development games.
We then trained that centered candidate vote policy for three updates on 12
games. On eight fresh held-out games, Town wins changed from 4 before training
to 3 after; random Town votes won 5. The [run report](runs/town-centered-vote-20261004-110540/report.json)
contains the settings and measurements. The adapter is saved in the Modal
Volume. This run confirms that the gradient and LoRA update work, while the
held-out result gives no evidence of better play. The next experiment should
test how discussion communicates a Detective's private result before we train
another action family.

The [scripted Detective reveal experiment](docs/scripted_detective_reveal.md)
now supplies that control. On 100 paired seeds, Town wins were 50 when the
Detective used its private evidence only to vote, 59 when it publicly shared a
true positive result and Villagers followed the claim, and 42 when Mafia could
counterclaim after that reveal. These are deterministic policy rules, not Qwen
speech training; they show why credibility and response to claims belong in the
next policy design.

## Detective reveal/hold RL pilot

These runs use the v0.4 scheduled-discussion rules. The
[open-floor policy design](docs/open_floor_policy.md) changes when agents may
speak and makes disclosure wording part of the learnable action; v0.4 win
rates do not measure that game.

[reveal_train.py](src/training/reveal_train.py) asks Qwen on every Detective
speaking turn where it has privately confirmed a living Mafia player. Qwen
chooses Reveal or Hold; either spends the speaking turn. The trainer uses two
counterbalanced Yes/No prompts to score the choices and updates a LoRA adapter
from complete-game Town wins or losses. Other actions use declared scripted
policies. Training samples natural games until four contain a reveal decision
per update; held-out evaluation keeps every seed, including games with no
eligible turn.

The [pilot report](runs/detective-reveal-20261004-120608/report.json) covers
two updates, 29 sampled training games, eight games with eligible decisions,
and 64 held-out games across counterclaim and bluff opponents. Town won 31/64
before and after training. A matched always-Reveal control also won 31/64;
always-Hold won 23/64. Qwen chose Reveal on all 45 eligible held-out turns
both before and after. The median probability assigned to Reveal increased
from 99.89% to 99.98%, and adapter weights changed, but the pilot found no
new timing behavior or win-rate gain. The replay's **Reveal RL** dataset shows
each decision and its Reveal probability.

Run another pilot with [reveal_modal_job.py](src/training/reveal_modal_job.py)
after reviewing the policy and evaluation design:

```bash
uv run modal run src/training/reveal_modal_job.py \
  --updates 2 --eligible-games-per-update 4 --eval-games 32
```
