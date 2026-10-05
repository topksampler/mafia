# Six-hour RL experiment plan (working draft)

The purpose of this session is to learn reinforcement learning hands-on while
building a Mafia agent. The first success criterion is **a measured policy
improvement on held-out games**, not merely a completed training run.

## 1. Define and verify the game

Use the [v0 rules](game_rules.md): six agents, night first, one Mafia, one
Detective, no Doctor, three discussion opportunities per survivor, equal
starting voice, and one-for-one overpowering. Keep the Godfather/referee
deterministic under a seed and keep private facts out of other agents'
observations. Save the ruleset version and configuration with every game.

Status: implemented in the single `env/` package with focused rule, privacy,
and OpenEnv WebSocket tests.

## 2. Establish baselines before training a language model

Run random and scripted agents on fixed evaluation seeds. Record Town/Mafia
win rate, completed day discussions, nights, overpower actions, and invalid
model actions. Compare candidate and baseline on the same seeds and use
separate random streams for speech, voting, and night actions.

A 200-game random-versus-random check and a ten-game frozen-Qwen diagnostic
have now run. The Qwen sample exposed self-votes and a fixed-seat tendency in
Detective investigations. Scripted opponents and a role-balanced held-out
comparison remain future evaluation work. The earlier vote-only prototype was
removed while consolidating the project around OpenEnv.

## 3. Put an open-weight Qwen player into the environment

Use LiteLLM for inference routing and keep training separate from the
inference adapter. The first hosted policy uses
[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
through Hugging Face's Nscale provider. We also tried
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), but its listed
Featherless AI provider returned 503 during the first game. One complete
ten-game Qwen diagnostic now exists; inspect its action validity, role
secrecy, latency, and conversation before running a larger sample. The hosted
model is frozen.

## 4. Make the first language-model RL update

The first experiment trains Town vote targets. Qwen sees the acting player's
public and private observations, scores legal targets, and samples a target.
Seeded random policies handle all other actions. At the end of a game, Town's
win/loss reward updates a LoRA adapter with episodic REINFORCE and a moving
reward baseline. This isolates the reward and gradient before training speech
and floor decisions. The OpenEnv wrapper and trainer use the same rules engine;
the trainer runs it in process to avoid an incompatible OpenEnv/Transformers
dependency combination in one GPU image.

The first Modal run completed six updates over 48 training games. LoRA weights
changed and the adapter was saved. On 24 held-out seeds, the original policy
won 16 Town games and the trained policy won 15; the random Town baseline won
12. This establishes a working RL loop, but does not meet the improvement
criterion yet. The [vote policy audit](vote_policy_audit.md) found a strong
player-seat preference: neither policy voted for P0 in the held-out games,
despite many opportunities. The adapter often shifted votes toward P3. The
next frozen-policy probes removed the direct digit comparison and then made
candidate prompts exactly invariant to player-name swaps. The invariant policy
passed that test but won only 9 of the same 24 development games. The vote
representation is no longer the only obstacle: random discussion shares no
useful private information. We should design speech and coordinated voting
before increasing training or updating all 4B base weights. Final evaluation
must use fresh seeds because the 24 games have informed policy design.

A small follow-up run trained the label-invariant candidate vote policy for
three updates on 12 games. On eight fresh held-out games, it went from 4 Town
wins before training to 3 after, while random Town votes won 5. The LoRA
weights changed and all votes were valid, but this did not show better play.
The [report](../runs/town-centered-vote-20261004-110540/report.json) is saved
locally, and its adapter is in the Modal Volume. We should design how Town
shares and weighs information during discussion before another GPU run.

The next control used five scripted discussion policies on 100 paired fresh
seeds, with no model calls or training. Town won 50 games when only the
Detective voted on private knowledge, 59 when they also revealed a positive
result and Villagers followed it, and 42 when Mafia could make a reactive
false counterclaim. The [experiment write-up](scripted_detective_reveal.md)
shows the per-seed vote changes and a replayable example. This supports
designing a reveal/hold learning action, while showing that Villager trust and
Mafia deception must be part of its evaluation.

The [reveal/hold policy design](reveal_hold_policy.md) now offers Qwen the
choice on every eligible Detective speech turn, with the current observation.
No reveal round is predefined. Scripted first-slot and last-round timing
controls against reactive Mafia produced 42 versus 51 Town wins on the first
100 paired seeds and 50 versus 57 on a fresh 100; late speech avoided some
counterclaims but also missed some chances to reveal. This confirms that
timing matters while leaving the choice for the policy to learn.

### Watch the training run

Modal hosts a [live chart dashboard](https://lalith1403--mafia-trackio-dashboard-dashboard.modal.run)
that the developer can open while training runs. The trainer publishes each
game reward, a 20-game rolling average, policy loss, gradient norm, weight
change, and held-out win rate. It also saves TensorBoard event files with the
run's adapter and report. Only numeric metrics and run settings are served on
the public-read dashboard; game transcripts and credentials are not there.

At regular checkpoints, evaluate on the same held-out game seeds and compare
against the random policy on those seeds. Track valid votes, warnings, and
forfeited turns separately from rewards. Record the model, ruleset, seed set,
role, and training settings. The first run also logs vote entropy. Non-finite
loss or gradients stop training. A training curve alone cannot establish that
the agent plays better; the held-out games are the check.

## 5. Extend from votes to manipulation

Let the language policy choose speech and overpower actions. Study how often
it takes another player's slot, whom it targets, how opponents respond, and
whether final votes shift. Evaluate against random, scripted, and earlier
policy snapshots. Avoid a numeric "room tilt" oracle in player observations:
the agent must infer it from public behavior.

## Open decisions before the next GPU run

- What a Detective should publicly claim, when to claim it, and how Villagers
  should weigh that claim against Mafia bluffs.
- Which speech or floor action to train after a frozen scripted policy shows
  whether information sharing changes voting.
- Whether to update all base weights after a more promising small experiment.
- Whether to add public, nonbinding reactions to make room-reading more
  observable in a later ruleset.
