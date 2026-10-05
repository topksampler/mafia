# First ten frozen-Qwen games

Model: `Qwen/Qwen3-4B-Instruct-2507` through Hugging Face Inference Providers (Nscale). One identical, frozen model policy controlled all six seats. Environment seeds were 1 through 10. No weights were updated.

## Observed results

| Measure | Result |
| --- | ---: |
| Town wins | 8 / 10 |
| Mafia wins | 2 / 10 |
| Draws | 0 |
| Action attempts | 267 |
| Accepted actions | 253 |
| First warnings | 9 |
| Forfeited turns after a second illegal move | 5 |
| Overpower attempts | 3 (1 accepted) |
| Input tokens | 381,243 |
| Output tokens | 14,874 |

These are diagnostic games. Ten outcomes are too few to estimate a stable win rate or claim that Qwen is stronger than another policy. We also used the same model for both teams, so the result mixes Town behavior, Mafia behavior, and the game's starting advantage.

## What the traces revealed

- Nine of the 14 rejected action attempts were self-votes. The observation already excluded the acting player from legal vote targets. Four of those sequences ended in a forfeited vote.
- The Detective selected P2 on the first night in 9 of 10 games. P2 happened to be Mafia in 4 of the 10 games. This fixed-seat tendency makes the ten-game Town win rate especially fragile.
- Two speeches exceeded the 500-character limit. One speech attempt used `vote` during discussion. Two overpower attempts chose a target without an open slot; the second forfeited that speaking turn.
- The first game showed a meaningful conversational pattern: Detective P3 correctly accused Mafia P2, while P2 challenged the private investigation claim, and Town eliminated P3. That is a single observed episode, not evidence of a reliable manipulation skill.

For context, a separate local random-versus-random check over 200 seeds gave 85 Town wins and 115 Mafia wins. The opponent policies and sample sizes differ, so that is not a direct head-to-head comparison.

The environment seed fixes roles and turn order. Qwen sampling used temperature 0.7 without a provider-side generation seed, so replaying a seed can produce a different conversation. The saved JSONL traces preserve the actual decisions. The aggregate machine-readable result is `qwen3-4b-instruct-2507-seeds1-10-eval.json`.
