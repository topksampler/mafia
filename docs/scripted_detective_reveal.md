# Scripted Detective reveal experiment

This is a controlled **scripted-policy experiment**, not an LLM evaluation or
a training run. It asks a narrow question before we train a reveal decision:
does a Detective's true information change the Town vote, and what happens
when Mafia lies in the same public format?

The [experiment code](../src/training/scripted_reveal.py) ran five arms on the
same 100 seeds, 70000–70099. Every arm uses the same six-player v0.4 rules and
separate seeded random streams for each actor and action phase. The first
night's roles, kill, and investigation match across arms for each seed.

| Arm | Detective and Villager rule | Mafia speech |
| --- | --- | --- |
| Random | All actions follow the existing seeded random policy | Random |
| Private vote | Detective votes for privately confirmed living Mafia but holds the result with an empty first speech; Villagers vote randomly | Random |
| Reveal | Same first speech slot, but Detective states a positive result; Villagers vote for the latest living claimant's current-day legal target | Random |
| Counterclaim | Same as Reveal | Mafia makes one false Detective claim after seeing a public Detective-style claim |
| Bluff | Same as Reveal | Mafia proactively makes one false Detective claim per day |

The Villager reads only public speech. It does not know who really holds the
Detective role. Claims must match an exact scripted sentence; other random
accusations are ignored by this simple response rule. Mafia knows that every
other player is Town in this one-Mafia ruleset and chooses a false target from
living opponents using its seeded random stream.

The Private vote and Reveal arms both spend the Detective's first eligible
speech slot as a normal `speak` action. One says nothing; the other gives the
true result. This holds the floor action fixed while testing the effect of
sharing the information and the Villager response it triggers.

## Results

| Arm | Town wins / 100 | Villager votes for Mafia | Villager claim-follow votes that hit Mafia |
| --- | ---: | ---: | ---: |
| Random | 44 | 189 / 618 | — |
| Private vote | 50 | 185 / 601 | — |
| Reveal | 59 | 211 / 562 | 58 / 58 |
| Counterclaim | 42 | 191 / 633 | 31 / 107 |
| Bluff | 28 | 115 / 555 | 52 / 386 |

The paired outcomes show where complete-game results changed:

- Random → Private vote: seven Town losses became wins; one win became a loss.
- Private vote → Reveal: nine losses became wins; no wins became losses.
- Reveal → Counterclaim: 17 wins became losses; none moved the other way.
- Reveal → proactive Bluff: 35 wins became losses; four losses became wins.

In 17 seeds, the Detective found Mafia on the first night and survived dawn.
Those first-night conditions are identical across arms. Town won 11 of those
17 under Private vote, all 17 under Reveal, and four under Counterclaim.

[Seed 70014](../runs/scripted-reveal-seeds-70000-70099/games.jsonl) is a useful
replay: on day 2, Detective P3 privately knows that P1 is Mafia. Private voting
fails because P3's vote is alone. A public reveal gets Villager P5 to vote P1,
eliminating Mafia. In the counterclaim arm, Mafia P1 falsely claims P3 is
Mafia; P5 follows that later claim, and Town loses.

The [report](../runs/scripted-reveal-seeds-70000-70099/report.json) has
per-seed outcomes, paired flips, vote counts, and claim context.
The [full traces](../runs/scripted-reveal-seeds-70000-70099/games.jsonl) have
one replay-ready JSON object per game, with every action, actor observation,
public event change, policy reason, final role map, and outcome. They contain
private observations and are intended for local evaluation.

## What this establishes

Sharing a true private result can coordinate Town voting under this response
rule. It also creates an opening for Mafia manipulation when Villagers trust
the latest claim without checking credibility. The proactive Bluff arm is a
stress test of that weak response rule; its large effect is not an estimate
of how a trained LLM Mafia would behave.

These scripted controls do not show that Qwen has learned a reveal strategy.
The next learning action to discuss is **reveal or hold** after a positive
investigation. We need to settle when that choice is offered, what holding
means for the remaining speech slots, and which mix of honest and deceptive
opponents should appear during training and held-out evaluation. The Villager
response policy also needs deliberate design before treating Town win rate as
a measure of the Detective's communication skill.
