# Learnable Detective reveal/hold policy

The Detective's timing strategy belongs to the model. The
[scripted reveal experiment](scripted_detective_reveal.md) and
[timing probe](../src/training/reveal_timing_probe.py) are controls for
understanding the game; neither tells Qwen when to speak.

## One decision, offered repeatedly

On **every actual Detective speaking turn** where their private investigation
has identified a living Mafia player, the environment passes the current
observation to the policy. This includes day and discussion round, living
players, all public speech and votes so far, and the Detective's own private
investigation result. It does not include the referee's role map or other
players' private information.

Qwen chooses one of two legal actions:

| Choice | Public action | Turn cost |
| --- | --- | --- |
| Reveal | Say the true positive investigation in a fixed, explicit sentence | One speaking turn |
| Hold | Say nothing | One speaking turn |

The choice is offered again at the next eligible Detective speaking turn.
After an earlier reveal, choosing Reveal again repeats or reasserts the true
claim. A Mafia counterclaim, changing votes, an overpower event, or a new day
can therefore change Qwen's next choice. If a turn is taken by overpowering,
there is no Detective decision for that lost slot. The
[action adapter](../src/training/reveal_choice.py) implements this boundary
without choosing a time for the model.

For this first learning problem, a negative investigation does not open this
binary action. Clearing Town players and choosing whom to investigate are
later action families.

## Where learning happens

1. The game engine gives Qwen the Detective's current view.
2. Qwen produces a probability for Reveal and Hold and samples one during
   training. The controller turns that choice into a legal `speak` action.
3. The game continues with Villager and Mafia policies. Every later eligible
   Detective slot gets a fresh Qwen decision from the new public state.
4. At game end, Town receives +1 for a win, -1 for a loss, and 0 for a draw.
   REINFORCE updates the chosen action probabilities from all eligible
   decisions in that game. A moving reward baseline and per-game decision
   normalization reduce noise. Games with no eligible decision produce no
   reveal-policy gradient.

This is sparse credit assignment: a win does not tell us which individual
speech turn mattered. We will log each choice, its probability, the current
round, prior public claims, counterclaims, later votes, and the final outcome.
We will not reward the act of revealing itself, because that would replace
the strategy goal with a scripted preference.

Eligible choices were rare in the timing controls: only 24 of the first 100
games and 27 of the fresh 100 had any Detective speech turn with a living
positive target. A 12-game training run like the earlier vote experiment
would usually contain very few useful reveal decisions. We can deliberately
sample more eligible training episodes for gradient efficiency, but held-out
evaluation must use the natural game distribution.

The initial optimizer can update a LoRA adapter on the open-weight Qwen 4B
model. That is still reinforcement learning: sampled choices affect game
outcomes, and their probabilities change through the policy gradient. The
action definition does not depend on LoRA versus full-weight training.

## Why timing is worth learning

Against a scripted reactive Mafia counterclaim, we compared two fixed timing
controls. Both spent every eligible Detective speech turn as `speak`; one
revealed at the first opportunity and the other held until discussion round
three.

| Paired seed set | Reveal first: Town wins | Hold until round three: Town wins |
| --- | ---: | ---: |
| 70000–70099 | 42 / 100 | 51 / 100 |
| 71000–71099 | 50 / 100 | 57 / 100 |

The later control gave Mafia fewer chances to counterclaim, but it also
sometimes lost the chance to reveal at all. The
[first timing report](../runs/reveal-timing-seeds-70000-70099/report.json)
and [fresh timing report](../runs/reveal-timing-seeds-71000-71099/report.json)
contain paired outcomes and full traces. In seed 70017, an early reveal lets
Mafia make a later false claim and Town loses; a round-three reveal leaves no
Mafia response slot and Town wins. In seed 70014, waiting misses the final
Detective slot and does not help. These are scripted mechanisms, not learned
Qwen behavior.

## Opponents and evaluation

An always-trusting Villager makes Reveal artificially strong when Mafia never
lies and artificially weak when Mafia can make the newest claim. Training and
evaluation therefore need several declared opponent conditions: random
speech, reactive counterclaims, proactive bluffs, and a less gullible
Villager response. The mix used for learning is an explicit design choice;
the opponent condition should not be handed to Qwen as a hidden-state hint.

Before a GPU run, we should decide the Qwen prompt and two-action scoring
scheme and check for label bias, as the vote policy's digit labels caused a
real failure. Evaluation should compare frozen Qwen, trained Qwen, always
Hold, always Reveal, first-slot, and last-round controls on **new seeds**
separate from the 70000 and 71000 design probes. Report Town wins together
with reveal frequency by round, missed opportunities, counterclaims, and
subsequent vote changes. A rising training reward alone will not establish
that the strategy improved.
