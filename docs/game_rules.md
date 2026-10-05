# Mafia game specification, v0.4 historical rules

The scheduled discussion below is the ruleset used for saved v0.4 games and
training reports. The [open-floor design draft](open_floor_policy.md) records
the replacement being designed with the user. Do not use v0.4 results as
evidence about the new discussion rules.

This document records the rules we are using for the first hands-on RL experiment.
The Godfather is the environment/referee, **not** a player. All six player agents
begin with equal discussion power. Future versions may assign power as a random
trait; that is outside v0.

## Players and objective

- Six players: one Mafia, one Detective, and four Villagers. The Detective and
  Villagers form Town. There is no Doctor.
- Town wins when Mafia is eliminated. Mafia wins as soon as living Mafia count
  reaches living Town count.
- A game unresolved after five day votes is a draw. Winning players receive
  `+1`, losing players `-1`, and all players receive `0` for a draw.

## Order of play

1. **Night 1 begins immediately.** Mafia privately selects one living non-Mafia
   player to kill. The Detective privately selects one other living player and
   receives a yes/no Mafia answer from the Godfather.
2. The Godfather resolves the kill at dawn and announces the victim. If the
   Detective was targeted, their investigation still happened, but they die
   before speaking and cannot pass its result to Town.
3. **Day 1 discussion.** Every survivor starts with three scheduled speaking
   opportunities. The Godfather shuffles speaker order from the game seed and
   rotates that order between rounds. Messages become public immediately.
4. **Day 1 vote.** Every survivor privately votes for another living player.
   After all votes are in, the Godfather reveals the ballot. A unique
   highest-vote player is eliminated. If the top vote ties, all survivors
   vote again with choices restricted to the tied leaders. There is no new
   discussion round. The runoff ballot is revealed after everyone votes.
   If the runoff also ties, the Godfather draws one of its top tied leaders
   using the game seed. Every day vote eliminates exactly one player.
5. Check victory. If nobody has won, Night 2 begins, then Day 2, and so on.

## Discussion and overpowering

- On a scheduled opportunity, an agent can `speak` a message of up to 500
  characters. The message may be empty; this consumes the turn and means the
  agent has nothing more to say for that opportunity. An agent may also say
  "I'm done" in ordinary speech. The Godfather never verifies statements or
  role claims.
- Instead of speaking in its current slot, an agent may `overpower` another
  survivor. This **uses one of the actor's three turns** and reserves the
  target's next unclaimed slot. When that slot arrives, the actor's message
  appears in the target's place and the target loses that speaking chance.
  The replacement message may also be empty. A public floor-claim event
  records who targeted whom.
- An agent may not overpower itself or a player with no remaining open slot.
  All agents have the same rules and opportunities in v0. No player has a
  stronger starting voice.
- Discussion remains bounded for training: three scheduled slots per living
  player, including slots that are later taken by another player.

## Invalid moves and referee announcements

- On the first illegal move in a turn, the Godfather issues a stern public
  warning. The move takes no action, does not consume the turn, and the same
  player tries again. A legal retry proceeds normally.
- If the retry is also illegal, the Godfather publicly dismisses that turn.
  No action is taken: a forfeited speaking slot is silent, a forfeited vote is
  an abstention, a forfeited Mafia night action causes no kill, and a forfeited
  Detective night action produces no investigation answer. The game then
  advances to the next turn or phase. Even if every day vote is forfeited,
  the runoff and seeded draw still eliminate one player.
- Both announcements enter the public transcript. At night, they do not name
  the actor or attempted target, preserving hidden roles and choices. A
  warning or dismissal has no intermediate game reward; terminal win/loss
  rewards still apply when the game ends.
- The Detective's yes/no investigation result remains private; it is not a
  public rule announcement.
- OpenEnv observations identify whether the action was accepted or the turn
  was forfeited, so a controller can retry once and then move to the next
  player if necessary.

## Information boundaries

- A player observes their own role, the living players, public events, their
  own investigation answers (if any), and their legal actions.
- Mafia's target is hidden until dawn. The Detective's target and answer stay
  private. Votes stay hidden until all living players have voted.
- At dawn and after a vote elimination, the eliminated player's role is
  revealed. The referee log may contain every role for evaluation, but those
  roles are never placed in another player's observation.
- Speech, public claims, and floor actions can be deceptive. The Godfather
  reports actions and outcomes; it does not provide an oracle for "room tilt."
  Agents must infer that from what others say and do.

## Working choices to review

The conversation established the night-first order, six-player starting
point, no Doctor, three discussion opportunities, equal starting voice,
one-for-one overpowering cost, and a guaranteed elimination after a day vote.
The following implementation choices are provisional and easy to change:
mandatory private voting, no self-votes, full role reveal on elimination,
one runoff before a seeded draw, 500-character speech cap, and the five-day
draw limit.

## Next environment version

Add a seeded, randomly initialized **power trait** after v0 is understood.
Power should affect floor control through an explicit rule rather than through
a Godfather judgment of persuasive text. Evaluate it against v0 using the
same seeds, role balance, and opponent policies.
