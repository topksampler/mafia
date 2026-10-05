# First frozen-Qwen Mafia rollout

- Model: `Qwen/Qwen3-4B-Instruct-2507` through Hugging Face Inference Providers (Nscale)
- Game seed: `1`
- Result: Mafia won after the second day vote
- Decisions: 36 attempts; 35 accepted; one public warning for a speech longer than 500 characters; no forfeits
- Actions: two kills, one investigation, 25 speech attempts, eight votes, no overpower actions
- Token usage: 58,949 input and 2,142 output tokens

## What happened

**Night 1:** Mafia killed P1, a Villager. Detective P3 privately investigated P2 and learned that P2 was Mafia.

**Day 1:** P3 publicly claimed the investigation result. P2 denied it, claimed to be a Villager, and challenged the lack of a *public* investigation announcement. The Godfather's investigation answer was correctly private; P2's argument was a player claim. P0, P2, and P4 voted for P3. P3 and P5 voted for P2. P3 was eliminated, revealing the Detective role.

**Night 2:** Mafia killed P5, a Villager.

**Day 2:** P2's first speech exceeded the 500-character limit. The Godfather warned P2 publicly and kept the turn with P2. P2's legal retry was accepted. P2 and P4 voted for P0; P0 voted for P4. P0 was eliminated, and Mafia reached parity with Town.

This single game shows one plausible manipulation pattern, not a measured policy improvement. The model was frozen throughout. The full evaluator trace, including each prompt and raw reply, is in `qwen3-4b-instruct-2507-seed1.jsonl`. That trace contains private player observations and must stay with the evaluator.
