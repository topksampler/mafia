"""REINFORCE on a 3-armed bandit. Same update rule as src/training/vote.py:387.

    loss = -advantage * log_prob(action) - entropy_coef * entropy

Arm i pays +1 with probability WIN[i], else -1 (like Mafia win/loss).
Expected reward per arm: 2p-1 -> [-0.4, 0.0, +0.4]. Arm 2 is best.

Try:  python learn/01_bandit.py
      python learn/01_bandit.py --baseline ema
      python learn/01_bandit.py --lr 1.0 --seeds 20
"""
import argparse
import numpy as np

WIN = np.array([0.3, 0.5, 0.7])


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def run(seed, steps, lr, baseline_mode, entropy_coef, batch):
    rng = np.random.default_rng(seed)
    theta = np.zeros(3)            # the policy's parameters (logits)
    baseline = 0.0
    for _ in range(steps):
        pi = softmax(theta)
        actions = rng.choice(3, size=batch, p=pi)
        rewards = np.where(rng.random(batch) < WIN[actions], 1.0, -1.0)
        advantages = rewards - (baseline if baseline_mode == "ema" else 0.0)

        grad = np.zeros(3)
        for a, adv in zip(actions, advantages):
            dlogp = -pi.copy()     # d log pi(a) / d theta = onehot(a) - pi
            dlogp[a] += 1.0
            grad += adv * dlogp
        grad /= batch
        # entropy bonus gradient: d H / d theta_i = -pi_i * (log pi_i + H)
        H = -(pi * np.log(pi)).sum()
        grad += entropy_coef * (-pi * (np.log(pi) + H))

        theta += lr * grad         # gradient ASCENT on reward
        baseline = 0.9 * baseline + 0.1 * rewards.mean()
    return softmax(theta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--baseline", choices=["none", "ema"], default="none")
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=8, help="games per update")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()

    finals = np.array([run(s, a.steps, a.lr, a.baseline, a.entropy, a.batch) for s in range(a.seeds)])
    best = finals[:, 2]
    print(f"config: lr={a.lr} baseline={a.baseline} entropy={a.entropy} batch={a.batch} steps={a.steps}")
    print(f"P(best arm) per seed: {np.round(best, 2)}")
    print(f"mean P(best arm)={best.mean():.2f}   seeds that picked wrong arm: {(finals.argmax(1) != 2).sum()}/{a.seeds}")
