"""Stage 4 experiment: compare adaptation mechanisms after an obstacle change.

Protocol: train goal-conditioned Q in the initial 8x8 env, collect a replay
buffer, insert a vertical barrier (cells (4,4),(5,4),(6,4)), then adapt with
each method for a fixed post-change budget and compare time-to-threshold.

Methods: scratch (retraining baseline), warm start (A), warm + plain replay
(A+B), warm + selective reset (A+C), full combo (A+B-filtered+C).

Run from the project root:

    python -m experiments.adaptation
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithms.goal_conditioned import GoalConditionedQLearner
from algorithms.knowledge_reuse import (
    adapt_post_change,
    collect_transitions,
    selective_reset,
    transfer_q_values,
    transition_consistent,
)
from environment.dynamic_gridworld import DynamicGridWorld
from experiments.retraining import retrain_from_scratch

WALL = [(3, c) for c in range(1, 7)]
BARRIER = [(3, 7)]  # seals the right-hand corridor passage; cross-wall routes must re-learn via (3, 0)
GOALS = [(7, 7), (7, 0), (0, 7), (0, 0)]
SEED = 0
PRE_EPISODES = 4000
POST_EPISODES = 1500
WINDOW = 100
THRESHOLD = 0.9


def moving_average(values, window=WINDOW):
    return np.convolve(np.asarray(values, dtype=float),
                       np.ones(window) / window, mode="valid")


def time_to_threshold(history, threshold=THRESHOLD, window=WINDOW):
    curve = moving_average(history["success"], window)
    hit = np.flatnonzero(curve >= threshold)
    return int(hit[0] + window) if len(hit) else None


def final_success(history, last: int = 200):
    return float(np.mean(history["success"][-last:]))


def main():
    env = DynamicGridWorld(width=8, height=8, obstacles=WALL, goals=GOALS, seed=SEED)
    _CFG = dict(alpha=0.2, gamma=0.95, epsilon_decay=0.997, epsilon_min=0.05)

    pretrained = GoalConditionedQLearner(8, 8, GOALS, seed=SEED, **_CFG)
    pretrained.train(env, episodes=PRE_EPISODES)
    pre_success, _ = pretrained.evaluate(env, episodes=80)
    buffer = collect_transitions(env, pretrained, episodes=200, seed=SEED)
    print(f"pre-change success={pre_success:.2f} buffer={len(buffer)} transitions")

    changed = env.apply_environment_change(add_obstacles=BARRIER)
    stale = sum(1 for (s, a, r, sp, t, g) in buffer._data
                if not transition_consistent(env, s, a, sp))
    print(f"change: {len(BARRIER)} new obstacles, {len(changed)} cells affected, "
          f"{stale}/{len(buffer)} stored transitions now inconsistent")

    results = {}

    def fresh():
        return GoalConditionedQLearner(8, 8, GOALS, seed=SEED, **_CFG)

    # Scratch retraining baseline (fresh values, epsilon 1.0 -> decay).
    _, hist = retrain_from_scratch(fresh, env, POST_EPISODES)
    results["scratch"] = hist

    def warmed():
        learner = fresh()
        learner.Q = transfer_q_values(pretrained.Q)
        return learner

    results["warm (A)"], _ = adapt_post_change(env, warmed(), POST_EPISODES)
    results["warm+replay (A+B)"], n_plain = adapt_post_change(
        env, warmed(), POST_EPISODES, buffer=buffer, replay_per_episode=50)
    reset_learner = warmed()
    n_reset = selective_reset(reset_learner.Q, changed, radius=2)
    results["warm+selective (A+C)"], _ = adapt_post_change(
        env, reset_learner, POST_EPISODES)
    combo_learner = warmed()
    selective_reset(combo_learner.Q, changed, radius=2)
    results["combo (A+Bf+C)"], n_filt = adapt_post_change(
        env, combo_learner, POST_EPISODES, buffer=buffer,
        replay_per_episode=50, consistency_filter=True)
    print(f"selective reset cleared {n_reset} states; "
          f"plain replay applied {n_plain} updates, filtered replay {n_filt}")

    print(f"\n{'method':22s} {'N_0.9':>7s} {'final':>7s}")
    for name, hist in results.items():
        print(f"{name:22s} {str(time_to_threshold(hist)):>7s} "
              f"{final_success(hist):7.2f}")

    _, ax = plt.subplots(figsize=(8, 4.5))
    for name, hist in results.items():
        ax.plot(moving_average(hist["success"]), label=name)
    ax.axhline(THRESHOLD, color="k", linestyle="--", label="90% threshold")
    ax.set_xlabel(f"Post-change episode ({WINDOW}-episode moving average)")
    ax.set_ylabel("Success rate")
    ax.set_title("Stage 4: adaptation after barrier insertion")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig("docs/adaptation_curves_stage4.png", dpi=100)
    print("saved docs/adaptation_curves_stage4.png")


if __name__ == "__main__":
    main()
