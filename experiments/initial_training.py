"""Stage 3 experiment: train baselines in the initial environment.

Builds one 8x8 environment with a wall corridor, trains tabular Q-learning
(single goal) and goal-conditioned Q-learning (four goals), evaluates greedy
success rates, and saves success-rate learning curves to
``docs/learning_curves_stage3.png``.

Run from the project root:

    python -m experiments.initial_training
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithms.bfs import bfs
from algorithms.dijkstra import dijkstra
from algorithms.goal_conditioned import GoalConditionedQLearner
from algorithms.q_learning import TabularQLearner
from environment.gridworld import GridWorld

WALL = [(3, c) for c in range(1, 7)]
GOALS = [(7, 7), (7, 0), (0, 7), (0, 0)]
SEED = 0


def moving_average(values, window: int = 100):
    values = np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    env = GridWorld(width=8, height=8, obstacles=WALL, goals=GOALS, seed=SEED)

    oracle_len = len(bfs(8, 8, WALL, (0, 0), (7, 7))) - 1
    _, oracle_cost = dijkstra(8, 8, WALL, (0, 0), (7, 7))

    t0 = time.perf_counter()
    q = TabularQLearner(8, 8, seed=SEED)
    q_hist = q.train(env, episodes=2000, start=(0, 0), goal=(7, 7))
    q_time = time.perf_counter() - t0
    q_success, q_len = q.evaluate(env, episodes=50, start=(0, 0), goal=(7, 7))

    t0 = time.perf_counter()
    gcq = GoalConditionedQLearner(8, 8, GOALS, seed=SEED)
    gcq_hist = gcq.train(env, episodes=4000)
    gcq_time = time.perf_counter() - t0
    gcq_success, gcq_len = gcq.evaluate(env, episodes=80)

    print(f"BFS shortest path (0,0)->(7,7): {oracle_len} steps")
    print(f"Dijkstra uniform cost (0,0)->(7,7): {oracle_cost}")
    print(
        f"TabularQ: success={q_success:.2f} mean_len={q_len:.1f} "
        f"time={q_time:.1f}s episodes=2000"
    )
    print(
        f"GoalCondQ: success={gcq_success:.2f} mean_len={gcq_len:.1f} "
        f"time={gcq_time:.1f}s episodes=4000"
    )

    _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(moving_average(q_hist["success"]), label="Tabular Q (single goal)")
    ax.plot(moving_average(gcq_hist["success"]), label="Goal-conditioned Q (4 goals)")
    ax.set_xlabel("Episode (100-episode moving average)")
    ax.set_ylabel("Success rate")
    ax.set_title("Stage 3: initial-environment learning curves")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig("docs/learning_curves_stage3.png", dpi=100)
    print("saved docs/learning_curves_stage3.png")


if __name__ == "__main__":
    main()
