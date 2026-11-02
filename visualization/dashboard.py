"""Shared dashboard helpers ( Streamlit-free so they stay unit-testable).

Plotting uses Matplotlib; data comes from the real Stage-6 CSV and from
small live-trained demo agents (fast: sub-second on laptop CPU).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms.bfs import bfs
from algorithms.goal_conditioned import GoalConditionedQLearner
from algorithms.q_learning import TabularQLearner
from environment.dynamic_gridworld import DynamicGridWorld


def load_results(path="docs/results_stage6.csv") -> pd.DataFrame:
    """Load the actual Stage-6 per-seed records."""
    return pd.read_csv(path)


def summarize(df: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Mean ± std per method for one experiment (NaN-aware for N_0.9)."""
    sub = df[df["exp"] == exp]
    return sub.groupby("method").agg(
        N_mean=("N_0.9", "mean"), N_std=("N_0.9", "std"),
        success_mean=("success", "mean"),
        meanlen_mean=("mean_len", "mean"),
        time_mean=("time", "mean"), reached=("reached", "sum"),
    ).reset_index()


def draw_grid(env, path=None, ax=None, title=""):
    """Render grid with obstacles, agent, goal and an optional route."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    grid = np.zeros((env.height, env.width))
    for (r, c) in env.obstacles:
        grid[r, c] = -1
    ax.imshow(grid, cmap="Greys", vmin=-1, vmax=1, origin="upper")
    if path:
        rows, cols = zip(*[tuple(cell) for cell in path])
        ax.plot(cols, rows, "b.-", linewidth=2, markersize=6)
    if getattr(env, "agent_pos", None) is not None:
        ax.plot(env.agent_pos[1], env.agent_pos[0], "bo", markersize=12)
    if getattr(env, "current_goal", None) is not None:
        ax.plot(env.current_goal[1], env.current_goal[0], "g*",
                markersize=16, markeredgecolor="k")
    ax.set_title(title)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="0.8", linewidth=0.5)
    return ax


def greedy_route(env, learner, goal, start, max_steps=200):
    """Roll out the greedy policy; returns (path, reached)."""
    state, _ = env.reset(start=start, goal=goal)
    path, reached = [state], False
    save = learner.epsilon
    learner.epsilon = 0.0
    try:
        for _ in range(max_steps):
            a = learner.act(state, goal, explore=False) \
                if hasattr(learner, "goals") else learner.act(state, explore=False)
            state, _, reached, truncated, _ = env.step(a)
            path.append(tuple(state))
            if reached or truncated:
                break
    finally:
        learner.epsilon = save
    return path, reached


def train_demo(width=6, height=6, episodes=800, seed=0):
    """Tiny live demo: returns (env, bfs_len, q_success, Qmax)."""
    obstacles = [(2, c) for c in range(1, 5)]
    env = DynamicGridWorld(width=width, height=height, obstacles=obstacles,
                           goals=[(height - 1, width - 1)], seed=seed)
    bfs_path = bfs(width, height, obstacles, (0, 0), (height - 1, width - 1))
    learner = TabularQLearner(height, width, seed=seed)
    learner.train(env, episodes=episodes, start=(0, 0),
                  goal=(height - 1, width - 1))
    success, _ = learner.evaluate(env, episodes=10, start=(0, 0),
                                  goal=(height - 1, width - 1))
    return env, len(bfs_path) - 1, success, np.max(learner.Q, axis=2), learner


def before_after_demo():
    """Stage-4 protocol at demo scale: old vs new optimal routes + envs."""
    wall = [(3, c) for c in range(1, 7)]
    before = DynamicGridWorld(width=8, height=8, obstacles=wall,
                              goals=[(7, 7)], seed=0)
    old_route = bfs(8, 8, wall, (0, 0), (7, 7))
    after = DynamicGridWorld(width=8, height=8, obstacles=wall,
                             goals=[(7, 7)], seed=0)
    after.apply_environment_change(add_obstacles=[(3, 7)])
    new_route = bfs(8, 8, wall + [(3, 7)], (0, 0), (7, 7))
    return before, old_route, after, new_route
