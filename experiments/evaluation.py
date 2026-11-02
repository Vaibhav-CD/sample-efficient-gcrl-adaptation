"""Stage 6: full experimental evaluation (Experiments A-E, multi-seed).

- A: initial learning (per-seed train + eval).
- B: new obstacle seals the corridor passage.
- C: slip 0 -> 0.2 (global stochasticity change).
- D: 2 retained + 2 unseen goals.
- E: barrier + slip 0.15 + 2 unseen goals combined.

Methods: scratch, warm (A), warm+replay (A+B), oracle-selective (A+C),
combo (A+Bf+C), CASR, CASR-no-reset (probe+filter only), CASR-no-filter
(reset+plain replay). Metrics: N_0.9 (CASR incl. probe episodes), final
greedy success, mean/median successful path length, adapt wall-clock time.

Run from the project root (takes ~2 min):

    python -m experiments.evaluation
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms.goal_conditioned import GoalConditionedQLearner
from algorithms.knowledge_reuse import (
    ReplayBuffer,
    adapt_post_change,
    collect_transitions,
    selective_reset,
)
from algorithms.proposed_method import casr_adapt, detect_region, run_probes
from environment.dynamic_gridworld import DynamicGridWorld
from experiments.adaptation import (
    WALL,
    moving_average,
    time_to_threshold,
)

GOALS = [(7, 7), (7, 0), (0, 7), (0, 0)]
UNSEEN = [(1, 1), (6, 6)]
SEEDS = [0, 1, 2, 3, 4]
PRE_EPISODES = 4000
POST_EPISODES = 1500
WINDOW = 100
THRESHOLD = 0.9
COLLECT_EPISODES = 200
CFG = dict(alpha=0.2, gamma=0.95, epsilon_decay=0.997, epsilon_min=0.05)
METHODS = ["scratch", "warm", "warm+replay", "oracle-selective", "combo",
           "CASR", "CASR-no-reset", "CASR-no-filter"]
CURVE_METHODS = ["scratch", "warm", "oracle-selective", "CASR"]


def make_env(seed: int, goals=None):
    return DynamicGridWorld(width=8, height=8, obstacles=WALL,
                            goals=list(goals or GOALS), seed=seed)


def fresh_learner(goals, seed: int):
    return GoalConditionedQLearner(8, 8, list(goals), seed=seed, **CFG)


def transfer_slices(source, new_goals):
    """New learner on ``new_goals``; copies slices of retained goals."""
    learner = fresh_learner(new_goals, 0)
    for i, g in enumerate(new_goals):
        if tuple(g) in source.goal_index:
            learner.Q[:, :, :, i] = source.Q[:, :, :, source.goal_index[tuple(g)]]
    return learner


def apply_change(env, key: str):
    """Apply experiment change; returns (oracle_cells, new_goals_or_None)."""
    if key == "B":
        return env.apply_environment_change(add_obstacles=[(3, 7)]), None
    if key == "C":
        return env.apply_environment_change(new_slip=0.2), None
    if key == "D":
        goals = [GOALS[0], GOALS[1], UNSEEN[0], UNSEEN[1]]
        return env.apply_environment_change(new_goals=goals), goals
    if key == "E":
        goals = [GOALS[0], GOALS[1], UNSEEN[0], UNSEEN[1]]
        env.apply_environment_change(add_obstacles=[(3, 7)])
        env.apply_environment_change(new_slip=0.15)
        return env.apply_environment_change(new_goals=goals), goals
    raise ValueError(key)


def adapt_method(name, env, pretrained, buffer, oracle_cells, new_goals, seed,
                 post_episodes=POST_EPISODES):
    """Run one adaptation; returns dict with learner, history, extra, time."""
    goals = new_goals or GOALS
    buf = buffer
    if new_goals is not None and buffer is not None:
        keep_goals = {tuple(g) for g in goals}
        buf = ReplayBuffer(seed=seed)
        buf._data = [t for t in buffer._data if t[5] in keep_goals]
    t0 = time.perf_counter()
    extra = 0
    if name == "scratch":
        learner = fresh_learner(goals, seed)
        history, _ = adapt_post_change(env, learner, post_episodes, eps_start=1.0)
    elif name == "warm":
        learner = transfer_slices(pretrained, goals)
        history, _ = adapt_post_change(env, learner, post_episodes)
    elif name == "warm+replay":
        learner = transfer_slices(pretrained, goals)
        history, _ = adapt_post_change(env, learner, post_episodes, buffer=buf,
                                       replay_per_episode=50)
    elif name == "oracle-selective":
        learner = transfer_slices(pretrained, goals)
        selective_reset(learner.Q, oracle_cells, radius=2)
        history, _ = adapt_post_change(env, learner, post_episodes)
    elif name == "combo":
        learner = transfer_slices(pretrained, goals)
        selective_reset(learner.Q, oracle_cells, radius=2)
        history, _ = adapt_post_change(env, learner, post_episodes, buffer=buf,
                                       replay_per_episode=50,
                                       consistency_filter=True)
    elif name == "CASR":
        learner = transfer_slices(pretrained, goals)
        report = casr_adapt(env, learner, buffer=buf, probe_episodes=30,
                            margin=1, replay_per_episode=50,
                            adapt_episodes=post_episodes, probe_seed=seed)
        history = report["history"]
        extra = report["probe_episodes"]
    elif name == "CASR-no-reset":
        learner = transfer_slices(pretrained, goals)
        probe = run_probes(env, learner, 30, seed=seed)
        blocked = set(probe["inferred_blocked"])
        keep = lambda s, a, sp: tuple(sp) not in blocked and tuple(s) not in blocked
        history, _ = adapt_post_change(env, learner, post_episodes, buffer=buf,
                                       replay_per_episode=50, keep=keep)
        extra = 30
    elif name == "CASR-no-filter":
        learner = transfer_slices(pretrained, goals)
        probe = run_probes(env, learner, 30, seed=seed)
        detection = detect_region(probe, env.height, env.width, margin=1)
        selective_reset(learner.Q, detection["seeds"], radius=1)
        history, _ = adapt_post_change(env, learner, post_episodes, buffer=buf,
                                       replay_per_episode=50)
        extra = 30
    else:
        raise ValueError(name)
    return {"learner": learner, "history": history, "extra": extra,
            "time": time.perf_counter() - t0}


def greedy_eval(env, learner, goals, episodes=50, seed=1000):
    """Greedy rollouts; returns (success_rate, [successful lengths])."""
    wins, lengths = 0, []
    save_eps = learner.epsilon
    learner.epsilon = 0.0
    try:
        for i in range(episodes):
            goal = goals[i % len(goals)]
            state, _ = env.reset(seed=seed + i, goal=goal)
            done = terminated = False
            while not done:
                action = learner.act(state, goal, explore=False)
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            wins += terminated
            if terminated:
                lengths.append(env.steps)
    finally:
        learner.epsilon = save_eps
    return wins / episodes, lengths


def run_all(seeds=SEEDS, pre_episodes=PRE_EPISODES, post_episodes=POST_EPISODES):
    """Full protocol; returns (records DataFrame, curves dict).

    curves[exp][method] = list over seeds of moving-average success arrays.
    """
    records, curves = [], {e: {m: [] for m in CURVE_METHODS} for e in "BCDE"}
    for seed in seeds:
        env = make_env(seed)
        pretrained = fresh_learner(GOALS, seed)
        t0 = time.perf_counter()
        pre_hist = pretrained.train(env, episodes=pre_episodes)
        pre_time = time.perf_counter() - t0
        pre_eval, pre_lens = greedy_eval(env, pretrained, GOALS)
        records.append({"exp": "A", "method": "initial-train", "seed": seed,
                        "N_0.9": time_to_threshold(pre_hist),
                        "success": pre_eval,
                        "mean_len": float(np.mean(pre_lens)),
                        "median_len": float(np.median(pre_lens)),
                        "time": pre_time, "reached": True})
        buffer = collect_transitions(env, pretrained, episodes=COLLECT_EPISODES,
                                     seed=seed)
        for exp in "BCDE":
            env_e = make_env(seed)
            learner_e = fresh_learner(GOALS, seed)
            learner_e.train(env_e, episodes=pre_episodes)
            oracle_cells, new_goals = apply_change(env_e, exp)
            goals = new_goals or GOALS
            for method in METHODS:
                out = adapt_method(method, env_e, learner_e, buffer,
                                   oracle_cells, new_goals, seed,
                                   post_episodes=post_episodes)
                n = time_to_threshold(out["history"])
                reached = n is not None
                succ, lens = greedy_eval(env_e, out["learner"], goals)
                records.append({
                    "exp": exp, "method": method, "seed": seed,
                    "N_0.9": (n + out["extra"]) if reached else np.nan,
                    "success": succ,
                    "mean_len": float(np.mean(lens)) if lens else np.nan,
                    "median_len": float(np.median(lens)) if lens else np.nan,
                    "time": out["time"], "reached": reached})
                if method in CURVE_METHODS:
                    curves[exp][method].append(
                        moving_average(out["history"]["success"]))
    return pd.DataFrame(records), curves


def plot_curves(curves, path="docs/eval_curves_stage6.png"):
    _, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, exp in zip(axes.flat, "BCDE"):
        for method in CURVE_METHODS:
            arr = np.array(curves[exp][method])
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            x = np.arange(len(mean))
            ax.plot(x, mean, label=method)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.axhline(THRESHOLD, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(f"Experiment {exp}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Post-change episode (100-ep moving avg)")
        ax.set_ylabel("Success rate (mean ± std)")
    axes.flat[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=100)


def main():
    df, curves = run_all()
    df.to_csv("docs/results_stage6.csv", index=False)
    summary = df.groupby(["exp", "method"]).agg(
        N_mean=("N_0.9", "mean"), N_std=("N_0.9", "std"),
        success_mean=("success", "mean"), success_std=("success", "std"),
        meanlen_mean=("mean_len", "mean"), medlen_mean=("median_len", "mean"),
        time_mean=("time", "mean"), reached=("reached", "sum"),
        n=("seed", "count")).reset_index()
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(summary.to_string(index=False))
    print("\nsaved docs/results_stage6.csv")
    plot_curves(curves)
    print("saved docs/eval_curves_stage6.png")


if __name__ == "__main__":
    main()
