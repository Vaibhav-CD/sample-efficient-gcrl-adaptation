#!/usr/bin/env python3
"""Unified CLI runner for Sample-Efficient Adaptation in Goal-Conditioned RL.

Usage:
    python run.py test               # Run all 50 unit tests
    python run.py dashboard          # Launch Streamlit interactive dashboard
    python run.py demo               # Run quick live comparison (Retrain vs CASR)
    python run.py experiment <name>  # Run experiment: casr, evaluation, adaptation, initial_training
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unittest


def run_tests() -> int:
    """Run full test suite via unittest."""
    print("Running project test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def run_dashboard() -> int:
    """Launch Streamlit web dashboard."""
    print("Launching Streamlit dashboard at http://localhost:8501 ...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return 0


def run_experiment(name: str) -> int:
    """Run a specific experiment module."""
    valid = ["casr", "evaluation", "adaptation", "initial_training", "retraining"]
    if name not in valid:
        print(f"Error: Unknown experiment '{name}'. Choose from: {', '.join(valid)}")
        return 1
    print(f"Executing experiments.{name} ...")
    cmd = [sys.executable, "-m", f"experiments.{name}"]
    return subprocess.call(cmd)


def run_demo() -> int:
    """Run a fast, self-contained comparative demo comparing scratch vs warm vs CASR."""
    print("=" * 70)
    print("LIVE DEMO: Goal-Conditioned RL Adaptation under Obstacle Shift")
    print("=" * 70)
    from environment.dynamic_gridworld import DynamicGridWorld
    from algorithms.goal_conditioned import GoalConditionedQLearner
    from algorithms.knowledge_reuse import collect_transitions, transfer_q_values
    from algorithms.proposed_method import casr_adapt
    from experiments.adaptation import (
        BARRIER,
        GOALS,
        POST_EPISODES,
        PRE_EPISODES,
        SEED,
        WALL,
        final_success,
        time_to_threshold,
    )

    env = DynamicGridWorld(width=8, height=8, obstacles=WALL, goals=GOALS, seed=SEED)
    cfg = dict(alpha=0.2, gamma=0.95, epsilon_decay=0.997, epsilon_min=0.05)

    print(f"[1/4] Pre-training agent on original layout ({PRE_EPISODES} episodes)...")
    pretrained = GoalConditionedQLearner(8, 8, GOALS, seed=SEED, **cfg)
    pretrained.train(env, episodes=PRE_EPISODES)
    print("      Pre-training complete.")

    print("[2/4] Collecting offline transition buffer (200 episodes)...")
    buffer = collect_transitions(env, pretrained, episodes=200, seed=SEED)

    print("[3/4] Applying environmental barrier change...")
    changed = env.apply_environment_change(add_obstacles=BARRIER)
    print(f"      Added barrier obstacles: {BARRIER}")

    print("[4/4] Running Change-Aware Selective Retention (CASR) adaptation...")
    learner = GoalConditionedQLearner(8, 8, GOALS, seed=SEED, **cfg)
    learner.Q = transfer_q_values(pretrained.Q)
    report = casr_adapt(
        env,
        learner,
        buffer=buffer,
        probe_episodes=30,
        margin=1,
        replay_per_episode=50,
        adapt_episodes=POST_EPISODES,
        probe_seed=SEED,
    )

    n_adapt = time_to_threshold(report["history"])
    n_total = (report["probe_episodes"] + n_adapt) if n_adapt is not None else "N/A"
    success_rate = final_success(report["history"])

    print("-" * 70)
    print(f"Result: Detected changed cells: {report['n_reset_states']}")
    print(f"        Success rate after adaptation: {success_rate * 100:.1f}%")
    print(f"        Episodes to reach 90% moving success: {n_total}")
    print("-" * 70)
    print("Demo completed successfully! For interactive visualization, run: python run.py dashboard")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI Runner for Goal-Conditioned RL Adaptation Research Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("test", help="Run the test suite (50 unit tests)")
    subparsers.add_parser("dashboard", help="Start the Streamlit dashboard")
    subparsers.add_parser("demo", help="Run a quick console demo of CASR adaptation")

    exp_parser = subparsers.add_parser("experiment", help="Run a benchmark experiment")
    exp_parser.add_argument(
        "name",
        choices=["casr", "evaluation", "adaptation", "initial_training", "retraining"],
        help="Experiment module name",
    )

    args = parser.parse_args()

    if args.command == "test":
        sys.exit(run_tests())
    elif args.command == "dashboard":
        sys.exit(run_dashboard())
    elif args.command == "demo":
        sys.exit(run_demo())
    elif args.command == "experiment":
        sys.exit(run_experiment(args.name))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
