"""Stage 5 study: CASR on the identical Stage-4 protocol.

Pre-training, buffer collection, change and budgets match
experiments.adaptation exactly (constants imported, not copied), so the
CASR row below is directly comparable to the Stage-4 table. Probe episodes
count toward the interaction budget: N_total = probe + adapt episodes.

Run from the project root:

    python -m experiments.casr
"""

from __future__ import annotations

import numpy as np

from algorithms.goal_conditioned import GoalConditionedQLearner
from algorithms.knowledge_reuse import collect_transitions, transfer_q_values
from algorithms.proposed_method import casr_adapt, score_detection
from environment.dynamic_gridworld import DynamicGridWorld
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


def main():
    env = DynamicGridWorld(width=8, height=8, obstacles=WALL, goals=GOALS, seed=SEED)
    cfg = dict(alpha=0.2, gamma=0.95, epsilon_decay=0.997, epsilon_min=0.05)
    pretrained = GoalConditionedQLearner(8, 8, GOALS, seed=SEED, **cfg)
    pretrained.train(env, episodes=PRE_EPISODES)
    buffer = collect_transitions(env, pretrained, episodes=200, seed=SEED)
    changed = env.apply_environment_change(add_obstacles=BARRIER)

    learner = GoalConditionedQLearner(8, 8, GOALS, seed=SEED, **cfg)
    learner.Q = transfer_q_values(pretrained.Q)
    report = casr_adapt(env, learner, buffer=buffer, probe_episodes=30,
                        margin=1, replay_per_episode=50,
                        adapt_episodes=POST_EPISODES, probe_seed=SEED)
    hist = report["history"]
    n_adapt = time_to_threshold(hist)
    n_total = (report["probe_episodes"] + n_adapt) if n_adapt is not None else None
    detection_score = score_detection(report["region"], changed)
    print(f"CASR: seeds={report['n_seeds']} reset_states={report['n_reset_states']} "
          f"inferred_blocked={report['inferred_blocked']}")
    print(f"CASR detection vs oracle changed cells: {detection_score}")
    print(f"CASR: N_adapt={n_adapt} probe={report['probe_episodes']} "
          f"N_total={n_total} final={final_success(hist):.2f} "
          f"probe_steps={report['probe_steps']} replay_applied={report['replay_applied']}")


if __name__ == "__main__":
    main()
