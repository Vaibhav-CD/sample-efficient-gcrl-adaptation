"""Stage 4 maintained tests: transfer, replay buffer, selective updating.

Run from the project root with the system interpreter (stdlib only):

    python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from algorithms.goal_conditioned import GoalConditionedQLearner
from algorithms.knowledge_reuse import (
    ReplayBuffer,
    adapt_post_change,
    collect_transitions,
    replay_updates,
    selective_reset,
    transfer_q_values,
    transition_consistent,
)
from environment.dynamic_gridworld import DynamicGridWorld


def make_env():
    return DynamicGridWorld(width=5, height=5, goals=[(4, 4)], seed=0)


class TestTransfer(unittest.TestCase):
    def test_transfer_copies_values_not_references(self):
        src = np.ones((5, 5, 4))
        dst = transfer_q_values(src)
        np.testing.assert_array_equal(dst, src)
        dst[0, 0, 0] = 99.0
        self.assertEqual(src[0, 0, 0], 1.0)


class TestReplayBuffer(unittest.TestCase):
    def test_add_sample_lengths(self):
        buf = ReplayBuffer(capacity=4, seed=0)
        for i in range(6):
            buf.add((0, 0), 1, -0.01, (0, 1), False)
        self.assertEqual(len(buf), 4)  # FIFO cap respected
        self.assertEqual(len(buf.sample(2)), 2)
        self.assertEqual(len(buf.sample(99)), 4)  # capped at contents

    def test_collect_transitions_runs(self):
        env = make_env()
        learner = GoalConditionedQLearner(5, 5, [(4, 4)], epsilon=0.5, seed=0)
        buf = collect_transitions(env, learner, episodes=3, seed=0)
        self.assertGreater(len(buf), 0)
        s, a, r, sp, t, g = buf.sample(1)[0]
        self.assertEqual(g, (4, 4))

    def test_replay_updates_improve_q(self):
        env = make_env()
        learner = GoalConditionedQLearner(5, 5, [(4, 4)], epsilon=0.0, seed=0)
        buf = ReplayBuffer()
        buf.add((4, 3), 1, 1.0, (4, 4), True, (4, 4))  # step into goal
        n = replay_updates(learner, buf, 1)
        self.assertEqual(n, 1)
        gi = learner.goal_index[(4, 4)]
        self.assertGreater(learner.Q[4, 3, 1, gi], 0.0)


class TestSelectiveUpdating(unittest.TestCase):
    def test_reset_only_hits_neighbourhood(self):
        Q = np.ones((5, 5, 4, 1))
        n = selective_reset(Q, [(2, 2)], radius=1)
        self.assertEqual(n, 5)  # centre + 4 neighbours
        self.assertEqual(Q[2, 2, 0, 0], 0.0)
        self.assertEqual(Q[0, 0, 0, 0], 1.0)  # far field untouched

    def test_consistent_transition_accepted(self):
        env = make_env()
        self.assertTrue(transition_consistent(env, (0, 0), 1, (0, 1)))

    def test_transition_into_new_obstacle_rejected(self):
        env = make_env()
        env.apply_environment_change(add_obstacles=[(0, 1)])
        self.assertFalse(transition_consistent(env, (0, 0), 1, (0, 1)))

    def test_wrong_outcome_rejected_when_deterministic(self):
        env = make_env()  # slip = 0
        self.assertFalse(transition_consistent(env, (0, 0), 1, (1, 1)))


class TestAdaptationLoop(unittest.TestCase):
    def test_adapt_returns_full_history(self):
        env = make_env()
        learner = GoalConditionedQLearner(5, 5, [(4, 4)], seed=0)
        hist, applied = adapt_post_change(env, learner, episodes=5)
        self.assertEqual(len(hist["success"]), 5)
        self.assertEqual(applied, 0)

    def test_adapt_with_filtered_replay_runs(self):
        env = make_env()
        learner = GoalConditionedQLearner(5, 5, [(4, 4)], seed=0)
        buf = collect_transitions(env, learner, episodes=2, seed=0)
        hist, applied = adapt_post_change(
            env, learner, episodes=5, buffer=buf,
            replay_per_episode=10, consistency_filter=True)
        self.assertEqual(len(hist["success"]), 5)
        self.assertGreater(applied, 0)


if __name__ == "__main__":
    unittest.main()
