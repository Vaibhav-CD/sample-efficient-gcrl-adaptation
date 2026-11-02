"""Stage 5 maintained tests: CASR probing, detection and adaptation.

Run from the project root with the system interpreter (stdlib only):

    python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from algorithms.proposed_method import (
    casr_adapt,
    detect_region,
    run_probes,
    score_detection,
)
from algorithms.q_learning import TabularQLearner
from environment.dynamic_gridworld import DynamicGridWorld


def bump_trap_env():
    """3x3 env where greedy Q drives the agent into a new obstacle."""
    env = DynamicGridWorld(width=3, height=3, goals=[(0, 2)], start=(0, 0),
                           max_steps=50, seed=0)
    env.reset()
    return env


class TestProbes(unittest.TestCase):
    def test_probes_leave_learner_untouched(self):
        env = bump_trap_env()
        learner = TabularQLearner(3, 3, epsilon=0.0, seed=0)
        before = learner.Q.copy()
        run_probes(env, learner, n_episodes=3, seed=0)
        np.testing.assert_array_equal(learner.Q, before)
        self.assertEqual(learner.epsilon, 0.0)

    def test_probes_infer_new_obstacle_from_bumps(self):
        env = bump_trap_env()
        env.apply_environment_change(add_obstacles=[(0, 1)])
        learner = TabularQLearner(3, 3, epsilon=0.0, seed=0)
        learner.Q[0, 0, 1] = 5.0  # RIGHT looks attractive but is now blocked
        probe = run_probes(env, learner, n_episodes=5, seed=0)
        self.assertIn((0, 1), probe["inferred_blocked"])
        self.assertIn((0, 0), probe["bump_states"])
        self.assertGreater(probe["probe_steps"], 0)


class TestDetection(unittest.TestCase):
    def test_threshold_and_margin(self):
        probe = {"surprises": {(0, 0): 2.0, (2, 2): 0.1, (2, 1): 0.1},
                 "bump_states": {(0, 0)}}
        out = detect_region(probe, 3, 3, margin=1, std_factor=1.0)
        self.assertIn((0, 0), out["seeds"])
        self.assertNotIn((2, 2), out["seeds"])
        self.assertIn((1, 0), out["region"])  # margin growth
        self.assertNotIn((2, 2), out["region"])

    def test_empty_probes_empty_region(self):
        out = detect_region({"surprises": {}, "bump_states": set()}, 3, 3)
        self.assertEqual(out["region"], set())

    def test_score_detection_math(self):
        score = score_detection({(0, 0), (9, 9)}, [(0, 0), (0, 1)])
        self.assertAlmostEqual(score["precision"], 0.5)
        self.assertAlmostEqual(score["recall"], 0.5)


class TestCasrEndToEnd(unittest.TestCase):
    def test_casr_report_structure_and_budget(self):
        env = DynamicGridWorld(width=4, height=4, goals=[(3, 3)], seed=0)
        learner = TabularQLearner(4, 4, seed=0)
        learner.train(env, episodes=50, start=(0, 0), goal=(3, 3))
        env.apply_environment_change(add_obstacles=[(1, 1)])
        report = casr_adapt(env, learner, buffer=None, probe_episodes=4,
                            margin=1, adapt_episodes=10)
        self.assertEqual(len(report["history"]["success"]), 10)
        self.assertEqual(report["probe_episodes"], 4)
        for cell in report["region"]:
            self.assertTrue(0 <= cell[0] < 4 and 0 <= cell[1] < 4)


if __name__ == "__main__":
    unittest.main()
