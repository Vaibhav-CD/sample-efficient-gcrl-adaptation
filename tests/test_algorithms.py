"""Stage 3 maintained tests: planners and tabular learners.

Run from the project root with the system interpreter (stdlib only):

    python -m unittest discover -s tests -v
"""

import unittest

from algorithms.bfs import bfs
from algorithms.dijkstra import dijkstra
from algorithms.goal_conditioned import GoalConditionedQLearner
from algorithms.q_learning import TabularQLearner
from environment.gridworld import DOWN, RIGHT, GridWorld
from experiments.retraining import retrain_from_scratch


class TestPlanners(unittest.TestCase):
    def test_bfs_manhattan_length_on_empty_grid(self):
        path = bfs(5, 5, [], (0, 0), (4, 4))
        self.assertEqual(len(path) - 1, 8)  # |dx| + |dy|
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))

    def test_bfs_avoids_obstacles(self):
        obstacles = [(1, 0), (1, 1), (1, 2)]
        path = bfs(4, 4, obstacles, (0, 0), (3, 0))
        self.assertIsNotNone(path)
        self.assertTrue(all(cell not in obstacles for cell in path))

    def test_bfs_unreachable_returns_none(self):
        obstacles = [(1, c) for c in range(3)]  # full wall on 3x3
        self.assertIsNone(bfs(3, 3, obstacles, (0, 0), (2, 0)))

    def test_bfs_start_is_goal(self):
        self.assertEqual(bfs(3, 3, [], (1, 1), (1, 1)), [(1, 1)])

    def test_dijkstra_uniform_matches_bfs_length(self):
        obstacles = [(1, 1)]
        path_bfs = bfs(4, 4, obstacles, (0, 0), (3, 3))
        path_d, cost = dijkstra(4, 4, obstacles, (0, 0), (3, 3))
        self.assertEqual(cost, len(path_bfs) - 1)

    def test_dijkstra_avoids_penalty_cell(self):
        # Crossing (0,1) costs 10; the detour via row 1 costs 4.
        cost_fn = lambda _f, t: 10.0 if t == (0, 1) else 1.0
        path, cost = dijkstra(3, 3, [], (0, 0), (0, 2), cost_fn)
        self.assertNotIn((0, 1), path)
        self.assertAlmostEqual(cost, 4.0)

    def test_dijkstra_unreachable(self):
        obstacles = [(1, c) for c in range(3)]
        path, cost = dijkstra(3, 3, obstacles, (0, 0), (2, 0))
        self.assertIsNone(path)
        self.assertEqual(cost, float("inf"))


class TestTabularQ(unittest.TestCase):
    def test_q_update_moves_toward_target(self):
        learner = TabularQLearner(3, 3, alpha=0.5, gamma=0.9, epsilon=0.0)
        learner.update((0, 0), RIGHT, 1.0, (0, 1), True)
        self.assertAlmostEqual(learner.Q[0, 0, RIGHT], 0.5)

    def test_q_learns_deterministic_route(self):
        env = GridWorld(width=4, height=4, goals=[(3, 3)], seed=0)
        learner = TabularQLearner(4, 4, seed=0)
        learner.train(env, episodes=800, start=(0, 0), goal=(3, 3))
        success, mean_len = learner.evaluate(
            env, episodes=20, start=(0, 0), goal=(3, 3)
        )
        self.assertGreaterEqual(success, 0.9)
        self.assertLessEqual(mean_len, 8)  # optimal is 6

    def test_q_greedy_action_is_argmax(self):
        learner = TabularQLearner(2, 2, epsilon=0.0)
        learner.Q[0, 0, DOWN] = 5.0
        self.assertEqual(learner.act((0, 0), explore=False), DOWN)


class TestGoalConditionedQ(unittest.TestCase):
    def test_hindsight_update_writes_relabelled_goal(self):
        env = GridWorld(width=3, height=3, goals=[(2, 2), (0, 2)], seed=0)
        learner = GoalConditionedQLearner(3, 3, [(2, 2), (0, 2)], epsilon=0.0)
        traj = [((0, 0), RIGHT, (0, 1)), ((0, 1), RIGHT, (0, 2))]
        before = learner.Q.copy()
        learner._hindsight_replay(env, traj)
        # Goal (0,2) was achieved, so its slice must have changed.
        gi = learner.goal_index[(0, 2)]
        self.assertFalse((learner.Q[:, :, :, gi] == before[:, :, :, gi]).all())

    def test_gcq_learns_two_goals(self):
        goals = [(3, 3), (0, 3)]
        env = GridWorld(width=4, height=4, goals=goals, seed=1)
        learner = GoalConditionedQLearner(4, 4, goals, seed=1)
        learner.train(env, episodes=2500)
        success, _ = learner.evaluate(env, episodes=40)
        self.assertGreaterEqual(success, 0.8)


class TestRetrainingBaseline(unittest.TestCase):
    def test_retrain_returns_fresh_trained_learner(self):
        env = GridWorld(width=4, height=4, goals=[(3, 3)], seed=2)
        learner, history = retrain_from_scratch(
            lambda: TabularQLearner(4, 4, seed=2),
            env,
            episodes=800,
            start=(0, 0),
            goal=(3, 3),
        )
        self.assertEqual(len(history["success"]), 800)
        success, _ = learner.evaluate(env, episodes=10, start=(0, 0), goal=(3, 3))
        self.assertGreaterEqual(success, 0.9)


if __name__ == "__main__":
    unittest.main()
