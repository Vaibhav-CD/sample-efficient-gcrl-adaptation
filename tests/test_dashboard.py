"""Stage 7 tests: dashboard helpers use real data and render figures.

Run from the project root with the system interpreter (stdlib only):

    python -m unittest discover -s tests -v
"""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.dashboard import (
    before_after_demo,
    draw_grid,
    greedy_route,
    load_results,
    summarize,
    train_demo,
)


class TestDashboard(unittest.TestCase):
    def test_load_results_has_real_rows(self):
        df = load_results()
        self.assertGreater(len(df), 100)
        for col in ("exp", "method", "seed", "N_0.9", "success"):
            self.assertIn(col, df.columns)

    def test_summarize_experiment_b(self):
        summ = summarize(load_results(), "B")
        self.assertIn("CASR", set(summ["method"]))
        row = summ[summ["method"] == "CASR"].iloc[0]
        self.assertGreater(row["N_mean"], 0)

    def test_demo_trains_and_routes(self):
        env, bfs_len, success, qmax, learner = train_demo(episodes=200)
        self.assertGreater(bfs_len, 0)
        self.assertGreaterEqual(success, 0.0)
        route, reached = greedy_route(env, learner, (5, 5), (0, 0))
        self.assertGreater(len(route), 1)
        self.assertIsInstance(reached, bool)

    def test_draw_grid_returns_axes(self):
        before, old_route, after, _ = before_after_demo()
        before.reset(start=(0, 0), goal=(7, 7))
        _, ax = plt.subplots()
        self.assertIsNotNone(draw_grid(before, path=old_route, ax=ax))
        self.assertGreater(len(old_route), 10)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
