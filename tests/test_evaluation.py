"""Stage 6 smoke test: the evaluation protocol runs end-to-end at tiny scale.

Run from the project root with the system interpreter (stdlib only):

    python -m unittest discover -s tests -v
"""

import unittest

from experiments.evaluation import METHODS, SEEDS, run_all


class TestEvaluationSmoke(unittest.TestCase):
    def test_protocol_completes_and_reports(self):
        df, curves = run_all(seeds=[0], pre_episodes=150, post_episodes=120)
        exps = set(df["exp"])
        self.assertEqual(exps, {"A", "B", "C", "D", "E"})
        self.assertEqual(set(df[df["exp"] == "B"]["method"]), set(METHODS))
        for col in ("N_0.9", "success", "mean_len", "median_len", "time"):
            self.assertIn(col, df.columns)
        self.assertTrue(((df["success"] >= 0) & (df["success"] <= 1)).all())
        self.assertEqual(set(curves), {"B", "C", "D", "E"})
        self.assertEqual(len(curves["B"]["CASR"]), 1)
        self.assertGreater(len(SEEDS), 1)  # real runs stay multi-seed


if __name__ == "__main__":
    unittest.main()
