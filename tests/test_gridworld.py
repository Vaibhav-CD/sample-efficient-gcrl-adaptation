"""Stage 2 maintained tests: grid-world movement, seeds, changes.

Run from the project root with the system interpreter (stdlib only):

    python -m unittest discover -s tests -v
"""

import unittest

from environment.dynamic_gridworld import DynamicGridWorld
from environment.gridworld import DOWN, LEFT, RIGHT, UP, GridWorld
from visualization.grid_visualizer import render_rgb


def make_env(**kwargs):
    params = dict(width=5, height=5, goals=[(4, 4)], seed=1)
    params.update(kwargs)
    return GridWorld(**params)


class TestMovement(unittest.TestCase):
    def test_valid_move_changes_position(self):
        env = make_env()
        env.reset(start=(2, 2), goal=(4, 4))
        obs, _, terminated, _, _ = env.step(RIGHT)
        self.assertEqual(obs, (2, 3))
        self.assertFalse(terminated)

    def test_all_four_actions(self):
        env = make_env()
        for action, expected in (
            (UP, (1, 2)),
            (RIGHT, (2, 3)),
            (DOWN, (3, 2)),
            (LEFT, (2, 1)),
        ):
            env.reset(start=(2, 2), goal=(4, 4))
            obs, _, _, _, _ = env.step(action)
            self.assertEqual(obs, expected)

    def test_wall_bump_stays_in_place(self):
        env = make_env()
        env.reset(start=(0, 0), goal=(4, 4))
        obs, reward, _, _, info = env.step(UP)
        self.assertEqual(obs, (0, 0))
        self.assertTrue(info["bumped"])
        self.assertAlmostEqual(reward, env.step_penalty)

    def test_obstacle_bump_stays_in_place(self):
        env = make_env(obstacles=[(1, 1)])
        env.reset(start=(1, 0), goal=(4, 4))
        obs, _, _, _, info = env.step(RIGHT)
        self.assertEqual(obs, (1, 0))
        self.assertTrue(info["bumped"])

    def test_invalid_action_raises(self):
        env = make_env()
        env.reset(start=(2, 2), goal=(4, 4))
        for bad in (-1, 4, "up", None):
            with self.assertRaises(ValueError, msg=f"action={bad!r}"):
                env.step(bad)

    def test_step_before_reset_raises(self):
        env = make_env()
        with self.assertRaises(RuntimeError):
            env.step(UP)

    def test_goal_reaching_terminates_with_reward(self):
        env = make_env()
        env.reset(start=(4, 3), goal=(4, 4))
        obs, reward, terminated, _, _ = env.step(RIGHT)
        self.assertEqual(obs, (4, 4))
        self.assertTrue(terminated)
        self.assertAlmostEqual(reward, env.goal_reward)

    def test_max_steps_truncates(self):
        env = make_env(max_steps=3)
        env.reset(start=(0, 0), goal=(4, 4))
        truncated = False
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(DOWN)
            self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_start_on_obstacle_raises(self):
        env = make_env(obstacles=[(0, 0)])
        with self.assertRaises(ValueError):
            env.reset(start=(0, 0), goal=(4, 4))


class TestSeeds(unittest.TestCase):
    def test_same_seed_same_random_start(self):
        starts = []
        for _ in range(2):
            env = make_env()
            pos, _ = env.reset(seed=42)
            starts.append(pos)
        self.assertEqual(starts[0], starts[1])

    def test_same_seed_same_slip_sequence(self):
        seqs = []
        for _ in range(2):
            env = make_env(slip=0.5)
            env.reset(seed=7, start=(2, 2), goal=(4, 4))
            seqs.append(
                [env.step(RIGHT)[4]["executed_action"] for _ in range(20)]
            )
        self.assertEqual(seqs[0], seqs[1])

    def test_deterministic_env_replays_exactly(self):
        trajs = []
        for _ in range(2):
            env = make_env()
            env.reset(start=(0, 0), goal=(4, 4))
            trajs.append([env.step(a)[0] for a in (RIGHT, RIGHT, DOWN, DOWN)])
        self.assertEqual(trajs[0], trajs[1])


class TestDynamicChanges(unittest.TestCase):
    def test_added_obstacle_blocks_path(self):
        env = DynamicGridWorld(width=5, height=5, goals=[(0, 4)], seed=0)
        env.reset(start=(0, 0), goal=(0, 4))
        self.assertEqual(env.step(RIGHT)[0], (0, 1))
        env.reset(start=(0, 0), goal=(0, 4))
        changed = env.apply_environment_change(add_obstacles=[(0, 1)])
        self.assertIn((0, 1), changed)
        self.assertIn((0, 0), changed)  # affected neighbour
        self.assertEqual(env.step(RIGHT)[0], (0, 0))  # now blocked

    def test_slip_change_marks_global_staleness(self):
        env = DynamicGridWorld(width=4, height=4, goals=[(3, 3)], seed=0)
        changed = env.apply_environment_change(new_slip=0.2)
        self.assertEqual(env.slip, 0.2)
        self.assertTrue(len(changed) > 0)

    def test_dynamic_obstacle_moves_each_step(self):
        env = DynamicGridWorld(
            width=4,
            height=4,
            goals=[(3, 3)],
            seed=0,
            dynamic_obstacles={"car": [(1, 1), (1, 2)]},
        )
        env.reset(start=(0, 0), goal=(3, 3))
        first = set(env.dynamic_positions)
        env.step(RIGHT)
        second = set(env.dynamic_positions)
        self.assertNotEqual(first, second)


class TestVisualization(unittest.TestCase):
    def test_render_rgb_shape_and_agent_pixel(self):
        import numpy as np

        env = make_env()
        env.reset(start=(0, 0), goal=(4, 4))
        img = render_rgb(env, cell_px=8)
        self.assertEqual(img.shape, (5 * 8, 5 * 8, 3))
        agent_block = img[0:8, 0:8]
        self.assertTrue((agent_block == np.array([30, 144, 255])).all())


if __name__ == "__main__":
    unittest.main()
