"""Configurable tabular grid-world for goal-conditioned RL.

State, action and transition definitions (also documented in
docs/stage2_environment.md):

- State: ``(row, col)`` cell of the agent on a ``height x width`` grid.
- Actions: ``0=up, 1=right, 2=down, 3=left`` (``NUM_ACTIONS = 4``).
- Transitions: the intended neighbour cell is entered with probability
  ``1 - slip``; with probability ``slip`` one of the other three neighbours
  is entered instead (uniform). Moves into walls or obstacles leave the
  agent in place. ``slip = 0`` is fully deterministic.
- Reward: ``goal_reward`` on entering the current goal cell, ``step_penalty``
  otherwise. Reaching the goal terminates the episode; exceeding
  ``max_steps`` truncates it.
"""

from __future__ import annotations

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
NUM_ACTIONS = 4
ACTION_DELTAS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}


class GridWorld:
    """Deterministic/stochastic 2D grid-world with obstacles and goals."""

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        obstacles=(),
        goals=(),
        start=None,
        slip: float = 0.0,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        max_steps: int = 200,
        seed: int = 0,
    ):
        if width < 2 or height < 2:
            raise ValueError("grid dimensions must be >= 2")
        if not 0.0 <= slip <= 1.0:
            raise ValueError("slip must be in [0, 1]")
        self.width = int(width)
        self.height = int(height)
        self.obstacles = frozenset((int(r), int(c)) for r, c in obstacles)
        for cell in self.obstacles:
            self._check_inside(cell, "obstacle")
        self.goals = [self._checked_cell(g, "goal") for g in goals]
        self.default_start = (
            None if start is None else self._checked_cell(start, "start")
        )
        if self.default_start is not None and self.default_start in self.obstacles:
            raise ValueError("start cell must not be an obstacle")
        for goal in self.goals:
            if goal in self.obstacles:
                raise ValueError("goal cell must not be an obstacle")
        self.slip = float(slip)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.agent_pos = None
        self.current_goal = None
        self.steps = 0

    # -- helpers ------------------------------------------------------
    def _check_inside(self, cell, name):
        r, c = int(cell[0]), int(cell[1])
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError(f"{name} cell {(r, c)} outside grid")
        return (r, c)

    def _checked_cell(self, cell, name):
        try:
            r, c = int(cell[0]), int(cell[1])
        except (TypeError, IndexError, ValueError):
            raise ValueError(f"{name} cell {cell!r} is not a (row, col) pair")
        return self._check_inside((r, c), name)

    def is_free(self, cell) -> bool:
        """True if the cell is inside the grid and not an obstacle."""
        r, c = cell
        return (
            0 <= r < self.height
            and 0 <= c < self.width
            and (r, c) not in self.obstacles
        )

    def free_cells(self, exclude_goals: bool = False):
        """All obstacle-free cells, optionally excluding goal cells."""
        goals = set(self.goals) if exclude_goals else set()
        return [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.obstacles and (r, c) not in goals
        ]

    # -- core API -----------------------------------------------------
    def reset(self, *, seed=None, start=None, goal=None):
        """Start a new episode. Returns ``(agent_pos, current_goal)``."""
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        start = self.default_start if start is None else self._checked_cell(start, "start")
        if start is None:
            candidates = self.free_cells()
            start = candidates[self.rng.integers(len(candidates))]
        if start in self.obstacles:
            raise ValueError(f"start cell {start} is blocked by an obstacle")
        if goal is None:
            if not self.goals:
                raise ValueError("no goals configured and none given to reset()")
            goal = self.goals[self.rng.integers(len(self.goals))]
        else:
            goal = self._checked_cell(goal, "goal")
        self.agent_pos = start
        self.current_goal = goal
        self.steps = 0
        return self.agent_pos, self.current_goal

    def step(self, action):
        """Apply one action. Returns ``(obs, reward, terminated, truncated, info)``."""
        if self.agent_pos is None:
            raise RuntimeError("call reset() before step()")
        if action not in ACTION_DELTAS:
            raise ValueError(
                f"invalid action {action!r}: must be 0=up, 1=right, 2=down, 3=left"
            )
        executed = self._sample_action(action)
        dr, dc = ACTION_DELTAS[executed]
        candidate = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        bumped = False
        if self.is_free(candidate):
            self.agent_pos = candidate
        else:
            bumped = True
        self.steps += 1
        terminated = self.agent_pos == self.current_goal
        reward = self.goal_reward if terminated else self.step_penalty
        truncated = not terminated and self.steps >= self.max_steps
        info = {"executed_action": executed, "bumped": bumped, "steps": self.steps}
        return self.agent_pos, reward, terminated, truncated, info

    def _sample_action(self, action: int) -> int:
        if self.slip <= 0.0 or self.rng.random() >= self.slip:
            return action
        others = [a for a in ACTION_DELTAS if a != action]
        return others[self.rng.integers(len(others))]

    # -- rendering ----------------------------------------------------
    def render_ascii(self) -> str:
        """Human-readable grid: A=agent, G=goal(s), *=both, #=obstacle."""
        rows = []
        for r in range(self.height):
            line = []
            for c in range(self.width):
                cell = (r, c)
                if cell in self.obstacles:
                    line.append("#")
                elif cell == self.agent_pos and cell == self.current_goal:
                    line.append("*")
                elif cell == self.agent_pos:
                    line.append("A")
                elif cell == self.current_goal:
                    line.append("G")
                elif cell in self.goals:
                    line.append("g")
                else:
                    line.append(".")
            rows.append("".join(line))
        return "\n".join(rows)
