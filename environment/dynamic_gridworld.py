"""Dynamic grid-world: applies environmental changes after training.

Covers the Stage 1 change types: new/blocked obstacles, changed transition
probabilities (slip), unseen goals, and moving (dynamic) obstacles. Records
the set of cells whose local dynamics changed so Stage 4/5 can target
selective updates at exactly those regions.
"""

from __future__ import annotations

from environment.gridworld import GridWorld


class DynamicGridWorld(GridWorld):
    """GridWorld plus post-training changes and cyclic dynamic obstacles."""

    def __init__(self, *args, dynamic_obstacles: dict = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_cycles = {
            str(k): [self._checked_cell(c, f"dynamic obstacle {k}") for c in cycle]
            for k, cycle in dict(dynamic_obstacles).items()
        }
        for cycle in self.dynamic_cycles.values():
            if not cycle:
                raise ValueError("dynamic obstacle cycles must be non-empty")
        self._dynamic_index = {k: 0 for k in self.dynamic_cycles}
        self.last_changed_cells = frozenset()

    @property
    def dynamic_positions(self) -> frozenset:
        """Current cells occupied by moving obstacles."""
        return frozenset(
            cycle[self._dynamic_index[k]]
            for k, cycle in self.dynamic_cycles.items()
        )

    def is_free(self, cell) -> bool:
        return super().is_free(cell) and tuple(cell) not in self.dynamic_positions

    def reset(self, *, seed=None, start=None, goal=None):
        self._dynamic_index = {k: 0 for k in self.dynamic_cycles}
        return super().reset(seed=seed, start=start, goal=goal)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.dynamic_cycles and not (terminated or truncated):
            self._advance_dynamic()
            if tuple(self.agent_pos) in self.dynamic_positions:
                info["hit_by_dynamic"] = True
        info["dynamic_positions"] = sorted(self.dynamic_positions)
        return obs, reward, terminated, truncated, info

    def _advance_dynamic(self):
        for k, cycle in self.dynamic_cycles.items():
            self._dynamic_index[k] = (self._dynamic_index[k] + 1) % len(cycle)

    def apply_environment_change(
        self, *, add_obstacles=(), remove_obstacles=(), new_slip=None, new_goals=None
    ) -> frozenset:
        """Modify the environment in place; returns changed cells.

        Changed cells = added/removed obstacle cells plus their 4-neighbours
        (whose transitions are affected), so selective methods know the
        potentially stale region. Slip and goal changes are global, so they
        conservatively mark every free cell as changed.
        """
        from environment.gridworld import ACTION_DELTAS

        changed = set()
        for cell in add_obstacles:
            cell = self._checked_cell(cell, "new obstacle")
            if cell not in self.obstacles:
                self.obstacles = self.obstacles | {cell}
                changed.add(cell)
        for cell in remove_obstacles:
            cell = self._checked_cell(cell, "removed obstacle")
            if cell in self.obstacles:
                self.obstacles = self.obstacles - {cell}
                changed.add(cell)
        # Neighbours of edited cells see changed transitions too.
        for (r, c) in list(changed):
            for dr, dc in ACTION_DELTAS.values():
                neighbour = (r + dr, c + dc)
                if 0 <= neighbour[0] < self.height and 0 <= neighbour[1] < self.width:
                    changed.add(neighbour)
        if new_slip is not None:
            if not 0.0 <= new_slip <= 1.0:
                raise ValueError("slip must be in [0, 1]")
            self.slip = float(new_slip)
            changed.update(self.free_cells())
        if new_goals is not None:
            self.goals = [self._checked_cell(g, "goal") for g in new_goals]
            changed.update(self.free_cells())
        self.last_changed_cells = frozenset(changed)
        return self.last_changed_cells
