# STAGE 2 — Environment Development: definitions and verification

**Status:** complete, awaiting approval.

## 1. State, action and transition definitions

- **State space.** The state is the agent cell `(row, col)` on a
  `height × width` grid. Obstacle cells are excluded from the reachable set.
  The observation returned by `reset()`/`step()` is `(agent_pos, current_goal)`.
- **Action space.** Four discrete actions: `0=up, 1=right, 2=down, 3=left`.
  Invalid actions (e.g. `-1`, `4`, `"up"`) raise `ValueError`.
- **Transition probabilities.** For intended action `a` with slip `ε`:
  `P(intended neighbour | s, a) = 1 − ε`, and each of the other three
  neighbours with probability `ε/3`. Moves into walls, static obstacles or
  dynamic obstacles leave the agent in place (a "bump", flagged in `info`).
  `ε = 0` is fully deterministic.
- **Reward.** `goal_reward` (default +1.0) on entering the current goal,
  `step_penalty` (default −0.01) otherwise. Goal entry terminates; reaching
  `max_steps` truncates.
- **Goals.** Multiple goals are configured; each episode draws one (or takes
  an explicit `goal=`). Unseen goals for Stage 6/Experiment D are simply goal
  cells never drawn during initial training.
- **Dynamic obstacles.** Cyclic cell sequences advanced once per step
  (`dynamic_positions`); contact is reported in `info["hit_by_dynamic"]`.
- **Environment changes.** `apply_environment_change()` adds/removes
  obstacles, sets a new slip, or replaces goals, and returns the set of cells
  whose local dynamics changed (edited cells plus 4-neighbours; every free
  cell for global slip/goal changes). This powers Stage 4/5 selective updates.

## 2. Mathematics used here

The environment realises the MDP `(S, A, P, R, γ)` from Stage 1: this stage
fixes `S`, `A`, `P` (via slip), `R` (via goal reward/penalty); `γ` and the
policy live in the learning code (Stage 3+). The slip model is a
stochastic-transition kernel with a reproducibility guarantee: identical seeds
reproduce identical trajectories.

## 3. Files created

- `environment/gridworld.py` — `GridWorld` (movement, rewards, seeds, ASCII art).
- `environment/dynamic_gridworld.py` — `DynamicGridWorld` (changes, movers,
  changed-cell reporting).
- `utils/config.py` — `EnvConfig` defaults shared by later experiments.
- `utils/reproducibility.py` — `seed_all()` for one-seed control.
- `visualization/grid_visualizer.py` — `render_rgb()`/`show()` (NumPy + Matplotlib).
- `tests/test_gridworld.py` — 16 maintained tests (stdlib `unittest`).

## 4. Tests and results

`python -m unittest discover -s tests -v` — **16/16 pass**, covering valid
moves in all four directions, wall/obstacle bumps, invalid-action and
pre-reset errors, goal termination, max-step truncation, seed reproducibility
(random starts, slip sequences, exact replay), obstacle insertion blocking a
path, slip-change staleness marking, dynamic-obstacle motion, and render shape
plus agent-pixel colour. A manual smoke run also moved an agent 8 steps,
saved an RGB render, and applied a combined obstacle+slip change.

Note: tests use stdlib `unittest` (no install needed) instead of `pytest`,
which is absent from this machine; `requirements.txt` still pins `pytest`
for machines that have it, and the test file is pytest-compatible.
