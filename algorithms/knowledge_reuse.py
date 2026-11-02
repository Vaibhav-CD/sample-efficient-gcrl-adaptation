"""Stage 4 mechanisms: transfer (A), replay (B), selective updating (C).

Each mechanism is independent and combinable:

- A. Transfer: copy an already-trained Q-table into a fresh learner so
  adaptation starts from old knowledge instead of zeros (warm start).
- B. Replay: keep transitions collected in the original environment and keep
  applying Q-updates to them after the change (extra training signal per
  interaction).
- C. Selective updating: use the environment's reported changed cells to
  (i) reset Q-values within a Manhattan radius of the change (drop stale
  values, keep far-field knowledge) and (ii) filter replayed transitions to
  those still consistent with the new dynamics.

All helpers are generic over Q-tables shaped ``(H, W, ...)`` (single-goal
``(H, W, 4)`` or goal-conditioned ``(H, W, 4, G)``).
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """FIFO store of ``(state, action, reward, next_state, terminated, goal)``."""

    def __init__(self, capacity: int = 20000, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self._data = []

    def add(self, state, action, reward, next_state, terminated, goal=None):
        if len(self._data) >= self.capacity:
            self._data.pop(0)
        self._data.append(
            (tuple(state), int(action), float(reward), tuple(next_state),
             bool(terminated), None if goal is None else tuple(goal))
        )

    def sample(self, n: int):
        n = min(int(n), len(self._data))
        idx = self.rng.choice(len(self._data), size=n, replace=False)
        return [self._data[i] for i in idx]

    def __len__(self):
        return len(self._data)


def transfer_q_values(source_Q: np.ndarray) -> np.ndarray:
    """Mechanism A: copy trained values as the adaptation starting point."""
    return np.array(source_Q, dtype=float, copy=True)


def selective_reset(Q: np.ndarray, changed_cells, radius: int = 2,
                    init_value: float = 0.0) -> int:
    """Mechanism C(i): reset Q for states near the change; returns count.

    Every state within Manhattan distance ``radius`` of any changed cell gets
    ``init_value`` across all actions (and goals). States outside keep their
    transferred knowledge. Operates in place on ``Q``.
    """
    height, width = Q.shape[0], Q.shape[1]
    changed = {(int(r), int(c)) for r, c in changed_cells}
    count = 0
    for r in range(height):
        for c in range(width):
            if any(abs(r - cr) + abs(c - cc) <= radius for cr, cc in changed):
                Q[r, c] = init_value
                count += 1
    return count


def expected_next_static(env, state, action):
    """Deterministic outcome of ``action`` under the current static map."""
    from environment.gridworld import ACTION_DELTAS

    dr, dc = ACTION_DELTAS[int(action)]
    candidate = (state[0] + dr, state[1] + dc)
    if (0 <= candidate[0] < env.height and 0 <= candidate[1] < env.width
            and candidate not in env.obstacles):
        return candidate
    return tuple(state)  # bump: agent stays


def transition_consistent(env, state, action, next_state) -> bool:
    """Mechanism C(ii): is this stored transition still possible?

    Checks the static map (bounds + obstacles). In a deterministic
    environment the recorded outcome must also equal the deterministic
    outcome; in a stochastic one any free outcome is plausible.
    """
    state, next_state = tuple(state), tuple(next_state)
    if state in env.obstacles or next_state in env.obstacles:
        return False
    if not (0 <= next_state[0] < env.height and 0 <= next_state[1] < env.width):
        return False
    if float(getattr(env, "slip", 0.0)) == 0.0:
        return expected_next_static(env, state, action) == next_state
    return True


def collect_transitions(env, learner, episodes: int, epsilon: float = 0.2,
                        seed: int = 0) -> ReplayBuffer:
    """Roll out a (trained) learner in ``env`` and store its transitions."""
    buf = ReplayBuffer(seed=seed)
    rng = np.random.default_rng(seed)
    goal_conditioned = hasattr(learner, "goals")
    old_eps = learner.epsilon
    learner.epsilon = float(epsilon)
    try:
        for _ in range(episodes):
            if goal_conditioned:
                goal = learner.goals[rng.integers(len(learner.goals))]
                state, _ = env.reset(goal=goal)
            else:
                goal = None
                state, _ = env.reset()
            done = False
            while not done:
                if goal_conditioned:
                    action = learner.act(state, goal, explore=True)
                else:
                    action = learner.act(state, explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                buf.add(state, action, reward, next_state, terminated, goal)
                state = next_state
                done = terminated or truncated
    finally:
        learner.epsilon = old_eps
    return buf


def replay_updates(learner, buffer: ReplayBuffer, n_updates: int, env=None,
                   consistency_filter: bool = False, keep=None) -> int:
    """Apply stored transitions as Q-updates; returns updates applied.

    ``keep`` is an optional caller-supplied predicate
    ``keep(state, action, next_state) -> bool`` (e.g. CASR's inferred-map
    filter); it applies in addition to ``consistency_filter``.
    """
    applied = 0
    for state, action, reward, next_state, terminated, goal in buffer.sample(n_updates):
        if consistency_filter:
            if env is None:
                raise ValueError("consistency filter needs the current env")
            if not transition_consistent(env, state, action, next_state):
                continue
        if keep is not None and not keep(state, action, next_state):
            continue
        if goal is None:
            learner.update(state, action, reward, next_state, terminated)
        else:
            learner.update(state, action, reward, next_state, terminated, goal)
        applied += 1
    return applied


def adapt_post_change(env, learner, episodes: int, buffer=None,
                      replay_per_episode: int = 0, consistency_filter: bool = False,
                      keep=None,
                      eps_start: float = 0.3, eps_decay: float = 0.997,
                      eps_min: float = 0.05):
    """Interact with the changed env; returns ``(history, replay_applied)``."""
    goal_conditioned = hasattr(learner, "goals")
    learner.epsilon = float(eps_start)
    history = {"success": [], "steps": []}
    replay_applied = 0
    for _ in range(episodes):
        if goal_conditioned:
            goal = learner.goals[learner.rng.integers(len(learner.goals))]
            state, _ = env.reset(goal=goal)
        else:
            goal = None
            state, _ = env.reset()
        done = terminated = False
        while not done:
            if goal_conditioned:
                action = learner.act(state, goal, explore=True)
            else:
                action = learner.act(state, explore=True)
            next_state, reward, t, truncated, _ = env.step(action)
            terminated = t
            if goal_conditioned:
                learner.update(state, action, reward, next_state, terminated, goal)
            else:
                learner.update(state, action, reward, next_state, terminated)
            state = next_state
            done = terminated or truncated
        if buffer is not None and replay_per_episode > 0:
            replay_applied += replay_updates(
                learner, buffer, replay_per_episode, env,
                consistency_filter=consistency_filter, keep=keep)
        history["success"].append(1 if terminated else 0)
        history["steps"].append(env.steps)
        learner.epsilon = max(eps_min, learner.epsilon * eps_decay)
    return history, replay_applied
