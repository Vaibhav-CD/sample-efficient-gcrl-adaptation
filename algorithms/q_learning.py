"""Tabular Q-learning for a single fixed goal.

Update applied after every transition (Watkins & Dayan, 1992):

    Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

with ``Q(s',a') = 0`` when ``s'`` is terminal. Behaviour is epsilon-greedy:
with probability ``epsilon`` pick a random action (explore), otherwise the
current argmax (exploit); ``epsilon`` decays geometrically per episode.
"""

from __future__ import annotations

import numpy as np


class TabularQLearner:
    """Q-table of shape ``(height, width, 4)`` for one goal."""

    def __init__(
        self,
        height: int,
        width: int,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: int = 0,
    ):
        self.height = int(height)
        self.width = int(width)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.rng = np.random.default_rng(int(seed))
        self.Q = np.zeros((self.height, self.width, 4))

    def act(self, state, explore: bool = True) -> int:
        """Epsilon-greedy action for a ``(row, col)`` state."""
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))
        return int(np.argmax(self.Q[state[0], state[1]]))

    def update(self, state, action, reward, next_state, terminated: bool):
        """Single Bellman backup toward the sampled target."""
        q = self.Q[state[0], state[1], action]
        target = reward
        if not terminated:
            target += self.gamma * np.max(self.Q[next_state[0], next_state[1]])
        self.Q[state[0], state[1], action] += self.alpha * (target - q)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes: int, start=None, goal=None, log_every: int = 100):
        """Interact for ``episodes``; returns history dict of lists."""
        history = {"success": [], "steps": []}
        for _ in range(episodes):
            state, _ = env.reset(start=start, goal=goal)
            done = False
            while not done:
                action = self.act(state, explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.update(state, action, reward, next_state, terminated)
                state = next_state
                done = terminated or truncated
            history["success"].append(1 if terminated else 0)
            history["steps"].append(env.steps)
            self.decay_exploration()
        history["log_every"] = log_every
        return history

    def evaluate(self, env, episodes: int = 50, start=None, goal=None, seed: int = 0):
        """Greedy rollouts; returns ``(success_rate, mean_steps_of_successes)``."""
        wins, lengths = 0, []
        for i in range(episodes):
            state, _ = env.reset(seed=seed + i, start=start, goal=goal)
            done = terminated = False
            while not done:
                action = self.act(state, explore=False)
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            wins += 1 if terminated else 0
            if terminated:
                lengths.append(env.steps)
        mean_len = float(np.mean(lengths)) if lengths else float("nan")
        return wins / episodes, mean_len
