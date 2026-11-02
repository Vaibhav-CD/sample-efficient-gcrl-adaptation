"""Goal-conditioned tabular Q-learning with hindsight relabelling.

One table ``Q(s, a, g)`` of shape ``(height, width, 4, n_goals)`` shares
structure across goals (the UVFA idea of Schaul et al., 2015, in tabular
form). Each episode targets a randomly drawn goal; failed episodes are
additionally relabelled: every stored transition is re-scored as if the goal
had been a state actually visited later in that episode (the HER idea of
Andrychowicz et al., 2017, in tabular form), turning failures into training
signal for other goals.
"""

from __future__ import annotations

import numpy as np


class GoalConditionedQLearner:
    """Tabular ``Q(s, a, g)`` with optional hindsight replay."""

    def __init__(
        self,
        height: int,
        width: int,
        goals,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.997,
        epsilon_min: float = 0.05,
        her_relabels: int = 2,
        seed: int = 0,
    ):
        self.height = int(height)
        self.width = int(width)
        self.goals = [tuple(g) for g in goals]
        self.goal_index = {g: i for i, g in enumerate(self.goals)}
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.her_relabels = int(her_relabels)
        self.rng = np.random.default_rng(int(seed))
        self.Q = np.zeros((self.height, self.width, 4, len(self.goals)))

    def act(self, state, goal, explore: bool = True) -> int:
        gi = self.goal_index[tuple(goal)]
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))
        return int(np.argmax(self.Q[state[0], state[1], :, gi]))

    def update(self, state, action, reward, next_state, terminated, goal):
        gi = self.goal_index[tuple(goal)]
        q = self.Q[state[0], state[1], action, gi]
        target = reward
        if not terminated:
            target += self.gamma * np.max(self.Q[next_state[0], next_state[1], :, gi])
        self.Q[state[0], state[1], action, gi] += self.alpha * (target - q)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes: int, log_every: int = 100):
        """Train with per-episode random goals; returns history dict."""
        history = {"success": [], "steps": []}
        for _ in range(episodes):
            goal = self.goals[self.rng.integers(len(self.goals))]
            state, _ = env.reset(goal=goal)
            done = terminated = False
            trajectory = []
            while not done:
                action = self.act(state, goal, explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.update(state, action, reward, next_state, terminated, goal)
                trajectory.append((state, action, next_state))
                state = next_state
                done = terminated or truncated
            self._hindsight_replay(env, trajectory)
            history["success"].append(1 if terminated else 0)
            history["steps"].append(env.steps)
            self.decay_exploration()
        history["log_every"] = log_every
        return history

    def _hindsight_replay(self, env, trajectory):
        """Re-score the episode's transitions for goals actually achieved."""
        if not trajectory or self.her_relabels <= 0:
            return
        visited = [s for s, _, _ in trajectory] + [trajectory[-1][2]]
        for _ in range(self.her_relabels):
            hindsight = visited[self.rng.integers(len(visited))]
            if hindsight not in self.goal_index:
                continue
            for state, action, next_state in trajectory:
                reached = next_state == hindsight
                reward = env.goal_reward if reached else env.step_penalty
                self.update(state, action, reward, next_state, reached, hindsight)

    def evaluate(self, env, episodes: int = 50, seed: int = 0):
        """Greedy rollouts over random goals; ``(success_rate, mean_len)``."""
        wins, lengths = 0, []
        for i in range(episodes):
            goal = self.goals[(seed + i) % len(self.goals)]
            state, _ = env.reset(seed=seed + i, goal=goal)
            done = terminated = False
            while not done:
                action = self.act(state, goal, explore=False)
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            wins += 1 if terminated else 0
            if terminated:
                lengths.append(env.steps)
        mean_len = float(np.mean(lengths)) if lengths else float("nan")
        return wins / episodes, mean_len
