"""Shared experiment configuration (Stage 2: environment defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Serializable defaults for grid-world construction."""

    width: int = 8
    height: int = 8
    obstacles: tuple = ()
    goals: tuple = ((7, 7),)
    start: tuple | None = None
    slip: float = 0.0
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    max_steps: int = 200
    seed: int = 0
    dynamic_obstacles: dict = field(default_factory=dict)

    def to_kwargs(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "obstacles": self.obstacles,
            "goals": self.goals,
            "start": self.start,
            "slip": self.slip,
            "step_penalty": self.step_penalty,
            "goal_reward": self.goal_reward,
            "max_steps": self.max_steps,
            "seed": self.seed,
        }
