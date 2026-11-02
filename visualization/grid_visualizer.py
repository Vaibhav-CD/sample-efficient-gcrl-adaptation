"""Grid-world visualisation: NumPy RGB render + Matplotlib display."""

from __future__ import annotations

import numpy as np

# Cell colours (R, G, B) in 0..255.
COLOR_FREE = (255, 255, 255)
COLOR_OBSTACLE = (30, 30, 30)
COLOR_AGENT = (30, 144, 255)
COLOR_GOAL = (50, 205, 50)
COLOR_DYNAMIC = (255, 140, 0)


def render_rgb(env, cell_px: int = 24) -> np.ndarray:
    """Render the environment to an ``(H*px, W*px, 3)`` uint8 array."""
    img = np.zeros(
        (env.height * cell_px, env.width * cell_px, 3), dtype=np.uint8
    )
    dynamic = getattr(env, "dynamic_positions", frozenset()) or frozenset()
    for r in range(env.height):
        for c in range(env.width):
            cell = (r, c)
            if cell in env.obstacles:
                color = COLOR_OBSTACLE
            elif cell in dynamic:
                color = COLOR_DYNAMIC
            elif cell == getattr(env, "agent_pos", None) and cell == getattr(
                env, "current_goal", None
            ):
                color = tuple((np.array(COLOR_AGENT) + np.array(COLOR_GOAL)) // 2)
            elif cell == getattr(env, "agent_pos", None):
                color = COLOR_AGENT
            elif cell == getattr(env, "current_goal", None):
                color = COLOR_GOAL
            else:
                color = COLOR_FREE
            img[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px] = color
    return img


def show(env, title: str = "grid-world", save_path: str | None = None):
    """Display the grid with Matplotlib; optionally save to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = render_rgb(env)
    _, ax = plt.subplots(figsize=(env.width * 0.7 + 1, env.height * 0.7 + 1))
    ax.imshow(img)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.close()
    return img
