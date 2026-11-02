"""Breadth-first search: shortest paths on deterministic unweighted grids.

BFS is a *planning oracle*, not a learner: it is given the full map
(dimensions + obstacles) and returns an optimal route without any environment
interaction. In this project it serves as an upper-bound reference for what a
perfectly informed agent could execute.
"""

from __future__ import annotations

from collections import deque

from environment.gridworld import ACTION_DELTAS


def bfs(width, height, obstacles, start, goal):
    """Shortest cell path from start to goal (inclusive), or None.

    All moves cost 1. Obstacle and out-of-grid cells are never entered.
    """
    start, goal = tuple(start), tuple(goal)
    blocked = {tuple(c) for c in obstacles}
    if start in blocked or goal in blocked:
        return None
    if start == goal:
        return [start]
    queue = deque([start])
    came_from = {start: None}
    while queue:
        current = queue.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if nxt in blocked or nxt in came_from:
                continue
            came_from[nxt] = current
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                return path[::-1]
            queue.append(nxt)
    return None


def bfs_on_env(env, start=None, goal=None):
    """Convenience wrapper reading map, start and goal from an environment."""
    start = env.agent_pos if start is None else tuple(start)
    goal = env.current_goal if goal is None else tuple(goal)
    return bfs(env.width, env.height, env.obstacles, start, goal)
