"""Dijkstra's algorithm: shortest paths on weighted grids.

Same oracle status as BFS, but each move has a caller-supplied cost, so it
finds minimum-*cost* routes (e.g. avoiding penalty cells). With uniform cost 1
it reduces to BFS. Used as the known-map reference for weighted environments.
"""

from __future__ import annotations

import heapq

from environment.gridworld import ACTION_DELTAS


def dijkstra(width, height, obstacles, start, goal, cost_fn=None):
    """Returns ``(path, total_cost)`` or ``(None, inf)`` if unreachable.

    ``cost_fn(from_cell, to_cell)`` gives the price of one move; default 1.0.
    """
    start, goal = tuple(start), tuple(goal)
    blocked = {tuple(c) for c in obstacles}
    if start in blocked or goal in blocked:
        return None, float("inf")
    step_cost = cost_fn or (lambda _f, _t: 1.0)
    best = {start: 0.0}
    came_from = {start: None}
    heap = [(0.0, start)]
    while heap:
        cost, current = heapq.heappop(heap)
        if cost > best[current]:
            continue
        if current == goal:
            break
        for dr, dc in ACTION_DELTAS.values():
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if nxt in blocked:
                continue
            new_cost = cost + float(step_cost(current, nxt))
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(heap, (new_cost, nxt))
    if goal not in came_from:
        return None, float("inf")
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    return path[::-1], best[goal]


def dijkstra_on_env(env, start=None, goal=None, cost_fn=None):
    """Convenience wrapper reading map, start and goal from an environment."""
    start = env.agent_pos if start is None else tuple(start)
    goal = env.current_goal if goal is None else tuple(goal)
    return dijkstra(env.width, env.height, env.obstacles, start, goal, cost_fn)
