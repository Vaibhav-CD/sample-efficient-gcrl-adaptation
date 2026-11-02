"""CASR — Change-Aware Selective Retention (the Stage-5 contribution).

Problem (from Stage 4): warm start + selective reset wins, but Stage 4
cheats twice — the reset location comes from the environment's oracle
``last_changed_cells`` and the radius ``k = 2`` is a guess (it hurt on the
small-barrier variant). CASR removes both crutches:

1. **Probe** — run short frozen-greedy episodes in the changed environment
   and record, per visited state, the absolute Bellman (TD) residual plus
   bump events. No learning happens during probing.
2. **Localise** — states whose mean surprise exceeds ``mean + std`` over
   visited states (adaptive threshold, no hand-tuned constant), plus
   bump states, form the seed set; the reset region is the seed set grown by
   a margin. The oracle changed cells are never consulted.
3. **Selectively reset** — re-initialise Q over the detected region only.
4. **Adapt** — online Q-learning plus replay filtered by an *inferred*
   blockage map built solely from probe bump observations
   (``position unchanged after a move  =>  entered cell is blocked``).

What is new vs. Stage 1/4 literature: PER replays high-TD-error transitions
but never invalidates values; DEER reweights by a learned discrepancy model
for deep agents; Stage-4 selective reset needs the oracle location. CASR is a
detection + hard-reset + inferred-map filter loop for tabular
goal-conditioned agents, with probe episodes honestly counted in the
interaction budget.
"""

from __future__ import annotations

import numpy as np

from algorithms.knowledge_reuse import adapt_post_change, selective_reset
from environment.gridworld import ACTION_DELTAS


def td_error(learner, state, action, reward, next_state, terminated, goal=None):
    """Absolute Bellman residual of one transition under current Q."""
    if goal is None:
        q = learner.Q[state[0], state[1], action]
        boot = 0.0 if terminated else float(np.max(learner.Q[next_state[0], next_state[1]]))
    else:
        gi = learner.goal_index[tuple(goal)]
        q = learner.Q[state[0], state[1], action, gi]
        boot = 0.0 if terminated else float(np.max(learner.Q[next_state[0], next_state[1], :, gi]))
    return abs(reward + learner.gamma * boot - q)


def run_probes(env, learner, n_episodes: int, seed: int = 0) -> dict:
    """Frozen-greedy probe episodes. Returns diagnostics dict.

    Keys: ``surprises`` (state -> mean abs TD error), ``visits``,
    ``bump_states``, ``inferred_blocked`` (cells observed to block entry),
    ``probe_episodes``, ``probe_steps``. The learner is not modified
    (Q and epsilon are restored).
    """
    goal_conditioned = hasattr(learner, "goals")
    rng = np.random.default_rng(int(seed))
    saved_Q, saved_eps = learner.Q.copy(), learner.epsilon
    errors, visits = {}, {}
    bump_states = set()
    inferred_blocked = set()
    steps = 0
    try:
        learner.epsilon = 0.0
        for _ in range(int(n_episodes)):
            if goal_conditioned:
                goal = learner.goals[rng.integers(len(learner.goals))]
                state, _ = env.reset(goal=goal)
            else:
                goal = None
                state, _ = env.reset()
            done = False
            while not done:
                action = (learner.act(state, goal, explore=False)
                          if goal_conditioned else learner.act(state, explore=False))
                next_state, reward, terminated, truncated, info = env.step(action)
                err = td_error(learner, state, action, reward, next_state, terminated, goal)
                errors.setdefault(tuple(state), []).append(err)
                visits[tuple(state)] = visits.get(tuple(state), 0) + 1
                if tuple(next_state) == tuple(state) and info.get("bumped"):
                    bump_states.add(tuple(state))
                    dr, dc = ACTION_DELTAS[int(action)]
                    candidate = (state[0] + dr, state[1] + dc)
                    if 0 <= candidate[0] < env.height and 0 <= candidate[1] < env.width:
                        inferred_blocked.add(candidate)
                steps += 1
                state = next_state
                done = terminated or truncated
    finally:
        learner.Q = saved_Q
        learner.epsilon = saved_eps
    surprises = {s: float(np.mean(v)) for s, v in errors.items()}
    return {"surprises": surprises, "visits": visits, "bump_states": bump_states,
            "inferred_blocked": inferred_blocked, "probe_episodes": int(n_episodes),
            "probe_steps": steps}


def detect_region(probe: dict, height: int, width: int, margin: int = 1,
                  std_factor: float = 1.0) -> dict:
    """Threshold surprises and grow the seed set; returns region diagnostics."""
    surprises = probe["surprises"]
    if not surprises:
        return {"seeds": set(), "region": set(), "threshold": float("inf")}
    values = np.array(sorted(surprises.values()))
    threshold = float(values.mean() + std_factor * values.std())
    seeds = {s for s, v in surprises.items() if v >= threshold}
    seeds |= set(probe["bump_states"])
    region = set()
    for (r, c) in seeds:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                if abs(dr) + abs(dc) <= margin:
                    cell = (r + dr, c + dc)
                    if 0 <= cell[0] < height and 0 <= cell[1] < width:
                        region.add(cell)
    return {"seeds": seeds, "region": region, "threshold": threshold}


def score_detection(region: set, oracle_cells) -> dict:
    """EVALUATION ONLY: precision/recall of ``region`` vs oracle changed cells."""
    oracle = {tuple(c) for c in oracle_cells}
    if not region:
        return {"precision": 0.0, "recall": 0.0}
    return {"precision": len(region & oracle) / len(region),
            "recall": len(region & oracle) / len(oracle) if oracle else 1.0}


def casr_adapt(env, learner, buffer=None, probe_episodes: int = 30, margin: int = 1,
               replay_per_episode: int = 50, adapt_episodes: int = 1500,
               eps_start: float = 0.3, eps_decay: float = 0.997,
               eps_min: float = 0.05, probe_seed: int = 0) -> dict:
    """Full CASR pipeline. Returns history plus detection diagnostics.

    Probe episodes are part of the interaction budget and reported as such.
    """
    probe = run_probes(env, learner, probe_episodes, seed=probe_seed)
    detection = detect_region(probe, env.height, env.width, margin=margin)
    n_reset = selective_reset(learner.Q, detection["seeds"], radius=margin)
    blocked = set(probe["inferred_blocked"])

    def keep(state, action, next_state):
        return tuple(next_state) not in blocked and tuple(state) not in blocked

    history, replay_applied = adapt_post_change(
        env, learner, adapt_episodes, buffer=buffer,
        replay_per_episode=replay_per_episode if buffer is not None else 0,
        keep=keep if buffer is not None else None,
        eps_start=eps_start, eps_decay=eps_decay, eps_min=eps_min)
    return {"history": history, "replay_applied": replay_applied,
            "n_reset_states": n_reset, "n_seeds": len(detection["seeds"]),
            "region": set(detection["region"]),
            "region_size": len(detection["region"]),
            "threshold": detection["threshold"],
            "n_inferred_blocked": len(blocked),
            "inferred_blocked": sorted(blocked),
            "probe_episodes": probe["probe_episodes"], "probe_steps": probe["probe_steps"]}
