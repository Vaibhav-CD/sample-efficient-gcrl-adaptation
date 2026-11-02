# STAGE 5 — CASR: Change-Aware Selective Retention

**Status:** complete, awaiting approval. Implementation:
`algorithms/proposed_method.py`; study: `experiments/casr.py`.

## 1. Limitation selected (from Stage 4)

Stage-4 selective reset won, but cheated twice: the reset location came from
the environment oracle (`last_changed_cells`) and the radius `k = 2` was a
guess — on the small-barrier variant it destroyed useful knowledge
(N_0.9 = 191 vs warm = 100). CASR keeps the win and removes both crutches:
detect the stale region from interaction, and let the evidence set its size.

## 2. Why this should help (one paragraph)

After a local change, Bellman residuals spike exactly where old values are
wrong: transitions near the change produce rewards and successors the old Q
never predicted, while far-field transitions backup almost as before. Short
frozen-policy probes therefore turn "where am I surprised?" into a sensor for
"what changed?", at the price of a few dozen probe episodes. Resetting only
the surprised neighbourhood restores an unbiased prior where it is needed and
keeps transferred knowledge everywhere else; replay filtered by probe-observed
blockages avoids re-injecting the stale targets that collapsed plain replay
to 0.81 in Stage 4.

## 3. Mathematical formulation

Probe episodes run the frozen greedy policy `π_S` in the changed MDP `M′`.
For a visited `(s, a, r, s′)` define the absolute Bellman residual under the
transferred table:

`δ(s) = |r + γ max_a′ Q_S(s′,a′,g) − Q_S(s,a,g)|`, averaged per state into
surprise `U(s) = mean δ(s)`.

Under unchanged dynamics `U(s)` is small (Bellman error of a converged
table); near the change it is large (wrong `r, s′` but old `Q_S`). With
`μ, σ` the mean/std of `U` over visited states, seed set
`C = {s : U(s) ≥ μ + σ} ∪ {bump states}`, reset region = Manhattan
`margin`-growth of `C`, and reset `Q_S[s] ← 0` on the region. Blockage
inference: a probe step with `s′ = s` and bump flag implies the entered
neighbour is blocked — one binary observation, no map access. Replay keeps a
stored transition iff neither endpoint is in the inferred blocked set.

Budget accounting is honest: `N_total = probe_episodes + N_adapt`, and probe
*steps* (796 here ≈ 5 adapt-episodes of steps) are reported alongside.

## 4. Pseudocode

```text
CASR(Q_S, buffer, changed env M′):
    # 1. PROBE (frozen policy, no updates)
    U, bumps, blocked ← probe(M′, π_S, P episodes)
    # 2. LOCALISE (adaptive threshold + margin growth)
    C ← {s : U(s) ≥ mean(U) + std(U)} ∪ bumps
    R ← margin-growth(C)
    # 3. SELECTIVE RESET
    Q[s] ← 0 for s in R          # Q ← Q_S elsewhere
    # 4. ADAPT (online Q-learning + inferred-map-filtered replay)
    repeat for adapt_episodes:
        rollout with ε-greedy, update Q online
        replay K stored transitions with endpoints ∉ blocked
```

## 5. Initial results (Stage-4 protocol, seed 0 — observed, not claimed)

| Method | N to 90% | Final |
|---|---|---|
| scratch | 170 | 1.00 |
| warm (A) | 136 | 1.00 |
| plain replay (A+B) | 457 | 0.81 |
| oracle selective (A+C) | 106 | 1.00 |
| combo (A+Bf+C) | 110 | 1.00 |
| **CASR (no oracle)** | **159 (30 probe + 129)** | **1.00** |

Detection diagnostics: 4 seeds, 13 states reset, inferred blocked = exactly
`{(3,7)}` (the true change, found from bumps alone); precision 0.23 / recall
0.75 vs oracle cells (oracle counts neighbours; the region intentionally
covers a margin). 796 probe steps, 70,525 filtered replay updates applied.

Reading: CASR beats retraining (159 < 170) with zero oracle information, but
trails oracle-aided reset (106) and plain warm start (136) — detection costs
30 probe episodes and an imperfect region. That overhead, not a victory lap,
is the honest Stage-6 target: multi-seed proof that detection pays off where
changes are larger or stochastic.

## 6. Contribution test (Stage-1 rule applied)

Surveyed methods with the same goal: PER (replays surprise, never resets),
DEER (learned discrepancy weights, deep regime), successor features (reward
change only), probabilistic reuse (whole policies). None combines
probe-detected hard reset with an inferred blockage map for tabular
goal-conditioned adaptation. The claim stays narrow — a lightweight
detection/reset/filter loop — and revises if Stage-6 literature checks find a
match.

## 7. Files and tests

- `algorithms/proposed_method.py` (probe, detect, score, CASR pipeline).
- Small backward-compatible `keep`-predicate extension in
  `algorithms/knowledge_reuse.py` (replay/adapt loop, all old calls unaffected).
- `experiments/casr.py`, `tests/test_proposed_method.py` (6 tests).
- Full suite: **45/45 pass**.
