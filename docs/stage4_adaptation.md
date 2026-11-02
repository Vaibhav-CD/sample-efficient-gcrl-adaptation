# STAGE 4 — Knowledge Reuse and Adaptation Mechanisms: measured comparison

**Status:** complete, awaiting approval.

## 1. Protocol (what was actually run)

`python -m experiments.adaptation`: goal-conditioned Q trained 4000 episodes
in the Stage-3 env (pre-change success 1.00), 200 rollout episodes stored
(1864 transitions), then obstacle `(3,7)` seals the right-hand corridor
passage — every cross-wall route must be re-learned via `(3,0)`. Five methods
adapt for 1500 post-change episodes: scratch, warm (A), warm+replay (A+B),
warm+selective (A+C, reset radius 2), combo (A + filtered replay + reset).
Metric: `N_0.9` = first post-change episode where the 100-episode moving
average reaches 0.9; `final` = mean of last 200 episodes.

## 2. Results (measured, seed 0)

| Method | N_0.9 (episodes) | Final success |
|---|---|---|
| scratch (retraining baseline) | 170 | 1.00 |
| warm start (A) | 136 | 1.00 |
| warm + plain replay (A+B) | 457 | 0.81 |
| warm + selective reset (A+C) | 106 | 1.00 |
| combo (A + filtered replay + reset) | 110 | 1.00 |

Diagnostics: 4 cells affected, 16 states reset, 136/1864 stored transitions
inconsistent; filtered replay applied 69,525 updates vs 75,000 unfiltered.

## 3. Reading the table (no improvement claimed without support)

- **A helps:** warm start (136) beats scratch (170) — old far-field knowledge
  transfers. Small but genuine jumpstart.
- **B alone harms — negative transfer demonstrated:** plain replay (457,
  final 0.81, never fully recovers) is far worse than scratch. Replaying
  transitions recorded under the old dynamics injects wrong Bellman targets
  episode after episode. This is the Stage-1 warning made concrete.
- **C fixes it:** selective reset (106) is the best time-to-threshold —
  dropping the 16 stale states while keeping everything else beats both
  warm start and scratch.
- **Filtering ≈ reset here:** combo (110) matches selective reset (106). Once
  stale values are cleared, online learning already avoids the poisoned
  region, so filtering adds little *in this change*. Its value should appear
  under larger stochastic changes — a Stage-6 question, not a claim.

## 4. Mathematical / algorithmic differences

- **A (transfer):** `Q_T ← Q_S`, then ordinary Q-updates. Prior over values;
  no data reuse. Cost: one array copy. Risk: stale values bias exploration.
- **B (replay):** extra updates `Q ← Q + αδ` on stored `(s,a,r,s′)` with
  `δ = r + γ max Q(s′) − Q(s,a)`. After the change, stored `r, s′` come from
  `P, R` but bootstrap against `Q` under `P′, R′` — biased targets, hence the
  0.81 collapse.
- **C (selective):** (i) `Q[s] ← 0` for `dist(s, changed) ≤ k` — a surgical
  return to the scratch prior exactly where staleness lives; (ii) replay only
  transitions with `P′(s′|s,a) > 0` under the current map (exact check when
  slip = 0, obstacle check otherwise). Cost: one neighbourhood sweep + one
  map lookup per replayed transition.

## 5. Failure cases examined

1. Plain replay after blockage (0.81 final) — stale-target poisoning.
2. Selective reset on the *small-barrier* variant (first run: N_0.9 = 191 vs
   warm = 100) — resetting destroys useful knowledge when the change barely
   matters. Lesson: reset radius must scale with detected change, not be
   fixed blindly. This motivates the Stage-5 probe-based calibration.
3. Scratch is slow but never poisoned — the honest floor every method must
   beat.

## 6. Trade-off verdict and Stage-5 direction

Best trade-off: **A + C**. Warm start keeps far-field knowledge; localised
reset removes exactly the stale part. Open problems for Stage 5: (i) fixed
radius `k = 2` is a guess — calibrate it from probe rollouts that *measure*
where values are wrong; (ii) the change location is currently handed to the
agent by the environment — detect it from prediction errors instead; (iii)
filtered replay needs a stochastic-change test where it can earn its keep.

## 7. Files

- `algorithms/knowledge_reuse.py` (buffer, transfer, reset, filter, adapt
  loop), `experiments/adaptation.py`, `tests/test_knowledge_reuse.py`
  (10 tests), `docs/adaptation_curves_stage4.png`, this note.

## 8. Tests — 39/39 pass

`python -m unittest discover -s tests`: 29 prior + 10 new.
