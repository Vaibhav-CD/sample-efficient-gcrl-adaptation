# Technical Presentation: Can Old Routes Survive New Walls?

## Audience guidance: one honest story in 10 slides, ~20 minutes + demo.

---

## Slide 1 — Motivation (2 min)

Robots meet closed corridors, slippery floors and new goals after training.
Retraining wastes everything; blind reuse drives into new walls. Question:
can an agent keep what still works and relearn only what broke — with fewer
interactions? Say plainly: this is about *training* cost, not driving skill.

## Slide 2 — Problem definition (2 min)

Grid agent trained to reach goals; then passages seal, slip changes, goals
move. Metric on screen: `N_0.9` — extra episodes until success is back at
90%. Demo cue: open the dashboard tab B; the sealed passage is visible.

## Slide 3 — Why existing methods fall short (2 min)

Table (from Stage 1): whole-policy reuse replays blocked routes; replay
replays stale transitions; successor features assume unchanged dynamics;
planners cheat with the full map. Punchline: nobody asks *which part* of
old knowledge went stale.

## Slide 4 — Research gap (1 min)

No lightweight tabular method detects *where* change struck, resets only
that region, and replays only still-true transitions. Falsifiable promise
on screen: beat retraining and warm start, or revise — not claim.

## Slide 5 — Math, simply (3 min)

- Policy `π(a|s,g)`: same crossing, different turn per goal.
- Hitting time: how long a good drive takes (execution).
- `N_0.9`: how long *learning* takes (our target — different thing).
- Bellman/Q-learning in one line: nudge each estimate toward observed
  reward plus discounted best-next; ε-greedy guarantees every corner gets
  visited. Demo cue: dashboard tab F heatmap — bright means near goal.

## Slide 6 — Proposed algorithm: CASR (2 min)

Probe frozen → surprise map → reset the surprised neighbourhood → relearn
with replay filtered by observed blockages. Pseudocode (4 lines, from the
report §7). Stress: probe episodes are counted in our own budget.

## Slide 7 — Simulation demonstration (3 min, live)

Dashboard tab A: train (~1 s), watch the greedy route vs BFS optimum. Tab
C: toggle methods on Experiment B bars. If the browser fails, fall back to
`docs/eval_curves_stage6.png`.

## Slide 8 — Experimental comparisons (2 min)

Setup in one breath: 8×8, 4000 train + 1500 adapt episodes, 5 seeds,
8 methods, Experiments A–E. State the oracles-vs-learners fairness rule.

## Slide 9 — Results, honestly (2 min)

B table headline: oracle combo 143, scratch 180, CASR 210, warm 279, plain
replay collapses (2/5, 0.77). C/D are floor effects — say so, show the
caveat (probes cost when nothing broke). Verdict on screen: **H1 fails**;
mechanism validated, detector too pricey. This slide is the credibility
of the whole talk.

## Slide 10 — Limitations and future directions (1 min)

5 seeds not 10, one grid, fixed probe budget, tabular slices don't share
across novel goals. Next: step-budgeted online probes, stronger slip/goal
shifts, then neural function approximation only with justification.

---

## Speaker checklist

- [ ] `streamlit run app.py` open + CSV tables loaded.
- [ ] Fallback PNGs ready (`docs/*.png`).
- [ ] Anticipated Q: "Why not deep RL?" → laptop-CPU constraint, tabular
      isolates the knowledge question from function approximation.
- [ ] Anticipated Q: "Is 210 vs 180 noise?" → std 31 vs 11 over 5 seeds,
      disclosed; 10-seed rerun is listed future work, not a claim.
