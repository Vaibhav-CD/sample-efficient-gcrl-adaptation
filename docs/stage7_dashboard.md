# STAGE 7 — Interactive Visualization Dashboard

**Status:** complete, awaiting approval. Entry: `app.py`
(`streamlit run app.py`); helpers: `visualization/dashboard.py`.

## 1. What was built (tabs A–G, as specified)

- **A. Live simulation** — slider-controlled training episodes, one-click
  train of a demo agent (~1 s), greedy route drawn on the grid with step
  count, BFS optimum and success/failure status.
- **B. Before/after** — original vs passage-sealed environments side by side
  with old (14-step) and new optimal routes; the broken corridor is visible.
- **C. Knowledge reuse** — method multiselect over the real Stage-6 CSV
  (Experiment B): bar chart of additional episodes to 90% plus success, time
  and reach counts. No hardcoded numbers.
- **D. Algorithm comparison** — BFS/Dijkstra computed live plus measured
  Stage-3/6 rows, with the oracle-vs-learner distinction captioned.
- **E. Learning curves** — the three saved real artifacts (Stages 3, 4, 6).
- **F. Q-heatmaps** — max-over-actions values from a live-trained agent,
  obstacles masked.
- **G. Experiment summary** — per-experiment table from the real CSV with a
  pointer to the per-seed rows.

## 2. Verification (honest)

- `py_compile` clean; full suite **50/50 pass** (4 new dashboard tests:
  CSV loads with >100 real rows, summary contains CASR, demo trains and
  routes, grid renders with a 10+-cell route).
- The headless serve check **could not run here**: this machine's sandbox
  denies socket creation (`PermissionError` on `socketpair` before the app
  even loads), so no local URL was reachable in verification. Please run
  `streamlit run app.py` from the project root and open the shown URL —
  all data callbacks were exercised by the tests except actual browser
  rendering.

## 3. Files

- `app.py`, `visualization/dashboard.py`, `tests/test_dashboard.py`,
  this note.
