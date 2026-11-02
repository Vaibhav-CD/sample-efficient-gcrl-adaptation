# Sample-Efficient Adaptation in Goal-Conditioned RL under Environment Changes

## Research report (Stages 1–6 record; no hypothetical data)

---

## 1. Abstract

An RL agent trained to navigate a 2D grid faces blocked passages, changed
slip and unseen goals after training. We ask whether reused goal-reaching
knowledge can cut the additional training interactions needed to recover a
high success rate. Starting from a literature survey, we identify a gap:
tabular goal-conditioned methods have no lightweight mechanism that
localises stale knowledge after abrupt local changes. We build a CPU-only
prototype (NumPy, no neural networks), implement planning oracles, tabular
Q-learning, goal-conditioned Q-learning with hindsight relabelling, three
adaptation mechanisms and a new method, Change-Aware Selective Retention
(CASR), which probes with a frozen policy, resets Q-values where
Bellman-residual surprise is high, and replays only probe-consistent
transitions. Over 5 seeds × 5 experiments: plain replay of stale
transitions collapses (Experiment B: 2/5 seeds reach 90%, final 0.77);
oracle selective reset + filtering is best (143 episodes); CASR with zero
oracle information reaches 90% in 210 episodes, beating warm start (279)
but not retraining from scratch (180). Our main hypothesis is therefore
reported as failed, with the mechanism validated and a cheaper detector
identified as the next step.

## 2. Introduction

Agents deployed in the real world meet changed environments: corridors
close, floors get slippery, goals move. Retraining from scratch wastes
every prior interaction; naive reuse risks negative transfer. This project
studies the middle ground on a laptop CPU: what exactly to keep, what to
discard, and how to tell the difference from interaction alone.

## 3. Problem statement

After training in MDP `M = (S, A, P, R, γ)`, the agent faces
`M′ = (S, A, P′, R′, γ)` with new obstacles, changed slip, or new goals.
Research question: can reused goal-reaching knowledge reduce post-change
interactions to a high success rate? Primary metric: `N_0.9`, post-change
episodes until a 100-episode moving success average reaches 0.9.

## 4. Literature review (condensed; full text: Stage-1 note)

UVFA (Schaul et al., 2015) conditions one value function on goals; HER
(Andrychowicz et al., 2017) relabels failures into signal; surveys (Colas
et al., 2022; Liu et al., 2022) cover fixed-environment goal
generalisation. Transfer RL (Taylor & Stone, 2009; Zhu et al., 2023) warns
of negative transfer. Probabilistic policy reuse (Fernández & Veloso,
2006) reuses whole policies; successor features (Barreto et al., 2017)
transfer across rewards, not dynamics. PER (Schaul et al., 2016)
prioritises TD-error, not staleness. Non-stationary analyses (Cheung et
al., 2020) target slow drift; DEER (arXiv:2509.15032, 2025) prioritises by
dynamics discrepancy for deep agents. Gap: no lightweight tabular
goal-conditioned mechanism that detects *where* change struck, resets only
affected values, and replays only consistent transitions.

## 5. Mathematical formulation (condensed; full text: Stage-1 note)

Goal-augmented MDP with `π(a|s,g)`; execution efficiency = expected hitting
time `H^π(s₀,g)`; training efficiency = interactions to learn a good
policy (our objective — never confuse the two). Bellman expectation and
optimality for `V^π(s,g)`, `Q*(s,a,g)`; Q-learning
`Q += α[r + γ max Q(s′) − Q]` as stochastic approximation of a contraction
with ε-greedy coverage. Transfer `Q_T ← Q_S` with reset mask; replay valid
only when stored targets match `P′, R′`; sample efficiency = Taylor &
Stone time-to-threshold on post-change samples.

## 6. Existing approaches

BFS/Dijkstra (map-given oracles, upper bounds only); tabular Q-learning;
goal-conditioned Q + HER; retraining-from-scratch procedure (the floor
every method must beat). Stage-3 measurements (8×8, wall, seed 0): BFS 14
steps, Dijkstra 14.0, tabular Q success 1.00 at 14.0 steps (2000 eps),
goal-conditioned Q success 1.00 at mean 6.5 over 4 goals (4000 eps).

## 7. Proposed methodology: CASR

Probe (frozen greedy, record `δ(s) = |r + γ max Q_S − Q_S|` per state and
bump-implied blockages) → localise (`U(s) ≥ mean + std`, margin growth) →
selectively reset → adapt (online Q + replay filtered by the inferred
blockage map). Distinctions: PER never resets; DEER reweights for deep
agents; Stage-4 reset needs the oracle location. Budgets count probe
episodes honestly (`N_total = probe + adapt`).

```text
CASR(Q_S, buffer, changed env M′):
    U, bumps, blocked ← probe(M′, π_S, P episodes)   # frozen, no updates
    C ← {s : U(s) ≥ mean(U) + std(U)} ∪ bumps
    Q[s] ← 0 for s in margin-growth(C)               # keep Q_S elsewhere
    repeat: ε-greedy rollout + online Q-updates
            replay K stored transitions with endpoints ∉ blocked
```

## 8. System architecture

```text
app.py                          Streamlit dashboard (Stage 7)
environment/gridworld.py        MDP: states, 4 actions, slip kernel, rewards
environment/dynamic_gridworld.py  changes, movers, changed-cell reporting
algorithms/{bfs,dijkstra,q_learning,goal_conditioned}.py  Stage-3 baselines
algorithms/knowledge_reuse.py   mechanisms A/B/C + adaptation loop (Stage 4)
algorithms/proposed_method.py   CASR (Stage 5)
experiments/{initial_training,retraining,adaptation,casr,evaluation}.py
visualization/{grid_visualizer,dashboard}.py
utils/{config,reproducibility}.py
tests/  50 tests (stdlib unittest, pytest-compatible)
docs/   per-stage notes, figures, results_stage6.csv
```

## 9. Experimental setup

8×8 grid, wall row, 4 goals, `α=0.2, γ=0.95, ε 1.0→0.997→0.05` (reuse:
`ε₀=0.3`); 4000 pre-train episodes, 200-episode buffer, 1500-episode
adapt budget; seeds 0–4. Experiments: A initial learning; B sealed
passage; C slip→0.2; D 2 retained + 2 unseen goals; E combined. Metrics:
greedy success, mean/median successful length, `N_0.9`, adapt time.

## 10. Results and analysis

A: `N_0.9` 135 ± 7, success 1.00. B: combo 143 ± 54, oracle 165 ± 56,
scratch 180 ± 11, CASR 210 ± 31, no-reset 238 ± 45, warm 279 ± 23,
plain replay 1037 (2/5, 0.77), CASR-no-filter 1477 (1/5, 0.84). C/D are
floor effects (all ≈ 100–256; policy already robust / new slices learn
fast) — they bound CASR's overhead, not its win. E: combo 108 ± 6, oracle
143, warm+replay 152 (0.90), warm 167 ± 47, CASR 176 ± 19, scratch
181 ± 18. Every number observed; per-seed rows in `results_stage6.csv`.

## 11. Ablation studies

Filter removal reproduces poisoning collapse (B: 1/5 reach) — load-bearing.
Reset removal costs ~28 episodes on B, ~28 on E — modest but consistent.
Probe cost (30 episodes) is not yet repaid by region quality — the revision
target.

## 12. Limitations

H1 as stated fails (CASR 210 > scratch 180 on B). Five seeds, not ten; one
grid size; C/D too easy; fixed probe budget wasteful when little changed;
episode-based budgets overstate probe cost in steps; tabular slices cannot
share across truly novel goals; dashboard browser rendering unverified in
sandbox.

## 13. Conclusion and future work

Selective reset + consistency filtering is validated by oracle runs; stale
replay is a demonstrated failure mode; autonomous detection works
(inferred the exact new cell) but must get cheaper — step-budgeted or
online probes. Next: stronger C/D variants, 10 seeds, probe-while-adapting,
then the presentation (Stage 9).

## 14. References

Watkins & Dayan (1992); Sutton & Barto (2018); Moore (1959); Dijkstra
(1959); Lin (1992); Schaul et al. (2015, 2016); Andrychowicz et al.
(2017); Barreto et al. (2017); Fernández & Veloso (2006); Taylor & Stone
(2009); Duan et al. (2016); Cheung et al. (2020); Colas et al. (2022);
Liu et al. (2022, arXiv:2201.08299); Zhu et al. (2023); Mnih et al.
(2015); DEER (arXiv:2509.15032, 2025). Full citations: Stage-1 note.
