# STAGE 3 — Baseline Algorithms: mathematics and measured performance

**Status:** complete, awaiting approval.

## 1. What each algorithm is (and is not)

- **BFS** (`algorithms/bfs.py`) — planning oracle with the full map. Expands
  states in order of distance from the start, so the first time it reaches the
  goal the path is shortest (every move costs 1). No learning, no interaction.
- **Dijkstra** (`algorithms/dijkstra.py`) — planning oracle with the full map
  plus per-move costs. A priority queue always expands the cheapest known
  state; with non-negative costs the first goal pop is optimal. Uniform cost
  reduces exactly to BFS. No learning, no interaction.
- **Tabular Q-learning** (`algorithms/q_learning.py`) — a genuine learner for
  one fixed goal. No map access; it learns only from `(s, a, r, s')` samples.
- **Goal-conditioned Q-learning** (`algorithms/goal_conditioned.py`) — a
  genuine learner for many goals sharing one `Q(s, a, g)` table plus
  hindsight relabelling of failed episodes.
- **Retraining baseline** (`experiments/retraining.py`) — the procedure, not
  an algorithm: after any change, train a fresh learner from scratch. Every
  Stage 4+ adaptation claim must beat this on post-change interactions.

## 2. Mathematics (from first principles)

**BFS optimality.** BFS visits cells in non-decreasing distance from the
start, because every edge adds exactly 1. When the goal is dequeued, all
shorter paths have already been exhausted — hence optimal. Cost:
`O(V + E)` time, `O(V)` memory.

**Dijkstra optimality.** The priority queue always holds the cheapest known
route to each frontier cell. Popping the minimum is safe: any alternative
route must pass through a queued cell of equal or greater cost (costs are
non-negative), so it cannot improve on it. Goal pop = optimal cost.

**Q-learning update.** The agent cannot evaluate the Bellman expectation
because `P` is unknown, so it substitutes one sample:

`Q(s,a) ← Q(s,a) + α [r + γ max Q(s′,a′) − Q(s,a)]`.

The bracket (TD-error) is an unbiased sample of the Bellman residual; the
update is stochastic approximation of a contraction (`γ < 1`), converging
when every `(s, a)` is visited infinitely often (hence ε-greedy) and `α`
decays. Terminal goals contribute `Q = 0` beyond the goal.

**Goal-conditioning + HER.** One table `Q(s, a, g)` generalises across goals
(tabular UVFA). A failed episode for goal `g` that visited `s′` is a
successful episode for goal `g′ = s′`; re-scoring its transitions with
`R_{g′}` (our `_hindsight_replay`) extracts training signal from failures.

**Execution vs. training efficiency.** BFS/Dijkstra have optimal execution
(path length) at zero training cost — but only because they are handed the
map. Learners pay training interactions to earn execution quality. Comparing
a planner's path length against a learner's as if both "learned" would be
dishonest; planners are upper bounds, learners compete on interactions.

## 3. Measured results (initial 8×8 environment, wall row, seed 0)

Environment: 8×8, wall `(3,1..6)`, goals `(7,7),(7,0),(0,7),(0,0)`.

| Method | Task | Result (measured) | Cost |
|---|---|---|---|
| BFS | (0,0)→(7,7) shortest path | 14 steps | 0 interactions (map given) |
| Dijkstra (uniform) | same | 14.0 | 0 interactions (map given) |
| Tabular Q (2000 eps) | single goal (7,7), fixed start | success 1.00, mean len 14.0 | 0.3 s CPU |
| Goal-cond Q + HER (4000 eps) | 4 goals, random starts | success 1.00, mean len 6.5¹ | 0.6 s CPU |

¹ Averaged over all four goals (some are nearby); single-goal length matches
the 14-step optimum. Curves: `docs/learning_curves_stage3.png`.

## 4. Files created

- `algorithms/{bfs,dijkstra,q_learning,goal_conditioned}.py`,
  `experiments/{initial_training,retraining}.py`,
  `tests/test_algorithms.py` (13 tests), this note, the learning-curve figure.

## 5. Tests — 29/29 pass

`python -m unittest discover -s tests`: 16 Stage-2 + 13 new (BFS length /
obstacles / unreachable / start-is-goal, Dijkstra uniform/penalty/
unreachable, Q update clipping / route learning / argmax, hindsight write /
two-goal learning, retraining procedure). Plus the real training run above.
