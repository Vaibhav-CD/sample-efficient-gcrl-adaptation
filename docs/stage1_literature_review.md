# STAGE 1 — Literature Review, Research Gap, and Problem Formulation

**Status:** complete, awaiting approval. No algorithm code is written in this
stage by design.

**Scope of this document:**

1. Surveys (goal-conditioned RL, transfer RL, policy reuse + replay,
   non-stationary RL, sample-efficient adaptation).
2. Algorithm comparison table.
3. Research gap.
4. Mathematical problem formulation from first principles.
5. Testable hypothesis and proposed direction.
6. Experimental plan for Stages 2–6.

---

## 1. Goal-conditioned reinforcement learning

### 1.1 From single-goal values to universal values

Classical Q-learning learns one value function for one reward function.
Goal-conditioned RL instead learns a single function conditioned on the goal.

The foundational step is **Universal Value Function Approximators (UVFA)** —
Schaul, Horgan, Gregor and Silver (2015). Instead of $V(s)$ or $Q(s,a)$, learn
$V(s,g)$ and $Q(s,a,g)$: one function that generalises over both states and
goals. In a navigation grid this is the difference between training one table
per goal and training one goal-augmented table that shares structure ("walls
block movement no matter which goal I want").

### 1.2 Sparse rewards and Hindsight Experience Replay

Goal-reaching rewards are naturally sparse (1 at the goal, 0 or −1 elsewhere),
so most episodes teach nothing. **Hindsight Experience Replay (HER)** —
Andrychowicz et al. (2017) — relabels failed trajectories: a trajectory that
missed goal $g$ but visited state $s'$ becomes a *successful* demonstration
for goal $g' = s'$. This converts waste transitions into training signal and
is the standard sample-efficiency trick for sparse goal-conditioned tasks.

### 1.3 Survey position

- Colas, Karch, Sigaud and Oudeyer (2022) survey intrinsically motivated
  goal-conditioned methods: how agents represent, generate and pursue their
  own goals.
- Liu et al. (2022, arXiv:2201.08299) survey goal-conditioned RL problems and
  solutions (sparse rewards, relabelling, sub-goals, multi-task generalisation).

**What this establishes for us:** goal-conditioning with relabelling is a
solved approach to *unseen goals in a fixed environment*. Our question is
different: goals *and* environment dynamics change after training.

## 2. Transfer reinforcement learning

The canonical framework is Taylor and Stone (2009): knowledge from source
task(s) $M_S$ improves learning on a target task $M_T$ along one or more axes
— jumpstart, asymptotic performance, total reward, time-to-threshold, or
transfer ratio (area under the learning curve). Their taxonomy (task mapping,
transferred representation, allowed learner) still organises the field, and Zhu
et al. (2023) extend it to deep transfer RL.

Central warning from this literature: **negative transfer**. Reused knowledge
helps only when source and target overlap; when the change invalidates old
knowledge, warm-starting can be *worse* than retraining from scratch. Any
honest adaptation claim must therefore test for negative transfer explicitly —
our Experiment B (blocked paths) is designed exactly for this.

## 3. Policy reuse and experience replay

### 3.1 Policy reuse

Fernández and Veloso (2006) introduced **probabilistic policy reuse**: keep a
library of past policies, follow a past policy with probability $\psi$ and
explore otherwise, decaying $\psi$ as the new policy improves. Strength:
immediate jumpstart. Weakness: it reuses whole policies, so one blocked
corridor poisons every reused episode — there is no notion of *which part* of
the old knowledge is stale.

**Successor features** (Barreto et al., 2017, 2018) factorise values into
dynamics features $\psi(s,a)$ and reward weights $w_g$, so a new goal (new
$w$) transfers instantly. Strength: elegant goal transfer. Weakness: it
assumes shared dynamics $\psi$; new obstacles *change the dynamics*, which is
precisely our setting.

### 3.2 Experience replay

Lin (1992) introduced replaying stored transitions; Mnih et al. (2015, DQN)
made it standard for breaking correlations. **Prioritized Experience Replay
(PER)** — Schaul et al. (2016) — replays high-TD-error transitions more often.
Strength: sample efficiency in stationary tasks. Weakness: TD-error measures
*surprise*, not *staleness*. After an environment change, an outdated
transition can have low TD-error under the old values yet actively mislead the
agent — PER will happily keep replaying it.

**Lesson:** neither whole-policy reuse nor TD-error prioritisation answers
"which old knowledge is still true?" That missing filter is where our gap sits.

## 4. RL in non-stationary environments

Stationary theory (Watkins and Dayan, 1992; Sutton and Barto, 2018) assumes
fixed $P$ and $R$. Non-stationary work studies what happens when they drift.

- Cheung, Simchi-Levi and Zhu (2020) analyse RL in non-stationary MDPs with
  regret bounds under a variation budget — principled but aimed at slowly
  drifting tabular MDPs with known budgets, not abrupt obstacle insertion.
- Recent off-policy work proposes **Discrepancy of Environment Dynamics (DoE)
  / DEER** (arXiv:2509.15032, 2025): prioritise replay by how much the
  environment shift affects values. This is the closest published answer to
  our question — but it targets deep off-policy control, not tabular
  goal-conditioned gridworlds, and it reweights rather than localises and
  resets stale values.

## 5. Sample-efficient adaptation

Taylor and Stone's metrics give us the vocabulary: we care about
**time-to-threshold** (interactions to reach, say, 90% of clean retraining
performance) and **transfer ratio**, not just final success. Meta-RL (e.g. RL²,
Duan et al., 2016) achieves few-shot adaptation but pays with massive
meta-training — the opposite of our lightweight constraint. Our question is
narrower and cheaper: after *one* source training run, adapt to *one* changed
environment with minimal extra samples on a laptop CPU.

---

## 6. Comparison table of existing methods

| Method (reference) | What it reuses | Handles new obstacles / blocked paths? | Handles unseen goals? | Staleness filter? | Cost / regime |
|---|---|---|---|---|---|
| Tabular Q-learning from scratch (Watkins & Dayan, 1992) | nothing | yes, by relearning everything | yes, by relearning | n/a (no reuse) | cheap per step, many steps |
| Goal-conditioned tabular Q + HER-style relabelling (Schaul et al., 2015; Andrychowicz et al., 2017) | shared goal-conditioned table | no change handling on its own | yes, in a fixed env | no | cheap, sample-efficient for goals |
| Probabilistic policy reuse (Fernández & Veloso, 2006) | whole past policies | brittle — replays blocked routes | partially | no (decays $\psi$ globally) | cheap, risk of negative transfer |
| Successor features (Barreto et al., 2017, 2018) | dynamics features $\psi$ | no — assumes shared dynamics | yes, instantly for new rewards | no (reward-only transfer) | cheap at transfer, needs feature learning |
| Experience replay / PER (Lin, 1992; Schaul et al., 2016) | past transitions | replays stale ones too | indirectly | no — prioritises TD-error, not freshness | cheap, misleads after shifts |
| DEER / DoE prioritisation (arXiv:2509.15032, 2025) | reweighted transitions | yes, by design | not its focus | yes, via dynamics discrepancy | deep off-policy, heavier regime |
| Non-stationary regret methods (Cheung et al., 2020) | windowed statistics | slow drift, not abrupt blockage | no | yes, via forgetting windows | theory-heavy, needs variation budget |
| BFS / Dijkstra (Moore, 1959; Dijkstra, 1959) | full map (planning, not learning) | optimal with known map | yes, by replanning | n/a | oracles / upper bounds only |

Reading the table row by row: nothing in the cheap tabular column both
localises stale knowledge after an abrupt local change *and* keeps the still
valid far-field knowledge. That empty cell is the gap.

---

## 7. Research gap (specific, falsifiable)

> **Gap:** In tabular goal-conditioned navigation, there is no lightweight
> mechanism that (i) detects *where* the environment changed, (ii) resets
> only the value knowledge affected by that change, and (iii) replays only
> the stored transitions still consistent with the new dynamics — so that
> adaptation avoids both full retraining and negative transfer from stale
> values.

Why this is genuinely distinct:

- vs. policy reuse: we invalidate *regions*, not whole policies.
- vs. PER: we filter by *consistency with new dynamics*, not TD-error.
- vs. successor features: we handle *dynamics* change, not reward change.
- vs. DEER: same intuition (discrepancy), but localised reset + tabular
  goal-conditioned regime + CPU-only constraint.
- vs. planning oracles: BFS/Dijkstra get the true map; our agent must detect
  change through interaction.

Falsification condition: if localised reset + consistency-filtered replay does
not beat both (a) warm-start Q-reuse and (b) retraining from scratch on
time-to-threshold after obstacle insertion, the gap claim fails and we revise
rather than claim novelty.

---

## 8. Mathematical formulation from first principles

### 8.1 Goal-augmented MDP

An MDP is a tuple $(S, A, P, R, \gamma)$:

- $S$: finite grid cells (free cells; walls/obstacles excluded).
- $A = \{\text{up, down, left, right}\}$, $|A| = 4$.
- $P(s' \mid s, a)$: transition probabilities. Deterministic means the
  intended neighbour with probability 1; stochastic ("slip") means the
  intended neighbour with probability $1 - \varepsilon$, else a random
  neighbour. After the environment change we have $P' \ne P$.
- $R_g(s, a, s')$: goal-conditioned reward, e.g. $1$ on entering $g$, $0$
  (or $-1$ per step) otherwise.
- $\gamma \in [0, 1)$: discount factor trading immediate vs. future reward.

The goal-augmented MDP adds a goal space $G \subseteq S$; each episode draws
$g \sim p(G)$ and conditions policy and values on it.

### 8.2 Goal-conditioned policy and expected hitting time

A goal-conditioned policy is $\pi(a \mid s, g)$: a distribution over actions
given current cell *and* desired goal. The same state maps to different
actions for different goals — that is the whole point of conditioning.

**Execution efficiency** is measured by expected hitting time: starting from
$s_0$ and following $\pi(\cdot \mid \cdot, g)$, how many steps until first
reaching $g$:

$$H^\pi(s_0, g) = \mathbb{E}_\pi[\min\{t \ge 0 : s_t = g\} \mid s_0].$$

Short hitting time = good route. This is *execution* efficiency (quality of
the final policy).

**Training efficiency** (our objective) is the number of environment
interactions (steps/samples) needed to *learn* a policy with
$H^\pi \le H^\star + \delta$ or success rate $\ge 1 - \delta$. Fewer
interactions = more sample-efficient. Never confuse the two: BFS has optimal
hitting time but zero training cost because it is given the map — it is not a
learner.

### 8.3 Bellman equations (goal-conditioned)

The return from $(s, g)$ under $\pi$ is
$G_t = \sum_k \gamma^k R_g(s_{t+k}, a_{t+k}, s_{t+k+1})$.
The value $V^\pi(s, g) = \mathbb{E}_\pi[G_t \mid s_t = s]$ satisfies the
Bellman expectation equation:

$$V^\pi(s,g) = \sum_a \pi(a\mid s,g)\sum_{s'} P(s'\mid s,a)\,[R_g(s,a,s') + \gamma V^\pi(s',g)],$$

i.e. "value = average over actions of average over next states of immediate
reward plus discounted future value." The optimal value satisfies Bellman
optimality:

$$V^\star(s,g) = \max_a \sum_{s'} P(s'\mid s,a)\,[R_g(s,a,s') + \gamma V^\star(s',g)],$$

and $Q^\star(s,a,g)$ is the same with the first action fixed:

$$Q^\star(s,a,g) = \sum_{s'} P(s'\mid s,a)\,[R_g(s,a,s') + \gamma \max_{a'} Q^\star(s',a',g)].$$

### 8.4 Q-learning updates (why they work, in one paragraph)

Q-learning replaces the unknown $P$ with samples: after observing
$(s, a, r, s')$, move the current estimate toward the Bellman target,

$$Q(s,a,g) \leftarrow Q(s,a,g) + \alpha\,[r + \gamma \max_{a'} Q(s',a',g) - Q(s,a,g)],$$

with learning rate $\alpha$. The bracket is the TD-error. Under standard
conditions (every $(s,a)$ visited infinitely often, decaying $\alpha$), this
stochastic approximation converges to $Q^\star$ because its expected update is
a contraction mapping (discount $\gamma < 1$ shrinks errors each backup).
Exploration vs. exploitation ($\varepsilon$-greedy) guarantees the "visited
infinitely often" part: with probability $\varepsilon$ act randomly
(explore), otherwise greedily (exploit).

### 8.5 Transfer, replay, and sample efficiency (formalised)

- **Transfer:** initialise target values from source values,
  $Q_{M_{T}}^{(0)} := Q_{M_S}^{(\text{final})}$ (warm start), optionally with
  a reset mask $M$: $Q^{(0)} := (1 - M) \odot Q_S + M \odot Q_{\text{init}}$.
  Our contribution is a principled choice of $M$.
- **Replay:** instead of learning only from the current transition, store
  transitions and re-apply the Q-update to sampled minibatches. It improves
  sample efficiency when replayed transitions are *valid* under $P'$;
  replaying transitions with $P(\cdot) \ne P'(\cdot)$ injects wrong Bellman
  targets — the negative-transfer mechanism of Stage 4.
- **Sample efficiency metric:** $N_\tau$ = interactions in the *changed*
  environment to first reach success rate $\tau$ (e.g. 0.9). Method X beats
  method Y iff $N^X_\tau < N^Y_\tau$ at equal final success — the
  time-to-threshold criterion of Taylor and Stone (2009) restricted to
  post-change samples.

---

## 9. Hypothesis and proposed direction

**H1 (main, testable):** After abrupt local changes (new obstacles, blocked
paths), a tabular goal-conditioned agent that (a) detects changed cells via
short probe rollouts, (b) resets Q-values within a $k$-hop neighbourhood of
detected changes, and (c) replays only stored transitions consistent with the
newly observed dynamics will reach 90% of clean-retraining success with fewer
post-change interactions than both full retraining and naive Q warm-start,
without asymptotic performance loss.

**H0 (null):** localisation and filtering provide no time-to-threshold benefit
over naive warm-start or scratch retraining.

**Working name of the Stage 5 method:** Change-Aware Selective Retention
(CASR): probe → localise → selectively reset → consistency-filtered replay.
Full derivation, pseudocode and implementation belong to Stage 5; Stage 4
builds the three components independently so the ablation is honest.

**Contribution test (decided now, applied in Stage 5):** if the literature
shows an identical tabular method, we revise the claim to the differing
component (e.g. the consistency filter or the $k$-hop reset rule) rather than
claim novelty.

## 10. Experimental plan (Stages 2–6)

- **Stage 2:** configurable gridworld (`gridworld.py`, `dynamic_gridworld.py`):
  sizes, seeds, random starts/goals, obstacles, 4 actions, slip $\varepsilon$,
  dynamic obstacles; seeded tests for legal/illegal moves and goal detection.
- **Stage 3:** baselines — BFS/Dijkstra as map-given oracles (upper bounds,
  never "learners"); tabular Q, goal-conditioned Q, scratch-retraining
  baseline; learning curves in the *initial* environment.
- **Stage 4:** independent A/B/C mechanisms (transfer, replay, selective
  update) + combinations; report negative-transfer cases explicitly.
- **Stage 5:** CASR specification and implementation; correctness tests.
- **Stage 6:** Experiments A–E (initial learning; new obstacles; changed
  slip; unseen goals; combined changes). Metrics: success rate, mean/median
  successful path length, post-change interactions, $N_{0.9}$ (interactions
  to 90%), adaptation speed, unseen-goal success, wall-clock time. Multiple
  seeds (≥5, target 10), mean ± std / 95% CI, ablations of probe, reset
  radius $k$, and replay filter. No fabricated numbers: every table states
  seeds, episodes and commit.

---

## References (genuine)

- Watkins, C. J. C. H. and Dayan, P. (1992). Q-learning. *Machine Learning*,
  8(3–4):279–292.
- Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction*, 2nd ed. MIT Press.
- Moore, E. F. (1959). The shortest path through a maze. *Proc. Int. Symp.
  Switching Theory*. (BFS.)
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
  *Numerische Mathematik*, 1:269–271.
- Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement
  learning, planning and teaching. *Machine Learning*, 8:293–321.
  (Experience replay.)
- Schaul, T., Horgan, D., Gregor, K. and Silver, D. (2015). Universal value
  function approximators. *ICML*.
- Schaul, T. et al. (2016). Prioritized experience replay. *ICLR*.
- Andrychowicz, M. et al. (2017). Hindsight experience replay. *NeurIPS*.
- Barreto, A. et al. (2017). Successor features for transfer in
  reinforcement learning. *NeurIPS*; extended JMLR 2018/2019 line.
- Fernández, F. and Veloso, M. (2006). Probabilistic policy reuse in a
  reinforcement learning agent. *AAMAS*.
- Taylor, M. E. and Stone, P. (2009). Transfer learning for reinforcement
  learning domains: a survey. *JMLR*, 10:1633–1685.
- Duan, Y. et al. (2016). RL²: Fast reinforcement learning via slow
  reinforcement learning. *NeurIPS*. (Meta-RL cost contrast.)
- Cheung, W. C., Simchi-Levi, D. and Zhu, R. (2020). Reinforcement learning
  for non-stationary Markov decision processes. *NeurIPS*.
- Colas, C., Karch, T., Sigaud, O. and Oudeyer, P.-Y. (2022). Intrinsically
  motivated goal-conditioned reinforcement learning: a short survey. *JAIR*,
  74.
- Liu, M. et al. (2022). Goal-conditioned reinforcement learning: problems
  and solutions. arXiv:2201.08299. (Survey.)
- Zhu, Z. et al. (2023). Transfer learning in deep reinforcement learning: a
  survey. *Neural Networks*. (Deep-transfer extension of Taylor & Stone.)
- Mnih, V. et al. (2015). Human-level control through deep reinforcement
  learning. *Nature*. (Replay in DQN.)
- Discrepancy-based replay for non-stationary environments (DEER / DoE),
  arXiv:2509.15032 (2025). (Closest related: discrepancy prioritisation for
  deep off-policy agents; ours differs in tabular goal-conditioned regime
  with localised reset.)
