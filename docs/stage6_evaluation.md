# STAGE 6 — Experimental Evaluation: what actually improved (5 seeds)

**Status:** complete, awaiting approval. Raw data: `docs/results_stage6.csv`;
curves: `docs/eval_curves_stage6.png`. Run: `python -m experiments.evaluation`.
All values below are mean ± std over seeds 0–4 (N_0.9 over seeds that reached
90%; `reached` column counts them).

## 1. Experiment A — initial learning

Goal-conditioned Q, 4000 episodes: N_0.9 = 135 ± 7, greedy success 1.00 ± 0.00,
mean length 6.9, 0.6 s. Stable starting point for every adaptation run.

## 2. Experiment B — blocked passage (the discriminative test)

| Method | N_0.9 | Final success | Reached |
|---|---|---|---|
| combo (oracle) | 143 ± 54 | 1.00 | 5/5 |
| oracle-selective | 165 ± 56 | 1.00 | 5/5 |
| scratch | 180 ± 11 | 1.00 | 5/5 |
| CASR | 210 ± 31 | 1.00 | 5/5 |
| CASR-no-reset | 238 ± 45 | 1.00 | 5/5 |
| warm | 279 ± 23 | 1.00 | 5/5 |
| warm+replay | 1037 ± 542 | 0.77 ± 0.09 | 2/5 |
| CASR-no-filter | 1477 (n=1) | 0.84 ± 0.09 | 1/5 |

Findings: stale values cause robust negative transfer — warm start (279) is
worse than scratch (180), and unfiltered replay collapses (2/5 reach 90%).
Oracle selective reset + filtering wins (143). CASR (210) beats warm and both
replay-poisoned variants but **loses to scratch** — detection overhead (30
probe episodes + imperfect region) exceeds the reset gain here.

## 3. Experiments C (slip → 0.2) and D (unseen goals) — floor effects

In C every reuse method sits at N = 100 (the window floor): the trained
policy is already robust to mild slip, so there is almost nothing to adapt
to — and CASR's probes become pure overhead (256). In D all methods land
between 100 and 139: two retained slices transfer, two nearby new slices
learn fast. Honest verdict: C and D as designed cannot discriminate methods.
They do contribute one real caveat: **when little adaptation is needed,
CASR's fixed probe budget is the most expensive option.** Stronger variants
(higher slip, distant goals) belong to future work, not to this report's
claims.

## 4. Experiment E — combined change

combo 108 ± 6, oracle 143, warm+replay 152 (0.90), warm 167 ± 47, CASR
176 ± 19, scratch 181 ± 18, no-filter 192 (0.90), no-reset 204 ± 40. Same
ordering as B, compressed: oracle combination best, CASR mid-pack at scratch
level, all at success ≥ 0.90 except the unfiltered variants.

## 5. Ablation (from B and E)

The replay filter is load-bearing: removing it (CASR-no-filter) reproduces
the poisoning collapse (B: 1/5 reach). The reset helps modestly (CASR 210 vs
no-reset 238 on B; 176 vs 204 on E). So both CASR components earn their keep,
but the probe cost (30 episodes) is not yet repaid by region quality.

## 6. Hypothesis verdict (H1 as stated: FAILS, and that is the finding)

H1 required CASR to beat *both* retraining and warm start on
time-to-threshold. It beats warm start on B (210 < 279) but not retraining
(210 > 180). The mechanism is validated — oracle selective reset + filtering
is the best method in B and E — but autonomous probe-based detection does not
yet pay for itself on this grid scale. The correct revision is not a bigger
claim but a cheaper detector: step-budgeted (not episode-budgeted) probes,
or probe-while-adapting instead of probe-then-adapt. No result was re-tuned
after seeing these numbers.

## 7. Costs and reproducibility

Adapt wall-clock per method: 0.2–1.2 s on laptop CPU (replay-heavy variants
cost more). Seeds 0–4, frozen protocol in `experiments/evaluation.py`, full
per-seed rows in `docs/results_stage6.csv`. Limitation: 5 seeds, not the
planned 10; single grid size; C/D too easy (see §3).

## 8. Files and tests — 46/46 pass

`experiments/evaluation.py`, `tests/test_evaluation.py` (protocol smoke
test), `docs/results_stage6.csv`, `docs/eval_curves_stage6.png`, this note.
