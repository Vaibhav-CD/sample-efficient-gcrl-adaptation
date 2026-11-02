# Sample-Efficient Adaptation in Goal-Conditioned RL under Environment Changes

Research prototype investigating whether an already-trained tabular agent can
reuse goal-reaching knowledge to adapt to environmental changes with fewer
additional interactions.

## Project status

- [x] **STAGE 1** — Literature review and research gap (this document set).
- [x] **STAGE 2** — Environment development ([notes](docs/stage2_environment.md)).
- [x] **STAGE 3** — Baseline algorithms ([notes](docs/stage3_baselines.md)).
- [x] **STAGE 4** — Knowledge reuse and adaptation mechanisms ([notes](docs/stage4_adaptation.md)).
- [x] **STAGE 5** — Proposed algorithm: CASR ([notes](docs/stage5_proposed_method.md)).
- [x] **STAGE 6** — Experimental evaluation ([notes](docs/stage6_evaluation.md), [data](docs/results_stage6.csv)).
- [x] **STAGE 7** — Visualization dashboard ([notes](docs/stage7_dashboard.md), run `streamlit run app.py`).
- [x] **STAGE 8** — Documentation and research report ([report](docs/research_report.md)).
- [x] **STAGE 9** — Presentation ([outline](docs/presentation.md)).

> Rule: work stops after each stage for explicit user approval.
> Do not start Stage 2 until approved.

## Stage 1 deliverables

- [Literature review and research gap](docs/stage1_literature_review.md) — surveys,
  comparison table, gap, math formulation, hypothesis, experimental plan.

## Planned architecture (not yet built)

```text
project/
    app.py
    environment/
    algorithms/
    experiments/
    visualization/
    utils/
    tests/
    docs/
    requirements.txt
    README.md
```

The skeleton is intentional: Stage 1 delivers documents only, no algorithm code,
per the "do not begin full implementation" rule.

## Reproducibility and honesty rules

- Tabular NumPy methods first; no deep networks without justification.
- Every adaptation claim must be backed by seeded experiments.
- Planning baselines (BFS/Dijkstra, full map access) are never compared as
  if they learned through interaction.
- Illustrative numbers are always labelled illustrative, never results.

## Requirements

See [requirements.txt](requirements.txt). CPU-only, pinned minimum versions.
