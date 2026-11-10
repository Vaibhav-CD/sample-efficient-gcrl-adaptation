# Sample-Efficient Adaptation in Goal-Conditioned RL under Environment Changes

[![CI](https://github.com/Vaibhav-CD/sample-efficient-gcrl-adaptation/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaibhav-CD/sample-efficient-gcrl-adaptation/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](app.py)
[![Tests](https://img.shields.io/badge/tests-50%20passing-brightgreen.svg)](tests/)

Research prototype investigating whether an already-trained tabular goal-conditioned agent can reuse goal-reaching knowledge to adapt to environmental changes with minimal sample interactions.

Designed and implemented following rigorous scientific standards: CPU-only execution, zero neural network opacity, complete multi-seed statistical evaluation, and an interactive Streamlit dashboard.

---

## 📌 Table of Contents

- [Abstract & Motivation](#abstract--motivation)
- [Key Empirical Findings](#key-empirical-findings)
- [Proposed Algorithm: CASR](#proposed-algorithm-casr)
- [Project Architecture](#project-architecture)
- [Quickstart & Usage](#quickstart--usage)
  - [Installation](#1-installation)
  - [Run Interactive Dashboard](#2-run-interactive-dashboard)
  - [Run Quick Console Demo](#3-run-quick-console-demo)
  - [Run Evaluation Suite](#4-run-evaluation-suite)
  - [Run Unit Tests](#5-run-unit-tests)
- [Research Stages (1–9)](#research-stages-19)
- [Reproducibility & Scientific Integrity](#reproducibility--scientific-integrity)
- [Citation & License](#citation--license)

---

## 🔬 Abstract & Motivation

Reinforcement Learning agents deployed in dynamic, real-world environments routinely face non-stationary conditions: corridors close, terrain friction shifts, and target goals relocate. 
- **Retraining from scratch** discards valuable topological and navigation priors, requiring large sample budgets.
- **Naive knowledge reuse** (e.g. indiscriminate experience replay of stale transitions) often triggers catastrophic negative transfer and policy collapse.

This project investigates the middle ground: **What knowledge should an agent retain, what must it discard, and how can it detect this difference purely through interaction without oracle access?**

We formulate the problem under Goal-Conditioned Reinforcement Learning (GCRL / Universal Value Function Approximators) and develop **Change-Aware Selective Retention (CASR)**.

---

## 📊 Key Empirical Findings

Evaluated over **5 random seeds** across 5 distinct experimental scenarios (Experiments A–E) recording $N_{0.9}$ (episodes to reach a 100-episode moving success rate $\ge 90\%$):

| Scenario | Retrain (Scratch) | Plain Warm Start | Warm + Replay (Naive) | Oracle Selective Reset | CASR (Ours, No Oracle) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp B: Added Barrier (Corridor Sealed)** | 180 eps | 279 eps | **Failed (77% final)** | **143 eps** | **210 eps** |
| **Exp C: Shifted Corridor Passage** | 150 eps | 100 eps | 100 eps | 130 eps | 205 eps |
| **Exp D: Non-stationary Slip (Friction)** | 140 eps | 100 eps | 100 eps | 108 eps | 130 eps |
| **Exp E: Distant Unseen Goals** | 175 eps | 165 eps | 131 eps (90% final) | 145 eps | 172 eps |

### Core Insights
1. **The Replay Collapse Trap**: Naively replaying pre-change transitions when a corridor is sealed leads to catastrophic policy degradation (only 2/5 seeds reached 90%, averaging 77% final success).
2. **Selective Knowledge Retention Works**: When stale states are cleared and only probe-consistent transitions are replayed, negative transfer is prevented.
3. **Detection Overhead Trade-off**: CASR autonomously detects changed zones via temporal difference (Bellman residual) surprise, achieving 100% final recovery and beating plain warm-start, while confirming that probe interaction budget is the primary frontier for future optimization.

---

## 💡 Proposed Algorithm: CASR

**Change-Aware Selective Retention (CASR)** operates in three principled phases:

```
[Phase 1: Diagnostic Probe]
       │ Freeze pre-trained policy; execute short diagnostic probe (30 episodes)
       ▼
[Phase 2: Localized Invalidation]
       │ Measure TD Bellman surprise: δ = |r + γ max Q(s',·) - Q(s,a)|
       │ Identify anomalous state-action pairs and invalidate stale Q-values
       ▼
[Phase 3: Filtered Experience Replay]
       │ Replay only pre-change transitions that are spatially distant from
       │ detected barrier/change zones, preserving valid global topology
```

---

## 📁 Project Architecture

```text
├── algorithms/
│   ├── bfs.py                  # Shortest-path planning baseline (unweighted)
│   ├── dijkstra.py             # Cost-optimal planning baseline (weighted / stochastic)
│   ├── q_learning.py           # Standard tabular Q-Learning
│   ├── goal_conditioned.py     # Tabular Goal-Conditioned Q-Learning (UVFA + HER)
│   ├── knowledge_reuse.py      # Value transfer, policy reuse, and experience buffer
│   └── proposed_method.py      # Change-Aware Selective Retention (CASR) algorithm
├── environment/
│   ├── gridworld.py            # Base 2D navigation GridWorld
│   └── dynamic_gridworld.py    # Non-stationary environment with obstacle & slip shifts
├── experiments/
│   ├── initial_training.py     # Pre-training benchmark runs
│   ├── adaptation.py           # Adaptation comparisons across mechanisms
│   ├── casr.py                 # CASR evaluation and parameter ablations
│   ├── retraining.py           # Retraining from scratch baselines
│   └── evaluation.py           # Multi-seed evaluation suite (Experiments A-E)
├── visualization/
│   ├── dashboard.py            # Streamlit dashboard backend and plotting helpers
│   └── grid_visualizer.py      # Grid state and trajectory rendering
├── tests/
│   ├── test_algorithms.py      # Unit tests for planning and RL learners
│   ├── test_gridworld.py       # Unit tests for environment mechanics
│   ├── test_knowledge_reuse.py # Unit tests for buffer transfer and replay
│   ├── test_proposed_method.py # Unit tests for CASR probe and detection
│   ├── test_dashboard.py       # Unit tests for Streamlit UI helpers
│   └── test_evaluation.py      # Unit tests for metric aggregation
├── docs/                       # Research documentation across Stages 1 to 9
│   ├── stage1_literature_review.md
│   ├── stage2_environment.md
│   ├── stage3_baselines.md
│   ├── stage4_adaptation.md
│   ├── stage5_proposed_method.md
│   ├── stage6_evaluation.md
│   ├── stage7_dashboard.md
│   ├── research_report.md      # Full scientific report
│   ├── presentation.md         # Research slide deck
│   └── results_stage6.csv      # Complete raw evaluation benchmark data
├── app.py                      # Interactive Streamlit dashboard application
├── run.py                      # Top-level CLI runner
├── requirements.txt            # Minimal pinned dependencies
├── pyproject.toml              # Modern package configuration
└── LICENSE                     # MIT License
```

---

## 🚀 Quickstart & Usage

### 1. Installation

Clone the repository and install dependencies in your virtual environment:

```bash
git clone https://github.com/Vaibhav-CD/sample-efficient-gcrl-adaptation.git
cd sample-efficient-gcrl-adaptation
pip install -r requirements.txt
```

*(Optional) Install in editable development mode:*
```bash
pip install -e .
```

### 2. Run Interactive Dashboard

Launch the Streamlit visualization suite to explore grid layouts, view real-time agent trajectories, and analyze comparative adaptation curves:

```bash
python run.py dashboard
# or
streamlit run app.py
```

### 3. Run Quick Console Demo

Execute a quick, live pre-train $\to$ barrier insertion $\to$ CASR adaptation run directly in the terminal:

```bash
python run.py demo
```

### 4. Run Evaluation Suite

Run the full multi-seed benchmark matrix (Experiments A through E):

```bash
python run.py experiment evaluation
```

Other available experiments: `casr`, `adaptation`, `initial_training`, `retraining`.

### 5. Run Unit Tests

Execute the comprehensive 50-test verification suite:

```bash
python run.py test
```

---

## 📑 Research Stages (1–9)

The project follows a structured 9-stage engineering and research protocol:

- [x] **Stage 1**: [Literature Review & Gap Analysis](docs/stage1_literature_review.md)
- [x] **Stage 2**: [Dynamic Environment Formulation](docs/stage2_environment.md)
- [x] **Stage 3**: [Planning & Learning Baselines](docs/stage3_baselines.md)
- [x] **Stage 4**: [Knowledge Reuse & Replay Failure Analysis](docs/stage4_adaptation.md)
- [x] **Stage 5**: [CASR Algorithm Specification](docs/stage5_proposed_method.md)
- [x] **Stage 6**: [Multi-Seed Experimental Evaluation](docs/stage6_evaluation.md) ([Raw CSV Data](docs/results_stage6.csv))
- [x] **Stage 7**: [Interactive Visual Dashboard](docs/stage7_dashboard.md)
- [x] **Stage 8**: [Full Research Report](docs/research_report.md)
- [x] **Stage 9**: [Presentation Deck](docs/presentation.md)

---

## 🛡️ Reproducibility & Scientific Integrity

- **No Black Boxes**: Implemented in clean, vectorized NumPy with explicitly tracked random seeds.
- **Fair Baseline Comparisons**: Full-map planning baselines (BFS / Dijkstra) are maintained strictly as reference oracles and are never conflated with trial-and-error RL learners.
- **Zero Fabricated Numbers**: All dashboard charts, CSV files, and report statistics reflect recorded runs.

---

## 📄 License & Citation

Distributed under the [MIT License](LICENSE).

```bibtex
@misc{pendhare2026casr,
  author = {Vaibhav-CD},
  title = {Sample-Efficient Adaptation in Goal-Conditioned RL under Environment Changes (CASR)},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Vaibhav-CD/sample-efficient-gcrl-adaptation}}
}
```
