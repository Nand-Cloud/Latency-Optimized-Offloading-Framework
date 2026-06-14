# Deep Reinforcement Learning-Based Adaptive Task Offloading in Edge – Fog – Cloud Environments for Latency Optimization

This repository implements a simulation + DRL training pipeline for benchmarking a hierarchical multi-objective Double-DQN offloading agent against multiple baselines. It produces CSV tables and publication-quality figures.

## Requirements

- Python 3.10+ (tested with Python 3.11)
- PyTorch 2.12.x (used by the DRL agents)
- Optional: NS-3 integration is stubbed behind the same environment interface.

Install dependencies:

```bat
pip install -r requirements.txt
```

## Project Layout

- `src/env/` — Gym-style environment interface and a pure-Python discrete-event simulator (`SimEnv`). NS-3 adapter stub lives behind the same interface.
- `src/agents/` — Agents and baselines (hierarchical DDQN, flat DQN variants, A3C, PPO, tabular Q, rule-based, cloud-only).
- `src/training/` — Training loop.
- `src/experiments/` — Experiment drivers (seed sweep, weight grid / Pareto, scalability sweep).
- `src/analysis/` — Statistics, table generation, and figure generation.

## Configuration

Edit `config.yaml` to change:

- topology (IoT/edge/fog/cloud counts, bandwidth, delays, capacities)
- workload model (Poisson arrival rate, task data sizes, cycles, deadline)
- reward model coefficients
- DRL hyperparameters
- which seeds to run

## How to Run

### 1) Run unit tests (smoke)

```bat
python -m pytest -q
```

### 2) Quick run (fast smoke)

Runs the seed sweep and generates the main tables/figures.

```bat
python run_all.py --quick --skip-weight-grid --skip-scalability
```

### 3) Figures-only

Rebuilds tables/figures from existing CSVs in `results/raw/`.

```bat
python run_all.py --figures-only
```

### 4) Full run

Full reproduction (may be slow):

```bat
python run_all.py
```

## Outputs

- Raw per-task/per-episode data:
  - `results/raw/*.csv`
- Aggregated paper tables:
  - `results/tables/*.csv`
  - `results/tables/significance_tests.csv`
- Figures (PNG + PDF):
  - `figures/*.png`
  - `figures/*.pdf`

## Notes on NS-3

NS-3 integration is isolated behind `src/env/ns3_adapter.py`. If NS-3 is available later, swap the adapter while keeping the same `OffloadingEnv` interface.


#
