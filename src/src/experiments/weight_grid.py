"""Weight-grid sweep over (w_L, w_E) for the latency–energy Pareto front.

Each method is trained and evaluated at every weighting in the grid and over all
seeds, so the trade-off curve (and its variability) can be plotted per method.
"""
from __future__ import annotations

import os

import pandas as pd

from ..agents.registry import ALL_AGENTS
from ..common.config import Config
from ..common.logging_utils import get_logger
from ..training.runner import run_config

log = get_logger("weight_grid")


def run(cfg: Config, quick: bool = False, device: str = "cpu",
        agents: list[str] | None = None) -> pd.DataFrame:
    agents = agents or ALL_AGENTS
    seeds = cfg.seeds_for(quick)
    eval_eps = 3 if quick else 5

    rows = []
    total = len(agents) * len(cfg.weight_grid) * len(seeds)
    done = 0
    for agent_name in agents:
        for weight in cfg.weight_grid:
            for seed in seeds:
                out = run_config(cfg, agent_name, seed, weight, quick=quick,
                                 eval_episodes=eval_eps, device=device)
                rows.extend(out.records)
                done += 1
                log.info("weight-grid %s w=%s seed=%d (%d/%d)",
                         agent_name, weight, seed, done, total)

    tasks = pd.DataFrame(rows)
    raw = cfg.abspath(cfg.paths.raw_dir)
    os.makedirs(raw, exist_ok=True)
    tasks.to_csv(os.path.join(raw, "weight_grid_tasks.csv"), index=False)
    log.info("weight grid complete: %d task rows", len(tasks))
    return tasks
