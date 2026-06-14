"""Seed sweep: every method over all seeds at the default weight.

Produces the raw per-task logs behind the headline latency, joint
latency–energy, and task-distribution tables, plus the per-episode convergence
curves. This is the core comparison experiment.
"""
from __future__ import annotations

import os

import pandas as pd

from ..agents.registry import ALL_AGENTS
from ..common.config import Config
from ..common.logging_utils import get_logger
from ..training.runner import run_config

log = get_logger("run_seeds")


def run(cfg: Config, quick: bool = False, device: str = "cpu",
        agents: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    agents = agents or ALL_AGENTS
    seeds = cfg.seeds_for(quick)
    weight = cfg.default_weight
    eval_eps = 3 if quick else 5

    task_rows, conv_rows = [], []
    total = len(agents) * len(seeds)
    done = 0
    for agent_name in agents:
        for seed in seeds:
            out = run_config(cfg, agent_name, seed, weight, quick=quick,
                             eval_episodes=eval_eps, device=device)
            task_rows.extend(out.records)
            conv_rows.extend(out.convergence)
            done += 1
            log.info("seed-sweep %s seed=%d (%d/%d)", agent_name, seed, done, total)

    tasks = pd.DataFrame(task_rows)
    conv = pd.DataFrame(conv_rows)

    raw = cfg.abspath(cfg.paths.raw_dir)
    os.makedirs(raw, exist_ok=True)
    tasks.to_csv(os.path.join(raw, "seed_sweep_tasks.csv"), index=False)
    conv.to_csv(os.path.join(raw, "convergence.csv"), index=False)
    log.info("seed sweep complete: %d task rows, %d convergence rows",
             len(tasks), len(conv))
    return tasks, conv
