"""Scalability sweep: vary IoT node count and task arrival rate.

For each method, sweep the number of IoT nodes (with arrival rate fixed) and the
arrival rate (with node count fixed), at the default weight, to measure how
latency and energy scale with offered load.
"""
from __future__ import annotations

import os

import pandas as pd

from ..agents.registry import ALL_AGENTS
from ..common.config import Config
from ..common.logging_utils import get_logger
from ..training.runner import run_config

log = get_logger("scalability")


def run(cfg: Config, quick: bool = False, device: str = "cpu",
        agents: list[str] | None = None) -> pd.DataFrame:
    agents = agents or ALL_AGENTS
    seeds = cfg.seeds_for(quick)
    weight = cfg.default_weight
    eval_eps = 2 if quick else 4

    node_counts = cfg.scalability.node_counts[:3] if quick else cfg.scalability.node_counts
    arrival_rates = (
        cfg.scalability.arrival_rates[:3] if quick else cfg.scalability.arrival_rates
    )

    rows = []
    for agent_name in agents:
        # Sweep node count (arrival rate at default).
        for n in node_counts:
            for seed in seeds:
                out = run_config(cfg, agent_name, seed, weight, quick=quick,
                                 iot_nodes=n, eval_episodes=eval_eps, device=device)
                for r in out.records:
                    r["sweep"] = "nodes"
                rows.extend(out.records)
            log.info("scalability %s nodes=%d", agent_name, n)
        # Sweep arrival rate (node count at default).
        for rate in arrival_rates:
            for seed in seeds:
                out = run_config(cfg, agent_name, seed, weight, quick=quick,
                                 arrival_rate=rate, eval_episodes=eval_eps, device=device)
                for r in out.records:
                    r["sweep"] = "arrival_rate"
                rows.extend(out.records)
            log.info("scalability %s arrival_rate=%.1f", agent_name, rate)

    tasks = pd.DataFrame(rows)
    raw = cfg.abspath(cfg.paths.raw_dir)
    os.makedirs(raw, exist_ok=True)
    tasks.to_csv(os.path.join(raw, "scalability_tasks.csv"), index=False)
    log.info("scalability complete: %d task rows", len(tasks))
    return tasks
