"""Build the paper's tables (CSV) from the raw per-task logs.

Tables produced:
* ``latency_table.csv``       -- avg / peak / std / % reduction with 95% CIs
* ``latency_energy_table.csv``-- joint latency & energy with 95% CIs
* ``task_distribution.csv``   -- fraction of tasks routed to each tier
* ``significance_tests.csv``  -- proposed vs each baseline, paired p-values
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..agents.registry import ALL_AGENTS, PROPOSED
from ..common.config import Config
from ..env.interface import TIER_NAMES
from .stats import mean_ci, paired_test

REFERENCE = "Cloud-Only"  # baseline against which % latency reduction is reported


def _ordered_agents(df: pd.DataFrame) -> list[str]:
    present = set(df["agent"].unique())
    return [a for a in ALL_AGENTS if a in present]


def _per_seed_means(df: pd.DataFrame, metric: str) -> dict[str, np.ndarray]:
    """Mean of ``metric`` per (agent, seed) -> array per agent, seed-aligned."""
    grouped = df.groupby(["agent", "seed"])[metric].mean().reset_index()
    out = {}
    for agent in _ordered_agents(df):
        sub = grouped[grouped["agent"] == agent].sort_values("seed")
        out[agent] = sub[metric].to_numpy()
    return out


def build_latency_table(tasks: pd.DataFrame) -> pd.DataFrame:
    per_seed = _per_seed_means(tasks, "latency_ms")
    ref_mean = float(np.mean(per_seed[REFERENCE])) if REFERENCE in per_seed else np.nan
    rows = []
    for agent in _ordered_agents(tasks):
        seed_means = per_seed[agent]
        ci = mean_ci(seed_means)
        all_lat = tasks[tasks["agent"] == agent]["latency_ms"].to_numpy()
        reduction = (
            100.0 * (ref_mean - ci.mean) / ref_mean if ref_mean and not np.isnan(ref_mean)
            else np.nan
        )
        rows.append({
            "agent": agent,
            "avg_latency_ms": round(ci.mean, 3),
            "ci95_low": round(ci.lo, 3),
            "ci95_high": round(ci.hi, 3),
            "peak_latency_ms": round(float(all_lat.max()), 3),
            "std_latency_ms": round(float(all_lat.std()), 3),
            "reduction_vs_cloud_pct": round(reduction, 2) if not np.isnan(reduction) else np.nan,
        })
    return pd.DataFrame(rows)


def build_latency_energy_table(tasks: pd.DataFrame) -> pd.DataFrame:
    lat = _per_seed_means(tasks, "latency_ms")
    ene = _per_seed_means(tasks, "energy_j")
    rows = []
    for agent in _ordered_agents(tasks):
        lci = mean_ci(lat[agent])
        eci = mean_ci(ene[agent])
        rows.append({
            "agent": agent,
            "avg_latency_ms": round(lci.mean, 3),
            "latency_ci95": round(lci.half_width, 3),
            "avg_energy_j": round(eci.mean, 4),
            "energy_ci95": round(eci.half_width, 4),
        })
    return pd.DataFrame(rows)


def build_task_distribution(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent in _ordered_agents(tasks):
        sub = tasks[tasks["agent"] == agent]
        counts = sub["tier"].value_counts(normalize=True)
        rows.append({
            "agent": agent,
            **{f"{TIER_NAMES[t].lower()}_pct": round(100.0 * counts.get(t, 0.0), 2)
               for t in range(len(TIER_NAMES))},
        })
    return pd.DataFrame(rows)


def build_significance_table(tasks: pd.DataFrame) -> pd.DataFrame:
    lat = _per_seed_means(tasks, "latency_ms")
    ene = _per_seed_means(tasks, "energy_j")
    rows = []
    for agent in _ordered_agents(tasks):
        if agent == PROPOSED:
            continue
        lt = paired_test(lat[PROPOSED], lat[agent])
        et = paired_test(ene[PROPOSED], ene[agent])
        rows.append({
            "comparison": f"{PROPOSED} vs {agent}",
            "latency_mean_diff_ms": round(lt["mean_diff"], 3),
            "latency_t_pvalue": _fmt_p(lt["t_pvalue"]),
            "latency_wilcoxon_pvalue": _fmt_p(lt["wilcoxon_pvalue"]),
            "energy_mean_diff_j": round(et["mean_diff"], 4),
            "energy_t_pvalue": _fmt_p(et["t_pvalue"]),
            "energy_wilcoxon_pvalue": _fmt_p(et["wilcoxon_pvalue"]),
            "n_seeds": lt["n"],
        })
    return pd.DataFrame(rows)


def _fmt_p(p: float) -> float:
    return round(p, 6) if not np.isnan(p) else np.nan


def build_all(cfg: Config, tasks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {
        "latency_table": build_latency_table(tasks),
        "latency_energy_table": build_latency_energy_table(tasks),
        "task_distribution": build_task_distribution(tasks),
        "significance_tests": build_significance_table(tasks),
    }
    tdir = cfg.abspath(cfg.paths.tables_dir)
    os.makedirs(tdir, exist_ok=True)
    for name, df in out.items():
        df.to_csv(os.path.join(tdir, f"{name}.csv"), index=False)
    return out
