"""Publication-quality matplotlib figures (PNG + PDF) from the raw logs.

Figures produced:
* ``convergence``  -- per-episode training reward per method
* ``scalability``  -- latency & energy vs IoT node count and arrival rate
* ``pareto``       -- latency–energy Pareto front per method across the weight grid
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless / file output only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..agents.registry import ALL_AGENTS  # noqa: E402
from ..common.config import Config  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _ordered(df: pd.DataFrame) -> list[str]:
    present = set(df["agent"].unique())
    return [a for a in ALL_AGENTS if a in present]


def _save(fig, cfg: Config, name: str) -> None:
    fdir = cfg.abspath(cfg.paths.figures_dir)
    os.makedirs(fdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_convergence(cfg: Config, conv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent in _ordered(conv):
        sub = conv[conv["agent"] == agent].groupby("episode")["reward"].mean()
        ax.plot(sub.index, sub.values, label=agent, linewidth=1.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean reward per step")
    ax.set_title("Training convergence")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, cfg, "convergence")


def plot_scalability(cfg: Config, scal: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    sweeps = [
        ("nodes", "iot_nodes", "IoT node count"),
        ("arrival_rate", "arrival_rate", "Arrival rate (tasks/s)"),
    ]
    for col, (sweep, xcol, xlabel) in enumerate(sweeps):
        sub = scal[scal["sweep"] == sweep]
        for row, (metric, ylabel) in enumerate(
            [("latency_ms", "Avg latency (ms)"), ("energy_j", "Avg energy (J)")]
        ):
            ax = axes[row][col]
            for agent in _ordered(sub):
                g = sub[sub["agent"] == agent].groupby(xcol)[metric].mean()
                ax.plot(g.index, g.values, marker="o", markersize=3,
                        label=agent, linewidth=1.4)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Scalability: latency & energy vs offered load")
    _save(fig, cfg, "scalability")


def plot_pareto(cfg: Config, grid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for agent in _ordered(grid):
        sub = grid[grid["agent"] == agent]
        pts = (
            sub.groupby(["w_latency", "w_energy"])[["latency_ms", "energy_j"]]
            .mean()
            .reset_index()
            .sort_values("latency_ms")
        )
        ax.plot(pts["latency_ms"], pts["energy_j"], marker="o", markersize=5,
                label=agent, linewidth=1.4)
    ax.set_xlabel("Avg latency (ms)")
    ax.set_ylabel("Avg energy (J)")
    ax.set_title("Latency–energy Pareto front (weight grid)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, cfg, "pareto")


def build_all(cfg: Config, conv: pd.DataFrame | None,
              scal: pd.DataFrame | None, grid: pd.DataFrame | None) -> None:
    if conv is not None and not conv.empty:
        plot_convergence(cfg, conv)
    if scal is not None and not scal.empty:
        plot_scalability(cfg, scal)
    if grid is not None and not grid.empty:
        plot_pareto(cfg, grid)
