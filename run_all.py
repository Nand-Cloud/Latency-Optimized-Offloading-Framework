"""Single entry point: regenerate every table and figure the paper needs.

Usage
-----
    python run_all.py              # full reproduction (30 seeds x 500 episodes)
    python run_all.py --quick      # fast smoke run (few seeds/episodes)
    python run_all.py --figures-only   # rebuild tables/figures from existing CSVs

Stages: seed sweep -> weight grid -> scalability -> tables -> figures.
All artifacts land in results/ (CSV) and figures/ (PNG + PDF).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

# Ensure the repo root is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analysis import figures, tables  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.experiments import run_seeds, scalability, weight_grid  # noqa: E402

log = get_logger("run_all")


def _read_if_exists(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce all experiments.")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run with reduced seeds/episodes")
    ap.add_argument("--figures-only", action="store_true",
                    help="rebuild tables/figures from existing raw CSVs only")
    ap.add_argument("--device", default="cpu", help="torch device (cpu/cuda)")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--skip-weight-grid", action="store_true")
    ap.add_argument("--skip-scalability", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    raw = cfg.abspath(cfg.paths.raw_dir)

    if args.figures_only:
        log.info("figures-only: rebuilding from existing CSVs in %s", raw)
        tasks = _read_if_exists(os.path.join(raw, "seed_sweep_tasks.csv"))
        conv = _read_if_exists(os.path.join(raw, "convergence.csv"))
        grid = _read_if_exists(os.path.join(raw, "weight_grid_tasks.csv"))
        scal = _read_if_exists(os.path.join(raw, "scalability_tasks.csv"))
        if tasks is not None:
            tables.build_all(cfg, tasks)
        figures.build_all(cfg, conv, scal, grid)
        log.info("figures-only complete.")
        return

    t0 = time.time()
    log.info("=== Stage 1/5: seed sweep ===")
    tasks, conv = run_seeds.run(cfg, quick=args.quick, device=args.device)

    grid = None
    if not args.skip_weight_grid:
        log.info("=== Stage 2/5: weight grid (Pareto) ===")
        grid = weight_grid.run(cfg, quick=args.quick, device=args.device)

    scal = None
    if not args.skip_scalability:
        log.info("=== Stage 3/5: scalability sweep ===")
        scal = scalability.run(cfg, quick=args.quick, device=args.device)

    log.info("=== Stage 4/5: tables ===")
    tbls = tables.build_all(cfg, tasks)
    for name, df in tbls.items():
        log.info("table %s:\n%s", name, df.to_string(index=False))

    log.info("=== Stage 5/5: figures ===")
    figures.build_all(cfg, conv, scal, grid)

    log.info("All done in %.1fs. Tables -> %s, Figures -> %s",
             time.time() - t0,
             cfg.abspath(cfg.paths.tables_dir), cfg.abspath(cfg.paths.figures_dir))


if __name__ == "__main__":
    main()