"""Statistics helpers: confidence intervals and paired significance tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class CI:
    mean: float
    lo: float
    hi: float
    half_width: float


def mean_ci(values, confidence: float = 0.95) -> CI:
    """Mean with a two-sided t-based confidence interval."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    mean = float(a.mean()) if n else float("nan")
    if n < 2:
        return CI(mean, mean, mean, 0.0)
    sem = stats.sem(a)
    half = float(sem * stats.t.ppf((1 + confidence) / 2.0, n - 1))
    return CI(mean, mean - half, mean + half, half)


def paired_test(proposed, baseline) -> dict:
    """Paired two-sided significance test (per-seed paired samples).

    Returns the paired t-test p-value and the Wilcoxon signed-rank p-value as a
    non-parametric companion. Samples must be aligned (e.g., by seed).
    """
    a = np.asarray(proposed, dtype=float)
    b = np.asarray(baseline, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    out = {"n": n, "t_pvalue": float("nan"), "wilcoxon_pvalue": float("nan"),
           "mean_diff": float(np.mean(a - b)) if n else float("nan")}
    if n >= 2 and np.any(a - b != 0):
        out["t_pvalue"] = float(stats.ttest_rel(a, b).pvalue)
        try:
            out["wilcoxon_pvalue"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            pass  # e.g. all differences zero
    return out
