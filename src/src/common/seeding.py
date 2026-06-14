"""Deterministic seeding across Python, NumPy and PyTorch.

Call :func:`seed_everything` at the start of every run so results are
reproducible for a given seed.
"""
from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # torch is optional for pure-env smoke tests
    torch = None


def seed_everything(seed: int) -> np.random.Generator:
    """Seed all RNGs and return a NumPy Generator for env-local randomness."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
