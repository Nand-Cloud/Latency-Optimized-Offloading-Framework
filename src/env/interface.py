"""Gym-style environment interface — the single swap point for NS-3.

Both the pure-Python discrete-event simulator (:mod:`src.env.sim_env`) and the
future NS-3 binding (:mod:`src.env.ns3_adapter`) implement :class:`OffloadingEnv`.
Agents depend only on this interface, never on a concrete simulator, so NS-3 can
be dropped in later without touching any agent code.

Action convention
-----------------
The action is a 2-tuple ``(tier, resource)`` reflecting the two-level hierarchy:

* ``tier``     -- high-level decision in ``{0:Edge, 1:Fog, 2:Cloud}``
* ``resource`` -- low-level intra-tier resource level in ``range(n_resource_levels)``

Flat agents that treat the action as a single discrete choice can use
:func:`encode_action` / :func:`decode_action` to map between the composite
``(tier, resource)`` and a flat index in ``range(n_tiers * n_resource_levels)``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

# Tier indices (high-level action space).
TIER_EDGE, TIER_FOG, TIER_CLOUD = 0, 1, 2
TIER_NAMES = ["Edge", "Fog", "Cloud"]

Action = Tuple[int, int]  # (tier, resource_level)


@dataclass
class Spaces:
    """Lightweight description of observation/action dimensions."""

    state_dim: int
    n_tiers: int
    n_resource_levels: int

    @property
    def flat_action_dim(self) -> int:
        return self.n_tiers * self.n_resource_levels


def encode_action(tier: int, resource: int, n_resource_levels: int) -> int:
    """Map a ``(tier, resource)`` pair to a flat discrete index."""
    return tier * n_resource_levels + resource


def decode_action(flat: int, n_resource_levels: int) -> Action:
    """Inverse of :func:`encode_action`."""
    return flat // n_resource_levels, flat % n_resource_levels


@dataclass
class StepResult:
    """Per-step outcome. ``info`` carries the raw per-task metrics that the
    experiment logger records (latency, energy, chosen tier, deadline hit)."""

    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class OffloadingEnv(ABC):
    """Abstract single-agent MDP for Edge–Fog–Cloud task offloading."""

    spaces: Spaces

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the simulator and return the initial 6-dim state."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Apply ``action`` to the current task and advance one decision epoch."""

    def close(self) -> None:  # optional override
        pass
