"""Common agent API shared by the proposed method and all baselines.

Every agent — learning or fixed-policy — implements this interface so the
trainer and experiment drivers treat them uniformly and they all see the same
state, action, and reward signal (fairness requirement from the spec).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ..env.interface import Action


class Agent(ABC):
    name: str = "agent"
    #: Whether the trainer should run a learning update / epsilon decay loop.
    learns: bool = True

    @abstractmethod
    def act(self, state: np.ndarray, explore: bool = True) -> Action:
        """Return a ``(tier, resource)`` action for ``state``."""

    def observe(self, state: np.ndarray, action: Action, reward: float,
                next_state: np.ndarray, done: bool) -> None:
        """Record a transition (no-op for non-learning agents)."""

    def update(self) -> float | None:
        """Perform a learning step; return a loss value if one occurred."""
        return None

    def end_episode(self) -> None:
        """Hook for per-episode bookkeeping (e.g. epsilon decay)."""

    # -- persistence (optional) -------------------------------------------
    def save(self, path: str) -> None:  # pragma: no cover - optional
        pass

    def load(self, path: str) -> None:  # pragma: no cover - optional
        pass


class FixedPolicyAgent(Agent):
    """Base for non-learning baselines (cloud-only, rule-based)."""

    learns = False

    def __init__(self, n_resource_levels: int, default_resource: int | None = None):
        # Default to the highest resource level (fastest) unless overridden.
        self.default_resource = (
            default_resource if default_resource is not None else n_resource_levels - 1
        )
