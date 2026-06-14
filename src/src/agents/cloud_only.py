"""Cloud-Only baseline: every admitted task is offloaded to the cloud tier."""
from __future__ import annotations

import numpy as np

from ..common.config import Config
from ..env.interface import Action, TIER_CLOUD
from .base import FixedPolicyAgent


class CloudOnly(FixedPolicyAgent):
    name = "Cloud-Only"

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None,
                 device: str = "cpu"):
        super().__init__(cfg.agent.n_resource_levels)

    def act(self, state: np.ndarray, explore: bool = True) -> Action:
        return TIER_CLOUD, self.default_resource
