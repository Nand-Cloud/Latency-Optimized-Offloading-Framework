"""Rule-Based baseline: static thresholds on the observed state.

A simple, fixed heuristic of the kind operators deploy without learning:
route to the Edge while it is lightly loaded and the payload is small, escalate
to the Fog under moderate load, and fall back to the Cloud when congested. Uses
the state features (CPU utilization and packet size) directly.
"""
from __future__ import annotations

import numpy as np

from ..common.config import Config
from ..env.interface import Action, TIER_CLOUD, TIER_EDGE, TIER_FOG
from .base import FixedPolicyAgent

# State layout: [avail_bw, queue_len, last_delay, packet_size, cpu_util, energy_ctx]
IDX_PACKET = 3
IDX_UTIL = 4


class RuleBased(FixedPolicyAgent):
    name = "Rule-Based"

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None,
                 device: str = "cpu", util_low: float = 0.4, util_high: float = 0.75,
                 packet_small: float = 0.3):
        super().__init__(cfg.agent.n_resource_levels)
        self.util_low = util_low
        self.util_high = util_high
        self.packet_small = packet_small

    def act(self, state: np.ndarray, explore: bool = True) -> Action:
        util = float(state[IDX_UTIL])
        packet = float(state[IDX_PACKET])
        if util <= self.util_low and packet <= self.packet_small:
            tier = TIER_EDGE
        elif util <= self.util_high:
            tier = TIER_FOG
        else:
            tier = TIER_CLOUD
        return tier, self.default_resource
