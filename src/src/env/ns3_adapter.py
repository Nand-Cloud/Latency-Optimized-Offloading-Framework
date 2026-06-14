"""NS-3 binding stub — implements the same :class:`OffloadingEnv` interface.

The intended design couples NS-3 (v3.38) to the Python agent over a file-based /
memory-mapped IPC channel: NS-3 writes the current network state to a shared
region each 50 ms decision epoch, Python writes back the routing decision, and
NS-3 advances the simulation. Because this adapter satisfies the exact same
interface as :class:`~src.env.sim_env.SimEnv`, it can be swapped in without
changing any agent, trainer, or experiment code.

This is a stub: NS-3 is not installed in this environment, so the methods raise
``NotImplementedError`` with notes on what the real binding must do. The
pure-Python :class:`SimEnv` is the active environment for all experiments.
"""
from __future__ import annotations

import numpy as np

from ..common.config import Config
from .interface import OffloadingEnv, Spaces, StepResult


class NS3Env(OffloadingEnv):
    def __init__(self, cfg: Config, weight: tuple[float, float] | None = None,
                 ipc_dir: str = "ns3_ipc"):
        self.cfg = cfg
        self.weight = weight or cfg.default_weight
        self.ipc_dir = ipc_dir
        self.spaces = Spaces(
            state_dim=cfg.agent.state_dim,
            n_tiers=cfg.agent.n_tiers,
            n_resource_levels=cfg.agent.n_resource_levels,
        )

    def reset(self, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError(
            "NS-3 binding not available. The real implementation would launch the "
            "NS-3 v3.38 scenario, open the shared memory-mapped IPC region in "
            f"'{self.ipc_dir}', and read the initial 6-dim state written by NS-3. "
            "Use SimEnv for pure-Python experiments."
        )

    def step(self, action) -> StepResult:
        raise NotImplementedError(
            "NS-3 binding not available. The real implementation would write the "
            "(tier, resource) decision to the IPC region, signal NS-3 to advance one "
            "50 ms epoch, then read back the resulting state, per-task latency and "
            "energy, and assemble a StepResult identical in shape to SimEnv's."
        )
