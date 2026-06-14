"""Topology construction and the Poisson task workload generator."""
from __future__ import annotations

from typing import List

import numpy as np

from ..common.config import Config
from .models import Task, Tier
from .interface import TIER_NAMES


def build_tiers(cfg: Config, rng: np.random.Generator) -> List[Tier]:
    """Create the [Edge, Fog, Cloud] tiers from config.

    Cloud link delay is variable (WAN); a value is sampled per build and the
    simulator re-samples it per task via :func:`sample_cloud_delay`.
    """
    t = cfg.topology
    edge = Tier(
        name=TIER_NAMES[0],
        bandwidth_mbps=t.edge.bandwidth_mbps,
        link_delay_ms=t.edge.link_delay_ms,
        cpu_freq_ghz=t.edge.cpu_freq_ghz,
        capacity=t.edge.capacity,
    )
    fog = Tier(
        name=TIER_NAMES[1],
        bandwidth_mbps=t.fog.bandwidth_mbps,
        link_delay_ms=t.fog.link_delay_ms,
        cpu_freq_ghz=t.fog.cpu_freq_ghz,
        capacity=t.fog.capacity,
    )
    cloud = Tier(
        name=TIER_NAMES[2],
        bandwidth_mbps=t.cloud.bandwidth_mbps,
        link_delay_ms=sample_cloud_delay(cfg, rng),
        cpu_freq_ghz=t.cloud.cpu_freq_ghz,
        capacity=t.cloud.capacity,
    )
    return [edge, fog, cloud]


def sample_cloud_delay(cfg: Config, rng: np.random.Generator) -> float:
    lo = cfg.topology.cloud.link_delay_ms_min
    hi = cfg.topology.cloud.link_delay_ms_max
    return float(rng.uniform(lo, hi))


class TaskGenerator:
    """Generates admitted tasks for IoT nodes with Poisson arrivals.

    Per decision epoch the simulator asks for the next task to route. Arrival
    intensity scales with ``iot_nodes * arrival_rate``; data size and compute
    demand are exponentially distributed around the configured means.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator,
                 iot_nodes: int | None = None, arrival_rate: float | None = None):
        self.cfg = cfg
        self.rng = rng
        self.iot_nodes = iot_nodes if iot_nodes is not None else cfg.topology.iot_nodes
        self.arrival_rate = (
            arrival_rate if arrival_rate is not None else cfg.topology.arrival_rate
        )

    def next_task(self) -> Task:
        wl = self.cfg.workload
        data = float(self.rng.exponential(wl.data_size_bits_mean))
        cycles = float(self.rng.exponential(wl.cycles_mean))
        # Guard against pathological near-zero draws.
        data = max(data, wl.data_size_bits_mean * 0.05)
        cycles = max(cycles, wl.cycles_mean * 0.05)
        return Task(data_size_bits=data, cycles=cycles, deadline_ms=wl.deadline_ms)

    @property
    def offered_load(self) -> float:
        """Aggregate arrival intensity (tasks/s) — drives queue build-up."""
        return self.iot_nodes * self.arrival_rate
