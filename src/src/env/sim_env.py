"""Pure-Python discrete-event Edge–Fog–Cloud offloading environment.

Implements :class:`~src.env.interface.OffloadingEnv` with no external simulator
dependency, so the entire pipeline runs without NS-3. The same interface is
implemented by :mod:`src.env.ns3_adapter` for a future NS-3 binding.

Dynamics (per 50 ms decision epoch)
-----------------------------------
* One admitted task is presented to the agent, which chooses ``(tier, resource)``.
* End-to-end delay and energy for that task are computed from :mod:`.models`
  using the chosen tier's *current backlog*.
* Queues evolve: the routed task plus a background load (from the other IoT
  arrivals) are added, then each tier drains at its service rate. Routing
  pressure therefore builds congestion the policy must learn to balance.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..common.config import Config
from . import models
from .admission import admit
from .interface import OffloadingEnv, Spaces, StepResult, TIER_NAMES
from .models import Task, Tier
from .topology import TaskGenerator, build_tiers, sample_cloud_delay

# Low-level controller resource levels (fraction of a tier's CPU frequency).
RESOURCE_FACTORS = [0.5, 0.75, 1.0]


class SimEnv(OffloadingEnv):
    def __init__(self, cfg: Config, weight: tuple[float, float] | None = None,
                 iot_nodes: int | None = None, arrival_rate: float | None = None):
        self.cfg = cfg
        self.w_latency, self.w_energy = weight if weight else cfg.default_weight
        self._iot_nodes = iot_nodes
        self._arrival_rate = arrival_rate
        self.spaces = Spaces(
            state_dim=cfg.agent.state_dim,
            n_tiers=cfg.agent.n_tiers,
            n_resource_levels=cfg.agent.n_resource_levels,
        )
        self.epoch_ms = cfg.training.decision_epoch_ms
        # Populated on reset.
        self.rng: np.random.Generator | None = None
        self.tiers: List[Tier] = []
        self.queues: np.ndarray = np.zeros(self.spaces.n_tiers)
        self.mean_service_ms: np.ndarray = np.zeros(self.spaces.n_tiers)
        self.gen: TaskGenerator | None = None
        self.current_task: Task | None = None
        self.last_delay_ms = 0.0
        self.last_energy_j = 0.0
        self.t = 0
        self.max_steps = cfg.training.steps_per_episode

    # -- setup -------------------------------------------------------------
    def set_horizon(self, steps: int) -> None:
        self.max_steps = steps

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.tiers = build_tiers(self.cfg, self.rng)
        self.gen = TaskGenerator(self.cfg, self.rng, self._iot_nodes, self._arrival_rate)
        # Mean service time per tier = processing of an average task at full clock.
        avg = Task(0.0, self.cfg.workload.cycles_mean, self.cfg.workload.deadline_ms)
        self.mean_service_ms = np.array(
            [models.processing_delay_ms(avg, tier, 1.0) for tier in self.tiers]
        )
        self.queues = np.zeros(self.spaces.n_tiers)
        self.last_delay_ms = 0.0
        self.last_energy_j = 0.0
        self.t = 0
        self.current_task = self._next_admitted_task()
        return self._make_state(self.current_task)

    # -- core --------------------------------------------------------------
    def step(self, action) -> StepResult:
        assert self.current_task is not None and self.rng is not None
        tier_idx, resource_idx = int(action[0]), int(action[1])
        tier = self.tiers[tier_idx]
        resource_factor = RESOURCE_FACTORS[resource_idx]
        cap = tier.capacity
        backlog = float(self.queues[tier_idx])

        # Cloud WAN delay is re-sampled per task (variable link).
        if tier.name == TIER_NAMES[2]:
            tier.link_delay_ms = sample_cloud_delay(self.cfg, self.rng)

        task = self.current_task
        delay_ms = models.end_to_end_delay_ms(
            task, tier, resource_factor, backlog, self.mean_service_ms[tier_idx]
        )
        energy_j = models.energy_joules(
            task, tier, resource_factor,
            self.cfg.model.tx_power_w, self.cfg.model.kappa,
        )

        util = float(np.clip(backlog / cap, 0.0, 1.0))
        over_capacity = backlog >= cap

        m = self.cfg.model
        r = models.reward(
            delay_ms, energy_j,
            queue_term=util, util_term=util, packet_bits=task.data_size_bits,
            w_latency=self.w_latency, w_energy=self.w_energy,
            deadline_ms=task.deadline_ms,
            alpha_Q=m.alpha_Q, alpha_U=m.alpha_U, beta=m.beta,
            energy_norm=m.energy_norm,
            deadline_penalty=m.deadline_penalty, capacity_penalty=m.capacity_penalty,
            over_capacity=over_capacity,
        )

        self._advance_queues(tier_idx)
        self.last_delay_ms = delay_ms
        self.last_energy_j = energy_j
        self.t += 1
        done = self.t >= self.max_steps

        info = {
            "latency_ms": delay_ms,
            "energy_j": energy_j,
            "tier": tier_idx,
            "tier_name": tier.name,
            "resource": resource_idx,
            "deadline_met": delay_ms <= task.deadline_ms,
            "over_capacity": over_capacity,
            "category": task.category,
        }

        self.current_task = None if done else self._next_admitted_task()
        next_state = (
            np.zeros(self.spaces.state_dim, dtype=np.float32)
            if done else self._make_state(self.current_task)
        )
        return StepResult(state=next_state, reward=r, done=done, info=info)

    # -- helpers -----------------------------------------------------------
    def _next_admitted_task(self) -> Task:
        """Draw tasks until one passes admission (Simple tasks served locally)."""
        assert self.gen is not None
        a = self.cfg.workload.admission
        while True:
            task = self.gen.next_task()
            if admit(task, a.simple_max_cycles, a.moderate_max_cycles):
                return task

    def _advance_queues(self, chosen: int) -> None:
        """Add the routed task + background arrivals, then drain each tier."""
        assert self.gen is not None
        epoch_s = self.epoch_ms / 1000.0
        arrivals = self.gen.offered_load * epoch_s
        background = max(0.0, arrivals - 1.0) / self.spaces.n_tiers

        self.queues[chosen] += 1.0
        self.queues += background
        drain = self.epoch_ms / self.mean_service_ms  # tasks served per epoch per tier
        self.queues = np.maximum(0.0, self.queues - drain)

    def _make_state(self, task: Task) -> np.ndarray:
        """6-dim state: avail bandwidth, queue length, last delay, packet size,
        CPU utilization, energy/twin-context feature (all normalised)."""
        caps = np.array([t.capacity for t in self.tiers], dtype=float)
        utils = np.clip(self.queues / caps, 0.0, 1.0)
        mean_util = float(utils.mean())

        avail_bw = 1.0 - mean_util
        total_cap = caps.sum()
        queue_len = float(np.clip(self.queues.sum() / total_cap, 0.0, 1.0))
        last_delay = float(np.clip(self.last_delay_ms / task.deadline_ms, 0.0, 2.0))
        packet_size = float(
            np.clip(task.data_size_bits / (self.cfg.workload.data_size_bits_mean * 4.0),
                    0.0, 1.0)
        )
        cpu_util = mean_util
        # Energy / digital-twin context: recent energy blended with task complexity.
        cat_code = 1.0 if task.category == "Complex" else 0.5
        energy_ctx = float(
            np.clip(0.5 * (self.last_energy_j / self.cfg.model.energy_norm)
                    + 0.5 * cat_code, 0.0, 2.0)
        )
        return np.array(
            [avail_bw, queue_len, last_delay, packet_size, cpu_util, energy_ctx],
            dtype=np.float32,
        )
