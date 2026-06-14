"""Physical models: tasks, tiers, latency, energy, and the reward function.

These are pure functions / plain dataclasses with no simulator state, which
keeps them trivially unit-testable against hand-computed values
(see ``tests/test_models.py`` and ``tests/test_reward.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

GHZ = 1.0e9  # cycles per second per GHz
MBPS = 1.0e6  # bits per second per Mbps


@dataclass
class Task:
    """A unit of work routed by the agent."""

    data_size_bits: float   # transmission payload (bits)
    cycles: float           # compute demand (CPU cycles)
    deadline_ms: float      # soft deadline for the deadline constraint
    category: str = "Moderate"  # Simple | Moderate | Complex (admission output)


@dataclass
class Tier:
    """A compute tier (Edge / Fog / Cloud) and the link reaching it."""

    name: str
    bandwidth_mbps: float
    link_delay_ms: float
    cpu_freq_ghz: float
    capacity: int


def transmission_delay_ms(task: Task, tier: Tier) -> float:
    """Payload transmission time = data / bandwidth (ms)."""
    return (task.data_size_bits / (tier.bandwidth_mbps * MBPS)) * 1000.0


def processing_delay_ms(task: Task, tier: Tier, resource_factor: float) -> float:
    """Compute time = cycles / (effective frequency) (ms).

    ``resource_factor`` is the low-level controller's intra-tier allocation in
    ``(0, 1]``: a larger share gives a faster effective clock.
    """
    eff_freq = tier.cpu_freq_ghz * GHZ * resource_factor
    return (task.cycles / eff_freq) * 1000.0


def queuing_delay_ms(queue_len: float, mean_service_ms: float) -> float:
    """Backlog already at the tier must drain before this task runs."""
    return max(0.0, queue_len) * mean_service_ms


def end_to_end_delay_ms(
    task: Task,
    tier: Tier,
    resource_factor: float,
    queue_len: float,
    mean_service_ms: float,
) -> float:
    """transmission + link + queuing + processing (ms)."""
    return (
        tier.link_delay_ms
        + transmission_delay_ms(task, tier)
        + queuing_delay_ms(queue_len, mean_service_ms)
        + processing_delay_ms(task, tier, resource_factor)
    )


def energy_joules(
    task: Task,
    tier: Tier,
    resource_factor: float,
    tx_power_w: float,
    kappa: float,
) -> float:
    """Energy = tx_power * tx_delay + kappa * cycles * freq^2.

    The transmission term uses the transmission delay in *seconds*; the compute
    term follows the standard dynamic-power model where energy grows with the
    square of the operating frequency (so a higher ``resource_factor`` finishes
    faster but costs more energy — the low-level trade-off).
    """
    tx_delay_s = transmission_delay_ms(task, tier) / 1000.0
    eff_freq = tier.cpu_freq_ghz * GHZ * resource_factor
    tx_energy = tx_power_w * tx_delay_s
    compute_energy = kappa * task.cycles * (eff_freq ** 2)
    return tx_energy + compute_energy


def reward(
    delay_ms: float,
    energy_j: float,
    queue_term: float,
    util_term: float,
    packet_bits: float,
    w_latency: float,
    w_energy: float,
    *,
    deadline_ms: float,
    alpha_Q: float,
    alpha_U: float,
    beta: float,
    energy_norm: float,
    deadline_penalty: float,
    capacity_penalty: float,
    over_capacity: bool,
) -> float:
    """Scalarised multi-objective reward (negative cost).

    ``-(w_L*(D + alpha_Q*Q + alpha_U*U + P/beta) + w_E*E_norm) - penalty``

    Delay is normalised by the deadline and energy by ``energy_norm`` so the two
    objectives live on a comparable scale before weighting. Constraint
    violations (deadline miss, tier at capacity) add a Lagrangian-style penalty.
    """
    d_norm = delay_ms / deadline_ms
    e_norm = energy_j / energy_norm
    p_norm = packet_bits / beta

    latency_cost = d_norm + alpha_Q * queue_term + alpha_U * util_term + p_norm
    cost = w_latency * latency_cost + w_energy * e_norm

    penalty = 0.0
    if delay_ms > deadline_ms:
        penalty += deadline_penalty
    if over_capacity:
        penalty += capacity_penalty

    return -(cost) - penalty
