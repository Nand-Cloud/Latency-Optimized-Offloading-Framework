"""Load and validate ``config.yaml`` into typed dataclasses.

A single :class:`Config` object is threaded through the env, agents, training,
experiments, and analysis so every component reads the same parameters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import yaml

# Repo root = three levels up from this file (src/src/common/config.py -> repo).
# Path: <repo>/src/src/common/config.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")


@dataclass
class TierConfig:
    count: int
    bandwidth_mbps: float
    link_delay_ms: float
    cpu_freq_ghz: float
    capacity: int


@dataclass
class CloudConfig:
    count: int
    bandwidth_mbps: float
    link_delay_ms_min: float
    link_delay_ms_max: float
    cpu_freq_ghz: float
    capacity: int


@dataclass
class TopologyConfig:
    iot_nodes: int
    arrival_rate: float
    edge: TierConfig
    fog: TierConfig
    cloud: CloudConfig


@dataclass
class AdmissionConfig:
    simple_max_cycles: float
    moderate_max_cycles: float
    local_service_ms: float


@dataclass
class WorkloadConfig:
    data_size_bits_mean: float
    cycles_mean: float
    deadline_ms: float
    admission: AdmissionConfig


@dataclass
class ModelConfig:
    tx_power_w: float
    kappa: float
    energy_norm: float
    alpha_Q: float
    alpha_U: float
    beta: float
    deadline_penalty: float
    capacity_penalty: float


@dataclass
class AgentConfig:
    state_dim: int
    n_tiers: int
    n_resource_levels: int
    hidden: List[int]
    lr: float
    gamma: float
    target_sync_steps: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    replay_capacity: int
    batch_size: int
    entropy_coef: float
    value_coef: float
    ppo_clip: float
    ppo_epochs: int
    gae_lambda: float


@dataclass
class TrainingConfig:
    episodes: int
    steps_per_episode: int
    decision_epoch_ms: float
    quick_episodes: int
    quick_steps_per_episode: int


@dataclass
class ScalabilityConfig:
    node_counts: List[int]
    arrival_rates: List[float]


@dataclass
class PathsConfig:
    results_dir: str
    raw_dir: str
    tables_dir: str
    figures_dir: str


@dataclass
class SeedsConfig:
    list: List[int]
    quick_count: int


@dataclass
class Config:
    seeds: SeedsConfig
    topology: TopologyConfig
    workload: WorkloadConfig
    model: ModelConfig
    weight_grid: List[Tuple[float, float]]
    default_weight: Tuple[float, float]
    agent: AgentConfig
    training: TrainingConfig
    scalability: ScalabilityConfig
    paths: PathsConfig
    raw: dict = field(default_factory=dict, repr=False)

    # -- convenience -------------------------------------------------------
    def seeds_for(self, quick: bool) -> List[int]:
        return self.seeds.list[: self.seeds.quick_count] if quick else self.seeds.list

    def episodes_for(self, quick: bool) -> int:
        return self.training.quick_episodes if quick else self.training.episodes

    def steps_for(self, quick: bool) -> int:
        return (
            self.training.quick_steps_per_episode
            if quick
            else self.training.steps_per_episode
        )

    def abspath(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)

    def ensure_dirs(self) -> None:
        for d in (
            self.paths.results_dir,
            self.paths.raw_dir,
            self.paths.tables_dir,
            self.paths.figures_dir,
        ):
            os.makedirs(self.abspath(d), exist_ok=True)


def load_config(path: str | None = None) -> Config:
    """Read ``config.yaml`` and build a validated :class:`Config`."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    topo = raw["topology"]
    topology = TopologyConfig(
        iot_nodes=int(topo["iot_nodes"]),
        arrival_rate=float(topo["arrival_rate"]),
        edge=TierConfig(**_tier(topo["edge"])),
        fog=TierConfig(**_tier(topo["fog"])),
        cloud=CloudConfig(
            count=int(topo["cloud"]["count"]),
            bandwidth_mbps=float(topo["cloud"]["bandwidth_mbps"]),
            link_delay_ms_min=float(topo["cloud"]["link_delay_ms_min"]),
            link_delay_ms_max=float(topo["cloud"]["link_delay_ms_max"]),
            cpu_freq_ghz=float(topo["cloud"]["cpu_freq_ghz"]),
            capacity=int(topo["cloud"]["capacity"]),
        ),
    )

    wl = raw["workload"]
    workload = WorkloadConfig(
        data_size_bits_mean=float(wl["data_size_bits_mean"]),
        cycles_mean=float(wl["cycles_mean"]),
        deadline_ms=float(wl["deadline_ms"]),
        admission=AdmissionConfig(
            simple_max_cycles=float(wl["admission"]["simple_max_cycles"]),
            moderate_max_cycles=float(wl["admission"]["moderate_max_cycles"]),
            local_service_ms=float(wl["admission"]["local_service_ms"]),
        ),
    )

    model = ModelConfig(**{k: float(v) for k, v in raw["model"].items()})

    agent_raw = dict(raw["agent"])
    agent = AgentConfig(
        state_dim=int(agent_raw["state_dim"]),
        n_tiers=int(agent_raw["n_tiers"]),
        n_resource_levels=int(agent_raw["n_resource_levels"]),
        hidden=[int(h) for h in agent_raw["hidden"]],
        lr=float(agent_raw["lr"]),
        gamma=float(agent_raw["gamma"]),
        target_sync_steps=int(agent_raw["target_sync_steps"]),
        epsilon_start=float(agent_raw["epsilon_start"]),
        epsilon_end=float(agent_raw["epsilon_end"]),
        epsilon_decay=float(agent_raw["epsilon_decay"]),
        replay_capacity=int(agent_raw["replay_capacity"]),
        batch_size=int(agent_raw["batch_size"]),
        entropy_coef=float(agent_raw["entropy_coef"]),
        value_coef=float(agent_raw["value_coef"]),
        ppo_clip=float(agent_raw["ppo_clip"]),
        ppo_epochs=int(agent_raw["ppo_epochs"]),
        gae_lambda=float(agent_raw["gae_lambda"]),
    )

    tr = raw["training"]
    training = TrainingConfig(
        episodes=int(tr["episodes"]),
        steps_per_episode=int(tr["steps_per_episode"]),
        decision_epoch_ms=float(tr["decision_epoch_ms"]),
        quick_episodes=int(tr["quick_episodes"]),
        quick_steps_per_episode=int(tr["quick_steps_per_episode"]),
    )

    sc = raw["scalability"]
    scalability = ScalabilityConfig(
        node_counts=[int(n) for n in sc["node_counts"]],
        arrival_rates=[float(r) for r in sc["arrival_rates"]],
    )

    paths = PathsConfig(**raw["paths"])

    seeds = SeedsConfig(
        list=[int(s) for s in raw["seeds"]["list"]],
        quick_count=int(raw["seeds"]["quick_count"]),
    )

    return Config(
        seeds=seeds,
        topology=topology,
        workload=workload,
        model=model,
        weight_grid=[(float(a), float(b)) for a, b in raw["weight_grid"]],
        default_weight=(float(raw["default_weight"][0]), float(raw["default_weight"][1])),
        agent=agent,
        training=training,
        scalability=scalability,
        paths=paths,
        raw=raw,
    )


def _tier(d: dict) -> dict:
    return {
        "count": int(d["count"]),
        "bandwidth_mbps": float(d["bandwidth_mbps"]),
        "link_delay_ms": float(d["link_delay_ms"]),
        "cpu_freq_ghz": float(d["cpu_freq_ghz"]),
        "capacity": int(d["capacity"]),
    }
