"""Agent registry: maps method names to constructors.

All agents share the signature ``(cfg, rng, device)`` so experiment drivers can
build any method uniformly. ``ALL_AGENTS`` is ordered with the proposed method
first, matching the table/figure layout in the paper.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from ..common.config import Config
from .a3c import A3C
from .base import Agent
from .cloud_only import CloudOnly
from .flat_double_dqn import FlatDoubleDQN
from .flat_dqn import FlatDQN
from .hierarchical_ddqn import HierarchicalDoubleDQN
from .ppo import PPO
from .rule_based import RuleBased
from .tabular_q import TabularQ

AgentFactory = Callable[[Config, np.random.Generator, str], Agent]

REGISTRY: Dict[str, AgentFactory] = {
    HierarchicalDoubleDQN.name: HierarchicalDoubleDQN,
    CloudOnly.name: CloudOnly,
    RuleBased.name: RuleBased,
    TabularQ.name: TabularQ,
    FlatDQN.name: FlatDQN,
    FlatDoubleDQN.name: FlatDoubleDQN,
    A3C.name: A3C,
    PPO.name: PPO,
}

#: Display/iteration order — proposed method first, then baselines.
ALL_AGENTS: List[str] = list(REGISTRY.keys())
PROPOSED: str = HierarchicalDoubleDQN.name


def make_agent(name: str, cfg: Config, rng: np.random.Generator,
               device: str = "cpu") -> Agent:
    if name not in REGISTRY:
        raise KeyError(f"Unknown agent '{name}'. Known: {ALL_AGENTS}")
    return REGISTRY[name](cfg, rng, device)
