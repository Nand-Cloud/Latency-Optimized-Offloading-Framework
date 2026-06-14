"""Neural network definitions (PyTorch).

* :class:`MLP`           -- generic trunk (the paper's FC 6-64-64-N ReLU stack).
* :class:`QNetwork`      -- state -> Q-values over a discrete action set (DQN family).
* :class:`HierarchicalQNet` -- two heads realising the policy factorisation
  ``pi_high(tier|S) * pi_low(resource|S,tier)`` for the proposed agent.
* :class:`ActorCriticNet` -- shared trunk with policy logits + value (A3C / PPO).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    """FC state -> Q(a) for a flat discrete action space."""

    def __init__(self, state_dim: int, hidden: List[int], n_actions: int):
        super().__init__()
        self.body = MLP(state_dim, hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class HierarchicalQNet(nn.Module):
    """Shared trunk feeding a high-level (tier) head and a low-level (resource)
    head. The low head emits ``n_tiers * n_resource`` values; the row for the
    selected tier gives ``Q_low(resource | S, tier)``.
    """

    def __init__(self, state_dim: int, hidden: List[int], n_tiers: int,
                 n_resource: int):
        super().__init__()
        self.n_tiers = n_tiers
        self.n_resource = n_resource
        # Shared representation, then two linear heads.
        trunk_out = hidden[-1]
        trunk_layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden:
            trunk_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)
        self.high_head = nn.Linear(trunk_out, n_tiers)
        self.low_head = nn.Linear(trunk_out, n_tiers * n_resource)

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        q_high = self.high_head(z)
        q_low = self.low_head(z).view(-1, self.n_tiers, self.n_resource)
        return q_high, q_low


class ActorCriticNet(nn.Module):
    """Shared trunk with a categorical policy head and a scalar value head."""

    def __init__(self, state_dim: int, hidden: List[int], n_actions: int):
        super().__init__()
        trunk_out = hidden[-1]
        trunk_layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden:
            trunk_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)
        self.policy = nn.Linear(trunk_out, n_actions)
        self.value = nn.Linear(trunk_out, 1)

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        return self.policy(z), self.value(z).squeeze(-1)
