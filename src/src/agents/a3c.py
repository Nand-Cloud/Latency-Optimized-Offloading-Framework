"""A3C baseline (synchronous single-worker / A2C-style).

On-policy advantage actor-critic over the three tier actions with a shared trunk,
GAE advantage estimation, and an entropy bonus. Trajectories are collected over
an episode and the network is updated at the episode boundary. Resource level is
fixed to the default.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from ..common.config import Config
from ..env.interface import Action
from .base import Agent
from .networks import ActorCriticNet


class A3C(Agent):
    name = "A3C"
    learns = True

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None,
                 device: str = "cpu"):
        a = cfg.agent
        self.cfg = cfg
        self.device = torch.device(device)
        self.rng = rng or np.random.default_rng()
        self.n_tiers = a.n_tiers
        self.default_resource = a.n_resource_levels - 1
        self.gamma = a.gamma
        self.lam = a.gae_lambda
        self.entropy_coef = a.entropy_coef
        self.value_coef = a.value_coef
        self.torch_gen = torch.Generator().manual_seed(int(self.rng.integers(1 << 31)))

        self.net = ActorCriticNet(a.state_dim, a.hidden, a.n_tiers).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=a.lr)
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._dones: List[float] = []

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> Action:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.net(s)
        if explore:
            probs = torch.softmax(logits, dim=1)
            tier = int(torch.multinomial(probs, 1, generator=self.torch_gen).item())
        else:
            tier = int(torch.argmax(logits, dim=1).item())
        return tier, self.default_resource

    def observe(self, state, action, reward, next_state, done) -> None:
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(int(action[0]))
        self._rewards.append(float(reward))
        self._dones.append(float(done))

    def _compute_gae(self, values: torch.Tensor) -> torch.Tensor:
        rewards = self._rewards
        dones = self._dones
        adv = torch.zeros(len(rewards), device=self.device)
        last = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards) - 1 else values[t + 1].item()
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * mask - values[t].item()
            last = delta + self.gamma * self.lam * mask * last
            adv[t] = last
        return adv

    def update(self) -> float | None:
        return None  # on-policy update happens at the episode boundary

    def end_episode(self) -> None:
        if not self._states:
            return
        states = torch.as_tensor(np.stack(self._states), device=self.device)
        actions = torch.as_tensor(self._actions, device=self.device)
        logits, values = self.net(states)
        advantages = self._compute_gae(values.detach())
        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.log_softmax(logits, dim=1)
        chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(torch.softmax(logits, dim=1) * log_probs).sum(dim=1).mean()

        policy_loss = -(chosen * advantages).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optim.step()
        self._states.clear(); self._actions.clear()
        self._rewards.clear(); self._dones.clear()

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
