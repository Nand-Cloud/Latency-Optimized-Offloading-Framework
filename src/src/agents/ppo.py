"""PPO / Actor-Critic baseline (clipped surrogate objective).

On-policy PPO over the three tier actions with a shared actor-critic trunk, GAE
advantages, and several optimisation epochs over each episode's trajectory using
the clipped policy objective. Resource level is fixed to the default.
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


class PPO(Agent):
    name = "PPO"
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
        self.clip = a.ppo_clip
        self.epochs = a.ppo_epochs
        self.entropy_coef = a.entropy_coef
        self.value_coef = a.value_coef
        self.torch_gen = torch.Generator().manual_seed(int(self.rng.integers(1 << 31)))

        self.net = ActorCriticNet(a.state_dim, a.hidden, a.n_tiers).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=a.lr)
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._dones: List[float] = []
        self._logp: List[float] = []
        self._values: List[float] = []

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> Action:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(s)
        log_probs = torch.log_softmax(logits, dim=1)
        if explore:
            probs = torch.softmax(logits, dim=1)
            tier = int(torch.multinomial(probs, 1, generator=self.torch_gen).item())
        else:
            tier = int(torch.argmax(logits, dim=1).item())
        self._pending = (float(log_probs[0, tier].item()), float(value.item()))
        return tier, self.default_resource

    def observe(self, state, action, reward, next_state, done) -> None:
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(int(action[0]))
        self._rewards.append(float(reward))
        self._dones.append(float(done))
        logp, value = getattr(self, "_pending", (0.0, 0.0))
        self._logp.append(logp)
        self._values.append(value)

    def _compute_gae(self, values: List[float]) -> torch.Tensor:
        adv = torch.zeros(len(self._rewards), device=self.device)
        last = 0.0
        for t in reversed(range(len(self._rewards))):
            next_val = 0.0 if t == len(self._rewards) - 1 else values[t + 1]
            mask = 1.0 - self._dones[t]
            delta = self._rewards[t] + self.gamma * next_val * mask - values[t]
            last = delta + self.gamma * self.lam * mask * last
            adv[t] = last
        return adv

    def update(self) -> float | None:
        return None  # PPO updates at the episode boundary

    def end_episode(self) -> None:
        if not self._states:
            return
        states = torch.as_tensor(np.stack(self._states), device=self.device)
        actions = torch.as_tensor(self._actions, device=self.device)
        old_logp = torch.as_tensor(self._logp, dtype=torch.float32, device=self.device)
        advantages = self._compute_gae(self._values)
        returns = advantages + torch.as_tensor(self._values, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            logits, values = self.net(states)
            log_probs = torch.log_softmax(logits, dim=1)
            chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(chosen - old_logp)
            clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = nn.functional.mse_loss(values, returns)
            entropy = -(torch.softmax(logits, dim=1) * log_probs).sum(dim=1).mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            self.optim.step()

        for buf in (self._states, self._actions, self._rewards,
                    self._dones, self._logp, self._values):
            buf.clear()

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
