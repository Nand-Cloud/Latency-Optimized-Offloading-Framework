"""Flat DQN baseline (and shared DQN core for the Double-DQN variant).

A standard single-level DQN over the three tier actions (FC 6-64-64-3), with a
fixed default resource level. The ``double`` flag switches the target rule
between vanilla DQN (``max`` over the target net) and Double-DQN (online net
selects, target net evaluates) — reused by :mod:`.flat_double_dqn`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..common.config import Config
from ..env.interface import Action
from .base import Agent
from .networks import QNetwork
from .replay import ReplayBuffer


class FlatDQN(Agent):
    name = "Flat-DQN"
    learns = True
    double = False

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None,
                 device: str = "cpu"):
        a = cfg.agent
        self.cfg = cfg
        self.device = torch.device(device)
        self.rng = rng or np.random.default_rng()
        self.n_tiers = a.n_tiers
        self.default_resource = a.n_resource_levels - 1
        self.gamma = a.gamma
        self.batch_size = a.batch_size
        self.target_sync = a.target_sync_steps

        self.online = QNetwork(a.state_dim, a.hidden, a.n_tiers).to(self.device)
        self.target = QNetwork(a.state_dim, a.hidden, a.n_tiers).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim = torch.optim.Adam(self.online.parameters(), lr=a.lr)
        self.replay = ReplayBuffer(a.replay_capacity, self.rng)

        self.epsilon = a.epsilon_start
        self.eps_end = a.epsilon_end
        self.eps_decay = a.epsilon_decay
        self.learn_steps = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> Action:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_tiers)), self.default_resource
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        tier = int(torch.argmax(self.online(s), dim=1).item())
        return tier, self.default_resource

    def observe(self, state, action, reward, next_state, done) -> None:
        # Only the tier dimension is learned; resource is fixed.
        self.replay.push(state, action[0], reward, next_state, done)

    def update(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = torch.as_tensor(states, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)

        q_sa = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                next_act = torch.argmax(self.online(next_states), dim=1)
                next_val = self.target(next_states).gather(
                    1, next_act.unsqueeze(1)).squeeze(1)
            else:
                next_val = self.target(next_states).max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * next_val

        loss = nn.functional.mse_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def end_episode(self) -> None:
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def save(self, path: str) -> None:
        torch.save(self.online.state_dict(), path)

    def load(self, path: str) -> None:
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())
