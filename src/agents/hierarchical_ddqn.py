"""Proposed method: hierarchical, multi-objective Double-DQN.

The policy factorises as ``pi_high(tier|S) * pi_low(resource|S, tier)`` realised
by :class:`~src.agents.networks.HierarchicalQNet`. Both heads are trained with the
**Double-DQN** target rule (online net selects the action, target net evaluates
it) against the shared scalarised multi-objective reward produced by the env.
The multi-objective behaviour is obtained by training across the (w_L, w_E)
weight grid, which traces out the latency–energy Pareto front.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..common.config import Config
from ..env.interface import Action, decode_action, encode_action
from .base import Agent
from .networks import HierarchicalQNet
from .replay import ReplayBuffer


class HierarchicalDoubleDQN(Agent):
    name = "Proposed-HierDDQN"
    learns = True

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None,
                 device: str = "cpu"):
        a = cfg.agent
        self.cfg = cfg
        self.device = torch.device(device)
        self.rng = rng or np.random.default_rng()
        self.n_tiers = a.n_tiers
        self.n_resource = a.n_resource_levels
        self.gamma = a.gamma
        self.batch_size = a.batch_size
        self.target_sync = a.target_sync_steps

        self.online = HierarchicalQNet(a.state_dim, a.hidden, a.n_tiers,
                                       a.n_resource_levels).to(self.device)
        self.target = HierarchicalQNet(a.state_dim, a.hidden, a.n_tiers,
                                       a.n_resource_levels).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim = torch.optim.Adam(self.online.parameters(), lr=a.lr)
        self.replay = ReplayBuffer(a.replay_capacity, self.rng)

        self.epsilon = a.epsilon_start
        self.eps_end = a.epsilon_end
        self.eps_decay = a.epsilon_decay
        self.learn_steps = 0

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> Action:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_high, q_low = self.online(s)
        if explore and self.rng.random() < self.epsilon:
            tier = int(self.rng.integers(self.n_tiers))
        else:
            tier = int(torch.argmax(q_high, dim=1).item())
        if explore and self.rng.random() < self.epsilon:
            resource = int(self.rng.integers(self.n_resource))
        else:
            resource = int(torch.argmax(q_low[0, tier], dim=0).item())
        return tier, resource

    # -- learning ----------------------------------------------------------
    def observe(self, state, action, reward, next_state, done) -> None:
        flat = encode_action(action[0], action[1], self.n_resource)
        self.replay.push(state, flat, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = torch.as_tensor(states, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)
        tiers = torch.as_tensor(actions // self.n_resource, device=self.device)
        resources = torch.as_tensor(actions % self.n_resource, device=self.device)

        q_high, q_low = self.online(states)
        q_high_sa = q_high.gather(1, tiers.unsqueeze(1)).squeeze(1)
        # Q_low for the chosen (tier, resource).
        q_low_tier = q_low[torch.arange(len(tiers)), tiers]          # [B, n_resource]
        q_low_sa = q_low_tier.gather(1, resources.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_high_on, next_q_low_on = self.online(next_states)
            next_q_high_tg, next_q_low_tg = self.target(next_states)

            # High head: online selects tier', target evaluates it.
            next_tier = torch.argmax(next_q_high_on, dim=1)
            high_next_val = next_q_high_tg.gather(1, next_tier.unsqueeze(1)).squeeze(1)
            high_target = rewards + self.gamma * (1.0 - dones) * high_next_val

            # Low head: condition on the online-selected tier', then Double-DQN
            # over resources within that tier.
            idx = torch.arange(len(next_tier))
            low_on_tier = next_q_low_on[idx, next_tier]              # [B, n_resource]
            next_res = torch.argmax(low_on_tier, dim=1)
            low_tg_tier = next_q_low_tg[idx, next_tier]
            low_next_val = low_tg_tier.gather(1, next_res.unsqueeze(1)).squeeze(1)
            low_target = rewards + self.gamma * (1.0 - dones) * low_next_val

        loss = nn.functional.mse_loss(q_high_sa, high_target) + \
            nn.functional.mse_loss(q_low_sa, low_target)
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

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(self.online.state_dict(), path)

    def load(self, path: str) -> None:
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())
