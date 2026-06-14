"""Tabular Q-Learning baseline.

The continuous 6-dim state is discretised into a coarse grid; a Q-table over the
three tier actions is learned with standard one-step Q-learning and an
epsilon-greedy policy. Resource level is fixed to the default (the tabular method
controls only the high-level tier choice).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..common.config import Config
from ..env.interface import Action
from .base import Agent


class TabularQ(Agent):
    name = "Tabular-Q"
    learns = True

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None,
                 device: str = "cpu", bins: int = 4):
        a = cfg.agent
        self.rng = rng or np.random.default_rng()
        self.n_tiers = a.n_tiers
        self.default_resource = a.n_resource_levels - 1
        self.bins = bins
        self.gamma = a.gamma
        self.lr = 0.1
        self.epsilon = a.epsilon_start
        self.eps_end = a.epsilon_end
        self.eps_decay = a.epsilon_decay
        self.q: dict = defaultdict(lambda: np.zeros(self.n_tiers))
        self._last = None  # (key, action) pending update

    def _key(self, state: np.ndarray) -> tuple:
        # State features are already roughly normalised to [0, ~2].
        clipped = np.clip(np.asarray(state) / 2.0, 0.0, 1.0)
        return tuple((clipped * (self.bins - 1)).round().astype(int).tolist())

    def act(self, state: np.ndarray, explore: bool = True) -> Action:
        key = self._key(state)
        if explore and self.rng.random() < self.epsilon:
            tier = int(self.rng.integers(self.n_tiers))
        else:
            tier = int(np.argmax(self.q[key]))
        self._last = (key, tier)
        return tier, self.default_resource

    def observe(self, state, action, reward, next_state, done) -> None:
        if self._last is None:
            return
        key, tier = self._last
        next_key = self._key(next_state)
        target = reward + (0.0 if done else self.gamma * np.max(self.q[next_key]))
        self.q[key][tier] += self.lr * (target - self.q[key][tier])

    def end_episode(self) -> None:
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
