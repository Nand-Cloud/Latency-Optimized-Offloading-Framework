"""Experience replay buffer (capacity 2000, batch 32 by default)."""
from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator | None = None):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = rng or np.random.default_rng()

    def push(self, state, action_flat: int, reward: float, next_state, done: bool) -> None:
        self.buffer.append(
            (np.asarray(state, dtype=np.float32), int(action_flat), float(reward),
             np.asarray(next_state, dtype=np.float32), bool(done))
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
