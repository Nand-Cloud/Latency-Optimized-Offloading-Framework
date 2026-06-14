"""Replay buffer: capacity eviction and sample shapes."""
import numpy as np

from src.agents.replay import ReplayBuffer


def _push_n(buf, n):
    for i in range(n):
        s = np.full(6, i, dtype=np.float32)
        buf.push(s, i % 3, float(i), s + 1, i % 2 == 0)


def test_capacity_eviction():
    buf = ReplayBuffer(capacity=10)
    _push_n(buf, 25)
    assert len(buf) == 10  # only the most recent 10 retained


def test_sample_shapes():
    buf = ReplayBuffer(capacity=100, rng=np.random.default_rng(0))
    _push_n(buf, 50)
    states, actions, rewards, next_states, dones = buf.sample(8)
    assert states.shape == (8, 6)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, 6)
    assert dones.shape == (8,)
    assert actions.dtype == np.int64


def test_sample_within_pushed_range():
    buf = ReplayBuffer(capacity=100, rng=np.random.default_rng(1))
    _push_n(buf, 5)
    _, actions, _, _, _ = buf.sample(20)
    assert set(np.unique(actions)).issubset({0, 1, 2})
