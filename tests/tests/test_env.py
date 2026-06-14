"""Smoke tests for the discrete-event environment and admission."""
import numpy as np

from src.common.config import load_config
from src.env.admission import SIMPLE, classify
from src.env.sim_env import RESOURCE_FACTORS, SimEnv


def _cfg():
    return load_config()


def test_reset_returns_six_dim_state():
    env = SimEnv(_cfg())
    state = env.reset(seed=0)
    assert state.shape == (6,)
    assert np.all(np.isfinite(state))


def test_step_returns_valid_result():
    cfg = _cfg()
    env = SimEnv(cfg)
    env.set_horizon(20)
    env.reset(seed=1)
    res = env.step((1, 2))  # Fog, full resource
    assert res.state.shape == (6,)
    assert np.isfinite(res.reward)
    assert res.info["tier"] == 1
    assert 0.0 <= res.info["latency_ms"]
    assert res.info["energy_j"] > 0.0
    assert res.info["category"] in ("Moderate", "Complex")


def test_episode_terminates_at_horizon():
    env = SimEnv(_cfg())
    env.set_horizon(15)
    env.reset(seed=2)
    done_at = None
    for t in range(15):
        res = env.step((0, len(RESOURCE_FACTORS) - 1))
        if res.done:
            done_at = t
            break
    assert done_at == 14


def test_admission_filters_simple_tasks():
    # Simple tasks (below simple_max) are served locally and never routed.
    assert classify(1.0, simple_max=1.0e7, moderate_max=1.0e8) == SIMPLE
    env = SimEnv(_cfg())
    env.set_horizon(30)
    env.reset(seed=3)
    for _ in range(30):
        res = env.step((2, 0))
        assert res.info["category"] != SIMPLE
        if res.done:
            break


def test_all_tiers_selectable():
    env = SimEnv(_cfg())
    env.set_horizon(9)
    env.reset(seed=4)
    seen = set()
    for tier in range(3):
        for resource in range(3):
            res = env.step((tier, resource))
            seen.add(res.info["tier"])
            if res.done:
                break
    assert seen == {0, 1, 2}
