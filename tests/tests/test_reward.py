"""Reward function behaviour: sign, weighting, and constraint penalties."""
from src.env.models import reward

BASE = dict(
    queue_term=0.2, util_term=0.2, packet_bits=2.0e6,
    deadline_ms=100.0, alpha_Q=0.5, alpha_U=0.5, beta=1.0e6,
    energy_norm=5.0, deadline_penalty=2.0, capacity_penalty=1.0,
)


def test_reward_is_negative_cost():
    r = reward(delay_ms=50.0, energy_j=1.0, w_latency=0.5, w_energy=0.5,
               over_capacity=False, **BASE)
    assert r < 0.0


def test_deadline_penalty_applied():
    common = dict(energy_j=1.0, w_latency=0.5, w_energy=0.5,
                  over_capacity=False, **BASE)
    met = reward(delay_ms=50.0, **common)
    missed = reward(delay_ms=150.0, **common)
    # Missing the deadline must cost strictly more than the delay increase alone.
    assert missed < met - BASE["deadline_penalty"] + 0.001


def test_capacity_penalty_applied():
    common = dict(delay_ms=50.0, energy_j=1.0, w_latency=0.5, w_energy=0.5, **BASE)
    ok = reward(over_capacity=False, **common)
    over = reward(over_capacity=True, **common)
    assert abs((ok - over) - BASE["capacity_penalty"]) < 1e-9


def test_weight_extremes_select_objective():
    # Pure-latency weighting ignores energy; pure-energy ignores latency.
    lat_only_cheap_e = reward(delay_ms=50.0, energy_j=0.1, w_latency=1.0,
                              w_energy=0.0, over_capacity=False, **BASE)
    lat_only_costly_e = reward(delay_ms=50.0, energy_j=100.0, w_latency=1.0,
                               w_energy=0.0, over_capacity=False, **BASE)
    assert abs(lat_only_cheap_e - lat_only_costly_e) < 1e-9

    ene_only_a = reward(delay_ms=10.0, energy_j=1.0, w_latency=0.0,
                        w_energy=1.0, over_capacity=False, **BASE)
    ene_only_b = reward(delay_ms=90.0, energy_j=1.0, w_latency=0.0,
                        w_energy=1.0, over_capacity=False, **BASE)
    assert abs(ene_only_a - ene_only_b) < 1e-9
