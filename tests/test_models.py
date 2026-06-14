"""Latency and energy models vs hand-computed values."""
import math

from src.env.models import (
    Task, Tier, energy_joules, processing_delay_ms, transmission_delay_ms,
)

TIER = Tier(name="Edge", bandwidth_mbps=100.0, link_delay_ms=2.0,
            cpu_freq_ghz=2.0, capacity=8)
TASK = Task(data_size_bits=1.0e6, cycles=1.0e8, deadline_ms=100.0)


def test_transmission_delay():
    # 1e6 bits / 100 Mbps = 0.01 s = 10 ms
    assert math.isclose(transmission_delay_ms(TASK, TIER), 10.0, rel_tol=1e-9)


def test_processing_delay_full_resource():
    # 1e8 cycles / 2e9 Hz = 0.05 s = 50 ms
    assert math.isclose(processing_delay_ms(TASK, TIER, 1.0), 50.0, rel_tol=1e-9)


def test_processing_delay_scales_with_resource():
    full = processing_delay_ms(TASK, TIER, 1.0)
    half = processing_delay_ms(TASK, TIER, 0.5)
    assert math.isclose(half, 2.0 * full, rel_tol=1e-9)


def test_energy_components():
    # tx: 0.5 W * 0.01 s = 0.005 J ; compute: 1e-27 * 1e8 * (2e9)^2 = 0.4 J
    e = energy_joules(TASK, TIER, 1.0, tx_power_w=0.5, kappa=1.0e-27)
    assert math.isclose(e, 0.405, rel_tol=1e-9)


def test_energy_increases_with_resource_factor():
    low = energy_joules(TASK, TIER, 0.5, tx_power_w=0.5, kappa=1.0e-27)
    high = energy_joules(TASK, TIER, 1.0, tx_power_w=0.5, kappa=1.0e-27)
    assert high > low
