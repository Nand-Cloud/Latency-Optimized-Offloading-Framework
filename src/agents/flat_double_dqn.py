"""Flat Double-DQN baseline: FlatDQN with the Double-DQN target rule."""
from __future__ import annotations

from .flat_dqn import FlatDQN


class FlatDoubleDQN(FlatDQN):
    name = "Flat-DoubleDQN"
    double = True
