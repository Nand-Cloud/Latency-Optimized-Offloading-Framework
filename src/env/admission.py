"""Edge-AI admission module.

Tasks are classified by compute demand into Simple / Moderate / Complex.
*Simple* tasks are served locally in ~1 ms and never reach the DRL agent — they
are filtered out before a decision epoch. Moderate/Complex tasks are admitted and
routed by the agent; the category is also surfaced as part of the energy/twin
context feature in the state vector.
"""
from __future__ import annotations

from .models import Task

SIMPLE = "Simple"
MODERATE = "Moderate"
COMPLEX = "Complex"


def classify(cycles: float, simple_max: float, moderate_max: float) -> str:
    if cycles <= simple_max:
        return SIMPLE
    if cycles <= moderate_max:
        return MODERATE
    return COMPLEX


def admit(task: Task, simple_max: float, moderate_max: float) -> bool:
    """Tag the task category in place; return True if it reaches the agent."""
    task.category = classify(task.cycles, simple_max, moderate_max)
    return task.category != SIMPLE
