"""Episode training loop and evaluation rollout.

The trainer is agent-agnostic: it drives any :class:`~src.agents.base.Agent`
through the standard reset/act/step/observe/update cycle, performs learning
updates and epsilon decay for learning agents, and skips them for fixed-policy
baselines. Per-episode means form the convergence curve; a final greedy rollout
produces the per-task records used for tables, statistics, and figures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..agents.base import Agent
from ..env.interface import OffloadingEnv


def episode_seed(base: int, ep: int) -> int:
    return base * 100000 + ep


def eval_seed(base: int, ep: int) -> int:
    return base * 100000 + 90000 + ep


@dataclass
class TrainHistory:
    reward: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)
    energy_j: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)


class Trainer:
    def __init__(self, env: OffloadingEnv, agent: Agent, episodes: int,
                 steps: int, base_seed: int):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.steps = steps
        self.base_seed = base_seed
        if hasattr(env, "set_horizon"):
            env.set_horizon(steps)  # type: ignore[attr-defined]

    def train(self) -> TrainHistory:
        hist = TrainHistory()
        for ep in range(self.episodes):
            state = self.env.reset(seed=episode_seed(self.base_seed, ep))
            ep_reward = 0.0
            lat, ene, losses = [], [], []
            for _ in range(self.steps):
                action = self.agent.act(state, explore=True)
                res = self.env.step(action)
                self.agent.observe(state, action, res.reward, res.state, res.done)
                if self.agent.learns:
                    loss = self.agent.update()
                    if loss is not None:
                        losses.append(loss)
                ep_reward += res.reward
                lat.append(res.info["latency_ms"])
                ene.append(res.info["energy_j"])
                state = res.state
                if res.done:
                    break
            self.agent.end_episode()
            hist.reward.append(ep_reward / max(1, self.steps))
            hist.latency_ms.append(float(np.mean(lat)))
            hist.energy_j.append(float(np.mean(ene)))
            hist.loss.append(float(np.mean(losses)) if losses else float("nan"))
        return hist

    def evaluate(self, episodes: int) -> List[Dict]:
        """Greedy rollout; return one record per routed task."""
        records: List[Dict] = []
        for ep in range(episodes):
            state = self.env.reset(seed=eval_seed(self.base_seed, ep))
            for step in range(self.steps):
                action = self.agent.act(state, explore=False)
                res = self.env.step(action)
                rec = {"eval_episode": ep, "step": step, "reward": res.reward}
                rec.update(res.info)
                records.append(rec)
                state = res.state
                if res.done:
                    break
        return records
