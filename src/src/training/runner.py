"""Run a single (agent, seed, weight, topology) configuration end to end.

Builds the environment and agent, trains, evaluates, and returns both the
per-task evaluation records (tagged with run metadata) and the per-episode
convergence history. This is the unit of work the experiment drivers fan out
over seeds, weights, and scalability settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..agents.registry import make_agent
from ..common.config import Config
from ..common.seeding import seed_everything
from ..env.sim_env import SimEnv
from .trainer import Trainer


@dataclass
class RunOutput:
    records: List[Dict]          # per-task evaluation rows (with metadata)
    convergence: List[Dict]      # per-episode rows (reward/latency/energy/loss)


def run_config(
    cfg: Config,
    agent_name: str,
    seed: int,
    weight: Tuple[float, float],
    quick: bool = False,
    iot_nodes: int | None = None,
    arrival_rate: float | None = None,
    eval_episodes: int = 5,
    device: str = "cpu",
) -> RunOutput:
    rng = seed_everything(seed)
    env = SimEnv(cfg, weight=weight, iot_nodes=iot_nodes, arrival_rate=arrival_rate)
    agent = make_agent(agent_name, cfg, rng, device)

    episodes = cfg.episodes_for(quick)
    steps = cfg.steps_for(quick)
    trainer = Trainer(env, agent, episodes=episodes, steps=steps, base_seed=seed)

    history = trainer.train()
    eval_records = trainer.evaluate(episodes=eval_episodes)

    meta = {
        "agent": agent_name,
        "seed": seed,
        "w_latency": weight[0],
        "w_energy": weight[1],
        "iot_nodes": iot_nodes if iot_nodes is not None else cfg.topology.iot_nodes,
        "arrival_rate": arrival_rate if arrival_rate is not None
        else cfg.topology.arrival_rate,
    }
    for r in eval_records:
        r.update(meta)

    convergence = [
        {
            "episode": i,
            "reward": history.reward[i],
            "latency_ms": history.latency_ms[i],
            "energy_j": history.energy_j[i],
            "loss": history.loss[i],
            **meta,
        }
        for i in range(len(history.reward))
    ]
    return RunOutput(records=eval_records, convergence=convergence)
