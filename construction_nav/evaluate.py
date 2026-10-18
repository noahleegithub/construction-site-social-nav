"""Predictor evaluation: ADE/FDE vs forecast horizon and agent count.

Evaluation is *open-loop on closed-loop rollouts*, the standard trajectory
prediction protocol: the environment is advanced under the nominal policy, at
every step the predictor is asked to forecast, and its forecast is compared with
the true positions that the rollout actually visited. This matches exactly how
the training data was produced (see :mod:`construction_nav.data.generate`).

Both unimodal predictors (mean only) and multimodal predictors (``K`` samples,
scored with best-of-K) are supported, so constant velocity, reduced Trajectron++
and the diffusion model can be compared through one function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from construction_nav import config as C
from construction_nav import metrics
from construction_nav.env import ConstructionEnv, generate_scene

DEFAULT_HORIZONS = (1, 2, 4, 8, 16)


@dataclass
class PredictorScore:
    """Per-horizon ADE / FDE (metres) plus bookkeeping."""

    name: str
    ade: dict[int, float] = field(default_factory=dict)
    fde: dict[int, float] = field(default_factory=dict)
    n_steps: int = 0
    n_agents: int = 0

    def row(self) -> str:
        parts = [f"{self.name:<24s}"]
        for h in sorted(self.ade):
            parts.append(f"H{h}:{self.ade[h]:5.2f}/{self.fde[h]:5.2f}")
        return "  ".join(parts)


def evaluate_predictor(
    predictor,
    seeds,
    horizons=DEFAULT_HORIZONS,
    num_agents: int = C.DEFAULT_AGENTS,
    arena: float = C.ARENA_SIZE,
    steps: int = 90,
    stride: int = 1,
) -> PredictorScore:
    """Score ``predictor`` on rollouts of one scene per seed.

    ``stride`` predicts only every ``stride``-th step (the ground truth still
    comes from the full rollout), which keeps the diffusion sampler's cost down
    on a CPU.
    """
    horizons = tuple(sorted(set(horizons)))
    sums = {h: [0.0, 0.0, 0] for h in horizons}   # ade, fde, count

    for seed in seeds:
        scene = generate_scene(seed=int(seed), num_agents=num_agents, arena=arena)
        env = ConstructionEnv(scene)
        outputs = []
        for i in range(steps):
            if i % stride == 0:
                outputs.append((i, predictor.predict(env.observe())))
            env.step()

        traj = np.array(env.agent_trajectories, dtype=float)   # (M, T+1, 2)
        n_frames = traj.shape[1]
        for i, out in outputs:
            for h in horizons:
                if i + 1 + h > n_frames:
                    continue
                gt = traj[:, i + 1 : i + 1 + h, :]              # (M, h, 2)
                if out.samples is not None:
                    pred = out.samples[:, :, :h, :]
                    a = metrics.min_ade(pred, gt)
                    f = metrics.min_fde(pred, gt)
                else:
                    pred = out.mean[:, :h, :]
                    a = metrics.ade(pred, gt)
                    f = metrics.fde(pred, gt)
                sums[h][0] += a
                sums[h][1] += f
                sums[h][2] += 1

    score = PredictorScore(
        name=predictor.name,
        n_agents=num_agents,
        n_steps=max(sums[h][2] for h in horizons),
    )
    for h in horizons:
        _, _, count = sums[h]
        if count:
            score.ade[h] = sums[h][0] / count
            score.fde[h] = sums[h][1] / count
    return score


def evaluate_many(
    predictors: dict[str, object],
    seeds,
    horizons=DEFAULT_HORIZONS,
    num_agents: int = C.DEFAULT_AGENTS,
    arena: float = C.ARENA_SIZE,
    steps: int = 90,
    stride: int = 1,
    log=print,
) -> dict[str, PredictorScore]:
    """Score several predictors on the same seeds."""
    results: dict[str, PredictorScore] = {}
    for name, predictor in predictors.items():
        score = evaluate_predictor(
            predictor, seeds, horizons=horizons, num_agents=num_agents,
            arena=arena, steps=steps, stride=stride,
        )
        results[name] = score
        log(score.row())
    return results


def scores_to_json(results: dict[str, PredictorScore], num_agents: int | None = None) -> dict:
    """Serialise a set of scores (per agent count) for a results file."""
    return {
        "num_agents": num_agents,
        "predictors": {
            name: {
                "ade": {str(h): float(v) for h, v in s.ade.items()},
                "fde": {str(h): float(v) for h, v in s.fde.items()},
                "n_steps": s.n_steps,
            }
            for name, s in results.items()
        },
    }
