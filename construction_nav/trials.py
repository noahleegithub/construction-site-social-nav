"""Randomized-trial evaluation of robot planners (proposal Section 5).

Runs a planner on a batch of randomized construction sites and aggregates the
Eq. (3) cost ``J = T_travel + lambda_1 * integral(max(0, -h_C)) + lambda_2 * T_freeze``
together with success, collision and freeze statistics. Used by
``scripts/evaluate_planners.py`` for the planner comparison, the ``lambda_1``
sweep and the failure-case study.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from construction_nav import config as C
from construction_nav.env import ConstructionEnv, generate_scene
from construction_nav.metrics import summarize_trial

DEFAULT_MAX_STEPS = 4000


@dataclass
class TrialResult:
    """Outcome of one closed-loop trial."""

    seed: int
    num_agents: int
    reached: bool
    travel_time: float
    intrusion_integral: float
    freeze_time: float
    n_collision_steps: int
    J: float
    goals_reached: int = 0
    # Distinct contact episodes, summed over agents: an agent that stays inside
    # the hard radius for ten steps contributes one, not ten.
    n_collision_events: int = 0

    @property
    def collided(self) -> bool:
        return self.n_collision_events > 0


def run_trial(
    planner,
    seed: int,
    num_agents: int = C.DEFAULT_AGENTS,
    arena: float = C.ARENA_SIZE,
    max_steps: int = DEFAULT_MAX_STEPS,
    lambda_intrusion: float = 1.0,
    lambda_freeze: float = 0.5,
    return_env: bool = False,
) -> TrialResult | tuple[TrialResult, ConstructionEnv]:
    """Run one closed-loop trial and summarise it.

    ``planner=None`` runs the built-in goal-seeking path follower, which is
    the no-avoidance floor of the comparison.
    """
    scene = generate_scene(seed=seed, num_agents=num_agents, arena=arena)
    env = ConstructionEnv(scene)
    reached = False
    for _ in range(max_steps):
        info = env.step(None if planner is None else planner(env))
        if info.robot_reached_goal:
            # Keep going to the next goal; the run ends after the last one.
            if not env.advance_robot_goal():
                reached = True
                break
    summary = summarize_trial(env, lambda_intrusion, lambda_freeze)
    result = TrialResult(
        seed=int(seed),
        num_agents=int(num_agents),
        reached=bool(reached),
        travel_time=float(summary["travel_time"]),
        intrusion_integral=float(summary["intrusion_integral"]),
        freeze_time=float(summary["freeze_time"]),
        n_collision_steps=int(summary["n_collision_steps"]),
        J=float(summary["J"]),
        goals_reached=int(env.robot.goals_reached),
        n_collision_events=int(summary["n_collision_events"]),
    )
    if return_env:
        return result, env
    return result


@dataclass
class PlannerReport:
    """Aggregated results for one planner over a set of trials."""

    name: str
    trials: list[TrialResult] = field(default_factory=list)

    def add(self, result: TrialResult) -> None:
        self.trials.append(result)

    # -- aggregates --------------------------------------------------------

    def _mean(self, attr: str) -> float:
        return float(np.mean([getattr(t, attr) for t in self.trials])) if self.trials else 0.0

    @property
    def success_rate(self) -> float:
        return float(np.mean([t.reached for t in self.trials])) if self.trials else 0.0

    @property
    def collision_rate(self) -> float:
        return float(np.mean([t.collided for t in self.trials])) if self.trials else 0.0

    @property
    def mean_J(self) -> float:
        return self._mean("J")

    @property
    def mean_travel_time(self) -> float:
        return self._mean("travel_time")

    @property
    def mean_intrusion(self) -> float:
        return self._mean("intrusion_integral")

    @property
    def mean_freeze_time(self) -> float:
        return self._mean("freeze_time")

    @property
    def n_collisions(self) -> int:
        """Total contact episodes over all trials (the headline count)."""
        return int(sum(t.n_collision_events for t in self.trials))

    @property
    def n_collision_steps(self) -> int:
        """Total violating steps; kept to show how long contacts lasted."""
        return int(sum(t.n_collision_steps for t in self.trials))

    def to_json(self) -> dict:
        return {
            "planner": self.name,
            "n_trials": len(self.trials),
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "n_collision_events": self.n_collisions,
            "n_collision_steps": self.n_collision_steps,
            "mean_J": self.mean_J,
            "mean_travel_time": self.mean_travel_time,
            "mean_intrusion": self.mean_intrusion,
            "mean_freeze_time": self.mean_freeze_time,
            "trials": [t.__dict__ for t in self.trials],
        }


def evaluate_planners(
    planners: dict[str, object],
    seeds,
    num_agents: int = C.DEFAULT_AGENTS,
    arena: float = C.ARENA_SIZE,
    max_steps: int = DEFAULT_MAX_STEPS,
    lambda_intrusion: float = 1.0,
    lambda_freeze: float = 0.5,
    log=print,
) -> dict[str, PlannerReport]:
    """Run every planner on the same randomized sites and aggregate."""
    seeds = list(seeds)
    reports: dict[str, PlannerReport] = {}
    for name, planner in planners.items():
        report = PlannerReport(name=name)
        for seed in seeds:
            report.add(
                run_trial(
                    planner, seed, num_agents=num_agents, arena=arena,
                    max_steps=max_steps, lambda_intrusion=lambda_intrusion,
                    lambda_freeze=lambda_freeze,
                )
            )
        reports[name] = report
        log(
            f"{name:<32} success {report.success_rate:5.1%}  "
            f"J {report.mean_J:7.2f}  T {report.mean_travel_time:5.1f}s  "
            f"intrusion {report.mean_intrusion:6.2f}  "
            f"contacts {report.n_collisions} ({report.n_collision_steps} steps)"
        )
    return reports


def collect_failure_cases(
    planner,
    seeds,
    num_agents: int = C.DEFAULT_AGENTS,
    arena: float = C.ARENA_SIZE,
    max_steps: int = DEFAULT_MAX_STEPS,
    limit: int = 4,
    log=print,
) -> list[tuple[TrialResult, ConstructionEnv]]:
    """Return the ``limit`` hardest trials: genuine failures first, then near-misses.

    Ranked by (timed out, collided, intrusion, cost), so the figure always shows
    the worst cases even when the planner never outright fails.
    """
    results: list[tuple[TrialResult, ConstructionEnv]] = []
    for seed in seeds:
        result, env = run_trial(
            planner, seed, num_agents=num_agents, arena=arena,
            max_steps=max_steps, return_env=True,
        )
        results.append((result, env))

    if not results:
        return []
    results.sort(
        key=lambda item: (
            item[0].reached,
            item[0].n_collision_events == 0,
            -item[0].intrusion_integral,
            -item[0].J,
        )
    )
    chosen = results[:limit]
    for result, _env in chosen:
        log(
            f"case seed={result.seed} reached={result.reached} "
            f"contacts={result.n_collision_events} "
            f"intrusion={result.intrusion_integral:.2f} t={result.travel_time:.1f}s"
        )
    return chosen
