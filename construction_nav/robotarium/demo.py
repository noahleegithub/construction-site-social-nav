"""Social-navigation demos on the Robotarium simulator.

Builds miniature construction sites that fit the Robotarium arena, runs the
robot policy alongside the scripted agents, and reports both the site-frame
metrics (Eq. (2)/(3)) and the Robotarium's own collision / boundary validation
(``debug()``).

Seeds are *ranked* by a cheap site-frame trial (goal reached, no collisions, and
a close encounter of the requested kind) and then confirmed on the Robotarium, so
the shipped demos are interesting **and** successful rather than arbitrary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from construction_nav.env import ConstructionEnv
from construction_nav.planner import FixedRadiusPlanner
from construction_nav.render_anim import animate_trial
from construction_nav.render_trials import plot_trial
from construction_nav.robotarium.arena import robotarium_scene
from construction_nav.robotarium.bridge import run_robotarium_trial


@dataclass
class DemoCase:
    """A named Robotarium demo scenario."""

    name: str
    description: str
    encounter: str            # "human" | "vehicle" | "any"
    seed: int = 0
    num_agents: int = 5


DEFAULT_CASES = [
    DemoCase("social_navigation", "Robot crosses the site while agents work and travel.",
             encounter="any"),
    DemoCase("human_encounter", "Robot passes close to a human worker (needs yielding).",
             encounter="human"),
    DemoCase("vehicle_encounter", "Robot encounters heavy machinery on a shared corridor.",
             encounter="vehicle"),
]


def site_trial(
    scene, encounter: str = "any", steps: int = 4000, planner=None
) -> dict:
    """Run one site-frame trial and return success + encounter statistics."""
    env = ConstructionEnv(scene)
    planner = planner or FixedRadiusPlanner(use_oracle=False)
    reached = False
    min_dist = float("inf")
    for _ in range(steps):
        info = env.step(planner(env))
        for agent in scene.agents:
            if encounter != "any" and agent.kind != encounter:
                continue
            min_dist = min(min_dist, float(np.linalg.norm(env.robot.pos - agent.pos)))
        if info.robot_reached_goal:
            reached = True
            break
    return {
        "seed": scene.seed,
        "reached": reached,
        "collisions": env.n_collision_steps,
        "intrusion_integral": env.intrusion_integral,
        "min_encounter": min_dist,
        "travel_time": env.travel_time,
    }


def rank_seeds(
    encounter: str,
    num_agents: int,
    n_seeds: int = 8,
    close_fraction: float = 0.45,
    exclude: set[int] | None = None,
    planner=None,
) -> list[dict]:
    """Order seeds so that safe, successful, close-encounter scenarios come first.

    Ordering is: reached-and-collision-free first, then fewest collisions, then a
    genuine encounter (minimum robot--agent distance inside ``close_fraction`` of
    the arena), then the quickest trial. ``exclude`` drops seeds already used by
    earlier demo cases so the shipped figures show distinct scenarios.
    """
    exclude = exclude or set()
    seeds = [s for s in range(n_seeds) if s not in exclude]
    stats = [
        site_trial(robotarium_scene(seed=seed, num_agents=num_agents), encounter, planner=planner)
        for seed in seeds
    ]
    if not stats:
        return []

    close_threshold = close_fraction * robotarium_scene(
        seed=seeds[0], num_agents=num_agents
    ).arena_size

    def key(s: dict):
        safe = s["reached"] and s["collisions"] == 0
        close = s["min_encounter"] <= close_threshold
        return (
            0 if safe else 1,
            s["collisions"],
            0 if close else 1,
            s["travel_time"],
            s["min_encounter"],
        )

    return sorted(stats, key=key)


def run_demo(
    case: DemoCase,
    out_dir: str | os.PathLike = "figures",
    barrier_mode: str = "all",
    max_steps: int = 4000,
    animate: bool = True,
    seed_attempts: int = 4,
    n_seeds: int = 8,
    exclude: set[int] | None = None,
    planner=None,
) -> dict:
    """Run one demo case: rank seeds, confirm on the Robotarium, then plot it."""
    ranked = rank_seeds(
        case.encounter, case.num_agents, n_seeds=n_seeds, exclude=exclude,
        planner=planner,
    )
    if not ranked:  # every candidate seed was claimed by an earlier case
        ranked = rank_seeds(case.encounter, case.num_agents, n_seeds=n_seeds,
                            planner=planner)

    scene = None
    robotarium = None
    chosen = None
    fallback = None
    for stats in ranked[:seed_attempts]:
        candidate = robotarium_scene(seed=stats["seed"], num_agents=case.num_agents)
        result = run_robotarium_trial(
            candidate, planner=planner or FixedRadiusPlanner(use_oracle=False),
            barrier_mode=barrier_mode, max_steps=max_steps,
        )
        record = (stats, candidate, result)
        if fallback is None:
            fallback = record
        if result.reached_goal and result.validation.get("robots_too_close", 0) == 0:
            chosen = record
            break
        if chosen is None and result.reached_goal:
            chosen = record
    chosen = chosen or fallback
    stats, _, robotarium = chosen

    # Build a *fresh* scene for the plotted site trial: the Robotarium run above
    # mutates the poses of the scene it was given, so reusing it would start the
    # trial from post-run agent positions.
    scene = robotarium_scene(seed=stats["seed"], num_agents=case.num_agents)
    planner = planner or FixedRadiusPlanner(use_oracle=False)
    paths, summary = plot_trial(scene, out_dir, name=f"demo_{case.name}", planner=planner)
    if animate:
        paths += animate_trial(
            scene, out_dir, name=f"demo_{case.name}", planner=planner, steps=4000,
        )

    return {
        "case": case.name,
        "description": case.description,
        "seed": scene.seed,
        "num_agents": case.num_agents,
        "arena": scene.arena_size,
        "encounter_kind": case.encounter,
        "min_encounter": stats["min_encounter"],
        "site": summary,
        "robotarium": {
            "reached": robotarium.reached_goal,
            "t": robotarium.t,
            "steps": robotarium.steps,
            "validation": robotarium.validation,
            "site_collision_steps": robotarium.n_collision_steps,
        },
        "figures": [str(p) for p in paths],
    }


def run_all(
    out_dir: str | os.PathLike = "figures",
    cases: list[DemoCase] | None = None,
    **kwargs,
) -> list[dict]:
    """Run every demo case, giving each a distinct seed where possible."""
    cases = cases or DEFAULT_CASES
    results: list[dict] = []
    used: set[int] = set()
    for case in cases:
        result = run_demo(case, out_dir=out_dir, exclude=used, **kwargs)
        results.append(result)
        used.add(result["seed"])
    return results
