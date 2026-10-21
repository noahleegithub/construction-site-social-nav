"""Trial plots: recorded trajectories over the worksite and metric charts."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from construction_nav import config as C
from construction_nav.env import ConstructionEnv
from construction_nav.graph import route_points
from construction_nav.render import (
    STYLE,
    draw_hazard_field,
    draw_highway,
    draw_workstations,
)


def plot_trial(
    scene,
    out_dir: str | os.PathLike = "figures",
    name: str = "trial",
    planner=None,
    steps: int = 3500,
    dpi: int = 200,
) -> tuple[list[Path], dict]:
    """Run a trial and plot every agent's and the robot's path over the arena."""
    # The environment advances the scene's agents in place; copy so callers can
    # render the same scene more than once (e.g. PNG then GIF) from t = 0.
    scene = copy.deepcopy(scene)
    env = ConstructionEnv(scene)
    reached_at = None
    for _ in range(steps):
        info = env.step(None if planner is None else planner(env))
        if info.robot_reached_goal:
            if env.advance_robot_goal():
                continue
            reached_at = env.t
            break

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    arena = scene.arena_size

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_hazard_field(ax, arena)
    for highway in scene.highways:
        draw_highway(ax, highway, scene)
    draw_workstations(ax, scene)

    for i, traj in enumerate(env.agent_trajectories):
        arr = np.array(traj)
        kind = scene.agents[i].kind
        ax.plot(arr[:, 0], arr[:, 1], color=STYLE[kind]["color"], lw=1.0, alpha=0.7,
                zorder=5)
        ax.plot(arr[0, 0], arr[0, 1], marker="o", ms=4,
                color=STYLE[kind]["color"], zorder=6)
        ax.plot(arr[-1, 0], arr[-1, 1], marker="x", ms=5,
                color=STYLE[kind]["color"], zorder=6)

    # The planned itinerary (every leg), so the executed path can be compared
    # with what the robot intended before the crowd got in the way.
    planned = route_points(scene, scene.planned_route)
    ax.plot(planned[:, 0], planned[:, 1],
            linestyle=(0, (5, 5)), color=STYLE["goal_color"], lw=2.0,
            alpha=0.9, zorder=7, label="planned route")

    robot = np.array(env.robot_trajectory)
    ax.plot(robot[:, 0], robot[:, 1], color=STYLE["robot_color"], lw=2.2, zorder=8,
            label="robot path")
    ax.plot(robot[0, 0], robot[0, 1], marker="o", ms=9,
            color=STYLE["robot_color"], markeredgecolor="black", zorder=9)
    goal = scene.final_goal_pos
    ax.plot(goal[0], goal[1], marker="*", ms=20, color=STYLE["goal_color"],
            markeredgecolor="black", zorder=9, label="goal")

    ax.set_xlim(0, arena)
    ax.set_ylim(0, arena)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Trial paths  seed={scene.seed}  M={len(scene.agents)}  "
        f"goals={env.robot.goals_reached}/{len(env.robot.goals)}  t={env.t:.1f}s  "
        f"collisions={env.n_collision_steps}",
        fontsize=11,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)

    png = out_path / f"{name}.png"
    pdf = out_path / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "reached": reached_at is not None,
        "t": env.t,
        "travel_time": env.travel_time,
        "intrusion_integral": env.intrusion_integral,
        "freeze_time": env.freeze_time,
        "n_collision_steps": env.n_collision_steps,
    }
    return [png, pdf], summary


def plot_metric_curve(
    curve: np.ndarray | list[float],
    x: np.ndarray | list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_dir: str | os.PathLike,
    name: str,
    dpi: int = 200,
) -> Path:
    """Generic single-series line chart for evaluation sweeps."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, curve, marker="o", color="#1f6feb")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    png = out_path / f"{name}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png
