"""Benchmark charts: ADE / FDE versus forecast horizon and agent count."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from construction_nav.evaluate import PredictorScore

# Consistent colours across figures.
COLORS = {
    "constant velocity": "#1f77b4",
    "trajectron++ (reduced)": "#ff7f0e",
    "diffusion (K=6)": "#2ca02c",
}
FALLBACK = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _color(name: str, i: int) -> str:
    return COLORS.get(name, FALLBACK[i % len(FALLBACK)])


def _save(fig, out_dir, name: str, dpi: int = 220) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        p = out_path / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


def plot_ade_vs_horizon(
    results: Mapping[str, PredictorScore],
    out_dir: str | os.PathLike,
    num_agents: int | None = None,
    dpi: int = 220,
) -> list[Path]:
    """Two panels: ADE and FDE against forecast horizon ``H``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (name, s) in enumerate(results.items()):
        hs = sorted(s.ade)
        c = _color(name, i)
        axes[0].plot(hs, [s.ade[h] for h in hs], "o-", color=c, label=name)
        axes[1].plot(hs, [s.fde[h] for h in hs], "s--", color=c, label=name)

    for ax, title in zip(axes, ("ADE vs horizon", "FDE vs horizon")):
        ax.set_xlabel("forecast horizon H (steps)")
        ax.set_ylabel("error (m)")
        ax.set_title(title + (f"  (M={num_agents})" if num_agents else ""))
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted({h for s in results.values() for h in s.ade}))
    axes[0].legend(fontsize=9)
    fig.suptitle("Forecast error vs horizon", y=1.02)
    return _save(fig, out_dir, "benchmark_horizon", dpi=dpi)


def plot_training_curves(
    histories: Mapping[str, dict],
    out_dir: str | os.PathLike,
    dpi: int = 220,
) -> list[Path]:
    """Train/val loss curves for the learned predictors."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (name, hist) in enumerate(histories.items()):
        c = _color(name, i)
        ax.plot(hist["train"], color=c, label=f"{name} (train)")
        if hist.get("val"):
            ax.plot(hist["val"], color=c, linestyle="--", label=f"{name} (val)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("Training curves")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "training_curves", dpi=dpi)


def plot_planner_comparison(
    reports: Mapping[str, object],
    out_dir: str | os.PathLike,
    dpi: int = 220,
) -> list[Path]:
    """Grouped bars: collisions, cost ``J``, intrusion and travel time."""
    names = list(reports)
    order = sorted(names, key=lambda n: reports[n].mean_J)
    colors = [_color(n, i) for i, n in enumerate(order)]

    panels = [
        ("collision steps (total)", [reports[n].n_collisions for n in order]),
        ("cost J", [reports[n].mean_J for n in order]),
        ("intrusion integral", [reports[n].mean_intrusion for n in order]),
        ("travel time (s)", [reports[n].mean_travel_time for n in order]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    short = [n.replace("joint planner ", "joint ").replace(" (reduced)", "") for n in order]
    for ax, (title, values) in zip(axes.ravel(), panels):
        ax.bar(range(len(order)), values, color=colors)
        ax.set_title(title)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(short, rotation=20, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    mean_success = float(np.mean([reports[n].success_rate for n in order]))
    n_trials = len(next(iter(reports.values())).trials)
    fig.suptitle(
        f"Randomized-trial planner comparison  "
        f"(goal reached in {mean_success:.0%} of {n_trials} trials)", y=1.0
    )
    fig.tight_layout()
    return _save(fig, out_dir, "planner_comparison", dpi=dpi)


def plot_lambda_sweep(
    sweeps: Mapping[str, Mapping[float, object]],
    out_dir: str | os.PathLike,
    dpi: int = 220,
) -> list[Path]:
    """Cost / intrusion / travel time as the intrusion weight ``lambda_1`` varies.

    ``sweeps`` maps a series label (e.g. "with safety filter") to its own
    ``{lambda_1: report}`` mapping, so the soft-term trade-off can be shown with
    and without the reactive hard-constraint filter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for j, (label, sweep) in enumerate(sweeps.items()):
        lambdas = sorted(sweep)
        reports = [sweep[l] for l in lambdas]
        color = FALLBACK[j % len(FALLBACK)]
        for ax, (title, attr) in zip(
            axes,
            (("cost J", "mean_J"), ("intrusion integral", "mean_intrusion"),
             ("travel time (s)", "mean_travel_time")),
        ):
            ax.plot(lambdas, [getattr(r, attr) for r in reports], "o-",
                    color=color, label=label)
            ax.set_xlabel(r"$\lambda_1$")
            ax.set_title(title)
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Intrusion-weight sweep (joint diffusion planner)", y=1.03)
    return _save(fig, out_dir, "lambda_sweep", dpi=dpi)


def _draw_scene(ax, scene) -> None:
    """Draw the static worksite (hazard field, highways, zones, workstations)."""
    from construction_nav.render import (
        draw_hazard_field,
        draw_highway,
        draw_workstations,
        draw_zones,
    )

    arena = scene.arena_size
    draw_hazard_field(ax, arena)
    for highway in scene.highways:
        draw_highway(ax, highway, scene)
    draw_zones(ax, scene)
    draw_workstations(ax, scene)
    return arena


def plot_failure_cases(
    failures,
    out_dir: str | os.PathLike,
    dpi: int = 220,
) -> list[Path]:
    """Grid of failed trials (collisions marked) for the joint planner."""
    from construction_nav.render import STYLE

    if not failures:
        return []
    n = len(failures)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 6.0 * rows), squeeze=False)
    for ax, (result, env) in zip(axes.ravel(), failures):
        scene = env.scene
        arena = scene.arena_size
        _draw_scene(ax, scene)
        for i, traj in enumerate(env.agent_trajectories):
            arr = np.array(traj)
            kind = scene.agents[i].kind
            ax.plot(arr[:, 0], arr[:, 1], color=STYLE[kind]["color"], lw=1.0, alpha=0.7)
        robot = np.array(env.robot_trajectory)
        ax.plot(robot[:, 0], robot[:, 1], color=STYLE["robot_color"], lw=2.2, label="robot")
        ax.plot(robot[0, 0], robot[0, 1], marker="s", ms=8, color=STYLE["robot_color"],
                markeredgecolor="black")
        goal = scene.goal_pos
        ax.plot(goal[0], goal[1], marker="*", ms=18, color=STYLE["goal_color"],
                markeredgecolor="black", label="goal")

        # Highlight steps where the robot was inside another agent's collision disk.
        robot_arr = robot
        worst = None
        for i, traj in enumerate(env.agent_trajectories):
            arr = np.array(traj)
            k = min(len(arr), len(robot_arr))
            d = np.linalg.norm(arr[:k] - robot_arr[:k], axis=-1)
            hit = d < scene.agents[i].D_S
            if hit.any():
                ax.plot(robot_arr[:k][hit, 0], robot_arr[:k][hit, 1], "o",
                        ms=7, mfc="none", mec="red", mew=1.6)
                if worst is None or d.min() < worst:
                    worst = d.min()
        ax.set_xlim(0, arena)
        ax.set_ylim(0, arena)
        ax.set_aspect("equal")
        ax.set_title(
            f"seed={result.seed}  reached={result.reached}  "
            f"collisions={result.n_collision_steps}  "
            f"intrusion={result.intrusion_integral:.1f}",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Failure / hard cases (joint diffusion planner)", y=0.995)
    fig.tight_layout()
    return _save(fig, out_dir, "failure_cases", dpi=dpi)


def plot_ade_vs_agents(
    by_agents: Mapping[int, Mapping[str, PredictorScore]],
    out_dir: str | os.PathLike,
    horizon: int = 16,
    dpi: int = 220,
) -> list[Path]:
    """ADE / FDE at a fixed horizon against the number of agents ``M``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    agents = sorted(by_agents)
    names = list(next(iter(by_agents.values())).keys())
    for i, name in enumerate(names):
        c = _color(name, i)
        ade = [by_agents[m][name].ade.get(horizon, np.nan) for m in agents]
        fde = [by_agents[m][name].fde.get(horizon, np.nan) for m in agents]
        axes[0].plot(agents, ade, "o-", color=c, label=name)
        axes[1].plot(agents, fde, "s--", color=c, label=name)

    for ax, title in zip(axes, (f"ADE at H={horizon}", f"FDE at H={horizon}")):
        ax.set_xlabel("number of agents M")
        ax.set_ylabel("error (m)")
        ax.set_title(title)
        ax.set_xticks(agents)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Forecast error vs agent count", y=1.02)
    return _save(fig, out_dir, "benchmark_agents", dpi=dpi)
