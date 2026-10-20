"""Schematic MotionDiffuser-style prediction graphics.

These are *illustrative* figures, not real model output: a synthetic multimodal
sampler stands in for MotionDiffuser so the graphics can be produced without a
checkpoint, dataset, or GPU. Two static figures are written:

    diffusion_sequence  - reverse diffusion where sampled forward-reachable
                          endpoints converge onto a few final paths for one
                          focus agent.
    pipeline            - horizontal schematic of MotionDiffuser's five
                          pipeline steps, framed for the construction site.

Ported from ``visualize_predictions.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from construction_nav.entities import Agent, Scene
from construction_nav.graph import (
    arc_lengths,
    point_on_polyline,
    project_arc_length,
    route_points,
)
from construction_nav.render import render_arena_backdrop

HORIZON_STEPS = 16                 # forecast horizon H
DT = 0.5                           # seconds per step
N_SAMPLES = 16                     # trajectories sampled at t = T
FINAL_MODES_MAX = 3                # samples converge to a random 1..3 paths
DIFFUSION_STEPS = 5                # panels in the denoising sequence
NOMINAL_SPEED = {"human": 1.2, "vehicle": 3.0}   # m/s
ENDPOINT_RANGE = (5.0, 25.0)       # meters, forward-reachable endpoint band
ENDPOINT_HALF_ANGLE = np.deg2rad(60.0)
ZOOM_RADIUS = 18.0                 # meters, half-width of the zoomed crop


@dataclass
class Mode:
    """One sampled future trajectory (a diffusion mode)."""

    weight: float
    positions: np.ndarray           # (H, 2) meters
    speed_scale: float
    dwell_fraction: float
    lateral: float


def _focus_agent(scene: Scene) -> Agent:
    """Pick the moving agent with the longest route (best figure subject)."""
    moving = [a for a in scene.agents if a.state == "moving"]
    pool = moving or list(scene.agents)

    def route_len(agent: Agent) -> float:
        pts = route_points(scene, agent.route)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    return max(pool, key=route_len)


def _arc_position(points: np.ndarray, pos: np.ndarray) -> float:
    return project_arc_length(points, pos)


def sample_modes(
    scene: Scene,
    agent: Agent,
    rng: np.random.Generator,
    n_modes: int = FINAL_MODES_MAX,
    horizon: int = HORIZON_STEPS,
    dt: float = DT,
) -> list[Mode]:
    """Sample multimodal futures by rolling the agent along its route with
    per-mode speed, dwell, and lateral variation."""
    points = route_points(scene, agent.route)
    cum, _ = arc_lengths(points)
    total = float(cum[-1])
    s0 = _arc_position(points, agent.pos)
    speed = NOMINAL_SPEED[agent.kind]
    horizon_time = horizon * dt

    weights = rng.dirichlet(np.ones(n_modes))
    modes: list[Mode] = []
    for k in range(n_modes):
        speed_scale = rng.uniform(0.5, 1.5)
        dwell_fraction = rng.uniform(0.0, 0.7)
        lateral = rng.uniform(-1.5, 1.5)
        positions = np.zeros((horizon, 2))
        for j in range(horizon):
            t = (j + 1) * dt
            moving_time = max(0.0, t - dwell_fraction * horizon_time)
            s = min(total, s0 + speed * speed_scale * moving_time)
            pos, heading = point_on_polyline(points, s)
            normal = np.array([-heading[1], heading[0]])
            positions[j] = pos + normal * lateral
        modes.append(Mode(weight=float(weights[k]), positions=positions,
                          speed_scale=speed_scale, dwell_fraction=dwell_fraction,
                          lateral=lateral))
    return modes


def sample_endpoints(
    scene: Scene,
    agent: Agent,
    rng: np.random.Generator,
    n_samples: int = N_SAMPLES,
) -> np.ndarray:
    """Sample endpoints in the agent's forward-reachable region (inside arena)."""
    arena = scene.arena_size
    lo, hi = ENDPOINT_RANGE
    # Scale the reachable band with the arena (small arenas -> shorter reach).
    k = arena / 60.0
    lo, hi = lo * k, hi * k
    endpoints = np.zeros((n_samples, 2))
    for i in range(n_samples):
        for _ in range(200):
            distance = rng.uniform(lo, hi)
            angle = rng.uniform(-ENDPOINT_HALF_ANGLE, ENDPOINT_HALF_ANGLE)
            direction = _rotate(agent.heading, angle)
            endpoint = agent.pos + direction * distance
            if np.all(endpoint >= 1.0) and np.all(endpoint <= arena - 1.0):
                endpoints[i] = endpoint
                break
        else:
            endpoints[i] = np.clip(endpoint, 1.0, arena - 1.0)
    return endpoints


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([cos * vector[0] - sin * vector[1], sin * vector[0] + cos * vector[1]])


def bezier_path(
    start: np.ndarray,
    start_heading: np.ndarray,
    end: np.ndarray,
    n_steps: int = HORIZON_STEPS,
) -> np.ndarray:
    """Cubic Bezier from ``start`` to ``end`` leaving along ``start_heading`` and
    arriving along the start->end direction, sampled at ``n_steps`` waypoints."""
    delta = end - start
    distance = float(np.linalg.norm(delta))
    end_heading = delta / distance if distance > 1e-9 else start_heading
    control = 0.35 * distance
    p0 = start
    p1 = start + start_heading * control
    p2 = end - end_heading * control
    p3 = end
    t = np.linspace(0.0, 1.0, n_steps)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2 + t**3 * p3)


def assign_modes(endpoints: np.ndarray, mode_ends: np.ndarray) -> np.ndarray:
    """Assign each sample to its nearest mode, guaranteeing every mode is used."""
    n_modes = len(mode_ends)
    distances = np.linalg.norm(endpoints[:, None, :] - mode_ends[None, :, :], axis=2)
    assign = np.argmin(distances, axis=1)
    counts = np.bincount(assign, minlength=n_modes)
    for k in np.where(counts == 0)[0]:
        for i in np.argsort(distances[:, k]):
            if counts[assign[i]] > 1:
                counts[assign[i]] -= 1
                assign[i] = k
                counts[k] += 1
                break
    return assign


def converge_sequence(
    initial_paths: np.ndarray,
    clean_paths: np.ndarray,
    n_steps: int = DIFFUSION_STEPS,
) -> list[np.ndarray]:
    """Fake reverse diffusion: interpolate each sampled path toward its assigned
    clean (mode) path with a cosine schedule."""
    steps: list[np.ndarray] = []
    for i in range(n_steps):
        alpha = 0.5 * (1.0 - np.cos(np.pi * i / (n_steps - 1))) if n_steps > 1 else 1.0
        steps.append((1.0 - alpha) * initial_paths + alpha * clean_paths)
    return steps


def render_diffusion_sequence(
    scene: Scene,
    out_dir: str | os.PathLike,
    n_samples: int = N_SAMPLES,
    final_modes_max: int = FINAL_MODES_MAX,
    n_steps: int = DIFFUSION_STEPS,
    dpi: int = 300,
) -> list[Path]:
    rng = np.random.default_rng(scene.seed + 101)
    agent = _focus_agent(scene)
    arena = scene.arena_size

    backdrop = plt.imread(render_arena_backdrop(scene, out_dir))

    endpoints = sample_endpoints(scene, agent, rng, n_samples)
    initial_paths = np.stack([bezier_path(agent.pos, agent.heading, e) for e in endpoints])

    n_final = int(rng.integers(1, final_modes_max + 1))
    modes = sample_modes(scene, agent, rng, n_modes=n_final)
    mode_ends = np.array([m.positions[-1] for m in modes])
    assign = assign_modes(endpoints, mode_ends)
    clean_paths = np.stack([modes[k].positions for k in assign])
    steps = converge_sequence(initial_paths, clean_paths, n_steps)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, n_final))

    half = ZOOM_RADIUS * (arena / 60.0)
    center = agent.pos if arena >= 2 * half else np.full(2, arena / 2.0)
    center = np.clip(center, half, arena - half) if arena >= 2 * half else center
    lo = center - half
    hi = center + half

    fig, axes = plt.subplots(1, n_steps, figsize=(3.1 * n_steps, 3.6))
    if n_steps == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(backdrop, extent=[0, arena, 0, arena], origin="upper",
                  interpolation="bilinear", zorder=0)
        trajectory = steps[i]
        for s in range(n_samples):
            ax.plot(trajectory[s, :, 0], trajectory[s, :, 1],
                    color=colors[assign[s]], linewidth=1.3, alpha=0.6, zorder=5)
        ax.plot(agent.pos[0], agent.pos[1], marker="o", markersize=6,
                markerfacecolor="black", markeredgecolor="white", zorder=7)
        ax.annotate("", xy=agent.pos + agent.heading * 2.5 * (arena / 60.0),
                    xytext=agent.pos,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4), zorder=7)
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        fraction = 1.0 - i / (n_steps - 1) if n_steps > 1 else 0.0
        if i == 0:
            ax.set_title("t = T", fontsize=10)
        elif i == n_steps - 1:
            ax.set_title("t = 0", fontsize=10)
        else:
            ax.set_title(f"t = {fraction:g}T", fontsize=10)

    handles = [plt.Line2D([], [], color=colors[k], lw=2, label="") for k in range(n_final)]
    fig.legend(handles=handles, loc="lower center", ncol=n_final, frameon=False, fontsize=8)
    fig.suptitle("MotionDiffuser - Reverse Diffusion Process\n"
                 f"focus agent: {agent.kind}, seed={scene.seed}", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    return _save(fig, out_dir, "diffusion_sequence", dpi)


def render_pipeline(out_dir: str | os.PathLike, dpi: int = 300) -> list[Path]:
    """Schematic of MotionDiffuser's five pipeline steps."""
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    steps = [
        ("1. Scene context\nencoding",
         "worksite map + agent\nhistories \u2192 context\nvector (Transformer encoder)"),
        ("2. PCA trajectory\ncompression",
         "multi-agent futures \u2192\nlow-dim latent space\n(smoothness & faster denoising)"),
        ("3. Diffusion process\n(DDPM)",
         "train: forward noising\n+ L2 denoising loss\ninference: reverse denoising\nfrom Gaussian noise"),
        ("4. Constrained &\nguided sampling",
         "differentiable costs\ninjected to encourage safe paths"),
        ("5. Permutation-invariant\njoint decoding",
         "outputs trajectories for all\nagents + robot"),
    ]
    n = len(steps)
    width, height = 2.0, 1.9
    gap = (14.0 - n * width) / (n + 1)
    y = 2.9
    xs: list[float] = []
    for i, (title, detail) in enumerate(steps):
        x = gap + i * (width + gap)
        xs.append(x)
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                    boxstyle="round,pad=0.08,rounding_size=0.12",
                                    linewidth=1.5, edgecolor="#37474f",
                                    facecolor="#dfe6ea", zorder=2))
        ax.text(x + width / 2.0, y + height - 0.42, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", zorder=3)
        ax.text(x + width / 2.0, y + (height - 0.7) / 2.0, detail, ha="center",
                va="center", fontsize=7.3, color="#455a64", zorder=3)

    for i in range(n - 1):
        start = (xs[i] + width, y + height / 2.0)
        end = (xs[i + 1], y + height / 2.0)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.5, color="#37474f", zorder=1))

    fig.tight_layout()
    return _save(fig, out_dir, "pipeline", dpi)


def _save(fig, out_dir: str | os.PathLike, name: str, dpi: int) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    png = out_path / f"{name}.png"
    pdf = out_path / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]
