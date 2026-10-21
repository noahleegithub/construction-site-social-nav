"""Animated rendering of closed-loop construction-site trials.

Produces GIFs (PillowWriter, no ffmpeg needed) and optionally MP4s of a rollout
so social-navigation behaviour can be inspected frame by frame.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon, Wedge
from matplotlib.transforms import Affine2D

from construction_nav import config as C
from construction_nav.env import ConstructionEnv
from construction_nav.render import (
    STYLE,
    body_scale,
    draw_hazard_field,
    draw_highway,
    draw_robot_route,
    draw_workstations,
)


def rollout(env: ConstructionEnv, planner=None, steps: int = 900) -> dict:
    """Advance the environment, recording poses/headings/states for animation.

    ``planner`` is a callable ``(env) -> (v, omega)`` or ``None`` for the
    built-in goal-seeking controller.
    """
    frames: list[dict] = []
    reached = False

    def snapshot():
        frames.append(
            {
                "t": env.t,
                "agent_pos": np.array([a.pos for a in env.scene.agents]),
                "agent_heading": np.array([a.heading for a in env.scene.agents]),
                "agent_kind": [a.kind for a in env.scene.agents],
                "agent_state": [a.state for a in env.scene.agents],
                "robot_pos": env.robot.pos.copy(),
                "robot_heading": env.robot.heading.copy(),
            }
        )

    snapshot()
    for _ in range(steps):
        cmd = None if planner is None else planner(env)
        info = env.step(cmd)
        snapshot()
        if info.robot_reached_goal and not env.advance_robot_goal():
            reached = True
            break

    return {"frames": frames, "reached": reached, "t": env.t}


def animate_trial(
    scene,
    out_dir: str | os.PathLike = "figures",
    name: str = "trial",
    planner=None,
    steps: int = 900,
    fps: int = 12,
    dpi: int = 120,
    fmt: str = "gif",
) -> list[Path]:
    """Render a rollout to an animation and return the written paths."""
    # Copy the scene: the rollout advances agent poses in place, so reusing the
    # caller's scene would leave it at the final state.
    scene = copy.deepcopy(scene)
    env = ConstructionEnv(scene)
    record = rollout(env, planner=planner, steps=steps)
    frames = record["frames"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    arena = scene.arena_size

    fig, ax = plt.subplots(figsize=(8, 7.5))
    draw_hazard_field(ax, arena)
    for highway in scene.highways:
        draw_highway(ax, highway, scene)
    draw_workstations(ax, scene)
    draw_robot_route(ax, scene)
    ax.set_xlim(0, arena)
    ax.set_ylim(0, arena)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    kinds = frames[0]["agent_kind"]
    n_agents = len(kinds)
    layout = scene.resolved_layout
    scale = body_scale(scene)

    # Persistent artists.
    fov_wedges = [
        Wedge((0, 0), layout.fov_range, 0, 0, facecolor=STYLE["fov_color"],
              edgecolor="none", alpha=0.18, zorder=3)
        for _ in range(n_agents)
    ]
    blind_wedges = [
        Wedge((0, 0), layout.blind_range, 0, 0, facecolor=STYLE["blind_color"],
              edgecolor="none", alpha=0.14, zorder=3)
        for _ in range(n_agents)
    ]
    for w in fov_wedges + blind_wedges:
        ax.add_patch(w)

    human_circles = [
        Circle(
            (0, 0), STYLE["human"]["size"] * scale,
            facecolor=STYLE["human"]["color"], edgecolor="none", zorder=7,
        )
        for i, k in enumerate(kinds) if k == "human"
    ]
    human_idx = [i for i, k in enumerate(kinds) if k == "human"]
    for c in human_circles:
        ax.add_patch(c)

    vlen = 2.6 * STYLE["vehicle"]["size"] / 1.6 * scale
    vwid = 1.3 * STYLE["vehicle"]["size"] / 1.6 * scale
    vehicle_polys = []
    vehicle_idx = [i for i, k in enumerate(kinds) if k == "vehicle"]
    for _ in vehicle_idx:
        poly = Polygon(
            np.zeros((4, 2)),
            closed=True,
            facecolor=STYLE["vehicle"]["color"],
            edgecolor="black", linewidth=0.5, zorder=7,
        )
        ax.add_patch(poly)
        vehicle_polys.append(poly)

    robot_circle = Circle(
        (0, 0), STYLE["human"]["size"] * scale,
        facecolor=STYLE["robot_color"], edgecolor="black", linewidth=0.8, zorder=9,
    )
    ax.add_patch(robot_circle)
    trail, = ax.plot([], [], color=STYLE["robot_color"], lw=1.4, alpha=0.7, zorder=8)
    title = ax.set_title("")

    rxs: list[float] = []
    rys: list[float] = []

    def _set_wedge(wedge, pos, theta_center_deg, rng, half_deg):
        wedge.set_center((float(pos[0]), float(pos[1])))
        wedge.set_radius(rng)
        wedge.set_theta1(theta_center_deg - half_deg)
        wedge.set_theta2(theta_center_deg + half_deg)

    def update(frame_i: int):
        f = frames[frame_i]
        for i in range(n_agents):
            pos = f["agent_pos"][i]
            theta = float(np.degrees(np.arctan2(f["agent_heading"][i][1], f["agent_heading"][i][0])))
            _set_wedge(fov_wedges[i], pos, theta, layout.fov_range, np.degrees(C.FOV_HALF_ANGLE))
            back = theta + 180.0
            _set_wedge(
                blind_wedges[i], pos, back, layout.blind_range,
                np.degrees(C.BLIND_HALF_ANGLE[f["agent_kind"][i]]),
            )
        for c, i in zip(human_circles, human_idx):
            c.set_center((float(f["agent_pos"][i][0]), float(f["agent_pos"][i][1])))
            c.set_alpha(STYLE["working_alpha"] if f["agent_state"][i] == "working" else STYLE["moving_alpha"])
        for poly, i in zip(vehicle_polys, vehicle_idx):
            theta = float(np.degrees(np.arctan2(f["agent_heading"][i][1], f["agent_heading"][i][0])))
            corners = np.array([
                [-vlen / 2, -vwid / 2], [vlen / 2, -vwid / 2],
                [vlen / 2, vwid / 2], [-vlen / 2, vwid / 2],
            ])
            rot = Affine2D().rotate_deg(theta).translate(*f["agent_pos"][i])
            poly.set_xy(rot.transform(corners))
            poly.set_alpha(STYLE["working_alpha"] if f["agent_state"][i] == "working" else STYLE["moving_alpha"])

        rpos = f["robot_pos"]
        robot_circle.set_center((float(rpos[0]), float(rpos[1])))
        rxs.append(float(rpos[0]))
        rys.append(float(rpos[1]))
        trail.set_data(rxs, rys)
        title.set_text(f"t = {f['t']:.1f} s")
        return []

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False,
    )

    paths: list[Path] = []
    if fmt == "gif":
        gif = out_path / f"{name}.gif"
        anim.save(gif, writer=PillowWriter(fps=fps), dpi=dpi)
        paths.append(gif)
    else:
        mp4 = out_path / f"{name}.mp4"
        anim.save(mp4, fps=fps, dpi=dpi)
        paths.append(mp4)
    plt.close(fig)
    return paths
