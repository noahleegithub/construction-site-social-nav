"""Static matplotlib rendering of a construction worksite scene.

Ported from ``visualize_scene.py``. Draws the graph-structured worksite
(highways, junctions, workstations), heterogeneous agents with vision cones and
blindspots, and the robot with its route and goal.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle, Wedge
from matplotlib.transforms import Affine2D

from construction_nav import config as C
from construction_nav.entities import Agent, Highway, Scene
from construction_nav.graph import route_points

STYLE = {
    "human": {"color": "#1f6feb", "size": 0.55},
    "vehicle": {"color": "#d2762b", "size": 1.6},
    "working_alpha": 0.4,
    "moving_alpha": 1.0,
    "fov_color": "#f2c744",
    "blind_color": "#c0392b",
    "hazard_face": "#e74c3c",
    "junction_color": "#37474f",
    "highway": {"face": "#dfe6ea", "edge": "#90a4ae"},
    "robot_color": "#27ae60",
    "goal_color": "#f1c40f",
}


# ---------------------------------------------------------------------------
# Environment layers
# ---------------------------------------------------------------------------


def draw_hazard_field(ax, arena: float) -> None:
    """Off-highway ground is high-risk; corridors are carved out on top."""
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            arena,
            arena,
            facecolor=STYLE["hazard_face"],
            edgecolor=STYLE["hazard_face"],
            alpha=0.12,
            hatch="///",
            linewidth=0.0,
            zorder=0,
            label="off-highway high-risk area",
        )
    )


def _corridor_quad(p_u: np.ndarray, p_v: np.ndarray, width: float) -> np.ndarray:
    direction = p_v - p_u
    length = np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]]) / max(length, 1e-9)
    half = normal * (width / 2.0)
    return np.array([p_u + half, p_v + half, p_v - half, p_u - half])


def draw_highway(ax, highway: Highway, scene: Scene) -> None:
    style = STYLE["highway"]
    p_u = scene.nodes[highway.u].pos
    p_v = scene.nodes[highway.v].pos
    quad = _corridor_quad(p_u, p_v, highway.width)

    ax.add_patch(
        Polygon(
            quad,
            closed=True,
            facecolor=style["face"],
            edgecolor="none",
            linewidth=0.0,
            antialiased=False,
            alpha=1.0,
            zorder=1.4,
        )
    )
    if C.SHOW_CENTERLINES:
        ax.plot(
            [p_u[0], p_v[0]],
            [p_u[1], p_v[1]],
            linestyle=(0, (6, 4)),
            color=style["edge"],
            linewidth=0.7,
            alpha=0.6,
            zorder=1.5,
        )


def draw_junctions(ax, scene: Scene) -> None:
    hubs = np.array([n.pos for n in scene.nodes if n.role == "junction"])
    if len(hubs):
        ax.plot(
            hubs[:, 0], hubs[:, 1],
            marker="o", markersize=5, linestyle="none",
            markerfacecolor=STYLE["junction_color"],
            markeredgecolor="white", markeredgewidth=0.6,
            zorder=4, label="intersection",
        )


def draw_workstations(ax, scene: Scene) -> None:
    for kind, label in (("human", "human workstation"), ("vehicle", "vehicle workstation")):
        pts = np.array([n.pos for n in scene.nodes if n.role == "workstation" and n.kind == kind])
        if len(pts):
            ax.plot(
                pts[:, 0], pts[:, 1],
                marker="s", markersize=9, linestyle="none",
                markerfacecolor=STYLE[kind]["color"],
                markeredgecolor="black", markeredgewidth=0.6,
                zorder=4.5, label=label,
            )


# ---------------------------------------------------------------------------
# Dynamic layers
# ---------------------------------------------------------------------------


def draw_fov_cone(ax, agent: Agent, fov_range: float = C.FOV_RANGE_REF) -> None:
    center = agent.heading_deg
    ax.add_patch(
        Wedge(
            agent.pos, fov_range,
            center - np.rad2deg(C.FOV_HALF_ANGLE),
            center + np.rad2deg(C.FOV_HALF_ANGLE),
            facecolor=STYLE["fov_color"], edgecolor="none", alpha=0.2, zorder=3,
        )
    )


def draw_blindspot(ax, agent: Agent, blind_range: float = C.BLIND_RANGE_REF) -> None:
    center = agent.heading_deg + 180.0
    half_angle = C.BLIND_HALF_ANGLE[agent.kind]
    ax.add_patch(
        Wedge(
            agent.pos, blind_range,
            center - np.rad2deg(half_angle),
            center + np.rad2deg(half_angle),
            facecolor=STYLE["blind_color"], edgecolor=STYLE["blind_color"],
            alpha=0.16, hatch="xx", linewidth=0.0, zorder=3,
        )
    )


def draw_agent(ax, agent: Agent) -> None:
    color = STYLE[agent.kind]["color"]
    alpha = STYLE["working_alpha"] if agent.state == "working" else STYLE["moving_alpha"]

    if agent.kind == "human":
        marker = Circle(agent.pos, STYLE["human"]["size"], transform=ax.transData)
        marker.set_facecolor(color)
        marker.set_alpha(alpha)
        marker.set_zorder(7)
        ax.add_patch(marker)
    else:
        length = 2.6 * STYLE["vehicle"]["size"] / 1.6
        width = 1.3 * STYLE["vehicle"]["size"] / 1.6
        rect = Rectangle(
            (-length / 2.0, -width / 2.0), length, width,
            facecolor=color, edgecolor="black", linewidth=0.5,
            alpha=alpha, zorder=7,
        )
        rect.set_transform(
            Affine2D().rotate_deg(agent.heading_deg).translate(*agent.pos) + ax.transData
        )
        ax.add_patch(rect)

    if agent.state == "moving":
        ax.annotate(
            "",
            xy=agent.pos + agent.heading * 2.2,
            xytext=agent.pos,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, alpha=alpha),
            zorder=7.5,
        )


def draw_robot_goal(ax, scene: Scene) -> None:
    route = route_points(scene, scene.robot_route)
    ax.plot(
        route[:, 0], route[:, 1],
        linestyle=(0, (5, 5)), color=STYLE["goal_color"], linewidth=1.6,
        zorder=6, label="robot route",
    )
    ax.plot(
        scene.robot_pos[0], scene.robot_pos[1],
        marker="s", markersize=13,
        markerfacecolor=STYLE["robot_color"], markeredgecolor="black",
        markeredgewidth=0.8, zorder=9, label="robot",
    )
    ax.annotate(
        "",
        xy=scene.robot_pos + scene.robot_heading * 3.2,
        xytext=scene.robot_pos,
        arrowprops=dict(arrowstyle="-|>", color=STYLE["robot_color"], lw=2.0),
        zorder=9,
    )
    goal = scene.goal_pos
    ax.plot(
        goal[0], goal[1],
        marker="*", markersize=22,
        markerfacecolor=STYLE["goal_color"], markeredgecolor="black",
        markeredgewidth=0.8, zorder=9, label="goal",
    )


def _legend_handles(show_cones: bool) -> list:
    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=8,
               markerfacecolor=STYLE["human"]["color"], markeredgecolor="black",
               label="human workstation"),
        Line2D([], [], marker="s", linestyle="none", markersize=8,
               markerfacecolor=STYLE["vehicle"]["color"], markeredgecolor="black",
               label="vehicle workstation"),
    ]
    if C.SHOW_JUNCTIONS:
        handles.append(
            Line2D([], [], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=STYLE["junction_color"], markeredgecolor="white",
                   label="intersection")
        )
    handles += [
        Patch(facecolor=STYLE["highway"]["face"], edgecolor="none", label="highway"),
        Patch(facecolor=STYLE["hazard_face"], alpha=0.15, hatch="///",
              edgecolor=STYLE["hazard_face"], label="off-highway high-risk area"),
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=STYLE["human"]["color"], markeredgecolor="none",
               label="human agent"),
        Line2D([], [], marker="s", linestyle="none", markersize=7,
               markerfacecolor=STYLE["vehicle"]["color"], markeredgecolor="black",
               label="vehicle agent"),
        Line2D([], [], color=STYLE["goal_color"], linestyle=(0, (5, 5)), label="robot route"),
    ]
    if show_cones:
        handles += [
            Patch(facecolor=STYLE["fov_color"], alpha=0.3, edgecolor="none", label="field of view"),
            Patch(facecolor=STYLE["blind_color"], alpha=0.3, hatch="xx",
                  edgecolor=STYLE["blind_color"], label="blindspot"),
        ]
    return handles


# ---------------------------------------------------------------------------
# Top-level renders
# ---------------------------------------------------------------------------


def render_arena_backdrop(
    scene: Scene,
    out_dir: str | os.PathLike = "figures",
    dpi: int = 300,
    pixels: int = 2400,
) -> Path:
    """Arena-only backdrop whose pixel extent is exactly ``[0, arena]^2``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    arena = scene.arena_size

    fig = plt.figure(figsize=(pixels / dpi, pixels / dpi), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, arena)
    ax.set_ylim(0.0, arena)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_hazard_field(ax, arena)
    for highway in scene.highways:
        draw_highway(ax, highway, scene)
    if C.SHOW_JUNCTIONS:
        draw_junctions(ax, scene)
    draw_workstations(ax, scene)

    path = out_path / "scene_arena.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def render_scene(
    scene: Scene,
    out_dir: str | os.PathLike = "figures",
    show_cones: bool = True,
    dpi: int = 300,
) -> list[Path]:
    """Render the scene to PNG + PDF and return the written paths."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    arena_png = render_arena_backdrop(scene, out_path, dpi=dpi)

    fig, ax = plt.subplots(figsize=(11, 10))
    arena = scene.arena_size

    draw_hazard_field(ax, arena)
    for highway in scene.highways:
        draw_highway(ax, highway, scene)
    if C.SHOW_JUNCTIONS:
        draw_junctions(ax, scene)

    if show_cones:
        layout = scene.resolved_layout
        for agent in scene.agents:
            draw_fov_cone(ax, agent, layout.fov_range)
            draw_blindspot(ax, agent, layout.blind_range)

    draw_workstations(ax, scene)
    for agent in scene.agents:
        draw_agent(ax, agent)
    draw_robot_goal(ax, scene)

    ax.set_xlim(0, arena)
    ax.set_ylim(0, arena)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    n_shared = sum(h.access == C.SHARED for h in scene.highways)
    ax.set_title(
        "Construction-site social navigation environment (graph-structured)\n"
        f"seed={scene.seed}   M={len(scene.agents)} "
        f"({scene.num_humans} humans, {scene.num_vehicles} vehicles)   "
        f"{len(scene.nodes)} nodes / {len(scene.highways)} highways "
        f"({n_shared} shared)",
        fontsize=12,
    )
    ax.set_axisbelow(True)
    ax.legend(
        handles=_legend_handles(show_cones),
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.5,
    )

    png = out_path / "scene.png"
    pdf = out_path / "scene.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf, arena_png]
