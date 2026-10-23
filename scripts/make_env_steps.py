"""Layer-by-layer construction of a worksite scene.

Writes one PNG per construction layer plus a combined grid, for slides:

1. arena + off-highway hazard field
2. junction graph (nodes and edges)
3. walkable corridors
4. workstations and their approach spokes
5. agents, with vision cones and blind spots
6. the robot, its planned route and goal

Each layer reuses the same drawing primitives as ``figures/scene.png`` so the
sequence ends exactly at the shipped scene.  Corridors are drawn as plain
walkable space: which agents may use which corridor is a routing constraint in
``graph.adjacency``, not something the picture needs to encode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from construction_nav import config as C
from construction_nav.env import generate_scene
from construction_nav.render import (
    STYLE,
    body_scale,
    draw_agent,
    draw_blindspot,
    draw_fov_cone,
    draw_hazard_field,
    draw_highway,
    draw_robot_goal,
    draw_workstations,
)

LAYER_TITLES = [
    "1. Arena and off-highway hazard field",
    "2. Junction graph",
    "3. Walkable corridors",
    "4. Workstations and approach spokes",
    "5. Heterogeneous agents: vision and blind spots",
    "6. Robot, planned route and goal",
]


def _graph(ax, scene) -> None:
    """Junction nodes and the edges joining them."""
    hubs = [n for n in scene.nodes if n.role == "junction"]
    index = {id(n): i for i, n in enumerate(scene.nodes)}
    lookup = {i: n.pos for i, n in enumerate(scene.nodes)}
    for highway in scene.highways:
        a, b = lookup[highway.u], lookup[highway.v]
        ax.plot(
            [a[0], b[0]], [a[1], b[1]],
            color="#546e7a", lw=1.1, ls=(0, (4, 4)), zorder=3.5,
        )
    del index
    if hubs:
        pts = np.array([n.pos for n in hubs])
        ax.plot(
            pts[:, 0], pts[:, 1],
            marker="o", markersize=6, linestyle="none",
            markerfacecolor=STYLE["junction_color"], markeredgecolor="white",
            markeredgewidth=0.7, zorder=4,
        )


def _spokes(ax, scene) -> None:
    """The corridor stub that joins each workstation to the network."""
    for highway in scene.highways:
        node_u = scene.nodes[highway.u]
        node_v = scene.nodes[highway.v]
        if node_u.role == "workstation" or node_v.role == "workstation":
            a, b = node_u.pos, node_v.pos
            # Neutral slate: in gold these read as the robot's route, which
            # arrives in layer 6 and must be the only gold path on the map.
            ax.plot(
                [a[0], b[0]], [a[1], b[1]],
                color="#7d8b93", lw=1.6, alpha=0.85, zorder=4.1,
            )


def _frame(ax, scene, title: str) -> None:
    arena = scene.arena_size
    ax.set_xlim(0, arena)
    ax.set_ylim(0, arena)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_layer(ax, scene, layer: int) -> None:
    """Draw layers ``1..layer`` cumulatively."""
    draw_hazard_field(ax, scene.arena_size)
    if layer >= 2:
        _graph(ax, scene)
    if layer >= 3:
        for highway in scene.highways:
            draw_highway(ax, highway, scene)
    if layer >= 4:
        _spokes(ax, scene)
        draw_workstations(ax, scene)
    if layer >= 5:
        layout = scene.resolved_layout
        for agent in scene.agents:
            draw_fov_cone(ax, agent, layout.fov_range)
            draw_blindspot(ax, agent, layout.blind_range)
        scale = body_scale(scene)
        for agent in scene.agents:
            draw_agent(ax, agent, scale=scale)
    if layer >= 6:
        draw_robot_goal(ax, scene)
    _frame(ax, scene, LAYER_TITLES[layer - 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--arena", type=float, default=C.ARENA_SIZE)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    scene = generate_scene(seed=args.seed, num_agents=args.agents, arena=args.arena)
    out_dir = args.out_dir / "env_steps"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for layer in range(1, 7):
        fig, ax = plt.subplots(figsize=(7.6, 7.2))
        draw_layer(ax, scene, layer)
        slug = LAYER_TITLES[layer - 1][3:].split(":")[0].lower()
        slug = "".join(ch if ch.isalnum() else "_" for ch in slug).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        path = out_dir / f"{layer:02d}_{slug}.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    grid, axes = plt.subplots(2, 3, figsize=(19, 13))
    for layer, ax in enumerate(axes.ravel(), start=1):
        draw_layer(ax, scene, layer)
        ax.tick_params(labelsize=7)
    grid.tight_layout()
    grid_path = args.out_dir / "env_steps.png"
    grid.savefig(grid_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(grid)
    written.append(grid_path)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
