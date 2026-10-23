"""Animated rendering of closed-loop construction-site trials.

Produces GIFs (PillowWriter, no ffmpeg needed) and optionally MP4s of a rollout
so social-navigation behaviour can be inspected frame by frame.

:func:`animate_trial` renders one planner over one scene.
:func:`animate_comparison` renders several planners over the *same* scene side by
side, in lockstep, so a difference in the picture cannot come from a different
site or a different clock.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle, Wedge
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


def rollout(
    env: ConstructionEnv,
    planner=None,
    steps: int = 900,
    lambda_intrusion: float = 1.0,
    lambda_freeze: float = 0.5,
) -> dict:
    """Advance the environment, recording poses/headings/states for animation.

    ``planner`` is a callable ``(env) -> (v, omega)`` or ``None`` for the
    built-in path-following controller.
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
                "goals": env.robot.goals_reached,
                "contacts": env.n_collision_events,
                "J": env.travel_time
                + lambda_intrusion * env.intrusion_integral
                + lambda_freeze * env.freeze_time,
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

    if frames:
        # The final snapshot is taken before the robot is promoted to its last
        # goal, so correct the last frame's counter.
        frames[-1]["goals"] = env.robot.goals_reached
    return {"frames": frames, "reached": reached, "t": env.t}


class _ScenePanel:
    """One animated work-site panel: static background plus moving actors."""

    def __init__(
        self, ax, scene, label: str = "", zoom: float = 5.0, zoom_frac: float = 0.36
    ) -> None:
        self.ax = ax
        self.label = label
        self.zoom = zoom
        # Fraction of the drawn map the magnifier occupies; a single-panel
        # animation needs less of it than a five-panel grid.
        self.zoom_frac = zoom_frac
        self.layout = scene.resolved_layout
        self.scale = body_scale(scene)
        arena = scene.arena_size

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

        self.fov_wedges: list[Wedge] = []
        self.blind_wedges: list[Wedge] = []
        self.kinds: list[str] = []
        self.human: dict[int, Circle] = {}
        self.vehicle: dict[int, Polygon] = {}

        # The stats live in the title rather than a caption box: the magnifier
        # sits over the bottom-right corner, where the caption used to be.
        self.title = ax.set_title(
            f"{label}\nt = 0.0 s" if label else "t = 0.0 s", fontsize=9,
        )
        self._rx: list[float] = []
        self._ry: list[float] = []

    def add_actors(self, frame: dict) -> None:
        """Create the moving artists once the first frame's agents are known."""
        self.kinds = list(frame["agent_kind"])
        n = len(self.kinds)
        for _ in range(n):
            fov = Wedge((0, 0), self.layout.fov_range, 0, 0,
                        facecolor=STYLE["fov_color"], edgecolor="none",
                        alpha=0.18, zorder=3)
            blind = Wedge((0, 0), self.layout.blind_range, 0, 0,
                          facecolor=STYLE["blind_color"], edgecolor="none",
                          alpha=0.14, zorder=3)
            self.ax.add_patch(fov)
            self.ax.add_patch(blind)
            self.fov_wedges.append(fov)
            self.blind_wedges.append(blind)

        for i, kind in enumerate(self.kinds):
            if kind == "human":
                circle = Circle((0, 0), STYLE["human"]["size"] * self.scale,
                                facecolor=STYLE["human"]["color"],
                                edgecolor="none", zorder=7)
                self.ax.add_patch(circle)
                self.human[i] = circle
            else:
                poly = Polygon(np.zeros((4, 2)), closed=True,
                               facecolor=STYLE["vehicle"]["color"],
                               edgecolor="black", linewidth=0.5, zorder=7)
                self.ax.add_patch(poly)
                self.vehicle[i] = poly

        self.robot = Circle((0, 0), STYLE["human"]["size"] * self.scale,
                            facecolor=STYLE["robot_color"], edgecolor="black",
                            linewidth=0.8, zorder=9)
        self.ax.add_patch(self.robot)
        (self.trail,) = self.ax.plot([], [], color=STYLE["robot_color"],
                                     lw=1.4, alpha=0.7, zorder=8)

        # The magnifier is created by :meth:`build_zoom`, once the figure has
        # been drawn and the equal-aspect axes' real box is known.
        self.zax = None
        self.z_human: dict[int, Circle] = {}
        self.z_vehicle: dict[int, Polygon] = {}

    def build_zoom(self) -> None:
        """Add the robot-centred magnifier in the map's bottom-right corner.

        Placed from the *drawn* axes box: ``set_aspect("equal")`` shrinks the
        axes inside its layout slot, so an inset positioned in axes fractions
        can land outside the map frame.
        """
        fig = self.ax.figure
        slot = self.ax.get_position()           # layout slot, figure fractions
        fig_w, fig_h = fig.get_size_inches()
        # ``set_aspect("equal")`` letterboxes the data inside that slot: work
        # out the square the map is actually drawn in, then margin inside it.
        slot_w, slot_h = slot.width * fig_w, slot.height * fig_h
        drawn_in = min(slot_w, slot_h)
        drawn_x = slot.x0 + (slot_w - drawn_in) / 2 / fig_w
        drawn_y = slot.y0 + (slot_h - drawn_in) / 2 / fig_h
        drawn_w, drawn_h = drawn_in / fig_w, drawn_in / fig_h

        if self.zoom_frac <= 0.0:
            return                      # single-panel renderings skip the magnifier

        margin = 0.05 / fig_w
        side_in = self.zoom_frac * drawn_in
        sx, sy = side_in / fig_w, side_in / fig_h
        x0 = drawn_x + drawn_w - sx - margin
        y0 = drawn_y + margin
        self.zax = fig.add_axes([x0, y0, sx, sy], zorder=20)
        self.zax.set_facecolor("white")
        self.zax.set_xticks([])
        self.zax.set_yticks([])
        for spine in self.zax.spines.values():
            spine.set_linewidth(0.8)

        for i, kind in enumerate(self.kinds):
            if kind == "human":
                circle = Circle((0, 0), STYLE["human"]["size"] * self.scale,
                                facecolor=STYLE["human"]["color"],
                                edgecolor="none", zorder=3)
                self.zax.add_patch(circle)
                self.z_human[i] = circle
            else:
                poly = Polygon(np.zeros((4, 2)), closed=True,
                               facecolor=STYLE["vehicle"]["color"],
                               edgecolor="black", linewidth=0.6, zorder=3)
                self.zax.add_patch(poly)
                self.z_vehicle[i] = poly
        self.z_robot = Circle((0, 0), STYLE["human"]["size"] * self.scale,
                              facecolor=STYLE["robot_color"], edgecolor="black",
                              linewidth=0.9, zorder=5)
        self.zax.add_patch(self.z_robot)
        (self.z_heading,) = self.zax.plot([], [], color="black", lw=1.0, zorder=6)
        label = self.zax.text(
            0.03, 0.97, f"robot view ({2 * self.zoom:.0f} m)",
            transform=self.zax.transAxes, fontsize=7, va="top", ha="left",
            zorder=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.2},
        )
        del label
        # Outline of the magnified window on the map.
        self.z_box = Rectangle((0, 0), 2 * self.zoom, 2 * self.zoom,
                               fill=False, edgecolor="black", lw=0.7,
                               linestyle=(0, (2, 2)), alpha=0.6, zorder=10)
        self.ax.add_patch(self.z_box)

    # -- frame update ------------------------------------------------------

    def _wedge(self, wedge, pos, centre_deg, radius, half_deg) -> None:
        wedge.set_center((float(pos[0]), float(pos[1])))
        wedge.set_radius(radius)
        wedge.set_theta1(centre_deg - half_deg)
        wedge.set_theta2(centre_deg + half_deg)

    def update(self, frame: dict) -> None:
        for i in range(len(self.kinds)):
            pos = frame["agent_pos"][i]
            heading = frame["agent_heading"][i]
            theta = float(np.degrees(np.arctan2(heading[1], heading[0])))
            self._wedge(self.fov_wedges[i], pos, theta, self.layout.fov_range,
                        np.degrees(C.FOV_HALF_ANGLE))
            self._wedge(self.blind_wedges[i], pos, theta + 180.0,
                        self.layout.blind_range,
                        np.degrees(C.BLIND_HALF_ANGLE[self.kinds[i]]))

        alpha = {
            "working": STYLE["working_alpha"],
            "moving": STYLE["moving_alpha"],
        }
        for i, circle in self.human.items():
            pos = frame["agent_pos"][i]
            circle.set_center((float(pos[0]), float(pos[1])))
            circle.set_alpha(alpha[frame["agent_state"][i]])

        vlen = 2.6 * STYLE["vehicle"]["size"] / 1.6 * self.scale
        vwid = 1.3 * STYLE["vehicle"]["size"] / 1.6 * self.scale
        corners = np.array([
            [-vlen / 2, -vwid / 2], [vlen / 2, -vwid / 2],
            [vlen / 2, vwid / 2], [-vlen / 2, vwid / 2],
        ])
        for i, poly in self.vehicle.items():
            heading = frame["agent_heading"][i]
            theta = float(np.degrees(np.arctan2(heading[1], heading[0])))
            rot = Affine2D().rotate_deg(theta).translate(*frame["agent_pos"][i])
            poly.set_xy(rot.transform(corners))
            poly.set_alpha(alpha[frame["agent_state"][i]])

        rpos = frame["robot_pos"]
        self.robot.set_center((float(rpos[0]), float(rpos[1])))
        self._rx.append(float(rpos[0]))
        self._ry.append(float(rpos[1]))
        self.trail.set_data(self._rx, self._ry)

        # Magnifier, centred on the robot (absent in single-panel renderings).
        if self.zax is not None:
            self._update_zoom(frame, rpos, corners, alpha)

        stats = (
            f"t = {frame['t']:.1f} s   goals {frame['goals']}/2   "
            f"contacts {frame['contacts']}   J = {frame['J']:.1f}"
        )
        self.title.set_text(f"{self.label}\n{stats}" if self.label else stats)

    def _update_zoom(self, frame: dict, rpos, corners, alpha) -> None:
        self.zax.set_xlim(rpos[0] - self.zoom, rpos[0] + self.zoom)
        self.zax.set_ylim(rpos[1] - self.zoom, rpos[1] + self.zoom)
        self.zax.set_aspect("equal")
        self.z_box.set_xy((rpos[0] - self.zoom, rpos[1] - self.zoom))
        for i, circle in self.z_human.items():
            pos = frame["agent_pos"][i]
            circle.set_center((float(pos[0]), float(pos[1])))
            circle.set_alpha(alpha[frame["agent_state"][i]])
        for i, poly in self.z_vehicle.items():
            heading = frame["agent_heading"][i]
            theta = float(np.degrees(np.arctan2(heading[1], heading[0])))
            rot = Affine2D().rotate_deg(theta).translate(*frame["agent_pos"][i])
            poly.set_xy(rot.transform(corners))
            poly.set_alpha(alpha[frame["agent_state"][i]])
        self.z_robot.set_center((float(rpos[0]), float(rpos[1])))
        rh = frame["robot_heading"]
        self.z_heading.set_data(
            [rpos[0], rpos[0] + 1.6 * rh[0]],
            [rpos[1], rpos[1] + 1.6 * rh[1]],
        )


def _save(fig, anim, out_path: Path, name: str, fmt: str, fps: int, dpi: int) -> list[Path]:
    out_path.mkdir(parents=True, exist_ok=True)
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

    fig, ax = plt.subplots(figsize=(8, 7.5))
    # A single panel is large enough to read; the magnifier would only hide
    # part of the map, so it is only used in the comparison grids.
    panel = _ScenePanel(ax, scene, zoom_frac=0.0)
    panel.add_actors(frames[0])
    fig.canvas.draw()
    panel.build_zoom()

    def update(i: int):
        panel.update(frames[i])
        return []

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / fps, blit=False)
    return _save(fig, anim, Path(out_dir), name, fmt, fps, dpi)


def closest_frame(records: list[dict]) -> int:
    """Index of the frame where the robot came closest to any agent.

    The final frame is a boring still (everyone back at their stations), so the
    contact sheet is taken at the moment of the encounter instead.
    """
    best_gap, best_index = np.inf, 0
    for record in records:
        for i, frame in enumerate(record["frames"]):
            gap = float(
                np.min(np.linalg.norm(frame["agent_pos"] - frame["robot_pos"], axis=1))
            )
            if gap < best_gap:
                best_gap, best_index = gap, i
    return best_index


def _spare_cell_legend(ax) -> None:
    """Turn an unused grid cell into a key for the panels."""
    ax.axis("off")
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=STYLE["human"]["color"], markeredgecolor="none",
               label="human agent"),
        Line2D([], [], marker="s", linestyle="none", markersize=8,
               markerfacecolor=STYLE["vehicle"]["color"], markeredgecolor="black",
               label="vehicle agent"),
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=STYLE["robot_color"], markeredgecolor="black",
               label="robot"),
        Line2D([], [], marker="*", linestyle="none", markersize=12,
               markerfacecolor=STYLE["goal_color"], markeredgecolor="black",
               label="goal"),
        Line2D([], [], linestyle=(0, (5, 5)), color=STYLE["goal_color"],
               label="planned robot route"),
        Patch(facecolor=STYLE["fov_color"], alpha=0.5, edgecolor="none",
              label="field of view"),
        Patch(facecolor=STYLE["blind_color"], alpha=0.5, edgecolor="none",
              label="blind arc"),
    ]
    ax.legend(handles=handles, loc="center", frameon=False, fontsize=9,
              title="key", title_fontsize=9)


def _comparison_figure(scene, records, labels, ncols, note=None):
    """Build the grid of panels for one set of recorded rollouts."""
    nrows = int(np.ceil(len(labels) / ncols))
    # Square panels: an equal-aspect axes shrinks inside its layout box, and a
    # non-square box would let the magnifier poke out of the map frame.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.6 * nrows))
    axes = np.atleast_2d(axes)
    if note:
        fig.suptitle(note, fontsize=12)
    # Explicit spacing: two-line panel titles need more room than tight_layout
    # gives when the titles are set after the layout has been computed. Applied
    # before the panels are built so the magnifiers can be placed from the
    # final axes boxes.
    fig.subplots_adjust(
        top=0.90 if note else 0.94,
        bottom=0.04, left=0.02, right=0.98,
        hspace=0.32, wspace=0.06,
    )

    panels: list[_ScenePanel] = []
    for cell, label in enumerate(labels):
        ax = axes[cell // ncols, cell % ncols]
        panel = _ScenePanel(ax, scene, label=label)
        panel.add_actors(records[cell]["frames"][0])
        panels.append(panel)
    for cell in range(len(labels), nrows * ncols):
        _spare_cell_legend(axes[cell // ncols, cell % ncols])

    fig.canvas.draw()          # resolves the equal-aspect axes boxes
    for panel in panels:
        panel.build_zoom()
    return fig, panels


def animate_comparison(
    scene,
    entries: list[tuple[str, object]],
    out_dir: str | os.PathLike = "figures",
    name: str = "comparison",
    steps: int = 900,
    fps: int = 10,
    dpi: int = 80,
    fmt: str = "gif",
    ncols: int = 3,
    note: str | None = None,
    still: bool = False,
    still_index: int | None = None,
) -> tuple[list[Path], dict]:
    """Render several planners on the same scene, side by side, in lockstep.

    ``entries`` is ``[(label, planner-or-None), ...]``. Every planner runs on its
    own copy of the scene from the same seed, so all panels share the site and
    the clock; a panel whose trial finishes early simply holds its last frame.
    With ``still=True`` the closest-approach frame (or ``still_index``) is also
    written as a PNG contact sheet.
    Returns the written paths and a per-planner summary.
    """
    records: list[dict] = []
    for _label, planner in entries:
        env = ConstructionEnv(copy.deepcopy(scene))
        records.append(rollout(env, planner=planner, steps=steps))

    labels = [label for label, _planner in entries]
    longest = max(len(r["frames"]) for r in records)
    out_path = Path(out_dir)
    paths: list[Path] = []

    if still:
        index = closest_frame(records) if still_index is None else still_index
        fig, panels = _comparison_figure(scene, records, labels, ncols, note)
        for panel, record in zip(panels, records):
            frames = record["frames"]
            panel.update(frames[min(index, len(frames) - 1)])
        out_path.mkdir(parents=True, exist_ok=True)
        png = out_path / f"{name}.png"
        fig.savefig(png, dpi=max(dpi, 150), bbox_inches="tight")
        plt.close(fig)
        paths.append(png)

    fig, panels = _comparison_figure(scene, records, labels, ncols, note)

    def update(i: int):
        for panel, record in zip(panels, records):
            frames = record["frames"]
            panel.update(frames[min(i, len(frames) - 1)])
        return []

    anim = FuncAnimation(fig, update, frames=longest,
                         interval=1000 / fps, blit=False)
    paths += _save(fig, anim, out_path, name, fmt, fps, dpi)

    summary = {
        label: {
            "reached": record["reached"],
            "t": record["t"],
            "contacts": record["frames"][-1]["contacts"],
            "J": record["frames"][-1]["J"],
        }
        for label, record in zip(labels, records)
    }
    return paths, summary
