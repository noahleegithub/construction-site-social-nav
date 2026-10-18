"""Planner interface and shared helpers.

A planner is a callable that inspects the environment and returns a unicycle
command ``(v, omega)`` for the robot. ``oracle_futures`` provides the
perfect-knowledge future trajectories used by the fixed-radius baseline, and
``unicycle_command`` maps a desired planar velocity to ``(v, omega)``.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np

from construction_nav import config as C
from construction_nav.dynamics import wrap_angle
from construction_nav.env import ConstructionEnv, Observation


class Planner(ABC):
    """Base class for robot policies."""

    @abstractmethod
    def __call__(self, env: ConstructionEnv) -> tuple[float, float]:
        """Return a ``(v, omega)`` command for the robot."""

    def reset(self) -> None:  # pragma: no cover - stateless default
        """Optional per-trial reset hook."""

    @property
    def name(self) -> str:
        return type(self).__name__


def unicycle_command(
    robot_pos: np.ndarray,
    robot_theta: float,
    desired_velocity: np.ndarray,
    max_speed: float,
    max_omega: float = C.ROBOT_MAX_OMEGA,
    heading_gain: float = C.HEADING_GAIN,
) -> tuple[float, float]:
    """Convert a desired planar velocity to a unicycle ``(v, omega)`` command.

    Speed is scaled by alignment so the robot slows while turning hard.
    """
    speed = float(np.linalg.norm(desired_velocity))
    if speed < 1e-6:
        return 0.0, 0.0
    err = wrap_angle(np.arctan2(desired_velocity[1], desired_velocity[0]) - robot_theta)
    align = max(0.0, float(np.cos(err)))
    v = min(speed, max_speed) * (0.25 + 0.75 * align)
    omega = float(np.clip(heading_gain * err, -max_omega, max_omega))
    return v, omega


def oracle_futures(env: ConstructionEnv, horizon: int = C.HORIZON_STEPS) -> np.ndarray:
    """Ground-truth future agent positions from a cloned environment.

    Returns ``(M, H, 2)``. This is the "perfect knowledge" assumption of the
    fixed-radius planner baseline (Section 5).
    """
    ghost = copy.deepcopy(env)
    futures = []
    for _ in range(horizon):
        ghost.step(robot_cmd=(0.0, 0.0))
        futures.append(np.array([a.pos for a in ghost.scene.agents]))
    return np.stack(futures, axis=1)  # (M, H, 2)


def goal_velocity(env: ConstructionEnv) -> np.ndarray:
    """Unit velocity pointing from the robot toward its goal."""
    delta = env.robot.goal - env.robot.pos
    d = float(np.linalg.norm(delta))
    if d < 1e-6:
        return np.zeros(2)
    return delta / d


def route_direction(env: ConstructionEnv) -> np.ndarray:
    """Unit heading along the robot's graph route at its lookahead point.

    Falls back to the straight-line goal direction if the route is degenerate.
    """
    from construction_nav.graph import point_on_polyline, project_arc_length, route_length

    route = env.robot.route
    if not route:
        return goal_velocity(env)
    points = np.array([env.scene.nodes[i].pos for i in route])
    total = route_length(points)
    s = project_arc_length(points, env.robot.pos)
    target, _ = point_on_polyline(points, min(s + env.layout.lookahead, total))
    delta = target - env.robot.pos
    d = float(np.linalg.norm(delta))
    if d < 1e-6:
        return goal_velocity(env)
    return delta / d
