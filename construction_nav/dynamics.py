"""Heterogeneous dynamics integrators.

Humans are modeled with unicycle dynamics, vehicles with bicycle dynamics
(Section 2/3). Both are integrated with a midpoint (RK2) step for stability at
the environment timestep, then reduced to the shared robot representation for
the Robotarium bridge.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(theta: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def heading_of(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)])


def theta_of(heading: np.ndarray) -> float:
    return float(np.arctan2(heading[1], heading[0]))


# ---------------------------------------------------------------------------
# Humans: unicycle  (x' = v cos t, y' = v sin t, t' = omega)
# ---------------------------------------------------------------------------


def unicycle_step(
    pos: np.ndarray, theta: float, v: float, omega: float, dt: float
) -> tuple[np.ndarray, float]:
    """Integrate one unicycle step with a midpoint rule."""
    theta_mid = theta + 0.5 * omega * dt
    pos_next = pos + (v * dt) * heading_of(theta_mid)
    theta_next = wrap_angle(theta + omega * dt)
    return pos_next, theta_next


# ---------------------------------------------------------------------------
# Vehicles: bicycle  (x' = v cos t, y' = v sin t, t' = v tan(delta) / L)
# ---------------------------------------------------------------------------


def bicycle_turn_rate(v: float, delta: float, wheelbase: float) -> float:
    return float(v * np.tan(delta) / wheelbase)


def bicycle_step(
    pos: np.ndarray,
    theta: float,
    v: float,
    delta: float,
    dt: float,
    wheelbase: float,
) -> tuple[np.ndarray, float]:
    """Integrate one bicycle step with a midpoint rule (rear-axle reference)."""
    omega = bicycle_turn_rate(v, delta, wheelbase)
    theta_mid = theta + 0.5 * omega * dt
    pos_next = pos + (v * dt) * heading_of(theta_mid)
    theta_next = wrap_angle(theta + omega * dt)
    return pos_next, theta_next


# ---------------------------------------------------------------------------
# Reduction used by the Robotarium bridge
# ---------------------------------------------------------------------------


def bicycle_to_unicycle_omega(v: float, delta: float, wheelbase: float) -> float:
    """Turn rate a vehicle executes; the Robotarium only simulates unicycles, so
    a vehicle is represented there by feeding this omega to a unicycle robot."""
    return bicycle_turn_rate(v, delta, wheelbase)


def step_agent(pos, theta, kind, v, turn, dt, wheelbase, max_turn):
    """Dispatch to the right integrator.

    ``turn`` is the human turn rate (rad/s) or the vehicle steering angle (rad);
    ``max_turn`` clamps it (rad/s for humans, rad for vehicles).
    """
    turn = float(np.clip(turn, -max_turn, max_turn))
    if kind == "vehicle":
        return bicycle_step(pos, theta, v, turn, dt, wheelbase)
    return unicycle_step(pos, theta, v, turn, dt)
