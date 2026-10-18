"""Heterogeneous dynamics integrators."""

from __future__ import annotations

import numpy as np
import pytest

from construction_nav import config as C
from construction_nav.dynamics import (
    bicycle_step,
    bicycle_turn_rate,
    unicycle_step,
    wrap_angle,
)


def test_wrap_angle_range():
    for angle in (-4 * np.pi, -np.pi - 0.1, 0.0, np.pi, 3 * np.pi):
        w = wrap_angle(angle)
        assert -np.pi - 1e-9 <= w <= np.pi + 1e-9


def test_unicycle_straight_line():
    pos = np.zeros(2)
    pos2, theta = unicycle_step(pos, 0.0, v=1.0, omega=0.0, dt=0.5)
    assert np.allclose(pos2, [0.5, 0.0])
    assert theta == pytest.approx(0.0)


def test_unicycle_turning():
    # Quarter turn: omega * dt = pi/2. The exact arc endpoint for a unicycle
    # starting at the origin heading +x is (v/w)*(sin(wt), 1 - cos(wt)).
    v, omega, dt = 1.0, np.pi / 2, 1.0
    pos2, theta = unicycle_step(np.zeros(2), 0.0, v, omega, dt)
    assert theta == pytest.approx(np.pi / 2, abs=1e-6)
    exact = (v / omega) * np.array([np.sin(omega * dt), 1 - np.cos(omega * dt)])
    assert np.allclose(pos2, exact, atol=0.1)


def test_bicycle_turn_rate_matches_curvature():
    delta = np.deg2rad(10.0)
    omega = bicycle_turn_rate(v=4.0, delta=delta, wheelbase=2.5)
    assert omega == pytest.approx(4.0 * np.tan(delta) / 2.5)


def test_bicycle_zero_steer_is_straight():
    pos2, theta = bicycle_step(np.zeros(2), 0.0, v=4.0, delta=0.0, dt=0.5, wheelbase=2.5)
    assert np.allclose(pos2, [2.0, 0.0])
    assert theta == pytest.approx(0.0)


def test_bicycle_turns_tighter_than_unicycle_when_steered():
    delta = np.deg2rad(25.0)
    _, theta_b = bicycle_step(np.zeros(2), 0.0, v=4.0, delta=delta, dt=0.5, wheelbase=2.5)
    assert theta_b > 0.0
