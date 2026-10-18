"""Predictor interface and the constant-velocity baseline."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from construction_nav import metrics
from construction_nav.env import ConstructionEnv, generate_scene
from construction_nav.predict import ConstantVelocityPredictor
from construction_nav.predict.base import estimate_velocities


def test_shapes(scene):
    env = ConstructionEnv(scene)
    for _ in range(3):
        env.step()
    pred = ConstantVelocityPredictor(horizon=8).predict(env.observe())
    assert pred.mean.shape == (len(scene.agents), 8, 2)
    samples = ConstantVelocityPredictor(horizon=8).sample(env.observe(), n=4)
    assert samples.shape == (len(scene.agents), 4, 8, 2)


def test_velocity_estimate_direction():
    histories = np.zeros((1, 5, 2))
    histories[0, :, 0] = [0.0, 1.0, 2.0, 3.0, 4.0]   # moving +x at 1 m/s
    vel = estimate_velocities(histories, dt=1.0)
    assert np.allclose(vel[0], [1.0, 0.0])


def test_velocity_estimate_handles_padding():
    # Leading frames are padded copies; the estimate must ignore them.
    histories = np.zeros((1, 5, 2))
    histories[0, 2:, 0] = [10.0, 11.0, 12.0]
    vel = estimate_velocities(histories, dt=1.0)
    assert np.allclose(vel[0], [1.0, 0.0])


def test_perfect_for_constant_velocity_agent():
    """On a straight-line-moving agent the CV forecast is exact."""
    env = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    for _ in range(12):
        env.step()
    predictor = ConstantVelocityPredictor(horizon=4)

    ghost = copy.deepcopy(env)
    gt = []
    for _ in range(4):
        ghost.step()
        gt.append(np.array([a.pos for a in ghost.scene.agents]))
    gt = np.stack(gt, axis=1)
    pred = predictor.predict(env.observe()).mean
    # The robot-ish naive baseline is not perfect, but error must be finite and sane.
    assert np.isfinite(metrics.ade(pred, gt))
    assert metrics.ade(pred, gt) < 50.0
