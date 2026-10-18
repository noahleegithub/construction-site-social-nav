"""ADE / FDE, barrier terms, and the navigation cost."""

from __future__ import annotations

import numpy as np
import pytest

from construction_nav import metrics


def test_ade_fde_closed_form():
    gt = np.zeros((2, 3, 2))
    pred = np.zeros((2, 3, 2))
    pred[0] = [1.0, 1.0]     # constant offset of sqrt(2)
    pred[1, -1] = [3.0, 4.0]  # final offset of 5
    # ADE = mean over 2 agents x 3 steps of per-step distances
    expected_ade = (3 * np.sqrt(2) + 0 + 0 + 5) / 6
    assert metrics.ade(pred, gt) == pytest.approx(expected_ade)
    # FDE = mean of final-step distances
    assert metrics.fde(pred, gt) == pytest.approx((np.sqrt(2) + 5) / 2)


def test_min_ade_best_of_k():
    gt = np.zeros((1, 2, 2))
    samples = np.zeros((1, 3, 2, 2))
    samples[0, 0] = 10.0   # bad mode
    samples[0, 1] = 1.0    # good mode: offset (1, 1) -> distance sqrt(2)
    samples[0, 2] = 5.0
    assert metrics.min_ade(samples, gt) == pytest.approx(np.sqrt(2))
    assert metrics.min_fde(samples, gt) == pytest.approx(np.sqrt(2))


def test_barrier_terms():
    p = np.zeros(2)
    q = np.array([3.0, 4.0])       # distance 5
    assert metrics.h_S(p, q, D_S=0.5) == pytest.approx(25 - 0.25)
    assert metrics.h_C(p, q, D_C=1.0) == pytest.approx(25 - 1.0)
    # intrusion is max(0, -h_C): zero when outside the safety radius.
    assert metrics.intrusion(p, q[None, :], np.array([1.0]))[0] == 0.0
    near = np.array([[0.1, 0.0]])
    assert metrics.intrusion(p, near, np.array([1.0]))[0] == pytest.approx(1.0 - 0.01)


def test_min_terms():
    robot = np.zeros(2)
    agents = np.array([[1.0, 0.0], [0.2, 0.0]])
    D_S = np.array([0.4, 0.4])
    assert metrics.min_h_S(robot, agents, D_S) == pytest.approx(0.04 - 0.16)


def test_navigation_cost_composition():
    J = metrics.navigation_cost(travel_time=10.0, intrusion_integral=2.0,
                                freeze_time=1.0, lambda_intrusion=1.0,
                                lambda_freeze=0.5)
    assert J == pytest.approx(10.0 + 2.0 + 0.5)


def test_summarize_trial():
    from construction_nav.env import ConstructionEnv, generate_scene

    env = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    for _ in range(20):
        env.step()
    summary = metrics.summarize_trial(env)
    assert set(summary) >= {"travel_time", "intrusion_integral", "freeze_time", "J", "collided"}
