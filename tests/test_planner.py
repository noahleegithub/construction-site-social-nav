"""Planner interface, helpers, and the fixed-radius baseline."""

from __future__ import annotations

import numpy as np
import pytest

from construction_nav import config as C
from construction_nav.env import ConstructionEnv, generate_scene
from construction_nav.planner import FixedRadiusPlanner, oracle_futures, unicycle_command
from construction_nav.planner.base import goal_velocity


def test_unicycle_command_forward():
    v, omega = unicycle_command(np.zeros(2), 0.0, np.array([2.0, 0.0]), max_speed=2.0)
    assert v == pytest.approx(2.0)
    assert omega == pytest.approx(0.0)


def test_unicycle_command_reverses_when_backwards():
    # Desired velocity points behind the robot: heading error is +/-pi, so the
    # robot must turn hard and slow down.
    v, omega = unicycle_command(np.zeros(2), 0.0, np.array([-1.0, 0.0]), max_speed=2.0)
    assert abs(omega) > 0.0
    assert 0.0 <= v <= 2.0


def test_goal_velocity_points_at_goal():
    env = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    env.robot.pos = np.zeros(2)
    env.robot.goal = np.array([0.0, 5.0])
    assert np.allclose(goal_velocity(env), [0.0, 1.0])


def test_oracle_futures_shape(scene):
    env = ConstructionEnv(scene)
    futures = oracle_futures(env, horizon=6)
    assert futures.shape == (len(scene.agents), 6, 2)


def test_oracle_futures_are_ground_truth():
    """The oracle's first step must equal the true next agent poses."""
    env = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    for _ in range(5):
        env.step()
    futures = oracle_futures(env, horizon=1)

    from construction_nav import metrics

    ghost = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    for _ in range(5):
        ghost.step()
    ghost.step((0.0, 0.0))
    truth = np.array([a.pos for a in ghost.scene.agents])
    assert np.allclose(futures[:, 0, :], truth)


def test_fixed_radius_reaches_goal():
    env = ConstructionEnv(generate_scene(seed=0, num_agents=7))
    planner = FixedRadiusPlanner(use_oracle=True)
    reached = False
    for _ in range(4000):
        info = env.step(planner(env))
        if info.robot_reached_goal:
            reached = True
            break
    assert reached


def _static_obstacle_env(seed=0, num_agents=5):
    """An env with one vehicle-sized obstacle parked on the robot's route.

    All other agents are frozen in distant corners so the test isolates a single
    static obstacle.
    """
    from construction_nav.graph import point_on_polyline, route_length

    scene = generate_scene(seed=seed, num_agents=num_agents)
    env = ConstructionEnv(scene)
    for agent in scene.agents:
        agent.state = "working"
        agent.dwell_remaining = 1e9
        agent.speed = 0.0

    arena = scene.arena_size
    corners = np.array([[0.1, 0.1], [arena - 0.1, 0.1], [0.1, arena - 0.1],
                        [arena - 0.1, arena - 0.1]])
    for i, agent in enumerate(scene.agents[1:], start=1):
        agent.pos = corners[i % len(corners)].copy()

    points = np.array([scene.nodes[i].pos for i in scene.robot_route])
    total = route_length(points)
    pos, heading = point_on_polyline(points, 0.45 * total)
    blocker = scene.agents[0]
    blocker.kind = "vehicle"
    blocker.pos = pos
    blocker.heading = heading
    blocker.D_S, blocker.D_C = 1.5, 3.0
    return env, blocker


def test_naive_controller_hits_static_obstacle():
    env, _ = _static_obstacle_env()
    min_h_S = float("inf")
    for _ in range(2000):
        info = env.step()
        min_h_S = min(min_h_S, info.min_h_S)
        if info.robot_reached_goal:
            break
    assert min_h_S < 0.0, "the naive route follower should drive into the obstacle"


def test_fixed_radius_avoids_static_obstacle():
    env, _ = _static_obstacle_env()
    planner = FixedRadiusPlanner(use_oracle=True)
    for _ in range(2000):
        info = env.step(planner(env))
        assert info.min_h_S >= 0.0, "avoidance planner collided with the obstacle"
        if info.robot_reached_goal:
            break


def test_oracle_planner_never_collides_across_seeds():
    """With perfect knowledge the fixed-radius planner stays collision-free."""
    for seed in range(5):
        env = ConstructionEnv(generate_scene(seed=seed, num_agents=7))
        planner = FixedRadiusPlanner(use_oracle=True)
        for _ in range(1500):
            info = env.step(planner(env))
            assert info.min_h_S >= 0.0
            if info.robot_reached_goal:
                break


def test_oracle_planner_no_worse_than_naive():
    def run(seed, planner):
        env = ConstructionEnv(generate_scene(seed=seed, num_agents=7))
        for _ in range(1500):
            info = env.step(None if planner is None else planner(env))
            if info.robot_reached_goal:
                break
        return env.n_collision_steps

    for seed in range(5):
        assert run(seed, FixedRadiusPlanner(use_oracle=True)) <= run(seed, None)


def test_planner_is_callable_with_env(mini_scene):
    planner = FixedRadiusPlanner(use_oracle=False)
    env = ConstructionEnv(mini_scene)
    for _ in range(10):
        v, omega = planner(env)
        assert -C.ROBOT_MAX_OMEGA - 1e-9 <= omega <= C.ROBOT_MAX_OMEGA + 1e-9
        assert v >= 0.0
        env.step((v, omega))
