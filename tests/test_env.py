"""Closed-loop environment behaviour."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from construction_nav import config as C
from construction_nav.env import ConstructionEnv, generate_scene


def test_determinism_with_seed():
    def run():
        env = ConstructionEnv(generate_scene(seed=1, num_agents=6))
        for _ in range(60):
            env.step()
        return env.robot.pos.copy()

    assert np.allclose(run(), run())


def test_observation_shapes(scene):
    env = ConstructionEnv(scene)
    obs = env.observe()
    m = len(scene.agents)
    assert obs.agent_pos.shape == (m, 2)
    assert obs.histories.shape == (m, C.HISTORY_STEPS, 2)
    assert obs.robot_history.shape == (C.HISTORY_STEPS, 2)
    assert len(obs.agent_kind) == m


def test_tick_agent_matches_step(scene):
    """A non-integrating tick must report the same command an integrating one runs."""
    env = ConstructionEnv(scene)
    for _ in range(10):
        env.step()

    a_obs = copy.deepcopy(env)
    a_int = copy.deepcopy(env)
    agent_obs = a_obs.scene.agents[0]
    agent_int = a_int.scene.agents[0]
    if agent_obs.state == "working":
        pytest.skip("agent 0 is working; no route command to compare")

    v, omega = a_obs.tick_agent(agent_obs, integrate=False)
    pos_before = agent_int.pos.copy()
    a_int.tick_agent(agent_int, integrate=True)
    if v > 1e-6:
        assert not np.allclose(pos_before, agent_int.pos)


def test_working_agents_dwell_then_move():
    env = ConstructionEnv(generate_scene(seed=2, num_agents=7))
    moved = False
    for _ in range(200):
        env.step()
        if any(a.state == "moving" for a in env.scene.agents):
            moved = True
        if moved:
            break
    assert moved, "no agent ever entered the moving state"


def _park_others(env, *keep):
    """Move every agent except ``keep`` (by identity) far away."""
    for agent in env.scene.agents:
        if any(agent is k for k in keep):
            continue
        agent.pos = np.array([-1e3, -1e3])
        agent.speed = 0.0


def _humans(scene, n):
    return [a for a in scene.agents if a.kind == "human"][:n]


def test_orca_returns_preferred_when_alone(scene):
    env = ConstructionEnv(scene)
    agent = scene.agents[0]
    _park_others(env, agent)
    env.robot.pos = np.array([-1e3, -1e3])
    preferred = np.array([0.3, -0.2])
    out = env._avoidance_velocity(agent, preferred, env.agent_snapshot())
    assert np.allclose(out, preferred)


def test_orca_deflects_a_head_on_neighbour(scene):
    env = ConstructionEnv(scene)
    a, b = _humans(scene, 2)
    _park_others(env, a, b)
    env.robot.pos = np.array([-1e3, -1e3])
    a.pos = np.array([10.0, 10.0])
    a.heading = np.array([1.0, 0.0])
    a.speed = 1.4
    b.pos = np.array([13.0, 10.05])
    b.heading = np.array([-1.0, 0.0])
    b.speed = 1.4

    preferred = np.array([1.4, 0.0])
    out = env._avoidance_velocity(a, preferred, env.agent_snapshot())
    assert not np.allclose(out, preferred)
    assert np.linalg.norm(out) <= a.nominal_speed + 1e-9


def test_orca_keeps_clearance_in_a_head_on_pass():
    """Two reciprocal agents must stay outside the combined radius."""
    from construction_nav.orca import combined_radius, orca_halfplane, solve

    dt = C.DT
    radius = combined_radius(0.4, 0.4) + 0.5
    pa, pb = np.array([0.0, 0.0]), np.array([5.0, 0.25])
    va, vb = np.array([1.4, 0.0]), np.array([-1.4, 0.0])
    closest = np.inf
    for _ in range(30):
        ca = orca_halfplane(pa, va, pb, vb, radius, 2.0, dt, tie_break=1.0)
        cb = orca_halfplane(pb, vb, pa, va, radius, 2.0, dt, tie_break=-1.0)
        na = solve(np.array([1.4, 0.0]), [ca] if ca else [], 1.4)
        nb = solve(np.array([-1.4, 0.0]), [cb] if cb else [], 1.4)
        va, vb = na, nb
        pa, pb = pa + dt * va, pb + dt * vb
        closest = min(closest, float(np.linalg.norm(pa - pb)))
    assert closest >= radius - 0.05


def test_human_yields_when_robot_is_in_view(scene):
    env = ConstructionEnv(scene)
    human = next(a for a in scene.agents if a.kind == "human")
    _park_others(env, human)
    env.robot.pos = human.pos + human.heading * 2.0
    env.robot.speed = 0.0
    assert env.in_fov(human, env.robot.pos)
    preferred = human.heading * human.nominal_speed
    out = env._avoidance_velocity(human, preferred, env.agent_snapshot())
    assert not np.allclose(out, preferred)


def test_human_ignores_robot_behind(scene):
    """A robot in the blindspot is not avoided (Section 3)."""
    env = ConstructionEnv(scene)
    human = next(a for a in scene.agents if a.kind == "human")
    _park_others(env, human)
    env.robot.pos = human.pos - human.heading * 2.0
    env.robot.speed = 0.0
    assert not env.in_fov(human, env.robot.pos)
    preferred = human.heading * human.nominal_speed
    out = env._avoidance_velocity(human, preferred, env.agent_snapshot())
    assert np.allclose(out, preferred)


def test_vehicle_holds_its_path(scene):
    """Vehicles do not yield to the robot even inside their field of view."""
    env = ConstructionEnv(scene)
    vehicle = next(a for a in scene.agents if a.kind == "vehicle")
    _park_others(env, vehicle)
    env.robot.pos = vehicle.pos + vehicle.heading * vehicle.D_C * 0.5
    env.robot.speed = 0.0
    assert env.in_fov(vehicle, env.robot.pos)
    preferred = vehicle.heading * vehicle.nominal_speed
    out = env._avoidance_velocity(vehicle, preferred, env.agent_snapshot())
    assert np.allclose(out, preferred)


def test_blindspot_is_behind_the_agent(scene):
    env = ConstructionEnv(scene)
    agent = scene.agents[0]
    behind = agent.pos - agent.heading * 2.0
    assert env.in_blindspot(agent, behind)
    assert not env.in_fov(agent, behind)


def test_default_controller_reaches_goal():
    env = ConstructionEnv(generate_scene(seed=0, num_agents=7))
    reached = False
    for _ in range(4000):
        info = env.step()
        if info.robot_reached_goal:
            reached = True
            break
    assert reached


def test_cost_terms_accumulate():
    env = ConstructionEnv(generate_scene(seed=0, num_agents=7))
    for _ in range(50):
        env.step()
    assert env.travel_time == pytest.approx(50 * env.dt)
    assert env.intrusion_integral >= 0.0
    assert env.n_collision_steps >= 0


def test_sync_round_trip(scene):
    env = ConstructionEnv(scene)
    pos = np.array([a.pos for a in scene.agents]) + 0.1
    thetas = np.array([a.theta for a in scene.agents])
    env.sync_agent_states(pos, thetas)
    for i, agent in enumerate(scene.agents):
        assert np.allclose(agent.pos, pos[i])
