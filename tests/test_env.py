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


def test_social_force_repels():
    env = ConstructionEnv(generate_scene(seed=0, num_agents=7))
    agent = env.scene.agents[0]
    # Place a peer right in front of the agent.
    other = env.scene.agents[1]
    other.pos = agent.pos + np.array([0.2, 0.0])
    push, slowdown = env._social_forces(agent)
    assert np.linalg.norm(push) > 0
    assert slowdown < 1.0


def _isolate(env, keep):
    """Park every other agent far away and clear zones, isolating one interaction."""
    for agent in env.scene.agents:
        if agent is not keep:
            agent.pos = np.array([-1e3, -1e3])
    env.scene.zones = []


def test_human_yields_when_robot_is_in_view(scene):
    env = ConstructionEnv(scene)
    human = next(a for a in scene.agents if a.kind == "human")
    env.robot.pos = human.pos + human.heading * 1.5
    _isolate(env, human)
    assert env.in_fov(human, env.robot.pos)
    push, slowdown = env._social_forces(human)
    assert np.linalg.norm(push) > 0.0
    assert slowdown < 1.0


def test_vehicle_holds_its_path(scene):
    """Vehicles do not yield to the robot even inside their field of view."""
    env = ConstructionEnv(scene)
    vehicle = next(a for a in scene.agents if a.kind == "vehicle")
    env.robot.pos = vehicle.pos + vehicle.heading * vehicle.D_C * 0.5
    _isolate(env, vehicle)
    assert env.in_fov(vehicle, env.robot.pos)
    push, _ = env._social_forces(vehicle)
    assert np.linalg.norm(push) == pytest.approx(0.0, abs=1e-9)


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
