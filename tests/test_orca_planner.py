"""The ORCA baseline: the robot runs the agents' own policy."""

from __future__ import annotations

import numpy as np
import pytest

from construction_nav import config as C
from construction_nav.env import ConstructionEnv, generate_scene
from construction_nav.planner import OrcaPlanner


def test_orca_planner_is_a_pure_peer_when_alone():
    """With nobody nearby the ORCA solve must return the preferred velocity."""
    env = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    planner = OrcaPlanner()
    for agent in env.scene.agents:
        agent.pos = env.robot.pos + np.array([40.0, 40.0])
    target = planner._constraints(env)
    assert target == []


def test_orca_planner_yields_to_a_vehicle():
    """The robot takes the human side of the split, so it gives way."""
    env = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    robot = env.robot
    other = env.scene.agents[0]
    other.kind = "vehicle"
    other.D_S = env.layout.radii["vehicle"]["D_S"]
    other.pos = robot.pos + np.array([3.0, 0.0])
    other.heading = np.array([-1.0, 0.0])
    other.speed = other.nominal_speed
    # Put everyone else out of range so only this pair matters.
    for agent in env.scene.agents[1:]:
        agent.pos = np.array([0.5, 0.5])

    constraints = OrcaPlanner()._constraints(env)
    assert constraints, "a vehicle in range must produce a constraint"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_orca_planner_reaches_both_goals_without_collisions(seed):
    env = ConstructionEnv(generate_scene(seed=seed, num_agents=7))
    planner = OrcaPlanner()
    done = False
    for _ in range(3000):
        v, omega = planner(env)
        assert 0.0 <= v <= env.layout.robot_speed + 1e-9
        assert abs(omega) <= C.ROBOT_MAX_OMEGA + 1e-9
        info = env.step((v, omega))
        assert info.min_h_S >= 0.0
        if info.robot_reached_goal and not env.advance_robot_goal():
            done = True
            break
    assert done, f"ORCA did not reach both goals on seed {seed}"
    assert env.robot.goals_reached == len(env.robot.goals)


def test_orca_planner_reports_its_name():
    assert OrcaPlanner().name == "ORCA"
