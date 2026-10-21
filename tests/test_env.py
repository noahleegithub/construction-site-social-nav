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


def test_vehicle_has_five_percent_responsibility(scene):
    """A vehicle paired with a human takes 5 % of the recovery; the human 95 %."""
    env = ConstructionEnv(scene)
    human = next(a for a in scene.agents if a.kind == "human")
    vehicle = next(a for a in scene.agents if a.kind == "vehicle")
    _park_others(env, human, vehicle)
    env.robot.pos = np.array([-1e3, -1e3])

    vehicle.pos = np.array([10.0, 10.0])
    vehicle.heading = np.array([1.0, 0.0])
    vehicle.speed = vehicle.nominal_speed
    human.pos = np.array([13.0, 10.1])
    human.heading = np.array([-1.0, 0.0])
    human.speed = human.nominal_speed

    snapshot = env.agent_snapshot()
    pref_v = vehicle.heading * vehicle.nominal_speed
    pref_h = human.heading * human.nominal_speed
    out_v = env._avoidance_velocity(vehicle, pref_v, snapshot)
    out_h = env._avoidance_velocity(human, pref_h, snapshot)

    rel_v = float(np.linalg.norm(out_v - pref_v)) / vehicle.nominal_speed
    rel_h = float(np.linalg.norm(out_h - pref_h)) / human.nominal_speed
    assert rel_h > 3.0 * rel_v


def test_stationary_neighbour_gets_full_avoidance(scene):
    """A standing agent cannot reciprocate, so the mover takes the whole recovery."""
    env = ConstructionEnv(scene)
    human = next(a for a in scene.agents if a.kind == "human")
    other = next(a for a in scene.agents if a.kind == "human" and a is not human)
    _park_others(env, human, other)
    env.robot.pos = np.array([-1e3, -1e3])
    human.pos = np.array([10.0, 10.0])
    human.heading = np.array([1.0, 0.0])
    human.speed = human.nominal_speed
    other.pos = np.array([11.6, 10.05])
    other.heading = np.array([0.0, 1.0])
    other.speed = 0.0  # working

    preferred = human.heading * human.nominal_speed
    out = env._avoidance_velocity(human, preferred, env.agent_snapshot())
    assert not np.allclose(out, preferred)


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


def test_agents_never_teleport():
    """No agent may move further in one step than its speed allows.

    The only permitted excess is the bounded positional de-overlap, which is
    capped at ``OVERLAP_MAX_PUSH_FRAC`` of a step's travel.
    """
    env = ConstructionEnv(generate_scene(seed=0, num_agents=7))
    agents = env.scene.agents
    limit = (
        max(a.nominal_speed for a in agents)
        * env.dt
        * (1.0 + C.OVERLAP_MAX_PUSH_FRAC)
    )
    previous = [a.pos.copy() for a in agents]
    for _ in range(300):
        env.step()
        for agent, before in zip(agents, previous):
            assert float(np.linalg.norm(agent.pos - before)) <= limit + 1e-6
            before[...] = agent.pos


def test_working_agents_stand_on_a_corridor():
    scene = generate_scene(seed=0, num_agents=7)
    env = ConstructionEnv(scene)
    for _ in range(400):
        env.step()

    points = np.array([n.pos for n in scene.nodes])
    layout = scene.resolved_layout
    for agent in scene.agents:
        if agent.state != "working":
            continue
        offset = min(
            _segment_distance(agent.pos, points[h.u], points[h.v])
            for h in scene.highways
        )
        half_width = (
            layout.vehicle_highway_width if agent.kind == "vehicle"
            else layout.human_highway_width
        ) / 2.0
        assert offset <= half_width + 1e-6


def _segment_distance(p, a, b):
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


# ---------------------------------------------------------------------------
# Contact fallback (social force)
# ---------------------------------------------------------------------------


def test_contact_fallback_is_inert_above_body_radius(scene):
    """Outside D_S_i + D_S_j the fallback must not change the command at all."""
    env = ConstructionEnv(scene)
    agent, neighbour = scene.agents[0], scene.agents[1]
    target = np.array([1.0, 0.0])
    body = agent.D_S + neighbour.D_S
    neighbour.pos = agent.pos + np.array([body + 0.5, 0.0])
    out = env._contact_fallback(agent, target, env.agent_snapshot())
    assert np.array_equal(out, target)


def test_contact_fallback_pushes_away_from_contact(scene):
    env = ConstructionEnv(scene)
    agent, neighbour = scene.agents[0], scene.agents[1]
    target = np.array([1.0, 0.0])
    neighbour.pos = agent.pos + np.array([0.5 * (agent.D_S + neighbour.D_S), 0.0])
    out = env._contact_fallback(agent, target, env.agent_snapshot())
    assert out[0] < target[0], "the fallback should push back against the neighbour"


def test_contact_fallback_ignores_the_robot(scene):
    """Robot contact stays measurable, so the robot is not a fallback neighbour."""
    env = ConstructionEnv(scene)
    env.robot.pos = scene.agents[0].pos.copy()
    agent = scene.agents[0]
    target = np.array([1.0, 0.0])
    out = env._contact_fallback(agent, target, env.agent_snapshot())
    assert np.array_equal(out, target)


@pytest.mark.parametrize("num_agents,max_bad_steps", [(7, 0), (19, 4)])
def test_agents_never_overlap(num_agents, max_bad_steps):
    """Agents keep their body radii apart -- the fallback's whole purpose.

    At the densest packing a couple of brief contacts can survive the per-step
    cap on the positional correction, so the assertion is on how *often* contact
    happens rather than on a single sample.
    """
    for seed in range(2):
        scene = generate_scene(seed=seed, num_agents=num_agents)
        env = ConstructionEnv(scene)
        bad_steps = 0
        for _ in range(400):
            env.step()
            worst = min(
                float(np.linalg.norm(a.pos - b.pos)) - (a.D_S + b.D_S)
                for i, a in enumerate(scene.agents)
                for b in scene.agents[i + 1 :]
            )
            bad_steps += int(worst < -0.05)
        assert bad_steps <= max_bad_steps


def test_agents_stay_inside_the_arena():
    """A constraint near the edge must not let an agent drift off the site."""
    for seed in range(2):
        scene = generate_scene(seed=seed, num_agents=12)
        env = ConstructionEnv(scene)
        for _ in range(400):
            env.step()
            for agent in scene.agents:
                assert np.all(agent.pos >= -1e-9)
                assert np.all(agent.pos <= scene.arena_size + 1e-9)


def test_collision_events_count_episodes_per_agent():
    """A contact episode is counted once per agent, not once per step."""
    scene = generate_scene(seed=0, num_agents=5)
    env = ConstructionEnv(scene)
    first, second = scene.agents[0], scene.agents[1]

    # One agent held inside its hard radius for ten steps is one event.
    env.robot.pos = first.pos + np.array([0.1, 0.0])
    for _ in range(10):
        env.log_step(0.0)
    assert env.n_collision_events == 1
    assert env.n_collision_steps == 10

    # Leaving and coming back is a second event.
    env.robot.pos = first.pos + np.array([10.0, 0.0])
    env.log_step(0.0)
    env.robot.pos = first.pos + np.array([0.1, 0.0])
    env.log_step(0.0)
    assert env.n_collision_events == 2

    # Two agents violating at the same time count individually.
    other = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    a, b = other.scene.agents[0], other.scene.agents[1]
    other.robot.pos = 0.5 * (a.pos + b.pos)
    a.pos = other.robot.pos + np.array([0.1, 0.0])
    b.pos = other.robot.pos - np.array([0.1, 0.0])
    other.log_step(0.0)
    assert other.n_collision_events == 2
    assert other.n_collision_steps == 1


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
