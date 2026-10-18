"""Joint predictor+planner: safety filter, conditioned diffusion, trial harness."""

from __future__ import annotations

import numpy as np
import pytest

from construction_nav import config as C
from construction_nav.env import ConstructionEnv, generate_scene
from construction_nav.planner import FixedRadiusPlanner, JointPlanner
from construction_nav.predict import ConstantVelocityPredictor
from construction_nav.trials import PlannerReport, evaluate_planners, run_trial


def _cv_planner(**kwargs) -> JointPlanner:
    return JointPlanner(
        predictor=ConstantVelocityPredictor(horizon=C.HORIZON_STEPS), **kwargs
    )


# ---------------------------------------------------------------------------
# Planner behaviour (torch-free)
# ---------------------------------------------------------------------------


def test_joint_planner_returns_valid_unicycle_commands():
    env = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    planner = _cv_planner()
    v, omega = planner(env)
    assert 0.0 <= v <= env.layout.robot_speed
    assert -C.ROBOT_MAX_OMEGA <= omega <= C.ROBOT_MAX_OMEGA
    assert np.isfinite([v, omega]).all()


def test_joint_planner_oracle_reaches_goal_without_collisions():
    for seed in (0, 1):
        scene = generate_scene(seed=seed, num_agents=5)
        env = ConstructionEnv(scene)
        planner = JointPlanner(oracle=True)
        reached = False
        for _ in range(2000):
            info = env.step(planner(env))
            if info.robot_reached_goal:
                reached = True
                break
        assert reached, f"oracle joint planner timed out on seed {seed}"
        assert env.n_collision_steps == 0


def test_joint_planner_beats_fixed_radius_on_intrusion():
    """On the same sites the prediction-based planner intrudes less than the
    perfect-knowledge fixed-radius baseline."""
    seeds = (0, 1, 2)
    joint = JointPlanner(predictor=ConstantVelocityPredictor(horizon=C.HORIZON_STEPS))
    fixed = FixedRadiusPlanner(use_oracle=True)
    joint_intrusion = []
    fixed_intrusion = []
    for seed in seeds:
        for planner, sink in ((joint, joint_intrusion), (fixed, fixed_intrusion)):
            env = ConstructionEnv(generate_scene(seed=seed, num_agents=5))
            for _ in range(2000):
                info = env.step(planner(env))
                if info.robot_reached_goal:
                    break
            sink.append(env.intrusion_integral)
    assert np.mean(joint_intrusion) < np.mean(fixed_intrusion)


def test_reactive_guard_scales_speed_down_near_an_agent():
    env = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    planner = _cv_planner(guard_factor=0.5)
    speed = env.layout.robot_speed
    agent = env.scene.agents[0]

    # A distant agent leaves the command untouched.
    agent.pos = env.robot.pos + np.array([10.0, 0.0])
    v_far, _ = planner._guard(env, speed, 0.0)
    assert v_far == pytest.approx(speed)

    # At (almost) the collision radius the speed is driven to ~zero.
    agent.pos = env.robot.pos + np.array([agent.D_S + 1e-3, 0.0])
    v_near, _ = planner._guard(env, speed, 0.0)
    assert v_near < 0.05 * speed


def test_joint_planner_brakes_when_all_manoeuvres_are_unsafe():
    """With an agent parked on the robot's route the planner slows down."""
    from construction_nav.planner import route_direction

    env = ConstructionEnv(generate_scene(seed=0, num_agents=5))
    planner = _cv_planner()
    direction = route_direction(env)
    agent = env.scene.agents[0]
    # Park an agent just over one collision radius ahead, on the robot's path.
    agent.pos = env.robot.pos + direction * (agent.D_S * 1.05)
    v, _ = planner(env)
    assert v < env.layout.robot_speed


# ---------------------------------------------------------------------------
# Trial harness
# ---------------------------------------------------------------------------


def test_run_trial_and_report_aggregate():
    planner = JointPlanner(oracle=True)
    result = run_trial(planner, seed=0, num_agents=5, max_steps=2000)
    assert result.reached
    assert result.n_collision_steps == 0
    assert result.travel_time > 0.0
    assert result.J >= result.travel_time      # lambda_1, lambda_2 >= 0

    reports = evaluate_planners(
        {"oracle": JointPlanner(oracle=True)}, seeds=(0, 1), num_agents=5,
        max_steps=2000, log=lambda *a: None,
    )
    report = reports["oracle"]
    assert isinstance(report, PlannerReport)
    assert report.success_rate == 1.0
    assert report.n_collisions == 0
    payload = report.to_json()
    assert payload["n_trials"] == 2
    assert payload["mean_J"] > 0.0


# ---------------------------------------------------------------------------
# Safety constraints inside reverse diffusion (require torch)
# ---------------------------------------------------------------------------


def test_project_robot_safety_pushes_out_of_collision_disk():
    torch = pytest.importorskip("torch")
    from construction_nav.predict.diffusion import project_robot_safety

    robot = torch.tensor([[[0.0, 0.0]]])                 # (K=1, H=1, 2)
    agents = torch.tensor([[[[0.1, 0.0]]]])              # (K=1, M=1, H=1, 2)
    ds = torch.tensor([0.5])
    projected = project_robot_safety(robot, agents, ds)
    dist = (projected - agents[:, 0]).norm(dim=-1)
    assert float(dist.min()) == pytest.approx(0.5, abs=1e-4)
    # The robot is pushed radially outward by the deficit (0.5 - 0.1 = 0.4).
    assert float(projected[0, 0, 0]) == pytest.approx(-0.4, abs=1e-4)


def test_conditioned_diffusion_respects_hard_constraints():
    torch = pytest.importorskip("torch")
    from construction_nav.predict.diffusion import (
        DiffusionSchedule,
        JointDenoiser,
        ddim_sample_conditioned,
    )

    model = JointDenoiser(horizon=4, d_model=16, n_heads=2)
    schedule = DiffusionSchedule(n_steps=20, dt=0.5)
    hist = torch.rand(1, 2, 8, 2)                        # (B=1, N=2, T=8, 2) in [0, 1]
    kind = torch.zeros(1, 2, dtype=torch.long)
    state = torch.zeros(1, 2, dtype=torch.long)
    plan = torch.full((4, 2), 0.5)                       # robot pinned to the centre
    ds = torch.tensor([0.2])

    samples = ddim_sample_conditioned(
        model, schedule, hist, kind, state, plan, ds, n_samples=3, n_steps=5
    )
    assert samples.shape == (3, 2, 4, 2)
    # The robot row (last entity) is projected onto the safe set w.r.t. the agent.
    dist = (samples[:, -1] - samples[:, 0]).norm(dim=-1)   # (K, H)
    assert float(dist.min()) >= 0.2 - 1e-3
