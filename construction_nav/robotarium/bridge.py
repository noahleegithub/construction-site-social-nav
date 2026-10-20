"""Bridge to the Robotarium simulator.

The construction worksite (tens of metres) is mapped by a uniform scale into the
Robotarium's 3.2 m x 2.0 m arena. Every entity -- the robot and all ``M`` agents
-- is instantiated as a Robotarium unicycle robot; agents are driven by the same
route-following + ORCA avoidance controller used in :mod:`construction_nav.env`,
and bicycle vehicles are represented by their curvature-equivalent turn rate
(``omega = v tan(delta) / L``). The Robotarium therefore owns all integration and
validates collisions / boundaries via ``debug()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from construction_nav import config as C
from construction_nav.entities import Scene
from construction_nav.env import ConstructionEnv
from construction_nav.robotarium.arena import RobotariumTransform


@dataclass
class RobotariumResult:
    """Outcome of a Robotarium run."""

    reached_goal: bool
    t: float
    steps: int
    n_collision_steps: int                 # from our metrics (site frame)
    robotarium_poses: np.ndarray           # (3, N, T) raw Robotarium poses
    site_trajectories: np.ndarray          # (N, T, 2) in construction metres
    validation: dict = field(default_factory=dict)


def run_robotarium_trial(
    scene: Scene,
    planner=None,
    max_steps: int = 4000,
    show_figure: bool = False,
    sim_in_real_time: bool = False,
    barrier_mode: str = "all",
    skip_initialization: bool | None = None,
) -> RobotariumResult:
    """Run one construction-site trial on the Robotarium simulator.

    ``planner`` is a callable ``(env) -> (v, omega)`` in construction units; the
    default is the environment's built-in goal-seeking controller.

    ``barrier_mode`` controls the Robotarium's control-barrier-certificate safety
    filter: ``"all"`` (default) filters every robot -- correct because on the
    physical testbed the agents are GRITSbots too -- ``"robot"`` filters only the
    navigator, and ``"none"`` disables it.
    """
    import rps.robotarium as robotarium
    from rps.utilities.barrier_certificates import create_uni_barrier_certificate

    if barrier_mode not in {"all", "robot", "none"}:
        raise ValueError("barrier_mode must be 'all', 'robot', or 'none'")

    n_agents = len(scene.agents)
    n_robots = 1 + n_agents
    transform = RobotariumTransform(scene.arena_size)

    # Initial conditions in Robotarium units (robot first, then agents).
    site_pos = np.vstack([scene.robot_pos[None, :], np.array([a.pos for a in scene.agents])])
    thetas = np.concatenate(
        [[np.arctan2(scene.robot_heading[1], scene.robot_heading[0])],
         [a.theta for a in scene.agents]]
    )
    ic = np.zeros((3, n_robots))
    ic[:2, :] = transform.to_robotarium(site_pos).T
    ic[2, :] = thetas

    if skip_initialization is None:
        skip_initialization = C.ROBOTARIUM_SKIP_INITIALIZATION

    rps = robotarium.Robotarium(
        number_of_robots=n_robots,
        initial_conditions=ic,
        show_figure=show_figure,
        sim_in_real_time=sim_in_real_time,
        skip_initialization=skip_initialization,
    )
    barrier = create_uni_barrier_certificate() if barrier_mode != "none" else None

    # The Robotarium owns time (TIME_STEP); keep the environment in sync.
    env = ConstructionEnv(scene, dt=rps.TIME_STEP)

    poses_log: list[np.ndarray] = []
    site_log: list[np.ndarray] = []
    reached = False

    # Robotarium requires get_poses() before every step().
    poses = rps.get_poses()
    _sync(env, poses, transform)
    poses_log.append(poses.copy())
    site_log.append(transform.to_site(poses[:2, :].T))

    for _ in range(max_steps):
        # 1. Commands in construction units.
        if planner is None:
            robot_cmd = env.default_robot_command()
        else:
            robot_cmd = planner(env)
        # Agents build their ORCA constraints from one frozen snapshot, and need
        # the robot's velocity for this step to treat it as a moving neighbour.
        env.robot.speed = float(robot_cmd[0])
        snapshot = env.agent_snapshot()
        agent_cmds = [
            env.tick_agent(a, integrate=False, snapshot=snapshot)
            for a in env.scene.agents
        ]

        # 2. Assemble Robotarium unicycle velocities (2 x N).
        U = np.zeros((2, n_robots))
        U[0, 0] = robot_cmd[0] * transform.scale
        U[1, 0] = robot_cmd[1]
        for m, (v, omega) in enumerate(agent_cmds):
            U[0, m + 1] = v * transform.scale
            U[1, m + 1] = omega

        # 3. Optional barrier certificate. On hardware every agent is a physical
        #    robot, so "all" (default) keeps the whole swarm collision-free; the
        #    navigator's command is what our planner contributes.
        if barrier is not None:
            U_safe = barrier(U.copy(), poses)
            if barrier_mode == "all":
                U = U_safe
            else:  # "robot"
                U[:, 0] = U_safe[:, 0]

        # 4. Integrate on the Robotarium and read the new state.
        rps.set_velocities(np.arange(n_robots), U)
        rps.step()
        poses = rps.get_poses()
        poses_log.append(poses.copy())
        site_log.append(transform.to_site(poses[:2, :].T))

        _sync(env, poses, transform, robot_speed=float(robot_cmd[0]))
        info = env.log_step(robot_speed=robot_cmd[0])
        if info.robot_reached_goal:
            reached = True
            break

    validation = _validation(rps)

    return RobotariumResult(
        reached_goal=reached,
        t=env.t,
        steps=len(poses_log),
        n_collision_steps=env.n_collision_steps,
        robotarium_poses=np.stack(poses_log, axis=2),
        site_trajectories=np.stack(site_log, axis=1),
        validation=validation,
    )


def _sync(
    env: ConstructionEnv,
    poses: np.ndarray,
    transform: RobotariumTransform,
    robot_speed: float | None = None,
) -> None:
    site = transform.to_site(poses[:2, :].T)   # (N, 2)
    thetas = poses[2, :]
    env.sync_robot_state(site[0], float(thetas[0]), speed=robot_speed)
    env.sync_agent_states(site[1:], thetas[1:])


def _validation(rps) -> dict:
    """Extract the Robotarium's own collision / boundary validation counters."""
    errors = getattr(rps, "_errors", {}) or {}
    return {
        "robots_too_close": int(errors.get("robots_too_close", 0)),
        "robots_outside_boundaries": int(errors.get("robots_outside_boundaries", 0)),
        "exceeded_actuator_limits": int(errors.get("exceeded_actuator_limits", 0)),
    }
