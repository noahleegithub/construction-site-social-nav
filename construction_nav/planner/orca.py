"""ORCA baseline: the robot is just another peer in the crowd.

Section 5's first baseline. The robot runs exactly the policy every other agent
runs -- the same ``orca.solve`` call, the same reciprocal responsibility split,
the same stationary-neighbour rule -- with a preferred velocity taken from its
graph route. It is *not* given any forecast: everything it knows is the current
snapshot of poses and velocities.

The point of the row is to answer "why not just be a well-behaved peer?". ORCA
reasons only over the next ``ORCA_TIME_HORIZON`` seconds and only against
instantaneous velocities, so it has no way to anticipate a worker stepping out
of a blind spot or a vehicle committing to a turn.
"""

from __future__ import annotations

import numpy as np

from construction_nav import config as C
from construction_nav.env import ConstructionEnv
from construction_nav.orca import combined_radius, orca_halfplane, solve
from construction_nav.planner.base import Planner, route_direction, unicycle_command


class OrcaPlanner(Planner):
    """The agents' own reciprocal-avoidance policy, driving the robot."""

    def __init__(
        self,
        forecast=None,
        horizon: float = C.ORCA_TIME_HORIZON,
    ) -> None:
        self.forecast = forecast
        self.horizon = horizon

    @property
    def name(self) -> str:
        if self.forecast is None:
            return "ORCA"
        label = getattr(self.forecast, "name", type(self.forecast).__name__)
        return f"ORCA + forecast ({label})"

    def _constraints(self, env: ConstructionEnv) -> list[tuple[np.ndarray, float]]:
        """ORCA half-planes against every agent the robot can see.

        Mirrors :meth:`ConstructionEnv._avoidance_velocity` exactly, with the
        robot on the human side of the responsibility split (so it yields to
        vehicles, as a person would).
        """
        robot = env.robot
        layout = env.layout
        pos = self._neighbour_states(env)
        own_weight = C.ORCA_RESPONSIBILITY["human"]
        own_vel = robot.speed * np.array([np.cos(robot.theta), np.sin(robot.theta)])

        constraints: list[tuple[np.ndarray, float]] = []
        for index, (nbr_pos, nbr_vel, nbr_kind, nbr_ds, still) in enumerate(pos):
            if float(np.linalg.norm(nbr_pos - robot.pos)) > layout.orca_neighbor_dist:
                continue
            radius = combined_radius(robot.D_S, nbr_ds) + layout.orca_safety_margin
            if still:
                share = 1.0
            else:
                share = own_weight / (own_weight + C.ORCA_RESPONSIBILITY[nbr_kind])
            constraint = orca_halfplane(
                robot.pos,
                own_vel,
                nbr_pos,
                nbr_vel,
                radius,
                self.horizon,
                env.dt,
                share=share,
                tie_break=1.0,
            )
            if constraint is not None:
                constraints.append(constraint)
        return constraints

    def _neighbour_states(self, env: ConstructionEnv):
        """Neighbour states to constrain against: now, and (with a forecast) later.

        A forecast adds the neighbour's predicted state at the horizon as a
        *second* constraint rather than replacing the current one. Replacing it
        would leave the robot free to drive through the space the neighbour
        currently occupies, which is exactly what a forecast should prevent.
        """
        agents = env.scene.agents
        snapshot = env.agent_snapshot()
        states = [
            (
                snapshot["pos"][i],
                snapshot["vel"][i],
                snapshot["kind"][i],
                float(snapshot["D_S"][i]),
                float(np.linalg.norm(snapshot["vel"][i])) < 1e-6,
            )
            for i in range(len(agents))
        ]
        if self.forecast is None:
            return states

        out = self.forecast.predict(env.observe())
        steps = out.mean.shape[1]
        k = min(max(int(round(self.horizon / env.dt)) - 1, 0), steps - 1)
        predicted = out.mean[:, k, :]
        chord = (predicted - snapshot["pos"]) / max(self.horizon, 1e-6)
        states.extend(
            (
                predicted[i],
                chord[i],
                snapshot["kind"][i],
                float(snapshot["D_S"][i]),
                float(np.linalg.norm(chord[i])) < 1e-6,
            )
            for i in range(len(agents))
        )
        return states

    def __call__(self, env: ConstructionEnv) -> tuple[float, float]:
        robot = env.robot
        speed = env.layout.robot_speed
        if env._robot_reached_goal():
            return 0.0, 0.0
        preferred = route_direction(env) * speed
        target = solve(preferred, self._constraints(env), speed)
        return unicycle_command(robot.pos, robot.theta, target, max_speed=speed)
