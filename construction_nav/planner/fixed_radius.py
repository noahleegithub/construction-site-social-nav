"""Fixed-radius avoidance planner with perfect knowledge (planner baseline).

Section 5's planner baseline: the robot knows every other agent's true future
trajectory and simply adds a repulsive velocity whenever it would come within a
fixed radius. It ignores blindspots and heterogeneity, so it isolates the value
of the learned predictor + planner.
"""

from __future__ import annotations

import numpy as np

from construction_nav import config as C
from construction_nav.env import ConstructionEnv
from construction_nav.planner.base import (
    Planner,
    route_direction,
    oracle_futures,
    unicycle_command,
)


class FixedRadiusPlanner(Planner):
    def __init__(
        self,
        avoid_radius: float | None = None,
        avoid_gain: float = 3.0,
        horizon: int = C.HORIZON_STEPS,
        use_oracle: bool = True,
    ) -> None:
        self.avoid_radius = avoid_radius
        self.avoid_gain = avoid_gain
        self.horizon = horizon
        self.use_oracle = use_oracle

    def _futures(self, env: ConstructionEnv) -> np.ndarray:
        if self.use_oracle:
            return oracle_futures(env, self.horizon)
        from construction_nav.predict import ConstantVelocityPredictor

        predictor = ConstantVelocityPredictor(horizon=self.horizon, dt=env.dt)
        return predictor.predict(env.observe()).mean

    def __call__(self, env: ConstructionEnv) -> tuple[float, float]:
        robot = env.robot
        layout = env.layout
        speed = layout.robot_speed
        # Follow the graph route, then add fixed-radius avoidance on top so the
        # robot still advances along its planned corridor.
        desired = route_direction(env) * speed

        # Repulsions are expressed as a *fraction of the robot's nominal speed*
        # (times ``avoid_gain``) so the behaviour is identical at every arena
        # scale: an unscaled term would swamp the tiny miniature-scale speed and
        # leave the robot oscillating in place.
        futures = self._futures(env)  # (M, H, 2)
        agents = env.scene.agents
        for m, agent in enumerate(agents):
            radius = self.avoid_radius if self.avoid_radius is not None else agent.D_C
            deltas = robot.pos - futures[m]  # (H, 2)
            dists = np.linalg.norm(deltas, axis=1)
            k = int(np.argmin(dists))  # only the closest future point repels
            d = float(dists[k])
            if 1e-6 < d < radius:
                desired = desired + (
                    self.avoid_gain * speed * (1.0 - d / radius) * deltas[k] / d
                )

        if env._robot_reached_goal():
            return 0.0, 0.0
        return unicycle_command(robot.pos, robot.theta, desired, max_speed=speed)

    @property
    def name(self) -> str:
        return "fixed-radius (perfect knowledge)"
