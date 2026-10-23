"""Constant-velocity baseline predictor.

The proposal's first baseline (Section 5). Each agent's velocity is estimated by
finite differences on its observed history and extrapolated linearly. The
including-the-robot forecast is just the robot's own history extrapolated.
"""

from __future__ import annotations

import numpy as np

from construction_nav.env import Observation
from construction_nav.predict.base import Predictor, PredictorOutput, estimate_velocities


class ConstantVelocityPredictor(Predictor):
    def predict(self, obs: Observation) -> PredictorOutput:
        vel = estimate_velocities(obs.histories, self.dt)          # (M, 2)
        last = obs.histories[:, -1, :]                             # (M, 2)
        steps = (np.arange(1, self.horizon + 1) * self.dt)         # (H,)
        mean = last[:, None, :] + vel[:, None, :] * steps[None, :, None]
        return PredictorOutput(mean=mean)

    @property
    def name(self) -> str:
        return "Constant Velocity"
