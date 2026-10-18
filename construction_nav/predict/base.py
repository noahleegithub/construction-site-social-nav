"""Predictor interface shared by every forecast model.

A predictor consumes the observation produced by
:meth:`construction_nav.env.ConstructionEnv.observe` (agent histories, poses,
types) and returns future positions for the ``M`` agents over a horizon ``H``.

Multimodal models (diffusion, Trajectron++-style CVAE) additionally return ``K``
samples with mode probabilities so that best-of-K metrics
(:func:`construction_nav.metrics.min_ade`) can be computed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from construction_nav import config as C
from construction_nav.env import Observation


@dataclass
class PredictorOutput:
    """A forecast.

    ``mean``      : (M, H, 2) most likely / averaged future.
    ``samples``   : (M, K, H, 2) optional multimodal samples.
    ``weights``   : (M, K) optional per-agent sample weights (sum to 1).
    """

    mean: np.ndarray
    samples: np.ndarray | None = None
    weights: np.ndarray | None = None
    aux: dict = field(default_factory=dict)


class Predictor(ABC):
    """Base class for trajectory predictors."""

    #: whether :meth:`sample` returns more than the single mean trajectory
    multimodal: bool = False

    def __init__(self, horizon: int = C.HORIZON_STEPS, dt: float = C.DT) -> None:
        self.horizon = horizon
        self.dt = dt

    @abstractmethod
    def predict(self, obs: Observation) -> PredictorOutput:
        """Return a forecast for the current observation."""

    def sample(self, obs: Observation, n: int) -> np.ndarray:
        """Return ``(M, n, H, 2)`` samples; unimodal models tile the mean."""
        out = self.predict(obs).mean
        return np.repeat(out[:, None, :, :], n, axis=1)

    @property
    def name(self) -> str:
        return type(self).__name__


def estimate_velocities(histories: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference velocity (``M, 2``) from the tail of padded histories.

    ``histories`` is ``(M, T, 2)``; padded leading frames are handled by walking
    back to the first frame that actually changed.
    """
    m, t, _ = histories.shape
    vel = np.zeros((m, 2))
    for i in range(m):
        h = histories[i]
        j = t - 1
        while j > 0 and np.allclose(h[j], h[j - 1]):
            j -= 1
        if j > 0:
            vel[i] = (h[j] - h[j - 1]) / dt
    return vel
