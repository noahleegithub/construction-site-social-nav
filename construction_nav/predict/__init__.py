"""Trajectory predictors.

The constant-velocity baseline is always available; the learned predictors
require the optional ``torch`` extra and are imported lazily via
:func:`load_diffusion` / :func:`load_trajectron` (or by importing their modules
directly), so the core package works without torch installed.
"""

from __future__ import annotations

from construction_nav.predict.base import (
    Predictor,
    PredictorOutput,
    estimate_velocities,
)
from construction_nav.predict.constant_velocity import ConstantVelocityPredictor

__all__ = [
    "Predictor",
    "PredictorOutput",
    "ConstantVelocityPredictor",
    "estimate_velocities",
    "load_diffusion",
    "load_trajectron",
]


def load_diffusion(path, device: str = "cpu"):
    """Load a trained diffusion predictor (requires torch)."""
    from construction_nav.predict.diffusion import load_predictor

    return load_predictor(path, device=device)


def load_trajectron(path, device: str = "cpu"):
    """Load a trained reduced Trajectron++ predictor (requires torch)."""
    from construction_nav.predict.trajectron import load_predictor

    return load_predictor(path, device=device)
