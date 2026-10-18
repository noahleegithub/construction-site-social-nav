"""Construction-site social navigation: environment, predictors, planners.

This package implements the system described in "Heterogeneous Trajectory
Prediction and Social Navigation for Construction Site Robots" (CSCE 689):

    One robot navigates a graph-structured construction worksite toward a goal
    while M >= 5 heterogeneous agents (human workers on unicycle dynamics and
    vehicles on bicycle dynamics) work at and travel between stations, obey
    vision cones and blindspots, avoid each other with a social-force model,
    and skirt static high-risk zones. The robot observes agent history,
    forecasts futures (ADE / FDE), plans a socially compliant path, and is
    evaluated both in this environment and on the Robotarium simulator.

Subpackages
-----------
config      : physical / algorithmic constants.
entities    : dataclasses (nodes, highways, agents, robot, zones, scene).
graph       : graph construction and polyline helpers.
dynamics    : unicycle and bicycle integrators.
zones       : static high-risk zones and hazard potentials.
env         : scene generation and the closed-loop simulation.
metrics     : ADE / FDE, control-barrier terms, and the navigation cost J.
evaluate    : open-loop predictor evaluation (ADE / FDE vs horizon and M).
trials      : randomized-trial planner evaluation (cost J and ablations).
data        : rollout dataset generation for the learned predictors.
predict     : trajectory predictors (constant velocity, diffusion, ...).
planner     : robot policies (fixed-radius baseline, joint planner, ...).
robotarium  : bridge to the Robotarium Python simulator.
render      : static and animated matplotlib rendering.
"""

from __future__ import annotations

__version__ = "0.1.0"

from construction_nav.entities import Agent, Highway, Node, Robot, Scene, Zone
from construction_nav.env import ConstructionEnv, generate_scene

__all__ = [
    "__version__",
    "Agent",
    "Highway",
    "Node",
    "Robot",
    "Scene",
    "Zone",
    "ConstructionEnv",
    "generate_scene",
]
