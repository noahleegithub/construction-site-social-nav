"""Coordinate mapping between the construction worksite and the Robotarium.

The worksite is ``[0, arena]^2`` metres; the Robotarium arena is
``[-1.6, 1.6] x [-1.0, 1.0]`` metres. A single uniform scale (so geometry stays
similar) fits the worksite inside the arena with a small margin, centred on the
origin. Angles are preserved; linear velocities scale, angular velocities do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from construction_nav import config as C

#: Robotarium arena boundary [x_min, x_max, y_min, y_max] (metres).
BOUNDARIES = np.array([-1.6, 1.6, -1.0, 1.0])


@dataclass(frozen=True)
class RobotariumTransform:
    """Affine map between worksite metres and Robotarium metres."""

    arena: float
    margin: float = C.ROBOTARIUM_MARGIN

    @property
    def scale(self) -> float:
        width = (BOUNDARIES[1] - BOUNDARIES[0]) - 2 * self.margin
        height = (BOUNDARIES[3] - BOUNDARIES[2]) - 2 * self.margin
        return float(min(width / self.arena, height / self.arena))

    def to_robotarium(self, pos: np.ndarray) -> np.ndarray:
        """Map worksite positions ``(..., 2)`` to Robotarium coordinates."""
        pos = np.asarray(pos, dtype=float)
        return (pos - self.arena / 2.0) * self.scale

    def to_site(self, pos: np.ndarray) -> np.ndarray:
        """Map Robotarium positions ``(..., 2)`` back to worksite metres."""
        pos = np.asarray(pos, dtype=float)
        return pos / self.scale + self.arena / 2.0

    def to_robotarium_velocity(self, vel: np.ndarray) -> np.ndarray:
        return np.asarray(vel, dtype=float) * self.scale

    def to_site_velocity(self, vel: np.ndarray) -> np.ndarray:
        return np.asarray(vel, dtype=float) / self.scale


#: Robotarium robot collision diameter (metres), from the simulator constants.
ROBOTARIUM_COLLISION_DIAMETER = 0.135


def robotarium_scene(
    seed: int = 0,
    num_agents: int = 5,
    arena: float = C.ROBOTARIUM_ARENA,
):
    """Build a miniature construction site that fits the Robotarium.

    The arena is ``ROBOTARIUM_ARENA`` metres and every agent's collision radius is
    floored at the Robotarium's own robot size mapped into site units, so the M
    agents are physically consistent with the ~0.11 m GRITSbots they are
    instantiated as there.
    """
    from construction_nav.env import generate_scene

    transform = RobotariumTransform(arena)
    radius_floor = ROBOTARIUM_COLLISION_DIAMETER / transform.scale
    return generate_scene(
        seed=seed, num_agents=num_agents, arena=arena, radius_floor=radius_floor
    )
