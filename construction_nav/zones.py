"""Static high-risk work zones and their hazard potentials (Section 3).

Zones are non-overlapping disks placed off the highway graph, so routes can
cross their hazard field while still being reachable. The hazard potential is a
smooth, monotone falloff used as a repulsive field for agents and as a cost term
for the robot planner.
"""

from __future__ import annotations

import numpy as np

from construction_nav import config as C
from construction_nav.entities import Zone
from construction_nav.graph import sample_in_annulus


def sample_zones(
    rng: np.random.Generator,
    layout: C.Layout,
    stations: list[np.ndarray],
    junctions: list[np.ndarray],
    n_zones: int = C.N_ZONES,
) -> list[Zone]:
    """Place non-overlapping high-risk zones inside the arena, away from nodes."""
    arena = layout.arena
    zones: list[Zone] = []
    r_min, r_max = layout.zone_radius_range
    for _ in range(n_zones):
        placed = False
        for _ in range(C.MAX_TRIES):
            radius = rng.uniform(r_min, r_max)
            center = rng.uniform(radius, arena - radius, size=2)
            if center[1] > arena - radius or center[0] > arena - radius:
                continue
            if any(np.linalg.norm(center - z.center) < z.radius + radius + 2.0 * layout.k for z in zones):
                continue
            if any(np.linalg.norm(center - p) < layout.zone_min_sep for p in stations):
                continue
            if any(np.linalg.norm(center - p) < layout.zone_min_sep for p in junctions):
                continue
            zones.append(Zone(center=center, radius=radius))
            placed = True
            break
        if not placed:
            radius = 0.5 * r_min
            center = sample_in_annulus(rng, np.full(2, arena / 2), 0.3 * arena, 0.45 * arena)
            center = np.clip(center, radius, arena - radius)
            zones.append(Zone(center=center, radius=radius))
    return zones


def hazard(pos: np.ndarray, zones: list[Zone], scale: float) -> float:
    """Scalar hazard in [0, 1] at ``pos``: 1 at a zone centre, decaying smoothly.

    ``h = sum_z exp(-(d_z / scale)^2)`` saturates near 1 inside a zone.
    """
    total = 0.0
    for z in zones:
        d = float(np.linalg.norm(pos - z.center))
        total += float(np.exp(-((d / scale) ** 2)))
    return total


def zone_repulsion(pos: np.ndarray, zones: list[Zone], layout: C.Layout) -> np.ndarray:
    """Repulsive vector pushing ``pos`` away from nearby zones (for agents)."""
    force = np.zeros(2)
    for z in zones:
        delta = pos - z.center
        d = float(np.linalg.norm(delta))
        if d < 1e-6:
            continue
        if d > z.radius + layout.sf_radius:
            continue
        force += (
            C.SF_ZONE_GAIN
            * float(np.exp(-((d / layout.zone_hazard_radius) ** 2)))
            * delta / d
        )
    return force
