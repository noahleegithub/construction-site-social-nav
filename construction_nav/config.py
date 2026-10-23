"""Physical and algorithmic constants for the construction-site environment.

The environment is **scale-invariant**: :class:`Layout` derives every length and
speed from the arena side length ``arena`` relative to a 60 m reference site. So
``arena=60`` reproduces the reference construction worksite (tens of metres,
metre-per-second workers), while ``arena≈3`` yields a geometrically and
dynamically similar miniature site that fits the Robotarium's 3.2 m x 2.0 m
arena with its ~0.11 m robots.

Time is not scaled: arcs take the same wall-clock time at every size, and angular
limits / controller gains are dimensionless, so they are shared constants.

Sections map to the design:

    arena / graph layout      -> Section 3 (environment simulation)
    dynamics limits           -> Section 2 (unicycle humans, bicycle vehicles)
    sensing (FOV / blindspot) -> Section 3
    radii (D_S / D_C)         -> Section 2, Eq. (2)
    forecast / cost           -> Section 2 (Eq. (1)) and Eq. (3)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Arena and reference-site layout (values at the 60 m reference size)
# ---------------------------------------------------------------------------

ARENA_REF = 60.0                  # reference worksite side length, meters
ARENA_SIZE = ARENA_REF            # default arena
DEFAULT_AGENTS = 7                # M >= 5 other agents per the problem statement
MIN_AGENTS = 5                    # M floor: 6 entities including the robot
MAX_AGENTS = 19                   # M ceiling: 20 entities including the robot
VEHICLE_FRACTION = 0.35           # share of the non-robot agents that are vehicles

# Workstations are the lower bound; ``generate_scene`` grows them with the agent
# count so every agent can spawn at a distinct station.
N_HUMAN_STATIONS = 5
N_VEHICLE_STATIONS = 3
N_JUNCTIONS = 5

JUNCTION_RADIUS_FRAC = 0.28       # central mesh disk, fraction of arena
STATION_INNER_FRAC = 0.32         # workstations live outside this central disk
STATION_MARGIN_FRAC = 0.04        # inset from the arena boundary
JUNCTION_MIN_SEP_REF = 12.0       # meters between junction nodes
STATION_MIN_SEP_REF = 9.0         # meters between same-kind workstation nodes
SPAWN_CROSS_KIND_CLEARANCE_REF = 12.0  # meters between human/vehicle at spawn
NODE_CLEARANCE_REF = 1.0          # meters a workstation keeps clear of a junction

HUMAN_HIGHWAY_WIDTH_REF = 3.0
VEHICLE_HIGHWAY_WIDTH_REF = 9.0
SHARED_HIGHWAY_WIDTH_REF = 9.0

RADII_REF = {
    "human": {"D_S": 0.4, "D_C": 1.2},
    "vehicle": {"D_S": 1.5, "D_C": 3.0},
}

FOV_RANGE_REF = 4.5
BLIND_RANGE_REF = 3.5

HUMAN_SPEED_REF = 1.4             # m/s nominal
VEHICLE_SPEED_REF = 1.0           # m/s nominal (site vehicles move slower than people)
VEHICLE_WHEELBASE_REF = 2.5       # m

LOOKAHEAD_REF = 6.0               # m lookahead along the route
ARRIVAL_RADIUS_REF = 1.5          # m within which a station counts as reached

# ---------------------------------------------------------------------------
# Reciprocal collision avoidance (ORCA)
# ---------------------------------------------------------------------------

ORCA_TIME_HORIZON = 4.0           # s: look-ahead over which conflicts are avoided
ORCA_NEIGHBOR_DIST_REF = 6.0      # m: neighbour interaction radius
# Recovery-weight per kind: the pair splits the recovery vector in proportion
# ``w_i / (w_i + w_j)``. A vehicle paired with a human therefore takes
# 1/19 / (1/19 + 1) = 5 %, and the human does the other 95 % -- everyone yields
# to vehicles. Like kinds always split 50/50.
ORCA_RESPONSIBILITY = {"human": 1.0, "vehicle": 1.0 / 19.0}
# Extra clearance on top of the combined radius. A human covers 0.7 m per 0.5 s
# step at the reference scale, so without a margin the discretisation alone can
# tunnel a pair through the constraint.
ORCA_SAFETY_MARGIN_REF = 0.35     # m

# ---------------------------------------------------------------------------
# Contact fallback (social force)
# ---------------------------------------------------------------------------

# ORCA reasons in velocity space over a 4 s horizon and can still let a pair
# drift into contact -- a slow vehicle with a 4.3 m turning circle, or two
# agents converging on the same station berth. Below the *body* radius
# ``D_S_i + D_S_j`` an exponential repulsion takes over, and any residual
# overlap is resolved positionally. The fallback is inert above that radius, so
# ordinary navigation is pure ORCA.
SOCIAL_FORCE_GAIN = 2.0           # correction speed, in units of nominal speed
SOCIAL_FORCE_DECAY_REF = 0.30     # m: exponential decay length of the fallback
SOCIAL_FORCE_MAX_MULT = 3.0       # cap on the correction, x nominal speed
# Positional de-overlap is capped per agent per step so recovery from a deep
# overlap takes a few steps instead of teleporting.
OVERLAP_MAX_PUSH_FRAC = 0.5       # x speed * dt
OVERLAP_RELAX_ITERS = 4           # separation passes before the cap is applied

ROBOT_RADII_REF = {"D_S": 0.4, "D_C": 1.2}
ROBOT_SPEED_REF = 1.4             # m/s nominal

# ---------------------------------------------------------------------------
# Access sets: which traffic may use an edge.
# ---------------------------------------------------------------------------

HUMAN = frozenset({"human"})
VEHICLE = frozenset({"vehicle"})
SHARED = frozenset({"human", "vehicle"})

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------

DT = 0.5                          # default simulation timestep, seconds

# ---------------------------------------------------------------------------
# Sensing geometry (radians)
# ---------------------------------------------------------------------------

FOV_HALF_ANGLE = np.deg2rad(60.0)
BLIND_HALF_ANGLE = {"human": np.deg2rad(40.0), "vehicle": np.deg2rad(80.0)}

# ---------------------------------------------------------------------------
# Dimensionless dynamics limits / controller gains
# ---------------------------------------------------------------------------

HUMAN_MAX_OMEGA = np.deg2rad(120.0)   # rad/s turn-rate limit
VEHICLE_MAX_STEER = np.deg2rad(30.0)  # rad steering limit
HEADING_GAIN = 2.5                    # controller gain on heading error
ROBOT_MAX_OMEGA = np.deg2rad(120.0)

DWELL_RANGE = (4.0, 14.0)         # seconds spent "working" at a station
WORKING_PROBABILITY = 0.45        # fraction of agents that start stationary

# ---------------------------------------------------------------------------
# Forecast and navigation cost (Eq. (1) and Eq. (3))
# ---------------------------------------------------------------------------

HORIZON_STEPS = 16                # forecast horizon H
HISTORY_STEPS = 8                 # observation window length
LAMBDA_INTRUSION = 1.0            # lambda_1: safety-radius intrusion weight
LAMBDA_FREEZE = 0.5               # lambda_2: freezing-time weight

# ---------------------------------------------------------------------------
# Robotarium bridge (arena is 3.2 m x 2.0 m; unicycle robots only)
# ---------------------------------------------------------------------------

ROBOTARIUM_ARENA = 3.0            # arena size used for Robotarium demos, metres
ROBOTARIUM_SKIP_INITIALIZATION = True
ROBOTARIUM_MARGIN = 0.08          # m inset from the boundaries for placements

# ---------------------------------------------------------------------------
# Resolved per-arena layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """All scale-dependent lengths and speeds resolved for a given arena size."""

    arena: float
    k: float                                   # arena / ARENA_REF

    # graph layout (metres)
    junction_min_sep: float
    station_min_sep: float
    spawn_clearance: float
    node_clearance: float
    human_highway_width: float
    vehicle_highway_width: float
    shared_highway_width: float

    # sensing (metres)
    fov_range: float
    blind_range: float

    # collision / safety radii (metres)
    radii: dict
    robot_radii: dict

    # dynamics (metres, metres/second)
    human_speed: float
    vehicle_speed: float
    wheelbase: float
    robot_speed: float

    # route following and interaction (metres)
    lookahead: float
    arrival_radius: float
    orca_neighbor_dist: float
    orca_safety_margin: float
    social_force_decay: float

    @classmethod
    def for_arena(cls, arena: float, radius_floor: float = 0.0) -> "Layout":
        """Resolve the layout for ``arena``.

        ``radius_floor`` raises every collision radius to at least that many
        metres; it is used to make a miniature site physically consistent
        with the Robotarium's ~0.11 m robots (see
        :func:`construction_nav.robotarium.arena.robotarium_scene`). Safety
        radii keep their reference ratio to the collision radius, so the
        ``D_C > D_S`` hierarchy is preserved at every scale.
        """
        k = arena / ARENA_REF

        def floored(d_s_ref: float, d_c_ref: float) -> tuple[float, float]:
            d_s = max(d_s_ref * k, radius_floor)
            # Preserve the reference D_C / D_S ratio when the floor binds.
            d_c = max(d_c_ref * k, radius_floor * (d_c_ref / d_s_ref))
            return d_s, d_c

        def radii(kind: str) -> dict:
            d_s, d_c = floored(RADII_REF[kind]["D_S"], RADII_REF[kind]["D_C"])
            return {"D_S": d_s, "D_C": d_c}

        robot_d_s, robot_d_c = floored(
            ROBOT_RADII_REF["D_S"], ROBOT_RADII_REF["D_C"]
        )

        return cls(
            arena=arena,
            k=k,
            junction_min_sep=JUNCTION_MIN_SEP_REF * k,
            station_min_sep=STATION_MIN_SEP_REF * k,
            spawn_clearance=SPAWN_CROSS_KIND_CLEARANCE_REF * k,
            node_clearance=NODE_CLEARANCE_REF * k,
            human_highway_width=HUMAN_HIGHWAY_WIDTH_REF * k,
            vehicle_highway_width=VEHICLE_HIGHWAY_WIDTH_REF * k,
            shared_highway_width=SHARED_HIGHWAY_WIDTH_REF * k,
            fov_range=FOV_RANGE_REF * k,
            blind_range=BLIND_RANGE_REF * k,
            radii={kind: radii(kind) for kind in RADII_REF},
            robot_radii={"D_S": robot_d_s, "D_C": robot_d_c},
            human_speed=HUMAN_SPEED_REF * k,
            vehicle_speed=VEHICLE_SPEED_REF * k,
            wheelbase=VEHICLE_WHEELBASE_REF * k,
            robot_speed=ROBOT_SPEED_REF * k,
            lookahead=LOOKAHEAD_REF * k,
            arrival_radius=ARRIVAL_RADIUS_REF * k,
            orca_neighbor_dist=max(ORCA_NEIGHBOR_DIST_REF * k, 2.5 * radius_floor),
            orca_safety_margin=max(
                ORCA_SAFETY_MARGIN_REF * k, 0.5 * radius_floor
            ),
            social_force_decay=max(
                SOCIAL_FORCE_DECAY_REF * k, 0.5 * radius_floor
            ),
        )


# Backwards-compatible aliases (reference-site values).
JUNCTION_MIN_SEP = JUNCTION_MIN_SEP_REF
STATION_MIN_SEP = STATION_MIN_SEP_REF
SPAWN_CROSS_KIND_CLEARANCE = SPAWN_CROSS_KIND_CLEARANCE_REF
HUMAN_HIGHWAY_WIDTH = HUMAN_HIGHWAY_WIDTH_REF
VEHICLE_HIGHWAY_WIDTH = VEHICLE_HIGHWAY_WIDTH_REF
SHARED_HIGHWAY_WIDTH = SHARED_HIGHWAY_WIDTH_REF
RADII = RADII_REF
FOV_RANGE = FOV_RANGE_REF
BLIND_RANGE = BLIND_RANGE_REF
HUMAN_SPEED = HUMAN_SPEED_REF
VEHICLE_SPEED = VEHICLE_SPEED_REF
VEHICLE_WHEELBASE = VEHICLE_WHEELBASE_REF
LOOKAHEAD = LOOKAHEAD_REF
ARRIVAL_RADIUS = ARRIVAL_RADIUS_REF
ROBOT_RADII = ROBOT_RADII_REF
ROBOT_SPEED = ROBOT_SPEED_REF

MAX_TRIES = 800
SPAWN_TRIES = 200
SHOW_JUNCTIONS = False
SHOW_CENTERLINES = False
