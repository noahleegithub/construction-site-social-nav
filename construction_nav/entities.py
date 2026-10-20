"""Entity dataclasses: graph nodes, highways, agents, robot, scene.

Ported from ``visualize_scene.py`` (which was already annotated for this split)
and extended with the dynamic state needed by
:mod:`construction_nav.env`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from construction_nav import config as C


# ---------------------------------------------------------------------------
# Worksite graph
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A graph vertex: a workstation leaf or a neutral interior junction."""

    kind: str | None                # "human" | "vehicle" | None (junction)
    role: str                       # "workstation" | "junction"
    pos: np.ndarray                 # (2,) meters


@dataclass
class Highway:
    """A graph edge rendered as a corridor of true metric width."""

    u: int                          # node index
    v: int
    access: frozenset               # subset of {"human", "vehicle"}
    width: float

    def kind_label(self) -> str:
        if self.access == C.SHARED:
            return "shared"
        if self.access == C.VEHICLE:
            return "vehicle"
        return "human"


@dataclass
class Agent:
    """A heterogeneous dynamic agent with a cyclic two-station route.

    ``heading`` is the unit forward vector; ``theta`` is its angle. Position is
    in meters. The ``state`` machine alternates between ``"working"`` (dwelling
    at a station) and ``"moving"`` (travelling along ``route`` toward the next
    station).
    """

    kind: str                       # "human" | "vehicle"
    pos: np.ndarray                 # (2,) meters
    heading: np.ndarray             # (2,) unit vector
    state: str                      # "working" | "moving"
    D_S: float                      # collision radius
    D_C: float                      # safety radius
    station_a: int = -1             # start workstation node index
    station_b: int = -1             # end workstation node index
    route: list[int] = field(default_factory=list)

    # --- dynamic state -----------------------------------------------------
    speed: float = 0.0              # current linear velocity (m/s)
    nominal_speed: float = C.HUMAN_SPEED_REF  # scale-resolved target speed (m/s)
    omega: float = 0.0              # human turn rate (rad/s)
    steer: float = 0.0              # vehicle steering angle (rad)
    at_station: int = -1            # station the agent dwells at when working
    target_station: int = -1        # station currently being approached
    dwell_remaining: float = 0.0    # seconds left in "working"
    trip_route: list[int] = field(default_factory=list)  # node indices, current trip
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def heading_deg(self) -> float:
        return float(np.rad2deg(np.arctan2(self.heading[1], self.heading[0])))

    @property
    def theta(self) -> float:
        return float(np.arctan2(self.heading[1], self.heading[0]))


@dataclass
class Robot:
    """The single navigator: a unicycle-modeled robot with a goal."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2))
    route: list[int] = field(default_factory=list)
    goal_node: int = -1
    D_S: float = C.ROBOT_RADII["D_S"]
    D_C: float = C.ROBOT_RADII["D_C"]
    speed: float = 0.0
    omega: float = 0.0
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def theta(self) -> float:
        return float(np.arctan2(self.heading[1], self.heading[0]))

    @property
    def heading_deg(self) -> float:
        return float(np.rad2deg(self.theta))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    """A generated construction worksite snapshot."""

    arena_size: float
    nodes: list[Node] = field(default_factory=list)
    highways: list[Highway] = field(default_factory=list)
    agents: list[Agent] = field(default_factory=list)
    robot_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    robot_heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    robot_route: list[int] = field(default_factory=list)
    goal_node: int = -1
    seed: int = 0
    n_human_stations: int = C.N_HUMAN_STATIONS
    n_vehicle_stations: int = C.N_VEHICLE_STATIONS
    n_junctions: int = C.N_JUNCTIONS
    layout: "C.Layout | None" = None

    @property
    def num_humans(self) -> int:
        return sum(a.kind == "human" for a in self.agents)

    @property
    def num_vehicles(self) -> int:
        return sum(a.kind == "vehicle" for a in self.agents)

    @property
    def goal_pos(self) -> np.ndarray:
        return self.nodes[self.goal_node].pos if self.goal_node >= 0 else np.zeros(2)

    @property
    def resolved_layout(self) -> "C.Layout":
        """The scale-dependent layout for this scene's arena."""
        return self.layout if self.layout is not None else C.Layout.for_arena(self.arena_size)

    def positions(self) -> np.ndarray:
        return np.array([n.pos for n in self.nodes])

    def station_indices(self, kind: str) -> list[int]:
        return [
            i for i, n in enumerate(self.nodes)
            if n.role == "workstation" and n.kind == kind
        ]

    def junction_indices(self) -> list[int]:
        return [i for i, n in enumerate(self.nodes) if n.role == "junction"]
