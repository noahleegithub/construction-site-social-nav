"""Scene generation and the closed-loop construction-site simulation.

``generate_scene`` produces a static, reproducible worksite snapshot;
``ConstructionEnv`` advances it in time: heterogeneous agents follow their
two-station routes (transitioning between ``working`` and ``moving``), avoid one
another with a social-force model, respect vision cones and blindspots, and
skirt static high-risk zones, while a single robot is driven by an external
policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from construction_nav import config as C
from construction_nav.dynamics import step_agent, wrap_angle
from construction_nav.entities import Agent, Highway, Node, Robot, Scene
from construction_nav.graph import (
    adjacency,
    arc_lengths,
    build_mesh,
    place,
    place_station,
    point_on_polyline,
    project_arc_length,
    route_length,
    sample_in_annulus,
    sample_in_disk,
    shortest_path,
)
from construction_nav.zones import hazard, sample_zones, zone_repulsion


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------


def generate_scene(
    seed: int = 0,
    num_agents: int = C.DEFAULT_AGENTS,
    arena: float = C.ARENA_SIZE,
    n_human_stations: int = C.N_HUMAN_STATIONS,
    n_vehicle_stations: int = C.N_VEHICLE_STATIONS,
    n_junctions: int = C.N_JUNCTIONS,
    n_zones: int = C.N_ZONES,
    radius_floor: float = 0.0,
) -> Scene:
    """Generate a representative static worksite snapshot per Section 3."""
    if n_junctions < 2:
        raise ValueError("n_junctions must be >= 2 to form a shared junction mesh")
    if n_human_stations < 2 or n_vehicle_stations < 2:
        raise ValueError("each kind needs >= 2 workstations for agents to cycle between")
    if num_agents < 1:
        raise ValueError("num_agents must be >= 1")

    rng = np.random.default_rng(seed)
    layout = C.Layout.for_arena(arena, radius_floor=radius_floor)
    center = np.full(2, arena / 2.0)
    junction_radius = C.JUNCTION_RADIUS_FRAC * arena
    station_r = (C.STATION_RADIUS_FRAC[0] * arena, C.STATION_RADIUS_FRAC[1] * arena)

    nodes: list[Node] = []

    # 1. Neutral interior junctions.
    junction_idx: list[int] = []
    j_positions: list[np.ndarray] = []
    for _ in range(n_junctions):
        pos = place(
            rng,
            lambda: sample_in_disk(rng, center, junction_radius),
            j_positions,
            layout.junction_min_sep,
        )
        junction_idx.append(len(nodes))
        j_positions.append(pos)
        nodes.append(Node(kind=None, role="junction", pos=pos))

    # 2. Junction mesh: spanning tree + extra loop edges, all shared.
    highways = build_mesh(
        junction_idx,
        np.array([n.pos for n in nodes]),
        n_extra=max(0, n_junctions - 1),
        width=layout.shared_highway_width,
    )

    # 3. Peripheral workstation leaves, each attached by one spoke.
    workstation_specs = [("human", n_human_stations), ("vehicle", n_vehicle_stations)]
    stations_by_kind: dict[str, list[np.ndarray]] = {"human": [], "vehicle": []}
    for kind, count in workstation_specs:
        width = (
            layout.human_highway_width if kind == "human"
            else layout.vehicle_highway_width
        )
        other = "vehicle" if kind == "human" else "human"
        for _ in range(count):
            pos = place_station(
                rng,
                lambda: sample_in_annulus(rng, center, station_r[0], station_r[1]),
                j_positions,
                stations_by_kind[kind],
                stations_by_kind[other],
                same_sep=layout.station_min_sep,
                cross_sep=layout.spawn_clearance,
                node_clearance=layout.node_clearance,
            )
            idx = len(nodes)
            nodes.append(Node(kind=kind, role="workstation", pos=pos))
            stations_by_kind[kind].append(pos)
            nearest = min(
                junction_idx,
                key=lambda j: float(np.linalg.norm(pos - nodes[j].pos)),
            )
            highways.append(
                Highway(
                    u=idx,
                    v=nearest,
                    access=C.HUMAN if kind == "human" else C.VEHICLE,
                    width=width,
                )
            )

    # 4. Static high-risk zones, off the graph.
    station_pts = stations_by_kind["human"] + stations_by_kind["vehicle"]
    zones = sample_zones(rng, layout, station_pts, j_positions, n_zones=n_zones)

    scene = Scene(
        arena_size=arena,
        nodes=nodes,
        highways=highways,
        zones=zones,
        seed=seed,
        n_human_stations=n_human_stations,
        n_vehicle_stations=n_vehicle_stations,
        n_junctions=n_junctions,
        layout=layout,
    )

    # 5. Robot: shuttle between the two farthest-apart human workstations.
    human_ws = scene.station_indices("human")
    vehicle_ws = scene.station_indices("vehicle")
    adj_human = adjacency(scene, "human")
    adj_vehicle = adjacency(scene, "vehicle")

    pairs = [
        (float(np.linalg.norm(nodes[a].pos - nodes[b].pos)), a, b)
        for i, a in enumerate(human_ws)
        for b in human_ws[i + 1 :]
    ]
    _, start, goal = max(pairs)
    robot_route = shortest_path(adj_human, start, goal)
    if robot_route is None:
        raise RuntimeError("no robot route between human workstations")
    pts = np.array([nodes[i].pos for i in robot_route])
    heading = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
    scene.robot_route = robot_route
    scene.robot_pos = pts[0].copy()
    scene.robot_heading = heading
    scene.goal_node = goal

    # 6. Agents: two same-kind workstations each, shortest route across the mesh.
    n_vehicles = max(1, int(round(num_agents * C.VEHICLE_FRACTION)))
    n_vehicles = min(n_vehicles, num_agents - 1)
    kinds = ["vehicle"] * n_vehicles + ["human"] * (num_agents - n_vehicles)
    rng.shuffle(kinds)

    used_start: dict[str, set[int]]
    used_start = {"human": {start}, "vehicle": set()}  # robot occupies its human station
    for kind in kinds:
        stations = vehicle_ws if kind == "vehicle" else human_ws
        adj = adj_vehicle if kind == "vehicle" else adj_human

        # Prefer an unoccupied start station so no two agents spawn coincident;
        # the partner station is likewise spread out when possible.
        free_starts = [s for s in stations if s not in used_start[kind]]
        start_pool = free_starts or list(stations)
        a = int(rng.choice(start_pool))
        partners = [s for s in stations if s != a]
        free_partners = [s for s in partners if s not in used_start[kind]]
        b = int(rng.choice(free_partners or partners))
        used_start[kind].add(a)

        route = shortest_path(adj, a, b)
        if route is None:
            raise RuntimeError("no route between same-kind workstations")
        radii = layout.radii[kind]
        scene.agents.append(
            _make_spaced_agent(rng, scene, kind, a, b, route, radii, layout)
        )

    _validate_scene(scene)
    return scene


# ---------------------------------------------------------------------------
# Agent construction helpers
# ---------------------------------------------------------------------------


def _make_agent(
    rng: np.random.Generator,
    scene: Scene,
    kind: str,
    a: int,
    b: int,
    route: list[int],
    radii: dict,
    layout: C.Layout,
) -> Agent:
    points = np.array([scene.nodes[i].pos for i in route])
    state = "working" if rng.random() < C.WORKING_PROBABILITY else "moving"

    # Agents always spawn at their (distinct) start station ``a`` so no two
    # agents of a kind begin coincident; moving agents start near that station.
    if state == "working":
        pos = points[0].copy()
        heading = points[1] - points[0]
    else:
        cum, _ = arc_lengths(points)
        s = rng.uniform(0.05, 0.45) * cum[-1]
        pos, heading = point_on_polyline(points, s)

    norm = np.linalg.norm(heading)
    if norm > 1e-9:
        heading = heading / norm
    else:
        ang = rng.uniform(0, 2 * np.pi)
        heading = np.array([np.cos(ang), np.sin(ang)])

    agent = Agent(
        kind=kind,
        pos=np.asarray(pos),
        heading=heading,
        state=state,
        D_S=radii["D_S"],
        D_C=radii["D_C"],
        station_a=a,
        station_b=b,
        route=list(route),
        nominal_speed=layout.vehicle_speed if kind == "vehicle" else layout.human_speed,
    )
    _prime_agent_dynamics(rng, agent, scene)
    return agent


def _prime_agent_dynamics(rng: np.random.Generator, agent: Agent, scene: Scene) -> None:
    """Initialize the working/moving state machine for a freshly built agent."""
    a, b = agent.station_a, agent.station_b
    pa, pb = scene.nodes[a].pos, scene.nodes[b].pos
    # Which end is the agent at (only meaningful when working)?
    at_a = float(np.linalg.norm(agent.pos - pa)) <= float(np.linalg.norm(agent.pos - pb))

    if agent.state == "working":
        agent.at_station = a if at_a else b
        agent.target_station = b if at_a else a
        agent.dwell_remaining = float(rng.uniform(*C.DWELL_RANGE))
    else:
        # Moving: head toward whichever endpoint is farther away.
        agent.at_station = b if at_a else a
        agent.target_station = b if at_a else a
        agent.dwell_remaining = 0.0
        agent.trip_route = _trip(agent, scene)

    agent.speed = agent.nominal_speed
    agent.history = [agent.pos.copy()]


def _trip(agent: Agent, scene: Scene) -> list[int]:
    """Node route from the agent's current station to its target station."""
    if agent.route and agent.route[0] == agent.at_station:
        return list(agent.route)
    if agent.route and agent.route[-1] == agent.at_station:
        return list(reversed(agent.route))
    adj = adjacency(scene, agent.kind)
    path = shortest_path(adj, agent.at_station, agent.target_station)
    return path or list(agent.route)


def _spawn_margins(scene: Scene, kind: str, pos: np.ndarray) -> tuple[float, float]:
    """Closest distance to (opposite kind, same kind) at a candidate spawn.

    The robot is human-like, so for a human agent it counts as same-kind, and
    for a vehicle it counts as opposite-kind.
    """
    same = [float(np.linalg.norm(pos - ag.pos)) for ag in scene.agents if ag.kind == kind]
    cross = [float(np.linalg.norm(pos - ag.pos)) for ag in scene.agents if ag.kind != kind]
    robot_gap = float(np.linalg.norm(pos - scene.robot_pos))
    if kind == "human":
        same.append(robot_gap)
    else:
        cross.append(robot_gap)
    return (
        min(cross) if cross else float("inf"),
        min(same) if same else float("inf"),
    )


def _make_spaced_agent(
    rng: np.random.Generator,
    scene: Scene,
    kind: str,
    a: int,
    b: int,
    route: list[int],
    radii: dict,
    layout: C.Layout,
) -> Agent:
    best: Agent | None = None
    best_score = -float("inf")
    same_sep = 0.5 * layout.station_min_sep
    for _ in range(C.SPAWN_TRIES):
        candidate = _make_agent(rng, scene, kind, a, b, route, radii, layout)
        cross, same = _spawn_margins(scene, kind, candidate.pos)
        if cross >= layout.spawn_clearance and same >= same_sep:
            return candidate
        score = min(cross - layout.spawn_clearance, same - same_sep)
        if score > best_score:
            best_score = score
            best = candidate
    assert best is not None
    return best


def _validate_scene(scene: Scene) -> None:
    arena = scene.arena_size
    layout = scene.layout or C.Layout.for_arena(arena)
    hubs = scene.junction_indices()
    human_ws = scene.station_indices("human")
    vehicle_ws = scene.station_indices("vehicle")
    assert len(human_ws) == scene.n_human_stations, "wrong number of human workstations"
    assert len(vehicle_ws) == scene.n_vehicle_stations, "wrong number of vehicle workstations"
    assert len(hubs) == scene.n_junctions, "wrong number of junctions"
    assert scene.num_humans >= 1 and scene.num_vehicles >= 1, "need heterogeneous agents"
    assert any(h.access == C.SHARED for h in scene.highways), "need shared corridors"
    assert any(h.access != C.SHARED for h in scene.highways), "need restricted corridors"

    for h in scene.highways:
        if scene.nodes[h.u].role == "junction" and scene.nodes[h.v].role == "junction":
            assert h.access == C.SHARED, "junction mesh corridors must be shared"

    # Agents must not start on top of each other or the robot. Opposite-kind
    # pairs (and any agent vs the robot) are held to the full cross-kind buffer.
    for i, ag in enumerate(scene.agents):
        assert np.linalg.norm(ag.pos - scene.robot_pos) > 1e-6, "agent spawned on the robot"
        for other in scene.agents[i + 1:]:
            assert np.linalg.norm(ag.pos - other.pos) > 1e-6, "agents spawned coincident"
            if ag.kind != other.kind:
                assert np.linalg.norm(ag.pos - other.pos) >= layout.spawn_clearance * 0.5, (
                    "human and vehicle agents start too close"
                )

    degree = {i: 0 for i in range(len(scene.nodes))}
    for h in scene.highways:
        degree[h.u] += 1
        degree[h.v] += 1
    for ws in human_ws + vehicle_ws:
        assert degree[ws] == 1, "workstation must be a degree-1 leaf"

    for kind, stations, adj in (
        ("human", human_ws, adjacency(scene, "human")),
        ("vehicle", vehicle_ws, adjacency(scene, "vehicle")),
    ):
        seen = {stations[0]}
        stack = [stations[0]]
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        assert all(s in seen for s in stations), f"{kind} network is disconnected"

    for agent in scene.agents:
        assert agent.route and agent.route[0] != agent.route[-1]
    assert scene.robot_route

    for node in scene.nodes:
        assert np.all(node.pos >= 0.0) and np.all(node.pos <= arena), "node outside arena"
    for agent in scene.agents:
        assert np.all(agent.pos >= 0.0) and np.all(agent.pos <= arena), "agent outside arena"

    for z in scene.zones:
        assert z.radius > 0.0
        assert np.all(z.center - z.radius >= 0.0), "zone outside arena"
        assert np.all(z.center + z.radius <= arena), "zone outside arena"


# ---------------------------------------------------------------------------
# Observation container
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """Everything a predictor / planner needs at one timestep."""

    t: float
    agent_pos: np.ndarray                 # (M, 2)
    agent_theta: np.ndarray               # (M,)
    agent_kind: list[str]
    agent_state: list[str]
    histories: np.ndarray                 # (M, T, 2) padded to HISTORY_STEPS
    robot_pos: np.ndarray                 # (2,)
    robot_theta: float
    robot_history: np.ndarray             # (T, 2)
    arena: float = C.ARENA_SIZE           # site side length, metres (for scaling)


# ---------------------------------------------------------------------------
# Closed-loop environment
# ---------------------------------------------------------------------------


@dataclass
class StepInfo:
    """Per-step bookkeeping returned by :meth:`ConstructionEnv.step`."""

    t: float
    robot_reached_goal: bool
    min_h_S: float                        # min over agents of Eq. (2) hard term
    min_h_C: float                        # min over agents of Eq. (2) safe term
    collision: bool                       # any h_S < 0 this step
    froze: bool                           # robot commanded ~zero while not done


@dataclass
class ConstructionEnv:
    """Advance a :class:`Scene` in time with heterogeneous agents."""

    scene: Scene
    dt: float = C.DT
    use_social_force: bool = True
    seed: int | None = None
    rng: np.random.Generator = field(init=False)
    robot: Robot = field(init=False)
    layout: C.Layout = field(init=False)
    t: float = field(init=False, default=0.0)
    # accumulated performance logs (Eq. (3))
    travel_time: float = field(init=False, default=0.0)
    freeze_time: float = field(init=False, default=0.0)
    intrusion_integral: float = field(init=False, default=0.0)
    n_collision_steps: int = field(init=False, default=0)
    robot_trajectory: list[np.ndarray] = field(init=False, default_factory=list)
    agent_trajectories: list[list[np.ndarray]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.scene.seed if self.seed is None else self.seed)
        self.layout = self.scene.resolved_layout
        self.robot = Robot(
            pos=self.scene.robot_pos.copy(),
            heading=self.scene.robot_heading.copy(),
            goal=self.scene.goal_pos.copy(),
            route=list(self.scene.robot_route),
            goal_node=self.scene.goal_node,
            D_S=self.layout.robot_radii["D_S"],
            D_C=self.layout.robot_radii["D_C"],
        )
        self._reset_logs()

    # -- setup ------------------------------------------------------------

    def _reset_logs(self) -> None:
        self.t = 0.0
        self.travel_time = 0.0
        self.freeze_time = 0.0
        self.intrusion_integral = 0.0
        self.n_collision_steps = 0
        self.robot.history = [self.robot.pos.copy()]
        self.robot_trajectory = [self.robot.pos.copy()]
        for agent in self.scene.agents:
            agent.history = [agent.pos.copy()]
        self.agent_trajectories = [[a.pos.copy()] for a in self.scene.agents]

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self._reset_logs()
        return self.observe()

    # -- observation ------------------------------------------------------

    def observe(self) -> Observation:
        hist_len = C.HISTORY_STEPS
        m = len(self.scene.agents)
        histories = np.zeros((m, hist_len, 2))
        for i, agent in enumerate(self.scene.agents):
            h = agent.history[-hist_len:]
            pad = hist_len - len(h)
            histories[i, :pad] = h[0]
            histories[i, pad:] = np.array(h)

        rh = self.robot.history[-hist_len:]
        pad = hist_len - len(rh)
        robot_history = np.zeros((hist_len, 2))
        robot_history[:pad] = rh[0]
        robot_history[pad:] = np.array(rh)

        return Observation(
            t=self.t,
            agent_pos=np.array([a.pos for a in self.scene.agents]),
            agent_theta=np.array([a.theta for a in self.scene.agents]),
            agent_kind=[a.kind for a in self.scene.agents],
            agent_state=[a.state for a in self.scene.agents],
            histories=histories,
            robot_pos=self.robot.pos.copy(),
            robot_theta=self.robot.theta,
            robot_history=robot_history,
            arena=self.scene.arena_size,
        )

    # -- stepping ---------------------------------------------------------

    def step(self, robot_cmd: tuple[float, float] | None = None) -> StepInfo:
        """Advance one timestep.

        ``robot_cmd`` is ``(v, omega)``; when ``None`` a built-in goal-seeking
        pure-pursuit controller is used (handy for demos and the baseline).
        """
        for agent in self.scene.agents:
            self.tick_agent(agent)

        if robot_cmd is None:
            robot_cmd = self.default_robot_command()
        v, omega = robot_cmd
        v = float(np.clip(v, 0.0, self.layout.robot_speed))
        omega = float(np.clip(omega, -C.ROBOT_MAX_OMEGA, C.ROBOT_MAX_OMEGA))

        from construction_nav.dynamics import unicycle_step

        self.robot.pos, theta = unicycle_step(
            self.robot.pos, self.robot.theta, v, omega, self.dt
        )
        self.robot.heading = np.array([np.cos(theta), np.sin(theta)])
        self.robot.speed = v
        self.robot.omega = omega

        return self.log_step(v)

    def log_step(self, robot_speed: float) -> StepInfo:
        """Record the current state and accumulate the Eq. (3) cost terms.

        Shared by :meth:`step` and the Robotarium bridge, which owns the actual
        integration but still needs identical bookkeeping.
        """
        self.robot.history.append(self.robot.pos.copy())
        self.robot_trajectory.append(self.robot.pos.copy())
        for i, agent in enumerate(self.scene.agents):
            agent.history.append(agent.pos.copy())
            self.agent_trajectories[i].append(agent.pos.copy())

        min_h_S, min_h_C = self._safety_terms()
        collision = min_h_S < 0.0
        self.n_collision_steps += int(collision)
        self.intrusion_integral += float(np.sum(self._intrusions()) * self.dt)

        reached = self._robot_reached_goal()
        froze = (not reached) and robot_speed < 1e-3
        self.freeze_time += self.dt * float(froze)
        self.travel_time += self.dt * float(not reached)

        self.t += self.dt
        return StepInfo(
            t=self.t,
            robot_reached_goal=reached,
            min_h_S=min_h_S,
            min_h_C=min_h_C,
            collision=collision,
            froze=froze,
        )

    # -- external-state sync (Robotarium bridge) --------------------------

    def sync_agent_states(
        self, positions: np.ndarray, thetas: np.ndarray
    ) -> None:
        """Overwrite agent poses with externally integrated (Robotarium) state."""
        for i, agent in enumerate(self.scene.agents):
            agent.pos = np.asarray(positions[i], dtype=float).copy()
            agent.heading = np.array([np.cos(thetas[i]), np.sin(thetas[i])])
            agent.pos = np.clip(agent.pos, 0.0, self.scene.arena_size)

    def sync_robot_state(self, pos: np.ndarray, theta: float) -> None:
        self.robot.pos = np.asarray(pos, dtype=float).copy()
        self.robot.heading = np.array([np.cos(theta), np.sin(theta)])

    # -- agents -----------------------------------------------------------

    def tick_agent(self, agent: Agent, integrate: bool = True) -> tuple[float, float]:
        """Advance one agent, or (with ``integrate=False``) only compute its command.

        Returns the unicycle command ``(v, omega)`` the agent executes, where
        ``omega`` is the curvature-equivalent turn rate for bicycle vehicles.
        Using ``integrate=False`` lets an external integrator (the Robotarium)
        own the agent's motion while this class owns its behaviour.
        """
        if agent.state == "working":
            agent.dwell_remaining -= self.dt
            agent.speed = 0.0
            agent.omega = 0.0
            if agent.dwell_remaining <= 0.0:
                self._begin_trip(agent)
            return 0.0, 0.0

        v, turn, max_turn = self._route_command(agent)
        if agent.state == "working":
            # ``_route_command`` detected arrival this step and snapped the agent
            # onto the station; nothing to integrate.
            agent.speed = 0.0
            agent.omega = 0.0
            return 0.0, 0.0

        if agent.kind == "vehicle":
            agent.steer = float(np.clip(turn, -max_turn, max_turn))
            agent.omega = float(v * np.tan(agent.steer) / self.layout.wheelbase)
        else:
            agent.omega = float(np.clip(turn, -max_turn, max_turn))

        if integrate:
            agent.pos, theta = step_agent(
                agent.pos, agent.theta, agent.kind, v, turn, self.dt,
                self.layout.wheelbase, max_turn,
            )
            agent.heading = np.array([np.cos(theta), np.sin(theta)])
        agent.speed = v
        return agent.speed, agent.omega

    def _begin_trip(self, agent: Agent) -> None:
        agent.at_station, agent.target_station = agent.target_station, agent.at_station
        agent.state = "moving"
        agent.trip_route = _trip(agent, self.scene)

    def _route_command(self, agent: Agent) -> tuple[float, float, float]:
        """Pure-pursuit route following blended with social forces.

        Returns ``(v, turn, max_turn)``; ``turn`` is an omega (humans) or a
        steering angle (vehicles) and ``max_turn`` bounds it. Handles arrival,
        which flips the agent to ``"working"``.
        """
        points = np.array([self.scene.nodes[i].pos for i in agent.trip_route])
        total = route_length(points)
        s = project_arc_length(points, agent.pos)

        # Arrival: within the arrival radius of the final station.
        if total - s <= self.layout.arrival_radius:
            agent.pos = points[-1].copy()
            agent.state = "working"
            agent.at_station = agent.trip_route[-1]
            agent.dwell_remaining = float(self.rng.uniform(*C.DWELL_RANGE))
            return 0.0, 0.0, C.HUMAN_MAX_OMEGA

        target_pt, heading_dir = point_on_polyline(points, min(s + self.layout.lookahead, total))
        desired = target_pt - agent.pos
        norm = float(np.linalg.norm(desired))
        if norm < 1e-6:
            desired = heading_dir
        else:
            desired = desired / norm

        push, slowdown = self._social_forces(agent)
        if np.linalg.norm(push) > 0:
            desired = desired + push
            dn = float(np.linalg.norm(desired))
            desired = desired / dn if dn > 1e-6 else heading_dir

        heading_error = wrap_angle(np.arctan2(desired[1], desired[0]) - agent.theta)
        align = max(0.0, float(np.cos(heading_error)))
        v = agent.nominal_speed * (0.25 + 0.75 * align) * slowdown

        turn = C.HEADING_GAIN * heading_error
        max_turn = C.VEHICLE_MAX_STEER if agent.kind == "vehicle" else C.HUMAN_MAX_OMEGA
        return float(v), float(turn), float(max_turn)

    def _social_forces(self, agent: Agent) -> tuple[np.ndarray, float]:
        """Return (heading push, speed slowdown factor) for one agent."""
        if not self.use_social_force:
            return np.zeros(2), 1.0

        push = np.zeros(2)
        proximity = 0.0

        for other in self.scene.agents:
            if other is agent:
                continue
            delta = agent.pos - other.pos
            d = float(np.linalg.norm(delta))
            if d > self.layout.sf_radius or d < 1e-6:
                continue
            closeness = 1.0 - d / self.layout.sf_radius
            push += C.SF_AGENT_GAIN * closeness * delta / d
            proximity = max(proximity, closeness)

        # Robot: humans yield when it is in their field of view; vehicles hold
        # their path (Section 3).
        delta = agent.pos - self.robot.pos
        d = float(np.linalg.norm(delta))
        if d <= self.layout.sf_radius and d > 1e-6:
            if agent.kind == "human":
                in_view = self._in_cone(
                    agent.pos, agent.theta, self.robot.pos,
                    C.FOV_HALF_ANGLE, self.layout.fov_range,
                )
                if in_view:
                    closeness = 1.0 - d / self.layout.sf_radius
                    push += C.SF_ROBOT_GAIN * closeness * delta / d
                    proximity = max(proximity, closeness)

        push += zone_repulsion(agent.pos, self.scene.zones, self.layout)

        slowdown = 1.0 - C.SF_MAX_SLOWDOWN * min(1.0, proximity)
        return push, float(np.clip(slowdown, 1.0 - C.SF_MAX_SLOWDOWN, 1.0))

    # -- robot helpers ----------------------------------------------------

    def default_robot_command(self) -> tuple[float, float]:
        """Goal-seeking pure pursuit along the robot route."""
        points = np.array([self.scene.nodes[i].pos for i in self.robot.route])
        total = route_length(points)
        s = project_arc_length(points, self.robot.pos)
        target_pt, _ = point_on_polyline(points, min(s + self.layout.lookahead, total))
        desired = target_pt - self.robot.pos
        dist = float(np.linalg.norm(desired))
        if dist < 1e-3:
            return 0.0, 0.0
        desired = desired / dist
        err = wrap_angle(np.arctan2(desired[1], desired[0]) - self.robot.theta)
        align = max(0.0, float(np.cos(err)))
        v = self.layout.robot_speed * (0.25 + 0.75 * align)
        omega = float(np.clip(C.HEADING_GAIN * err, -C.ROBOT_MAX_OMEGA, C.ROBOT_MAX_OMEGA))
        if self._robot_reached_goal():
            return 0.0, 0.0
        return v, omega

    def _robot_reached_goal(self) -> bool:
        return float(np.linalg.norm(self.robot.pos - self.robot.goal)) <= self.layout.arrival_radius

    # -- safety / cost ----------------------------------------------------

    def _intrusions(self) -> np.ndarray:
        """Per-agent max(0, -h_C) (Eq. (3) integrand)."""
        return np.array(
            [
                max(0.0, -(self._h_C(self.robot.pos, a.pos, a.D_C)))
                for a in self.scene.agents
            ]
        )

    def _safety_terms(self) -> tuple[float, float]:
        h_S = [
            self._h_S(self.robot.pos, a.pos, a.D_S) for a in self.scene.agents
        ]
        h_C = [
            self._h_C(self.robot.pos, a.pos, a.D_C) for a in self.scene.agents
        ]
        return (min(h_S) if h_S else float("inf"), min(h_C) if h_C else float("inf"))

    @staticmethod
    def _h_S(p: np.ndarray, q: np.ndarray, D_S: float) -> float:
        return float(np.sum((p - q) ** 2) - D_S**2)

    @staticmethod
    def _h_C(p: np.ndarray, q: np.ndarray, D_C: float) -> float:
        return float(np.sum((p - q) ** 2) - D_C**2)

    # -- sensing ----------------------------------------------------------

    @staticmethod
    def _in_cone(
        pos: np.ndarray,
        theta: float,
        target: np.ndarray,
        half_angle: float,
        rng_range: float,
    ) -> bool:
        delta = target - pos
        d = float(np.linalg.norm(delta))
        if d > rng_range or d < 1e-6:
            return False
        ang = float(np.arctan2(delta[1], delta[0]))
        return abs(wrap_angle(ang - theta)) <= half_angle

    def in_fov(self, agent: Agent, target: np.ndarray) -> bool:
        return self._in_cone(agent.pos, agent.theta, target, C.FOV_HALF_ANGLE, self.layout.fov_range)

    def in_blindspot(self, agent: Agent, target: np.ndarray) -> bool:
        return self._in_cone(
            agent.pos,
            wrap_angle(agent.theta + np.pi),
            target,
            C.BLIND_HALF_ANGLE[agent.kind],
            self.layout.blind_range,
        )

    def hazard_at(self, pos: np.ndarray) -> float:
        return hazard(pos, self.scene.zones, self.layout.zone_hazard_radius)
