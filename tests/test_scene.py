"""Scene generation invariants and scale invariance."""

from __future__ import annotations

import numpy as np
import pytest

from construction_nav import config as C
from construction_nav.env import generate_scene
from construction_nav.graph import adjacency, coverage_fraction, shortest_path


def test_scene_counts(scene):
    assert len(scene.station_indices("human")) == scene.n_human_stations
    assert len(scene.station_indices("vehicle")) == scene.n_vehicle_stations
    assert len(scene.junction_indices()) == scene.n_junctions
    assert scene.num_humans >= 1 and scene.num_vehicles >= 1


def test_workstations_are_leaves(scene):
    degree = {i: 0 for i in range(len(scene.nodes))}
    for h in scene.highways:
        degree[h.u] += 1
        degree[h.v] += 1
    for ws in scene.station_indices("human") + scene.station_indices("vehicle"):
        assert degree[ws] == 1


def test_mesh_corridors_shared(scene):
    for h in scene.highways:
        if scene.nodes[h.u].role == "junction" and scene.nodes[h.v].role == "junction":
            assert h.access == C.SHARED


def test_traffic_networks_connected(scene):
    for kind in ("human", "vehicle"):
        stations = scene.station_indices(kind)
        adj = adjacency(scene, kind)
        seen = {stations[0]}
        stack = [stations[0]]
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        assert all(s in seen for s in stations)
        assert shortest_path(adj, stations[0], stations[1]) is not None


def test_no_spawn_overlap(scene):
    pos = np.array([a.pos for a in scene.agents])
    for i in range(len(pos)):
        assert np.linalg.norm(pos[i] - scene.robot_pos) > 1e-6
        for j in range(i + 1, len(pos)):
            assert np.linalg.norm(pos[i] - pos[j]) > 1e-6


def test_seed_reproducibility():
    a = generate_scene(seed=3, num_agents=6)
    b = generate_scene(seed=3, num_agents=6)
    assert [n.pos.tolist() for n in a.nodes] == [n.pos.tolist() for n in b.nodes]
    assert [ag.kind for ag in a.agents] == [ag.kind for ag in b.agents]
    assert np.allclose(a.robot_pos, b.robot_pos)


def test_scale_invariance_of_layout():
    big = generate_scene(seed=0, num_agents=7, arena=60.0)
    small = generate_scene(seed=0, num_agents=7, arena=3.0)
    # Positions are geometrically similar: normalising by arena makes them equal.
    kb = big.resolved_layout.k
    ks = small.resolved_layout.k
    assert kb == pytest.approx(1.0)
    assert ks == pytest.approx(0.05)
    # A layout quantity scales linearly with arena.
    assert small.resolved_layout.radii["human"]["D_S"] == pytest.approx(
        big.resolved_layout.radii["human"]["D_S"] * ks
    )
    nb = np.array([n.pos for n in big.nodes]) / big.arena_size
    ns = np.array([n.pos for n in small.nodes]) / small.arena_size
    assert np.allclose(nb, ns, atol=1e-6)


def test_radius_floor():
    scene = generate_scene(seed=0, num_agents=5, arena=3.0, radius_floor=0.22)
    for agent in scene.agents:
        assert agent.D_S >= 0.22
        assert agent.D_C >= 0.22


def test_walkable_area_spans_the_site():
    """The network should reach well beyond the central mesh."""
    scene = generate_scene(seed=0, num_agents=7)
    layout = scene.resolved_layout
    # Corridors are 9 m wide at the reference site, so a probe of one corridor
    # width measures "reachable ground", not just the centrelines.
    frac = coverage_fraction(scene, radius=layout.shared_highway_width)
    assert frac > 0.70, f"only {frac:.2f} of the arena is near the network"
    # And the nodes themselves must reach the outer part of the site.
    nodes = np.array([n.pos for n in scene.nodes])
    assert nodes.max() > 0.8 * scene.arena_size
    assert nodes.min() < 0.2 * scene.arena_size


@pytest.mark.parametrize("num_agents", [5, 12, 19])
def test_agent_count_range_and_stations(num_agents):
    scene = generate_scene(seed=0, num_agents=num_agents)
    assert len(scene.agents) == num_agents
    assert C.MIN_AGENTS <= num_agents <= C.MAX_AGENTS
    # Every agent gets its own start station; the robot occupies a human one.
    starts = [a.station_a for a in scene.agents]
    assert len(set(starts)) == len(starts)
    assert scene.n_human_stations >= scene.num_humans + 1
    assert scene.n_vehicle_stations >= scene.num_vehicles


def test_spawn_never_starts_in_collision():
    """Even at the density ceiling, spawns must be outside each other's D_S."""
    for seed in range(4):
        scene = generate_scene(seed=seed, num_agents=C.MAX_AGENTS)
        agents = scene.agents
        for i, a in enumerate(agents):
            assert np.linalg.norm(a.pos - scene.robot_pos) > a.D_S
            for b in agents[i + 1:]:
                assert np.linalg.norm(a.pos - b.pos) > max(a.D_S, b.D_S)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        generate_scene(n_junctions=1)
    with pytest.raises(ValueError):
        generate_scene(n_human_stations=1)
    with pytest.raises(ValueError):
        generate_scene(num_agents=0)
    with pytest.raises(ValueError):
        generate_scene(num_agents=C.MIN_AGENTS - 1)
    with pytest.raises(ValueError):
        generate_scene(num_agents=C.MAX_AGENTS + 1)
    with pytest.raises(ValueError):
        generate_scene(num_agents=19, n_human_stations=5)
