"""Scene generation invariants and scale invariance."""

from __future__ import annotations

import numpy as np
import pytest

from construction_nav import config as C
from construction_nav.env import generate_scene
from construction_nav.graph import adjacency, shortest_path


def test_scene_counts(scene):
    assert len(scene.station_indices("human")) == scene.n_human_stations
    assert len(scene.station_indices("vehicle")) == scene.n_vehicle_stations
    assert len(scene.junction_indices()) == scene.n_junctions
    assert len(scene.zones) == C.N_ZONES
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


def test_zones_inside_arena(scene):
    for z in scene.zones:
        assert z.radius > 0
        assert np.all(z.center - z.radius >= 0.0)
        assert np.all(z.center + z.radius <= scene.arena_size)


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


def test_invalid_arguments():
    with pytest.raises(ValueError):
        generate_scene(n_junctions=1)
    with pytest.raises(ValueError):
        generate_scene(n_human_stations=1)
    with pytest.raises(ValueError):
        generate_scene(num_agents=0)
