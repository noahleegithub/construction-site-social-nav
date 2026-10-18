"""Graph construction and polyline geometry helpers.

Contains the worksite-graph machinery (mesh spanning tree, access-aware
adjacency, shortest paths) and the arc-length / point-on-polyline utilities used
for route following. Ported from ``visualize_scene.py``.
"""

from __future__ import annotations

import heapq

import numpy as np

from construction_nav import config as C
from construction_nav.entities import Highway, Scene, Zone


# ---------------------------------------------------------------------------
# Placement sampling
# ---------------------------------------------------------------------------


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def unit_from_angle(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)])


def sample_in_disk(rng: np.random.Generator, center: np.ndarray, radius: float) -> np.ndarray:
    r = radius * np.sqrt(rng.random())
    return center + r * unit_from_angle(rng.uniform(0.0, 2.0 * np.pi))


def sample_in_annulus(
    rng: np.random.Generator, center: np.ndarray, r_min: float, r_max: float
) -> np.ndarray:
    r = rng.uniform(r_min, r_max)
    return center + r * unit_from_angle(rng.uniform(0.0, 2.0 * np.pi))


def place(
    rng: np.random.Generator,
    sampler,
    existing: list[np.ndarray],
    min_sep: float,
) -> np.ndarray:
    """Rejection-sample a point from ``sampler`` at least ``min_sep`` from others."""
    dist = min_sep
    for _ in range(3):
        for _ in range(C.MAX_TRIES):
            pos = sampler()
            if all(np.linalg.norm(pos - e) >= dist for e in existing):
                return pos
        dist *= 0.7
    return sampler()


def place_station(
    rng: np.random.Generator,
    sampler,
    junctions: list[np.ndarray],
    same_kind: list[np.ndarray],
    cross_kind: list[np.ndarray],
    same_sep: float,
    cross_sep: float,
    node_clearance: float = 1.0,
) -> np.ndarray:
    """Place a workstation: same-kind separation plus a wider cross-kind buffer."""
    for _ in range(4):
        for _ in range(C.MAX_TRIES):
            pos = sampler()
            if (
                all(np.linalg.norm(pos - j) >= node_clearance for j in junctions)
                and all(np.linalg.norm(pos - s) >= same_sep for s in same_kind)
                and all(np.linalg.norm(pos - c) >= cross_sep for c in cross_kind)
            ):
                return pos
        same_sep *= 0.7
        cross_sep *= 0.7
        node_clearance *= 0.7
    return sampler()


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def build_mesh(
    junction_idx: list[int],
    positions: np.ndarray,
    n_extra: int,
    width: float = C.SHARED_HIGHWAY_WIDTH_REF,
) -> list[Highway]:
    """Spanning tree plus extra loop edges; every mesh corridor is shared."""
    pairs = [
        (float(np.linalg.norm(positions[a] - positions[b])), a, b)
        for i, a in enumerate(junction_idx)
        for b in junction_idx[i + 1 :]
    ]
    pairs.sort(key=lambda p: p[0])

    dsu = _DSU(len(junction_idx))
    local = {node: i for i, node in enumerate(junction_idx)}
    tree: set[tuple[int, int]] = set()
    for _, a, b in pairs:
        if dsu.union(local[a], local[b]):
            tree.add((min(a, b), max(a, b)))

    highways = [
        Highway(u=a, v=b, access=C.SHARED, width=width)
        for a, b in sorted(tree)
    ]

    extras = [p for p in pairs if (min(p[1], p[2]), max(p[1], p[2])) not in tree]
    for _, a, b in extras[:n_extra]:
        highways.append(Highway(u=a, v=b, access=C.SHARED, width=width))
    return highways


# ---------------------------------------------------------------------------
# Access-aware routing
# ---------------------------------------------------------------------------


def adjacency(scene: Scene, kind: str) -> dict[int, list[tuple[int, float]]]:
    pos = scene.positions()
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(scene.nodes))}
    for h in scene.highways:
        if kind in h.access:
            w = float(np.linalg.norm(pos[h.u] - pos[h.v]))
            adj[h.u].append((h.v, w))
            adj[h.v].append((h.u, w))
    return adj


def shortest_path(
    adj: dict[int, list[tuple[int, float]]], src: int, dst: int
) -> list[int] | None:
    dist = {src: 0.0}
    prev: dict[int, int] = {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, np.inf):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def route_points(scene: Scene, route: list[int]) -> np.ndarray:
    return np.array([scene.nodes[i].pos for i in route])


# ---------------------------------------------------------------------------
# Polyline geometry
# ---------------------------------------------------------------------------


def arc_lengths(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg = np.diff(points, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    return cum, lengths


def point_on_polyline(points: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Position and unit heading at arc length ``s`` along a polyline."""
    cum, lengths = arc_lengths(points)
    total = cum[-1]
    s = float(np.clip(s, 0.0, total))
    idx = int(np.searchsorted(cum, s, side="right") - 1)
    idx = max(0, min(idx, len(lengths) - 1))
    if lengths[idx] < 1e-9:
        return points[idx], np.array([1.0, 0.0])
    t = (s - cum[idx]) / lengths[idx]
    heading = (points[idx + 1] - points[idx]) / lengths[idx]
    return points[idx] + t * (points[idx + 1] - points[idx]), heading


def project_arc_length(points: np.ndarray, pos: np.ndarray) -> float:
    """Arc length along a polyline closest to ``pos`` (perpendicular projection)."""
    cum, lengths = arc_lengths(points)
    best_s, best_d = 0.0, float("inf")
    for i, seg in enumerate(np.diff(points, axis=0)):
        length = lengths[i]
        if length < 1e-9:
            continue
        t = float(np.clip(np.dot(pos - points[i], seg) / (length * length), 0.0, 1.0))
        projection = points[i] + t * seg
        distance = float(np.linalg.norm(pos - projection))
        if distance < best_d:
            best_d = distance
            best_s = float(cum[i] + t * length)
    return best_s


def route_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def point_in_zone(pos: np.ndarray, zone: Zone, margin: float = 0.0) -> bool:
    return float(np.linalg.norm(pos - zone.center)) <= zone.radius + margin
