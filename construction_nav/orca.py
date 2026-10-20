"""ORCA: optimal reciprocal collision avoidance (pure numpy).

Implements the ORCA construction of van den Berg et al. (2011) -- the same
half-plane formulation used by RVO2 -- and a small exact 2-D linear program.

Each constraint is a half-plane ``n . v >= c`` on the *own* velocity ``v``. A
pair of agents builds one constraint each, so when both honour theirs the pair
is guaranteed collision-free for ``time_horizon`` seconds. Splitting the
recovery vector ``u`` in half is what makes the avoidance *reciprocal* rather
than one-sided.

The solver minimises ``||v - v_pref||`` over the intersection of the
half-planes and the speed disk, by enumerating the vertices of that feasible
polygon. With at most ~20 neighbours there are a few hundred candidates per
agent per step, so an exact enumeration is cheaper than an iterative solver and
cannot get stuck.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _det(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def orca_halfplane(
    pos: np.ndarray,
    vel: np.ndarray,
    other_pos: np.ndarray,
    other_vel: np.ndarray,
    combined_radius: float,
    time_horizon: float,
    time_step: float,
    share: float = 0.5,
    tie_break: float = 1.0,
) -> tuple[np.ndarray, float] | None:
    """ORCA half-plane ``n . v >= c`` for ``pos``/``vel`` against one neighbour.

    ``combined_radius`` is the closest centre-to-centre distance the pair should
    keep. ``share`` is this agent's fraction of the recovery vector (0.5 for
    reciprocal avoidance, 1.0 for full responsibility). ``tie_break`` only
    matters for exactly collinear head-on geometry, where it deterministically
    selects the passing side.

    Returns ``None`` when the constraint is inactive (zero-width recovery).
    """
    rel_pos = other_pos - pos
    rel_vel = vel - other_vel
    dist_sq = float(rel_pos @ rel_pos)
    radius_sq = combined_radius * combined_radius
    inv_tau = 1.0 / time_horizon

    if dist_sq > radius_sq:
        # No overlap: project onto the (truncated) velocity obstacle.
        w = rel_vel - inv_tau * rel_pos
        w_len_sq = float(w @ w)
        dot = float(w @ rel_pos)
        if dot < 0.0 and dot * dot > radius_sq * w_len_sq:
            # Nearest boundary is the cut-off circle.
            w_len = np.sqrt(w_len_sq)
            if w_len < _EPS:
                return None
            unit_w = w / w_len
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (combined_radius * inv_tau - w_len) * unit_w
        else:
            # Nearest boundary is one of the two cone legs.
            leg = np.sqrt(max(dist_sq - radius_sq, 0.0))
            side = _det(rel_pos, w)
            if abs(side) < _EPS:
                side = tie_break
            if side > 0.0:
                direction = np.array([
                    rel_pos[0] * leg - rel_pos[1] * combined_radius,
                    rel_pos[0] * combined_radius + rel_pos[1] * leg,
                ]) / dist_sq
            else:
                direction = -np.array([
                    rel_pos[0] * leg + rel_pos[1] * combined_radius,
                    -rel_pos[0] * combined_radius + rel_pos[1] * leg,
                ]) / dist_sq
            u = float(rel_vel @ direction) * direction - rel_vel
    else:
        # Already overlapping: resolve at the simulation timescale.
        inv_dt = 1.0 / time_step
        w = rel_vel - inv_dt * rel_pos
        w_len = float(np.linalg.norm(w))
        if w_len < _EPS:
            return None
        unit_w = w / w_len
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (combined_radius * inv_dt - w_len) * unit_w

    n = np.array([-direction[1], direction[0]])
    point = vel + share * u
    return n, float(n @ point)


def solve(
    preferred: np.ndarray,
    constraints: list[tuple[np.ndarray, float]],
    max_speed: float,
) -> np.ndarray:
    """Velocity closest to ``preferred`` satisfying all half-planes and the speed disk."""
    if not constraints:
        return _clip_speed(preferred, max_speed)

    def feasible(v: np.ndarray) -> bool:
        if float(v @ v) > max_speed * max_speed + 1e-9:
            return False
        return all(float(n @ v) - c >= -1e-9 for n, c in constraints)

    if feasible(preferred):
        return preferred

    candidates: list[np.ndarray] = []

    # Projection of the preferred velocity onto each constraint boundary.
    for n, c in constraints:
        candidates.append(preferred + (c - float(n @ preferred)) * n)

    # Intersections of constraint pairs.
    for i, (n1, c1) in enumerate(constraints):
        for n2, c2 in constraints[i + 1:]:
            denom = _det(n1, n2)
            if abs(denom) < 1e-9:
                continue
            candidates.append(
                np.array([(c1 * n2[1] - c2 * n1[1]) / denom,
                          (n1[0] * c2 - n2[0] * c1) / denom])
            )

    # Constraint boundary meeting the speed circle (turning at full speed).
    for n, c in constraints:
        if abs(c) > max_speed:
            continue
        base = c * n
        perp = np.array([n[1], -n[0]]) * np.sqrt(max(max_speed**2 - c**2, 0.0))
        candidates.append(base + perp)
        candidates.append(base - perp)

    best: np.ndarray | None = None
    best_cost = np.inf
    for v in candidates:
        if not feasible(v):
            continue
        cost = float(np.sum((v - preferred) ** 2))
        if cost < best_cost:
            best_cost, best = cost, v
    if best is not None:
        return best

    # Infeasible (e.g. boxed in on all sides): keep the least-violating point so
    # the agent still makes progress instead of deadlocking.
    best_violation = np.inf
    fallback = np.zeros(2)
    for v in candidates:
        violation = sum(max(0.0, c - float(n @ v)) for n, c in constraints)
        violation += max(0.0, float(np.linalg.norm(v)) - max_speed)
        if violation < best_violation:
            best_violation, fallback = violation, v
    return _clip_speed(fallback, max_speed)


def _clip_speed(v: np.ndarray, max_speed: float) -> np.ndarray:
    speed = float(np.linalg.norm(v))
    if speed <= max_speed or speed < _EPS:
        return v
    return v * (max_speed / speed)


def combined_radius(D_S_self: float, D_S_other: float) -> float:
    """Centre distance the pair should keep.

    ``D_S`` is a minimum centre distance, so the conservative choice for the
    pair is the larger of the two: a small agent next to a large one keeps the
    large one's clearance. For human--robot pairs both are equal to Eq. (2)'s
    ``D_S``; for vehicles the pair keeps the vehicle's larger radius.
    """
    return max(D_S_self, D_S_other)
