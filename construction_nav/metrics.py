"""Evaluation metrics: ADE / FDE (Eq. (1)), barrier terms (Eq. (2)), cost J.

All functions take plain ``numpy`` arrays so they can score any predictor or
planner without depending on the model implementations.

Conventions
-----------
* A prediction is ``(..., H, 2)`` positions (meters); ground truth matches.
* A multi-modal prediction adds a leading mode axis: ``(K, M, H, 2)`` or
  ``(M, K, H, 2)`` depending on the predictor; helpers accept either the last
  two axes as ``(H, 2)``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Displacement errors  (Eq. (1))
# ---------------------------------------------------------------------------


def ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average displacement error over all agents and forecast steps."""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final displacement error averaged over agents."""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    return float(np.mean(np.linalg.norm(pred[..., -1, :] - gt[..., -1, :], axis=-1)))


def min_ade(pred_samples: np.ndarray, gt: np.ndarray) -> float:
    """Best-of-K ADE: minimum over modes, averaged over agents and steps.

    ``pred_samples`` is ``(M, K, H, 2)``; ``gt`` is ``(M, H, 2)``.
    """
    pred_samples = np.asarray(pred_samples, dtype=float)
    gt = np.asarray(gt, dtype=float)
    per_mode = np.linalg.norm(pred_samples - gt[:, None, :, :], axis=-1)  # (M,K,H)
    per_agent_step = per_mode.min(axis=1)                                # (M,H)
    return float(np.mean(per_agent_step))


def min_fde(pred_samples: np.ndarray, gt: np.ndarray) -> float:
    """Best-of-K final displacement error.

    ``pred_samples`` is ``(M, K, H, 2)``; ``gt`` is ``(M, H, 2)``.
    """
    pred_samples = np.asarray(pred_samples, dtype=float)
    gt = np.asarray(gt, dtype=float)
    per_mode = np.linalg.norm(
        pred_samples[..., -1, :] - gt[:, None, -1, :], axis=-1
    )  # (M,K)
    return float(np.mean(per_mode.min(axis=1)))


def displacement_curve(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-step mean displacement error, shape ``(H,)`` (for error-vs-horizon plots)."""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=0)


# ---------------------------------------------------------------------------
# Barrier terms  (Eq. (2))
# ---------------------------------------------------------------------------


def h_S(p: np.ndarray, q: np.ndarray, D_S: float) -> float:
    """Hard collision term: ``||p - q||^2 - D_S^2`` (>= 0 is collision-free)."""
    return float(np.sum((np.asarray(p) - np.asarray(q)) ** 2) - D_S**2)


def h_C(p: np.ndarray, q: np.ndarray, D_C: float) -> float:
    """Safety-radius term: ``||p - q||^2 - D_C^2`` (>= 0 is non-intrusive)."""
    return float(np.sum((np.asarray(p) - np.asarray(q)) ** 2) - D_C**2)


def min_h_S(robot_pos: np.ndarray, agent_pos: np.ndarray, D_S: np.ndarray) -> float:
    """Minimum hard term between the robot and every agent."""
    agent_pos = np.asarray(agent_pos, dtype=float)
    D_S = np.asarray(D_S, dtype=float)
    terms = np.sum((agent_pos - np.asarray(robot_pos)) ** 2, axis=-1) - D_S**2
    return float(terms.min()) if terms.size else float("inf")


def min_h_C(robot_pos: np.ndarray, agent_pos: np.ndarray, D_C: np.ndarray) -> float:
    """Minimum safety term between the robot and every agent."""
    agent_pos = np.asarray(agent_pos, dtype=float)
    D_C = np.asarray(D_C, dtype=float)
    terms = np.sum((agent_pos - np.asarray(robot_pos)) ** 2, axis=-1) - D_C**2
    return float(terms.min()) if terms.size else float("inf")


def intrusion(robot_pos: np.ndarray, agent_pos: np.ndarray, D_C: np.ndarray) -> np.ndarray:
    """Per-agent ``max(0, -h_C)`` -- the Eq. (3) integrand."""
    agent_pos = np.asarray(agent_pos, dtype=float)
    D_C = np.asarray(D_C, dtype=float)
    h = np.sum((agent_pos - np.asarray(robot_pos)) ** 2, axis=-1) - D_C**2
    return np.maximum(0.0, -h)


# ---------------------------------------------------------------------------
# Navigation cost  (Eq. (3))
# ---------------------------------------------------------------------------


def navigation_cost(
    travel_time: float,
    intrusion_integral: float,
    freeze_time: float,
    lambda_intrusion: float = 1.0,
    lambda_freeze: float = 0.5,
) -> float:
    """``J = T_travel + lambda_1 * integral(max(0, -h_C)) + lambda_2 * T_freeze``."""
    return float(
        travel_time
        + lambda_intrusion * intrusion_integral
        + lambda_freeze * freeze_time
    )


def summarize_trial(
    env,
    lambda_intrusion: float = 1.0,
    lambda_freeze: float = 0.5,
) -> dict:
    """Collapse a finished :class:`~construction_nav.env.ConstructionEnv` run.

    Returns travel/intrusion/freeze components of Eq. (3) plus collision stats.
    """
    J = navigation_cost(
        env.travel_time,
        env.intrusion_integral,
        env.freeze_time,
        lambda_intrusion=lambda_intrusion,
        lambda_freeze=lambda_freeze,
    )
    return {
        "travel_time": env.travel_time,
        "intrusion_integral": env.intrusion_integral,
        "freeze_time": env.freeze_time,
        "n_collision_steps": env.n_collision_steps,
        "collided": env.n_collision_steps > 0,
        "J": J,
    }
