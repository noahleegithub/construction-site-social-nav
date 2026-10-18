"""Shared observation -> tensor conversion for the joint (all-entities) models.

The learned predictors see ``N = M + 1`` entities: the ``M`` agents followed by
the robot (matching :func:`construction_nav.env.observe`). Positions are divided
by the arena so the models are scale-free.
"""

from __future__ import annotations

import numpy as np
import torch

from construction_nav.data import KIND_TO_ID, ROBOT_KIND_ID, STATE_TO_ID
from construction_nav.env import Observation

ROBOT_STATE_ID = STATE_TO_ID["moving"]


def obs_tensors(
    obs: Observation, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(hist, kind, state)`` with a leading batch axis of size 1."""
    hist = np.concatenate([obs.histories, obs.robot_history[None]], axis=0)
    hist = hist / float(obs.arena)
    kind = np.array(
        [KIND_TO_ID[k] for k in obs.agent_kind] + [ROBOT_KIND_ID], dtype=np.int64
    )
    state = np.array(
        [STATE_TO_ID[s] for s in obs.agent_state] + [ROBOT_STATE_ID], dtype=np.int64
    )
    return (
        torch.as_tensor(hist, dtype=torch.float32, device=device)[None],
        torch.as_tensor(kind, device=device)[None],
        torch.as_tensor(state, device=device)[None],
    )
