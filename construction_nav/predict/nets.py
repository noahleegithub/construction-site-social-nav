"""Small ``torch`` building blocks shared by the learned predictors.

Everything here is deliberately tiny so the whole project trains and evaluates on
a CPU in seconds-to-minutes, and is *scale-free*: inputs are arena-normalised
positions (see :mod:`construction_nav.data.generate`), so one model covers both
the 60 m reference site and the miniature Robotarium site.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EntityEncoder(nn.Module):
    """Encodes per-entity history into a ``d_model`` embedding.

    A GRU over the ``(T, 2)`` past positions, plus learned embeddings for the
    entity type and activity state, so the model can tell humans, vehicles and
    the robot apart (Section 3's heterogeneity).
    """

    def __init__(self, d_model: int = 64, n_kinds: int = 3, n_states: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRU(2, d_model, batch_first=True)
        self.kind = nn.Embedding(n_kinds, d_model)
        self.state = nn.Embedding(n_states, d_model)
        # Explicit last-position + velocity features: these are exactly what the
        # constant-velocity baseline uses, so giving them to the encoder lets the
        # learned models match it instead of having to rediscover finite
        # differences from a small GRU.
        self.motion = nn.Linear(4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hist: torch.Tensor, kind: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, n, t, _ = hist.shape
        h, _ = self.gru(hist.reshape(b * n, t, 2))
        h = h[:, -1].reshape(b, n, self.d_model)
        vel = hist[:, :, -1, :] - hist[:, :, -2, :]
        motion = torch.cat([hist[:, :, -1, :], vel], dim=-1)
        h = h + self.kind(kind) + self.state(state) + self.motion(motion)
        return self.norm(h)


class JointMixer(nn.Module):
    """Self-attention across the entity tokens -> a *joint* (multimodal) model."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 1) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TimestepEmbedding(nn.Module):
    """Embeds a normalised diffusion timestep in ``[0, 1]`` into ``d_model``."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t.view(-1, 1).float())
