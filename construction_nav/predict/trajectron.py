"""Reduced Trajectron++ baseline (design Section 5).

An in-repo, CPU-sized re-implementation of the essential Trajectron++ ingredients:
a recurrent history encoder, a **discrete conditional VAE** (a categorical latent
modelling the interaction modes), and a decoder that emits a future trajectory per
mode. It is *joint* over the ``N = M + 1`` entities (agents + robot), like the
diffusion model, so a single latent explains a whole scene's joint future.

It is deliberately much smaller than the published Trajectron++ (no graph
attention, no depthwise encoders): the point is a like-for-like multimodal
baseline against which the diffusion predictor is compared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from construction_nav.data import Dataset
from construction_nav.data.dataset import make_loader
from construction_nav.env import Observation
from construction_nav.predict.base import Predictor, PredictorOutput
from construction_nav.predict.encode import obs_tensors
from construction_nav.predict.nets import EntityEncoder, JointMixer


class TrajectronModel(torch.nn.Module):
    """Joint encoder + discrete-CVAE latent + per-entity decoder."""

    def __init__(
        self,
        horizon: int,
        n_modes: int = 6,
        n_kinds: int = 3,
        n_states: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_modes = n_modes
        self.encoder = EntityEncoder(d_model, n_kinds, n_states)
        self.mixer = JointMixer(d_model, n_heads, n_layers)

        self.future_enc = torch.nn.Sequential(
            torch.nn.Linear(horizon * 2, d_model), torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.post_head = torch.nn.Linear(2 * d_model, n_modes)      # q(z | history, future)
        self.prior_head = torch.nn.Sequential(                         # p(z | history)
            torch.nn.Linear(d_model, d_model), torch.nn.SiLU(),
            torch.nn.Linear(d_model, n_modes),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(d_model + n_modes, 2 * d_model), torch.nn.SiLU(),
            torch.nn.Linear(2 * d_model, horizon * 2),
        )

    def context(self, hist: torch.Tensor, kind: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.encoder(hist, kind, state))          # (B, N, d)

    def posterior_logits(self, ctx: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        b, n, h, _ = future.shape
        fe = self.future_enc(future.reshape(b, n, h * 2))
        return self.post_head(torch.cat([ctx, fe], dim=-1))          # (B, N, K)

    def prior_logits(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.prior_head(ctx)                                  # (B, N, K)

    def decode(self, ctx: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode futures; ``z`` may be ``(B,N,K)`` or ``(B,N,...,K)``.

        A single categorical sample ``(B, N, K)`` yields ``(B, N, H, 2)``; the
        all-modes one-hot ``(B, N, K, K)`` yields ``(B, N, K, H, 2)``.
        """
        b, n, d = ctx.shape
        extra = z.shape[2:-1]
        ctx = ctx.view(b, n, *([1] * len(extra)), d).expand(b, n, *extra, d)
        out = self.decoder(torch.cat([ctx, z], dim=-1))
        return out.reshape(b, n, *extra, self.horizon, 2)

    @torch.no_grad()
    def decode_all_modes(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode every mode; returns ``(samples (B,N,K,H,2), probs (B,N,K))``."""
        b, n, d = ctx.shape
        k = self.n_modes
        z = torch.eye(k, device=ctx.device).view(1, 1, k, k).expand(b, n, k, k)
        samples = self.decode(ctx, z)
        probs = F.softmax(self.prior_logits(ctx), dim=-1)
        return samples, probs


def _batch(batch: dict, device: str) -> tuple[torch.Tensor, ...]:
    return (
        batch["hist"].to(device),
        batch["kind"].to(device),
        batch["state"].to(device),
        batch["future"].to(device),
    )


def train_trajectron(
    data: Dataset,
    val_data: Dataset | None = None,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 2e-3,
    n_modes: int = 6,
    beta: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
    log=print,
) -> tuple[TrajectronModel, dict]:
    """Train with reconstruction + categorical-KL (discrete CVAE) objectives."""
    torch.manual_seed(seed)
    model = TrajectronModel(horizon=data.horizon, n_modes=n_modes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = make_loader(data, batch_size=batch_size, shuffle=True, seed=seed)

    history: dict[str, list[float]] = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            hist, kind, state, future = _batch(batch, device)
            delta = future - hist[:, :, -1:, :]
            ctx = model.context(hist, kind, state)
            post_logits = model.posterior_logits(ctx, delta)
            prior_logits = model.prior_logits(ctx)

            # Straight-through categorical sample from the posterior.
            z = F.gumbel_softmax(post_logits, tau=1.0, hard=True)
            pred = model.decode(ctx, z)
            recon = F.mse_loss(pred, delta)

            log_post = F.log_softmax(post_logits, dim=-1)
            log_prior = F.log_softmax(prior_logits, dim=-1)
            post = log_post.exp()
            kl = (post * (log_post - log_prior)).sum(dim=-1).mean()

            loss = recon + beta * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * hist.shape[0]
            count += hist.shape[0]
        history["train"].append(total / max(count, 1))
        if val_data is not None:
            history["val"].append(eval_trajectron_loss(model, val_data, beta, device))
        if epoch == 1 or epoch % max(epochs // 8, 1) == 0 or epoch == epochs:
            msg = f"epoch {epoch:3d}  train {history['train'][-1]:.4f}"
            if history["val"]:
                msg += f"  val {history['val'][-1]:.4f}"
            log(msg)
    return model, history


@torch.no_grad()
def eval_trajectron_loss(
    model: TrajectronModel,
    data: Dataset,
    beta: float = 0.05,
    device: str = "cpu",
    batch_size: int = 128,
) -> float:
    model.eval()
    loader = make_loader(data, batch_size=batch_size, shuffle=False)
    total, count = 0.0, 0
    for batch in loader:
        hist, kind, state, future = _batch(batch, device)
        delta = future - hist[:, :, -1:, :]
        ctx = model.context(hist, kind, state)
        post_logits = model.posterior_logits(ctx, delta)
        prior_logits = model.prior_logits(ctx)
        z = F.gumbel_softmax(post_logits, tau=1.0, hard=True)
        recon = F.mse_loss(model.decode(ctx, z), delta)
        log_post = F.log_softmax(post_logits, dim=-1)
        log_prior = F.log_softmax(prior_logits, dim=-1)
        kl = (log_post.exp() * (log_post - log_prior)).sum(dim=-1).mean()
        total += float(recon + beta * kl) * hist.shape[0]
        count += hist.shape[0]
    return total / max(count, 1)


class TrajectronPredictor(Predictor):
    """Multimodal predictor backed by a trained :class:`TrajectronModel`."""

    multimodal = True

    def __init__(self, model: TrajectronModel, device: str = "cpu") -> None:
        super().__init__(horizon=model.horizon)
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, obs: Observation) -> PredictorOutput:
        m = len(obs.agent_kind)
        hist, kind, state = obs_tensors(obs, device=self.device)
        with torch.no_grad():
            ctx = self.model.context(hist, kind, state)
            samples, probs = self.model.decode_all_modes(ctx)   # (1,N,K,H,2), (1,N,K)

        # Decoded futures are displacements from the last observed position.
        pos0 = hist[:, :, -1, :]                                # (1, N, 2)
        samples = samples + pos0[:, :, None, None, :]

        arena = float(obs.arena)
        samples = samples[0].cpu().numpy() * arena              # (N, K, H, 2)
        probs = probs[0].cpu().numpy()                          # (N, K)

        agent_samples = samples[:m]                             # (M, K, H, 2)
        agent_weights = probs[:m]
        agent_weights = agent_weights / np.maximum(agent_weights.sum(axis=1, keepdims=True), 1e-9)
        mean = (agent_samples * agent_weights[:, :, None, None]).sum(axis=1)
        return PredictorOutput(mean=mean, samples=agent_samples, weights=agent_weights)

    @property
    def name(self) -> str:
        return "trajectron++ (reduced)"


def save_predictor(path: str | Path, predictor: TrajectronPredictor) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": predictor.model.state_dict(),
            "horizon": predictor.model.horizon,
            "n_modes": predictor.model.n_modes,
        },
        path,
    )
    return path


def load_predictor(path: str | Path, device: str = "cpu") -> TrajectronPredictor:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model = TrajectronModel(horizon=int(ckpt["horizon"]), n_modes=int(ckpt["n_modes"]))
    model.load_state_dict(ckpt["state_dict"])
    return TrajectronPredictor(model, device=device)
