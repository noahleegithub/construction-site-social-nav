"""MotionDiffuser-style conditional diffusion predictor (design Section 4).

A *joint*, natively-multimodal model: one denoiser over all ``N = M + 1`` entities
(the agents **and** the robot), conditioned on their observed histories. Self
attention across the entity tokens lets the agents react to one another instead
of forecasting in isolation.

This is a deliberately small stand-in for the full MotionDiffuser: a GRU history
encoder (plus explicit last-position and velocity features), one transformer block
over the entity tokens -- that self-attention is what makes the forecast *joint* --
and a DDIM sampler with ``K`` samples per observation. Everything is CPU-trainable
and arena-normalised, so one model covers the 60 m reference site and the miniature
Robotarium site.

Three choices matter on a CPU budget and are documented where they are implemented:
the network predicts ``x0`` directly (stable for small models / few steps), uses a
cosine noise schedule (so the last step really is pure noise), and diffs the
*residual over a constant-velocity prior* (so it starts from the CV baseline's
accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from construction_nav.data import Dataset
from construction_nav.data.dataset import make_loader
from construction_nav.env import Observation
from construction_nav.predict.base import Predictor, PredictorOutput
from construction_nav.predict.encode import obs_tensors
from construction_nav.predict.nets import EntityEncoder, JointMixer, TimestepEmbedding


# ---------------------------------------------------------------------------
# Denoiser + diffusion schedule
# ---------------------------------------------------------------------------


class JointDenoiser(torch.nn.Module):
    """Predicts the (residual) joint future ``(N, H, 2)`` from noisy input."""

    def __init__(
        self,
        horizon: int,
        n_kinds: int = 3,
        n_states: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.encoder = EntityEncoder(d_model, n_kinds, n_states)
        self.future_proj = torch.nn.Linear(horizon * 2, d_model)
        self.step_emb = TimestepEmbedding(d_model)
        self.mixer = JointMixer(d_model, n_heads, n_layers)
        self.out = torch.nn.Linear(d_model, horizon * 2)

    def forward(
        self,
        noisy: torch.Tensor,      # (B, N, 2H) displacement-to-predict
        hist: torch.Tensor,       # (B, N, T, 2)
        kind: torch.Tensor,       # (B, N)
        state: torch.Tensor,      # (B, N)
        t: torch.Tensor,          # (B,) normalised to [0, 1]
    ) -> torch.Tensor:
        # The model predicts the *displacement* to each future step relative to
        # the last observed position (the anchor is added back after sampling);
        # the encoder carries the position + velocity features.
        cond = self.encoder(hist, kind, state)
        x = self.future_proj(noisy) + cond + self.step_emb(t)[:, None, :]
        x = self.mixer(x)
        return self.out(x)


@dataclass
class DiffusionSchedule:
    """Cosine noise schedule (Nichol & Dhariwal) plus the constant-velocity prior.

    A cosine ``alpha_bar`` reaches ~0 at the final step, so DDIM sampling can
    start from standard normal noise -- a linear-beta schedule over only ~50
    steps does *not* fully destroy the signal, which leaves sampling outside the
    training distribution and the model unable to denoise.

    The model diffuses the *residual over a constant-velocity prior* (rather than
    the raw displacement). The prior is exactly the CV baseline, so the model
    never does worse than CV at short horizons and spends its capacity modelling
    the interaction-driven corrections that make the forecast multimodal.
    """

    n_steps: int = 100
    dt: float = 0.5
    s: float = 0.008

    def __post_init__(self) -> None:
        import math

        steps = torch.arange(self.n_steps + 1, dtype=torch.float64) / self.n_steps
        f = torch.cos((steps + self.s) / (1.0 + self.s) * (math.pi / 2.0)) ** 2
        alpha_bar = (f / f[0]).clamp(1e-4, 0.9999)[1:]
        self.alpha_bar = alpha_bar.float()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def cv_prior(hist: torch.Tensor, horizon: int, dt: float) -> torch.Tensor:
    """Constant-velocity extrapolation of the last observed frame; ``(B, N, H, 2)``."""
    vel = (hist[:, :, -1, :] - hist[:, :, -2, :]) / dt
    steps = torch.arange(1, horizon + 1, device=hist.device).float() * dt
    return vel[:, :, None, :] * steps[None, None, :, None]


def residual_target(future: torch.Tensor, hist: torch.Tensor, dt: float) -> torch.Tensor:
    """Target = future minus the constant-velocity prior and the anchor frame."""
    horizon = future.shape[2]
    pos0 = hist[:, :, -1, :]
    return future - pos0[:, :, None, :] - cv_prior(hist, horizon, dt)


def _flatten_future(x: torch.Tensor) -> torch.Tensor:
    b, n, h, _ = x.shape
    return x.reshape(b, n, h * 2)


def project_robot_safety(
    robot: torch.Tensor,
    agents: torch.Tensor,
    ds_agents: torch.Tensor,
    margin: float = 0.0,
    iters: int = 2,
) -> torch.Tensor:
    """Push the robot rows out of every agent's collision disk.

    ``robot`` is ``(K, H, 2)``, ``agents`` is ``(K, M, H, 2)`` and ``ds_agents``
    is ``(M,)``. This is the projection onto the *hard* constraint set used as
    classifier-style guidance inside reverse diffusion.
    """
    r = robot.clone()
    need = (ds_agents + margin).view(1, -1, 1)          # (1, M, 1)
    for _ in range(iters):
        delta = r[:, None, :, :] - agents               # (K, M, H, 2)
        dist = delta.norm(dim=-1, keepdim=True)         # (K, M, H, 1)
        deficit = (need.unsqueeze(-1) - dist).clamp_min(0.0)
        if float(deficit.max()) == 0.0:
            break
        r = r + (delta / dist.clamp_min(1e-6) * deficit).sum(dim=1)
    return r


@torch.no_grad()
def ddim_sample_conditioned(
    model: JointDenoiser,
    schedule: DiffusionSchedule,
    hist: torch.Tensor,
    kind: torch.Tensor,
    state: torch.Tensor,
    robot_plan: torch.Tensor,
    ds_agents: torch.Tensor,
    n_samples: int,
    n_steps: int = 10,
    safety_margin: float = 0.0,
    project_safety: bool = True,
) -> torch.Tensor:
    """Sample agent futures conditioned on a fixed robot plan (inpainting).

    The robot row of the joint sample is *pinned* to ``robot_plan`` (normalised
    ``(H, 2)``) at every reverse step, so the agent futures are generated
    conditioned on how the agents would react to that plan. With
    ``project_safety`` the pinned row is projected onto the hard-constraint safe
    set at every step -- the safety constraint is injected into reverse
    diffusion rather than checked only afterwards.

    ``hist`` is ``(1, N, T, 2)``; returns absolute ``(K, N, H, 2)``.
    """
    b, n = hist.shape[:2]
    assert b == 1, "conditioned sampling is defined for a single observation"
    horizon = model.horizon
    robot_idx = n - 1
    dt = schedule.dt
    pos0 = hist[:, :, -1, :]                             # (1, N, 2)
    cv = cv_prior(hist, horizon, dt)                     # (1, N, H, 2)
    plan = robot_plan.reshape(1, horizon, 2).expand(n_samples, horizon, 2)

    hist_k = hist.expand(n_samples, n, -1, -1)
    kind_k = kind.expand(n_samples, n)
    state_k = state.expand(n_samples, n)
    x = torch.randn(n_samples, n, horizon * 2, device=hist.device)

    steps = torch.linspace(schedule.n_steps - 1, 0, n_steps).long()
    for i, t in enumerate(steps):
        tb = torch.full((n_samples,), int(t), device=hist.device)
        ab = schedule.alpha_bar[int(t)]
        x0 = model(x, hist_k, kind_k, state_k, tb.float() / schedule.n_steps)
        x0 = x0.clamp(-2.0, 3.0).reshape(n_samples, n, horizon, 2)
        absolute = x0 + cv + pos0[:, :, None, :]         # (K, N, H, 2)

        # Pin the robot row to the plan and (optionally) project it to safety.
        robot_row = plan.clone()
        if project_safety:
            robot_row = project_robot_safety(
                robot_row, absolute[:, :robot_idx], ds_agents, safety_margin
            )
        absolute = absolute.clone()
        absolute[:, robot_idx] = robot_row
        x0 = (absolute - cv - pos0[:, :, None, :]).reshape(n_samples, n, horizon * 2)

        if i < len(steps) - 1:
            ab_prev = schedule.alpha_bar[int(steps[i + 1])]
            eps = (x - ab.sqrt() * x0) / (1.0 - ab).sqrt()
            x = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps
        else:
            x = x0

    return x.reshape(n_samples, n, horizon, 2) + cv + pos0[:, :, None, :]


@torch.no_grad()
def ddim_sample(
    model: JointDenoiser,
    schedule: DiffusionSchedule,
    hist: torch.Tensor,
    kind: torch.Tensor,
    state: torch.Tensor,
    n_samples: int,
    n_steps: int = 10,
) -> torch.Tensor:
    """Deterministic DDIM sampling; returns absolute ``(B, K, N, H, 2)``."""
    b, n = hist.shape[:2]
    horizon = model.horizon
    hist = hist.repeat_interleave(n_samples, dim=0)
    kind = kind.repeat_interleave(n_samples, dim=0)
    state = state.repeat_interleave(n_samples, dim=0)
    x = torch.randn(b * n_samples, n, horizon * 2, device=hist.device)

    steps = torch.linspace(schedule.n_steps - 1, 0, n_steps).long()
    for i, t in enumerate(steps):
        t_batch = torch.full((b * n_samples,), int(t), device=hist.device)
        ab = schedule.alpha_bar[int(t)]
        # The denoiser predicts x0 directly (stable for small models / few steps).
        x0 = model(x, hist, kind, state, t_batch.float() / schedule.n_steps).clamp(-2.0, 3.0)
        if i < len(steps) - 1:
            ab_prev = schedule.alpha_bar[int(steps[i + 1])]
            eps = (x - ab.sqrt() * x0) / (1.0 - ab).sqrt()
            x = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps
        else:
            x = x0

    # residual -> absolute: add the CV prior and the anchor position back.
    x = x.reshape(b * n_samples, n, horizon, 2)
    x = x + cv_prior(hist, horizon, schedule.dt) + hist[:, :, -1, :][:, :, None, :]
    return x.reshape(b, n_samples, n, horizon, 2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _batch(batch: dict, device: str) -> tuple[torch.Tensor, ...]:
    return (
        batch["hist"].to(device),
        batch["kind"].to(device),
        batch["state"].to(device),
        batch["future"].to(device),
    )


@torch.no_grad()
def eval_diffusion_loss(
    model: JointDenoiser,
    schedule: DiffusionSchedule,
    data: Dataset,
    batch_size: int = 128,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    model.eval()
    loader = make_loader(data, batch_size=batch_size, shuffle=False, seed=seed)
    total, count = 0.0, 0
    for batch in loader:
        hist, kind, state, future = _batch(batch, device)
        x0 = _flatten_future(residual_target(future, hist, schedule.dt))
        noise = torch.randn_like(x0)
        t = torch.randint(0, schedule.n_steps, (x0.shape[0],), device=x0.device)
        xt = schedule.q_sample(x0, t, noise)
        pred = model(xt, hist, kind, state, t.float() / schedule.n_steps)
        total += float(F.mse_loss(pred, x0)) * x0.shape[0]
        count += x0.shape[0]
    return total / max(count, 1)


def train_diffusion(
    data: Dataset,
    val_data: Dataset | None = None,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 2e-3,
    diffusion_steps: int = 100,
    seed: int = 0,
    device: str = "cpu",
    log=print,
) -> tuple[JointDenoiser, dict, DiffusionSchedule]:
    """Train the denoiser with the standard noise-prediction objective."""
    torch.manual_seed(seed)
    model = JointDenoiser(horizon=data.horizon).to(device)
    schedule = DiffusionSchedule(n_steps=diffusion_steps, dt=data.dt)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = make_loader(data, batch_size=batch_size, shuffle=True, seed=seed)

    history: dict[str, list[float]] = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            hist, kind, state, future = _batch(batch, device)
            x0 = _flatten_future(residual_target(future, hist, schedule.dt))
            noise = torch.randn_like(x0)
            t = torch.randint(0, diffusion_steps, (x0.shape[0],), device=x0.device)
            xt = schedule.q_sample(x0, t, noise)
            pred = model(xt, hist, kind, state, t.float() / diffusion_steps)
            loss = F.mse_loss(pred, x0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * x0.shape[0]
            count += x0.shape[0]
        history["train"].append(total / max(count, 1))
        if val_data is not None:
            history["val"].append(
                eval_diffusion_loss(model, schedule, val_data, device=device)
            )
        if epoch == 1 or epoch % max(epochs // 8, 1) == 0 or epoch == epochs:
            msg = f"epoch {epoch:3d}  train {history['train'][-1]:.4f}"
            if history["val"]:
                msg += f"  val {history['val'][-1]:.4f}"
            log(msg)
    return model, history, schedule


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class DiffusionPredictor(Predictor):
    """Wraps a trained :class:`JointDenoiser` as a multimodal predictor."""

    multimodal = True

    def __init__(
        self,
        model: JointDenoiser,
        diffusion_steps: int = 100,
        n_samples: int = 6,
        sample_steps: int = 20,
        dt: float = 0.5,
        device: str = "cpu",
    ) -> None:
        super().__init__(horizon=model.horizon, dt=dt)
        self.device = device
        self.model = model.to(device).eval()
        self.schedule = DiffusionSchedule(n_steps=diffusion_steps, dt=dt)
        self.n_samples = n_samples
        self.sample_steps = sample_steps

    # -- tensor preparation ------------------------------------------------

    def _tensors(self, obs: Observation) -> tuple[torch.Tensor, ...]:
        return obs_tensors(obs, device=self.device)

    # -- inference ---------------------------------------------------------

    def sample_joint(
        self,
        obs: Observation,
        n_samples: int | None = None,
        n_steps: int | None = None,
        robot_plan: np.ndarray | None = None,
        ds_agents: np.ndarray | None = None,
        safety_margin: float = 0.0,
    ) -> np.ndarray:
        """Joint forecast for **all** entities, including the robot.

        Returns absolute ``(K, N, H, 2)``. When ``robot_plan`` (site metres,
        ``(H, 2)``) is given the agents are forecast *conditioned* on it
        (inpainting); when ``ds_agents`` is also given, the robot row is projected
        onto the hard-constraint safe set at every reverse step, i.e. the safety
        constraint is injected into reverse diffusion.
        """
        n_samples = self.n_samples if n_samples is None else n_samples
        n_steps = self.sample_steps if n_steps is None else n_steps
        arena = float(obs.arena)
        hist, kind, state = self._tensors(obs)

        if robot_plan is None and ds_agents is None:
            samples = ddim_sample(
                self.model, self.schedule, hist, kind, state,
                n_samples=n_samples, n_steps=n_steps,
            )[0]
        else:
            if robot_plan is None or ds_agents is None:
                raise ValueError("conditioned sampling needs both robot_plan and ds_agents")
            plan = torch.as_tensor(
                np.asarray(robot_plan, dtype=np.float32) / arena, device=self.device
            )
            ds = torch.as_tensor(
                np.asarray(ds_agents, dtype=np.float32) / arena, device=self.device
            )
            samples = ddim_sample_conditioned(
                self.model, self.schedule, hist, kind, state, plan, ds,
                n_samples=n_samples, n_steps=n_steps,
                safety_margin=safety_margin / arena,
            )  # already (K, N, H, 2)
        return samples.cpu().numpy() * arena

    def predict(self, obs: Observation) -> PredictorOutput:
        m = len(obs.agent_kind)
        hist, kind, state = self._tensors(obs)
        samples = ddim_sample(
            self.model, self.schedule, hist, kind, state,
            n_samples=self.n_samples, n_steps=self.sample_steps,
        )[0]                                   # (K, N, H, 2) normalised
        samples = samples.cpu().numpy() * float(obs.arena)

        agent_samples = np.transpose(samples[:, :m], (1, 0, 2, 3))  # (M, K, H, 2)
        mean = agent_samples.mean(axis=1)
        weights = np.full((m, self.n_samples), 1.0 / self.n_samples)
        return PredictorOutput(mean=mean, samples=agent_samples, weights=weights)

    @property
    def name(self) -> str:
        return f"diffusion (K={self.n_samples})"


def save_predictor(path: str | Path, predictor: DiffusionPredictor) -> Path:
    """Serialise a trained diffusion predictor (weights + sampling config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": predictor.model.state_dict(),
            "horizon": predictor.model.horizon,
            "diffusion_steps": predictor.schedule.n_steps,
            "dt": predictor.schedule.dt,
            "n_samples": predictor.n_samples,
            "sample_steps": predictor.sample_steps,
        },
        path,
    )
    return path


def load_predictor(path: str | Path, device: str = "cpu") -> DiffusionPredictor:
    """Load a predictor written by :func:`save_predictor`."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model = JointDenoiser(horizon=int(ckpt["horizon"]))
    model.load_state_dict(ckpt["state_dict"])
    return DiffusionPredictor(
        model,
        diffusion_steps=int(ckpt["diffusion_steps"]),
        n_samples=int(ckpt["n_samples"]),
        sample_steps=int(ckpt["sample_steps"]),
        dt=float(ckpt.get("dt", 0.5)),
        device=device,
    )
