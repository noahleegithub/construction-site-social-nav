"""Dataset generation, learned predictors and open-loop evaluation.

Every test that touches a ``torch`` model skips cleanly when the optional ML extra
is not installed, so the core suite still runs on a torch-free checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from construction_nav import config as C
from construction_nav.data import (
    KIND_TO_ID,
    ROBOT_KIND_ID,
    STATE_TO_ID,
    build_dataset,
    episode_windows,
    load_dataset,
    save_dataset,
)
from construction_nav.env import generate_scene
from construction_nav.evaluate import evaluate_predictor
from construction_nav.predict import ConstantVelocityPredictor


# ---------------------------------------------------------------------------
# Dataset generation (torch-free)
# ---------------------------------------------------------------------------


def test_dataset_shapes_and_normalisation():
    data = build_dataset(range(2), num_agents=5, steps=40)
    s, n, t, two = data.hist.shape
    assert two == 2
    assert n == 6                       # M agents + the robot
    assert t == C.HISTORY_STEPS
    assert data.future.shape[:3] == (s, n, C.HORIZON_STEPS)
    assert data.kind.shape == (s, n)
    assert data.state.shape == (s, n)
    # Normalised positions live inside the unit square.
    assert data.hist.min() >= 0.0 and data.hist.max() <= 1.0
    assert data.future.min() >= 0.0 and data.future.max() <= 1.0


def test_dataset_entity_layout_matches_observation():
    data = build_dataset(range(1), num_agents=5, steps=40)
    # The robot is the last entity and is labelled as such in every window.
    assert np.all(data.kind[:, -1] == ROBOT_KIND_ID)
    assert set(np.unique(data.kind)) <= set(KIND_TO_ID.values())
    assert set(np.unique(data.state)) <= set(STATE_TO_ID.values())


def test_dataset_round_trip(tmp_path):
    data = build_dataset(range(2), num_agents=5, steps=40)
    path = save_dataset(tmp_path / "d.npz", data)
    back = load_dataset(path)
    assert np.allclose(back.hist, data.hist)
    assert np.allclose(back.future, data.future)
    assert np.array_equal(back.kind, data.kind)
    assert back.arena == pytest.approx(data.arena)
    assert back.horizon == data.horizon


def test_episode_windows_history_is_continuous():
    scene = generate_scene(seed=0, num_agents=5, arena=C.ARENA_SIZE)
    hist, future, _kind, _state = episode_windows(scene, steps=40)
    # The final history frame is the last observation, and the first future frame
    # is the very next step.
    assert hist.shape[0] == future.shape[0]
    # History is time-ordered: consecutive frames are (usually) a small step apart.
    diffs = np.linalg.norm(np.diff(hist[0, 0], axis=0), axis=-1)
    assert np.all(diffs >= 0)


def test_scale_invariance_of_dataset():
    """The same seed at two arena sizes yields identical normalised windows."""
    big = build_dataset([0], num_agents=5, arena=60.0, steps=40)
    small = build_dataset([0], num_agents=5, arena=3.0, steps=40)
    assert big.hist.shape == small.hist.shape
    assert np.allclose(big.hist, small.hist, atol=1e-5)
    assert np.allclose(big.future, small.future, atol=1e-5)


# ---------------------------------------------------------------------------
# Evaluation harness with the always-available baseline
# ---------------------------------------------------------------------------


def test_evaluate_constant_velocity_returns_finite_scores():
    predictor = ConstantVelocityPredictor(horizon=C.HORIZON_STEPS)
    score = evaluate_predictor(
        predictor, seeds=range(2), horizons=(2, 8), num_agents=5, steps=40
    )
    assert set(score.ade) == {2, 8}
    assert all(np.isfinite(v) for v in score.ade.values())
    # Error grows with the horizon on average for the CV baseline.
    assert score.ade[8] > score.ade[2]


# ---------------------------------------------------------------------------
# Learned predictors (require torch)
# ---------------------------------------------------------------------------


torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tiny_data():
    return (
        build_dataset(range(3), num_agents=5, steps=40),
        build_dataset(range(3, 4), num_agents=5, steps=40),
    )


def test_diffusion_train_and_predict(tiny_data):
    from construction_nav.env import ConstructionEnv
    from construction_nav.predict.diffusion import DiffusionPredictor, train_diffusion

    train, val = tiny_data
    model, history, schedule = train_diffusion(
        train, val, epochs=2, batch_size=32, log=lambda *a: None
    )
    assert history["train"][-1] < history["train"][0]

    predictor = DiffusionPredictor(
        model, diffusion_steps=schedule.n_steps, n_samples=4, sample_steps=5
    )
    env = ConstructionEnv(generate_scene(seed=99, num_agents=5))
    out = predictor.predict(env.observe())
    m = 5
    assert out.mean.shape == (m, model.horizon, 2)
    assert out.samples.shape == (m, 4, model.horizon, 2)
    assert np.all(np.isfinite(out.samples))
    assert out.weights.shape == (m, 4)
    assert np.allclose(out.weights.sum(axis=1), 1.0)


def test_trajectron_train_and_predict(tiny_data):
    from construction_nav.env import ConstructionEnv
    from construction_nav.predict.trajectron import (
        TrajectronPredictor,
        train_trajectron,
    )

    train, val = tiny_data
    model, history = train_trajectron(
        train, val, epochs=2, batch_size=32, n_modes=4, log=lambda *a: None
    )
    assert history["train"][-1] < history["train"][0]

    predictor = TrajectronPredictor(model)
    env = ConstructionEnv(generate_scene(seed=99, num_agents=5))
    out = predictor.predict(env.observe())
    m = 5
    assert out.samples.shape == (m, 4, model.horizon, 2)
    assert np.all(np.isfinite(out.samples))
    assert np.allclose(out.weights.sum(axis=1), 1.0)
    # The reported mean is the weighted mixture of the modes.
    mixture = (out.samples * out.weights[:, :, None, None]).sum(axis=1)
    assert np.allclose(out.mean, mixture, atol=1e-5)


def test_diffusion_predictor_save_load(tiny_data, tmp_path):
    from construction_nav.predict.diffusion import (
        DiffusionPredictor,
        load_predictor,
        save_predictor,
        train_diffusion,
    )

    train, val = tiny_data
    model, _h, schedule = train_diffusion(
        train, val, epochs=1, batch_size=32, log=lambda *a: None
    )
    predictor = DiffusionPredictor(
        model, diffusion_steps=schedule.n_steps, n_samples=3, sample_steps=4
    )
    path = save_predictor(tmp_path / "d.pt", predictor)
    loaded = load_predictor(path)
    assert loaded.horizon == predictor.horizon
    assert loaded.n_samples == predictor.n_samples
    assert loaded.schedule.n_steps == predictor.schedule.n_steps


def test_cv_prior_is_linear_extrapolation():
    from construction_nav.predict.diffusion import cv_prior, residual_target

    # A perfectly linear history means the residual over the CV prior is zero.
    t = torch.arange(8).float()
    hist = torch.stack([t, 2.0 * t], dim=-1)          # (8, 2)
    hist = hist.reshape(1, 1, 8, 2)                   # (B=1, N=1, T=8, 2)
    horizon, dt = 4, 1.0
    prior = cv_prior(hist, horizon, dt)               # (1, 1, H, 2)
    vel = hist[:, :, -1, :] - hist[:, :, -2, :]       # (1, 1, 2)
    steps = torch.arange(1, horizon + 1).float()[None, None, :, None]
    expected = vel[:, :, None, :] * steps             # (1, 1, H, 2)
    assert torch.allclose(prior, expected)

    future = hist[:, :, -1, :][:, :, None, :] + prior
    assert torch.allclose(residual_target(future, hist, dt), torch.zeros_like(prior))
