"""Dataset generation for the learned predictors.

A *sample* is one observation window and the future it should forecast, for all
``N = M + 1`` entities (the ``M`` agents plus the robot):

    hist    (N, T_obs, 2)   past positions, normalised by the arena
    future  (N, H, 2)       true future positions, normalised by the arena
    kind    (N,)            entity type id (human / vehicle / robot)
    state   (N,)            working / moving flag at the window's last frame

Positions are divided by the arena side length, so a single model trained on the
60 m reference site transfers exactly to the miniature Robotarium site (the
problem is scale-invariant; see :mod:`construction_nav.config`).

Generation is pure ``numpy`` -- it does **not** import torch, so the environment
and the baseline predictors keep working without the optional ML extras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from construction_nav import config as C
from construction_nav.env import ConstructionEnv, generate_scene

#: entity kinds, in id order
KINDS: tuple[str, ...] = ("human", "vehicle", "robot")
KIND_TO_ID = {k: i for i, k in enumerate(KINDS)}

#: agent activity states, in id order
STATES: tuple[str, ...] = ("working", "moving")
STATE_TO_ID = {s: i for i, s in enumerate(STATES)}

ROBOT_KIND_ID = KIND_TO_ID["robot"]


@dataclass
class Dataset:
    """In-memory dataset arrays (normalised positions unless ``arena_norm=False``)."""

    hist: np.ndarray        # (S, N, T, 2) float32
    future: np.ndarray      # (S, N, H, 2) float32
    kind: np.ndarray        # (S, N) int8
    state: np.ndarray       # (S, N) int8
    arena: float
    dt: float

    def __len__(self) -> int:
        return int(self.hist.shape[0])

    @property
    def num_entities(self) -> int:
        return int(self.hist.shape[1])

    @property
    def history_len(self) -> int:
        return int(self.hist.shape[2])

    @property
    def horizon(self) -> int:
        return int(self.future.shape[2])


def episode_windows(
    scene,
    steps: int = 100,
    dt: float = C.DT,
    planner=None,
    history: int = C.HISTORY_STEPS,
    horizon: int = C.HORIZON_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll one scene out and slice it into ``(hist, future, kind, state)`` windows."""
    env = ConstructionEnv(scene, dt=dt)
    state_seq: list[list[int]] = [[STATE_TO_ID[a.state] for a in scene.agents]]
    for _ in range(steps):
        info = env.step(None if planner is None else planner(env))
        state_seq.append([STATE_TO_ID[a.state] for a in scene.agents])
        if info.robot_reached_goal:
            break

    # (N, T + 1, 2): agents first, robot last (matches the observation layout).
    agent_pos = np.asarray(env.agent_trajectories, dtype=float)      # (M, T+1, 2)
    robot_pos = np.asarray(env.robot_trajectory, dtype=float)[None]  # (1, T+1, 2)
    pos = np.concatenate([agent_pos, robot_pos], axis=0)             # (N, T+1, 2)

    n = pos.shape[0]
    t_total = pos.shape[1] - 1
    # ``state_seq`` has one entry per frame (initial frame included); the robot is
    # always "moving" for the model's purposes.
    states_agents = np.asarray(state_seq, dtype=np.int8).T            # (M, T+1)
    states = np.vstack([states_agents, np.ones((1, states_agents.shape[1]), np.int8)])

    kind = np.array([KIND_TO_ID[a.kind] for a in scene.agents] + [ROBOT_KIND_ID], dtype=np.int8)

    hists, futures, kinds, state0 = [], [], [], []
    for i in range(history - 1, t_total - horizon):
        hists.append(pos[:, i - history + 1 : i + 1, :])
        futures.append(pos[:, i + 1 : i + horizon + 1, :])
        kinds.append(kind)
        state0.append(states[:, i])

    if not hists:
        empty = np.zeros((0, n, history, 2), np.float32)
        return empty, np.zeros((0, n, horizon, 2), np.float32), np.zeros((0, n), np.int8), np.zeros((0, n), np.int8)

    return (
        np.asarray(hists, dtype=np.float32),
        np.asarray(futures, dtype=np.float32),
        np.asarray(kinds, dtype=np.int8),
        np.asarray(state0, dtype=np.int8),
    )


def build_dataset(
    seeds,
    num_agents: int = C.DEFAULT_AGENTS,
    arena: float = C.ARENA_SIZE,
    steps: int = 100,
    planner=None,
    history: int = C.HISTORY_STEPS,
    horizon: int = C.HORIZON_STEPS,
    normalize: bool = True,
) -> Dataset:
    """Generate a dataset by rolling out one scene per seed."""
    hists, futures, kinds, states = [], [], [], []
    for seed in seeds:
        scene = generate_scene(seed=int(seed), num_agents=num_agents, arena=arena)
        h, f, k, s = episode_windows(
            scene, steps=steps, planner=planner, history=history, horizon=horizon
        )
        if len(h):
            hists.append(h)
            futures.append(f)
            kinds.append(k)
            states.append(s)

    if not hists:
        raise RuntimeError("no windows generated; increase `steps`")

    hist = np.concatenate(hists, axis=0)
    future = np.concatenate(futures, axis=0)
    kind = np.concatenate(kinds, axis=0)
    state = np.concatenate(states, axis=0)

    if normalize:
        hist = hist / arena
        future = future / arena

    return Dataset(hist=hist, future=future, kind=kind, state=state, arena=arena, dt=C.DT)


def save_dataset(path: str | Path, data: Dataset) -> Path:
    """Write a :class:`Dataset` to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hist=data.hist,
        future=data.future,
        kind=data.kind,
        state=data.state,
        arena=np.float32(data.arena),
        dt=np.float32(data.dt),
    )
    return path


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset written by :func:`save_dataset`."""
    with np.load(Path(path)) as z:
        return Dataset(
            hist=z["hist"].astype(np.float32),
            future=z["future"].astype(np.float32),
            kind=z["kind"].astype(np.int8),
            state=z["state"].astype(np.int8),
            arena=float(z["arena"]),
            dt=float(z["dt"]),
        )


def train_val_test_split(
    num_agents: int = C.DEFAULT_AGENTS,
    arena: float = C.ARENA_SIZE,
    steps: int = 100,
    n_train: int = 40,
    n_val: int = 8,
    n_test: int = 12,
    seed_offset: int = 0,
    **kwargs,
) -> tuple[Dataset, Dataset, Dataset]:
    """Build disjoint-seed train / val / test datasets."""
    train = build_dataset(
        range(seed_offset, seed_offset + n_train),
        num_agents=num_agents, arena=arena, steps=steps, **kwargs
    )
    val = build_dataset(
        range(seed_offset + n_train, seed_offset + n_train + n_val),
        num_agents=num_agents, arena=arena, steps=steps, **kwargs
    )
    test = build_dataset(
        range(seed_offset + n_train + n_val, seed_offset + n_train + n_val + n_test),
        num_agents=num_agents, arena=arena, steps=steps, **kwargs
    )
    return train, val, test
