"""Torch ``Dataset``/``DataLoader`` wrappers around :class:`Dataset` arrays.

Kept separate from :mod:`construction_nav.data.generate` so that dataset
*generation* stays torch-free.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from construction_nav.data.generate import Dataset


class WindowDataset(TorchDataset):
    """Turns a :class:`Dataset` into ``(hist, kind, state) -> future`` samples."""

    def __init__(self, data: Dataset) -> None:
        self.hist = torch.as_tensor(data.hist, dtype=torch.float32)
        self.future = torch.as_tensor(data.future, dtype=torch.float32)
        self.kind = torch.as_tensor(data.kind, dtype=torch.long)
        self.state = torch.as_tensor(data.state, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.hist.shape[0])

    def __getitem__(self, i: int):
        return {
            "hist": self.hist[i],
            "kind": self.kind[i],
            "state": self.state[i],
            "future": self.future[i],
        }


def make_loader(
    data: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader:
    """Build a deterministic (seeded) ``DataLoader`` over ``data``."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        WindowDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def batch_to_tensors(batch: dict) -> dict:
    """Move a loader batch to float/long tensors (no-op for CPU-only setups)."""
    return {
        "hist": batch["hist"],
        "kind": batch["kind"],
        "state": batch["state"],
        "future": batch["future"],
    }
