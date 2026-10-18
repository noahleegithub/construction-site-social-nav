"""Benchmark charts: ADE / FDE versus forecast horizon and agent count."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from construction_nav.evaluate import PredictorScore

# Consistent colours across figures.
COLORS = {
    "constant velocity": "#1f77b4",
    "trajectron++ (reduced)": "#ff7f0e",
    "diffusion (K=6)": "#2ca02c",
}
FALLBACK = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _color(name: str, i: int) -> str:
    return COLORS.get(name, FALLBACK[i % len(FALLBACK)])


def _save(fig, out_dir, name: str, dpi: int = 220) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        p = out_path / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


def plot_ade_vs_horizon(
    results: Mapping[str, PredictorScore],
    out_dir: str | os.PathLike,
    num_agents: int | None = None,
    dpi: int = 220,
) -> list[Path]:
    """Two panels: ADE and FDE against forecast horizon ``H``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (name, s) in enumerate(results.items()):
        hs = sorted(s.ade)
        c = _color(name, i)
        axes[0].plot(hs, [s.ade[h] for h in hs], "o-", color=c, label=name)
        axes[1].plot(hs, [s.fde[h] for h in hs], "s--", color=c, label=name)

    for ax, title in zip(axes, ("ADE vs horizon", "FDE vs horizon")):
        ax.set_xlabel("forecast horizon H (steps)")
        ax.set_ylabel("error (m)")
        ax.set_title(title + (f"  (M={num_agents})" if num_agents else ""))
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted({h for s in results.values() for h in s.ade}))
    axes[0].legend(fontsize=9)
    fig.suptitle("Forecast error vs horizon", y=1.02)
    return _save(fig, out_dir, "benchmark_horizon", dpi=dpi)


def plot_training_curves(
    histories: Mapping[str, dict],
    out_dir: str | os.PathLike,
    dpi: int = 220,
) -> list[Path]:
    """Train/val loss curves for the learned predictors."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (name, hist) in enumerate(histories.items()):
        c = _color(name, i)
        ax.plot(hist["train"], color=c, label=f"{name} (train)")
        if hist.get("val"):
            ax.plot(hist["val"], color=c, linestyle="--", label=f"{name} (val)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("Training curves")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "training_curves", dpi=dpi)


def plot_ade_vs_agents(
    by_agents: Mapping[int, Mapping[str, PredictorScore]],
    out_dir: str | os.PathLike,
    horizon: int = 16,
    dpi: int = 220,
) -> list[Path]:
    """ADE / FDE at a fixed horizon against the number of agents ``M``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    agents = sorted(by_agents)
    names = list(next(iter(by_agents.values())).keys())
    for i, name in enumerate(names):
        c = _color(name, i)
        ade = [by_agents[m][name].ade.get(horizon, np.nan) for m in agents]
        fde = [by_agents[m][name].fde.get(horizon, np.nan) for m in agents]
        axes[0].plot(agents, ade, "o-", color=c, label=name)
        axes[1].plot(agents, fde, "s--", color=c, label=name)

    for ax, title in zip(axes, (f"ADE at H={horizon}", f"FDE at H={horizon}")):
        ax.set_xlabel("number of agents M")
        ax.set_ylabel("error (m)")
        ax.set_title(title)
        ax.set_xticks(agents)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Forecast error vs agent count", y=1.02)
    return _save(fig, out_dir, "benchmark_agents", dpi=dpi)
