#!/usr/bin/env python
"""Benchmark the predictors: ADE / FDE versus horizon H and agent count M.

    python scripts/benchmark.py --agents 5 7 9 --epochs 60 --out-dir figures

For every agent count in the sweep the script builds disjoint train / val / test
datasets, trains the diffusion and reduced-Trajectron++ predictors, and scores all
three models (plus the constant-velocity baseline) with open-loop evaluation on
closed-loop rollouts. It writes two figures and a JSON summary.

Checkpoints are cached under ``checkpoints/bench_<M>.pt`` and reused unless
``--retrain`` is given, so re-plotting is cheap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from construction_nav.data import train_val_test_split
from construction_nav.evaluate import (
    DEFAULT_HORIZONS,
    evaluate_many,
    scores_to_json,
)
from construction_nav.predict import ConstantVelocityPredictor
from construction_nav.render_metrics import plot_ade_vs_agents, plot_ade_vs_horizon


def _predictors(agents: int, checkpoints: Path, args, retrain: bool, log=print):
    import torch  # noqa: F401

    from construction_nav.predict.diffusion import (
        DiffusionPredictor,
        load_predictor as load_diffusion,
        save_predictor as save_diffusion,
        train_diffusion,
    )
    from construction_nav.predict.trajectron import (
        TrajectronPredictor,
        load_predictor as load_trajectron,
        save_predictor as save_trajectron,
        train_trajectron,
    )

    # The diffusion model has a fixed horizon; the constant-velocity baseline is
    # built to match it.
    horizon = None
    diffusion_ckpt = checkpoints / f"bench_{agents}_diffusion.pt"
    trajectron_ckpt = checkpoints / f"bench_{agents}_trajectron.pt"

    if not retrain and diffusion_ckpt.exists() and trajectron_ckpt.exists():
        log(f"  loading cached checkpoints for M={agents}")
        diffusion = load_diffusion(diffusion_ckpt)
        trajectron = load_trajectron(trajectron_ckpt)
        horizon = diffusion.horizon
    else:
        train, val, _ = train_val_test_split(
            num_agents=agents, arena=args.arena, steps=args.steps,
            n_train=args.train_scenes, n_val=args.val_scenes, n_test=args.test_scenes,
            seed_offset=args.seed * 1000 + agents,
        )
        log(f"  M={agents}: {len(train)} train windows, N={train.num_entities}")
        d_model, _hist, schedule = train_diffusion(
            train, val, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            diffusion_steps=args.diffusion_steps, seed=args.seed,
        )
        diffusion = DiffusionPredictor(
            d_model, diffusion_steps=schedule.n_steps, n_samples=args.modes,
            sample_steps=args.sample_steps, dt=train.dt,
        )
        save_diffusion(diffusion_ckpt, diffusion)
        t_model, _th = train_trajectron(
            train, val, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            n_modes=args.modes, seed=args.seed,
        )
        trajectron = TrajectronPredictor(t_model)
        save_trajectron(trajectron_ckpt, trajectron)
        horizon = diffusion.horizon

    return {
        "constant velocity": ConstantVelocityPredictor(horizon=horizon),
        "trajectron++ (reduced)": trajectron,
        "diffusion (K=6)": diffusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agents", type=int, nargs="+", default=[5, 7, 9])
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--arena", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--train-scenes", type=int, default=16)
    parser.add_argument("--val-scenes", type=int, default=6)
    parser.add_argument("--test-scenes", type=int, default=8)
    parser.add_argument("--eval-stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain", action="store_true", help="ignore cached checkpoints")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    by_agents: dict[int, dict] = {}
    for agents in args.agents:
        print(f"\n=== M = {agents} ===")
        predictors = _predictors(agents, args.checkpoint_dir, args, args.retrain)
        test_seeds = range(
            args.seed * 1000 + agents + args.train_scenes + args.val_scenes,
            args.seed * 1000 + agents + args.train_scenes + args.val_scenes + args.test_scenes,
        )
        scores = evaluate_many(
            predictors, test_seeds, horizons=tuple(args.horizons), num_agents=agents,
            arena=args.arena, steps=args.steps, stride=args.eval_stride,
        )
        by_agents[agents] = scores

    # A horizon figure for the mid-sized scenario, and an agent-count sweep.
    mid = args.agents[len(args.agents) // 2]
    written = plot_ade_vs_horizon(by_agents[mid], args.out_dir, num_agents=mid)
    written += plot_ade_vs_agents(by_agents, args.out_dir, horizon=max(args.horizons))
    for path in written:
        print(f"wrote {path}")

    payload = {
        "arena": args.arena,
        "seed": args.seed,
        "epochs": args.epochs,
        "eval_stride": args.eval_stride,
        "agents": args.agents,
        "horizons": args.horizons,
        "results": {str(m): scores_to_json(s, m) for m, s in by_agents.items()},
    }
    out = args.out_dir / "benchmark.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
