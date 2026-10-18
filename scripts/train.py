#!/usr/bin/env python
"""Train the learned predictors (diffusion + reduced Trajectron++).

    python scripts/train.py --agents 7 --epochs 80 --out-dir figures

Builds train / validation / test datasets by rolling out randomized construction
sites, trains both models on the train split (early-stopped on validation), saves
checkpoints under ``checkpoints/`` and writes the training curves to ``figures/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from construction_nav.data import train_val_test_split
from construction_nav.render_metrics import plot_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--arena", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=90, help="rollout steps per scene")
    parser.add_argument("--train-scenes", type=int, default=32)
    parser.add_argument("--val-scenes", type=int, default=8)
    parser.add_argument("--test-scenes", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--modes", type=int, default=6, help="K for both models")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    import torch  # noqa: F401  (fail early with a clear error if missing)

    from construction_nav.predict.diffusion import (
        DiffusionPredictor,
        save_predictor as save_diffusion,
        train_diffusion,
    )
    from construction_nav.predict.trajectron import (
        TrajectronPredictor,
        save_predictor as save_trajectron,
        train_trajectron,
    )

    print(f"building datasets (M={args.agents}, {args.train_scenes} train scenes)...")
    train, val, _ = train_val_test_split(
        num_agents=args.agents, arena=args.arena, steps=args.steps,
        n_train=args.train_scenes, n_val=args.val_scenes, n_test=args.test_scenes,
        seed_offset=args.seed * 1000,
    )
    print(f"train windows: {len(train)}  val windows: {len(val)}  "
          f"N={train.num_entities}  T={train.history_len}  H={train.horizon}")

    print("\ntraining diffusion predictor...")
    d_model, d_hist, schedule = train_diffusion(
        train, val, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        diffusion_steps=args.diffusion_steps, seed=args.seed,
    )
    diffusion = DiffusionPredictor(
        d_model, diffusion_steps=schedule.n_steps, n_samples=args.modes,
        sample_steps=args.sample_steps, dt=train.dt,
    )
    ckpt = save_diffusion(args.checkpoint_dir / "diffusion.pt", diffusion)
    print(f"saved {ckpt}")

    print("\ntraining reduced Trajectron++...")
    t_model, t_hist = train_trajectron(
        train, val, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        n_modes=args.modes, seed=args.seed,
    )
    trajectron = TrajectronPredictor(t_model)
    ckpt = save_trajectron(args.checkpoint_dir / "trajectron.pt", trajectron)
    print(f"saved {ckpt}")

    written = plot_training_curves(
        {"diffusion": d_hist, "trajectron++ (reduced)": t_hist}, args.out_dir
    )
    for path in written:
        print(f"wrote {path}")

    summary = {
        "agents": args.agents,
        "arena": args.arena,
        "epochs": args.epochs,
        "train_windows": len(train),
        "val_windows": len(val),
        "diffusion": {"train": d_hist["train"], "val": d_hist["val"]},
        "trajectron": {"train": t_hist["train"], "val": t_hist["val"]},
    }
    out = args.out_dir / "training_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
