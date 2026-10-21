#!/usr/bin/env python
"""Regenerate the static project figures (scene, predictions, sample trials).

    python scripts/make_figures.py --out-dir figures --seed 0 --agents 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from construction_nav import config as C
from construction_nav.env import generate_scene
from construction_nav.planner import OrcaPlanner, FixedRadiusPlanner, JointPlanner
from construction_nav.predict import ConstantVelocityPredictor
from construction_nav.render import render_scene
from construction_nav.render_anim import animate_trial
from construction_nav.render_predictions import render_diffusion_sequence, render_pipeline
from construction_nav.render_trials import plot_trial


def _make_planner(name: str):
    if name == "joint":
        return JointPlanner(predictor=ConstantVelocityPredictor(horizon=C.HORIZON_STEPS))
    if name == "orca":
        return OrcaPlanner()
    if name == "fixed-radius":
        return FixedRadiusPlanner(use_oracle=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--arena", type=float, default=60.0)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--planner", choices=["orca", "joint", "fixed-radius", "pursuit"],
                        default="joint")
    parser.add_argument("--animate", action="store_true", help="also write a GIF")
    args = parser.parse_args()

    scene = generate_scene(seed=args.seed, num_agents=args.agents, arena=args.arena)
    written = []
    written += render_scene(scene, out_dir=args.out_dir, dpi=args.dpi)
    written += render_diffusion_sequence(scene, args.out_dir, dpi=args.dpi)
    written += render_pipeline(args.out_dir, dpi=args.dpi)

    planner = _make_planner(args.planner)
    paths, summary = plot_trial(
        scene, args.out_dir, name="trial", planner=planner, steps=3500
    )
    written += paths
    if args.animate:
        written += animate_trial(
            scene, args.out_dir, name="trial", planner=planner, steps=900
        )

    for path in written:
        print(f"wrote {path}")
    print(summary)


if __name__ == "__main__":
    main()
