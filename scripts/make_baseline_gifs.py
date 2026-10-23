"""Baseline GIFs: the same scene, five planners, side by side.

Each situation is one GIF plus a final-frame contact sheet. Every panel runs on
its own copy of the *same* seed, so the site, the crowd and the clock are
identical and the only difference in the picture is the robot's policy.

    python scripts/make_baseline_gifs.py --situation all
    python scripts/make_baseline_gifs.py --situation head_on_pedestrian --steps 400

Seeds come from ``figures/midterm/situations.json`` (written by
``scripts/find_situations.py``); pass ``--rescan`` to rebuild that file first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from construction_nav import config as C
from construction_nav.env import generate_scene
from construction_nav.planner import FixedRadiusPlanner, JointPlanner, OrcaPlanner
from construction_nav.predict import ConstantVelocityPredictor
from construction_nav.render_anim import animate_comparison

# Panel labels: the README carries the full names, the panels take a wrapped
# short form so six titles fit across a slide.
SHORT = {
    "Path following": "Path following",
    "ORCA": "ORCA",
    "ORCA + Constant Velocity forecasting": "ORCA + Constant Velocity\nforecasting",
    "fixed-radius avoidance + perfect trajectory forecasting":
        "fixed-radius avoidance +\nperfect trajectory forecasting",
    "MPC + perfect trajectory forecasting":
        "MPC +\nperfect trajectory forecasting",
}

SITUATIONS = {
    "head_on_pedestrian": "Head-on pedestrian",
    "blind_spot_worker": "Worker in the robot's blind arc",
    "crossing_vehicle": "Crossing vehicle",
    "dense_crowd": "Dense crowd (M = 19)",
}


def _planners() -> list[tuple[str, object]]:
    """The five rows of the baseline table, in table order."""
    return [
        ("Path following", None),
        ("ORCA", OrcaPlanner()),
        (
            "ORCA + Constant Velocity forecasting",
            OrcaPlanner(forecast=ConstantVelocityPredictor(horizon=C.HORIZON_STEPS)),
        ),
        (
            "fixed-radius avoidance + perfect trajectory forecasting",
            FixedRadiusPlanner(use_oracle=True),
        ),
        ("MPC + perfect trajectory forecasting", JointPlanner(oracle=True)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--situation", default="all",
        choices=["all", *SITUATIONS],
    )
    parser.add_argument("--situations-file", type=Path,
                        default=Path("figures/baselines/situations.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/baselines"))
    parser.add_argument("--arena", type=float, default=C.ARENA_SIZE)
    parser.add_argument(
        "--steps", type=int, default=260,
        help="step budget per panel; a finished trial holds its last frame, "
             "so a larger budget only adds duplicate frames",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=72)
    args = parser.parse_args()

    chosen = json.loads(args.situations_file.read_text())
    names = list(SITUATIONS) if args.situation == "all" else [args.situation]

    for name in names:
        row = chosen[name]
        seed, agents = int(row["seed"]), int(row["num_agents"])
        print(f"\n=== {SITUATIONS[name]}  seed={seed}  M={agents} ===")
        scene = generate_scene(seed=seed, num_agents=agents, arena=args.arena)
        entries = [(SHORT[label], planner) for label, planner in _planners()]
        encounter = row.get("t")
        still_index = None
        if encounter is not None:
            # The scan records the time of the peak signature; the contact
            # sheet is taken at that step so it shows the encounter.
            still_index = int(round(encounter / C.DT))
        paths, summary = animate_comparison(
            scene,
            entries,
            out_dir=args.out_dir,
            name=name,
            steps=args.steps,
            fps=args.fps,
            dpi=args.dpi,
            note=f"{SITUATIONS[name]}   seed {seed}   M = {agents}",
            still=True,
            still_index=still_index,
        )
        for label, stats in summary.items():
            flat = label.replace("\n", " ")
            print(
                f"  {flat:<58} reached={stats['reached']}  "
                f"t={stats['t']:6.1f}s  contacts={stats['contacts']}  "
                f"J={stats['J']:6.2f}"
            )
        for path in paths:
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
