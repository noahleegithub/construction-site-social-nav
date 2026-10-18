#!/usr/bin/env python
"""Evaluate robot planners on randomized trials: comparison, lambda_1 sweep, failures.

    python scripts/evaluate_planners.py --agents 7 --trials 16 --out-dir figures

Runs a set of planners on identical randomized construction sites and reports the
Eq. (3) cost ``J``, success rate, collisions, intrusion and travel time. It also
sweeps the intrusion weight ``lambda_1`` for the joint diffusion planner and
collects failure cases, writing:

    figures/planner_comparison.{png,pdf}
    figures/lambda_sweep.{png,pdf}
    figures/failure_cases.{png,pdf}
    figures/planner_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from construction_nav import config as C
from construction_nav.planner import FixedRadiusPlanner, JointPlanner
from construction_nav.render_metrics import (
    plot_failure_cases,
    plot_lambda_sweep,
    plot_planner_comparison,
)
from construction_nav.trials import collect_failure_cases, evaluate_planners


def _load_predictors(checkpoint_dir: Path, log=print):
    """Load the trained predictors; missing checkpoints are skipped with a note."""
    predictors: dict[str, object] = {}
    import torch  # noqa: F401

    from construction_nav.predict import ConstantVelocityPredictor

    predictors["constant velocity"] = ConstantVelocityPredictor(horizon=C.HORIZON_STEPS)

    dpath = checkpoint_dir / "diffusion.pt"
    tpath = checkpoint_dir / "trajectron.pt"
    if dpath.exists():
        from construction_nav.predict import load_diffusion

        predictors["diffusion"] = load_diffusion(dpath)
    else:
        log(f"  (no {dpath}; skipping the diffusion planner)")
    if tpath.exists():
        from construction_nav.predict import load_trajectron

        predictors["trajectron++ (reduced)"] = load_trajectron(tpath)
    else:
        log(f"  (no {tpath}; skipping the Trajectron++ planner)")
    return predictors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agents", type=int, default=C.DEFAULT_AGENTS)
    parser.add_argument("--arena", type=float, default=C.ARENA_SIZE)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--lambda-intrusion", type=float, default=1.0)
    parser.add_argument("--lambda-freeze", type=float, default=0.5)
    parser.add_argument("--lambda-sweep", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--failure-scan", type=int, default=32,
                        help="seeds scanned for failure cases")
    parser.add_argument("--failure-limit", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.trials))
    predictors = _load_predictors(args.checkpoint_dir)

    # -- planner comparison -------------------------------------------------
    planners: dict[str, object] = {
        "fixed-radius (oracle)": FixedRadiusPlanner(use_oracle=True),
        "joint planner (oracle)": JointPlanner(oracle=True),
    }
    for name, predictor in predictors.items():
        planners[f"joint planner ({name})"] = JointPlanner(predictor=predictor)
    if "diffusion" in predictors:
        planners["joint planner (diffusion + safety diffusion)"] = JointPlanner(
            predictor=predictors["diffusion"], safety_diffusion=True
        )

    print(f"\n=== planner comparison (M={args.agents}, {len(seeds)} trials) ===")
    reports = evaluate_planners(
        planners, seeds, num_agents=args.agents, arena=args.arena,
        max_steps=args.max_steps, lambda_intrusion=args.lambda_intrusion,
        lambda_freeze=args.lambda_freeze,
    )
    written = plot_planner_comparison(reports, args.out_dir)

    # -- lambda_1 sweep -----------------------------------------------------
    sweeps: dict[str, dict[float, object]] = {}
    if "diffusion" in predictors:
        print("\n=== lambda_1 sweep (joint diffusion planner) ===")
        for series, guarded in (("with safety filter", True), ("without safety filter", False)):
            sweep: dict[float, object] = {}
            for lam in args.lambda_sweep:
                planner = JointPlanner(
                    predictor=predictors["diffusion"],
                    lambda_intrusion=lam,
                    lambda_freeze=args.lambda_freeze,
                    hard_filter=guarded,
                )
                report = list(
                    evaluate_planners(
                        {f"lambda={lam}": planner}, seeds, num_agents=args.agents,
                        arena=args.arena, max_steps=args.max_steps,
                        lambda_intrusion=args.lambda_intrusion,
                        lambda_freeze=args.lambda_freeze,
                    ).values()
                )[0]
                sweep[float(lam)] = report
            sweeps[series] = sweep
        written += plot_lambda_sweep(sweeps, args.out_dir)

    # -- failure cases ------------------------------------------------------
    failures: list = []
    if "diffusion" in predictors:
        print("\n=== failure / hard cases (joint diffusion planner) ===")
        planner = JointPlanner(
            predictor=predictors["diffusion"],
            lambda_intrusion=args.lambda_intrusion,
            lambda_freeze=args.lambda_freeze,
        )
        failures = collect_failure_cases(
            planner, range(args.seed_start, args.seed_start + args.failure_scan),
            num_agents=args.agents, arena=args.arena, max_steps=args.max_steps,
            limit=args.failure_limit,
        )
        written += plot_failure_cases(failures, args.out_dir)

    for path in written:
        print(f"wrote {path}")

    payload = {
        "num_agents": args.agents,
        "arena": args.arena,
        "trials": args.trials,
        "seed_start": args.seed_start,
        "lambda_intrusion": args.lambda_intrusion,
        "lambda_freeze": args.lambda_freeze,
        "planners": {name: r.to_json() for name, r in reports.items()},
        "lambda_sweep": {
            series: {
                str(lam): {
                    "mean_J": r.mean_J,
                    "mean_intrusion": r.mean_intrusion,
                    "mean_travel_time": r.mean_travel_time,
                    "success_rate": r.success_rate,
                }
                for lam, r in sweep.items()
            }
            for series, sweep in sweeps.items()
        },
        "failures": [
            {
                "seed": r.seed,
                "reached": r.reached,
                "n_collision_steps": r.n_collision_steps,
                "travel_time": r.travel_time,
            }
            for r, _env in failures
        ],
    }
    out = args.out_dir / "planner_eval.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
