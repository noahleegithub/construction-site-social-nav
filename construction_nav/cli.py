"""Command-line interface for the construction-site social navigation project.

Subcommands
-----------
``scene``       render a static worksite figure (and the arena backdrop).
``trial``       run one closed-loop trial and plot the paths.
``predictions`` render the MotionDiffuser-style schematic figures.
``robotarium``  run the social-navigation demos on the Robotarium simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from construction_nav import config as C


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the scene")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--agents", type=int, default=None,
                       help=f"non-robot agents M ({C.MIN_AGENTS}-{C.MAX_AGENTS})")
    group.add_argument("--total-agents", type=int, default=None,
                       help="agents including the robot (6-20); M = total - 1")
    parser.add_argument("--arena", type=float, default=C.ARENA_SIZE,
                        help="worksite side length in metres")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"),
                        help="output directory")
    parser.add_argument("--dpi", type=int, default=300, help="raster resolution")


def _agent_count(args) -> int:
    """Resolve ``--agents`` / ``--total-agents`` to M (non-robot agents)."""
    if getattr(args, "total_agents", None) is not None:
        total = int(args.total_agents)
        if not C.MIN_AGENTS + 1 <= total <= C.MAX_AGENTS + 1:
            raise SystemExit(
                f"--total-agents must be in [{C.MIN_AGENTS + 1}, {C.MAX_AGENTS + 1}]"
            )
        return total - 1
    count = int(C.DEFAULT_AGENTS if args.agents is None else args.agents)
    if not C.MIN_AGENTS <= count <= C.MAX_AGENTS:
        raise SystemExit(
            f"--agents must be in [{C.MIN_AGENTS}, {C.MAX_AGENTS}] "
            f"(total entities {C.MIN_AGENTS + 1}-{C.MAX_AGENTS + 1})"
        )
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="construction-nav", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_scene = sub.add_parser("scene", help="render a static worksite figure")
    _add_common(p_scene)
    p_scene.add_argument("--no-cones", action="store_true", help="hide FOV / blindspots")

    p_trial = sub.add_parser("trial", help="run one closed-loop trial")
    _add_common(p_trial)
    p_trial.add_argument("--steps", type=int, default=2000, help="max simulation steps")
    p_trial.add_argument("--planner", choices=["pursuit", "fixed-radius", "joint"],
                         default="pursuit")
    p_trial.add_argument("--animate", action="store_true", help="also write a GIF")

    p_pred = sub.add_parser("predictions", help="render MotionDiffuser-style schematics")
    p_pred.add_argument("--seed", type=int, default=0)
    p_pred.add_argument("--agents", type=int, default=7)
    p_pred.add_argument("--out-dir", type=Path, default=Path("figures"))
    p_pred.add_argument("--dpi", type=int, default=300)

    p_rob = sub.add_parser("robotarium", help="run Robotarium social-navigation demos")
    p_rob.add_argument("--out-dir", type=Path, default=Path("figures"))
    p_rob.add_argument("--case", default=None, help="specific demo case name")
    p_rob.add_argument("--agents", type=int, default=5, help="agents in the miniature site")
    p_rob.add_argument("--seeds", type=int, default=8, help="seeds scanned per case")
    p_rob.add_argument("--max-steps", type=int, default=4000)
    p_rob.add_argument("--barrier", choices=["all", "robot", "none"], default="all")
    p_rob.add_argument("--planner", choices=["fixed-radius", "joint"],
                       default="fixed-radius")
    p_rob.add_argument("--no-animate", action="store_true")
    p_rob.add_argument("--json", type=Path, default=None, help="write a JSON summary here")

    p_check = sub.add_parser("check", help="validate that scenes and envs build")
    _add_common(p_check)
    p_check.add_argument("--trials", type=int, default=5)
    return parser


def _make_planner(name: str):
    from construction_nav import config as C
    from construction_nav.planner import FixedRadiusPlanner, JointPlanner
    from construction_nav.predict import ConstantVelocityPredictor

    if name == "fixed-radius":
        return FixedRadiusPlanner(use_oracle=True)
    if name == "joint":
        return JointPlanner(predictor=ConstantVelocityPredictor(horizon=C.HORIZON_STEPS))
    return None


def cmd_scene(args) -> int:
    from construction_nav.env import generate_scene
    from construction_nav.render import render_scene

    scene = generate_scene(seed=args.seed, num_agents=_agent_count(args), arena=args.arena)
    written = render_scene(scene, out_dir=args.out_dir, show_cones=not args.no_cones, dpi=args.dpi)
    for path in written:
        print(f"wrote {path}")
    return 0


def cmd_trial(args) -> int:
    from construction_nav.env import generate_scene
    from construction_nav.render_anim import animate_trial
    from construction_nav.render_trials import plot_trial

    scene = generate_scene(seed=args.seed, num_agents=_agent_count(args), arena=args.arena)
    planner = _make_planner(args.planner)

    paths, summary = plot_trial(scene, args.out_dir, name="trial", planner=planner, steps=args.steps)
    if args.animate:
        paths += animate_trial(scene, args.out_dir, name="trial", planner=planner, steps=args.steps)
    for path in paths:
        print(f"wrote {path}")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_predictions(args) -> int:
    from construction_nav.env import generate_scene
    from construction_nav.render_predictions import render_diffusion_sequence, render_pipeline

    scene = generate_scene(seed=args.seed, num_agents=args.agents)
    written = render_diffusion_sequence(scene, args.out_dir)
    written += render_pipeline(args.out_dir)
    for path in written:
        print(f"wrote {path}")
    return 0


def cmd_robotarium(args) -> int:
    import dataclasses

    from construction_nav.robotarium.demo import DEFAULT_CASES, run_all

    cases = DEFAULT_CASES
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            raise SystemExit(f"unknown case {args.case!r}")
    cases = [dataclasses.replace(c, num_agents=args.agents) for c in cases]
    results = run_all(out_dir=args.out_dir, cases=cases, barrier_mode=args.barrier,
                      max_steps=args.max_steps, animate=not args.no_animate,
                      n_seeds=args.seeds, planner=_make_planner(args.planner))
    print(json.dumps(results, indent=2))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    return 0


def cmd_check(args) -> int:
    from construction_nav.env import ConstructionEnv, generate_scene

    for seed in range(args.trials):
        scene = generate_scene(seed=seed, num_agents=_agent_count(args), arena=args.arena)
        env = ConstructionEnv(scene)
        for _ in range(50):
            env.step()
        print(f"seed={seed} nodes={len(scene.nodes)} "
              f"agents={len(scene.agents)} ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "scene": cmd_scene,
        "trial": cmd_trial,
        "predictions": cmd_predictions,
        "robotarium": cmd_robotarium,
        "check": cmd_check,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
