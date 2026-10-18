#!/usr/bin/env python
"""Run the social-navigation demos on the Robotarium simulator.

    python scripts/robotarium_demo.py --out-dir figures --agents 5

Each demo builds a miniature construction site that fits the Robotarium arena,
runs the robot policy alongside scripted agents in both the native environment
and the Robotarium, and reports the site metrics plus the Robotarium's own
collision / boundary validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from construction_nav.robotarium.demo import DEFAULT_CASES, run_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--case", default=None, help="run a single demo case by name")
    parser.add_argument("--barrier", choices=["all", "robot", "none"], default="all")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--no-animate", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cases = DEFAULT_CASES
    if args.case:
        cases = [c for c in cases if c.name == args.case] or None
        if cases is None:
            raise SystemExit(f"unknown case {args.case!r}")

    results = run_all(
        out_dir=args.out_dir,
        cases=cases,
        barrier_mode=args.barrier,
        max_steps=args.max_steps,
        animate=not args.no_animate,
    )
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
