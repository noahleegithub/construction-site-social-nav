"""Pick the seed that best illustrates each social-navigation situation.

The baseline GIFs are only persuasive if the scene actually contains the
encounter they claim to show, so rather than hand-picking seeds this scans a
range and scores each trial against a geometric signature:

``head_on``        a *moving* human in front of the robot (bearing within
                   ``HEAD_ON_CONE`` of the robot's heading), within ``NEAR``
                   metres, closing on it, and with the robot inside that human's
                   field of view -- a genuine mutual encounter.
``blind_spot``     a moving human with the robot outside its field of view
                   (bearing past the FOV half-angle) and closing inside
                   ``BLIND_NEAR`` metres -- the case only prediction can help.
``crossing``       a vehicle whose velocity is within ``CROSS_DEG`` of
                   perpendicular to the robot's heading, at close range.
``dense_crowd``    the packing-limit crowd (``M = 19``); scored by ORCA's
                   intrusion integral, i.e. how crowded the robot's run actually
                   was.

Scores are ``(peak closeness)`` for the first three and the intrusion integral
for the crowd, so the maximum is the most legible example rather than merely the
first one that trips the test.

    python scripts/find_situations.py --out figures/baselines/situations.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from construction_nav import config as C
from construction_nav.env import ConstructionEnv, generate_scene
from construction_nav.planner import OrcaPlanner

NEAR = 4.0          # m: head-on "close enough to matter"
HEAD_ON_CONE = np.deg2rad(35.0)
BLIND_NEAR = 1.2    # m: blind-spot "about to touch"
CROSS_DEG = np.deg2rad(35.0)   # tolerance around perpendicular
CROSS_NEAR = 3.5    # m


def _bearing(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    delta = to_pos - from_pos
    return float(np.arctan2(delta[1], delta[0]))


def _angle_gap(a: float, b: float) -> float:
    return float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def scan(seed: int, num_agents: int, arena: float, planner) -> dict:
    """Run one trial and return the best score seen for each signature."""
    scene = generate_scene(seed=seed, num_agents=num_agents, arena=arena)
    env = ConstructionEnv(scene)
    layout = env.layout
    fov_half = float(C.FOV_HALF_ANGLE)

    best = {
        "head_on": 0.0, "blind_spot": 0.0, "crossing": 0.0,
        # Time of the busiest instant, used to place the dense-crowd still.
        "intrusion_rate": 0.0,
    }
    previous_intrusion = 0.0

    for _ in range(700):
        info = env.step(planner(env))
        intrusion_rate = env.intrusion_integral - previous_intrusion
        previous_intrusion = env.intrusion_integral
        if intrusion_rate > best["intrusion_rate"]:
            best["intrusion_rate"] = intrusion_rate
            best["intrusion_t"] = env.t
        if info.robot_reached_goal:
            # Two-goal trial: stepping onto goal 1 does not end the run.
            env.advance_robot_goal()
        robot = env.robot
        rpos = robot.pos
        rtheta = float(np.arctan2(robot.heading[1], robot.heading[0]))

        for agent in env.scene.agents:
            d = float(np.linalg.norm(agent.pos - rpos))
            bearing = _bearing(rpos, agent.pos)
            robot_in_agent_frame = _bearing(agent.pos, rpos)
            seen = _angle_gap(robot_in_agent_frame, agent.theta) <= fov_half

            if agent.kind == "human" and agent.state == "moving":
                # Closing speed along the line of sight (positive = approaching).
                los = (rpos - agent.pos) / max(d, 1e-9)
                closing = float(np.dot(agent.speed * agent.heading, los))
                if (
                    d < NEAR
                    and _angle_gap(bearing, rtheta) < HEAD_ON_CONE
                    and seen
                    and closing > 0.3
                ):
                    score = (NEAR - d) * closing
                    if score > best["head_on"]:
                        best["head_on"] = score
                        best["head_on_t"] = env.t
                if d < BLIND_NEAR and not seen and closing > 0.3:
                    score = (BLIND_NEAR - d) * closing
                    if score > best["blind_spot"]:
                        best["blind_spot"] = score
                        best["blind_spot_t"] = env.t

            if agent.kind == "vehicle" and d < CROSS_NEAR:
                v = agent.speed * agent.heading
                speed = float(agent.speed)
                if speed > 1e-3:
                    gap = _angle_gap(_bearing(np.zeros(2), v), rtheta)
                    if abs(gap - np.pi / 2) < CROSS_DEG:
                        score = CROSS_NEAR - d
                        if score > best["crossing"]:
                            best["crossing"] = score
                            best["crossing_t"] = env.t

        if env.robot.goals_reached >= 2:
            break

    best.setdefault("head_on_t", 0.0)
    best.setdefault("blind_spot_t", 0.0)
    best.setdefault("crossing_t", 0.0)
    best.setdefault("intrusion_t", 0.0)
    best["intrusion"] = float(env.intrusion_integral)
    best["reached"] = bool(env.robot.goals_reached >= 2)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=36)
    parser.add_argument("--crowd-seeds", type=int, default=12)
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--crowd-agents", type=int, default=19)
    parser.add_argument("--arena", type=float, default=C.ARENA_SIZE)
    parser.add_argument("--out", type=Path, default=Path("figures/baselines/situations.json"))
    args = parser.parse_args()

    planner = OrcaPlanner()
    seeds = range(args.seed_start, args.seed_start + args.seed_count)

    table: dict[str, dict[int, dict]] = {}
    for seed in seeds:
        try:
            table[seed] = scan(seed, args.agents, args.arena, planner)
        except AssertionError as exc:
            print(f"seed {seed}: skipped ({exc})")
            continue
        print(
            f"seed {seed}: head_on {table[seed]['head_on']:.2f}  "
            f"blind {table[seed]['blind_spot']:.2f}  "
            f"crossing {table[seed]['crossing']:.2f}  "
            f"intrusion {table[seed]['intrusion']:.2f}  "
            f"reached {table[seed]['reached']}"
        )

    crowd: dict[int, dict] = {}
    # At the packing limit some seeds cannot place every agent without an
    # initial overlap; the generator rejects those, so skip rather than fail.
    for seed in range(args.seed_start, args.seed_start + args.crowd_seeds):
        try:
            crowd[seed] = scan(seed, args.crowd_agents, args.arena, planner)
        except AssertionError as exc:
            print(f"seed {seed} (M={args.crowd_agents}): skipped ({exc})")
            continue
        print(
            f"seed {seed} (M={args.crowd_agents}): "
            f"intrusion {crowd[seed]['intrusion']:.2f}  "
            f"reached {crowd[seed]['reached']}"
        )

    def best(scored: dict[int, dict], key: str, require_reached: bool = True) -> int:
        pool = {
            s: row for s, row in scored.items()
            if row["reached"] or not require_reached
        } or scored
        return max(pool, key=lambda s: pool[s][key])

    def row(scored: dict[int, dict], key: str, agents: int) -> dict:
        seed = best(scored, key)
        return {
            "seed": seed,
            "num_agents": agents,
            "score": scored[seed][key],
            # Time of the peak signature: the baseline GIFs cut their still
            # frame here so the contact sheet shows the encounter, not the end.
            "t": scored[seed][f"{key}_t"],
        }

    crowd_seed = best(crowd, "intrusion", require_reached=False)
    chosen = {
        "head_on_pedestrian": row(table, "head_on", args.agents),
        "blind_spot_worker": row(table, "blind_spot", args.agents),
        "crossing_vehicle": row(table, "crossing", args.agents),
        # The crowd has no single encounter: anchor the still on the busiest
        # 0.5 s step instead.
        "dense_crowd": {
            "seed": crowd_seed,
            "num_agents": args.crowd_agents,
            "score": crowd[crowd_seed]["intrusion"],
            "t": crowd[crowd_seed]["intrusion_t"],
        },
    }

    print("\nchosen:")
    for name, entry in chosen.items():
        where = "" if entry["t"] is None else f", peak at t={entry['t']:.1f}s"
        print(
            f"  {name}: seed {entry['seed']} "
            f"(M={entry['num_agents']}, score {entry['score']:.2f}{where})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(chosen, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
