# Heterogeneous Trajectory Prediction and Social Navigation for Construction Site Robots

CSCE 689 course project. One robot navigates a **graph-structured construction
worksite** toward a goal while `M >= 5` heterogeneous agents work at and travel
between stations. Agents are **humans** (unicycle dynamics) or **vehicles**
(bicycle dynamics); they alternate between working and moving, have vision cones
and blindspots, avoid one another with a **social-force model**, and skirt static
**high-risk zones**. The robot observes agent histories, forecasts their futures
(ADE / FDE), plans a socially compliant path, and is evaluated both in the native
simulation and on the **Robotarium** simulator.

The design follows the proposal's sections:

| Section | Topic | Where |
|---|---|---|
| 2 | Problem formulation, Eqs. (1)–(3) | `metrics.py`, `config.py` |
| 3 | Environment simulation | `env.py`, `entities.py`, `graph.py`, `zones.py`, `dynamics.py` |
| 4 | Joint predictor + planner (MotionDiffuser-style) | `predict/`, `planner/` |
| 5 | Evaluation, baselines, ablations | `evaluate` CLI (Slice 3), `render_trials.py` |
| — | Robotarium validation | `robotarium/` |

## Status

Development proceeds in three vertical slices. **Slice 1 is complete**:

- Scale-invariant worksite generator (graph, workstations, high-risk zones).
- Heterogeneous closed-loop simulation (unicycle humans, bicycle vehicles,
  working/moving state machine, social forces, FOV/blindspot sensing).
- Metrics: ADE / FDE (Eq. 1), barrier terms `h_S`/`h_C` (Eq. 2), cost `J` (Eq. 3).
- Predictor interface + **constant-velocity** baseline.
- Planner interface + **fixed-radius (perfect-knowledge)** baseline.
- Robotarium bridge (all entities as Robotarium robots) and social-navigation demos.
- Static + animated rendering, and a test suite (54 tests).

Slices 2 and 3 (learned diffusion predictor, reduced Trajectron++ baseline,
joint planner, evaluation sweeps and ablations) are described in
[Roadmap](#roadmap).

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
# The Robotarium simulator ships from GitHub, not PyPI:
.venv/bin/pip install "git+https://github.com/robotarium/robotarium_python_simulator"
```

Learned predictors (Slice 2+) additionally need PyTorch: `pip install -e ".[ml]"`.

## Quick start

```bash
# Render a static worksite figure (writes figures/scene.{png,pdf}, scene_arena.png)
python -m construction_nav scene --seed 0 --agents 7

# Run one closed-loop trial and plot the paths (add --animate for a GIF)
python -m construction_nav trial --seed 0 --agents 7 --planner fixed-radius --animate

# MotionDiffuser-style schematic figures
python -m construction_nav predictions --seed 0

# Social-navigation demos on the Robotarium
python -m construction_nav robotarium --agents 5 --json figures/robotarium_demo.json

# Regenerate every static figure at once
python scripts/make_figures.py --out-dir figures --animate
```

## Environment

A worksite is generated from a seed by `generate_scene`:

- **Junctions** — neutral interior vertices, connected by a shared corridor mesh
  (a Euclidean spanning tree plus extra loop edges).
- **Workstations** — degree-1 leaves (human or vehicle) attached by a single
  spoke; human corridors are narrow (3 m at 60 m scale), vehicle corridors wide
  (9 m), and mesh corridors shared.
- **High-risk zones** — static disks placed off the graph; their hazard field
  repels agents and penalises the robot.
- **Agents** — `M >= 5`, each holding two same-kind workstations and cycling
  `working -> moving -> working`. Humans are unicycles (`v`, `omega`); vehicles
  are bicycles (`v`, `delta`, `omega = v tan(delta) / L`).
- **Robot** — a unicycle shuttling between the two farthest human workstations.

### Scale invariance

Every length and speed is derived from the arena size via
`config.Layout.for_arena(arena)`, relative to a 60 m reference site. So
`arena=60` is the reference construction site, and `arena ≈ 3` is a
geometrically and dynamically similar miniature that fits the Robotarium arena
(3.2 m x 2.0 m) with its ~0.11 m robots. Time, angles, and controller gains are
not scaled. This is what lets the same code run the large simulated experiments
and the hardware-scale Robotarium demos.

## Dynamics and social behaviour

| Quantity | Humans | Vehicles |
|---|---|---|
| Model | unicycle | bicycle (`omega = v tan(delta)/L`) |
| Nominal speed | 1.4 m/s | 4.0 m/s |
| Turn limit | 120 deg/s | 30 deg steering |
| Collision radius `D_S` | 0.4 m | 1.5 m |
| Safety radius `D_C` | 1.2 m | 3.0 m |

(Radii/speeds at the 60 m reference scale; all scale linearly with the arena.)

Agents follow their route by pure-pursuit blended with a social-force push from
nearby agents and high-risk zones, and a speed slowdown under crowding.
**Humans yield to the robot when it is inside their field of view; vehicles hold
their path** (Section 3), while every agent carries a rear *blindspot* cone.

## Metrics (proposal Eqs. 1–3)

- `ade(pred, gt)` / `fde(pred, gt)` — mean / final displacement error.
- `min_ade` / `min_fde` — best-of-K over multimodal samples (K=1 for unimodal).
- `h_S = ||p_R - p_m||^2 - D_S^2` (hard, `>= 0`), `h_C = ||p_R - p_m||^2 - D_C^2`
  (safety), per Eq. (2).
- `J = T_travel + lambda_1 * sum_m integral max(0, -h_C^m) dt + lambda_2 * T_freeze`
  (Eq. 3) via `metrics.navigation_cost` / `summarize_trial`.

## Baselines

- **Constant velocity** (`predict.ConstantVelocityPredictor`) — finite-difference
  velocity extrapolated over the horizon.
- **Fixed-radius, perfect knowledge** (`planner.FixedRadiusPlanner`) — the robot
  follows its graph route by pure pursuit and, for each agent, adds a repulsive
  velocity for the closest point of that agent's true future trajectory (from an
  oracle rollout) inside a fixed radius `D_C`; repulsions are scaled by the
  robot's nominal speed so behaviour is identical at every arena scale. This is
  the planner baseline that isolates the value of the learned predictor plus
  planner. (`use_oracle=False` swaps the oracle for the constant-velocity
  predictor.)

## Robotarium

`construction_nav.robotarium` maps the worksite into the Robotarium arena with a
single uniform scale and instantiates **the robot and all `M` agents as
Robotarium unicycle robots**. Agents are driven by the same route-following and
social-force controller; vehicles are represented by their curvature-equivalent
turn rate `omega = v tan(delta)/L`. Two approximations are documented:

1. The Robotarium only simulates unicycles, so bicycle vehicles are reduced to a
   unicycle with the matching instantaneous turn rate.
2. The Robotarium's robots are ~0.11 m in diameter and cannot shrink, so a
   miniature site (`arena = ROBOTARIUM_ARENA = 3 m`) is used and every agent's
   collision radius is *floored at the Robotarium's robot diameter* in site
   units (`robotarium_scene`), with safety radii keeping their reference
   `D_C / D_S` ratio. Without the floor the scaled agents would be far smaller
   than the physical robots and the simulator's collision checks would fire
   constantly.

`run_robotarium_trial` runs the loop, applies the unicycle **barrier
certificate** (`barrier_mode="all"` by default, since on hardware every agent is
a physical robot), and reports both the site-frame metrics and the Robotarium's
own `debug()` validation counters.

Demo cases are made reproducible and meaningful rather than arbitrary: seeds are
**ranked** by a cheap site-frame trial (goal reached, zero collisions, closest
encounter of the requested kind) and then confirmed on the Robotarium, taking the
first seed that completes safely. The shipped `figures/robotarium_demo.json`
records the chosen seed, the minimum encounter distance, the site metrics, and
the Robotarium validation for each case.

## Figures

`figures/` is regenerated with `scripts/make_figures.py` and
`scripts/robotarium_demo.py`:

- `scene.{png,pdf}` — static worksite with zoning, sensing, and routes.
- `scene_arena.png` — arena-only backdrop used behind overlays.
- `pipeline.{png,pdf}` — MotionDiffuser pipeline schematic.
- `diffusion_sequence.{png,pdf}` — reverse-diffusion schematic.
- `trial.{png,pdf,gif}` — closed-loop trial paths / animation.
- `demo_*.{png,pdf,gif}` — Robotarium social-navigation cases.

## Tests

```bash
python -m pytest -m "not slow"   # fast unit/integration tests
python -m pytest -m slow         # end-to-end Robotarium smoke test
```

Coverage includes dynamics integration, scene invariants and scale invariance,
environment determinism and the working/moving state machine, social-force
behaviour, ADE/FDE/barrier/cost closed forms, predictor shapes and velocity
estimation, planner goal-reaching and static-obstacle avoidance, Robotarium
coordinate mapping and demo seed ranking, and the Robotarium collision/boundary
validation.

## Layout

```
construction_nav/
  config.py         layout + physical constants (scale-invariant via Layout)
  entities.py       Node, Highway, Zone, Agent, Robot, Scene
  graph.py          mesh construction, routing, polyline geometry
  dynamics.py       unicycle / bicycle integrators
  zones.py          high-risk zones and hazard potentials
  env.py            generate_scene + ConstructionEnv (closed loop)
  metrics.py        ADE / FDE / barriers / cost J
  render.py         static scene rendering
  render_anim.py    rollout animation (GIF / MP4)
  render_trials.py  trajectory plots and metric charts
  render_predictions.py  MotionDiffuser-style schematics
  predict/          predictor interface + constant-velocity baseline
  planner/          planner interface + fixed-radius baseline
  robotarium/       arena mapping, bridge, demos
  cli.py            `python -m construction_nav ...`
scripts/            figure/demo entry points
tests/              pytest suite
```

## Roadmap

**Slice 2 — predictors.** Diffusion dataset from rollouts; a scaled-down
PyTorch conditional diffusion predictor (MotionDiffuser-style, jointly over all
agents including the robot) with real ADE/FDE vs horizon `H` and agent count `M`;
a reduced in-repo Trajectron++ (RNN + discrete CVAE) baseline; training and
benchmark scripts.

**Slice 3 — joint planner + evaluation.** A joint predictor+planner with safety
constraints injected into reverse diffusion; per-type `D_S`/`D_C` hard
constraints; randomized-trial sweeps (ADE/FDE vs `H`, `M`), ablations (predictor
quality vs planner; `lambda_1` sweep), failure cases, and the final Robotarium
social-navigation cases.

## Notes and limitations

- `D_S` is a collision radius and `D_C` a safety radius; the barrier terms use
  squared distances per Eq. (2).
- The Robotarium demo is a schematic at hardware scale; its collision notion is
  the GRITSbot body, which is larger than the scaled agents, so the site-frame
  metrics (Eq. 2/3) are the authoritative safety measures.
- Full-scale MotionDiffuser replication (and the optional speculative-decoding
  experiment) is out of scope for the course timeline; the diffusion model is
  deliberately small and CPU-trainable.
