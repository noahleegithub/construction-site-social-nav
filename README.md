# Heterogeneous Trajectory Prediction and Social Navigation for Construction Site Robots

CSCE 689 course project. One robot navigates a **graph-structured construction
worksite** toward a goal while `5 <= M <= 19` heterogeneous agents work at and
travel between stations. Agents are **humans** (unicycle dynamics) or **vehicles**
(bicycle dynamics); they alternate between working and moving, have vision cones
and blindspots, and avoid one another with **ORCA** reciprocal collision
avoidance. The robot
observes agent histories, forecasts their futures
(ADE / FDE), plans a socially compliant path, and is evaluated both in the native
simulation and on the **Robotarium** simulator.

The design follows the proposal's sections:

| Section | Topic | Where |
|---|---|---|
| 2 | Problem formulation, Eqs. (1)–(3) | `metrics.py`, `config.py` |
| 3 | Environment simulation | `env.py`, `entities.py`, `graph.py`, `dynamics.py` |
| 4 | Joint predictor + planner (MotionDiffuser-style) | `predict/`, `planner/joint.py` |
| 5 | Evaluation, baselines, ablations | `evaluate.py`, `trials.py`, `render_metrics.py`, `scripts/` |
| — | Robotarium validation | `robotarium/` |

## Status

Development proceeds in three vertical slices. **All three are complete**:

- Scale-invariant worksite generator (graph, workstations, 5–19 agents).
- Heterogeneous closed-loop simulation (unicycle humans, bicycle vehicles,
  working/moving state machine, ORCA reciprocal avoidance, FOV/blindspot sensing).
- Metrics: ADE / FDE (Eq. 1), barrier terms `h_S`/`h_C` (Eq. 2), cost `J` (Eq. 3).
- Predictors: **constant velocity**, a **reduced Trajectron++** (RNN + discrete
  CVAE), and a **MotionDiffuser-style conditional diffusion** model, both jointly
  over all agents *and* the robot.
- **Joint predictor + planner** with multimodal prediction, safety constraints
  projected **inside reverse diffusion**, a per-type `D_S`/`D_C` hard-constraint
  filter, and the `oracle` perfect-knowledge variant.
- Open-loop evaluation harness and a **benchmark of ADE/FDE vs horizon `H` and
  agent count `M`**.
- Randomized-trial **planner evaluation**: comparison against the fixed-radius
  baseline, a `lambda_1` sweep, and a failure/hard-case study.
- Robotarium bridge (all entities as Robotarium robots) and social-navigation demos,
  with either the fixed-radius or the joint planner.
- Static + animated rendering, and a test suite.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
# The Robotarium simulator ships from GitHub, not PyPI:
.venv/bin/pip install "git+https://github.com/robotarium/robotarium_python_simulator"
```

Learned predictors need PyTorch:

```bash
.venv/bin/pip install -e ".[ml]"        # CPU wheel, or:
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Quick start

```bash
# Render a static worksite figure (writes figures/scene.{png,pdf}, scene_arena.png)
python -m construction_nav scene --seed 0 --agents 7

# Run one closed-loop trial and plot the paths (add --animate for a GIF)
python -m construction_nav trial --seed 0 --agents 7 --planner joint --animate

# MotionDiffuser-style schematic figures
python -m construction_nav predictions --seed 0

# Train the learned predictors and plot their training curves
python scripts/train.py --agents 7 --epochs 80

# Benchmark ADE/FDE vs horizon H and agent count M (writes figures/benchmark*.png)
python scripts/benchmark.py --agents 5 7 9 --epochs 80

# Evaluate the planners: comparison, lambda_1 sweep, failure cases
python scripts/evaluate_planners.py --agents 7 --trials 16

# Social-navigation demos on the Robotarium (--planner joint for the learned policy)
python -m construction_nav robotarium --agents 5 --planner joint --json figures/robotarium_demo.json

# Regenerate every static figure at once
python scripts/make_figures.py --out-dir figures --animate
```

## Environment

A worksite is generated from a seed by `generate_scene`:

- **Junctions** — neutral interior vertices, connected by a shared corridor mesh
  (a Euclidean spanning tree plus extra loop edges).
- **Workstations** — degree-1 leaves (human or vehicle) attached by a single
  spoke. They are sampled uniformly from a **peripheral square ring** (the whole
  arena outside a central disk, including the corners), so the walkable network
  spans the site instead of a thin annulus. Human corridors are narrow (3 m at
  60 m scale), vehicle corridors wide (9 m), and mesh corridors shared. The
  number of workstations grows with the agent count so every agent gets its own
  start station.
- **Agents** — `5 <= M <= 19` (6–20 entities including the robot), each holding
  two same-kind workstations and cycling `working -> moving -> working`. Humans
  are unicycles (`v`, `omega`); vehicles are bicycles (`v`, `delta`,
  `omega = v tan(delta) / L`).
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

Agents follow their route by pure pursuit, then project the preferred velocity
through **ORCA** (optimal reciprocal collision avoidance). Every agent builds
half-plane constraints against all neighbours within `orca_neighbor_dist` and
picks the velocity closest to its preferred one that satisfies them, by exact
enumeration of the feasible polygon's vertices. Because the recovery vector is
split 50/50, avoidance is reciprocal rather than one-sided.

Three details make ORCA work with nonholonomic bodies:

- **Two-phase update.** `step()` freezes a snapshot (poses, velocities, radii)
  before commanding anyone; otherwise later agents would react to earlier
  agents' *new* velocities and reciprocity would break.
- **Reachable-velocity scaling.** A unicycle can only move along its heading, so
  a velocity the steering cannot realise within one step is scaled down instead
  of being driven at full speed — this is what stops a saturated-steering
  vehicle creeping into the obstacle it is turning away from.
- **Coarse-step margin.** At the 0.5 s timestep a vehicle covers 2 m per step,
  more than its own `D_S`, so `ORCA_SAFETY_MARGIN_REF` (0.5 m) keeps the
  combined radius above one step of travel.

**Humans yield to the robot when it is inside their field of view; vehicles hold
their path** (Section 3), while every agent carries a rear *blindspot* cone.
Vehicle corridors are wide (9 m) and mesh corridors shared.

| Quantity | Value (60 m reference) |
|---|---|
| ORCA time horizon | 2.0 s (not scaled) |
| Neighbour radius | 6 m |
| Combined radius | `max(D_S_i, D_S_j) + 0.5 m` |
| Responsibility split | 50 / 50 |

## Metrics (proposal Eqs. 1–3)

- `ade(pred, gt)` / `fde(pred, gt)` — mean / final displacement error.
- `min_ade` / `min_fde` — best-of-K over multimodal samples (K=1 for unimodal).
- `h_S = ||p_R - p_m||^2 - D_S^2` (hard, `>= 0`), `h_C = ||p_R - p_m||^2 - D_C^2`
  (safety), per Eq. (2).
- `J = T_travel + lambda_1 * sum_m integral max(0, -h_C^m) dt + lambda_2 * T_freeze`
  (Eq. 3) via `metrics.navigation_cost` / `summarize_trial`.

## Predictors

All learned predictors consume the shared `Observation` (agent histories, poses,
kinds, states) and return `PredictorOutput(mean, samples, weights)` for the `M`
agents over a horizon `H`. They are **joint** over `N = M + 1` entities — the
agents *and* the robot — and **scale-free**: positions are normalised by the arena,
so one model covers the 60 m site and the miniature Robotarium site.

### Dataset (`data/`)

`data.generate` rolls out randomized scenes under the nominal policy and slices
each rollout into observation windows for all `N` entities:

```
hist    (N, T, 2)   past positions / arena
future  (N, H, 2)   true future positions / arena
kind    (N,)        human / vehicle / robot
state   (N,)        working / moving
```

Train/val/test datasets use disjoint seeds, and generation is pure `numpy` so the
core package keeps working without PyTorch.

### Diffusion (`predict/diffusion.py`)

A deliberately small MotionDiffuser: a GRU history encoder (history + last
position + velocity + kind/state embeddings) feeding a transformer block over the
`N` entity tokens — that self-attention is what makes the forecast *joint* — and a
DDIM sampler returning `K` samples. Three design choices matter on a CPU budget:

- **x0-prediction** (not noise-prediction), which is far more stable for a small
  model and few sampling steps;
- a **cosine noise schedule**, so the final step really is pure noise;
- the network diffuses the **residual over a constant-velocity prior**, so it
  starts from the CV baseline's accuracy and spends its capacity on the
  interaction-driven corrections that make the forecast multimodal.

### Reduced Trajectron++ (`predict/trajectron.py`)

An in-repo, much smaller Trajectron++: the same joint encoder, a **discrete
conditional VAE** (categorical latent over interaction modes) trained with
reconstruction + categorical KL, and a per-entity decoder. Inference returns one
trajectory per mode with the prior's mode probabilities as weights.

## Planner: joint predictor + planner

`planner/joint.py` implements the proposal's coupled predictor+planner. At every
step it:

1. **Predicts** the agents with the multimodal predictor (constant velocity,
   reduced Trajectron++, or diffusion). In `safety_diffusion` mode the diffusion
   sampler is *conditioned* on the robot's intended route plan — the joint sample
   is inpainted so the robot row is pinned and the agents are generated
   conditioned on it — and the robot's rows are projected onto the hard-constraint
   safe set **inside reverse diffusion**. Safety is therefore injected into the
   sampling process rather than checked only afterwards, which is the
   MotionDiffuser-style coupling.
2. **Proposes** a small set of local manoeuvres by integrating the unicycle along
   the graph route at several speed scales and heading offsets.
3. **Scores** each manoeuvre with the Eq. (3) ingredients against every predicted
   mode and the *per-type* `D_S` / `D_C`: a progress term (travel-time proxy),
   `lambda_1 * sum max(0, -h_C)`, `lambda_2 * T_freeze`, and a hard-constraint
   feasibility test `h_S >= 0` in every mode with non-zero weight.
4. **Filters** the chosen command through a reactive clearance guard that scales
   the speed to zero as the clearance to any agent vanishes, enforcing the hard
   constraint against the *real* state even when the forecast is wrong. If no
   manoeuvre is feasible it maximises clearance instead of progress.

`oracle=True` replaces the predictor with perfect future knowledge, giving the
proposal's perfect-knowledge planner; combined with the `predictor=` variants this
is exactly the "predictor quality vs planner" ablation.

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

## Evaluation

> **Numbers below predate the ORCA switch.** They were measured with the
> social-force agents and the previous layout, and the checkpoints were trained
> on that distribution. Re-running `scripts/train.py`, `scripts/benchmark.py` and
> `scripts/evaluate_planners.py` is the next step and will refresh these tables,
> `figures/*.json` and the plotted figures.

`evaluate.evaluate_predictor` scores any predictor on closed-loop rollouts:
the environment is advanced under the nominal policy, the predictor is asked to
forecast at each step, and its forecast is compared with the positions the rollout
actually visited (open-loop evaluation, exactly how the training data was built).
Multimodal predictors are scored with best-of-`K` (`min_ade` / `min_fde`).

`scripts/benchmark.py` sweeps the **forecast horizon `H`** and the **agent count
`M`**, training one diffusion and one Trajectron++ model per `M` and scoring both
against the constant-velocity baseline. It writes `figures/benchmark_horizon.png`,
`figures/benchmark_agents.png`, and `figures/benchmark.json`.

Representative numbers (held-out test scenes, best-of-`K`, metres, `M = 7`):

| model | ADE `H=1` | ADE `H=4` | ADE `H=16` | FDE `H=16` |
|---|---|---|---|---|
| constant velocity | 0.15 | 0.48 | 2.71 | 5.97 |
| trajectron++ (reduced) | 0.40 | 0.78 | 2.68 | 5.30 |
| diffusion (`K=6`) | 0.58 | 0.62 | **2.28** | **4.59** |

Two trends stand out. Constant velocity wins at **short** horizons (the motion is
smooth and nearly linear at `dt = 0.5 s`), while the **joint** models win at long
horizons and as the crowd grows `M = 5 -> 9`, where interaction structure matters
most. The diffusion model is best or tied-best at `H = 16` for every `M` tested,
and improves relative to the baselines as `M` increases.

### Planner evaluation

`scripts/evaluate_planners.py` runs each planner on the same randomized sites
(out-of-sample seeds, `M = 7`) and reports Eq. (3). 16 trials each:

| planner | success | collision steps | `J` | travel (s) | intrusion |
|---|---|---|---|---|---|
| fixed-radius (oracle) | 100% | 0 | 36.59 | 34.7 | 1.91 |
| joint (oracle) | 100% | 0 | **35.28** | 35.1 | **0.16** |
| joint (constant velocity) | 100% | 4 | 36.61 | 35.3 | 1.27 |
| joint (trajectron++, reduced) | 100% | 2 | 35.85 | 35.1 | 0.75 |
| joint (diffusion) | 100% | 1 | 35.89 | 35.1 | 0.67 |
| joint (diffusion + safety diffusion) | 100% | 1 | 35.86 | 35.0 | 0.71 |

Every planner reaches the goal, so the differences are all in safety and cost.
The **predictor-quality ablation** is the three `joint (learned)` rows: a weaker
forecast (constant velocity) collides and intrudes more, and the ordering
constant-velocity → Trajectron++ → diffusion matches the long-horizon (`H = 16`)
ADE ordering of the predictors, i.e. forecast quality transfers to planner
safety. The **planner ablation** is the perfect-knowledge `joint (oracle)` row:
the learned-diffusion planner matches the oracle's cost while the oracle keeps
the largest margin. The fixed-radius baseline reaches the goal but with ~3x the
intrusion of the prediction-based planner.

Sweeping `lambda_1` (diffusion planner, with and without the reactive safety
filter) makes the trade-off explicit: going from `lambda_1 = 0` to `5` cuts the
intrusion integral from `0.77` to `0.40` and removes the remaining collision
steps, at the price of `J` rising from `35.5` to `37.3` (about +1.5 s of travel
time) -- the soft social cost is bought with time, as Eq. (3) intends.

## Robotarium

`construction_nav.robotarium` maps the worksite into the Robotarium arena with a
single uniform scale and instantiates **the robot and all `M` agents as
Robotarium unicycle robots**. Agents are driven by the same route-following and
ORCA avoidance controller; vehicles are represented by their curvature-equivalent
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
the Robotarium validation for each case. Pass `--planner joint` to drive the robot
with the joint planner instead of the fixed-radius baseline.

## Figures

`figures/` is regenerated with `scripts/make_figures.py`,
`scripts/train.py`, `scripts/benchmark.py`, `scripts/evaluate_planners.py` and
`python -m construction_nav robotarium`:

- `scene.{png,pdf}` — static worksite with zoning, sensing, and routes.
- `scene_arena.png` — arena-only backdrop used behind overlays.
- `pipeline.{png,pdf}` — MotionDiffuser pipeline schematic.
- `diffusion_sequence.{png,pdf}` — reverse-diffusion schematic.
- `trial.{png,pdf,gif}` — closed-loop trial paths / animation.
- `benchmark_horizon.{png,pdf}` — ADE/FDE vs horizon for every model.
- `benchmark_agents.{png,pdf}` — ADE/FDE vs agent count `M`.
- `training_curves.{png,pdf}` — diffusion and Trajectron++ loss curves.
- `planner_comparison.{png,pdf}` — success, cost `J`, intrusion, travel time.
- `lambda_sweep.{png,pdf}` — cost / intrusion / travel time vs `lambda_1`.
- `failure_cases.{png,pdf}` — failure / hard cases for the joint planner.
- `demo_*.{png,pdf,gif}` — Robotarium social-navigation cases.

## Tests

```bash
python -m pytest -m "not slow"   # fast unit/integration tests
python -m pytest -m slow         # end-to-end Robotarium smoke test
```

Coverage includes dynamics integration, scene invariants and scale invariance,
environment determinism and the working/moving state machine, ORCA reciprocity
and clearance, FOV-gated robot avoidance, ADE/FDE/barrier/cost closed forms,
predictor shapes and velocity
estimation, dataset generation and round-trip, the learned predictors (train,
predict, save/load) and the CV prior, planner goal-reaching and static-obstacle
avoidance, the joint planner (valid commands, oracle goal-reaching, the reactive
guard, safety-in-reverse-diffusion projection) and the trial harness, Robotarium
coordinate mapping and demo seed ranking, and the Robotarium collision/boundary
validation. Tests that need PyTorch skip cleanly without it.

## Layout

```
construction_nav/
  config.py         layout + physical constants (scale-invariant via Layout)
  entities.py       Node, Highway, Agent, Robot, Scene
  graph.py          mesh construction, routing, polyline geometry
  dynamics.py       unicycle / bicycle integrators
  orca.py           ORCA half-planes and the 2-D velocity solver
  env.py            generate_scene + ConstructionEnv (closed loop)
  metrics.py        ADE / FDE / barriers / cost J
  evaluate.py       open-loop predictor evaluation (ADE/FDE vs H, M)
  trials.py         randomized-trial planner evaluation (cost J, ablations)
  render.py         static scene rendering
  render_anim.py    rollout animation (GIF / MP4)
  render_trials.py  trajectory plots and metric charts
  render_metrics.py benchmark / planner charts and training curves
  render_predictions.py  MotionDiffuser-style schematics
  data/             rollout dataset generation + torch Dataset wrappers
  predict/          predictor interface, constant velocity, diffusion, Trajectron++
  planner/          planner interface, fixed-radius baseline, joint planner
  robotarium/       arena mapping, bridge, demos
  cli.py            `python -m construction_nav ...`
scripts/            train / benchmark / planner-eval / figure / demo entry points
tests/              pytest suite
checkpoints/        trained model weights (git-ignored)
```

## Roadmap

The three vertical slices in the proposal are implemented. Natural next steps,
deliberately left out of the course timeline:

- Full-scale MotionDiffuser (per-agent transformers, latent maps) and the optional
  speculative-decoding experiment.
- A CMOT->BEV map encoder so the robot reasons over the raw worksite image.
- Trajectron++'s graph attention and depthwise encoders for a faithful baseline.
- On-hardware Robotarium deployment (the bridge already runs the same loop).

## Notes and limitations

- `D_S` is a collision radius and `D_C` a safety radius; the barrier terms use
  squared distances per Eq. (2). ORCA uses `max(D_S_i, D_S_j)` as the pair's
  combined radius, which equals Eq. (2)'s `D_S` for human–robot pairs and is
  conservative (keeps the vehicle's larger radius) for vehicles.
- The environment no longer has circular high-risk zones; "high-risk" is the
  off-highway ground shown by the hazard hatch. Eq. (3) was never a function of
  the zones, so the cost model is unchanged.
- Agent counts run 5–19 non-robot agents (6–20 entities). At the top of that
  range the peripheral ring is densely packed, so workstation separations relax
  and the spawn clearances become soft (never below `D_S`).
- The 0.5 s timestep is coarse relative to vehicle speed (2 m per step). ORCA's
  0.5 m safety margin absorbs this at the reference scale; a few brief
  agent–agent overlaps remain possible at `M = 19`.
- ORCA is solved in velocity space and then projected onto the unicycle/bicycle
  kinematics, so its guarantee holds for the commanded velocity, not exactly for
  the integrated pose; the reachable-velocity scaling keeps the gap small.
- The Robotarium demo is a schematic at hardware scale; its collision notion is
  the GRITSbot body, which is larger than the scaled agents, so the site-frame
  metrics (Eq. 2/3) are the authoritative safety measures.
- Full-scale MotionDiffuser replication (and the optional speculative-decoding
  experiment) is out of scope for the course timeline; the diffusion model is
  deliberately small and CPU-trainable.
- Constant velocity is a strong baseline at short horizons on this data (smooth,
  near-constant-velocity motion at `dt = 0.5 s`), so the learned models are most
  useful at longer horizons and for the multimodal interaction cases the planner
  actually consumes. The benchmark reports both regimes rather than cherry-picking.
- The reduced Trajectron++ is a like-for-like multimodal baseline, not a faithful
  reproduction of the published architecture (no graph attention or depthwise
  encoders).
- The joint planner's hard-constraint test is evaluated against the *predicted*
  modes (an expected-value `h_S >= 0`), with the reactive clearance guard covering
  the true state; a safety filter over measured state (a true control barrier
  function) would harden this further.
- On the planner test set the joint planner never outright failed (timeout or
  sustained collision), so `figures/failure_cases.png` shows the **hardest** cases
  -- the trials with the largest intrusion, where an agent cut across the robot's
  route -- rather than catastrophic failures. Safety-diffusion conditioning
  changed little on these sites because the manoeuvre filter already dominates;
  it is most useful in denser scenes where the nominal plan itself is unsafe.
