"""Robotarium bridge: coordinate mapping and a short end-to-end smoke run."""

from __future__ import annotations

import numpy as np
import pytest

from construction_nav.robotarium.arena import (
    BOUNDARIES,
    ROBOTARIUM_COLLISION_DIAMETER,
    RobotariumTransform,
    robotarium_scene,
)

rps = pytest.importorskip("rps.robotarium")


def test_transform_round_trip():
    t = RobotariumTransform(arena=60.0)
    pos = np.array([[0.0, 0.0], [60.0, 60.0], [15.0, 42.0]])
    back = t.to_site(t.to_robotarium(pos))
    assert np.allclose(back, pos, atol=1e-9)


def test_transform_fits_inside_bounds():
    for arena in (3.0, 60.0):
        t = RobotariumTransform(arena=arena)
        corners = np.array([[0.0, 0.0], [arena, 0.0], [0.0, arena], [arena, arena]])
        rb = t.to_robotarium(corners)
        assert np.all(rb[:, 0] >= BOUNDARIES[0])
        assert np.all(rb[:, 0] <= BOUNDARIES[1])
        assert np.all(rb[:, 1] >= BOUNDARIES[2])
        assert np.all(rb[:, 1] <= BOUNDARIES[3])


def test_transform_scale_is_uniform_and_smaller_than_arena():
    t = RobotariumTransform(arena=3.0)
    assert 0.0 < t.scale < 3.0


def test_robotarium_scene_radius_floor():
    scene = robotarium_scene(seed=0, num_agents=5)
    transform = RobotariumTransform(scene.arena_size)
    min_site_radius = ROBOTARIUM_COLLISION_DIAMETER / transform.scale
    for agent in scene.agents:
        # Radii are stored in site metres; in Robotarium metres they must be at
        # least the simulator's robot diameter.
        assert agent.D_S * transform.scale >= ROBOTARIUM_COLLISION_DIAMETER - 1e-9


def test_rank_seeds_prefers_safe_close_encounters(monkeypatch):
    """Seeds that reach safely and quickly should sort first; exclusion works."""
    from construction_nav.robotarium import demo

    fake = {
        0: {"seed": 0, "reached": False, "collisions": 3, "min_encounter": 0.2,
            "travel_time": 100.0},
        1: {"seed": 1, "reached": True, "collisions": 0, "min_encounter": 0.5,
            "travel_time": 60.0},
        2: {"seed": 2, "reached": True, "collisions": 0, "min_encounter": 0.3,
            "travel_time": 40.0},
        3: {"seed": 3, "reached": True, "collisions": 4, "min_encounter": 0.1,
            "travel_time": 80.0},
    }
    monkeypatch.setattr(
        demo, "site_trial",
        lambda scene, encounter="any", steps=4000, planner=None: fake[scene.seed],
    )
    ranked = demo.rank_seeds("any", num_agents=5, n_seeds=4)
    # Safe seeds first (then fastest), then unsafe by collision count.
    assert [s["seed"] for s in ranked] == [2, 1, 0, 3]

    # Excluding a seed removes it from consideration (distinct demo cases).
    ranked = demo.rank_seeds("any", num_agents=5, n_seeds=4, exclude={2})
    assert [s["seed"] for s in ranked] == [1, 0, 3]


@pytest.mark.slow
def test_robotarium_trial_smoke():
    from construction_nav.planner import FixedRadiusPlanner
    from construction_nav.robotarium import run_robotarium_trial

    scene = robotarium_scene(seed=0, num_agents=5)
    result = run_robotarium_trial(
        scene,
        planner=FixedRadiusPlanner(use_oracle=False),
        max_steps=1500,
        barrier_mode="all",
    )
    assert result.steps > 1
    assert result.site_trajectories.shape[0] == len(scene.agents) + 1
    # The Robotarium's own pose validation must never fire (all robots stay put
    # inside the arena when the barrier certificate is enabled for everyone).
    assert result.validation.get("robots_outside_boundaries", 0) == 0
