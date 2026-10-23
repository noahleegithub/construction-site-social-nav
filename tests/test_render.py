"""Rendering invariants: body scale and non-mutation."""

from __future__ import annotations

import numpy as np

from construction_nav import config as C
from construction_nav.env import generate_scene
from construction_nav.render import body_scale, draw_agent
from construction_nav.render_anim import animate_comparison, animate_trial, closest_frame
from construction_nav.render_metrics import SHORT_NAMES, _short_label
from construction_nav.render_trials import plot_trial


def test_body_scale_tracks_the_arena():
    big = generate_scene(seed=0, num_agents=7, arena=60.0)
    small = generate_scene(seed=0, num_agents=7, arena=3.0)
    assert body_scale(big) == 1.0
    assert body_scale(small) == 3.0 / C.ARENA_REF


def test_agent_body_is_arena_relative():
    """A human must occupy the same fraction of the site at every scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for arena, expected_frac in ((60.0, 2 * 0.55 / 60.0), (3.0, 2 * 0.55 * 0.05 / 3.0)):
        scene = generate_scene(seed=0, num_agents=5, arena=arena)
        fig, ax = plt.subplots()
        draw_agent(ax, scene.agents[0], scale=body_scale(scene))
        circle = ax.patches[0]
        diameter = 2 * circle.get_radius()
        assert diameter / arena == expected_frac
        plt.close(fig)


def test_renderers_do_not_mutate_the_scene(tmp_path):
    scene = generate_scene(seed=0, num_agents=5, arena=3.0)
    before = np.array([a.pos for a in scene.agents])
    states = [a.state for a in scene.agents]
    history_len = [len(a.history) for a in scene.agents]

    plot_trial(scene, tmp_path, name="trial_check", steps=6, dpi=40)
    animate_trial(scene, tmp_path, name="anim_check", steps=6, fps=5, dpi=40)

    assert np.allclose(np.array([a.pos for a in scene.agents]), before)
    assert [a.state for a in scene.agents] == states
    assert [len(a.history) for a in scene.agents] == history_len


def test_short_labels_cover_every_baseline():
    """The figure axis labels must stay readable at five bars per panel."""
    from scripts.make_baseline_gifs import _planners

    for name, _planner in _planners():
        assert name in SHORT_NAMES, f"no compact label for {name!r}"
        assert "\n" in _short_label(name) or len(_short_label(name)) <= 6


def test_closest_frame_picks_the_nearest_approach():
    records = [
        {"frames": [
            {"agent_pos": np.array([[0.0, 0.0]]), "robot_pos": np.array([5.0, 0.0])},
            {"agent_pos": np.array([[0.0, 0.0]]), "robot_pos": np.array([0.5, 0.0])},
            {"agent_pos": np.array([[0.0, 0.0]]), "robot_pos": np.array([9.0, 0.0])},
        ]},
    ]
    assert closest_frame(records) == 1


def test_comparison_grid_shares_one_scene(tmp_path):
    """Every panel runs the same seed, so the panels must diverge only by policy."""
    from construction_nav.planner import OrcaPlanner

    scene = generate_scene(seed=3, num_agents=5, arena=3.0)
    before = np.array([a.pos for a in scene.agents])

    paths, summary = animate_comparison(
        scene,
        [("Path following", None), ("ORCA", OrcaPlanner())],
        out_dir=tmp_path,
        name="cmp",
        steps=8,
        fps=4,
        dpi=40,
        ncols=2,
        still=True,
        still_index=3,
    )
    assert (tmp_path / "cmp.gif").exists()
    assert (tmp_path / "cmp.png").exists()
    assert set(summary) == {"Path following", "ORCA"}
    # The caller's scene is untouched: each panel worked on its own copy.
    assert np.allclose(np.array([a.pos for a in scene.agents]), before)
    assert any(p.suffix == ".gif" for p in paths)
