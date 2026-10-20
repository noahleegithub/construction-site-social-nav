"""Rendering invariants: body scale and non-mutation."""

from __future__ import annotations

import numpy as np

from construction_nav import config as C
from construction_nav.env import generate_scene
from construction_nav.render import body_scale, draw_agent
from construction_nav.render_anim import animate_trial
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
