"""Shared pytest fixtures.

Scenes are cheap to generate, and several tests mutate agent state, so the
fixtures are function-scoped to keep tests independent.
"""

from __future__ import annotations

import pytest

from construction_nav.env import generate_scene


@pytest.fixture
def scene():
    return generate_scene(seed=0, num_agents=7)


@pytest.fixture
def mini_scene():
    from construction_nav.robotarium import robotarium_scene

    return robotarium_scene(seed=0, num_agents=5)
