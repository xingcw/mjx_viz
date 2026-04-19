"""Build interactive brax HTML viewers from per-step link pose trajectories."""

from __future__ import annotations

from typing import Optional

import numpy as np


def render_brax_html(
    mj_model,
    xpos: np.ndarray,
    xquat: np.ndarray,
    height: int = 480,
    max_frames: int = 500,
) -> Optional[str]:
    """Return a standalone HTML string showing the trajectory in brax's WebGL viewer.

    Args:
        mj_model: mujoco.MjModel. Must be the same model the rollout was run on.
        xpos: (T, num_links, 3) link positions in world frame. The world body
            (index 0) must already be excluded.
        xquat: (T, num_links, 4) link quaternions (wxyz), world body excluded.
        height: viewer height in pixels.
        max_frames: subsample trajectory to at most this many frames so the
            generated HTML stays reasonably small.

    Returns:
        HTML string, or None if the trajectory is empty or rendering fails.
    """
    if len(xpos) == 0:
        return None

    try:
        from brax.base import Transform
        from brax.io.html import render as brax_html_render
        from brax.io.mjcf import load_model as brax_load_model
    except ImportError as e:
        raise ImportError(
            "mjx_viz.html_render requires brax. Install with `pip install brax`."
        ) from e

    sys = brax_load_model(mj_model)

    total = len(xpos)
    step = max(1, total // max_frames)

    class _VizState:
        __slots__ = ("x",)

        def __init__(self, pos, rot):
            self.x = Transform(pos=pos, rot=rot)

    states = [_VizState(pos=xpos[i], rot=xquat[i]) for i in range(0, total, step)]

    return brax_html_render(sys, states, height=height, colab=False)


def write_html(path: str, html: str) -> None:
    """Save an HTML string to disk."""
    with open(path, "w") as f:
        f.write(html)
