"""Build interactive brax HTML viewers from per-step link pose trajectories."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


def render_brax_html(
    mj_model,
    xpos: np.ndarray,
    xquat: np.ndarray,
    height: Union[int, str] = "100vh",
    max_frames: int = 500,
    time_scale: float = 0.1,
) -> Optional[str]:
    """Return a standalone HTML string showing the trajectory in brax's WebGL viewer.

    Args:
        mj_model: mujoco.MjModel. Must be the same model the rollout was run on.
        xpos: (T, num_links, 3) link positions in world frame. The world body
            (index 0) must already be excluded.
        xquat: (T, num_links, 4) link quaternions (wxyz), world body excluded.
        height: viewer height. Pass an int (treated as pixels by brax's colab
            template) or a CSS length string like "100vh"/"100%". Default
            "100vh" makes the viewer fill its iframe; brax's non-colab template
            requires a CSS unit (a unitless int produces invalid CSS).
        max_frames: subsample trajectory to at most this many frames so the
            generated HTML stays reasonably small.
        time_scale: initial value of the viewer's Trajectory > timeScale
            slider (still adjustable in the panel). 1.0 = real-time, 0.1 = 10×
            slow-motion. Defaults to 0.1 because raw drone trajectories play
            back too fast at 1×.

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

    html = brax_html_render(sys, states, height=height, colab=False)

    # Override the THREE.AnimationMixer's default timeScale (1.0). The viewer's
    # lil-gui slider was bound *before* we ran (in Animator.load), so we also
    # have to call updateDisplay() on the timeScale controller for the slider
    # widget to reflect the new value — otherwise playback uses 0.1 but the
    # panel still reads 1.0.
    target = "var viewer = new Viewer(domElement, system);"
    inject = (
        f"{target}\n"
        f"      viewer.animator.mixer.timeScale = {float(time_scale)};\n"
        f"      viewer.gui.controllersRecursive().forEach("
        f"function (c) {{ if (c.property === 'timeScale') c.updateDisplay(); }});"
    )
    return html.replace(target, inject)


def write_html(path: str, html: str) -> None:
    """Save an HTML string to disk."""
    with open(path, "w") as f:
        f.write(html)
