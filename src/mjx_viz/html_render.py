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
    time_scale: float = 0.05,
    overlay_text: Optional[str] = None,
) -> Optional[str]:
    """Return a standalone HTML string showing the trajectory in brax's WebGL viewer.

    Args:
        mj_model: mujoco.MjModel. Must be the same model the rollout was run on.
        xpos: (T, num_links, 3) link positions in world frame. The world body
            (index 0) must already be excluded.
        xquat: (T, num_links, 4) link quaternions (wxyz), world body excluded.
        height: viewer height as a CSS value. Pass an int for pixels (e.g. 480)
            or a CSS string (e.g. "100vh", "80%"). Default "100vh" fills the
            containing iframe / tab.
        max_frames: subsample trajectory to at most this many frames so the
            generated HTML stays reasonably small.
        time_scale: initial playback speed multiplier. 1.0 plays at sim-rate;
            smaller values slow down the animation. The user can still override
            this live via the viewer's "Trajectory" panel.

    Returns:
        HTML string, or None if the trajectory is empty or rendering fails.
    """
    # brax template uses jinja `{{height | default('100vh', true)}}` with no
    # unit suffix — ints render as unitless CSS (invalid). Convert to "<N>px".
    if isinstance(height, (int, np.integer)):
        height = f"{int(height)}px"
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

    # Patch initial playback timeScale. The brax template contains exactly one
    # line `var viewer = new Viewer(domElement, system);`; we append a small
    # snippet that sets the animator's mixer.timeScale after load() has run.
    if time_scale is not None:
        target = "var viewer = new Viewer(domElement, system);"
        injection = (
            target
            + "\n      if (viewer.animator && viewer.animator.mixer) {"
            + f" viewer.animator.mixer.timeScale = {float(time_scale)};"
            + " }"
        )
        html = html.replace(target, injection, 1)

    # Inject a fixed-position overlay with per-episode stats.
    if overlay_text:
        # Minimal HTML escape on the text; avoid injecting raw user strings.
        safe = (
            overlay_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        overlay = (
            '<div style="position:fixed;top:12px;left:12px;z-index:9999;'
            "padding:6px 12px;background:rgba(24,27,36,0.85);color:#e0e0e6;"
            "font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;"
            'border:1px solid #2a2e3a;border-radius:6px;pointer-events:none;">'
            f"{safe}</div>"
        )
        html = html.replace("</body>", overlay + "</body>", 1)

    return html


def write_html(path: str, html: str) -> None:
    """Save an HTML string to disk."""
    with open(path, "w") as f:
        f.write(html)
