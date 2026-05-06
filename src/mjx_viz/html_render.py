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
    time_scale: float = 1.0,
    frame_dt: Optional[float] = None,
    show_overlay: bool = True,
    episode_reward: Optional[float] = None,
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
        time_scale: initial value of the viewer's Trajectory > timeScale slider.
            1.0 = play back at the same wall-clock rate as the simulation
            (assuming `frame_dt` is set correctly).
        frame_dt: wall-clock seconds represented by each consecutive entry of
            `xpos`/`xquat`. Brax's viewer puts keyframes at
            `frame_idx × system.opt.timestep`, so by default a trajectory
            recorded at one frame per env-step (`step_dt = timestep × n_frames`)
            plays back `n_frames` × too fast. Pass `frame_dt` (typically the
            env's `step_dt`) and we override `system.opt.timestep` so the
            timeline matches physical time. None ⇒ keep `mj_model.opt.timestep`
            (legacy behaviour; only correct when one frame == one physics
            timestep).
        show_overlay: if True, draw a top-left HUD inside the viewer that
            shows `step N | t = X.XXX s` updated per animation frame. The
            step count is computed from the animator's clock and the
            (subsample-adjusted) per-frame dt, so it tracks scrubbing too.
        episode_reward: optional cumulative env reward for this rollout. When
            provided, the HUD shows a static summary line above the dynamic
            step/time counter: `ep <T>  return <R>` where T is the recorded
            trajectory length and R is the cumulative reward. Pass None to
            keep the legacy single-line HUD.

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

    # Build a render-only deepcopy so we can mutate it without touching the
    # caller's mj_model. Two adjustments:
    #   1. opt.timestep ← frame_dt so brax's viewer (which builds keyframe
    #      times as `frame_idx × system.opt.timestep`) tracks physical time.
    #   2. Any geom flagged as "render-invisible" via alpha == 0 has its size
    #      zeroed. Brax's WebGL viewer (visualizer/js/system.js) ignores RGBA
    #      alpha entirely, so without this, a transparent collision proxy
    #      (e.g. a `cf_col` sphere with rgba="0 0 1 0") would overdraw the
    #      real visual meshes as a solid color. Zero size collapses the
    #      geometry to a degenerate point so three.js draws nothing.
    import copy
    import numpy as np
    mj_model = copy.deepcopy(mj_model)
    if frame_dt is not None:
        mj_model.opt.timestep = float(frame_dt)
    invisible_geom_ids = np.flatnonzero(mj_model.geom_rgba[:, 3] == 0.0)
    if invisible_geom_ids.size:
        mj_model.geom_size[invisible_geom_ids] = 0.0

    sys = brax_load_model(mj_model)

    total = len(xpos)
    step  = max(1, total // max_frames)
    # The viewer treats each *kept* state as a keyframe at
    # `kept_idx × sys.opt.timestep`. We subsample by `step`, so the wall-clock
    # gap between consecutive kept frames is `step × frame_dt`. To keep
    # playback aligned to physical time we have to bake that into the keyframe
    # spacing rather than the original frame_dt.
    if frame_dt is not None and step > 1:
        sys = sys.replace(
            opt=sys.opt.replace(timestep=float(frame_dt) * step),
        )
    effective_frame_dt = (frame_dt if frame_dt is not None
                         else float(mj_model.opt.timestep))

    class _VizState:
        __slots__ = ("x",)

        def __init__(self, pos, rot):
            self.x = Transform(pos=pos, rot=rot)

    states = [_VizState(pos=xpos[i], rot=xquat[i]) for i in range(0, total, step)]

    html = brax_html_render(sys, states, height=height, colab=False)

    target = "var viewer = new Viewer(domElement, system);"
    inject_lines = [target]

    # Set initial timeScale and refresh the lil-gui slider so the panel
    # reflects the value (the slider is bound during Animator.load before
    # this code executes).
    inject_lines.append(
        f"      viewer.animator.mixer.timeScale = {float(time_scale)};"
    )
    inject_lines.append(
        "      viewer.gui.controllersRecursive().forEach("
        "function (c) { if (c.property === 'timeScale') c.updateDisplay(); });"
    )

    if show_overlay:
        # Top-left HUD: optional static summary line (ep length + cumulative
        # reward when provided) above a dynamic step + time counter. The
        # dynamic line updates each animation frame by hooking into
        # Animator.update; the static line is baked once at render time.
        ep_len_recorded = total
        per_step_dt = float(effective_frame_dt)
        if episode_reward is not None:
            summary_line_js = (
                f"'ep {ep_len_recorded}  return {float(episode_reward):.2f}'"
            )
            has_summary_js = "true"
        else:
            summary_line_js = "''"
            has_summary_js = "false"
        inject_lines.append(f"""
      (function () {{
        var stepDt      = {per_step_dt};
        var totalFr     = {ep_len_recorded};
        var summaryLine = {summary_line_js};
        var hasSummary  = {has_summary_js};
        var hud      = document.createElement('div');
        hud.style.cssText = 'position:absolute;top:8px;left:8px;z-index:9999;'
          + 'font-family:Menlo, Monaco, Consolas, "Courier New", monospace;'
          + 'font-size:13px;font-weight:600;color:#fff;'
          + 'background:rgba(0,0,0,0.55);padding:4px 8px;border-radius:4px;'
          + 'pointer-events:none;letter-spacing:0.02em;'
          + 'white-space:pre;line-height:1.35;';
        function frameText(stepIdx, tSec) {{
          var dyn = 'step ' + stepIdx + ' / ' + (totalFr - 1)
                  + '   t = ' + tSec.toFixed(3) + ' s';
          return hasSummary ? (summaryLine + '\\n' + dyn) : dyn;
        }}
        hud.textContent = frameText(0, 0);
        var parent = domElement;
        if (getComputedStyle(parent).position === 'static') {{
          parent.style.position = 'relative';
        }}
        parent.appendChild(hud);

        var anim = viewer.animator;
        var origUpdate = anim.update.bind(anim);
        anim.update = function () {{
          origUpdate();
          var t  = this.action ? this.action.time : 0;
          var st = Math.min(totalFr - 1, Math.max(0, Math.round(t / stepDt)));
          hud.textContent = frameText(st, t);
        }};
      }})();""")

    inject = "\n".join(inject_lines)
    return html.replace(target, inject)


def write_html(path: str, html: str) -> None:
    """Save an HTML string to disk."""
    with open(path, "w") as f:
        f.write(html)
