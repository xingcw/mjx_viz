"""Render a per-step trajectory dict as a self-contained Plotly HTML.

The dashboard's view toggle expects a ``<rollout>_traj.html`` companion next to
each ``<rollout>.html``. This module produces that companion: drop ``signals``
in (any 1-D or 2-D ``(T,)``/``(T, k)`` numpy arrays keyed by name) and get back
an HTML string ready to write next to the brax viewer.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np


_AXIS_LABELS = ("x", "y", "z", "w")


def render_traj_html(
    signals: Mapping[str, np.ndarray],
    dt: float = 1.0,
    title: Optional[str] = None,
    height: int = 900,
    component_labels: Optional[Sequence[str]] = None,
) -> str:
    """Return a standalone Plotly HTML page with one subplot row per signal.

    Args:
        signals: mapping from name -> array. 1-D arrays render as a single trace;
            2-D ``(T, k)`` arrays render one trace per component, labelled with
            ``component_labels`` (defaults to ``x, y, z, w`` then numeric).
        dt: seconds per step. The x-axis is ``np.arange(T) * dt``.
        title: figure title; defaults to ``"trajectory"``.
        height: total figure height in px (split evenly across rows).
        component_labels: override per-column labels for 2-D signals.

    Returns:
        A self-contained HTML string with Plotly loaded from CDN.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError(
            "mjx_viz.plot_render requires plotly. Install with "
            "`pip install mjx-viz[plots]` or `pip install plotly`."
        ) from e

    items = [(name, np.asarray(arr)) for name, arr in signals.items() if arr is not None]
    if not items:
        raise ValueError("render_traj_html requires at least one signal")

    rows = len(items)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=[name for name, _ in items],
    )
    labels = tuple(component_labels) if component_labels is not None else _AXIS_LABELS

    for i, (name, arr) in enumerate(items, start=1):
        if arr.ndim == 1:
            t = np.arange(arr.shape[0]) * dt
            fig.add_trace(
                go.Scatter(x=t, y=arr, mode="lines", name=name, showlegend=False),
                row=i, col=1,
            )
        elif arr.ndim == 2:
            t = np.arange(arr.shape[0]) * dt
            for k in range(arr.shape[1]):
                lbl = labels[k] if k < len(labels) else str(k)
                fig.add_trace(
                    go.Scatter(x=t, y=arr[:, k], mode="lines",
                               name=f"{name}_{lbl}", legendgroup=name,
                               showlegend=(i == 1)),
                    row=i, col=1,
                )
        else:
            raise ValueError(f"signal {name!r} has unsupported shape {arr.shape}")

        fig.update_yaxes(title_text=name, row=i, col=1)

    fig.update_xaxes(title_text="time (s)", row=rows, col=1)
    fig.update_layout(
        title=title or "trajectory",
        height=height,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
        template="plotly_dark",
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=True)


def write_traj_html(
    path: Union[str, "os.PathLike[str]"],
    signals: Mapping[str, np.ndarray],
    **kwargs,
) -> None:
    """Convenience: render and write to disk."""
    import os  # noqa: F401

    html = render_traj_html(signals, **kwargs)
    with open(path, "w") as f:
        f.write(html)
