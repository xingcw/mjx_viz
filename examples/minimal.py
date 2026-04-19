"""Minimal example: run a rollout and save a brax HTML viewer.

Assumes the caller supplies:
  - env: MJX-style env with reset(rng) and step(data, action, info)
  - inference_fn: (obs, rng) -> (action, extras)
  - mj_model: the mujoco.MjModel that env was built on

Replace the stubs at the bottom with your own env + policy to run end-to-end.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from mjx_viz import (
    batched_rollout,
    episode_length,
    render_brax_html,
    slice_episode,
    write_html,
)


def save_rollouts_as_html(
    env,
    inference_fn,
    mj_model,
    rngs,
    out_dir: str,
    max_steps: int = 1000,
    prefix: str = "rollout",
    setup_info=None,
):
    """Run a batch of rollouts and save one HTML file per episode."""
    rollout = batched_rollout(env, inference_fn, max_steps, setup_info=setup_info)
    traj = rollout(rngs)

    alive_all = np.asarray(traj["alive"])
    xpos_all = np.asarray(traj["xpos"])
    xquat_all = np.asarray(traj["xquat"])

    os.makedirs(out_dir, exist_ok=True)
    for ep in range(alive_all.shape[0]):
        ep_len = episode_length(alive_all[ep])
        if ep_len == 0:
            continue
        ep_traj = slice_episode(
            {"xpos": xpos_all[ep], "xquat": xquat_all[ep]}, ep_len
        )
        html = render_brax_html(mj_model, ep_traj["xpos"], ep_traj["xquat"])
        if html is not None:
            write_html(os.path.join(out_dir, f"{prefix}_ep{ep}.html"), html)


if __name__ == "__main__":
    # Fill in your own env + policy here.
    raise SystemExit(
        "Import save_rollouts_as_html and pass your env/inference_fn/mj_model."
    )
