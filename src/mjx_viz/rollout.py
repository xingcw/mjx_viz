"""Generic JIT-compatible rollout helpers for MJX/brax-style environments.

The environment is expected to expose:
    reset(rng) -> (data, obs, info)
    step(data, action, info) -> (data, obs, reward, done, info)

and an ``inference_fn(obs, rng) -> (action, extras)`` policy. Everything is
traced under ``jax.lax.scan``, so per-step side effects are not supported --
collect whatever you need via ``collect_fn`` and post-process after the scan.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


InfoInit = Callable[[Dict[str, Any]], Dict[str, Any]]
CollectFn = Callable[..., Dict[str, jnp.ndarray]]


def _default_collect(data, obs, reward, done, info, alive) -> Dict[str, jnp.ndarray]:
    """Default per-step collector: link geometry for brax HTML + alive mask.

    Skips the world body (index 0) to match brax.io.html's expectations.
    """
    return {
        "alive": alive.astype(jnp.float32),
        "reward": reward * alive,
        "xpos": data.xpos[1:],
        "xquat": data.xquat[1:],
    }


def make_rollout(
    env,
    inference_fn,
    max_steps: int,
    setup_info: Optional[InfoInit] = None,
    collect_fn: Optional[CollectFn] = None,
) -> Callable[[jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Return a (rng) -> traj function that runs one episode under ``jax.lax.scan``.

    Args:
        env: environment exposing reset/step (see module docstring).
        inference_fn: policy callable ``(obs, rng) -> (action, extras)``.
        max_steps: rollout length (episodes shorter than this are masked out
            via the ``alive`` flag).
        setup_info: optional callable that mutates/returns the initial ``info``
            dict (e.g. to inject a task command or zero-init reward accumulators).
        collect_fn: optional per-step collector. Receives
            ``(data, obs, reward, done, info, alive)`` and returns a pytree of
            quantities to stack along time. Defaults to ``{alive, reward, xpos, xquat}``.

    Returns:
        Callable ``rollout(rng) -> traj_dict``. Each leaf of ``traj_dict`` has
        shape ``(max_steps, ...)``. Wrap with ``jax.jit`` / ``jax.vmap`` as needed.
    """
    collect = collect_fn or _default_collect

    def rollout(rng: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        data, obs, info = env.reset(rng)
        if setup_info is not None:
            info = setup_info(info)

        def scan_step(carry, _):
            data, obs, info, rng, terminated = carry
            rng = jax.random.fold_in(rng, 0)
            action, _ = inference_fn(obs, rng)
            next_data, next_obs, reward, done, next_info = env.step(data, action, info)

            alive = ~terminated
            new_terminated = terminated | done.astype(jnp.bool_)
            per_step = collect(next_data, next_obs, reward, done, next_info, alive)

            carry = (next_data, next_obs, next_info, rng, new_terminated)
            return carry, per_step

        init_carry = (data, obs, info, rng, jnp.bool_(False))
        _, traj = jax.lax.scan(scan_step, init_carry, None, length=max_steps)
        return traj

    return rollout


def batched_rollout(
    env,
    inference_fn,
    max_steps: int,
    setup_info: Optional[InfoInit] = None,
    collect_fn: Optional[CollectFn] = None,
) -> Callable[[jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Return a jit+vmap'd ``(rngs) -> traj`` function over a batch of rng seeds."""
    single = make_rollout(env, inference_fn, max_steps, setup_info, collect_fn)
    return jax.jit(jax.vmap(single))


def episode_length(alive: np.ndarray) -> int:
    """Number of live steps from an ``alive`` mask (non-jitted host helper)."""
    return int(np.asarray(alive).sum())


def slice_episode(traj: Dict[str, np.ndarray], ep_len: int) -> Dict[str, np.ndarray]:
    """Return a copy of ``traj`` truncated to the first ``ep_len`` steps.

    Works on a single-episode trajectory (leading axis is time). Use after
    pulling data back to host with ``np.asarray``.
    """
    return {k: np.asarray(v)[:ep_len] for k, v in traj.items()}
