# mjx_viz

Portable library for rolling out policies in brax/MJX envs, rendering episodes
as standalone brax HTML, and browsing them through a lightweight FastAPI
dashboard that auto-discovers outputs on disk.

## Components

- `mjx_viz.rollout` - `batched_rollout(env, inference_fn, max_steps, collect_fn)`
  returns a jit+vmap'd `(rngs) -> traj_dict` function. Environments must expose
  `reset(rng) -> (data, obs, info)` and
  `step(data, action, info) -> (data, obs, reward, done, info)`. For brax envs
  use a thin adapter (see `learning2sim2real/data_gen/check_hypernet_viz.py::BraxRolloutAdapter`).
- `mjx_viz.html_render` - `render_brax_html(mj_model, xpos, xquat)` returns a
  standalone HTML string; `write_html(path, html)` persists it.
- `mjx_viz.dashboard` - FastAPI app that scans a videos directory and serves
  the HTML episodes in a browsable UI.

## Dashboard

### Launch

From the repo root, point the dashboard at the directory holding your
`morph_*_rl/` and `morph_*_hypernet/` (or any `run/step_K/*.html`) subtrees:

```
~/.local/bin/uv run mjx-viz-dashboard \
    --videos-dir results/zero_shot_transfer/ant_sac_relu \
    --port 8501
```

Module form (when the entrypoint is not on `PATH`):

```
~/.local/bin/uv run python -m mjx_viz.dashboard.server \
    --videos-dir results/zero_shot_transfer/ant_sac_relu \
    --port 8501
```

CLI flags (defined in `src/mjx_viz/dashboard/server.py::main`):

| flag | default | meaning |
| --- | --- | --- |
| `--videos-dir` | `videos` | root directory that the dashboard scans |
| `--port` | `8501` | HTTP port to listen on |
| `--host` | `0.0.0.0` | bind address |
| `--title` | `mjx_viz Dashboard` | browser title |

### Accessing from a laptop when the server runs on a TPU VM

Open an SSH tunnel so `localhost:8501` on your laptop forwards to the VM:

```
gcloud compute ssh <tpu-vm> -- -L 8501:localhost:8501
```

Then visit `http://localhost:8501` in your browser. The dashboard process also
prints a ready-to-copy tunnel command on startup.

### Expected directory layouts

The dashboard auto-detects two layouts under `--videos-dir`:

- "viz" layout (single-shot runs, e.g. `zero_shot_transfer.py` outputs):
  ```
  <videos-dir>/
    morph_000_rl/
      morphology_metadata.json
      viz/
        rollout_ep0.html
        rollout_ep1.html
        ...
    morph_000_hypernet/
      morphology_metadata.json
      viz/
        rollout_ep0.html
        ...
  ```
- "step" layout (training runs with periodic checkpoints):
  ```
  <videos-dir>/
    run-YYYYMMDD-HHMMSS/
      metadata.json
      step_0/*.html
      step_1000/*.html
      ...
  ```

Runs are sorted newest-first by name; `morphology_metadata.json` fields (source,
run, eval_reward, best_step, etc.) are surfaced as chips in the UI.

### Live updates

The dashboard watches `--videos-dir` via SSE and streams new episodes into the
browser as files land on disk, so a long-running `zero_shot_transfer.py` job
will populate the UI morph by morph without refreshing.

## Install

The dashboard extras are declared in `pyproject.toml`:

```
uv pip install -e ".[dashboard]"      # fastapi, uvicorn, watchfiles
```

Plain rollout/render functionality does not require the dashboard extras.
