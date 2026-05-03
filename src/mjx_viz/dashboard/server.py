"""FastAPI server for the mjx_viz training visualization dashboard.

Run with:
    # Unified per-run layout (preferred):
    mjx-viz-dashboard --runs-dir runs
    # Each subdir of `runs/` is one run; videos live in
    # `<run>/rollouts/`, checkpoints in `<run>/checkpoints/`,
    # plus per-run `config.json` and optional `log.txt`.

    # Legacy (still supported for back-compat):
    mjx-viz-dashboard --videos-dir videos
    mjx-viz-dashboard --checkpoints-dir checkpoint_datasets/ant_sac_relu_morphology_50k
    mjx-viz-dashboard --videos-dir videos --checkpoints-dir checkpoint_datasets/...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Path-traversal guard for /api/ckpts/runs/{run_name}/... and /api/runs/.../config etc.
# Allows the optional `_<suffix>` postfix that DashboardSink writes when an
# `--exp-name` is set (e.g. run_00007_lr3e4_seed0).
_RUN_NAME_RE = re.compile(r"^(?:run|ckpt)_\d+(?:_[A-Za-z0-9._-]+)?$")


def _parse_run_name_from_url(name: str) -> str | None:
    return name if _RUN_NAME_RE.match(name) else None

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mjx_viz.dashboard.checkpoints import (
    load_dataset_config,
    load_final_rewards,
    load_reward_curve,
    scan_ckpt_runs,
)
from mjx_viz.dashboard.watcher import scan_runs, watch_sse

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    videos_dir: str | None = None,
    checkpoints_dir: str | None = None,
    title: str = "mjx_viz Dashboard",
    *,
    unified: bool = False,
    runs_dir: str | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Two modes:
      - Unified: pass `runs_dir` (and set `unified=True`). The same root
        serves videos, checkpoints, configs, and logs through a per-run
        nested layout (`<run>/rollouts/`, `<run>/checkpoints/`, etc.).
      - Legacy: pass `videos_dir` and/or `checkpoints_dir` separately
        with `unified=False`.
    """
    if unified:
        if runs_dir is None:
            raise ValueError("unified=True requires runs_dir")
        # In unified mode the same root drives both videos and checkpoint APIs.
        videos_dir = runs_dir
        checkpoints_dir = runs_dir

    if videos_dir is None and checkpoints_dir is None:
        raise ValueError(
            "create_app requires at least one of videos_dir or checkpoints_dir"
        )

    app = FastAPI(title=title)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        if videos_dir is not None:
            return (STATIC_DIR / "index.html").read_text()
        return (STATIC_DIR / "checkpoints.html").read_text()

    @app.get("/api/config")
    async def dashboard_config():
        return {
            "videos_enabled": videos_dir is not None,
            "checkpoints_enabled": checkpoints_dir is not None,
            "unified": unified,
        }

    def _resolve_videos_path(file_path: str) -> Path:
        """Translate the URL-relative `<run>/<rest>` path to an on-disk path.

        Unified layout inserts `/rollouts/` after the run-name segment so the
        frontend's existing `<run>/training_curves.html` and
        `<run>/step_N/<file>.html` URLs resolve correctly without the JS
        having to know about the new layout. Returns an absolute path; the
        caller still validates it stays under `videos_dir`.
        """
        if not unified:
            return Path(videos_dir) / file_path
        parts = Path(file_path).parts
        if not parts:
            return Path(videos_dir) / file_path
        # First segment is the run name; everything after lives under
        # `<run>/rollouts/`.
        return Path(videos_dir) / parts[0] / "rollouts" / Path(*parts[1:])

    if checkpoints_dir is not None:

        @app.get("/checkpoints", response_class=HTMLResponse)
        async def checkpoints_page():
            return (STATIC_DIR / "checkpoints.html").read_text()

        @app.get("/api/ckpts/dataset")
        async def ckpt_dataset_info():
            cfg = load_dataset_config(checkpoints_dir) or {}
            return {
                "root": str(Path(checkpoints_dir).resolve()),
                "eval_config": cfg,
            }

        @app.get("/api/ckpts/runs")
        async def list_ckpt_runs(
            include_final_reward: bool = Query(False),
            with_eval_only: bool = Query(False),
        ):
            """Lightweight run summary. Reward curves are NOT loaded here;
            fetch them per-run via /api/ckpts/runs/{run_name}/curve.

            with_eval_only=true filters to runs that actually have an eval
            pkl on disk (either morphology_eval_metrics.pkl or the legacy
            eval_metrics.pkl).

            include_final_reward=true additionally loads every eval pkl to
            attach a final_reward field -- useful for sorting in the UI
            but costs one pickle read per run.
            """
            runs = scan_ckpt_runs(checkpoints_dir, unified=unified)
            if with_eval_only:
                runs = [r for r in runs if r["eval_curve_path"] is not None]

            finals: dict[str, float] = {}
            if include_final_reward:
                finals = load_final_rewards(checkpoints_dir, unified=unified)

            out = []
            for r in runs:
                meta = r["morphology_metadata"] or {}
                out.append(
                    {
                        "name": r["name"],
                        "run_suffix": r["run_suffix"],
                        "train_done": r["train_done"],
                        "has_eval_curve": r["eval_curve_path"] is not None,
                        "eval_done_marker": r["eval_done_marker"],
                        "morphology_scales": meta.get("morphology_scales"),
                        "morphology_seed": meta.get("morphology_seed"),
                        "train_seed": meta.get("train_seed"),
                        "final_reward": finals.get(r["name"]),
                    }
                )
            return out

        @app.get("/api/ckpts/runs/{run_name}/curve")
        async def get_ckpt_curve(run_name: str):
            """Return the reward curve + metadata for one run."""
            if _parse_run_name_from_url(run_name) is None:
                raise HTTPException(400, f"Invalid run name: {run_name!r}")
            run_dir = Path(checkpoints_dir) / run_name
            if not run_dir.is_dir():
                raise HTTPException(404, f"{run_name} not found")
            curve = load_reward_curve(run_dir, unified=unified)
            if curve is None:
                raise HTTPException(
                    404,
                    f"No eval pkl (morphology_eval_metrics.pkl or "
                    f"eval_metrics.pkl) found in {run_name}",
                )
            meta_dir = run_dir / "checkpoints" if unified else run_dir
            meta_path = meta_dir / "morphology_metadata.json"
            meta = {}
            if meta_path.is_file():
                import json as _json

                meta = _json.loads(meta_path.read_text())
            return {
                "name": run_name,
                "rewards": curve["rewards"],
                "final_reward": curve["final_reward"],
                "num_ckpts": curve["num_ckpts"],
                "source": curve["source"],
                "morphology_metadata": meta,
            }

    if videos_dir is not None:

        @app.get("/videos", response_class=HTMLResponse)
        async def videos_page():
            return (STATIC_DIR / "index.html").read_text()

        @app.get("/api/runs")
        async def list_runs():
            """Return all runs with metadata and step counts."""
            runs = scan_runs(videos_dir, unified=unified)
            result = []
            for name, info in sorted(runs.items(), reverse=True):
                meta = info["meta"] or {}
                # Step keys are int (step_N) or the literal string "final".
                # Sort ints first ascending, then put "final" last.
                int_steps = sorted(s for s in info["steps"].keys() if isinstance(s, int))
                str_steps = [s for s in info["steps"].keys() if isinstance(s, str)]
                steps_sorted = int_steps + str_steps
                latest = int_steps[-1] if int_steps else (str_steps[0] if str_steps else None)
                result.append({
                    "name": name,
                    "created": meta.get("created", ""),
                    "num_steps": len(steps_sorted),
                    "steps": steps_sorted,
                    "latest_step": latest,
                    "has_config": unified and (Path(videos_dir) / name / "config.json").is_file(),
                    "has_log": unified and (Path(videos_dir) / name / "log.txt").is_file(),
                })
            return result

        @app.get("/api/runs/{run_name}/steps/{step}/episodes")
        async def list_episodes(run_name: str, step: str):
            """Return list of HTML filenames for a given run and step.

            `step` is either a stringified integer (`step_N`) or the literal
            "final" for the post-training rollout (unified mode only).
            """
            runs = scan_runs(videos_dir, unified=unified)
            run = runs.get(run_name)
            if not run:
                raise HTTPException(404, "Run not found")
            try:
                step_key: int | str = int(step)
            except ValueError:
                step_key = step
            if step_key not in run["steps"]:
                raise HTTPException(404, "Step not found")
            return run["steps"][step_key]

        @app.get("/api/files/{file_path:path}")
        async def serve_file(file_path: str):
            """Serve an HTML file from the videos directory.

            URL paths use the legacy shape `<run>/<file>` or
            `<run>/step_N/<file>` (frontend stays simple). In unified mode
            the on-disk path is `<run>/rollouts/<...>`.
            """
            full_path = _resolve_videos_path(file_path)
            try:
                full_path.resolve().relative_to(Path(videos_dir).resolve())
            except ValueError:
                raise HTTPException(403, "Access denied")
            if not full_path.is_file():
                raise HTTPException(404, "File not found")
            # training_curves.html is rewritten in place each eval — anything
            # else (per-step rollouts, traj plots) is write-once. Disable
            # caching for the mutating file so the iframe always sees fresh
            # panels; keep the long cache for the immutable rollouts.
            if full_path.name == "training_curves.html":
                cache_header = "no-cache, no-store, must-revalidate"
            else:
                cache_header = "public, max-age=3600"
            return FileResponse(
                str(full_path),
                media_type="text/html",
                headers={"Cache-Control": cache_header},
            )

        if unified:

            @app.get("/api/runs/{run_name}/config")
            async def get_run_config(run_name: str):
                """Return the per-run config.json (full CLI args + env cfg).

                Only available in unified mode; legacy layout had no per-run
                config because the dataset shared one `eval_config.yaml`.
                """
                if _parse_run_name_from_url(run_name) is None:
                    raise HTTPException(400, f"Invalid run name: {run_name!r}")
                config_path = Path(videos_dir) / run_name / "config.json"
                if not config_path.is_file():
                    raise HTTPException(404, "config.json not found for this run")
                import json as _json
                try:
                    return _json.loads(config_path.read_text())
                except _json.JSONDecodeError as e:
                    raise HTTPException(500, f"config.json parse error: {e}")

            @app.get("/api/runs/{run_name}/log", response_class=PlainTextResponse)
            async def get_run_log(
                run_name: str,
                tail: int | None = Query(None, ge=1, le=100000),
            ):
                """Return the per-run log.txt (captured by the launcher).

                ``tail`` returns only the last N lines so very long logs stay
                cheap to render in the iframe.
                """
                if _parse_run_name_from_url(run_name) is None:
                    raise HTTPException(400, f"Invalid run name: {run_name!r}")
                log_path = Path(videos_dir) / run_name / "log.txt"
                if not log_path.is_file():
                    raise HTTPException(404, "log.txt not found for this run")
                if tail is None:
                    return log_path.read_text(errors="replace")
                # Cheap tail: read the whole file, slice. log.txt is bounded
                # by training duration so this is fine in practice.
                lines = log_path.read_text(errors="replace").splitlines()
                return "\n".join(lines[-tail:])

        @app.get("/api/events")
        async def sse_events():
            """SSE stream of new-file events."""
            return StreamingResponse(
                watch_sse(videos_dir, unified=unified),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    return app


def main():
    parser = argparse.ArgumentParser(description="mjx_viz Dashboard")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--runs-dir", type=str, default=None,
        help="Root directory containing per-run unified folders "
             "(`<run>/rollouts/`, `<run>/checkpoints/`, `<run>/config.json`, "
             "`<run>/log.txt`). Mutually exclusive with the legacy "
             "--videos-dir / --checkpoints-dir flags.",
    )
    parser.add_argument(
        "--videos-dir", type=str, default=None,
        help="(Legacy) Root directory of training HTML rollouts laid out as "
             "<root>/<run>/step_N/*.html. At least one of --runs-dir / "
             "--videos-dir / --checkpoints-dir is required.",
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default=None,
        help="(Legacy) Root directory containing run_XXXXX/ subdirectories "
             "with morphology_eval_metrics.pkl. At least one of "
             "--runs-dir / --videos-dir / --checkpoints-dir is required.",
    )
    parser.add_argument("--title", type=str, default="mjx_viz Dashboard")
    args = parser.parse_args()

    if args.runs_dir is None and args.videos_dir is None and args.checkpoints_dir is None:
        parser.error(
            "at least one of --runs-dir / --videos-dir / --checkpoints-dir must be set"
        )
    if args.runs_dir is not None and (args.videos_dir or args.checkpoints_dir):
        parser.error(
            "--runs-dir is mutually exclusive with --videos-dir / --checkpoints-dir"
        )

    import uvicorn

    app = create_app(
        videos_dir=args.videos_dir,
        checkpoints_dir=args.checkpoints_dir,
        title=args.title,
        unified=args.runs_dir is not None,
        runs_dir=args.runs_dir,
    )
    print(f"Dashboard starting at http://{args.host}:{args.port}")
    if args.runs_dir:
        print(f"Runs directory:        {Path(args.runs_dir).resolve()}  (unified)")
    if args.videos_dir:
        print(f"Videos directory:      {Path(args.videos_dir).resolve()}")
    if args.checkpoints_dir:
        print(f"Checkpoints directory: {Path(args.checkpoints_dir).resolve()}")
    print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port} <tpu-vm>")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
