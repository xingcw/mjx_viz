"""FastAPI server for the mjx_viz training visualization dashboard.

Run with:
    mjx-viz-dashboard --videos-dir videos
    mjx-viz-dashboard --checkpoints-dir checkpoint_datasets/ant_sac_relu_morphology_50k
    mjx-viz-dashboard --videos-dir videos --checkpoints-dir checkpoint_datasets/...
    # or
    python -m mjx_viz.dashboard.server --videos-dir videos
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Path-traversal guard for /api/ckpts/runs/{run_name}/...
_RUN_NAME_RE = re.compile(r"^(?:run|ckpt)_\d+$")


def _parse_run_name_from_url(name: str) -> str | None:
    return name if _RUN_NAME_RE.match(name) else None

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
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
) -> FastAPI:
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
        }

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
            runs = scan_ckpt_runs(checkpoints_dir)
            if with_eval_only:
                runs = [r for r in runs if r["eval_curve_path"] is not None]

            finals: dict[str, float] = {}
            if include_final_reward:
                finals = load_final_rewards(checkpoints_dir)

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
            curve = load_reward_curve(run_dir)
            if curve is None:
                raise HTTPException(
                    404,
                    f"No eval pkl (morphology_eval_metrics.pkl or "
                    f"eval_metrics.pkl) found in {run_name}",
                )
            meta_path = run_dir / "morphology_metadata.json"
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
            runs = scan_runs(videos_dir)
            result = []
            for name, info in sorted(runs.items(), reverse=True):
                meta = info["meta"] or {}
                steps_sorted = sorted(info["steps"].keys())
                result.append({
                    "name": name,
                    "created": meta.get("created", ""),
                    "num_steps": len(steps_sorted),
                    "steps": steps_sorted,
                    "latest_step": steps_sorted[-1] if steps_sorted else None,
                })
            return result

        @app.get("/api/runs/{run_name}/steps/{step:int}/episodes")
        async def list_episodes(run_name: str, step: int):
            """Return list of HTML filenames for a given run and step."""
            runs = scan_runs(videos_dir)
            run = runs.get(run_name)
            if not run or step not in run["steps"]:
                raise HTTPException(404, "Run or step not found")
            return run["steps"][step]

        @app.get("/api/files/{file_path:path}")
        async def serve_file(file_path: str):
            """Serve a brax HTML file from the videos directory.

            Paths are relative to videos_dir, e.g.:
              /api/files/run-20260411-041700/step_0/forward_slow_ep0.html
              /api/files/step_0/forward_slow_ep0.html  (legacy flat layout)
            """
            full_path = Path(videos_dir) / file_path
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

        @app.get("/api/events")
        async def sse_events():
            """SSE stream of new-file events."""
            return StreamingResponse(
                watch_sse(videos_dir),
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
        "--videos-dir", type=str, default=None,
        help="Root directory containing training HTML rollouts. "
             "At least one of --videos-dir / --checkpoints-dir is required.",
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default=None,
        help="Root directory containing run_XXXXX/ subdirectories with "
             "morphology_eval_metrics.pkl. At least one of --videos-dir / "
             "--checkpoints-dir is required.",
    )
    parser.add_argument("--title", type=str, default="mjx_viz Dashboard")
    args = parser.parse_args()

    if args.videos_dir is None and args.checkpoints_dir is None:
        parser.error(
            "at least one of --videos-dir / --checkpoints-dir must be set"
        )

    import uvicorn

    app = create_app(
        videos_dir=args.videos_dir,
        checkpoints_dir=args.checkpoints_dir,
        title=args.title,
    )
    print(f"Dashboard starting at http://{args.host}:{args.port}")
    if args.videos_dir:
        print(f"Videos directory:      {Path(args.videos_dir).resolve()}")
    if args.checkpoints_dir:
        print(f"Checkpoints directory: {Path(args.checkpoints_dir).resolve()}")
    print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port} <tpu-vm>")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
