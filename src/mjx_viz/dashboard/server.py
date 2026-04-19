"""FastAPI server for the mjx_viz training visualization dashboard.

Run with:
    mjx-viz-dashboard --videos-dir videos
    # or
    python -m mjx_viz.dashboard.server --videos-dir videos
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mjx_viz.dashboard.watcher import scan_runs, watch_sse

STATIC_DIR = Path(__file__).parent / "static"


def create_app(videos_dir: str, title: str = "mjx_viz Dashboard") -> FastAPI:
    app = FastAPI(title=title)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/api/runs")
    async def list_runs():
        """Return all runs with metadata and step counts."""
        runs = scan_runs(videos_dir)
        result = []
        for name, info in sorted(runs.items(), reverse=True):
            meta = info["meta"] or {}
            steps_sorted = sorted(info["steps"].keys())
            # Infer layout: single-shot runs have exactly step 0 and a viz/ dir
            is_viz_layout = (
                name != "__default__"
                and (Path(videos_dir) / name / "viz").is_dir()
                and steps_sorted == [0]
            )
            result.append({
                "name": name,
                "created": meta.get("created", ""),
                "num_steps": len(steps_sorted),
                "steps": steps_sorted,
                "latest_step": steps_sorted[-1] if steps_sorted else None,
                "layout": "viz" if is_viz_layout else "step",
                "meta": meta,
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
        return FileResponse(
            str(full_path),
            media_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
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
        "--videos-dir", type=str, default="videos",
        help="Root directory containing training HTML outputs",
    )
    parser.add_argument("--title", type=str, default="mjx_viz Dashboard")
    args = parser.parse_args()

    import uvicorn

    app = create_app(args.videos_dir, title=args.title)
    print(f"Dashboard starting at http://{args.host}:{args.port}")
    print(f"Videos directory: {Path(args.videos_dir).resolve()}")
    print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port} <tpu-vm>")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
