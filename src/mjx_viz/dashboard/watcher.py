"""Filesystem scanner and SSE event source for the training dashboard."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator


def scan_runs(videos_dir: str) -> dict:
    """Walk the videos directory and return a structured index of all runs.

    Supports three layouts:
      - Named runs:  videos/{run_name}/step_{N}/*.html
      - Legacy flat: videos/step_{N}/*.html  (grouped under "__default__")
      - Single-shot: videos/{run_name}/viz/*.html  (no step subdir; shown as step 0)

    Run metadata is loaded from `run_meta.json` or (fallback) `morphology_metadata.json`.

    Returns:
        {run_name: {"meta": dict | None, "steps": {step_int: [filenames]}}}
    """
    root = Path(videos_dir)
    if not root.is_dir():
        return {}

    runs: dict = {}

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue

        # Legacy flat layout: videos/step_N/
        if entry.name.startswith("step_"):
            step_num = _parse_step(entry.name)
            if step_num is None:
                continue
            run = runs.setdefault("__default__", {"meta": None, "steps": {}})
            run["steps"][step_num] = _list_html(entry)
            continue

        # Named run layout
        meta = _load_meta(entry)

        # Single-shot layout: videos/{run_name}/viz/*.html
        viz_dir = entry / "viz"
        if viz_dir.is_dir():
            files = _list_html(viz_dir)
            if files:
                run = runs.setdefault(entry.name, {"meta": meta, "steps": {}})
                run["steps"][0] = files
                continue

        # Stepped layout: videos/{run_name}/step_N/
        run = runs.setdefault(entry.name, {"meta": meta, "steps": {}})
        for step_dir in sorted(entry.iterdir()):
            if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
                continue
            step_num = _parse_step(step_dir.name)
            if step_num is not None:
                run["steps"][step_num] = _list_html(step_dir)

    return runs


def _parse_step(name: str) -> int | None:
    """Extract the integer step from a directory name like 'step_2621440'."""
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _list_html(directory: Path) -> list[str]:
    """Return sorted list of .html filenames in a directory."""
    return sorted(f.name for f in directory.iterdir() if f.suffix == ".html")


def _load_meta(run_dir: Path) -> dict | None:
    """Load run_meta.json (or fallback morphology_metadata.json) from a run dir."""
    for fname in ("run_meta.json", "morphology_metadata.json"):
        meta_path = run_dir / fname
        if meta_path.is_file():
            try:
                return json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    return None


async def watch_sse(videos_dir: str, poll_interval: float = 2.0) -> AsyncIterator[str]:
    """Async generator that yields SSE-formatted events when HTML files change.

    Uses simple polling (works on any filesystem including NFS/network mounts).
    Each event is a ``data: ...`` line ready for the SSE protocol.
    """
    known = _snapshot(videos_dir)

    while True:
        await asyncio.sleep(poll_interval)
        current = _snapshot(videos_dir)
        new_files = current - known
        if new_files:
            known = current
            for fpath in sorted(new_files):
                parts = _parse_file_event(videos_dir, fpath)
                if parts:
                    payload = json.dumps(parts)
                    yield f"data: {payload}\n\n"


def _snapshot(videos_dir: str) -> set[str]:
    """Return set of all .html file paths under videos_dir."""
    result = set()
    root = Path(videos_dir)
    if root.is_dir():
        for html in root.rglob("*.html"):
            result.add(str(html))
    return result


def _parse_file_event(videos_dir: str, filepath: str) -> dict | None:
    """Parse a filepath into a structured event dict."""
    try:
        rel = os.path.relpath(filepath, videos_dir)
        parts = Path(rel).parts

        # Named run: run_name/step_N/filename.html
        if len(parts) == 3 and parts[1].startswith("step_"):
            step = _parse_step(parts[1])
            if step is not None:
                return {"type": "new_file", "run": parts[0],
                        "step": step, "file": parts[2]}

        # Single-shot: run_name/viz/filename.html
        if len(parts) == 3 and parts[1] == "viz":
            return {"type": "new_file", "run": parts[0],
                    "step": 0, "file": parts[2]}

        # Legacy flat: step_N/filename.html
        if len(parts) == 2 and parts[0].startswith("step_"):
            step = _parse_step(parts[0])
            if step is not None:
                return {"type": "new_file", "run": "__default__",
                        "step": step, "file": parts[1]}
    except (ValueError, IndexError):
        pass
    return None
