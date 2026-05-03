"""Filesystem scanner and SSE event source for the training dashboard."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator


def scan_runs(videos_dir: str, *, unified: bool = False) -> dict:
    """Walk the videos directory and return a structured index of all runs.

    Layouts:
      - unified=False (legacy):
          {root}/{run_name}/step_{N}/*.html
          {root}/step_{N}/*.html              (grouped under "__default__")
      - unified=True (per-run unified folder):
          {root}/{run_name}/rollouts/step_{N}/*.html
          {root}/{run_name}/rollouts/final/*.html
          {root}/{run_name}/rollouts/training_curves.html
          run_meta.json sits at {root}/{run_name}/run_meta.json

    Returns:
        {run_name: {"meta": dict | None, "steps": {step_key: [filenames]}}}
        where step_key is an int for `step_N` or the string "final" for the
        post-training rollout.
    """
    root = Path(videos_dir)
    if not root.is_dir():
        return {}

    runs: dict = {}

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue

        # Legacy flat layout: videos/step_N/
        if not unified and entry.name.startswith("step_"):
            step_num = _parse_step(entry.name)
            if step_num is None:
                continue
            run = runs.setdefault("__default__", {"meta": None, "steps": {}})
            run["steps"][step_num] = _list_html(entry)
            continue

        # Per-run layout. In unified mode we look inside `<run>/rollouts/`.
        meta = _load_meta(entry)
        scan_dir = entry / "rollouts" if unified else entry
        if not scan_dir.is_dir():
            continue
        run = runs.setdefault(entry.name, {"meta": meta, "steps": {}})
        for step_dir in sorted(scan_dir.iterdir()):
            if not step_dir.is_dir():
                continue
            if step_dir.name.startswith("step_"):
                step_num = _parse_step(step_dir.name)
                if step_num is not None:
                    run["steps"][step_num] = _list_html(step_dir)
            elif unified and step_dir.name == "final":
                # Post-training rollout. Surfaced under the synthetic key
                # "final" so the frontend can render it as a special entry
                # at the bottom of the per-run step list.
                run["steps"]["final"] = _list_html(step_dir)

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
    """Load run_meta.json from a run directory, or None."""
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return None


async def watch_sse(
    videos_dir: str, poll_interval: float = 2.0, *, unified: bool = False,
) -> AsyncIterator[str]:
    """Async generator that yields SSE-formatted events when HTML files change.

    Uses simple polling (works on any filesystem including NFS/network mounts).
    Each event is a ``data: ...`` line ready for the SSE protocol.

    Tracks mtimes (not just paths) so files rewritten in place — like
    ``training_curves.html`` — also fire events.
    """
    known = _snapshot(videos_dir)

    while True:
        await asyncio.sleep(poll_interval)
        current = _snapshot(videos_dir)
        changed = [
            p for p, mt in current.items()
            if known.get(p) != mt  # new path OR same path with newer mtime
        ]
        if changed:
            known = current
            for fpath in sorted(changed):
                parts = _parse_file_event(videos_dir, fpath, unified=unified)
                if parts:
                    payload = json.dumps(parts)
                    yield f"data: {payload}\n\n"


def _snapshot(videos_dir: str) -> dict[str, float]:
    """Return ``{path: mtime}`` for every .html file under videos_dir."""
    result: dict[str, float] = {}
    root = Path(videos_dir)
    if root.is_dir():
        for html in root.rglob("*.html"):
            try:
                result[str(html)] = html.stat().st_mtime
            except OSError:
                continue
    return result


def _parse_file_event(
    videos_dir: str, filepath: str, *, unified: bool = False,
) -> dict | None:
    """Parse a filepath into a structured event dict."""
    try:
        rel = os.path.relpath(filepath, videos_dir)
        parts = Path(rel).parts

        if unified:
            # Unified: <run>/rollouts/step_N/filename.html
            #          <run>/rollouts/final/filename.html
            #          <run>/rollouts/training_curves.html
            if len(parts) >= 3 and parts[1] == "rollouts":
                if len(parts) == 4 and parts[2].startswith("step_"):
                    step = _parse_step(parts[2])
                    if step is not None:
                        return {"type": "new_file", "run": parts[0],
                                "step": step, "file": parts[3]}
                if len(parts) == 4 and parts[2] == "final":
                    return {"type": "new_file", "run": parts[0],
                            "step": "final", "file": parts[3]}
                if len(parts) == 3:
                    return {"type": "run_file", "run": parts[0],
                            "file": parts[2]}
            # Run-level file (e.g. config.json refreshed): <run>/<file>
            if len(parts) == 2:
                return {"type": "run_file", "run": parts[0], "file": parts[1]}
            return None

        # Legacy named run: run_name/step_N/filename.html
        if len(parts) == 3 and parts[1].startswith("step_"):
            step = _parse_step(parts[1])
            if step is not None:
                return {"type": "new_file", "run": parts[0],
                        "step": step, "file": parts[2]}

        # Legacy flat: step_N/filename.html
        if len(parts) == 2 and parts[0].startswith("step_"):
            step = _parse_step(parts[0])
            if step is not None:
                return {"type": "new_file", "run": "__default__",
                        "step": step, "file": parts[1]}

        # Run-level file: run_name/<file>.html (e.g. training_curves.html).
        # Surfaced so the SSE stream fires on per-eval rewrites of run-scoped
        # artifacts; the frontend treats any event as "refresh".
        if len(parts) == 2 and not parts[0].startswith("step_"):
            return {"type": "run_file", "run": parts[0], "file": parts[1]}
    except (ValueError, IndexError):
        pass
    return None
