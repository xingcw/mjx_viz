"""Filesystem scanner for SAC/PPO checkpoint datasets.

Supports three on-disk layouts:

1. Morphology-randomized runs (data_gen/collect.py + eval_morphology_distributed.py):
       {root}/eval_config.yaml
       {root}/run_XXXXX/{morphology_metadata.json,
                        morphology_eval_metrics.pkl,
                        sac_params.pkl, sac_metrics.pkl,
                        finished.json, MORPHOLOGY_EVAL_DONE.txt}

2. Morphology runs without eval yet (data_gen/collect.py only):
       {root}/config.yaml
       {root}/run_XXXXX/{morphology_metadata.json, finished.json,
                         sac_params.pkl, sac_metrics.pkl}

3. Legacy flat per-checkpoint layout (older training runs):
       {root}/ckpt_NNNNNNNNNNNN/{eval_metrics.pkl, sac_metrics.pkl,
                                 sac_params.pkl, DONE.txt}

The scanner keys runs by their directory basename (run_name) so both
`run_XXXXX` and `ckpt_NNNNNNNNNNNN` are first-class.
"""

from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml


RUN_DIR_RE = re.compile(r"^(?:run|ckpt)_(\d+)(?:_[A-Za-z0-9._-]+)?$")

# Names to probe, in priority order. The first that exists wins.
_EVAL_PKL_NAMES = ("morphology_eval_metrics.pkl", "eval_metrics.pkl")
_TRAIN_DONE_NAMES = ("finished.json", "DONE.txt")
_EVAL_DONE_NAMES = ("MORPHOLOGY_EVAL_DONE.txt",)


def _parse_run_suffix(name: str) -> int | None:
    """Return the numeric suffix of a `run_*` or `ckpt_*` dir, else None."""
    m = RUN_DIR_RE.match(name)
    return int(m.group(1)) if m else None


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_yaml(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("r") as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return None


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        p = directory / n
        if p.is_file():
            return p
    return None


def load_dataset_config(root: str | Path) -> dict | None:
    """Return parsed eval_config.yaml or config.yaml at the dataset root.
    eval_config wins if both exist; returns None for legacy datasets.
    """
    root_p = Path(root)
    for candidate in ("eval_config.yaml", "config.yaml"):
        cfg = _read_yaml(root_p / candidate)
        if cfg is not None:
            return cfg
    return None


def _ckpt_search_dir(run_dir: Path, unified: bool) -> Path:
    """Where eval_metrics.pkl / morphology_metadata.json live for one run.

    In unified mode they sit under `<run>/checkpoints/` (with a per-run
    `run_meta.json` and `config.json` at the run-dir level). The legacy
    layout had them at `<run>/` directly.
    """
    return (run_dir / "checkpoints") if unified else run_dir


def scan_ckpt_runs(root: str | Path, *, unified: bool = False) -> list[dict]:
    """List all run directories (run_XXXXX or ckpt_NNNNN) under root.

    Does NOT load reward curves -- see load_reward_curve.

    Returned fields:
        name                  directory basename (unique within dataset)
        path                  absolute path to the run dir
        ckpt_path             absolute path to the dir containing eval_metrics
                              (== path in legacy mode, == path/checkpoints in
                              unified mode)
        run_suffix            int parsed from the name, for natural sorting
        morphology_metadata   dict | None
        train_done            bool (finished.json OR DONE.txt)
        eval_curve_path       str | None  (None if no eval pkl exists)
        eval_done_marker      bool (MORPHOLOGY_EVAL_DONE.txt present)
    """
    root_p = Path(root)
    if not root_p.is_dir():
        return []

    runs: list[dict] = []
    for entry in sorted(root_p.iterdir()):
        if not entry.is_dir():
            continue
        suffix = _parse_run_suffix(entry.name)
        if suffix is None:
            continue
        ckpt_dir = _ckpt_search_dir(entry, unified)
        eval_pkl = _first_existing(ckpt_dir, _EVAL_PKL_NAMES)
        # `finished.json` is written at run-dir level in unified mode
        # (more discoverable when browsing the run as a single folder);
        # legacy layout had it inside the checkpoints dir.
        train_done_dir = entry if unified else ckpt_dir
        train_done = _first_existing(train_done_dir, _TRAIN_DONE_NAMES) is not None
        eval_done_marker = _first_existing(ckpt_dir, _EVAL_DONE_NAMES) is not None
        runs.append(
            {
                "name": entry.name,
                "path": str(entry),
                "ckpt_path": str(ckpt_dir),
                "run_suffix": suffix,
                "morphology_metadata": _read_json(ckpt_dir / "morphology_metadata.json"),
                "train_done": train_done,
                "eval_curve_path": str(eval_pkl) if eval_pkl is not None else None,
                "eval_done_marker": eval_done_marker,
            }
        )
    # Sort by numeric suffix so ckpt_000019074236 and run_00000 both come
    # out in training-step / run-index order.
    runs.sort(key=lambda r: r["run_suffix"])
    return runs


@lru_cache(maxsize=4096)
def _load_curve_cached(path_str: str, mtime: float) -> tuple[tuple[float, ...], float | None]:
    """Cached pickle load keyed on (path, mtime). Returns (rewards_tuple, final_reward)."""
    with open(path_str, "rb") as f:
        data = pickle.load(f)
    arr = np.asarray(data["eval/episode_reward"], dtype=np.float32).reshape(-1)
    final = float(arr[-1]) if arr.size else None
    return tuple(float(x) for x in arr.tolist()), final


def load_reward_curve(run_path: str | Path, *, unified: bool = False) -> dict | None:
    """Return {"rewards": [...], "final_reward": float, "num_ckpts": int,
    "source": "<eval_pkl_basename>"} for the given run directory, or None
    if no eval pkl exists.

    `run_path` is the run directory. In unified mode the eval pkl lives
    one level deeper at `<run>/checkpoints/`.
    """
    run_p = Path(run_path)
    search_dir = _ckpt_search_dir(run_p, unified)
    eval_pkl = _first_existing(search_dir, _EVAL_PKL_NAMES)
    if eval_pkl is None:
        return None
    mtime = eval_pkl.stat().st_mtime
    rewards_tuple, final = _load_curve_cached(str(eval_pkl), mtime)
    return {
        "rewards": list(rewards_tuple),
        "final_reward": final,
        "num_ckpts": len(rewards_tuple),
        "source": eval_pkl.name,
    }


def load_final_rewards(root: str | Path, *, unified: bool = False) -> dict[str, float]:
    """Return {run_name: final_reward} for every run with an eval pkl."""
    out: dict[str, float] = {}
    for info in scan_ckpt_runs(root, unified=unified):
        if info["eval_curve_path"] is None:
            continue
        curve = load_reward_curve(info["path"], unified=unified)
        if curve is not None and curve["final_reward"] is not None:
            out[info["name"]] = curve["final_reward"]
    return out
