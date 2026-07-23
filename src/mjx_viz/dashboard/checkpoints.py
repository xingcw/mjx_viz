"""Filesystem scanner for SAC/PPO checkpoint datasets.

Supports five on-disk layouts:

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

4. LMDB checkpoint datasets (process/prepare_ckpts.py output; one LMDB
   per training run, run dirs named by bare run index):
       {root}/NNNNNN/{data.mdb, lock.mdb, runs.json}
   The reward curve is read from runs.json (per-checkpoint `train_ret`),
   so no LMDB open is needed to plot training progress.

5. Non-morphology runs, unified layout (data_gen/collect.py current schema,
   e.g. racing_pointmass_ppo_elu_v1; data_gen/train_racing_ppo.py writes the
   per-run config.json):
       {root}/config.yaml, eval_config.yaml
       {root}/run_XXXXX/{run_metadata.json, finished.json, config.json,
                         checkpoints/{eval_metrics.pkl, ppo_params.pkl,
                                      ppo_metrics.pkl, ckpt_steps.json,
                                      DONE.txt}}
   Same shape as layout 1 with `run_metadata.json` in place of
   `morphology_metadata.json` (no morphology fields, but the same
   `train_seed` / `domain_rand` keys) -- see `_METADATA_NAMES`.

The scanner keys runs by their directory basename (run_name) so
`run_XXXXX`, `ckpt_NNNNNNNNNNNN`, and bare `NNNNNN` are first-class.
"""

from __future__ import annotations

import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml


RUN_DIR_RE = re.compile(r"^(?:run|ckpt)_(\d+)(?:_[A-Za-z0-9._-]+)?$")
LMDB_RUN_DIR_RE = re.compile(r"^(\d+)$")

# Names to probe, in priority order. The first that exists wins.
_EVAL_PKL_NAMES = ("morphology_eval_metrics.pkl", "eval_metrics.pkl")
_TRAIN_DONE_NAMES = ("finished.json", "DONE.txt")
_EVAL_DONE_NAMES = ("MORPHOLOGY_EVAL_DONE.txt",)
# morphology_metadata.json is the older Ant-morphology name; run_metadata.json
# is what data_gen/collect.py currently writes for morphology-free datasets
# (e.g. racing). Both carry train_seed / domain_rand; only the former also
# carries morphology_scales / morphology_seed.
_METADATA_NAMES = ("morphology_metadata.json", "run_metadata.json")


def _parse_run_suffix(name: str) -> int | None:
    """Return the numeric suffix of a `run_*` or `ckpt_*` dir, else None."""
    m = RUN_DIR_RE.match(name)
    return int(m.group(1)) if m else None


def _lmdb_runs_json(run_dir: Path) -> Path | None:
    """Return `<run>/runs.json` if this dir is an LMDB dataset run, else None."""
    if LMDB_RUN_DIR_RE.match(run_dir.name) is None:
        return None
    candidate = run_dir / "runs.json"
    return candidate if candidate.is_file() else None


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


def load_run_metadata(run_dir: str | Path, *, unified: bool = False) -> dict:
    """Return the parsed per-run metadata dict for one run directory.

    Probes `_METADATA_NAMES` in priority order. In unified mode, run-level
    metadata (`run_metadata.json`, written by data_gen/collect.py alongside
    `finished.json` / `config.json`) sits at `<run>/`; the older Ant-morphology
    `morphology_metadata.json` sits inside `<run>/checkpoints/`. Both dirs are
    probed, run-level first, so either dataset shape resolves. Returns an
    empty dict if no metadata file exists.
    """
    run_dir = Path(run_dir)
    ckpt_dir = _ckpt_search_dir(run_dir, unified)
    search_dirs = (run_dir, ckpt_dir) if unified else (ckpt_dir,)
    for d in search_dirs:
        metadata_path = _first_existing(d, _METADATA_NAMES)
        if metadata_path is not None:
            return _read_json(metadata_path) or {}
    return {}


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
        runs_json = _lmdb_runs_json(entry)
        if runs_json is not None:
            # Layout 4: LMDB dataset run. The curve source is runs.json and
            # the run is by construction complete (runs.json is written when
            # dataset preparation finishes).
            runs.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "ckpt_path": str(entry),
                    "run_suffix": int(entry.name),
                    "morphology_metadata": None,
                    "train_done": True,
                    "eval_curve_path": str(runs_json),
                    "eval_done_marker": False,
                }
            )
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
        metadata = load_run_metadata(entry, unified=unified)
        runs.append(
            {
                "name": entry.name,
                "path": str(entry),
                "ckpt_path": str(ckpt_dir),
                "run_suffix": suffix,
                "morphology_metadata": metadata or None,
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


def _extract_lmdb_train_ret(path_str: str) -> list[float]:
    """Read the per-checkpoint train_ret curve out of an LMDB run's runs.json.

    Handles both a plain per-run json ({"metadata": ..., "checkpoints": ...})
    and the fused variant that nests it under the run id.
    """
    with open(path_str, "r") as f:
        data = json.load(f)
    run_name = Path(path_str).parent.name
    if run_name in data:
        data = data[run_name]
    metadata = data["metadata"]
    if "train_ret" in metadata:
        rewards = metadata["train_ret"]
    else:
        rewards = [ckpt["train_ret"] for ckpt in data["checkpoints"].values()]
    return [float(r) for r in rewards]


@lru_cache(maxsize=8192)
def _load_lmdb_curve_cached(path_str: str, mtime: float) -> tuple[tuple[float, ...], float | None]:
    """Cached runs.json curve load keyed on (path, mtime)."""
    rewards = _extract_lmdb_train_ret(path_str)
    final = float(rewards[-1]) if rewards else None
    return tuple(rewards), final


def _lmdb_final_reward(path_str: str) -> float | None:
    """ProcessPool worker: final train_ret of one LMDB run's runs.json."""
    rewards = _extract_lmdb_train_ret(path_str)
    return float(rewards[-1]) if rewards else None


def load_reward_curve(run_path: str | Path, *, unified: bool = False) -> dict | None:
    """Return {"rewards": [...], "final_reward": float, "num_ckpts": int,
    "source": "<curve_file_basename>"} for the given run directory, or None
    if no curve source exists.

    `run_path` is the run directory. In unified mode the eval pkl lives
    one level deeper at `<run>/checkpoints/`. LMDB dataset runs carry their
    curve in `<run>/runs.json` instead of an eval pkl.
    """
    run_p = Path(run_path)
    runs_json = _lmdb_runs_json(run_p)
    if runs_json is not None:
        mtime = runs_json.stat().st_mtime
        rewards_tuple, final = _load_lmdb_curve_cached(str(runs_json), mtime)
        return {
            "rewards": list(rewards_tuple),
            "final_reward": final,
            "num_ckpts": len(rewards_tuple),
            "source": runs_json.name,
        }
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


@lru_cache(maxsize=16)
def _lmdb_final_rewards_cached(paths: tuple[tuple[str, str], ...]) -> dict[str, float]:
    """Batched final-reward extraction for LMDB runs.

    `paths` is a tuple of (run_name, runs_json_path). Fans out over a
    process pool because parsing thousands of runs.json files is CPU-bound
    (the metadata blocks are large); cached on the exact run set so the
    dataset is only swept once per server process.
    """
    out: dict[str, float] = {}
    with ProcessPoolExecutor(max_workers=16) as pool:
        finals = pool.map(_lmdb_final_reward, [p for _, p in paths], chunksize=32)
        for (name, _), final in zip(paths, finals):
            if final is not None:
                out[name] = final
    return out


def load_final_rewards(root: str | Path, *, unified: bool = False) -> dict[str, float]:
    """Return {run_name: final_reward} for every run with a curve source."""
    out: dict[str, float] = {}
    lmdb_runs: list[tuple[str, str]] = []
    for info in scan_ckpt_runs(root, unified=unified):
        if info["eval_curve_path"] is None:
            continue
        if info["eval_curve_path"].endswith("runs.json"):
            lmdb_runs.append((info["name"], info["eval_curve_path"]))
            continue
        curve = load_reward_curve(info["path"], unified=unified)
        if curve is not None and curve["final_reward"] is not None:
            out[info["name"]] = curve["final_reward"]
    if lmdb_runs:
        out.update(_lmdb_final_rewards_cached(tuple(lmdb_runs)))
    return out
