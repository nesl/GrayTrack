
from __future__ import annotations

import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .dataset import Dataset, DatasetConfig, build_dataset, git_commit
from .evaluate import (trajectory_metrics, is_catastrophic, classify_difficulty,
                       classify_blind_gap, branch_density, blind_gap)
from .observation_model import SideChannelObservationModel
from .estimators import build_estimator

COND_PERIMETER_ONLY = "A_perimeter_only"
COND_ORACLE_INTERIOR = "B_oracle_interior"
COND_SIDECHANNEL = "C_sidechannel"
CONDITIONS = (COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR, COND_SIDECHANNEL)


def build_event_stream(ds: Dataset, traj, condition: str,
                       obs_model: SideChannelObservationModel | None = None):
    perim = ds.oracle_perimeter[traj.trajectory_id]
    interior = ds.oracle_interior[traj.trajectory_id]

    if condition == COND_PERIMETER_ONLY:
        return list(perim), []
    if condition == COND_ORACLE_INTERIOR:
        return list(perim) + list(interior), list(interior)
    if condition == COND_SIDECHANNEL:
        if obs_model is None:
            raise ValueError("condition C requires an observation model")
        window = ds.tracking_window(traj)
        corrupted = obs_model.corrupt(interior, window, ds.interior_sensors)
        return list(perim) + list(corrupted), list(corrupted)
    raise KeyError(f"unknown condition {condition!r}")


def run_single(ds: Dataset, traj, estimator_name: str, condition: str,
               obs_model: SideChannelObservationModel | None = None,
               est_kwargs: dict | None = None) -> dict:
    est_kwargs = dict(est_kwargs or {})
    est = build_estimator(estimator_name, ds.graph, **est_kwargs)
    events, interior_used = build_event_stream(ds, traj, condition, obs_model)
    out = est.run(events, traj.timestamps, vehicle_ids=[traj.vehicle_id])
    m = trajectory_metrics(out[traj.vehicle_id], traj)

    n_true = sum(1 for e in interior_used if not e.is_ghost)
    n_ghost = sum(1 for e in interior_used if e.is_ghost)
    return {
        "trajectory_id": traj.trajectory_id,
        "estimator": estimator_name,
        "condition": condition,
        **m,
        "catastrophic": is_catastrophic(m),
        "n_interior_events_used": len(interior_used),
        "n_true_events_used": n_true,
        "n_ghost_events_used": n_ghost,
        "n_oracle_interior_events": len(ds.oracle_interior[traj.trajectory_id]),
        "difficulty": classify_difficulty(traj),
        "blind_gap_stratum": classify_blind_gap(traj),
        "branch_density_per_km": branch_density(traj),
        "max_blind_gap_s": blind_gap(traj),
        "inside_length_m": traj.meta.get("inside_length_m"),
        "transit_duration_s": traj.meta.get("transit_duration_s"),
        "interior_coverage": traj.meta.get("interior_coverage"),
    }






_WORKER: dict = {}


def _worker_init(cfg_dict):
    cfg = DatasetConfig(**cfg_dict)
    _WORKER["ds"] = build_dataset(cfg, cache=True, verbose=False)
    _WORKER["by_id"] = {t.trajectory_id: t for t in _WORKER["ds"].trajectories}


def _worker_run(job):
    ds = _WORKER["ds"]
    traj = _WORKER["by_id"][job["trajectory_id"]]
    obs_model = None
    if job.get("obs") is not None:
        obs_model = SideChannelObservationModel(**job["obs"])
    row = run_single(ds, traj, job["estimator"], job["condition"],
                     obs_model=obs_model, est_kwargs=job.get("est_kwargs"))
    row.update(job.get("tags", {}))
    return row


def run_jobs(cfg: DatasetConfig, jobs, n_workers: int | None = None,
             progress_every: int = 200, verbose: bool = True) -> pd.DataFrame:
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    if n_workers == 1:
        _worker_init(cfg.as_dict())
        rows = []
        for i, job in enumerate(jobs):
            rows.append(_worker_run(job))
            if verbose and progress_every and (i + 1) % progress_every == 0:
                print(f"  [{i+1}/{len(jobs)}]", flush=True)
        return pd.DataFrame(rows)

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init,
                             initargs=(cfg.as_dict(),)) as pool:
        for i, row in enumerate(pool.map(_worker_run, jobs, chunksize=4)):
            rows.append(row)
            if verbose and progress_every and (i + 1) % progress_every == 0:
                print(f"  [{i+1}/{len(jobs)}]", flush=True)
    return pd.DataFrame(rows)






def save_results(df: pd.DataFrame, path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    sidecar = dict(meta)
    sidecar.setdefault("git_commit", git_commit())
    sidecar.setdefault("python", sys.version.split()[0])
    sidecar.setdefault("platform", platform.platform())
    sidecar.setdefault("n_rows", int(len(df)))
    sidecar.setdefault("numpy", np.__version__)
    with open(path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(sidecar, f, indent=2, default=str)


def aggregate(df: pd.DataFrame, group_cols, value_col: str = "rmse",
              n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    from .evaluate import summarise
    out = []
    for key, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(group_cols, key))
        rec.update(summarise(sub[value_col].values, n_boot=n_boot, seed=seed))
        rec["catastrophic_rate"] = (float(sub["catastrophic"].mean())
                                    if "catastrophic" in sub else float("nan"))
        for extra in ("mae", "p95", "max", "coverage"):
            if extra in sub:
                rec[f"{extra}_mean"] = float(np.nanmean(sub[extra].values))
        out.append(rec)
    return pd.DataFrame(out)


def load_dataset_config(path: str | None = None, **overrides) -> DatasetConfig:
    from .road_graph import EXP_ROOT
    path = path or os.path.join(EXP_ROOT, "configs", "dataset.yaml")
    params = {}
    if os.path.exists(path):
        try:
            import yaml
            with open(path) as f:
                params = yaml.safe_load(f) or {}
        except ImportError:
            params = _mini_yaml(path)
    params.update(overrides)
    valid = set(DatasetConfig().__dict__.keys())
    return DatasetConfig(**{k: v for k, v in params.items() if k in valid})


def _mini_yaml(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            if not v:
                continue
            if v.startswith("["):
                out[k.strip()] = [float(x) for x in v.strip("[]").split(",")]
            else:
                try:
                    out[k.strip()] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except ValueError:
                    out[k.strip()] = v
    return out
