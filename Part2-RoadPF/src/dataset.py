
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np

from .road_graph import load_road_graph, EXP_ROOT
from .schema import Trajectory, save_sensors, load_sensors, SENSOR_INTERIOR
from .sensors import (Region, place_perimeter_sensors, place_interior_sensors,
                      coverage_fraction)
from .scenario_gen import generate_trajectory, RouteRejected
from .oracle_events import generate_oracle_events

DATASET_DIR = os.path.join(EXP_ROOT, "dataset")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=EXP_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class DatasetConfig:
    def __init__(self, n_trajectories: int = 120, seed0: int = 5000,
                 region=(-130.0, 130.0, -130.0, 130.0),
                 interior_fov: float = 25.0, interior_density: float = 1.0,
                 interior_margin: float = 30.0, interior_seed: int = 42,
                 perimeter_fov: float = 15.0, perimeter_merge_radius: float = 20.0,
                 perimeter_noise_std: float = 2.0,
                 dt: float = 0.25, speed_min: float = 6.0, speed_max: float = 12.0,
                 speed_noise_frac: float = 0.15, min_inside_dist: float = 150.0,
                 min_interior_events: int = 1, name: str = "main"):
        self.n_trajectories = n_trajectories
        self.seed0 = seed0
        self.region = tuple(float(v) for v in region)
        self.interior_fov = interior_fov
        self.interior_density = interior_density
        self.interior_margin = interior_margin
        self.interior_seed = interior_seed
        self.perimeter_fov = perimeter_fov
        self.perimeter_merge_radius = perimeter_merge_radius
        self.perimeter_noise_std = perimeter_noise_std
        self.dt = dt
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.speed_noise_frac = speed_noise_frac
        self.min_inside_dist = min_inside_dist




        self.min_interior_events = min_interior_events
        self.name = name

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def cache_key(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    @property
    def region_obj(self) -> Region:
        return Region(*self.region)


def _max_blind_gap(traj, interior_events) -> float:
    t_in = float(traj.timestamps[traj.active_start])
    t_out = float(traj.timestamps[traj.active_end])
    ts = sorted([t_in] + [e.timestamp for e in interior_events] + [t_out])
    return float(np.max(np.diff(ts))) if len(ts) > 1 else (t_out - t_in)


class Dataset:

    def __init__(self, cfg: DatasetConfig, graph, sensors, trajectories,
                 oracle_perimeter, oracle_interior, manifest):
        self.cfg = cfg
        self.graph = graph
        self.sensors = sensors
        self.trajectories = trajectories
        self.oracle_perimeter = oracle_perimeter
        self.oracle_interior = oracle_interior
        self.manifest = manifest

    @property
    def region(self) -> Region:
        return self.cfg.region_obj

    @property
    def interior_sensors(self):
        return [s for s in self.sensors if s.role == SENSOR_INTERIOR]

    def tracking_window(self, traj) -> tuple[float, float]:
        return float(traj.timestamps[0]), float(traj.timestamps[-1])

    def __len__(self):
        return len(self.trajectories)


def build_dataset(cfg: DatasetConfig, cache: bool = True, verbose: bool = True) -> Dataset:
    graph = load_road_graph()
    region = cfg.region_obj
    cache_dir = os.path.join(DATASET_DIR, f"{cfg.name}_{cfg.cache_key()}")

    if cache and os.path.exists(os.path.join(cache_dir, "manifest.json")):
        if verbose:
            print(f"[dataset] loading cache {cache_dir}")
        return _load_cache(cfg, graph, cache_dir)

    perim_sensors, perim_stats, _edge_map = place_perimeter_sensors(
        graph, region, fov_radius=cfg.perimeter_fov,
        merge_radius=cfg.perimeter_merge_radius)
    interior_sensors = place_interior_sensors(
        graph, region, fov_radius=cfg.interior_fov, density=cfg.interior_density,
        seed=cfg.interior_seed, margin=cfg.interior_margin)
    sensors = perim_sensors + interior_sensors
    if verbose:
        print(f"[dataset] {len(perim_sensors)} perimeter + "
              f"{len(interior_sensors)} interior sensors")

    trajectories, oracle_p, oracle_i = [], {}, {}
    n_attempts = n_rejected_route = n_rejected_events = 0
    seed = cfg.seed0
    while len(trajectories) < cfg.n_trajectories:
        n_attempts += 1
        if n_attempts > cfg.n_trajectories * 40:
            raise RuntimeError(
                f"only produced {len(trajectories)}/{cfg.n_trajectories} "
                f"trajectories after {n_attempts} attempts")
        s = seed
        seed += 1
        try:
            tr = generate_trajectory(
                graph, region, seed=s, vehicle_id=0, dt=cfg.dt,
                speed_min=cfg.speed_min, speed_max=cfg.speed_max,
                speed_noise_frac=cfg.speed_noise_frac,
                min_inside_dist=cfg.min_inside_dist)
        except (RouteRejected, Exception) as exc:
            if isinstance(exc, RouteRejected):
                n_rejected_route += 1
                continue
            raise
        p, iv = generate_oracle_events(tr, sensors, region, seed=s,
                                       perimeter_noise_std=cfg.perimeter_noise_std)
        if len(iv) < cfg.min_interior_events:
            n_rejected_events += 1
            continue
        tr.meta["n_interior_events"] = len(iv)
        tr.meta["interior_coverage"] = coverage_fraction(
            interior_sensors, tr.xy[tr.active_slice])
        tr.meta["max_blind_gap_s"] = _max_blind_gap(tr, iv)
        trajectories.append(tr)
        oracle_p[tr.trajectory_id] = p
        oracle_i[tr.trajectory_id] = iv
        if verbose and len(trajectories) % 25 == 0:
            print(f"[dataset]   {len(trajectories)}/{cfg.n_trajectories}")

    n_ev = [len(oracle_i[t.trajectory_id]) for t in trajectories]
    transit = [t.meta["transit_duration_s"] for t in trajectories]
    inside = [t.meta["inside_length_m"] for t in trajectories]
    manifest = {
        "config": cfg.as_dict(),
        "cache_key": cfg.cache_key(),
        "git_commit": git_commit(),
        "region": region.as_dict(),
        "n_perimeter_sensors": len(perim_sensors),
        "n_interior_sensors": len(interior_sensors),
        "perimeter_stats": perim_stats,
        "n_trajectories": len(trajectories),
        "n_route_attempts": n_attempts,
        "n_rejected_no_route": n_rejected_route,
        "n_rejected_too_few_events": n_rejected_events,
        "interior_events_per_traj": {
            "mean": float(np.mean(n_ev)), "median": float(np.median(n_ev)),
            "min": int(np.min(n_ev)), "max": int(np.max(n_ev)),
        },
        "transit_duration_s": {
            "mean": float(np.mean(transit)), "min": float(np.min(transit)),
            "max": float(np.max(transit)),
        },
        "inside_length_m": {
            "mean": float(np.mean(inside)), "min": float(np.min(inside)),
            "max": float(np.max(inside)),
        },
        "interior_coverage_mean": float(np.mean(
            [t.meta["interior_coverage"] for t in trajectories])),
    }

    ds = Dataset(cfg, graph, sensors, trajectories, oracle_p, oracle_i, manifest)
    if cache:
        _save_cache(ds, cache_dir)
        if verbose:
            print(f"[dataset] cached to {cache_dir}")
    return ds






def _save_cache(ds: Dataset, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    save_sensors(ds.sensors, os.path.join(cache_dir, "sensors.json"),
                 extra={"region": ds.region.as_dict()})
    payload = {}
    meta = {}
    for tr in ds.trajectories:
        k = tr.trajectory_id
        payload[f"{k}__ts"] = tr.timestamps
        payload[f"{k}__xy"] = tr.xy
        payload[f"{k}__speed"] = tr.speed
        payload[f"{k}__heading"] = tr.heading
        payload[f"{k}__edge"] = tr.road_edge_id
        meta[k] = {"vehicle_id": tr.vehicle_id, "active_start": tr.active_start,
                   "active_end": tr.active_end, "meta": tr.meta}
    np.savez_compressed(os.path.join(cache_dir, "trajectories.npz"), **payload)
    with open(os.path.join(cache_dir, "trajectory_meta.json"), "w") as f:
        json.dump(meta, f)
    ev = {"perimeter": {k: [e.as_dict() for e in v]
                        for k, v in ds.oracle_perimeter.items()},
          "interior": {k: [e.as_dict() for e in v]
                       for k, v in ds.oracle_interior.items()}}
    with open(os.path.join(cache_dir, "oracle_events.json"), "w") as f:
        json.dump(ev, f)
    with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
        json.dump(ds.manifest, f, indent=2)


def _load_cache(cfg: DatasetConfig, graph, cache_dir: str) -> Dataset:
    from .schema import Event
    sensors = load_sensors(os.path.join(cache_dir, "sensors.json"))
    npz = np.load(os.path.join(cache_dir, "trajectories.npz"))
    with open(os.path.join(cache_dir, "trajectory_meta.json")) as f:
        meta = json.load(f)
    trajectories = []
    for k, m in meta.items():
        trajectories.append(Trajectory(
            trajectory_id=k, vehicle_id=m["vehicle_id"],
            timestamps=npz[f"{k}__ts"], xy=npz[f"{k}__xy"],
            speed=npz[f"{k}__speed"], heading=npz[f"{k}__heading"],
            road_edge_id=npz[f"{k}__edge"],
            active_start=m["active_start"], active_end=m["active_end"],
            meta=m["meta"]))
    trajectories.sort(key=lambda t: t.trajectory_id)
    with open(os.path.join(cache_dir, "oracle_events.json")) as f:
        ev = json.load(f)
    op = {k: [Event(**d) for d in v] for k, v in ev["perimeter"].items()}
    oi = {k: [Event(**d) for d in v] for k, v in ev["interior"].items()}
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return Dataset(cfg, graph, sensors, trajectories, op, oi, manifest)
