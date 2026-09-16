
from __future__ import annotations

import numpy as np

from .schema import (Event, Trajectory, EVENT_ENTRY, EVENT_EXIT, EVENT_PASSAGE,
                     SOURCE_PERIMETER, SOURCE_ORACLE_INTERIOR,
                     SENSOR_INTERIOR, SENSOR_PERIMETER)
from .sensors import Region, sensor_tree


def find_bursts(mask) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def interior_passage_events(traj: Trajectory, sensors, rng=None,
                            min_burst_len: int = 1,
                            source: str = SOURCE_ORACLE_INTERIOR) -> list[Event]:
    interior = [s for s in sensors if s.role == SENSOR_INTERIOR]
    if not interior:
        return []

    tree, slist = sensor_tree(interior)
    dists, idxs = tree.query(traj.xy, k=1)
    radii = np.array([s.fov_radius for s in slist])
    visible = dists <= radii[idxs]

    events: list[Event] = []
    for sensor_pos, sensor in enumerate(slist):
        mask = visible & (idxs == sensor_pos)
        for (bs, be) in find_bursts(mask):
            if (be - bs + 1) < min_burst_len:
                continue

            local = np.argmin(dists[bs:be + 1])
            t_close = float(traj.timestamps[bs + local])
            events.append(Event(
                timestamp=t_close,
                sensor_id=sensor.sensor_id,
                sensor_x=sensor.x,
                sensor_y=sensor.y,
                event_type=EVENT_PASSAGE,
                source=source,
                vehicle_id=traj.vehicle_id,
                t_enter=float(traj.timestamps[bs]),
                t_exit=float(traj.timestamps[be]),
            ))

    events.sort(key=lambda e: e.timestamp)
    return events


def perimeter_events(traj: Trajectory, sensors, region: Region, rng=None,
                     position_noise_std: float = 2.0) -> list[Event]:
    rng = rng if rng is not None else np.random.RandomState(0)
    perim = [s for s in sensors if s.role == SENSOR_PERIMETER]
    if not perim:
        return []
    tree, slist = sensor_tree(perim)

    events: list[Event] = []
    for idx, etype in ((traj.active_start, EVENT_ENTRY),
                       (traj.active_end, EVENT_EXIT)):
        xy = traj.xy[idx]
        _d, j = tree.query(xy, k=1)
        sensor = slist[int(j)]
        noisy = xy + rng.normal(0, position_noise_std, 2)
        events.append(Event(
            timestamp=float(traj.timestamps[idx]),
            sensor_id=sensor.sensor_id,
            sensor_x=float(noisy[0]),
            sensor_y=float(noisy[1]),
            event_type=etype,
            source=SOURCE_PERIMETER,
            vehicle_id=traj.vehicle_id,
            confidence=1.0,
            t_enter=float(traj.timestamps[idx]),
            t_exit=float(traj.timestamps[idx]),
        ))
    return events


def generate_oracle_events(traj: Trajectory, sensors, region: Region,
                           seed: int = 0, perimeter_noise_std: float = 2.0):
    rng = np.random.RandomState(seed)
    perim = perimeter_events(traj, sensors, region, rng=rng,
                             position_noise_std=perimeter_noise_std)
    interior = interior_passage_events(traj, sensors, rng=rng)
    return perim, interior


def summarise_events(trajs, sensors, region: Region, seed: int = 0) -> dict:
    n_interior, n_perim, covered = [], [], []
    from .sensors import coverage_fraction
    interior_sensors = [s for s in sensors if s.role == SENSOR_INTERIOR]
    for i, tr in enumerate(trajs):
        p, iv = generate_oracle_events(tr, sensors, region, seed=seed + i)
        n_perim.append(len(p))
        n_interior.append(len(iv))
        covered.append(coverage_fraction(interior_sensors, tr.xy[tr.active_slice]))
    return {
        "n_trajectories": len(trajs),
        "interior_events_mean": float(np.mean(n_interior)) if n_interior else 0.0,
        "interior_events_median": float(np.median(n_interior)) if n_interior else 0.0,
        "interior_events_min": int(np.min(n_interior)) if n_interior else 0,
        "interior_events_max": int(np.max(n_interior)) if n_interior else 0,
        "frac_with_zero_interior": float(np.mean(np.array(n_interior) == 0)) if n_interior else 0.0,
        "perimeter_events_mean": float(np.mean(n_perim)) if n_perim else 0.0,
        "interior_coverage_mean": float(np.mean(covered)) if covered else 0.0,
    }
