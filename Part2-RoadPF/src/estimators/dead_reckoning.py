
from __future__ import annotations

import numpy as np

from ..schema import SOURCE_PERIMETER, EVENT_ENTRY, EVENT_EXIT


class DRConfig:
    def __init__(self, avg_speed: float = 8.0, velocity_alpha: float = 0.5,
                 snap_to_road: bool = False, max_speed: float = 25.0,
                 seed: int = 0):
        self.avg_speed = avg_speed


        self.velocity_alpha = velocity_alpha
        self.snap_to_road = snap_to_road
        self.max_speed = max_speed
        self.seed = seed

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class DeadReckoningEstimator:
    name = "dead_reckoning"

    def __init__(self, graph, cfg: DRConfig | None = None):
        self.graph = graph
        self.cfg = cfg or DRConfig()

    def _entry_velocity(self, xy):
        _pt, edge_idx, _r = self.graph.snap(xy)
        theta = float(self.graph.edge_dir[edge_idx])
        return self.cfg.avg_speed * np.array([np.cos(theta), np.sin(theta)])

    def run(self, events, timestamps, vehicle_ids=None):
        cfg = self.cfg
        timestamps = np.asarray(timestamps, dtype=float)
        T = len(timestamps)
        dt = float(timestamps[1] - timestamps[0]) if T > 1 else 1.0

        if vehicle_ids is None:
            vehicle_ids = sorted({e.vehicle_id for e in events
                                  if e.source == SOURCE_PERIMETER
                                  and e.vehicle_id is not None})

        from .road_pf import _bin_events
        by_step = _bin_events(events, timestamps, dt)

        out = {v: np.full((T, 2), np.nan) for v in vehicle_ids}
        pos: dict[int, np.ndarray] = {}
        vel: dict[int, np.ndarray] = {}
        last_obs: dict[int, tuple[float, np.ndarray]] = {}
        retired: set[int] = set()
        self.association_log: list[dict] = []

        for t in range(T):
            for vid in list(pos):
                if vid not in retired:
                    pos[vid] = pos[vid] + vel[vid] * dt

            evs = by_step.get(t, [])
            perim = [e for e in evs if e.source == SOURCE_PERIMETER]
            interior = [e for e in evs if e.source != SOURCE_PERIMETER]

            for ev in perim:
                vid = ev.vehicle_id
                if vid is None or vid not in out:
                    continue
                z = np.array([ev.sensor_x, ev.sensor_y])
                if ev.event_type == EVENT_ENTRY or vid not in pos:
                    pos[vid] = z.copy()
                    vel[vid] = self._entry_velocity(z)
                else:
                    self._apply_observation(vid, z, timestamps[t], pos, vel, last_obs)
                last_obs[vid] = (float(timestamps[t]), z.copy())
                if ev.event_type == EVENT_EXIT:
                    retired.add(vid)


            live = [v for v in pos if v not in retired]
            for ev in interior:
                if not live:
                    break
                z = np.array([ev.sensor_x, ev.sensor_y])
                vid = min(live, key=lambda v: float(np.linalg.norm(pos[v] - z)))
                self._apply_observation(vid, z, timestamps[t], pos, vel, last_obs)
                last_obs[vid] = (float(timestamps[t]), z.copy())
                self.association_log.append({
                    "timestep": t, "timestamp": ev.timestamp,
                    "sensor_id": ev.sensor_id, "assigned_vehicle_id": vid,
                    "true_vehicle_id": None if ev.is_ghost else ev.vehicle_id,
                    "is_ghost": ev.is_ghost, "rejected": False,
                })

            for vid in out:
                if vid in pos:
                    p = pos[vid]
                    if cfg.snap_to_road:
                        p, _e, _r = self.graph.snap(p)
                    out[vid][t] = p

        return out

    def _apply_observation(self, vid, z, t_now, pos, vel, last_obs):
        cfg = self.cfg
        if vid in last_obs:
            t_prev, z_prev = last_obs[vid]
            gap = t_now - t_prev
            if gap > 1e-6:
                v_meas = (z - z_prev) / gap
                speed = float(np.linalg.norm(v_meas))
                if speed > cfg.max_speed:
                    v_meas = v_meas / speed * cfg.max_speed
                vel[vid] = (cfg.velocity_alpha * v_meas
                            + (1 - cfg.velocity_alpha) * vel[vid])
        pos[vid] = z.copy()
