
from __future__ import annotations

import numpy as np

from ..schema import SOURCE_PERIMETER, EVENT_ENTRY, EVENT_EXIT

CONFIDENCE_THRESHOLD = 0.2
P0_SCALE = 10.0
Q_SCALE = 0.5
R_EDGE_SCALE = 0.1
R_INNER_FAITHFUL = 3.0


class KFConfig:
    def __init__(self, fov_radius: float = 25.0, avg_speed: float = 8.0,
                 faithful_r_inner: bool = False,
                 perimeter_sigma: float = 2.0, seed: int = 0):
        self.fov_radius = fov_radius
        self.avg_speed = avg_speed
        self.faithful_r_inner = faithful_r_inner
        self.perimeter_sigma = perimeter_sigma
        self.seed = seed

    @property
    def r_inner_scale(self) -> float:
        if self.faithful_r_inner:
            return R_INNER_FAITHFUL
        return (self.fov_radius / np.sqrt(2.0)) ** 2

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["r_inner_scale"] = self.r_inner_scale
        d["r_edge_scale"] = R_EDGE_SCALE
        d["confidence_threshold"] = CONFIDENCE_THRESHOLD
        return d


class _CarKF:

    def __init__(self, vehicle_id, xy, dt, init_velocity):
        self.vehicle_id = vehicle_id
        self.state = np.array([xy[0], xy[1], init_velocity[0], init_velocity[1]],
                              dtype=float)
        self.P = np.eye(4) * P0_SCALE
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * Q_SCALE
        self.age = 0
        self.alive = True

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    def update(self, z, R):
        y = np.asarray(z, float) - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def xy(self):
        return self.state[:2].copy()


def identify_event_softmax(tracks, event_xy, threshold=CONFIDENCE_THRESHOLD):
    if not tracks:
        return None
    ids, costs = [], []
    for vid, car in tracks.items():
        dist = float(np.linalg.norm(car.xy - np.asarray(event_xy, float)))
        uncertainty = float(np.trace(car.P[:2, :2])) + 1e-6
        ids.append(vid)
        costs.append(dist / uncertainty)
    costs = np.clip(np.array(costs), 0, 100)
    exp_scores = np.exp(-costs)
    probs = exp_scores / np.sum(exp_scores)
    best = int(np.argmax(probs))
    if probs[best] < threshold:
        return None
    return ids[best]


class KalmanEstimator:

    name = "kalman"

    def __init__(self, graph, cfg: KFConfig | None = None):
        self.graph = graph
        self.cfg = cfg or KFConfig()

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

        R_edge = np.eye(2) * R_EDGE_SCALE
        R_inner = np.eye(2) * cfg.r_inner_scale

        from .road_pf import _bin_events
        by_step = _bin_events(events, timestamps, dt)

        tracks: dict[int, _CarKF] = {}
        out = {v: np.full((T, 2), np.nan) for v in vehicle_ids}
        retired: set[int] = set()
        self.association_log: list[dict] = []

        for t in range(T):
            for car in tracks.values():
                car.predict()

            evs = by_step.get(t, [])
            for ev in [e for e in evs if e.source == SOURCE_PERIMETER]:
                vid = ev.vehicle_id
                if vid is None or vid not in out:
                    continue
                z = np.array([ev.sensor_x, ev.sensor_y])
                if vid not in tracks or ev.event_type == EVENT_ENTRY:
                    tracks[vid] = _CarKF(vid, z, dt, self._entry_velocity(z))
                else:
                    tracks[vid].update(z, R_edge)
                if ev.event_type == EVENT_EXIT:
                    tracks[vid].update(z, R_edge)
                    retired.add(vid)

            live = {v: c for v, c in tracks.items() if v not in retired}
            for ev in [e for e in evs if e.source != SOURCE_PERIMETER]:
                z = np.array([ev.sensor_x, ev.sensor_y])
                matched = identify_event_softmax(live, z)
                if matched is not None:
                    tracks[matched].update(z, R_inner)
                self.association_log.append({
                    "timestep": t, "timestamp": ev.timestamp,
                    "sensor_id": ev.sensor_id, "assigned_vehicle_id": matched,
                    "true_vehicle_id": None if ev.is_ghost else ev.vehicle_id,
                    "is_ghost": ev.is_ghost, "rejected": matched is None,
                })

            for vid, car in tracks.items():
                if vid in out:
                    out[vid][t] = car.xy

        return out
