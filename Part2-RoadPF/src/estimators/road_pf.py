
from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..road_graph import angle_diff
from ..schema import (Event, SOURCE_PERIMETER, EVENT_ENTRY, EVENT_EXIT)
from .readouts import (DEFAULT_READOUT, READOUTS, WEIGHTED_MEAN, apply_readout,
                       posterior_diagnostics)

DEFAULT_NUM_PARTICLES = 2000


class PFConfig:

    def __init__(self, num_particles: int = DEFAULT_NUM_PARTICLES,
                 process_noise_var: float = 0.5,
                 avg_speed: float = 8.0,
                 perimeter_sigma: float = 2.0,
                 interior_sigma: float | None = None,
                 fov_radius: float = 25.0,
                 base_gate: float = 60.0,
                 adaptive_gate: bool = True,
                 rescue_fix: bool = True,
                 omni_init: bool = True,
                 rescue_drift_threshold: float = 10.0,
                 max_missed: int = 10_000,
                 seed: int = 0,
                 readout: str = DEFAULT_READOUT,
                 record_readouts=None,
                 cloud_digest: bool = False,
                 snapshot: bool = False):
        self.num_particles = num_particles
        self.process_noise_var = process_noise_var
        self.avg_speed = avg_speed
        self.perimeter_sigma = perimeter_sigma



        self.interior_sigma = (interior_sigma if interior_sigma is not None
                               else fov_radius / np.sqrt(2.0))
        self.fov_radius = fov_radius
        self.base_gate = base_gate
        self.adaptive_gate = adaptive_gate
        self.rescue_fix = rescue_fix
        self.omni_init = omni_init
        self.rescue_drift_threshold = rescue_drift_threshold
        self.max_missed = max_missed
        self.seed = seed

        if readout not in READOUTS:
            raise KeyError(f"unknown readout {readout!r}; have {sorted(READOUTS)}")


        self.readout = readout



        self.record_readouts = (tuple(READOUTS) if record_readouts is True
                                else (tuple(record_readouts) if record_readouts
                                      else ()))
        for name in self.record_readouts:
            if name not in READOUTS:
                raise KeyError(f"unknown readout {name!r}")

        self.cloud_digest = bool(cloud_digest)


        self.snapshot = bool(snapshot)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class _Track:

    def __init__(self, graph, vehicle_id, init_xy, cfg: PFConfig, rng,
                 init_heading=None):
        self.graph = graph
        self.vehicle_id = vehicle_id
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.num_particles
        self.missed_count = 0
        self.alive = True
        self.history: dict[int, np.ndarray] = {}

        self.readout_history: dict[str, dict[int, np.ndarray]] = {
            name: {} for name in cfg.record_readouts}
        self.diagnostics: dict[int, dict] = {}
        self.snapshots: dict[int, dict] = {}
        self._seed_particles(init_xy, init_heading)


    def _candidate_samples(self, xy, heading, k=20, cone_deg=60.0):
        _d, idx = self.graph.edge_tree.query(np.asarray(xy, float), k=k)
        idx = np.atleast_1d(idx)
        if heading is not None and not self.cfg.omni_init:
            dirs = self.graph.edge_dir[self.graph.sample_edge_idx[idx]]
            keep = idx[np.abs(angle_diff(dirs, heading)) < np.radians(cone_deg)]
            if len(keep):
                return keep
        return idx

    def _seed_particles(self, xy, heading=None, mask=None):
        cands = self._candidate_samples(xy, heading)
        pick = self.rng.choice(cands, self.n if mask is None else int(mask.sum()),
                               replace=True)
        r = np.clip(self.graph.sample_r[pick] + self.rng.normal(0, 0.3, len(pick)), 0, None)
        e = self.graph.sample_edge_idx[pick].astype(float)
        spd = np.abs(self.rng.normal(self.cfg.avg_speed, self.cfg.avg_speed * 0.2, len(pick)))
        if mask is None:
            self.r, self.e, self.rdot = r, e, spd
            self.log_w = np.full(self.n, -np.log(self.n))
        else:
            self.r[mask], self.e[mask], self.rdot[mask] = r, e, spd
            self.log_w = np.full(self.n, -np.log(self.n))


    def predict(self, dt):
        g, pnv = self.graph, self.cfg.process_noise_var
        n = self.n
        self.r = self.r + self.rdot * dt + self.rng.normal(
            0, np.sqrt(pnv * dt ** 3 / 3), n)
        self.rdot = np.clip(self.rdot + self.rng.normal(0, np.sqrt(pnv * dt), n),
                            0.001, None)



        for _ in range(50):
            e_int = self.e.astype(int)
            elen = g.edge_len[e_int]
            over = self.r > elen
            if not over.any():
                break
            for i in np.flatnonzero(over):
                ei = int(self.e[i])
                _fn, tn = g.edge_list[ei]
                out = g.outgoing.get(tn, [])
                if not out:
                    self.r[i] = g.edge_len[ei] - 0.001
                    continue
                self.r[i] -= g.edge_len[ei]
                self.e[i] = self._choose_outgoing(ei, out)
        self.r = np.clip(self.r, 0, None)

    def _choose_outgoing(self, cur_edge, out_edges, temperature=0.5):
        if len(out_edges) == 1:
            return out_edges[0]
        g = self.graph
        cur_dir = g.edge_dir[cur_edge]
        diffs = np.abs(angle_diff(g.edge_dir[np.asarray(out_edges)], cur_dir))
        lw = -diffs / temperature
        lw -= lw.max()
        w = np.exp(lw)
        return int(self.rng.choice(out_edges, p=w / w.sum()))


    def _weights(self):
        w = np.exp(self.log_w - self.log_w.max())
        s = w.sum()
        return w / s if s > 0 else np.full(self.n, 1.0 / self.n)

    def estimate(self):
        xy = self.graph.xy_from_re(self.r, self.e)
        return np.average(xy, axis=1, weights=self._weights())

    def report(self):
        if self.cfg.readout == WEIGHTED_MEAN:
            return self.estimate()
        xy = self.graph.xy_from_re(self.r, self.e)
        w = self._weights()
        return apply_readout(self.cfg.readout, self.graph, self.r, self.e, w, xy)

    def _emit(self, timestep, w, xy, mean_xy):
        cfg = self.cfg
        primary = (mean_xy if cfg.readout == WEIGHTED_MEAN
                   else apply_readout(cfg.readout, self.graph, self.r, self.e,
                                      w, xy))
        self.history[timestep] = np.asarray(primary, dtype=float).copy()
        for name in cfg.record_readouts:
            v = (mean_xy if name == WEIGHTED_MEAN
                 else apply_readout(name, self.graph, self.r, self.e, w, xy))
            self.readout_history[name][timestep] = np.asarray(v, dtype=float).copy()
        if cfg.record_readouts or cfg.snapshot:
            self.diagnostics[timestep] = posterior_diagnostics(
                self.graph, self.r, self.e, w, xy)
        if cfg.snapshot:
            self.snapshots[timestep] = {"r": self.r.copy(), "e": self.e.copy(),
                                        "w": np.asarray(w, float).copy(),
                                        "xy": np.asarray(xy, float).copy()}


    def update(self, meas_xy, sigma, timestep):
        xy = self.graph.xy_from_re(self.r, self.e)
        diff = xy - np.asarray(meas_xy, float)[:, None]
        self.log_w = self.log_w - 0.5 * np.sum(diff ** 2, axis=0) / sigma ** 2
        w = self._weights()
        self.log_w = np.log(np.maximum(w, 1e-300))
        mu = np.average(xy, axis=1, weights=w)



        self._emit(timestep, w, xy, mu)

        self.missed_count = 0
        n_eff = 1.0 / np.sum(w ** 2)
        if n_eff < self.n / 2:
            idx = self._systematic_resample(w)
            self.r, self.rdot, self.e = self.r[idx], self.rdot[idx], self.e[idx]
            self.log_w = np.full(self.n, -np.log(self.n))

        drift = float(np.linalg.norm(mu - np.asarray(meas_xy, float)))
        if drift > self.cfg.rescue_drift_threshold:
            frac = 0.90 if drift > 50 else (0.50 if drift > 20 else 0.25)
            self._rescue(meas_xy, frac)
            if self.cfg.rescue_fix:

                xy = self.graph.xy_from_re(self.r, self.e)
                mu = np.mean(xy, axis=1)


                self._emit(timestep, self._weights(), xy, mu)

        return mu

    def _systematic_resample(self, w):
        cumsum = np.cumsum(w)
        u = (self.rng.random_sample() + np.arange(self.n)) / self.n
        return np.clip(np.searchsorted(cumsum, u), 0, self.n - 1)

    def _rescue(self, meas_xy, frac):
        nr = int(self.n * frac)
        if nr <= 0:
            return
        cands = self._candidate_samples(meas_xy, None)
        pick = self.rng.choice(cands, nr, replace=True)
        avg_spd = float(np.mean(self.rdot))
        self.r[:nr] = self.graph.sample_r[pick] + self.rng.normal(0, 0.5, nr)
        self.e[:nr] = self.graph.sample_edge_idx[pick]
        self.rdot[:nr] = np.abs(self.rng.normal(avg_spd, avg_spd * 0.2, nr))
        self.log_w = np.full(self.n, -np.log(self.n))

    def predict_only(self, timestep):



        xy = self.graph.xy_from_re(self.r, self.e)
        w = self._weights()
        self._emit(timestep, w, xy, np.average(xy, axis=1, weights=w))
        self.missed_count += 1

    def reinit(self, meas_xy, heading=None):
        self._seed_particles(meas_xy, heading)
        self.missed_count = 0
        self.alive = True


class RoadParticleFilter:

    name = "road_pf"

    def __init__(self, graph, cfg: PFConfig | None = None):
        self.graph = graph
        self.cfg = cfg or PFConfig()

    def run(self, events, timestamps, vehicle_ids=None):
        cfg = self.cfg
        rng = np.random.RandomState(cfg.seed)
        timestamps = np.asarray(timestamps, dtype=float)
        T = len(timestamps)
        dt = float(timestamps[1] - timestamps[0]) if T > 1 else 1.0

        if vehicle_ids is None:
            vehicle_ids = sorted({e.vehicle_id for e in events
                                  if e.source == SOURCE_PERIMETER
                                  and e.vehicle_id is not None})
        by_step = _bin_events(events, timestamps, dt)

        tracks: dict[int, _Track] = {}
        out = {v: np.full((T, 2), np.nan) for v in vehicle_ids}
        retired: set[int] = set()


        self.association_log: list[dict] = []




        self.readout_outputs: dict[str, dict[int, np.ndarray]] = {
            name: {v: np.full((T, 2), np.nan) for v in vehicle_ids}
            for name in cfg.record_readouts}
        self.diagnostics: dict[int, dict[int, dict]] = {}
        self.snapshots: dict[int, dict[int, dict]] = {}
        self._digest = hashlib.sha1() if cfg.cloud_digest else None

        for t in range(T):
            evs = by_step.get(t, [])
            perim = [e for e in evs if e.source == SOURCE_PERIMETER]
            interior = [e for e in evs if e.source != SOURCE_PERIMETER]

            for tr in tracks.values():
                if tr.alive:
                    tr.predict(dt)


            for ev in perim:
                vid = ev.vehicle_id
                if vid is None or vid not in out:
                    continue
                mxy = np.array([ev.sensor_x, ev.sensor_y])
                if ev.event_type == EVENT_ENTRY or vid not in tracks:
                    tracks[vid] = _Track(self.graph, vid, mxy, cfg, rng)
                    tracks[vid].update(mxy, cfg.perimeter_sigma, t)
                else:
                    tracks[vid].update(mxy, cfg.perimeter_sigma, t)
                if ev.event_type == EVENT_EXIT:
                    retired.add(vid)


            live = {v: tr for v, tr in tracks.items()
                    if tr.alive and v not in retired}
            assigned = self._associate(live, interior)
            for vid, ei in assigned.items():
                ev = interior[ei]
                tracks[vid].update(np.array([ev.sensor_x, ev.sensor_y]),
                                   cfg.interior_sigma, t)
            claimed = set(assigned.values())
            for ei, ev in enumerate(interior):
                got = next((v for v, e in assigned.items() if e == ei), None)
                self.association_log.append({
                    "timestep": t, "timestamp": ev.timestamp,
                    "sensor_id": ev.sensor_id, "assigned_vehicle_id": got,
                    "true_vehicle_id": None if ev.is_ghost else ev.vehicle_id,
                    "is_ghost": ev.is_ghost,
                    "rejected": ei not in claimed,
                })

            for vid, tr in tracks.items():
                if t not in tr.history:
                    tr.predict_only(t)
                if tr.missed_count >= cfg.max_missed:
                    tr.alive = False

            for vid, tr in tracks.items():
                if vid in out and t in tr.history:
                    out[vid][t] = tr.history[t]
                    for name in cfg.record_readouts:
                        self.readout_outputs[name][vid][t] = tr.readout_history[name][t]

            if self._digest is not None:
                self._update_digest(t, tracks)

        for vid, tr in tracks.items():
            if tr.diagnostics:
                self.diagnostics[vid] = tr.diagnostics
            if tr.snapshots:
                self.snapshots[vid] = tr.snapshots
        self.cloud_digest = (self._digest.hexdigest()
                             if self._digest is not None else None)
        return out


    def _update_digest(self, t, tracks):
        h = self._digest
        h.update(str(t).encode())
        for vid in sorted(tracks):
            tr = tracks[vid]
            h.update(str(vid).encode())
            h.update(str(tr.missed_count).encode())
            h.update(b"1" if tr.alive else b"0")
            for arr in (tr.r, tr.rdot, tr.e, tr.log_w):
                h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())


    def _associate(self, live_tracks, interior_events):
        if not live_tracks or not interior_events:
            return {}
        vids = list(live_tracks.keys())
        preds = np.array([live_tracks[v].estimate() for v in vids])
        meas = np.array([[e.sensor_x, e.sensor_y] for e in interior_events])
        cost = np.linalg.norm(preds[:, None, :] - meas[None, :, :], axis=2)

        gates = np.array([self._gate_for(live_tracks[v]) for v in vids])[:, None]
        big = 1e6
        masked = np.where(cost <= gates, cost, big)
        rows, cols = linear_sum_assignment(masked)
        return {vids[r]: int(c) for r, c in zip(rows, cols) if masked[r, c] < big}

    def _gate_for(self, track):
        g = self.cfg.base_gate
        if not self.cfg.adaptive_gate:
            return g


        return g * min(1.0 + 0.15 * track.missed_count, 4.0)


def _bin_events(events, timestamps, dt):
    by_step: dict[int, list[Event]] = {}
    t0 = float(timestamps[0])
    T = len(timestamps)
    for e in events:
        idx = int(round((e.timestamp - t0) / dt))
        idx = max(0, min(T - 1, idx))
        by_step.setdefault(idx, []).append(e)
    for k in by_step:
        by_step[k].sort(key=lambda e: (e.source != SOURCE_PERIMETER, e.timestamp))
    return by_step
