
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np

from .schema import (Event, EVENT_PASSAGE, SOURCE_SIDECHANNEL_SIM,
                     SENSOR_INTERIOR)


TODO = "TODO"


def _is_todo(v) -> bool:
    return isinstance(v, str) and v.strip().upper() == TODO


@dataclass
class ObservationModelParams:

    false_negative_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    timing_jitter_std_s: Optional[float] = None






    timing_bias_s: Optional[float] = None







    apply_timing_bias: bool = False

    per_camera: dict = None



    timing_residuals_s: Optional[list] = None
    source: str = "uncalibrated"
    notes: str = ""

    def __post_init__(self):
        if self.per_camera is None:
            self.per_camera = {}

    @property
    def is_calibrated(self) -> bool:
        return (self.false_negative_rate is not None
                and self.false_positive_rate is not None
                and self.timing_jitter_std_s is not None)

    def missing_fields(self) -> list[str]:
        out = []
        for f in ("false_negative_rate", "false_positive_rate", "timing_jitter_std_s"):
            if getattr(self, f) is None:
                out.append(f)
        return out

    def as_dict(self) -> dict:
        return asdict(self)


def load_params(path: str) -> ObservationModelParams:
    with open(path) as f:
        doc = json.load(f)

    def num(key):
        v = doc.get(key)
        if v is None or _is_todo(v):
            return None
        return float(v)

    per_cam_raw = doc.get("per_camera", {}) or {}
    per_cam = {}
    for cam, sub in per_cam_raw.items():
        clean = {k: (None if _is_todo(v) else v) for k, v in (sub or {}).items()}
        per_cam[str(cam)] = clean

    residuals = doc.get("timing_residuals_s")
    if _is_todo(residuals):
        residuals = None

    return ObservationModelParams(
        false_negative_rate=num("false_negative_rate"),
        false_positive_rate=num("false_positive_rate"),
        timing_jitter_std_s=num("timing_jitter_std_s"),
        timing_bias_s=num("timing_bias_s"),
        apply_timing_bias=bool(doc.get("apply_timing_bias", False)),
        per_camera=per_cam,
        timing_residuals_s=residuals,
        source=doc.get("source", "uncalibrated"),
        notes=doc.get("notes", ""),
    )


class SideChannelObservationModel:

    def __init__(self, false_negative_rate: float = 0.0,
                 false_positive_rate: float = 0.0,
                 timing_jitter_std_s: float = 0.0,
                 timing_bias_s: float = 0.0,
                 per_camera_params: dict | None = None,
                 jitter_mode: str = "gaussian",
                 timing_residuals_s: Sequence[float] | None = None,
                 random_seed: int = 0):
        self.false_negative_rate = float(false_negative_rate)
        self.false_positive_rate = float(false_positive_rate)
        self.timing_jitter_std_s = float(timing_jitter_std_s)
        self.timing_bias_s = float(timing_bias_s)
        self.per_camera_params = per_camera_params or {}
        self.jitter_mode = jitter_mode
        self.timing_residuals_s = (np.asarray(timing_residuals_s, dtype=float)
                                   if timing_residuals_s is not None else None)
        if jitter_mode == "empirical" and self.timing_residuals_s is None:
            raise ValueError("jitter_mode='empirical' requires timing_residuals_s")
        self.rng = np.random.RandomState(random_seed)
        self.random_seed = random_seed


    @classmethod
    def from_params(cls, params: ObservationModelParams, random_seed: int = 0,
                    jitter_mode: str = "gaussian",
                    require_calibrated: bool = True,
                    timing_bias_s: float | None = None):
        if require_calibrated and not params.is_calibrated:
            raise ValueError(
                "Observation model is not calibrated; missing "
                f"{params.missing_fields()}. Supply measured detector rates "
                "and timing jitter, or pass "
                "require_calibrated=False for a clearly-labelled development run.")
        return cls(
            false_negative_rate=params.false_negative_rate or 0.0,
            false_positive_rate=params.false_positive_rate or 0.0,
            timing_jitter_std_s=params.timing_jitter_std_s or 0.0,
            timing_bias_s=float(timing_bias_s or 0.0),
            per_camera_params=params.per_camera,
            jitter_mode=jitter_mode,
            timing_residuals_s=params.timing_residuals_s,
            random_seed=random_seed,
        )

    def _param_for(self, sensor_id: str, name: str, default: float) -> float:
        sub = self.per_camera_params.get(str(sensor_id))
        if sub and sub.get(name) is not None:
            return float(sub[name])
        return float(default)


    def corrupt(self, oracle_events: Sequence[Event], tracking_window,
                sensor_nodes) -> list[Event]:
        t0, t1 = float(tracking_window[0]), float(tracking_window[1])
        out: list[Event] = []


        for ev in oracle_events:
            p_drop = self._param_for(ev.sensor_id, "false_negative_rate",
                                     self.false_negative_rate)
            if self.rng.random_sample() < p_drop:
                continue
            out.append(Event(
                timestamp=ev.timestamp, sensor_id=ev.sensor_id,
                sensor_x=ev.sensor_x, sensor_y=ev.sensor_y,
                event_type=EVENT_PASSAGE, source=SOURCE_SIDECHANNEL_SIM,
                vehicle_id=ev.vehicle_id, confidence=ev.confidence,
                is_ghost=False, true_timestamp=ev.timestamp,
                t_enter=ev.t_enter, t_exit=ev.t_exit,
            ))


        interior = [s for s in sensor_nodes if s.role == SENSOR_INTERIOR]
        if interior and self.false_positive_rate > 0 and len(oracle_events) > 0:
            expected = self.false_positive_rate * len(oracle_events)
            n_ghost = int(self.rng.poisson(expected))
            for _ in range(n_ghost):
                s = interior[self.rng.randint(len(interior))]
                t = float(self.rng.uniform(t0, t1))
                out.append(Event(
                    timestamp=t, sensor_id=s.sensor_id,
                    sensor_x=s.x, sensor_y=s.y,
                    event_type=EVENT_PASSAGE, source=SOURCE_SIDECHANNEL_SIM,
                    vehicle_id=None,
                    is_ghost=True, true_timestamp=None,
                ))







        if (self.timing_jitter_std_s > 0 or self.jitter_mode == "empirical"
                or self.timing_bias_s != 0.0):
            for ev in out:
                if ev.is_ghost:
                    continue
                sigma = self._param_for(ev.sensor_id, "timing_jitter_std_s",
                                        self.timing_jitter_std_s)
                bias = self._param_for(ev.sensor_id, "timing_bias_s",
                                       self.timing_bias_s)
                if self.jitter_mode == "empirical":
                    delta = float(self.rng.choice(self.timing_residuals_s))
                elif sigma > 0:
                    delta = float(self.rng.normal(0.0, sigma))
                else:
                    delta = 0.0




                ev.timestamp = float(np.clip(ev.timestamp + bias + delta, t0, t1))

        out.sort(key=lambda e: e.timestamp)
        return out

    def describe(self) -> dict:
        return {
            "false_negative_rate": self.false_negative_rate,
            "false_positive_rate": self.false_positive_rate,
            "timing_jitter_std_s": self.timing_jitter_std_s,
            "timing_bias_s": self.timing_bias_s,
            "jitter_mode": self.jitter_mode,
            "n_per_camera_overrides": len(self.per_camera_params),
            "random_seed": self.random_seed,
        }
