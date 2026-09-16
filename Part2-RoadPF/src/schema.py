
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd







SOURCE_PERIMETER = "perimeter"
SOURCE_ORACLE_INTERIOR = "oracle_interior"
SOURCE_SIDECHANNEL_SIM = "sidechannel_sim"
SOURCE_SIDECHANNEL_REAL = "sidechannel_real"
SOURCES = (SOURCE_PERIMETER, SOURCE_ORACLE_INTERIOR,
           SOURCE_SIDECHANNEL_SIM, SOURCE_SIDECHANNEL_REAL)


EVENT_ENTRY = "entry"
EVENT_EXIT = "exit"
EVENT_PASSAGE = "passage"
EVENT_TYPES = (EVENT_ENTRY, EVENT_EXIT, EVENT_PASSAGE)


SENSOR_PERIMETER = "perimeter"
SENSOR_INTERIOR = "interior"






@dataclass
class Sensor:

    sensor_id: str
    role: str
    x: float
    y: float
    fov_radius: float
    edge_idx: Optional[int] = None
    node_id: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Event:

    timestamp: float
    sensor_id: str
    sensor_x: float
    sensor_y: float
    event_type: str
    source: str
    vehicle_id: Optional[int] = None
    confidence: Optional[float] = None


    is_ghost: bool = False

    true_timestamp: Optional[float] = None





    t_enter: Optional[float] = None
    t_exit: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trajectory:

    trajectory_id: str
    vehicle_id: int
    timestamps: np.ndarray
    xy: np.ndarray
    speed: Optional[np.ndarray] = None
    heading: Optional[np.ndarray] = None
    road_edge_id: Optional[np.ndarray] = None



    active_start: int = 0
    active_end: int = -1
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.xy = np.asarray(self.xy, dtype=float)
        if self.active_end < 0:
            self.active_end = len(self.timestamps) - 1

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)

    @property
    def active_slice(self) -> slice:
        return slice(self.active_start, self.active_end + 1)

    def to_frame(self) -> pd.DataFrame:
        n = self.n_steps
        data = {
            "trajectory_id": [self.trajectory_id] * n,
            "vehicle_id": [self.vehicle_id] * n,
            "timestamp": self.timestamps,
            "x": self.xy[:, 0],
            "y": self.xy[:, 1],
        }
        if self.road_edge_id is not None:
            data["road_edge_id"] = self.road_edge_id
        if self.speed is not None:
            data["speed"] = self.speed
        if self.heading is not None:
            data["heading"] = self.heading
        return pd.DataFrame(data)






def events_to_frame(events: Sequence[Event]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=[
            "timestamp", "sensor_id", "sensor_x", "sensor_y", "event_type",
            "source", "vehicle_id", "confidence", "is_ghost", "true_timestamp"])
    return pd.DataFrame([e.as_dict() for e in events])


def frame_to_events(df: pd.DataFrame) -> list[Event]:
    out = []
    for rec in df.to_dict("records"):
        vid = rec.get("vehicle_id")
        if vid is not None and not (isinstance(vid, float) and np.isnan(vid)):
            vid = int(vid)
        else:
            vid = None
        conf = rec.get("confidence")
        if conf is not None and isinstance(conf, float) and np.isnan(conf):
            conf = None
        tts = rec.get("true_timestamp")
        if tts is not None and isinstance(tts, float) and np.isnan(tts):
            tts = None
        out.append(Event(
            timestamp=float(rec["timestamp"]),
            sensor_id=str(rec["sensor_id"]),
            sensor_x=float(rec["sensor_x"]),
            sensor_y=float(rec["sensor_y"]),
            event_type=str(rec["event_type"]),
            source=str(rec["source"]),
            vehicle_id=vid,
            confidence=conf,
            is_ghost=bool(rec.get("is_ghost", False)),
            true_timestamp=tts,
        ))
    return out


def save_events(events: Sequence[Event], path: str) -> None:
    events_to_frame(events).to_csv(path, index=False)


def load_events(path: str) -> list[Event]:
    return frame_to_events(pd.read_csv(path))


def save_sensors(sensors: Sequence[Sensor], path: str, extra: dict | None = None) -> None:
    doc = {
        "n_sensors": len(sensors),
        "n_perimeter": sum(1 for s in sensors if s.role == SENSOR_PERIMETER),
        "n_interior": sum(1 for s in sensors if s.role == SENSOR_INTERIOR),
        "sensors": [s.as_dict() for s in sensors],
    }
    if extra:
        doc.update(extra)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def load_sensors(path: str) -> list[Sensor]:
    with open(path) as f:
        doc = json.load(f)
    return [Sensor(**s) for s in doc["sensors"]]


def split_events(events: Sequence[Event]):
    perim = [e for e in events if e.source == SOURCE_PERIMETER]
    interior = [e for e in events if e.source != SOURCE_PERIMETER]
    return perim, interior


def sort_events(events: Sequence[Event]) -> list[Event]:
    return sorted(events, key=lambda e: (e.timestamp, e.source != SOURCE_PERIMETER))
