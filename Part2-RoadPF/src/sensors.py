
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .schema import Sensor, SENSOR_INTERIOR, SENSOR_PERIMETER


@dataclass(frozen=True)
class Region:

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, xy) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        return ((xy[:, 0] >= self.x_min) & (xy[:, 0] <= self.x_max)
                & (xy[:, 1] >= self.y_min) & (xy[:, 1] <= self.y_max))

    def contains_point(self, x: float, y: float) -> bool:
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area_km2(self) -> float:
        return self.width * self.height / 1e6

    def as_dict(self) -> dict:
        return {"x_min": self.x_min, "x_max": self.x_max,
                "y_min": self.y_min, "y_max": self.y_max,
                "width_m": self.width, "height_m": self.height,
                "area_km2": self.area_km2}






def classify_nodes(graph, region: Region):
    inside, outside = set(), set()
    for n, p in graph.node_pos.items():
        (inside if region.contains_point(p[0], p[1]) else outside).add(n)
    return inside, outside


def find_boundary_crossings(graph, region: Region):
    inside, _outside = classify_nodes(graph, region)
    inbound, outbound = [], []
    for idx, (fn, tn) in enumerate(graph.edge_list):
        f_in, t_in = fn in inside, tn in inside
        if f_in == t_in:
            continue
        p0 = np.asarray(graph.node_pos[fn], dtype=float)
        p1 = np.asarray(graph.node_pos[tn], dtype=float)
        cross = _segment_boundary_point(p0, p1, region)
        rec = (idx, int(fn), int(tn), cross)
        if t_in:
            inbound.append(rec)
        else:
            outbound.append(rec)
    return inbound, outbound


def _segment_boundary_point(p0, p1, region: Region) -> np.ndarray:
    a, b = 0.0, 1.0
    inside_a = region.contains_point(*p0)
    for _ in range(30):
        m = 0.5 * (a + b)
        pm = p0 + m * (p1 - p0)
        if region.contains_point(*pm) == inside_a:
            a = m
        else:
            b = m
    return p0 + 0.5 * (a + b) * (p1 - p0)


def place_perimeter_sensors(graph, region: Region, fov_radius: float = 15.0,
                            merge_radius: float = 20.0):
    inbound, outbound = find_boundary_crossings(graph, region)
    all_cross = inbound + outbound
    if not all_cross:
        return [], {"inbound_crossings": 0, "outbound_crossings": 0,
                    "merged_sensors": 0, "merge_radius_m": merge_radius}, {}

    pts = np.array([c[3] for c in all_cross], dtype=float)
    order = np.lexsort((pts[:, 1], pts[:, 0]))

    sensors: list[Sensor] = []
    placed: list[np.ndarray] = []
    tree = None
    edge_map: dict[int, str] = {}

    for i in order:
        idx, fn, tn, cross = all_cross[i]
        if tree is not None:
            d, j = tree.query(cross)
            if d < merge_radius:
                edge_map[idx] = sensors[int(j)].sensor_id
                continue
        sid = f"P{len(sensors):02d}"
        sensors.append(Sensor(
            sensor_id=sid, role=SENSOR_PERIMETER,
            x=float(cross[0]), y=float(cross[1]),
            fov_radius=fov_radius, edge_idx=int(idx),
            node_id=int(tn),
        ))
        placed.append(cross)
        tree = cKDTree(np.array(placed))
        edge_map[idx] = sid

    stats = {
        "inbound_crossings": len(inbound),
        "outbound_crossings": len(outbound),
        "merged_sensors": len(sensors),
        "merge_radius_m": merge_radius,
    }
    return sensors, stats, edge_map


def place_interior_sensors(graph, region: Region, fov_radius: float = 25.0,
                           density: float = 1.0, seed: int = 42,
                           min_edge_len: float = 1.5,
                           margin: float = 30.0):
    rng = np.random.RandomState(seed)
    min_spacing = 2.0 * fov_radius

    inner = Region(region.x_min + margin, region.x_max - margin,
                   region.y_min + margin, region.y_max - margin)

    candidates = []
    for idx in range(graph.n_edges):
        if graph.edge_len[idx] < min_edge_len:
            continue
        mid = 0.5 * (graph.edge_p0[idx] + graph.edge_p1[idx])
        if not inner.contains_point(mid[0], mid[1]):
            continue
        candidates.append((idx, mid))

    order = rng.permutation(len(candidates))
    sensors: list[Sensor] = []
    placed: list[np.ndarray] = []
    tree = None

    for i in order:
        idx, mid = candidates[int(i)]
        if tree is not None:
            d, _ = tree.query(mid)
            if d < min_spacing:
                continue
        sensors.append(Sensor(
            sensor_id=f"I{len(sensors):02d}", role=SENSOR_INTERIOR,
            x=float(mid[0]), y=float(mid[1]),
            fov_radius=fov_radius, edge_idx=int(idx),
        ))
        placed.append(mid)
        tree = cKDTree(np.array(placed))

    if density < 1.0 and sensors:
        n_keep = max(1, int(round(len(sensors) * density)))
        keep = sorted(rng.choice(len(sensors), n_keep, replace=False))
        sensors = [Sensor(sensor_id=f"I{i:02d}", role=sensors[k].role,
                          x=sensors[k].x, y=sensors[k].y,
                          fov_radius=sensors[k].fov_radius,
                          edge_idx=sensors[k].edge_idx, node_id=sensors[k].node_id)
                   for i, k in enumerate(keep)]

    return sensors


def sensor_tree(sensors):
    if not sensors:
        return None, []
    pts = np.array([[s.x, s.y] for s in sensors], dtype=float)
    return cKDTree(pts), list(sensors)


def coverage_fraction(sensors, xy) -> float:
    tree, slist = sensor_tree(sensors)
    if tree is None:
        return 0.0
    d, i = tree.query(np.asarray(xy, dtype=float), k=1)
    radii = np.array([s.fov_radius for s in slist])
    return float(np.mean(d <= radii[i]))
