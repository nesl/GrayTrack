
from __future__ import annotations

import os
import pickle
from functools import lru_cache

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(_SRC_DIR)

DEFAULT_NETWORK_PKL = os.path.join(EXP_ROOT, "data", "Town05_road_net_v3.pkl")


def angle_diff(a, b):
    d = np.asarray(a) - np.asarray(b)
    return (d + np.pi) % (2 * np.pi) - np.pi


class RoadGraph:

    def __init__(self, pkl_path: str | None = None):
        if pkl_path is None:
            pkl_path = DEFAULT_NETWORK_PKL
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Town05 road network pickle not found at {pkl_path}. "
                "Expected data/Town05_road_net_v3.pkl (tracked in git)."
            )
        self.pkl_path = pkl_path

        from stonesoup.types.graph import RoadNetwork

        with open(pkl_path, "rb") as f:
            net_dict = pickle.load(f)
        self.road_net = RoadNetwork.from_dict(net_dict)
        self.node_pos = nx.get_node_attributes(self.road_net, "pos")
        self.edge_list = list(self.road_net.edge_list)
        self.n_edges = len(self.edge_list)

        p0 = np.array([self.node_pos[fn] for fn, _ in self.edge_list], dtype=float)
        p1 = np.array([self.node_pos[tn] for _, tn in self.edge_list], dtype=float)
        delta = p1 - p0
        self.edge_len = np.linalg.norm(delta, axis=1)
        self.edge_dir = np.arctan2(delta[:, 1], delta[:, 0])
        self.edge_p0 = p0
        self.edge_p1 = p1

        self.outgoing: dict[int, list[int]] = {}
        for idx, (fn, _tn) in enumerate(self.edge_list):
            self.outgoing.setdefault(fn, []).append(idx)

        self.incoming: dict[int, list[int]] = {}
        for idx, (_fn, tn) in enumerate(self.edge_list):
            self.incoming.setdefault(tn, []).append(idx)

        self._build_edge_kdtree()




    def _build_edge_kdtree(self, spacing: float = 0.5):
        pts, eidx, rvals = [], [], []
        for idx in range(self.n_edges):
            elen = float(self.edge_len[idx])
            if elen < 0.01:
                continue
            n_s = max(2, int(elen / spacing) + 1)
            ts = np.linspace(0.0, 1.0, n_s)
            seg = self.edge_p0[idx][None, :] + ts[:, None] * (
                self.edge_p1[idx] - self.edge_p0[idx]
            )[None, :]
            pts.append(seg)
            eidx.append(np.full(n_s, idx, dtype=int))
            rvals.append(ts * elen)
        self.sample_points = np.vstack(pts)
        self.sample_edge_idx = np.concatenate(eidx)
        self.sample_r = np.concatenate(rvals)
        self.edge_tree = cKDTree(self.sample_points)




    def xy_from_re(self, r, e):
        from stonesoup.functions.graph import get_xy_from_range_edge

        return get_xy_from_range_edge(np.asarray(r, dtype=float),
                                      np.asarray(e, dtype=float), self.road_net)

    def snap(self, xy):
        _, idx = self.edge_tree.query(np.asarray(xy, dtype=float), k=1)
        idx = int(idx)
        return (self.sample_points[idx].copy(),
                int(self.sample_edge_idx[idx]),
                float(self.sample_r[idx]))

    def shortest_path_edges(self, src: int, dst: int):
        info = self.road_net.shortest_path(src, dst, path_type="both")
        return list(info["node"][(src, dst)]), list(info["edge"][(src, dst)])

    def route_length(self, route_edges) -> float:
        return float(sum(self.edge_len[e] for e in route_edges))




    @lru_cache(maxsize=1)
    def _node_bbox(self):
        arr = np.array(list(self.node_pos.values()), dtype=float)
        return arr[:, 0].min(), arr[:, 0].max(), arr[:, 1].min(), arr[:, 1].max()

    @property
    def bbox(self):
        return self._node_bbox()

    def branch_factor(self, edge_idx: int) -> int:
        _fn, tn = self.edge_list[edge_idx]
        return len(self.outgoing.get(tn, []))

    def route_topology_stats(self, route_edges) -> dict:
        if len(route_edges) == 0:
            return {"n_edges": 0, "n_junctions": 0, "mean_branch": 0.0,
                    "max_branch": 0, "junctions_per_km": 0.0}
        branches = [self.branch_factor(int(e)) for e in route_edges]
        length_km = max(self.route_length(route_edges) / 1000.0, 1e-6)
        n_junctions = int(sum(1 for b in branches if b > 1))
        return {
            "n_edges": int(len(route_edges)),
            "n_junctions": n_junctions,
            "mean_branch": float(np.mean(branches)),
            "max_branch": int(np.max(branches)),
            "junctions_per_km": float(n_junctions / length_km),
        }


_GRAPH_CACHE: dict[str, RoadGraph] = {}


def load_road_graph(pkl_path: str | None = None) -> RoadGraph:
    key = pkl_path or "__default__"
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = RoadGraph(pkl_path)
    return _GRAPH_CACHE[key]


if __name__ == "__main__":
    g = load_road_graph()
    print(f"pickle       : {g.pkl_path}")
    print(f"nodes        : {g.road_net.number_of_nodes()}")
    print(f"edges        : {g.n_edges}")
    print(f"bbox (x,y)   : {g.bbox}")
    print(f"edge len     : mean={g.edge_len.mean():.2f} m  max={g.edge_len.max():.2f} m")
    print(f"kdtree pts   : {len(g.sample_points)}")
