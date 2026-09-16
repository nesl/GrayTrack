
from __future__ import annotations

import numpy as np

WEIGHTED_MEAN = "weighted_mean"
MAP_PARTICLE = "map_particle"
DOMINANT_MODE_MEAN = "dominant_mode_mean"
DOMINANT_MODE_MEAN_NBR = "dominant_mode_mean_nbr"
PROJECTED_MEAN = "projected_mean"



DEFAULT_READOUT = WEIGHTED_MEAN


READOUTS = (WEIGHTED_MEAN, MAP_PARTICLE, DOMINANT_MODE_MEAN,
            DOMINANT_MODE_MEAN_NBR, PROJECTED_MEAN)

READOUT_LABEL = {
    WEIGHTED_MEAN: "weighted mean",
    MAP_PARTICLE: "MAP particle",
    DOMINANT_MODE_MEAN: "dominant-mode mean",
    DOMINANT_MODE_MEAN_NBR: "dominant-mode mean (1-hop)",
    PROJECTED_MEAN: "projected mean",
}






def distance_to_graph(graph, points, k: int = 24):
    P = np.atleast_2d(np.asarray(points, dtype=float))
    n = len(P)
    dist = np.full(n, np.nan)
    proj = np.full((n, 2), np.nan)
    eidx = np.full(n, -1, dtype=int)
    ok = np.isfinite(P).all(axis=1)
    if not ok.any():
        return dist, proj, eidx

    Q = P[ok]
    kk = int(min(k, len(graph.sample_points)))
    _d, idx = graph.edge_tree.query(Q, k=kk)
    idx = np.atleast_2d(idx)
    cand = np.asarray(graph.sample_edge_idx)[idx]
    a = graph.edge_p0[cand]
    b = graph.edge_p1[cand]
    ab = b - a
    L2 = (ab ** 2).sum(axis=-1)
    safe = np.where(L2 > 0, L2, 1.0)
    t = np.where(L2 > 0, ((Q[:, None, :] - a) * ab).sum(axis=-1) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    pr = a + t[..., None] * ab
    d = np.linalg.norm(pr - Q[:, None, :], axis=-1)
    j = np.argmin(d, axis=1)
    rows = np.arange(len(Q))

    dist[ok] = d[rows, j]
    proj[ok] = pr[rows, j]
    eidx[ok] = cand[rows, j]
    return dist, proj, eidx


def project_point(graph, xy, k: int = 24):
    dist, proj, eidx = distance_to_graph(graph, np.asarray(xy, float)[None, :], k=k)
    return proj[0], float(dist[0]), int(eidx[0])










def weighted_mean(graph, r, e, w, xy):
    return np.average(xy, axis=1, weights=w)


def map_particle(graph, r, e, w, xy):
    return np.array(xy[:, int(np.argmax(w))], dtype=float)


def _edge_masses(e, w):
    ei = np.asarray(e).astype(int)
    occ, inv = np.unique(ei, return_inverse=True)
    mass = np.bincount(inv, weights=np.asarray(w, dtype=float), minlength=len(occ))
    return ei, occ, mass


def _mean_on_edge(edge, ei, w, xy):
    sel = ei == edge
    ww = np.asarray(w, dtype=float)[sel]
    if ww.sum() <= 0:
        return np.asarray(xy[:, sel]).mean(axis=1)
    return np.average(xy[:, sel], axis=1, weights=ww)


def dominant_mode_mean(graph, r, e, w, xy):
    ei, occ, mass = _edge_masses(e, w)
    best = int(occ[int(np.argmax(mass))])
    return _mean_on_edge(best, ei, w, xy)


def _edge_neighbours(graph, edge: int):
    cache = getattr(graph, "_readout_edge_neighbours", None)
    if cache is None:
        cache = {}
        setattr(graph, "_readout_edge_neighbours", cache)
    hit = cache.get(edge)
    if hit is None:
        fn, tn = graph.edge_list[edge]
        nb = set()
        for node in (fn, tn):
            nb.update(int(x) for x in graph.outgoing.get(node, []))
            nb.update(int(x) for x in graph.incoming.get(node, []))
        nb.add(int(edge))
        hit = np.fromiter(sorted(nb), dtype=int)
        cache[edge] = hit
    return hit


def dominant_mode_mean_nbr(graph, r, e, w, xy):
    ei, occ, mass = _edge_masses(e, w)
    massmap = {int(k): float(v) for k, v in zip(occ, mass)}
    scores = np.empty(len(occ), dtype=float)
    for i, edge in enumerate(occ):
        nb = _edge_neighbours(graph, int(edge))
        scores[i] = sum(massmap.get(int(x), 0.0) for x in nb)
    best = int(occ[int(np.argmax(scores))])
    return _mean_on_edge(best, ei, w, xy)


def projected_mean(graph, r, e, w, xy):
    mu = np.average(xy, axis=1, weights=w)
    p, _d, _e = project_point(graph, mu)
    return p


_FUNCS = {
    WEIGHTED_MEAN: weighted_mean,
    MAP_PARTICLE: map_particle,
    DOMINANT_MODE_MEAN: dominant_mode_mean,
    DOMINANT_MODE_MEAN_NBR: dominant_mode_mean_nbr,
    PROJECTED_MEAN: projected_mean,
}



































MIN_MODE_MASS = 0.08




EUCLID_LINK_M = 12.0

CLUSTERINGS = ("graph", "euclid")
REPRESENTATIVES = ("proj", "edge")


def _graph_components(graph, occ):
    n = len(occ)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    at_node: dict[int, int] = {}
    for i, edge in enumerate(occ):
        fn, tn = graph.edge_list[int(edge)]
        for node in (int(fn), int(tn)):
            prev = at_node.get(node)
            if prev is None:
                at_node[node] = i
            else:
                union(prev, i)
    roots = np.array([find(i) for i in range(n)])
    _u, comp = np.unique(roots, return_inverse=True)
    return comp


def _euclid_components(xy, radius=EUCLID_LINK_M):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    P = np.asarray(xy, dtype=float).T
    n = len(P)
    pairs = cKDTree(P).query_pairs(r=float(radius), output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n), n
    m = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(m, directed=False)
    return lab, ncomp


def _dissolve(masses, centres, min_mass):
    masses = np.asarray(masses, dtype=float)
    strong = np.flatnonzero(masses >= min_mass)
    if strong.size == 0:
        strong = np.array([int(np.argmax(masses))])
    target = np.empty(len(masses), dtype=int)
    sc = np.asarray(centres, dtype=float)[strong]
    for i in range(len(masses)):
        if i in set(strong.tolist()):
            target[i] = i
            continue
        d = np.linalg.norm(sc - np.asarray(centres[i], dtype=float), axis=1)
        target[i] = int(strong[int(np.argmin(d))])
    return target


def topk_candidates(graph, r, e, w, xy, k=5, min_mode_mass=MIN_MODE_MASS,
                    cluster="graph", rep="proj", radius=EUCLID_LINK_M):
    if cluster not in CLUSTERINGS:
        raise ValueError(f"unknown cluster {cluster!r}; have {CLUSTERINGS}")
    if rep not in REPRESENTATIVES:
        raise ValueError(f"unknown rep {rep!r}; have {REPRESENTATIVES}")
    w = np.asarray(w, dtype=float)
    xy = np.asarray(xy, dtype=float)
    ei, occ, emass = _edge_masses(e, w)

    if cluster == "graph":
        comp_of_edge = _graph_components(graph, occ)
        edge_pos = {int(edge): i for i, edge in enumerate(occ)}
        lab = comp_of_edge[np.array([edge_pos[int(x)] for x in ei])]
        ncomp = int(comp_of_edge.max()) + 1 if len(comp_of_edge) else 0
    else:
        lab, ncomp = _euclid_components(xy, radius)

    masses = np.bincount(lab, weights=w, minlength=ncomp)
    keep = np.flatnonzero(masses > 0)
    centres = np.zeros((ncomp, 2))
    for c in keep:
        sel = lab == c
        centres[c] = np.average(xy[:, sel], axis=1, weights=w[sel])


    target = np.full(ncomp, -1, dtype=int)
    tgt_keep = _dissolve(masses[keep], centres[keep], float(min_mode_mass))
    for i, c in enumerate(keep):
        target[c] = keep[tgt_keep[i]]
    merged: dict[int, float] = {}
    for c in keep:
        merged[int(target[c])] = merged.get(int(target[c]), 0.0) + float(masses[c])



    first_idx = {int(c): int(np.flatnonzero(np.isin(lab, np.flatnonzero(target == c)))[0])
                 for c in merged}
    order = sorted(merged, key=lambda c: (-merged[c], first_idx[c]))[:int(k)]

    pts = []
    for c in order:
        sel = np.isin(lab, np.flatnonzero(target == c))
        if rep == "proj":
            mu = np.average(xy[:, sel], axis=1, weights=w[sel])
            p, _d, _e = project_point(graph, mu)
        else:
            sub_e = ei[sel]
            occ_s, inv_s = np.unique(sub_e, return_inverse=True)
            m_s = np.bincount(inv_s, weights=w[sel], minlength=len(occ_s))
            best = int(occ_s[int(np.argmax(m_s))])
            p = _mean_on_edge(best, ei, w, xy)
        pts.append(np.asarray(p, dtype=float))

    mm = np.array([merged[c] for c in order], dtype=float)
    tot = mm.sum()
    return np.asarray(pts, dtype=float), (mm / tot if tot > 0 else mm)


def topk_diagnostics(graph, r, e, w, xy, radius=EUCLID_LINK_M) -> dict:
    ei, occ, emass = _edge_masses(e, w)
    w = np.asarray(w, dtype=float)
    out = {"n_occupied_edges": int(len(occ)),
           "top_edge_mass": float(emass.max()) if len(emass) else np.nan}
    comp = _graph_components(graph, occ)
    gm = np.bincount(comp, weights=emass, minlength=int(comp.max()) + 1)
    gm = np.sort(gm)[::-1]
    out["n_graph_components"] = int(len(gm))
    out["graph_top_mass"] = float(gm[0])
    out["graph_second_mass"] = float(gm[1]) if len(gm) > 1 else 0.0
    lab, nc = _euclid_components(np.asarray(xy, float), radius)
    em = np.sort(np.bincount(lab, weights=w, minlength=nc))[::-1]
    out["n_euclid_components"] = int(np.count_nonzero(em))
    out["euclid_top_mass"] = float(em[0])
    out["euclid_second_mass"] = float(em[1]) if len(em) > 1 else 0.0
    return out


def apply_readout(name: str, graph, r, e, w, xy):
    try:
        fn = _FUNCS[name]
    except KeyError:
        raise KeyError(f"unknown readout {name!r}; have {sorted(_FUNCS)}") from None
    return np.asarray(fn(graph, r, e, w, xy), dtype=float)






def posterior_diagnostics(graph, r, e, w, xy) -> dict:
    ei, occ, mass = _edge_masses(e, w)
    w = np.asarray(w, dtype=float)
    order = np.argsort(mass)[::-1]
    top = mass[order]
    mu = np.average(xy, axis=1, weights=w)
    spread = np.sqrt(np.average(((xy - mu[:, None]) ** 2).sum(axis=0), weights=w))
    return {
        "n_occupied_edges": int(len(occ)),
        "dominant_edge_mass": float(top[0]),
        "second_edge_mass": float(top[1]) if len(top) > 1 else 0.0,
        "ess_frac": float(1.0 / np.sum(w ** 2) / len(w)),
        "weights_uniform": bool(np.ptp(w) <= 1e-12),
        "cloud_spread_m": float(spread),
    }


__all__ = [
    "WEIGHTED_MEAN", "MAP_PARTICLE", "DOMINANT_MODE_MEAN",
    "DOMINANT_MODE_MEAN_NBR", "PROJECTED_MEAN", "READOUTS", "READOUT_LABEL",
    "DEFAULT_READOUT", "apply_readout", "distance_to_graph", "project_point",
    "posterior_diagnostics",
    "topk_candidates", "topk_diagnostics", "CLUSTERINGS", "REPRESENTATIVES",
    "MIN_MODE_MASS", "EUCLID_LINK_M",
]
