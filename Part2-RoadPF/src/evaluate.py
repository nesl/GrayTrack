
from __future__ import annotations

import numpy as np


def position_errors(est_xy, gt_xy) -> np.ndarray:
    est_xy = np.asarray(est_xy, dtype=float)
    gt_xy = np.asarray(gt_xy, dtype=float)
    return np.sqrt(np.sum((est_xy - gt_xy) ** 2, axis=1))


def trajectory_metrics(est_xy, traj, fill_missing: bool = True) -> dict:
    est_xy = np.asarray(est_xy, dtype=float)
    sl = traj.active_slice
    est = est_xy[sl]
    gt = traj.xy[sl]
    n = len(gt)
    if n == 0:
        return _empty_metrics()

    valid = ~np.isnan(est).any(axis=1)
    coverage = float(valid.mean())
    if coverage == 0.0:
        return _empty_metrics(coverage=0.0, n_steps=n)

    if fill_missing and not valid.all():
        est = _forward_fill(est, valid)

    err = position_errors(est, gt)
    err = err[~np.isnan(err)]
    if err.size == 0:
        return _empty_metrics(coverage=coverage, n_steps=n)

    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(err)),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
        "final_error": float(err[-1]),
        "coverage": coverage,
        "n_steps": int(n),
    }


def _forward_fill(est, valid):
    est = est.copy()
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return est

    est[:idx[0]] = est[idx[0]]
    last = est[idx[0]]
    for i in range(idx[0], len(est)):
        if valid[i]:
            last = est[i]
        else:
            est[i] = last
    return est


def _empty_metrics(coverage=0.0, n_steps=0):
    return {"rmse": np.nan, "mae": np.nan, "median": np.nan, "p95": np.nan,
            "max": np.nan, "final_error": np.nan,
            "coverage": coverage, "n_steps": n_steps}






def is_catastrophic(metrics: dict, threshold_m: float = 100.0) -> bool:
    r = metrics.get("rmse")
    return bool(r is not None and np.isfinite(r) and r > threshold_m)


def association_accuracy(assignments, events) -> float:
    if not events:
        return float("nan")
    correct = 0
    for i, ev in enumerate(events):
        got = assignments.get(i)
        if ev.is_ghost or ev.vehicle_id is None:
            correct += int(got is None)
        else:
            correct += int(got == ev.vehicle_id)
    return correct / len(events)






def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0, statistic=np.mean):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, v.size, size=(n_boot, v.size))
    stats = statistic(v[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(v)), float(lo), float(hi)


def summarise(values, n_boot: int = 10_000, seed: int = 0) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    mean, lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=seed)
    return {
        "n": int(v.size),
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "sem": float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
        "median": float(np.median(v)) if v.size else float("nan"),
    }
























BRANCH_BINS = (
    ("straight_or_low_branching", -np.inf, 32.0),
    ("moderate_branching", 32.0, 37.0),
    ("junction_heavy", 37.0, np.inf),
)

BLIND_GAP_BINS = (
    ("short_gap", -np.inf, 10.0),
    ("medium_gap", 10.0, 15.0),
    ("long_gap", 15.0, np.inf),
)

DIFFICULTY_RULE = {
    "primary": "branch_density",
    "branch_density": (
        "forks per km over the in-region portion of the route; a fork is a "
        "traversed edge whose end node has >1 outgoing edge (the Town05 graph "
        "is lane-level, so out-degree >= 3 is not the right test). "
        "Bins: straight_or_low_branching < 32, moderate_branching 32-37, "
        "junction_heavy >= 37 (chosen once as round numbers near the tertiles "
        "of the 120-trajectory main dataset: 32.8 and 36.2; strata 28/58/34)."),
    "blind_gap": (
        "longest interval (s) between consecutive ORACLE interior events, "
        "including the entry->first-event and last-event->exit intervals. "
        "Bins: short_gap < 10, medium_gap 10-15, long_gap >= 15."),
    "note": ("Both axes depend only on the trajectory and the oracle event "
             "set, so a trajectory keeps the same stratum across every "
             "condition, corruption level and estimator."),
}


DIFFICULTY_BINS = BRANCH_BINS


def _bin(value, bins):
    for name, lo, hi in bins:
        if lo <= value < hi:
            return name
    return bins[-1][0]


def branch_density(traj) -> float:
    stats = traj.meta.get("inside_topology") or traj.meta.get("route_topology") or {}
    return float(stats.get("junctions_per_km", 0.0))


def blind_gap(traj) -> float:
    return float(traj.meta.get("max_blind_gap_s", np.nan))


def classify_difficulty(traj) -> str:
    return _bin(branch_density(traj), BRANCH_BINS)


def classify_blind_gap(traj) -> str:
    g = blind_gap(traj)
    if not np.isfinite(g):
        return "unknown"
    return _bin(g, BLIND_GAP_BINS)


def difficulty_table(trajs) -> dict:
    out = {"rule": DIFFICULTY_RULE, "branch_density": {}, "blind_gap": {}}
    for axis, bins, fn, cls in (
            ("branch_density", BRANCH_BINS, branch_density, classify_difficulty),
            ("blind_gap", BLIND_GAP_BINS, blind_gap, classify_blind_gap)):
        buckets = {name: [] for name, _lo, _hi in bins}
        buckets["unknown"] = []
        for tr in trajs:
            buckets.setdefault(cls(tr), []).append(fn(tr))
        for name, vals in buckets.items():
            vals = [v for v in vals if np.isfinite(v)]
            if not vals and name == "unknown":
                continue
            out[axis][name] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "min": float(np.min(vals)) if vals else float("nan"),
                "max": float(np.max(vals)) if vals else float("nan"),
            }
    return out
