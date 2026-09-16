
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.experiment import (COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR,
                            COND_SIDECHANNEL, build_event_stream, run_jobs,
                            save_results, aggregate, load_dataset_config)
from src.dataset import build_dataset
from src import detector_calibration as dc
from src.observation_model import SideChannelObservationModel, load_params
from src.evaluate import (difficulty_table, DIFFICULTY_RULE, BRANCH_BINS,
                          BLIND_GAP_BINS, bootstrap_ci)
from src.road_graph import EXP_ROOT

RESULTS = os.path.join(EXP_ROOT, "results")
FIGURES = os.path.join(EXP_ROOT, "figures")


ESTIMATOR_ORDER = ["dead_reckoning", "kalman", "road_pf"]
ESTIMATOR_LABEL = {"dead_reckoning": "Dead reckoning",
                   "kalman": "Kalman filter",
                   "road_pf": "Road-PF"}



ESTIMATOR_COLOR = {"dead_reckoning": "#0072B2", "kalman": "#D55E00",
                   "road_pf": "#009E73"}
ESTIMATOR_HATCH = {"dead_reckoning": "", "kalman": "///", "road_pf": "..."}

CONDITION_LABEL = {COND_PERIMETER_ONLY: "A: perimeter only",
                   COND_ORACLE_INTERIOR: "B: oracle interior",
                   COND_SIDECHANNEL: "C: side-channel"}
CONDITION_SHORT = {COND_PERIMETER_ONLY: "A\nperimeter\nonly",
                   COND_ORACLE_INTERIOR: "B\noracle\ninterior",
                   COND_SIDECHANNEL: "C\nside-\nchannel"}
STRATUM_LABEL = {"straight_or_low_branching": "low\nbranching",
                 "moderate_branching": "moderate\nbranching",
                 "junction_heavy": "junction\nheavy",
                 "short_gap": "short\ngap", "medium_gap": "medium\ngap",
                 "long_gap": "long\ngap"}

BRANCH_ORDER = [name for name, _lo, _hi in BRANCH_BINS]
GAP_ORDER = [name for name, _lo, _hi in BLIND_GAP_BINS]





DEV_CORRUPTION = {"false_negative_rate": 0.10,
                  "false_positive_rate": 0.10,
                  "timing_jitter_std_s": 0.5}






def resolve_observation_model(args, profile=None):
    if args.dev_corruption:
        return dict(DEV_CORRUPTION), "DEVELOPMENT_PLACEHOLDER", (
            "hand-set values for pipeline debugging — NOT a paper result")

    if args.obs_model:
        path = args.obs_model
    else:
        path = (profile or dc.load_profile()).observation_model_path
    params = load_params(path)
    if not params.is_calibrated:
        raise SystemExit(
            f"\nObservation model at {path} is not calibrated "
            f"(missing: {params.missing_fields()}).\n"
            "Condition C cannot be run as a paper result yet. Either:\n"
            "  1. select a calibrated --detector_profile or --obs_model, or\n"
            "  2. re-run with --dev_corruption for a clearly-labelled\n"
            "     development run, or\n"
            "  3. pass --skip_condition_c to run A and B only.\n")
    return ({"false_negative_rate": params.false_negative_rate,
             "false_positive_rate": params.false_positive_rate,
             "timing_jitter_std_s": params.timing_jitter_std_s,



             "timing_bias_s": ((params.timing_bias_s or 0.0)
                               if params.apply_timing_bias else 0.0),
             "per_camera_params": params.per_camera},
            os.path.basename(path), params.source)


def corruption_seed(trajectory_id: str, s: int) -> int:
    # Preserve the original seed namespace so renaming does not change results.
    return int(zlib.crc32(f"rq4|{trajectory_id}|{s}".encode())) % (2 ** 31)






def build_jobs(ds, estimators, conditions, est_kwargs, obs_kwargs, n_seeds,
               calib_name):
    jobs = []
    for est in estimators:
        for tr in ds.trajectories:
            for cond in (COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR):
                if cond not in conditions:
                    continue
                jobs.append({"trajectory_id": tr.trajectory_id, "estimator": est,
                             "condition": cond, "obs": None,
                             "est_kwargs": {**est_kwargs, "seed": 0},
                             "tags": {"corruption_seed": -1,
                                      "calibration": "n/a"}})
            if COND_SIDECHANNEL not in conditions:
                continue
            for s in range(n_seeds):
                jobs.append({
                    "trajectory_id": tr.trajectory_id, "estimator": est,
                    "condition": COND_SIDECHANNEL,
                    "obs": {**obs_kwargs,
                            "random_seed": corruption_seed(tr.trajectory_id, s)},
                    "est_kwargs": {**est_kwargs, "seed": s},
                    "tags": {"corruption_seed": s, "calibration": calib_name}})
    return jobs






def _digest_stream(ds, traj, condition, obs):
    obs_model = SideChannelObservationModel(**obs) if obs else None
    events, interior = build_event_stream(ds, traj, condition, obs_model)
    blob = json.dumps([[round(float(e.timestamp), 9), str(e.sensor_id),
                        e.event_type, e.source, round(float(e.sensor_x), 9),
                        round(float(e.sensor_y), 9), bool(e.is_ghost)]
                       for e in events])
    return (hashlib.sha1(blob.encode()).hexdigest()[:16], len(events),
            len(interior))


def verify_event_streams(ds, jobs):
    by_id = {t.trajectory_id: t for t in ds.trajectories}
    cells: dict[tuple, dict[str, tuple]] = {}
    for job in jobs:
        key = (job["condition"], job["trajectory_id"],
               job["tags"]["corruption_seed"])
        cells.setdefault(key, {})[job["estimator"]] = _digest_stream(
            ds, by_id[job["trajectory_id"]], job["condition"], job.get("obs"))

    bad = [k for k, v in cells.items() if len({d for d, _n, _i in v.values()}) > 1]
    if bad:
        raise AssertionError(
            f"event streams differ across estimators in {len(bad)} cells, "
            f"e.g. {bad[:3]} — the estimator comparison would be invalid")
    n_est = {len(v) for v in cells.values()}
    return {"check": "identical_event_streams_pre_run",
            "n_cells": len(cells),
            "estimators_per_cell": sorted(n_est),
            "n_cells_with_mismatched_digest": 0,
            "total_events_hashed": int(sum(n for v in cells.values()
                                           for _d, n, _i in v.values())),
            "passed": True}


def verify_run_rows(df, estimators):
    keys = ["condition", "trajectory_id", "corruption_seed"]
    cols = ["n_interior_events_used", "n_true_events_used",
            "n_ghost_events_used", "n_oracle_interior_events", "n_steps"]
    g = df.groupby(keys, dropna=False)
    mismatch = {c: int((g[c].nunique() > 1).sum()) for c in cols}
    n_est = g["estimator"].nunique()
    strata = df.groupby("trajectory_id")[["difficulty", "blind_gap_stratum"]].nunique()
    report = {
        "check": "identical_event_streams_post_run",
        "n_cells": int(len(n_est)),
        "cells_missing_an_estimator": int((n_est != len(estimators)).sum()),
        "cells_with_disagreeing_counts": mismatch,
        "trajectories_with_unstable_stratum": int(
            (strata["difficulty"] > 1).sum() + (strata["blind_gap_stratum"] > 1).sum()),
        "passed": (all(v == 0 for v in mismatch.values())
                   and int((n_est != len(estimators)).sum()) == 0
                   and int((strata > 1).sum().sum()) == 0),
    }
    if not report["passed"]:
        raise AssertionError(f"post-run identity check failed: {report}")
    return report


def kf_equals_dr_check(df):
    out = []
    for cond, sub in df.groupby("condition"):
        piv = sub.pivot_table(index=["trajectory_id", "corruption_seed"],
                              columns="estimator", values="rmse")
        if not {"kalman", "dead_reckoning"} <= set(piv.columns):
            continue
        d = (piv["kalman"] - piv["dead_reckoning"]).abs()
        rel = d / piv["dead_reckoning"].abs()
        out.append({
            "condition": cond, "n_pairs": int(d.notna().sum()),
            "n_identical_exact": int((d == 0).sum()),
            "n_identical_1e-6m": int((d < 1e-6).sum()),
            "frac_identical_1e-6m": float((d < 1e-6).mean()),
            "max_abs_rmse_diff_m": float(d.max()),
            "mean_abs_rmse_diff_m": float(d.mean()),
            "max_rel_rmse_diff": float(rel.max()),
        })
    return out






def by_difficulty(df, n_boot=10_000):
    frames = []
    for axis, col, order in (("branch_density", "difficulty", BRANCH_ORDER),
                             ("blind_gap", "blind_gap_stratum", GAP_ORDER)):
        a = aggregate(df, ["estimator", "condition", col], n_boot=n_boot)
        a = a.rename(columns={col: "stratum"})
        a.insert(0, "axis", axis)
        a["stratum"] = pd.Categorical(a["stratum"], order + ["unknown"],
                                      ordered=True)
        a["estimator"] = pd.Categorical(a["estimator"], ESTIMATOR_ORDER,
                                        ordered=True)
        frames.append(a.sort_values(["condition", "stratum", "estimator"]))
    out = pd.concat(frames, ignore_index=True)
    cols = ["axis", "stratum", "condition", "estimator", "n", "mean", "ci_lo",
            "ci_hi", "std", "sem", "median", "p95_mean", "max_mean", "mae_mean",
            "coverage_mean", "catastrophic_rate"]
    return out[[c for c in cols if c in out.columns]]


def stratum_profile(df):
    cols = [c for c in ("branch_density_per_km", "max_blind_gap_s",
                        "inside_length_m", "transit_duration_s",
                        "interior_coverage", "n_oracle_interior_events")
            if c in df.columns]
    per = df.groupby("trajectory_id").first()[cols + ["difficulty",
                                                      "blind_gap_stratum"]]
    out = {}
    for axis, col, order in (("branch_density", "difficulty", BRANCH_ORDER),
                             ("blind_gap", "blind_gap_stratum", GAP_ORDER)):
        rec = {}
        for name in order:
            g = per[per[col] == name]
            if g.empty:
                continue
            rec[name] = {"n": int(len(g)),
                         **{c: float(g[c].mean()) for c in cols}}
        out[axis] = rec
    out["axis_crosstab"] = {
        f"{a}|{b}": int(v) for (a, b), v in
        per.groupby(["difficulty", "blind_gap_stratum"]).size().items()}
    out["axis_correlation_pearson"] = float(
        per["branch_density_per_km"].corr(per["max_blind_gap_s"]))
    return out


def _per_trajectory(df, condition):
    sub = df[df["condition"] == condition]
    return sub.pivot_table(index="trajectory_id", columns="estimator",
                           values="rmse", aggfunc="mean")


def hypothesis_test(df, condition, baseline="kalman", champion="road_pf",
                    n_boot=10_000, seed=0):
    piv = _per_trajectory(df, condition)
    if not {baseline, champion} <= set(piv.columns):
        return {}
    strat = (df[df["condition"] == condition]
             .groupby("trajectory_id")[["difficulty", "blind_gap_stratum"]].first())
    piv = piv.join(strat)
    piv["gain_m"] = piv[baseline] - piv[champion]
    piv["gain_pct"] = 100.0 * piv["gain_m"] / piv[baseline]

    out = {"condition": condition, "baseline": baseline, "champion": champion,
           "axes": {}}
    for axis, col, order in (("branch_density", "difficulty", BRANCH_ORDER),
                             ("blind_gap", "blind_gap_stratum", GAP_ORDER)):
        rec = {}
        for name in order:
            g = piv[piv[col] == name]
            if g.empty:
                continue
            m, lo, hi = bootstrap_ci(g["gain_m"].values, n_boot=n_boot, seed=seed)
            rec[name] = {
                "n": int(len(g)),
                f"{baseline}_rmse_m": float(g[baseline].mean()),
                f"{champion}_rmse_m": float(g[champion].mean()),
                "gain_m": m, "gain_ci_lo": lo, "gain_ci_hi": hi,
                "gain_pct": float(100.0 * m / g[baseline].mean()),
                "significant": bool(lo > 0),
            }
        easy, hard = order[0], order[-1]
        if easy in rec and hard in rec:
            rng = np.random.RandomState(seed)
            ge = piv.loc[piv[col] == easy, "gain_m"].values
            gh = piv.loc[piv[col] == hard, "gain_m"].values
            be = ge[rng.randint(0, len(ge), (n_boot, len(ge)))].mean(axis=1)
            bh = gh[rng.randint(0, len(gh), (n_boot, len(gh)))].mean(axis=1)
            d = bh - be
            lo, hi = np.percentile(d, [2.5, 97.5])
            rec["interaction"] = {
                "comparison": f"{hard} minus {easy}",
                "delta_gain_m": float(gh.mean() - ge.mean()),
                "ci_lo": float(lo), "ci_hi": float(hi),
                "p_one_sided_gt_0": float((d <= 0).mean()),
                "supports_hypothesis": bool(lo > 0),
            }
        out["axes"][axis] = rec
    return out


def _boot_pearson(x, y, idx):
    xs, ys = x[idx], y[idx]
    xs = xs - xs.mean(axis=1, keepdims=True)
    ys = ys - ys.mean(axis=1, keepdims=True)
    den = np.sqrt((xs ** 2).sum(axis=1) * (ys ** 2).sum(axis=1))
    r = np.where(den > 0, (xs * ys).sum(axis=1) / np.where(den > 0, den, 1), np.nan)
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return float(lo), float(hi)


def _residualise(y, z):
    A = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def gain_vs_covariates(df, condition, baseline="kalman", champion="road_pf",
                       n_boot=10_000, seed=0):
    sub = df[df["condition"] == condition]
    piv = sub.pivot_table(index="trajectory_id", columns="estimator",
                          values="rmse", aggfunc="mean")
    if not {baseline, champion} <= set(piv.columns):
        return {}
    cols = [c for c in ("branch_density_per_km", "max_blind_gap_s",
                        "inside_length_m", "interior_coverage",
                        "n_oracle_interior_events") if c in sub.columns]
    cov = sub.groupby("trajectory_id")[cols].first()
    d = piv.join(cov).dropna()
    gain = (d[baseline] - d[champion]).values.astype(float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(gain), (n_boot, len(gain)))

    out = {"condition": condition, "baseline": baseline, "champion": champion,
           "n_trajectories": int(len(d)), "pearson_r_gain_vs": {}}
    for name in cols:
        x = d[name].values.astype(float)
        lo, hi = _boot_pearson(x, gain, idx)
        out["pearson_r_gain_vs"][name] = {
            "r": float(np.corrcoef(x, gain)[0, 1]), "ci_lo": lo, "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0)}
    if {"branch_density_per_km", "max_blind_gap_s"} <= set(cols):
        b = d["branch_density_per_km"].values.astype(float)
        g = d["max_blind_gap_s"].values.astype(float)
        rb = _residualise(gain, g)
        xb = _residualise(b, g)
        rg = _residualise(gain, b)
        xg = _residualise(g, b)
        lo_b, hi_b = _boot_pearson(xb, rb, idx)
        lo_g, hi_g = _boot_pearson(xg, rg, idx)
        out["partial_correlation"] = {
            "gain_vs_branch_density_controlling_blind_gap": {
                "r": float(np.corrcoef(xb, rb)[0, 1]), "ci_lo": lo_b, "ci_hi": hi_b},
            "gain_vs_blind_gap_controlling_branch_density": {
                "r": float(np.corrcoef(xg, rg)[0, 1]), "ci_lo": lo_g, "ci_hi": hi_g},
        }
    return out






def _fmt(m, lo, hi):
    if not np.isfinite(m):
        return "--"
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"


def _condition_c_note(calib_name, obs_kwargs):
    if calib_name == "DEVELOPMENT_PLACEHOLDER":
        return (r" --- \textbf{DEVELOPMENT PLACEHOLDER, NOT A PAPER RESULT}"
                r" (hand-set FN 0.10, ghosts 0.10/event, jitter $\sigma$ 0.5 s)")
    if not obs_kwargs:
        return ""
    return (rf" (detector-calibrated: FN {obs_kwargs['false_negative_rate']:.2f}, "
            rf"ghosts {obs_kwargs['false_positive_rate']:.3f}/event, "
            rf"jitter $\sigma$ {obs_kwargs['timing_jitter_std_s']:.2f}\,s)")


def write_table2(df, path, conditions, calib_name, n_traj, obs_kwargs=None):
    lines = [
        "% Table 2 — estimator comparison. Generated by",
        "% experiments/estimator_comparison/run_estimator_comparison.py; do not edit by hand.",
        "% RMSE in metres, mean over trajectories with a percentile bootstrap",
        "% 95% CI in brackets. p95 = mean over trajectories of the 95th",
        "% percentile of per-sample position error.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Estimator comparison on identical trajectories and identical "
        r"event streams (" + str(n_traj) + r" region transits, CARLA Town05). "
        r"Position RMSE in metres, mean [95\% bootstrap CI]. Strata follow the "
        r"frozen route-difficulty rule (branch density; longest predict-only "
        r"gap) and are estimator-independent.}",
        r"\label{tab:estimators}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Estimator & Overall & Junction-heavy & Long blind gap & p95 error \\",
        r"\midrule",
    ]
    overall = aggregate(df, ["estimator", "condition"])
    diff = by_difficulty(df)
    for ci, cond in enumerate(conditions):
        if ci:
            lines.append(r"\midrule")
        note = ""
        if cond == COND_PERIMETER_ONLY:
            note = (r" (no interior measurement: the CV Kalman filter reduces "
                    r"to dead reckoning)")
        elif cond == COND_SIDECHANNEL:
            note = _condition_c_note(calib_name, obs_kwargs)
        lines.append(r"\multicolumn{5}{l}{\emph{" +
                     CONDITION_LABEL[cond].replace(":", ".") + note + r"}} \\")
        for est in ESTIMATOR_ORDER:
            o = overall[(overall["estimator"] == est) &
                        (overall["condition"] == cond)]
            if o.empty:
                continue
            o = o.iloc[0]
            def cell(axis, stratum):
                r = diff[(diff["axis"] == axis) & (diff["stratum"] == stratum) &
                         (diff["condition"] == cond) & (diff["estimator"] == est)]
                if r.empty:
                    return "--"
                r = r.iloc[0]
                return _fmt(r["mean"], r["ci_lo"], r["ci_hi"])
            lines.append(
                f"{ESTIMATOR_LABEL[est]} & {_fmt(o['mean'], o['ci_lo'], o['ci_hi'])}"
                f" & {cell('branch_density', 'junction_heavy')}"
                f" & {cell('blind_gap', 'long_gap')}"
                f" & {o['p95_mean']:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))






def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.0,
        "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return plt


def _grouped_bars(ax, groups, agg, key_fn, ylabel=None, width=0.26):
    x = np.arange(len(groups), dtype=float)
    handles = []
    for i, est in enumerate(ESTIMATOR_ORDER):
        m, lo, hi = [], [], []
        for grp in groups:
            r = agg[key_fn(agg, grp, est)]
            if r.empty:
                m.append(np.nan); lo.append(0.0); hi.append(0.0)
            else:
                r = r.iloc[0]
                m.append(r["mean"])
                lo.append(max(r["mean"] - r["ci_lo"], 0.0))
                hi.append(max(r["ci_hi"] - r["mean"], 0.0))
        off = (i - (len(ESTIMATOR_ORDER) - 1) / 2) * width
        b = ax.bar(x + off, m, width * 0.92, color=ESTIMATOR_COLOR[est],
                   hatch=ESTIMATOR_HATCH[est], edgecolor="white", linewidth=0.5,
                   label=ESTIMATOR_LABEL[est], zorder=2)
        ax.errorbar(x + off, m, yerr=[lo, hi], fmt="none", ecolor="#333333",
                    elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
        handles.append(b)
    ax.set_xticks(x)
    ax.yaxis.grid(True, color="#d9d9d9", zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    return handles


def figure_overall(df, conditions, path):
    plt = _style()
    agg = aggregate(df, ["estimator", "condition"])
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    _grouped_bars(ax, conditions, agg,
                  lambda a, g, e: (a["condition"] == g) & (a["estimator"] == e),
                  ylabel="Position RMSE (m)")
    ax.set_xticklabels([CONDITION_SHORT[c] for c in conditions])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3,
              handlelength=1.1, columnspacing=0.9, handletextpad=0.4,
              borderpad=0.0)
    fig.savefig(path)
    plt.close(fig)


def figure_difficulty(df, conditions, path, counts):
    plt = _style()
    agg = by_difficulty(df)
    conds = [c for c in conditions if c != COND_PERIMETER_ONLY] or list(conditions)
    axes_spec = [("branch_density", BRANCH_ORDER, "Branch density (forks/km)"),
                 ("blind_gap", GAP_ORDER, "Longest predict-only gap (s)")]
    fig, axs = plt.subplots(len(conds), 2, figsize=(7.0, 1.9 * len(conds) + 0.6),
                            sharey=True, squeeze=False)
    for r, cond in enumerate(conds):
        for c, (axis, order, xlabel) in enumerate(axes_spec):
            ax = axs[r][c]
            _grouped_bars(
                ax, order, agg,
                lambda a, g, e, axis=axis, cond=cond: (
                    (a["axis"] == axis) & (a["stratum"] == g) &
                    (a["condition"] == cond) & (a["estimator"] == e)),
                ylabel="Position RMSE (m)" if c == 0 else None)
            ax.set_xticklabels([f"{STRATUM_LABEL[s]}\n(n={counts[axis].get(s, 0)})"
                                for s in order])
            if r == len(conds) - 1:
                ax.set_xlabel(xlabel)
            if c == 1:


                ax.set_title(CONDITION_LABEL[cond], loc="right", pad=3,
                             color="#444444")
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, handlelength=1.1,
               columnspacing=1.2, handletextpad=0.4,
               bbox_to_anchor=(0.5, 1.02 + 0.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trajectories", type=int, default=120)
    ap.add_argument("--estimators", nargs="+", default=ESTIMATOR_ORDER)
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="corruption realisations per trajectory in condition C")
    ap.add_argument("--num_particles", type=int, default=2000)
    ap.add_argument("--n_workers", type=int, default=os.cpu_count())
    ap.add_argument("--obs_model", default=None)
    ap.add_argument("--dev_corruption", action="store_true")
    ap.add_argument("--skip_condition_c", action="store_true")
    ap.add_argument("--pf_stability_seeds", type=int, default=5,
                    help="extra road-PF filter seeds on condition B, reported "
                         "as a Monte-Carlo stability check (0 disables)")
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--suffix", default="",
                    help="appended to every output stem, e.g. '.devcorruption', "
                         "so a placeholder run never overwrites the calibrated one")
    ap.add_argument("--out", default=None)
    dc.add_profile_argument(ap)
    args = ap.parse_args()





    profile = None if args.dev_corruption else dc.resolve(args)
    sfx = args.suffix or ("" if profile is None else dc.suffix_for(profile.name))
    if sfx and not sfx.startswith("."):
        sfx = "." + sfx
    if args.out is None:
        args.out = os.path.join(RESULTS, f"estimator_comparison{sfx}.csv")
    if args.dev_corruption and not sfx:
        print("[estimator_comparison] note: a --dev_corruption run without --suffix will "
              "overwrite the calibrated outputs at the same paths")

    cfg = load_dataset_config(n_trajectories=args.n_trajectories)
    ds = build_dataset(cfg, cache=True)
    print(f"[estimator_comparison] {len(ds)} trajectories, {len(ds.interior_sensors)} interior "
          f"sensors, {ds.manifest['n_perimeter_sensors']} perimeter sensors")

    obs_kwargs = calib_name = calib_source = None
    conditions = [COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR]
    if not args.skip_condition_c:
        obs_kwargs, calib_name, calib_source = resolve_observation_model(args, profile)
        conditions.append(COND_SIDECHANNEL)
        print(f"[estimator_comparison] condition C calibration: {calib_name} ({calib_source})")
        if calib_name == "DEVELOPMENT_PLACEHOLDER":
            print("[estimator_comparison] *** DEVELOPMENT RUN — condition C rows are NOT paper "
                  "results ***")

    est_kwargs = {"num_particles": args.num_particles,
                  "fov_radius": cfg.interior_fov,
                  "perimeter_sigma": cfg.perimeter_noise_std}

    jobs = build_jobs(ds, args.estimators, conditions, est_kwargs, obs_kwargs,
                      args.n_seeds, calib_name)
    print(f"[estimator_comparison] {len(jobs)} runs across {len(args.estimators)} estimators")

    pre = verify_event_streams(ds, jobs)
    print(f"[estimator_comparison] stream identity (pre-run): {pre['n_cells']} cells x "
          f"{pre['estimators_per_cell']} estimators, "
          f"{pre['total_events_hashed']} events hashed, 0 mismatches")

    df = run_jobs(cfg, jobs, n_workers=args.n_workers)
    post = verify_run_rows(df, args.estimators)
    print(f"[estimator_comparison] stream identity (post-run): PASS "
          f"({post['n_cells']} cells, 0 disagreeing event counts)")

    kf_dr = kf_equals_dr_check(df)
    for rec in kf_dr:
        print(f"[estimator_comparison] KF vs DR  {rec['condition']:20s} identical (<1e-6 m) in "
              f"{rec['n_identical_1e-6m']}/{rec['n_pairs']} runs "
              f"({100*rec['frac_identical_1e-6m']:.1f}%), max |dRMSE| = "
              f"{rec['max_abs_rmse_diff_m']:.3g} m")


    pf_stability = None
    if args.pf_stability_seeds and "road_pf" in args.estimators:
        sj = [{"trajectory_id": tr.trajectory_id, "estimator": "road_pf",
               "condition": COND_ORACLE_INTERIOR, "obs": None,
               "est_kwargs": {**est_kwargs, "seed": k},
               "tags": {"corruption_seed": -1, "calibration": "n/a",
                        "filter_seed": k}}
              for tr in ds.trajectories for k in range(args.pf_stability_seeds)]
        sdf = run_jobs(cfg, sj, n_workers=args.n_workers, verbose=False)
        per_seed = sdf.groupby("filter_seed")["rmse"].mean()
        pf_stability = {
            "condition": COND_ORACLE_INTERIOR, "n_seeds": args.pf_stability_seeds,
            "mean_rmse_per_seed_m": {int(k): float(v) for k, v in per_seed.items()},
            "spread_across_seeds_m": float(per_seed.max() - per_seed.min()),
            "within_trajectory_sd_mean_m": float(
                sdf.groupby("trajectory_id")["rmse"].std().mean()),
        }
        print(f"[estimator_comparison] road-PF filter-seed spread on B: "
              f"{pf_stability['spread_across_seeds_m']:.3f} m across "
              f"{args.pf_stability_seeds} seeds")

    counts = {"branch_density": df.groupby("difficulty")["trajectory_id"].nunique().to_dict(),
              "blind_gap": df.groupby("blind_gap_stratum")["trajectory_id"].nunique().to_dict()}

    headline = COND_SIDECHANNEL if COND_SIDECHANNEL in conditions else COND_ORACLE_INTERIOR
    hyp = {c: hypothesis_test(df, c, n_boot=args.n_boot)
           for c in conditions if c != COND_PERIMETER_ONLY}
    hyp_dr = {c: hypothesis_test(df, c, baseline="dead_reckoning",
                                 n_boot=args.n_boot)
              for c in conditions if c != COND_PERIMETER_ONLY}
    cont = {c: gain_vs_covariates(df, c, n_boot=args.n_boot)
            for c in conditions if c != COND_PERIMETER_ONLY}

    meta = {
        "experiment": "estimator_comparison",
        "question": "why road-constrained probabilistic inference?",
        "dataset_config": cfg.as_dict(),
        "dataset_manifest": ds.manifest,
        "conditions": conditions,
        "estimators": args.estimators,
        "estimator_kwargs": est_kwargs,
        "n_corruption_seeds": args.n_seeds,
        "corruption_seed_rule": "zlib.crc32('rq4|<trajectory_id>|<replicate>') % 2**31",
        "corruption_seeds": {tr.trajectory_id: [corruption_seed(tr.trajectory_id, s)
                                                for s in range(args.n_seeds)]
                             for tr in ds.trajectories},
        "filter_seeds": {"A_and_B": 0, "C": "replicate index 0..n_seeds-1"},
        "bootstrap": {"n_boot": args.n_boot, "seed": 0, "unit": "trajectory"},
        "observation_model": obs_kwargs,
        "calibration_rates": ({k: obs_kwargs[k] for k in
                               ("false_negative_rate", "false_positive_rate",
                                "timing_jitter_std_s")} if obs_kwargs else None),
        "calibration_source": calib_source,
        "calibration_file": calib_name,
        "detector_calibration": None if profile is None else profile.meta(),
        "per_camera_overrides_note": (
            "The bundled profile uses global rates. Custom per-camera keys "
            "must match synthetic sensor ids (I00, I01, ...)."),
        "is_paper_result": calib_name != "DEVELOPMENT_PLACEHOLDER",
        "difficulty_rule": DIFFICULTY_RULE,
        "difficulty_strata": difficulty_table(ds.trajectories),
        "stratum_route_profile": stratum_profile(df),
        "identity_checks": {"pre_run": pre, "post_run": post},
        "kf_equals_dead_reckoning_check": kf_dr,
        "pf_filter_seed_stability": pf_stability,
        "headline_condition": headline,
        "hypothesis_road_pf_vs_kalman": hyp,
        "hypothesis_road_pf_vs_dead_reckoning": hyp_dr,
        "hypothesis_continuous_gain_vs_covariates": cont,
        "out_of_scope": ["LaneGCN", "HiVT", "QCNet", "graph MIP"],
    }
    save_results(df, args.out, meta)
    print(f"[estimator_comparison] wrote {args.out}")

    agg = aggregate(df, ["estimator", "condition"], n_boot=args.n_boot)
    agg["estimator"] = pd.Categorical(agg["estimator"], ESTIMATOR_ORDER, ordered=True)
    agg = agg.sort_values(["condition", "estimator"])
    agg_path = args.out.replace(".csv", "_summary.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[estimator_comparison] wrote {agg_path}")

    diff = by_difficulty(df, n_boot=args.n_boot)
    diff_path = os.path.join(RESULTS, f"estimator_comparison_by_route_difficulty{sfx}.csv")
    save_results(diff, diff_path, {**meta, "table": "estimator x stratum x condition"})
    print(f"[estimator_comparison] wrote {diff_path}")

    tex = os.path.join(RESULTS, f"estimator_comparison_table2{sfx}.tex")
    write_table2(df, tex, conditions, calib_name or "n/a", len(ds), obs_kwargs)
    print(f"[estimator_comparison] wrote {tex}")

    os.makedirs(FIGURES, exist_ok=True)
    fig1 = os.path.join(FIGURES, f"estimator_comparison{sfx}.pdf")
    fig2 = os.path.join(FIGURES, f"estimator_comparison_route_difficulty{sfx}.pdf")
    figure_overall(df, conditions, fig1)
    figure_difficulty(df, conditions, fig2, counts)
    print(f"[estimator_comparison] wrote {fig1}, {fig2}")


    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\n" + agg[["condition", "estimator", "n", "mean", "ci_lo", "ci_hi",
                          "p95_mean", "catastrophic_rate"]].to_string(index=False))
        print("\n" + diff[diff["condition"] == headline][
            ["axis", "stratum", "estimator", "n", "mean", "ci_lo", "ci_hi",
             "p95_mean", "catastrophic_rate"]].to_string(index=False))

    print(f"\n[estimator_comparison] hypothesis (road-PF vs Kalman), condition {headline}:")
    for axis, rec in hyp[headline]["axes"].items():
        for name, r in rec.items():
            if name == "interaction":
                v = r
                print(f"  {axis:14s} {v['comparison']:42s} "
                      f"delta gain {v['delta_gain_m']:+6.2f} m "
                      f"[{v['ci_lo']:+.2f}, {v['ci_hi']:+.2f}] "
                      f"-> {'SUPPORTED' if v['supports_hypothesis'] else 'NOT supported'}")
            else:
                print(f"  {axis:14s} {name:28s} n={r['n']:3d} "
                      f"KF {r['kalman_rmse_m']:6.2f} -> PF {r['road_pf_rmse_m']:6.2f} m "
                      f"({r['gain_pct']:+5.1f}%, gain CI "
                      f"[{r['gain_ci_lo']:+.2f}, {r['gain_ci_hi']:+.2f}])")

    c = cont.get(headline) or {}
    if c:
        print(f"\n[estimator_comparison] continuous check (per-trajectory PF gain over KF, "
              f"n={c['n_trajectories']}), condition {headline}:")
        for name, r in c["pearson_r_gain_vs"].items():
            print(f"  r(gain, {name:26s}) = {r['r']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
                  f"{'  *' if r['significant'] else ''}")
        for name, r in c.get("partial_correlation", {}).items():
            print(f"  partial {name:52s} = {r['r']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")


if __name__ == "__main__":
    main()
