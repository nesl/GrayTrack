
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.experiment import (COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR,
                            COND_SIDECHANNEL, run_jobs, save_results, aggregate,
                            load_dataset_config)
from src.dataset import build_dataset
from src import detector_calibration as dc
from src.observation_model import load_params
from src.evaluate import bootstrap_ci, difficulty_table
from src.road_graph import EXP_ROOT

RESULTS = os.path.join(EXP_ROOT, "results")
FIGURES = os.path.join(EXP_ROOT, "figures")

PARAMS = ("false_negative_rate", "false_positive_rate", "timing_jitter_std_s")













BASELINE = {p: 0.0 for p in PARAMS}



BASELINE_BIAS = 0.0



GRID_SUFFIX = ""




FIG_SUFFIX = ""


def sweep_csv(spec: dict) -> str:
    return spec["csv"].replace(".csv", f"{GRID_SUFFIX}.csv")



SWEEPS = {
    "missing": {
        "param": "false_negative_rate",
        "levels": [0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
        "csv": "robustness_missing_sweep.csv",
        "pdf": "robustness_missing_sweep.pdf",
        "xlabel": "Missed-detection rate (%)",
        "title": "Missed detections",
        "xscale": 100.0,
        "unit": "%",
    },
    "false_positive": {
        "param": "false_positive_rate",
        "levels": [0.0, 0.05, 0.10, 0.20, 0.30],
        "csv": "robustness_false_positive_sweep.csv",
        "pdf": "robustness_false_positive_sweep.pdf",
        "xlabel": "Ghost events (% of true events)",
        "title": "Ghost events",
        "xscale": 100.0,
        "unit": "% of true interior events",
    },
    "timing_jitter": {
        "param": "timing_jitter_std_s",
        "levels": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        "csv": "robustness_timing_jitter_sweep.csv",
        "pdf": "robustness_timing_jitter_sweep.pdf",
        "xlabel": "Timing jitter $\\sigma$ (s)",
        "title": "Timing jitter",
        "xscale": 1.0,
        "unit": "s",
    },
}



COLOURS = {"road_pf": "#0072B2", "kalman": "#D55E00", "dead_reckoning": "#009E73"}
PRETTY = {"road_pf": "Road-PF", "kalman": "Kalman", "dead_reckoning": "Dead reck."}
MEAS_COLOUR = "#444444"





MEAS_LABEL = "measured (event eval)"




REFERENCE_MODE = "reference"


def stable_seed(*parts) -> int:
    blob = "|".join(repr(p) for p in parts).encode()
    return int(hashlib.sha1(blob).hexdigest()[:8], 16) % (2 ** 31)






def resolve_calibration(args, profile=None):
    if args.obs_model:
        path = args.obs_model
    else:
        path = (profile or dc.load_profile()).observation_model_path
    if not os.path.exists(path):
        return None, path, "calibrated config absent"
    params = load_params(path)
    if not params.is_calibrated:
        return None, path, f"config not calibrated (missing {params.missing_fields()})"

    res = params.timing_residuals_s
    res = np.asarray(res, dtype=float) if res is not None and len(res) else None
    calib = {
        "file": path,
        "source": params.source,
        "false_negative_rate": params.false_negative_rate,
        "false_positive_rate": params.false_positive_rate,
        "timing_jitter_std_s": params.timing_jitter_std_s,
        "n_per_camera_overrides": len(params.per_camera or {}),
        "timing_residuals_s": res,
    }
    if res is not None:
        calib.update({
            "n_residuals": int(len(res)),
            "residual_mean_s": float(np.mean(res)),
            "residual_std_s": float(np.std(res, ddof=1)),
            "residual_min_s": float(np.min(res)),
            "residual_max_s": float(np.max(res)),
        })
    return calib, path, "ok"


def calib_meta(calib):
    if calib is None:
        return {"available": False}
    d = {k: v for k, v in calib.items() if k != "timing_residuals_s"}
    d["available"] = True
    d["timing_residuals_available"] = calib.get("timing_residuals_s") is not None
    d["measured_operating_point"] = {
        "missing": calib["false_negative_rate"],
        "false_positive": calib["false_positive_rate"],
        "timing_jitter": calib["timing_jitter_std_s"],
    }
    return d


def measured_level(calib, sweep_key):
    if calib is None:
        return None
    return calib.get(SWEEPS[sweep_key]["param"])


def empirical_x(calib):
    if calib is None or calib.get("timing_residuals_s") is None:
        return None
    return float(calib["residual_std_s"])






def gaussian_points():
    pts = set()
    for spec in SWEEPS.values():
        for lvl in spec["levels"]:
            vals = dict(BASELINE)
            vals[spec["param"]] = float(lvl)
            pts.add((vals["false_negative_rate"], vals["false_positive_rate"],
                     vals["timing_jitter_std_s"], "gaussian"))
    return sorted(pts)


def empirical_point(calib):
    x = empirical_x(calib)
    if x is None:
        return []
    return [(BASELINE["false_negative_rate"],
             BASELINE["false_positive_rate"], x, "empirical")]


def build_jobs(ds, args, est_kwargs, points, residuals, with_references=True):
    jobs = []
    for est in args.estimators:
        for tr in ds.trajectories:
            for s in range(args.n_seeds):


                if with_references:
                    for cond in (COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR):
                        jobs.append({
                            "trajectory_id": tr.trajectory_id, "estimator": est,
                            "condition": cond, "obs": None,
                            "est_kwargs": {**est_kwargs, "seed": s},
                            "tags": {"corruption_seed": s,
                                     "jitter_mode": REFERENCE_MODE,
                                     **{p: np.nan for p in PARAMS}}})


                oseed = stable_seed(tr.trajectory_id, s)
                for fnr, fpr, jit, mode in points:
                    obs = {"false_negative_rate": fnr,
                           "false_positive_rate": fpr,
                           "timing_jitter_std_s": jit,
                           "timing_bias_s": BASELINE_BIAS,
                           "jitter_mode": mode,
                           "random_seed": oseed}
                    if mode == "empirical":
                        obs["timing_residuals_s"] = list(residuals)
                    jobs.append({
                        "trajectory_id": tr.trajectory_id, "estimator": est,
                        "condition": COND_SIDECHANNEL, "obs": obs,
                        "est_kwargs": {**est_kwargs, "seed": s},
                        "tags": {"corruption_seed": s, "jitter_mode": mode,
                                 "false_negative_rate": fnr,
                                 "false_positive_rate": fpr,
                                 "timing_jitter_std_s": jit,
                                 "obs_seed": oseed}})

    random.Random(0).shuffle(jobs)
    return jobs


def select_sweep(df, key, emp_x=None):
    spec = SWEEPS[key]
    p = spec["param"]
    others = [q for q in PARAMS if q != p]

    c = df["condition"] == COND_SIDECHANNEL
    m = c & (df["jitter_mode"] == "gaussian") & df[p].isin(spec["levels"])
    for q in others:


        m &= np.isclose(df[q].astype(float), BASELINE[q], rtol=0, atol=1e-12)
    sub = df[m].copy()
    sub["level"] = sub[p].astype(float)

    if key == "timing_jitter" and emp_x is not None:
        emp = df[c & (df["jitter_mode"] == "empirical")].copy()
        if len(emp):
            emp["level"] = float(emp_x)
            sub = pd.concat([sub, emp], ignore_index=True)

    ref = df[df["condition"] != COND_SIDECHANNEL].copy()
    ref["level"] = np.nan
    out = pd.concat([sub, ref], ignore_index=True)
    out["sweep"] = key
    return out






MEAN_COLS = ["rmse", "mae", "median", "p95", "max", "final_error", "coverage",
             "catastrophic", "n_interior_events_used", "n_true_events_used",
             "n_ghost_events_used", "n_oracle_interior_events"]


def per_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["estimator", "condition", "jitter_mode", "level", "trajectory_id"]
    cols = [c for c in MEAN_COLS if c in df.columns]
    return df.groupby(keys, dropna=False)[cols].mean().reset_index()


def summarise_sweep(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    pt = per_trajectory(df)
    keys = ["estimator", "condition", "jitter_mode", "level"]
    agg = aggregate(pt, keys, value_col="rmse", n_boot=n_boot, seed=seed)
    extra = (pt.groupby(keys, dropna=False)[
                 ["n_interior_events_used", "n_true_events_used",
                  "n_ghost_events_used", "median", "final_error"]]
             .mean().reset_index()
             .rename(columns={"median": "median_err_mean",
                              "final_error": "final_error_mean"}))
    out = agg.merge(extra, on=keys, how="left")
    return out.sort_values(keys, na_position="first").reset_index(drop=True)






def paired_vs_reference(df, est, level, ref_condition, n_boot, seed,
                        jitter_mode="gaussian"):
    c = df[(df["estimator"] == est) & (df["condition"] == COND_SIDECHANNEL)
           & (df["jitter_mode"] == jitter_mode) & (df["level"] == level)]
    r = df[(df["estimator"] == est) & (df["condition"] == ref_condition)]
    key = ["trajectory_id", "corruption_seed"]
    m = c[key + ["rmse"]].merge(r[key + ["rmse"]], on=key,
                                suffixes=("_c", "_ref"))
    if m.empty:
        return float("nan"), float("nan"), float("nan"), 0
    m["d"] = m["rmse_c"] - m["rmse_ref"]
    per_traj = m.groupby("trajectory_id")["d"].mean().values
    mean, lo, hi = bootstrap_ci(per_traj, n_boot=n_boot, seed=seed)
    return mean, lo, hi, len(per_traj)


def _interp(x0, y0, x1, y1):
    if y1 == y0:
        return float(x1)
    return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))


def _interp_y(levels, ys, x):
    return float(np.interp(x, levels, ys))


def crossover_analysis(df, sweep_key, estimators, n_boot, seed, calib=None):
    spec = SWEEPS[sweep_key]
    levels = spec["levels"]
    meas = measured_level(calib, sweep_key)
    rows, crossings = [], []
    for est in estimators:
        curve = []
        for lvl in levels:
            mean, lo, hi, n = paired_vs_reference(
                df, est, lvl, COND_PERIMETER_ONLY, n_boot, seed)
            rows.append({"sweep": sweep_key, "estimator": est, "level": lvl,
                         "jitter_mode": "gaussian", "n_trajectories": n,
                         "delta_vs_perimeter_mean": mean,
                         "delta_ci_lo": lo, "delta_ci_hi": hi,
                         "significantly_better_than_perimeter": bool(hi < 0)})
            curve.append((lvl, mean, hi))

        if calib is not None and calib.get("timing_residuals_s") is not None \
                and sweep_key == "timing_jitter":
            x = empirical_x(calib)
            mean, lo, hi, n = paired_vs_reference(
                df, est, x, COND_PERIMETER_ONLY, n_boot, seed,
                jitter_mode="empirical")
            rows.append({"sweep": sweep_key, "estimator": est, "level": x,
                         "jitter_mode": "empirical", "n_trajectories": n,
                         "delta_vs_perimeter_mean": mean,
                         "delta_ci_lo": lo, "delta_ci_hi": hi,
                         "significantly_better_than_perimeter": bool(hi < 0)})

        rec = {"sweep": sweep_key, "estimator": est,
               "param": spec["param"], "max_level_swept": levels[-1],
               "unit": spec["unit"], "measured_level": meas}
        for name, idx in (("mean", 1), ("sig", 2)):
            cross, inside = None, False
            for i in range(1, len(curve)):
                y0, y1 = curve[i - 1][idx], curve[i][idx]
                if y0 < 0 <= y1:
                    cross = _interp(curve[i - 1][0], y0, curve[i][0], y1)
                    inside = True
                    break
            if curve[0][idx] >= 0:
                cross, inside = float(levels[0]), True
            rec[f"crossover_{name}"] = cross
            rec[f"crossover_{name}_inside_range"] = inside
            rec[f"value_at_max_level_{name}"] = curve[-1][idx]
        if meas is not None:
            rec["delta_at_measured_mean"] = _interp_y(
                levels, [c[1] for c in curve], meas)
            rec["delta_at_measured_ci_hi"] = _interp_y(
                levels, [c[2] for c in curve], meas)
            if rec["crossover_sig_inside_range"]:
                rec["measured_safety_margin"] = float(rec["crossover_sig"] - meas)
            else:
                rec["measured_safety_margin"] = float(levels[-1] - meas)

        u = spec["unit"]
        sc = 100.0 if u.startswith("%") else 1.0
        if rec["crossover_sig_inside_range"]:
            rec["note"] = (f"C stops being significantly better than "
                           f"perimeter-only at {rec['crossover_sig'] * sc:.3g} {u}")
        else:
            rec["note"] = (
                f"no crossover within the swept range (max {levels[-1] * sc:.3g} "
                f"{u}); at that level C is still "
                f"{abs(rec['value_at_max_level_mean']):.1f} m better than "
                f"perimeter-only (upper 95% bound "
                f"{rec['value_at_max_level_sig']:+.1f} m)")
        crossings.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(crossings)


def degradation_stats(summary, sweep_key, estimators, calib=None):
    spec = SWEEPS[sweep_key]
    meas = measured_level(calib, sweep_key)
    out = []
    for est in estimators:
        s = summary[(summary["estimator"] == est)
                    & (summary["condition"] == COND_SIDECHANNEL)
                    & (summary["jitter_mode"] == "gaussian")].sort_values("level")
        s = s[s["level"].isin(spec["levels"])]
        if len(s) < 2:
            continue
        x = s["level"].values.astype(float)
        y = s["mean"].values.astype(float)
        a = summary[(summary["estimator"] == est)
                    & (summary["condition"] == COND_PERIMETER_ONLY)]["mean"]
        a = float(a.iloc[0]) if len(a) else float("nan")
        rec = {
            "sweep": sweep_key, "estimator": est,
            "rmse_at_0": float(y[0]), "rmse_at_max": float(y[-1]),
            "abs_increase_m": float(y[-1] - y[0]),
            "rel_increase_pct": float(100.0 * (y[-1] - y[0]) / y[0]),
            "slope_m_per_unit": float(np.polyfit(x, y, 1)[0]),
            "perimeter_only_rmse": a,
            "gap_closed_at_0_pct": float(100.0 * (a - y[0]) / a),
            "gap_closed_at_max_pct": float(100.0 * (a - y[-1]) / a),
        }
        if meas is not None:
            ym = _interp_y(list(x), list(y), meas)
            rec["measured_level"] = meas
            rec["rmse_at_measured"] = ym
            rec["rel_increase_at_measured_pct"] = float(100.0 * (ym - y[0]) / y[0])
            rec["gap_closed_at_measured_pct"] = float(100.0 * (a - ym) / a)
        out.append(rec)
    return pd.DataFrame(out)






def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "lines.linewidth": 1.4,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "grid.color": "#B0B0B0", "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    for side in ("top", "right"):
        plt.rcParams[f"axes.spines.{side}"] = False
    return plt


def y_limits(summaries, keys, estimators, logy=True, pad=0.10):
    lo, hi = np.inf, -np.inf
    for k in keys:
        s = summaries[k]
        s = s[s["estimator"].isin(estimators)]
        if s.empty:
            continue
        lo = min(lo, float(np.nanmin(s["ci_lo"].values)))
        hi = max(hi, float(np.nanmax(s["ci_hi"].values)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if logy:
        span = np.log10(hi) - np.log10(lo)
        return (10 ** (np.log10(lo) - pad * span),
                10 ** (np.log10(hi) + pad * span))
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def draw_panel(ax, summary, sweep_key, estimators, calib=None, annotate=True,
               logy=False, ylim=None):
    spec = SWEEPS[sweep_key]
    xs = spec["xscale"]
    x_levels = [lvl * xs for lvl in spec["levels"]]
    emp_x = empirical_x(calib) if sweep_key == "timing_jitter" else None


    meas = measured_level(calib, sweep_key)
    if meas is not None:
        ax.axvline(meas * xs, color=MEAS_COLOUR, linewidth=0.9,
                   linestyle=(0, (2, 2)), zorder=1)

    for est in estimators:
        col = COLOURS.get(est, "#666666")
        s = summary[(summary["estimator"] == est)
                    & (summary["condition"] == COND_SIDECHANNEL)
                    & (summary["jitter_mode"] == "gaussian")]
        s = s[s["level"].isin(spec["levels"])].sort_values("level")
        x = s["level"].values * xs
        ax.fill_between(x, s["ci_lo"].values, s["ci_hi"].values,
                        color=col, alpha=0.18, linewidth=0)
        ax.plot(x, s["mean"].values, color=col, marker="o", markersize=3.2,
                markeredgewidth=0, zorder=3)

        ref = summary[(summary["estimator"] == est)
                      & (summary["condition"] == COND_PERIMETER_ONLY)]
        if len(ref):
            r = ref.iloc[0]
            ax.axhspan(r["ci_lo"], r["ci_hi"], color=col, alpha=0.10, linewidth=0)
            ax.axhline(r["mean"], color=col, linestyle=(0, (4, 2)), linewidth=1.1)
        orc = summary[(summary["estimator"] == est)
                      & (summary["condition"] == COND_ORACLE_INTERIOR)]
        if len(orc):
            ax.axhline(float(orc.iloc[0]["mean"]), color=col,
                       linestyle=(0, (1, 1.6)), linewidth=1.0)

        if emp_x is not None:
            e = summary[(summary["estimator"] == est)
                        & (summary["condition"] == COND_SIDECHANNEL)
                        & (summary["jitter_mode"] == "empirical")]
            if len(e):
                e = e.iloc[0]
                ax.errorbar([e["level"] * xs], [e["mean"]],
                            yerr=[[max(e["mean"] - e["ci_lo"], 0)],
                                  [max(e["ci_hi"] - e["mean"], 0)]],
                            fmt="*", markersize=8, color=col, ecolor=col,
                            elinewidth=0.9, capsize=2, zorder=5)

    if logy:
        ax.set_yscale("log")
        from matplotlib.ticker import FixedLocator, FuncFormatter
        ticks = [10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))

    ax.set_xlabel(spec["xlabel"])
    lo_x, hi_x = min(x_levels), max(x_levels)
    ax.set_xlim(lo_x - 0.03 * hi_x, hi_x * 1.03)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if meas is not None and annotate:
        ax.annotate(MEAS_LABEL, xy=(meas * xs, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=8, color=MEAS_COLOUR,
                    rotation=90)
    return ax


def legend_handles(estimators, emp=False, measured=False):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOURS.get(e, "#666"), marker="o", markersize=3.2,
                markeredgewidth=0, label=PRETTY.get(e, e)) for e in estimators]
    h += [Line2D([], [], color="#555555", linestyle=(0, (4, 2)), linewidth=1.1,
                 label="perimeter-only (A)"),
          Line2D([], [], color="#555555", linestyle=(0, (1, 1.6)), linewidth=1.0,
                 label="oracle interior (B)")]
    if emp:
        h.append(Line2D([], [], color="#555555", linestyle="none", marker="*",
                        markersize=8, label="empirical jitter"))
    if measured:
        h.append(Line2D([], [], color=MEAS_COLOUR, linestyle=(0, (2, 2)),
                        linewidth=0.9, label=MEAS_LABEL))
    return h


def make_figures(summaries, estimators, calib, outdir, logy=True, suffix=""):
    plt = _style()
    os.makedirs(outdir, exist_ok=True)
    written = []
    has_emp = empirical_x(calib) is not None
    has_meas = calib is not None


    ylim = y_limits(summaries, list(SWEEPS), estimators, logy=logy)

    for key, spec in SWEEPS.items():
        fig, ax = plt.subplots(figsize=(3.3, 2.4))
        draw_panel(ax, summaries[key], key, estimators, calib=calib, logy=logy,
                   ylim=ylim)
        ax.set_ylabel("Trajectory RMSE (m)")


        ax.legend(handles=legend_handles(
                      estimators, emp=has_emp and key == "timing_jitter",
                      measured=has_meas),
                  loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=2,
                  fontsize=8)
        p = os.path.join(outdir, spec["pdf"].replace(".pdf", f"{suffix}.pdf"))
        fig.savefig(p)
        plt.close(fig)
        written.append(p)


    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for ax, key in zip(axes, SWEEPS):
        draw_panel(ax, summaries[key], key, estimators, calib=calib,
                   annotate=False, logy=logy, ylim=ylim)
        ax.set_title(SWEEPS[key]["title"], pad=3)
    axes[0].set_ylabel("Trajectory RMSE (m)")




    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.30,
                        wspace=0.10)
    fig.legend(handles=legend_handles(estimators, emp=has_emp, measured=has_meas),
               loc="upper center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 0.11))
    p = os.path.join(outdir, f"robustness_panel{suffix}.pdf")
    fig.savefig(p)
    plt.close(fig)
    written.append(p)
    return written






def postprocess(raws, args, base_meta, calib):
    summaries = {}
    for key, spec in SWEEPS.items():
        out = os.path.join(args.results_dir, sweep_csv(spec))
        meta = {**base_meta, "sweep": key, "swept_parameter": spec["param"],
                "levels": spec["levels"],
                "measured_level": measured_level(calib, key)}
        save_results(raws[key], out, meta)
        summ = summarise_sweep(raws[key], args.n_boot, args.boot_seed)
        summ.insert(0, "sweep", key)
        summ.to_csv(out.replace(".csv", "_summary.csv"), index=False)
        summaries[key] = summ
        print(f"[robustness] wrote {out} ({len(raws[key])} rows) + _summary.csv")

    paired_all, cross_all, deg_all = [], [], []
    for key in SWEEPS:
        pr, cr = crossover_analysis(raws[key], key, args.estimators,
                                    args.n_boot, args.boot_seed, calib=calib)
        paired_all.append(pr)
        cross_all.append(cr)
        deg_all.append(degradation_stats(summaries[key], key, args.estimators,
                                         calib=calib))
    paired = pd.concat(paired_all, ignore_index=True)
    cross = pd.concat(cross_all, ignore_index=True)
    deg = pd.concat(deg_all, ignore_index=True)

    save_results(paired, os.path.join(
        args.results_dir, f"robustness_paired_vs_perimeter{GRID_SUFFIX}.csv"),
        {**base_meta, "table": "paired difference C - A per sweep level"})
    cross.to_csv(os.path.join(args.results_dir,
                              f"robustness_crossover{GRID_SUFFIX}.csv"), index=False)
    deg.to_csv(os.path.join(args.results_dir,
                            f"robustness_degradation{GRID_SUFFIX}.csv"), index=False)

    written = make_figures(summaries, args.estimators, calib, args.figures_dir,
                           suffix=FIG_SUFFIX)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    for key, spec in SWEEPS.items():
        print(f"\n=== sweep: {key}  ({spec['param']}) ===")
        s = summaries[key]
        print(s[["estimator", "condition", "jitter_mode", "level", "n", "mean",
                 "ci_lo", "ci_hi", "p95_mean", "catastrophic_rate",
                 "n_true_events_used", "n_ghost_events_used"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n=== crossover: level at which C stops beating perimeter-only ===")
    print(cross.drop(columns=["note"]).to_string(index=False))
    for _, r in cross.iterrows():
        print(f"  [{r['sweep']}/{r['estimator']}] {r['note']}")
    print("\n=== degradation along each axis ===")
    print(deg.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\n[robustness] figures:", *written, sep="\n  ")
    return summaries, cross, deg


def make_base_meta(cfg, ds, args, calib):
    return {
        "experiment": "robustness",
        "dataset_config": cfg.as_dict(),
        "dataset_manifest": ds.manifest,
        "estimators": args.estimators,
        "estimator_kwargs": {"num_particles": args.num_particles,
                             "fov_radius": cfg.interior_fov,
                             "perimeter_sigma": cfg.perimeter_noise_std},
        "n_seeds": args.n_seeds,
        "sweep_baseline": args.sweep_baseline,
        "non_swept_held_at": dict(BASELINE),
        "non_swept_timing_bias_s": BASELINE_BIAS,
        "n_trajectories": len(ds),
        "conditions": [COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR, COND_SIDECHANNEL],
        "false_positive_rate_semantics": (
            "expected ghost count as a FRACTION of the number of true interior "
            "events on that trajectory, Poisson-distributed; NOT 1-precision"),
        "corruption_seed_rule": (
            "sha1(trajectory_id|seed_index)[:8] mod 2**31 — independent of the "
            "sweep axis and level (common random numbers across levels)"),
        "estimator_seed_rule": "seed index 0..n_seeds-1",
        "bootstrap": {"n_boot": args.n_boot, "seed": args.boot_seed,
                      "resampling_unit": "trajectory (seeds averaged first)"},
        "jitter_mode": args.jitter_mode,
        "calibration": calib_meta(calib),
        "difficulty_strata": difficulty_table(ds.trajectories),
        "notes": [
            "Three independent sweeps: one corruption parameter varies, the "
            "other two are held at zero.",
            "The zero-corruption point is shared by all three sweeps and is "
            "executed once; identical rows appear in each sweep CSV.",
            "Identical estimator configuration at every sweep point.",
            "Condition A/B reference rows carry level=NaN.",
            "Empirical jitter is a single operating point (the model ignores "
            "sigma in that mode), plotted at x = std(residuals).",
        ],
    }


def zero_equals_oracle(df, estimators):
    print("\n[robustness] correctness check — condition C at zero corruption vs oracle B")
    ok = True
    for est in estimators:
        b = df[(df["estimator"] == est) & (df["condition"] == COND_ORACLE_INTERIOR)]
        c = df[(df["estimator"] == est) & (df["condition"] == COND_SIDECHANNEL)
               & (df["jitter_mode"] == "gaussian")
               & (df["false_negative_rate"] == 0) & (df["false_positive_rate"] == 0)
               & (df["timing_jitter_std_s"] == 0)]
        key = ["trajectory_id", "corruption_seed"]
        m = b[key + ["rmse"]].merge(c[key + ["rmse"]], on=key, suffixes=("_b", "_c"))
        d = np.abs(m["rmse_b"].values - m["rmse_c"].values)
        good = bool(len(m) > 0 and np.nanmax(d) < 1e-9)
        ok &= good
        print(f"  {est:12s} n={len(m):5d}  max|RMSE_C0 - RMSE_B| = "
              f"{(np.nanmax(d) if len(m) else float('nan')):.3e}"
              f"   {'PASS' if good else 'FAIL'}")
    return ok






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trajectories", type=int, default=120)
    ap.add_argument("--estimators", nargs="+", default=["road_pf", "kalman"])
    ap.add_argument("--n_seeds", type=int, default=10,
                    help="corruption/estimator realisations per trajectory")
    ap.add_argument("--num_particles", type=int, default=2000)
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--jitter_mode", default="both",
                    choices=["gaussian", "empirical", "both"],
                    help="'empirical' resamples measured passage-detector timing residuals; "
                         "skipped with a warning if the calibrated config is absent")
    ap.add_argument("--obs_model", default=None)
    ap.add_argument("--figures_only", action="store_true",
                    help="rebuild figures from existing summary CSVs")
    ap.add_argument("--postprocess_only", action="store_true",
                    help="re-run summarisation, crossover analysis and figures "
                         "from the existing RAW sweep CSVs, without re-running "
                         "the grid. Recovers a run whose grid completed but "
                         "whose post-processing failed.")
    ap.add_argument("--append_empirical", action="store_true",
                    help="run ONLY the empirical-jitter operating point and merge "
                         "it into the existing sweep CSVs (does not re-run the grid)")
    ap.add_argument("--linear_y", action="store_true",
                    help="linear y-axis (default is log, which is needed because "
                         "perimeter-only sits several times above the curves)")
    ap.add_argument("--sweep_baseline", choices=["zero", "nominal"],
                    default="zero",
                    help="what the two NON-swept corruption parameters are held "
                         "at. 'zero' (default) isolates each channel and keeps "
                         "the grid detector-independent, reproducing the "
                         "historical CSVs. 'nominal' holds them at the selected "
                         "profile's measured operating point, showing how the "
                         "deployed system degrades; that grid is "
                         "calibration-dependent and is written to "
                         "profile-suffixed CSVs.")
    ap.add_argument("--results_dir", default=RESULTS)
    ap.add_argument("--figures_dir", default=FIGURES)
    dc.add_profile_argument(ap)
    args = ap.parse_args()











    profile = dc.resolve(args)
    sfx = dc.suffix_for(profile.name)

    calib, calib_path, calib_note = resolve_calibration(args, profile)

    global BASELINE, BASELINE_BIAS, GRID_SUFFIX, FIG_SUFFIX, MEAS_LABEL
    MEAS_LABEL = ("measured (event eval)" if profile.event_level
                  else "assumed operating point")

    if args.sweep_baseline == "nominal":
        if calib is None:
            raise SystemExit(
                "--sweep_baseline nominal needs a calibrated observation "
                f"model, but: {calib_note} ({calib_path})")
        BASELINE = {p: float(calib[p]) for p in PARAMS}
        BASELINE_BIAS = (float(profile.timing_bias_s or 0.0)
                         if profile.apply_timing_bias else 0.0)
        GRID_SUFFIX = f".nominal{sfx}"
        print(f"[robustness] sweep baseline: NOMINAL — non-swept parameters held at "
              f"fnr={BASELINE['false_negative_rate']:.6g}, "
              f"fpr={BASELINE['false_positive_rate']:.6g}, "
              f"jitter={BASELINE['timing_jitter_std_s']:.6g}s, "
              f"bias={BASELINE_BIAS:+.6g}s")
        print(f"[robustness] calibration-dependent grid -> CSVs suffixed "
              f"'{GRID_SUFFIX}'")
    else:
        print("[robustness] sweep baseline: ZERO — each channel isolated "
              "(historical, detector-independent grid)")



    FIG_SUFFIX = GRID_SUFFIX if args.sweep_baseline == "nominal" else sfx
    if calib is None:
        print(f"[robustness] no calibrated observation model: {calib_note} ({calib_path})")
        print("[robustness] measured-operating-point annotations and empirical jitter "
              "will be omitted")
    else:
        print(f"[robustness] calibration {os.path.basename(calib_path)}: "
              f"fnr={calib['false_negative_rate']:.4g} "
              f"fpr={calib['false_positive_rate']:.4g} "
              f"jitter_std={calib['timing_jitter_std_s']:.4g}s")
        if calib.get("timing_residuals_s") is not None:
            print(f"[robustness] {calib['n_residuals']} empirical timing residuals: "
                  f"mean {calib['residual_mean_s']:+.3f}s, "
                  f"std {calib['residual_std_s']:.3f}s, range "
                  f"[{calib['residual_min_s']:+.2f}, {calib['residual_max_s']:+.2f}]s")
        else:
            print("[robustness] calibrated config has no timing_residuals_s — "
                  "empirical jitter mode unavailable")



    if calib is not None:
        for key, spec in SWEEPS.items():
            m = measured_level(calib, key)
            if m is not None and not (spec["levels"][0] <= m <= spec["levels"][-1]):
                print(f"[robustness] WARNING: measured {spec['param']}={m} is OUTSIDE "
                      f"the swept range {spec['levels'][0]}..{spec['levels'][-1]}; "
                      f"extend the grid rather than extrapolating")

    logy = not args.linear_y

    if args.postprocess_only:


        cfg = load_dataset_config(n_trajectories=args.n_trajectories)
        ds = build_dataset(cfg, cache=True, verbose=False)
        raws = {}
        for key, spec in SWEEPS.items():
            path = os.path.join(args.results_dir, sweep_csv(spec))
            raws[key] = pd.read_csv(path, float_precision="round_trip")
            raws[key]["jitter_mode"] = raws[key]["jitter_mode"].fillna(
                REFERENCE_MODE)
            print(f"[robustness] loaded {path} ({len(raws[key])} rows)")
        postprocess(raws, args, make_base_meta(cfg, ds, args, calib), calib)
        return

    if args.figures_only:
        summaries = {}
        for key, spec in SWEEPS.items():
            summaries[key] = pd.read_csv(os.path.join(
                args.results_dir, sweep_csv(spec).replace(".csv", "_summary.csv")))
        written = make_figures(summaries, args.estimators, calib,
                               args.figures_dir, logy=logy, suffix=FIG_SUFFIX)
        print("[robustness] figures:", *written, sep="\n  ")
        return

    cfg = load_dataset_config(n_trajectories=args.n_trajectories)
    ds = build_dataset(cfg, cache=True)
    print(f"[robustness] {len(ds)} trajectories, {len(ds.interior_sensors)} interior + "
          f"{ds.manifest['n_perimeter_sensors']} perimeter sensors")
    print(f"[robustness] interior events/traj: {ds.manifest['interior_events_per_traj']}")

    est_kwargs = {"num_particles": args.num_particles,
                  "fov_radius": cfg.interior_fov,
                  "perimeter_sigma": cfg.perimeter_noise_std}
    base_meta = make_base_meta(cfg, ds, args, calib)
    residuals = calib.get("timing_residuals_s") if calib else None
    emp_x = empirical_x(calib)


    if args.append_empirical:
        pts = empirical_point(calib)
        if not pts:
            raise SystemExit("[robustness] --append_empirical needs a calibrated config "
                             "with timing_residuals_s; none found")
        jobs = build_jobs(ds, args, est_kwargs, pts, residuals,
                          with_references=False)
        print(f"[robustness] appending empirical-jitter point at "
              f"sigma={emp_x:.4g}s — {len(jobs)} runs")
        new = run_jobs(cfg, jobs, n_workers=args.n_workers, progress_every=500)
        new["level"] = float(emp_x)

        raws = {}
        for key, spec in SWEEPS.items():
            p = os.path.join(args.results_dir, sweep_csv(spec))
            old = pd.read_csv(p)


            old["jitter_mode"] = old["jitter_mode"].fillna(REFERENCE_MODE)
            if key == "timing_jitter":
                old = old[old["jitter_mode"] != "empirical"]
                add = new.copy()
                add["sweep"] = key
                old = pd.concat([old, add], ignore_index=True)
            raws[key] = old
        base_meta["notes"].append(
            "The empirical-jitter rows were appended by --append_empirical after "
            "the gaussian grid had been run; the grid itself is unchanged.")
        base_meta["corruption_zero_equals_oracle"] = zero_equals_oracle(
            raws["missing"], args.estimators)
        postprocess(raws, args, base_meta, calib)
        return


    points = list(gaussian_points())
    if args.jitter_mode in ("empirical", "both"):
        emp_pts = empirical_point(calib)
        if emp_pts:
            points += emp_pts
        else:
            print("[robustness] empirical jitter mode SKIPPED "
                  "(no calibrated timing residuals)")
    jobs = build_jobs(ds, args, est_kwargs, points, residuals)
    print(f"[robustness] {len(points)} unique corruption points, "
          f"{len(jobs)} runs ({args.n_seeds} seeds x {len(ds)} trajectories x "
          f"{len(args.estimators)} estimators)")

    df = run_jobs(cfg, jobs, n_workers=args.n_workers, progress_every=2000)
    print(f"[robustness] {len(df)} rows returned")
    if df["rmse"].isna().any():
        print(f"[robustness] WARNING: {int(df['rmse'].isna().sum())} rows have NaN rmse")

    base_meta["corruption_zero_equals_oracle"] = zero_equals_oracle(
        df, args.estimators)
    raws = {k: select_sweep(df, k, emp_x=emp_x) for k in SWEEPS}
    postprocess(raws, args, base_meta, calib)


if __name__ == "__main__":
    main()
