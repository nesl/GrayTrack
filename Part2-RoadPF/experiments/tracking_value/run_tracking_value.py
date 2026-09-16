
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.experiment import (CONDITIONS, COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR,
                            COND_SIDECHANNEL, run_jobs, save_results, aggregate,
                            load_dataset_config, build_event_stream)
from src.dataset import build_dataset
from src import detector_calibration as dc
from src.estimators import build_estimator
from src.observation_model import load_params, SideChannelObservationModel
from src.evaluate import difficulty_table, trajectory_metrics
from src.road_graph import EXP_ROOT
from src.schema import SENSOR_INTERIOR, SOURCE_PERIMETER

RESULTS = os.path.join(EXP_ROOT, "results")
FIGURES = os.path.join(EXP_ROOT, "figures")



COND_COLOUR = {COND_PERIMETER_ONLY: "#999999",
               COND_ORACLE_INTERIOR: "#009E73",
               COND_SIDECHANNEL: "#0072B2"}
COND_PRETTY = {COND_PERIMETER_ONLY: "A: perimeter only",
               COND_ORACLE_INTERIOR: "B: + oracle interior",
               COND_SIDECHANNEL: "C: + side-channel"}
EST_PRETTY = {"road_pf": "Road-PF", "kalman": "Kalman",
              "dead_reckoning": "Dead reckoning"}




DEV_CORRUPTION = {"false_negative_rate": 0.10,
                  "false_positive_rate": 0.10,
                  "timing_jitter_std_s": 0.5}


def stable_seed(*parts) -> int:
    blob = "|".join(repr(p) for p in parts).encode()
    return int(hashlib.sha1(blob).hexdigest()[:8], 16) % (2 ** 31)


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






def plot_bar(agg, estimators, path, calib_note, n_traj):
    conds = [COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR, COND_SIDECHANNEL]
    conds = [c for c in conds if c in set(agg["condition"])]
    idx = np.arange(len(estimators), dtype=float)
    w = 0.8 / len(conds)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for k, cond in enumerate(conds):
        means, los, his = [], [], []
        for est in estimators:
            row = agg[(agg["estimator"] == est) & (agg["condition"] == cond)]
            m = float(row["mean"].iloc[0]) if len(row) else np.nan
            means.append(m)
            los.append(m - float(row["ci_lo"].iloc[0]) if len(row) else 0.0)
            his.append(float(row["ci_hi"].iloc[0]) - m if len(row) else 0.0)
        ax.bar(idx + k * w - 0.4 + w / 2, means, width=w * 0.92,
               yerr=[los, his], capsize=2.5, color=COND_COLOUR[cond],
               label=COND_PRETTY[cond],
               error_kw={"elinewidth": 0.9, "capthick": 0.9})

    ax.set_xticks(idx)
    ax.set_xticklabels([EST_PRETTY.get(e, e) for e in estimators], fontsize=8)
    ax.set_ylabel("Trajectory RMSE (m)", fontsize=8)
    ax.tick_params(labelsize=7.5)



    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), columnspacing=1.1, handlelength=1.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(f"{n_traj} region transits, CARLA Town05\nC: {calib_note}",
                  fontsize=6.0, labelpad=5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_examples(ds, obs_kwargs, est_kwargs, estimator, path, seed=0,
                  n_examples=3):
    graph, region = ds.graph, ds.region
    interior = ds.interior_sensors


    trajs = sorted(ds.trajectories,
                   key=lambda t: t.meta.get("max_blind_gap_s", 0.0))
    if len(trajs) >= n_examples:
        picks = [trajs[len(trajs) // 6], trajs[len(trajs) // 2],
                 trajs[-1 - len(trajs) // 6]][:n_examples]
    else:
        picks = trajs[:n_examples]

    conds = [COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR]
    if obs_kwargs is not None:
        conds.append(COND_SIDECHANNEL)

    fig, axes = plt.subplots(len(picks), len(conds),
                             figsize=(2.3 * len(conds), 2.3 * len(picks)),
                             squeeze=False)
    for r, tr in enumerate(picks):
        for c, cond in enumerate(conds):
            ax = axes[r][c]
            obs = (SideChannelObservationModel(**obs_kwargs, random_seed=seed)
                   if cond == COND_SIDECHANNEL else None)
            events, interior_used = build_event_stream(ds, tr, cond, obs)
            est = build_estimator(estimator, graph, **{**est_kwargs, "seed": seed})
            out = est.run(events, tr.timestamps, vehicle_ids=[tr.vehicle_id])
            m = trajectory_metrics(out[tr.vehicle_id], tr)

            for s in interior:
                ax.add_patch(plt.Circle((s.x, s.y), s.fov_radius, color="0.9",
                                        lw=0, zorder=0))
            ax.add_patch(plt.Rectangle((region.x_min, region.y_min),
                                       region.width, region.height, fill=False,
                                       ec="0.55", lw=0.7, ls=(0, (4, 3)), zorder=1))

            sl = tr.active_slice
            ax.plot(tr.xy[sl, 0], tr.xy[sl, 1], "-", color="0.15", lw=1.5,
                    zorder=3, label="ground truth")
            e = out[tr.vehicle_id][sl]
            v = ~np.isnan(e).any(axis=1)
            ax.plot(e[v, 0], e[v, 1], "--", color=COND_COLOUR[cond], lw=1.3,
                    zorder=4, label="estimate")

            real = [ev for ev in interior_used if not ev.is_ghost]
            ghost = [ev for ev in interior_used if ev.is_ghost]
            if real:
                ax.scatter([ev.sensor_x for ev in real],
                           [ev.sensor_y for ev in real], marker="*", s=44,
                           c="black", zorder=6, label="passage event")
            if ghost:
                ax.scatter([ev.sensor_x for ev in ghost],
                           [ev.sensor_y for ev in ghost], marker="x", s=26,
                           c="#D55E00", linewidths=1.2, zorder=6, label="ghost")
            perim = [ev for ev in events if ev.source == SOURCE_PERIMETER]
            ax.scatter([ev.sensor_x for ev in perim],
                       [ev.sensor_y for ev in perim], marker="s", s=16,
                       facecolors="none", edgecolors="black", linewidths=0.8,
                       zorder=6, label="perimeter fix")

            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{COND_PRETTY[cond]}\nRMSE {m['rmse']:.0f} m",
                         fontsize=7)
            if c == 0:
                ax.set_ylabel(f"blind gap "
                              f"{tr.meta.get('max_blind_gap_s', float('nan')):.0f} s",
                              fontsize=7)




    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="0.15", lw=1.5, label="ground truth"),
        Line2D([], [], color="0.35", lw=1.3, ls="--", label="estimate"),
        Line2D([], [], color="black", marker="*", ls="none", ms=7,
               label="anonymous passage event"),
        Line2D([], [], color="#D55E00", marker="x", ls="none", ms=5,
               mew=1.2, label="ghost event"),
        Line2D([], [], color="black", marker="s", ls="none", ms=4,
               mfc="none", mew=0.8, label="perimeter fix (identified)"),
        Line2D([], [], color="0.85", marker="o", ls="none", ms=7,
               label="interior camera FOV"),
    ]
    fig.legend(handles=handles, fontsize=6.5, ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trajectories", type=int, default=120)
    ap.add_argument("--estimators", nargs="+",
                    default=["road_pf", "kalman", "dead_reckoning"])
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="corruption realisations per trajectory in condition C")
    ap.add_argument("--num_particles", type=int, default=2000)
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--obs_model", default=None)
    ap.add_argument("--dev_corruption", action="store_true")
    ap.add_argument("--skip_condition_c", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures_dir", default=FIGURES)
    ap.add_argument("--figures_only", action="store_true",
                    help="redraw from an existing results CSV without re-running")
    ap.add_argument("--example_estimator", default="road_pf")
    dc.add_profile_argument(ap)
    args = ap.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)




    profile = None if args.dev_corruption else dc.resolve(args)
    sfx = "" if profile is None else dc.suffix_for(profile.name)
    if args.out is None:
        args.out = os.path.join(RESULTS, f"tracking_value{sfx}.csv")

    cfg = load_dataset_config(n_trajectories=args.n_trajectories)
    ds = build_dataset(cfg, cache=True)
    print(f"[tracking_value] {len(ds)} trajectories, {len(ds.interior_sensors)} interior "
          f"sensors, {ds.manifest['n_perimeter_sensors']} perimeter sensors")
    print(f"[tracking_value] interior events/traj: {ds.manifest['interior_events_per_traj']}")

    obs_kwargs = calib_name = calib_source = None
    conditions = [COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR]
    if not args.skip_condition_c:
        obs_kwargs, calib_name, calib_source = resolve_observation_model(args, profile)
        conditions.append(COND_SIDECHANNEL)
        print(f"[tracking_value] condition C calibration: {calib_name} ({calib_source})")

    est_kwargs = {"num_particles": args.num_particles,
                  "fov_radius": cfg.interior_fov,
                  "perimeter_sigma": cfg.perimeter_noise_std}

    jobs = []
    for est in args.estimators:
        for tr in ds.trajectories:

            for cond in (COND_PERIMETER_ONLY, COND_ORACLE_INTERIOR):
                jobs.append({"trajectory_id": tr.trajectory_id, "estimator": est,
                             "condition": cond, "obs": None,
                             "est_kwargs": {**est_kwargs, "seed": 0},
                             "tags": {"corruption_seed": -1,
                                      "calibration": "n/a"}})
            if args.skip_condition_c:
                continue

            for s in range(args.n_seeds):
                seed = stable_seed(tr.trajectory_id, s)
                jobs.append({
                    "trajectory_id": tr.trajectory_id, "estimator": est,
                    "condition": COND_SIDECHANNEL,
                    "obs": {**obs_kwargs, "random_seed": seed},
                    "est_kwargs": {**est_kwargs, "seed": s},
                    "tags": {"corruption_seed": s, "calibration": calib_name}})

    if args.figures_only:
        df = pd.read_csv(args.out)
    else:
        print(f"[tracking_value] {len(jobs)} runs across {len(args.estimators)} estimators")
        df = run_jobs(cfg, jobs, n_workers=args.n_workers)

    meta = {
        "experiment": "tracking_value",
        "dataset_config": cfg.as_dict(),
        "dataset_manifest": ds.manifest,
        "conditions": conditions,
        "estimators": args.estimators,
        "n_corruption_seeds": args.n_seeds,
        "estimator_kwargs": est_kwargs,
        "observation_model": obs_kwargs,
        "calibration_source": calib_source,
        "calibration_file": calib_name,
        "detector_calibration": None if profile is None else profile.meta(),
        "difficulty_strata": difficulty_table(ds.trajectories),
    }
    if not args.figures_only:
        save_results(df, args.out, meta)
        print(f"[tracking_value] wrote {args.out}")


    agg = aggregate(df, ["estimator", "condition"])
    agg_path = args.out.replace(".csv", "_summary.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[tracking_value] wrote {agg_path}\n")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(agg[["estimator", "condition", "n", "mean", "ci_lo", "ci_hi",
                   "p95_mean", "catastrophic_rate"]].to_string(index=False))


    print()
    for est in args.estimators:
        sub = agg[agg["estimator"] == est].set_index("condition")
        if COND_PERIMETER_ONLY in sub.index and COND_SIDECHANNEL in sub.index:
            a = sub.loc[COND_PERIMETER_ONLY, "mean"]
            b = sub.loc[COND_ORACLE_INTERIOR, "mean"]
            c = sub.loc[COND_SIDECHANNEL, "mean"]
            print(f"[tracking_value] {est:16s} A={a:7.2f}m  B={b:7.2f}m ({100*(b-a)/a:+.1f}%)  "
                  f"C={c:7.2f}m ({100*(c-a)/a:+.1f}% vs A)")

    if obs_kwargs is None:
        calib_note = "not run"
    elif calib_name == "DEVELOPMENT_PLACEHOLDER":
        calib_note = "DEVELOPMENT PLACEHOLDER — not a paper result"
    else:
        who = "" if profile is None else f"{profile.label()}: "
        calib_note = (f"{who}FN {obs_kwargs['false_negative_rate']:.2f}, "
                      f"ghosts {obs_kwargs['false_positive_rate']:.2f}/event, "
                      f"jitter $\\sigma$={obs_kwargs['timing_jitter_std_s']:.2f}s")

    bar_path = os.path.join(args.figures_dir, f"tracking_value_bar{sfx}.pdf")
    plot_bar(agg, args.estimators, bar_path, calib_note, len(ds))
    print(f"\n[tracking_value] wrote {bar_path}")

    ex_path = os.path.join(args.figures_dir, f"tracking_value_example_trajectories{sfx}.pdf")
    plot_examples(ds, obs_kwargs, est_kwargs, args.example_estimator, ex_path)
    print(f"[tracking_value] wrote {ex_path}")


if __name__ == "__main__":
    main()
