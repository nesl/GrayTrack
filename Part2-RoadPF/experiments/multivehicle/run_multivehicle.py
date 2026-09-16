
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import build_dataset, git_commit
from src import detector_calibration as dc
from src.experiment import load_dataset_config, save_results
from src.estimators import build_estimator
from src.evaluate import trajectory_metrics, is_catastrophic, summarise
from src.observation_model import SideChannelObservationModel
from src.oracle_events import generate_oracle_events
from src.road_graph import EXP_ROOT, load_road_graph
from src.scenario_gen import generate_multi_vehicle_scenario, RouteRejected
from src.schema import SENSOR_INTERIOR, SOURCE_PERIMETER, Trajectory

RESULTS = os.path.join(EXP_ROOT, "results")
FIGURES = os.path.join(EXP_ROOT, "figures")

DEV_CORRUPTION = {"false_negative_rate": 0.10, "false_positive_rate": 0.10,
                  "timing_jitter_std_s": 0.5}


def build_scenarios(graph, region, cfg, n_vehicles, n_scenarios, seed0):
    out = []
    seed = seed0
    attempts = 0
    while len(out) < n_scenarios:
        attempts += 1
        if attempts > n_scenarios * 30:
            break
        s = seed
        seed += 1
        try:
            trajs = generate_multi_vehicle_scenario(
                graph, region, seed=s, n_vehicles=n_vehicles,
                overlap_target=0.6, max_stagger_s=12.0,
                dt=cfg.dt, speed_min=cfg.speed_min, speed_max=cfg.speed_max,
                speed_noise_frac=cfg.speed_noise_frac,
                min_inside_dist=cfg.min_inside_dist)
        except (RouteRejected, RuntimeError):
            continue
        out.append({"seed": s, "trajectories": trajs})
    return out


def pad_to_common_grid(trajs, dt):
    T = max(t.n_steps for t in trajs)
    grid = np.arange(T, dtype=float) * dt
    return grid, T


def closest_approach(trajs):
    best = np.inf
    for i in range(len(trajs)):
        for j in range(i + 1, len(trajs)):
            a, b = trajs[i], trajs[j]
            lo = max(a.active_start, b.active_start)
            hi = min(a.active_end, b.active_end)
            if hi <= lo:
                continue
            d = np.linalg.norm(a.xy[lo:hi + 1] - b.xy[lo:hi + 1], axis=1)
            best = min(best, float(d.min()))
    return best if np.isfinite(best) else float("nan")


def association_metrics(log):
    real = [r for r in log if not r["is_ghost"] and r["true_vehicle_id"] is not None]
    ghosts = [r for r in log if r["is_ghost"]]
    correct = sum(1 for r in real
                  if r["assigned_vehicle_id"] == r["true_vehicle_id"])


    by_vehicle: dict[int, list] = {}
    for r in sorted(real, key=lambda r: r["timestamp"]):
        by_vehicle.setdefault(r["true_vehicle_id"], []).append(r["assigned_vehicle_id"])
    switches = 0
    for _v, seq in by_vehicle.items():
        seq = [s for s in seq if s is not None]
        switches += sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return {
        "n_real_events": len(real),
        "n_ghost_events": len(ghosts),
        "assignment_accuracy": (correct / len(real)) if real else float("nan"),
        "n_events_rejected": sum(1 for r in real if r["assigned_vehicle_id"] is None),
        "ghost_rejection_rate": (sum(1 for r in ghosts if r["assigned_vehicle_id"] is None)
                                 / len(ghosts)) if ghosts else float("nan"),
        "id_switches": switches,
    }


def run_scenario(graph, region, cfg, scen, estimator_name, condition,
                 obs_model, est_kwargs):
    trajs = scen["trajectories"]
    sensors = scen["sensors"]
    dt = cfg.dt
    grid, T = pad_to_common_grid(trajs, dt)

    events = []
    interior_all = []
    for tr in trajs:
        p, iv = generate_oracle_events(tr, sensors, region, seed=scen["seed"] + tr.vehicle_id,
                                       perimeter_noise_std=cfg.perimeter_noise_std)
        events.extend(p)
        if condition == "A_perimeter_only":
            continue
        if condition == "B_oracle_interior":
            events.extend(iv)
            interior_all.extend(iv)
        else:
            window = (float(grid[0]), float(grid[-1]))
            interior_sensors = [s for s in sensors if s.role == SENSOR_INTERIOR]
            cor = obs_model.corrupt(iv, window, interior_sensors)
            events.extend(cor)
            interior_all.extend(cor)

    est = build_estimator(estimator_name, graph, **est_kwargs)
    vids = [t.vehicle_id for t in trajs]
    out = est.run(events, grid, vehicle_ids=vids)

    rows = []
    for tr in trajs:
        est_xy = out[tr.vehicle_id]


        padded = np.full((T, 2), np.nan)
        padded[:tr.n_steps] = tr.xy
        tr_pad = Trajectory(trajectory_id=tr.trajectory_id, vehicle_id=tr.vehicle_id,
                            timestamps=grid, xy=padded,
                            active_start=tr.active_start, active_end=tr.active_end,
                            meta=tr.meta)
        m = trajectory_metrics(est_xy, tr_pad)
        rows.append({"vehicle_id": tr.vehicle_id, **m,
                     "catastrophic": is_catastrophic(m)})

    assoc = association_metrics(getattr(est, "association_log", []))
    return rows, assoc, trajs, out, grid, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenarios", type=int, default=15)
    ap.add_argument("--n_vehicles", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--estimators", nargs="+", default=["road_pf", "kalman"])
    ap.add_argument("--conditions", nargs="+",
                    default=["B_oracle_interior", "C_sidechannel"])
    ap.add_argument("--num_particles", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=90000)
    ap.add_argument("--dev_corruption", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures_dir", default=FIGURES,
                    help="where to write the qualitative figure. Point this at "
                         "a scratch directory for smoke runs — otherwise a "
                         "2-scenario debug run silently overwrites the "
                         "paper figure of the same name.")
    ap.add_argument("--figures_only", action="store_true",
                    help="redraw the qualitative figure from an existing CSV; "
                         "re-runs only the one exemplar scenario")
    dc.add_profile_argument(ap)
    args = ap.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    _sfx = ("" if args.dev_corruption
            else dc.suffix_for(dc.active_profile_name()
                               if args.detector_profile is None
                               else args.detector_profile))
    if args.out is None:
        args.out = os.path.join(RESULTS, f"multivehicle_case_study{_sfx}.csv")
    fig_path = os.path.join(args.figures_dir,
                            f"multivehicle_case_study{_sfx}.pdf")

    cfg = load_dataset_config()
    ds = build_dataset(cfg, cache=True)
    graph, region, sensors = ds.graph, ds.region, ds.sensors




    profile = None if args.dev_corruption else dc.resolve(args)
    if args.dev_corruption:
        obs_kwargs, calib_name = dict(DEV_CORRUPTION), "DEVELOPMENT_PLACEHOLDER"
    else:
        p = profile.params()
        obs_kwargs = {"false_negative_rate": p.false_negative_rate,
                      "false_positive_rate": p.false_positive_rate,
                      "timing_jitter_std_s": p.timing_jitter_std_s,


                      "timing_bias_s": ((p.timing_bias_s or 0.0)
                                        if p.apply_timing_bias else 0.0),
                      "per_camera_params": p.per_camera}
        calib_name = os.path.basename(profile.observation_model_path)
    print(f"[mv] condition C calibration: {calib_name}")

    est_kwargs = {"num_particles": args.num_particles,
                  "fov_radius": cfg.interior_fov,
                  "perimeter_sigma": cfg.perimeter_noise_std, "seed": 0}

    all_rows = []
    exemplar = None
    for nv in args.n_vehicles:
        if args.figures_only and nv != max(args.n_vehicles):
            continue
        scens = build_scenarios(graph, region, cfg, nv, args.n_scenarios,
                                args.seed0 + nv * 1000)
        print(f"[mv] n_vehicles={nv}: {len(scens)} scenarios")
        for sc in scens:
            sc["sensors"] = sensors
            sc["min_separation_m"] = closest_approach(sc["trajectories"])
        for est_name in args.estimators:
            if args.figures_only and est_name != "road_pf":
                continue
            for cond in args.conditions:
                if args.figures_only and cond != "C_sidechannel":
                    continue
                for sc in scens:
                    if args.figures_only and exemplar is not None:
                        break
                    obs = (SideChannelObservationModel(**obs_kwargs,
                                                       random_seed=sc["seed"])
                           if cond == "C_sidechannel" else None)
                    rows, assoc, trajs, out, grid, events = run_scenario(
                        graph, region, cfg, sc, est_name, cond, obs, est_kwargs)
                    for r in rows:
                        all_rows.append({
                            "n_vehicles": nv, "scenario_seed": sc["seed"],
                            "estimator": est_name, "condition": cond,
                            "calibration": calib_name if cond == "C_sidechannel" else "n/a",
                            "min_separation_m": sc["min_separation_m"],
                            "min_pairwise_overlap": trajs[0].meta.get(
                                "min_pairwise_overlap", 1.0),
                            **r, **assoc})
                    if (exemplar is None and nv == 3 and est_name == "road_pf"
                            and cond == "C_sidechannel"
                            and np.isfinite(sc["min_separation_m"])
                            and sc["min_separation_m"] < 60):
                        exemplar = (sc, trajs, out, grid, events)

    if args.figures_only:
        if exemplar is None:
            print("[mv] no close-encounter exemplar found; figure not redrawn")
            return
        plot_exemplar(exemplar, sensors, region, fig_path)
        print(f"[mv] wrote {fig_path}")
        return

    df = pd.DataFrame(all_rows)
    meta = {"experiment": "multivehicle_case_study",
            "dataset_config": cfg.as_dict(),
            "n_scenarios_requested": args.n_scenarios,
            "n_vehicles": args.n_vehicles,
            "estimators": args.estimators,
            "conditions": args.conditions,
            "observation_model": obs_kwargs,
            "calibration_file": calib_name,
            "detector_calibration": None if profile is None else profile.meta(),
            "overlap_target": 0.6, "max_stagger_s": 12.0,
            "git_commit": git_commit()}
    save_results(df, args.out, meta)
    print(f"[mv] wrote {args.out}")

    summary = []

    nanmedian = lambda v: (float(np.nanmedian(v))
                           if np.isfinite(np.asarray(v, float)).any() else float("nan"))
    for (nv, est, cond), sub in df.groupby(["n_vehicles", "estimator", "condition"]):
        s = summarise(sub["rmse"].values)
        summary.append({
            "n_vehicles": nv, "estimator": est, "condition": cond,
            "n_tracks": len(sub), "rmse_mean": s["mean"],
            "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
            "assignment_accuracy": float(np.nanmean(sub["assignment_accuracy"])),
            "id_switches_mean": float(np.nanmean(sub["id_switches"])),
            "catastrophic_rate": float(sub["catastrophic"].mean()),
            "min_separation_m_median": nanmedian(sub["min_separation_m"]),
        })
    sdf = pd.DataFrame(summary).sort_values(["estimator", "condition", "n_vehicles"])
    spath = args.out.replace(".csv", "_summary.csv")
    sdf.to_csv(spath, index=False)
    print(f"[mv] wrote {spath}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(sdf.to_string(index=False))

    if exemplar is not None:
        plot_exemplar(exemplar, sensors, region, fig_path)
        print(f"\n[mv] wrote {fig_path}")
    else:
        print("\n[mv] no close-encounter 3-vehicle exemplar found for the figure")


def plot_exemplar(exemplar, sensors, region, path):
    sc, trajs, out, grid, events = exemplar
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))

    for s in sensors:
        if s.role == SENSOR_INTERIOR:
            ax.add_patch(plt.Circle((s.x, s.y), s.fov_radius, color="0.85",
                                    alpha=0.55, lw=0, zorder=0))
    ax.add_patch(plt.Rectangle((region.x_min, region.y_min), region.width,
                               region.height, fill=False, ec="0.4", lw=0.8,
                               ls=(0, (4, 3)), zorder=1))

    for i, tr in enumerate(trajs):
        c = colors[i % len(colors)]
        sl = tr.active_slice
        ax.plot(tr.xy[sl, 0], tr.xy[sl, 1], "-", color=c, lw=1.6, zorder=3,
                label=f"vehicle {tr.vehicle_id}")



        e = out[tr.vehicle_id][sl]
        v = ~np.isnan(e).any(axis=1)
        ax.plot(e[v, 0], e[v, 1], "--", color=c, lw=1.1, alpha=0.9, zorder=4)
        ax.plot(tr.xy[tr.active_start, 0], tr.xy[tr.active_start, 1], "o",
                color=c, ms=4, zorder=5)

    ev_i = [e for e in events if e.source != SOURCE_PERIMETER]
    if ev_i:
        ax.scatter([e.sensor_x for e in ev_i], [e.sensor_y for e in ev_i],
                   marker="*", s=42, c="black", zorder=6,
                   label="anonymous passage event")
    ev_p = [e for e in events if e.source == SOURCE_PERIMETER]
    if ev_p:
        ax.scatter([e.sensor_x for e in ev_p], [e.sensor_y for e in ev_p],
                   marker="s", s=18, facecolors="none", edgecolors="black",
                   linewidths=0.8, zorder=6, label="perimeter fix (identified)")

    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.set_title(f"{len(trajs)} vehicles, closest approach "
                 f"{sc['min_separation_m']:.0f} m\n"
                 f"solid = truth, dashed = road-PF estimate", fontsize=8)


    ax.legend(fontsize=5.8, ncol=3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), columnspacing=1.0, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
