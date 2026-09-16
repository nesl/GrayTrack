"""Animate one passage-events export run as a 2D GIF.

Takes a single run folder (with ``x/`` and ``y/``) and draws:
  - every camera XY from ``scripts/util.py`` up front (grey ``+``)
  - the vehicle as an arrow that follows the ground-truth (x, y) trail in ``y/``
  - a persistent trail behind the vehicle
  - markers once each event's frame is reached:
      blue circle  = predicted TP (x matched to y)
      orange circle = ghost FP (predicted in x, no nearby GT)
      red X         = missed FN (GT in y, no nearby prediction)

Optional ``--speed-pred-csv`` also draws a second vehicle whose pose is integrated
from that CSV's ``speed_pred`` along the truth heading (lower opacity). Truth stays
more opaque so both arrows are visible when they overlap.

Example:
  python plot_passage_run_gif.py \\
    /media/ubuntu/research/passage_events_export/2026_03_04_02_41_39_91

  python plot_passage_run_gif.py \\
    /media/ubuntu/research/passage_events_export_7/2026_07_29_13_00_42_446 \\
    --speed-pred-csv /media/ubuntu/research/carla_data_aug_car_visible_pred_7/..._car_visible_pred.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

# scripts/ is on sys.path so ``import util`` matches the rest of the repo.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SCRIPTS = SCRIPT_DIR.parent
if str(REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS))

import util  # noqa: E402

# Active configs from util.py, plus the commented sensors in that same list so
# detections on inactive cameras still land on a known Town05 XY.
_EXTRA_CAMERA_XY = {
    4: (35.000, -210.000),
    5: (27.500, 212.500),
    7: (62.500, -2.500),
    10: (127.500, 0.000),
    11: (132.500, -132.500),
    12: (132.500, 127.500),
    18: (-87.500, 0.000),
    19: (-87.500, -92.500),
    20: (-87.500, 87.500),
    21: (-75.000, 145.000),
    22: (-75.000, -137.500),
    23: (-162.500, -92.500),
    24: (-155.000, -5.000),
    25: (-160.000, 87.500),
    26: (-125.000, 45.000),
    27: (-125.000, -45.000),
    28: (-175.000, -137.500),
    29: (-175.000, 145.000),
}

DEFAULT_FIRST_FRAME = 512  # older exports; passage_events_export_7 uses 256
CAMERA_ID_RE = re.compile(r"_camera_(\d+)_")


def camera_xy_map() -> dict[int, tuple[float, float]]:
    cams = dict(_EXTRA_CAMERA_XY)
    for cfg in util.CAMERA_CONFIGS:
        cam_id = cfg["id"]
        if cam_id == "overhead":
            continue
        cams[int(cam_id)] = (float(cfg["pos"][0]), float(cfg["pos"][1]))
    return cams


def camera_id_from_name(name: str) -> int | None:
    match = CAMERA_ID_RE.search(name)
    return int(match.group(1)) if match else None


def resolve_first_frame(run_dir: Path) -> int:
    """Prefer first_frame from the export index so video-frame labels stay aligned."""
    idx_path = run_dir.parent / "passage_events_index.csv"
    if idx_path.is_file():
        df = pd.read_csv(idx_path)
        if "first_frame" in df.columns:
            if "run" in df.columns:
                rows = df.loc[df["run"].astype(str) == run_dir.name, "first_frame"]
                if len(rows):
                    return int(rows.iloc[0])
            return int(df["first_frame"].iloc[0])
    return DEFAULT_FIRST_FRAME


def load_trajectory(
    y_dir: Path, first_frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frame numbers, x, y from any truth CSV in the run (they match)."""
    y_files = sorted(y_dir.glob("*_rtp_marker_pred_truth.csv"))
    if not y_files:
        y_files = sorted(y_dir.glob("*.csv"))
    if not y_files:
        raise SystemExit(f"no truth CSVs in {y_dir}")

    df = pd.read_csv(y_files[0], usecols=["x", "y"])
    n = len(df)
    frames = np.arange(first_frame, first_frame + n, dtype=np.int64)
    return frames, df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)


def gt_passage_mids(y_path: Path, first_frame: int) -> list[int]:
    """Representative (mid) frame of each contiguous car_visible run."""
    cv = pd.read_csv(y_path, usecols=["car_visible"])["car_visible"].astype(bool).to_numpy()
    mids: list[int] = []
    i = 0
    while i < len(cv):
        if not cv[i]:
            i += 1
            continue
        j = i
        while j < len(cv) and cv[j]:
            j += 1
        mids.append(first_frame + (i + j - 1) // 2)
        i = j
    return mids


def classify_detections(
    x_dir: Path,
    y_dir: Path,
    cams: dict[int, tuple[float, float]],
    match_tol_frames: int = 20,
    first_frame: int = DEFAULT_FIRST_FRAME,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Match predicted x/ passages to y/ GT midpoints per camera.

    Returns (true_positives, ghosts/FP, misses/FN). Each marker is placed at the
    camera XY; ghosts use the predicted frame, misses use the GT mid frame.
    """
    tps: list[dict] = []
    fps: list[dict] = []
    fns: list[dict] = []

    for xp in sorted(x_dir.glob("*.csv")):
        cam_id = camera_id_from_name(xp.name)
        if cam_id is None or cam_id not in cams:
            continue
        yp = y_dir / f"{xp.stem}_truth.csv"
        if not yp.is_file():
            # Older exports used an extra _rtp_marker_pred before _truth.
            yp = y_dir / f"{xp.stem}_rtp_marker_pred_truth.csv"
        if not yp.is_file():
            continue

        pred_df = pd.read_csv(xp)
        pred_frames = (
            []
            if pred_df.empty or "frame_number" not in pred_df.columns
            else [int(v) for v in pred_df.frame_number]
        )
        confs = (
            [1.0] * len(pred_frames)
            if pred_df.empty or "confidence" not in pred_df.columns
            else [float(v) for v in pred_df.confidence]
        )
        gt_frames = gt_passage_mids(yp, first_frame)
        cx, cy = cams[cam_id]

        candidates = []
        for pi, pf in enumerate(pred_frames):
            for gi, gf in enumerate(gt_frames):
                delta = abs(pf - gf)
                if delta <= match_tol_frames:
                    candidates.append((delta, pi, gi))
        candidates.sort()
        used_p: set[int] = set()
        used_g: set[int] = set()
        for _, pi, gi in candidates:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            tps.append(
                {
                    "camera_id": cam_id,
                    "frame": pred_frames[pi],
                    "confidence": confs[pi],
                    "x": cx,
                    "y": cy,
                    "kind": "tp",
                }
            )
        for pi, pf in enumerate(pred_frames):
            if pi in used_p:
                continue
            fps.append(
                {
                    "camera_id": cam_id,
                    "frame": pf,
                    "confidence": confs[pi],
                    "x": cx,
                    "y": cy,
                    "kind": "ghost",
                }
            )
        for gi, gf in enumerate(gt_frames):
            if gi in used_g:
                continue
            fns.append(
                {
                    "camera_id": cam_id,
                    "frame": gf,
                    "confidence": "",
                    "x": cx,
                    "y": cy,
                    "kind": "miss",
                }
            )

    key = lambda e: (e["frame"], e["camera_id"])
    tps.sort(key=key)
    fps.sort(key=key)
    fns.sort(key=key)
    return tps, fps, fns


def heading_deg(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Arrow yaw from consecutive positions; hold last heading when stopped."""
    dx = np.diff(xs, prepend=xs[0])
    dy = np.diff(ys, prepend=ys[0])
    # Seed the first step from the first non-trivial move so frame 0 is not NaN.
    for i in range(1, len(dx)):
        if abs(dx[i]) + abs(dy[i]) > 1e-6:
            dx[0], dy[0] = dx[i], dy[i]
            break
    yaw = np.degrees(np.arctan2(dy, dx))
    for i in range(1, len(yaw)):
        if abs(dx[i]) + abs(dy[i]) <= 1e-6:
            yaw[i] = yaw[i - 1]
    return yaw


def load_speed_series(path: Path, n: int) -> np.ndarray:
    """Return speed_pred[:n] from a car_visible_pred CSV."""
    df = pd.read_csv(path)
    if "speed_pred" not in df.columns:
        raise SystemExit(f"{path} missing speed_pred column")
    speed_pred = pd.to_numeric(df["speed_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if speed_pred.size < n:
        raise SystemExit(f"{path}: {speed_pred.size} speed rows < trajectory length {n}")
    return speed_pred[:n]


def load_truth_speed_for_pred(pred_csv: Path, truth_dir: Path, n: int) -> np.ndarray:
    """Load the camera-matched truth speed (speed is per-camera, unlike shared x/y)."""
    stem = pred_csv.name
    if stem.endswith("_car_visible_pred.csv"):
        stem = stem[: -len("_car_visible_pred.csv")]
    truth_path = truth_dir / f"{stem}_truth.csv"
    if not truth_path.is_file():
        # Fall back to export y/ copy if the research truth dir is unavailable.
        truth_path = truth_dir / f"{stem}_rtp_marker_pred_truth.csv"
    if not truth_path.is_file():
        raise SystemExit(f"no truth CSV for speed pairing: tried {stem}_truth.csv under {truth_dir}")
    speed = pd.to_numeric(
        pd.read_csv(truth_path, usecols=["speed"])["speed"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    if speed.size < n:
        raise SystemExit(f"{truth_path}: {speed.size} speed rows < trajectory length {n}")
    return speed[:n]


def integrate_speed_along_heading(
    xs: np.ndarray,
    ys: np.ndarray,
    yaw_deg: np.ndarray,
    speed_mps: np.ndarray,
    speed_truth: np.ndarray,
    dt_s: float,
    active_speed_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Dead-reckon from scalar speed_pred along the truth heading.

    Outside frames where ground-truth speed is active (car not in that camera's
    label), snap to the truth pose so the overlay only drifts from speed error
    during the scored regime.
    """
    px = np.empty_like(xs)
    py = np.empty_like(ys)
    px[0] = xs[0]
    py[0] = ys[0]
    active = speed_truth > active_speed_thr
    for i in range(1, len(xs)):
        if not (active[i] or active[i - 1]):
            px[i] = xs[i]
            py[i] = ys[i]
            continue
        heading = np.deg2rad(yaw_deg[i])
        step = float(speed_mps[i]) * dt_s
        px[i] = px[i - 1] + step * np.cos(heading)
        py[i] = py[i - 1] + step * np.sin(heading)
    return px, py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Export run folder containing x/ and y/ (or y_truth/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="GIF path (default: <run_dir>/<run_name>_passage.gif)",
    )
    parser.add_argument("--stride", type=int, default=10, help="Keep every Nth truth frame")
    parser.add_argument("--fps", type=int, default=10, help="GIF playback FPS")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--arrow-length", type=float, default=12.0, help="Arrow length in meters")
    parser.add_argument(
        "--match-tol-frames",
        type=int,
        default=20,
        help="Max |pred−GT| frames to count a predicted passage as a true positive (~1 s @ 20 FPS)",
    )
    parser.add_argument(
        "--speed-pred-csv",
        type=Path,
        default=None,
        help="car_visible_pred CSV with speed_pred; draws a second (lower-opacity) vehicle",
    )
    parser.add_argument(
        "--sim-fps",
        type=float,
        default=20.0,
        help="Simulation FPS used to integrate speed_pred into a predicted pose",
    )
    parser.add_argument(
        "--truth-alpha",
        type=float,
        default=0.95,
        help="Opacity of the ground-truth vehicle (higher than --pred-alpha)",
    )
    parser.add_argument(
        "--pred-alpha",
        type=float,
        default=0.55,
        help="Opacity of the speed-predicted vehicle",
    )
    return parser.parse_args()


def resolve_y_dir(run_dir: Path) -> Path:
    if (run_dir / "y").is_dir():
        return run_dir / "y"
    if (run_dir / "y_truth").is_dir():
        return run_dir / "y_truth"
    raise SystemExit(f"no y/ or y_truth/ under {run_dir}")


def _offsets(events: list[dict], frame: int) -> np.ndarray:
    fired = [e for e in events if e["frame"] <= frame]
    if not fired:
        return np.empty((0, 2))
    return np.array([[e["x"], e["y"]] for e in fired], dtype=float)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    x_dir = run_dir / "x"
    y_dir = resolve_y_dir(run_dir)
    if not x_dir.is_dir():
        raise SystemExit(f"missing x/ under {run_dir}")

    cams = camera_xy_map()
    first_frame = resolve_first_frame(run_dir)
    frames, xs, ys = load_trajectory(y_dir, first_frame)
    tps, ghosts, misses = classify_detections(
        x_dir,
        y_dir,
        cams,
        match_tol_frames=args.match_tol_frames,
        first_frame=first_frame,
    )
    yaw = heading_deg(xs, ys)

    speed_pred: np.ndarray | None = None
    speed_truth: np.ndarray | None = None
    xs_pred: np.ndarray | None = None
    ys_pred: np.ndarray | None = None
    if args.speed_pred_csv is not None:
        pred_csv = args.speed_pred_csv.resolve()
        speed_pred = load_speed_series(pred_csv, len(frames))
        # Prefer the full truth build; export y/ is a verbatim copy of the same files.
        truth_speed_dir = Path("/media/ubuntu/research/carla_data_aug_infer_truth_no_warmup")
        if not truth_speed_dir.is_dir():
            truth_speed_dir = y_dir
        speed_truth = load_truth_speed_for_pred(pred_csv, truth_speed_dir, len(frames))
        xs_pred, ys_pred = integrate_speed_along_heading(
            xs, ys, yaw, speed_pred, speed_truth, dt_s=1.0 / args.sim_fps
        )
        visible = speed_truth > 0.5
        vis_mae = (
            float(np.mean(np.abs(speed_pred[visible] - speed_truth[visible])))
            if visible.any()
            else float("nan")
        )
        print(
            f"Speed overlay from {pred_csv.name}: "
            f"MAE={float(np.mean(np.abs(speed_pred - speed_truth))):.3f} m/s "
            f"(visible MAE={vis_mae:.3f} m/s over {int(visible.sum())} frames)"
        )

    # Keep stride frames plus every TP / ghost / miss so markers appear on time.
    keep = np.zeros(len(frames), dtype=bool)
    keep[:: max(args.stride, 1)] = True
    keep[-1] = True
    event_frames = {e["frame"] for e in (*tps, *ghosts, *misses)}
    for i, f in enumerate(frames):
        if int(f) in event_frames:
            keep[i] = True
    indices = np.flatnonzero(keep)

    pad = 15.0
    x_min, x_max = float(xs.min()) - pad, float(xs.max()) + pad
    y_min, y_max = float(ys.min()) - pad, float(ys.max()) + pad
    for cx, cy in cams.values():
        x_min = min(x_min, cx - pad)
        x_max = max(x_max, cx + pad)
        y_min = min(y_min, cy - pad)
        y_max = max(y_max, cy + pad)
    if xs_pred is not None and ys_pred is not None:
        x_min = min(x_min, float(xs_pred.min()) - pad)
        x_max = max(x_max, float(xs_pred.max()) + pad)
        y_min = min(y_min, float(ys_pred.min()) - pad)
        y_max = max(y_max, float(ys_pred.max()) + pad)

    out_path = args.output or (run_dir / f"{run_dir.name}_passage.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)

    cam_xs = [xy[0] for xy in cams.values()]
    cam_ys = [xy[1] for xy in cams.values()]
    ax.scatter(
        cam_xs,
        cam_ys,
        marker="+",
        c="0.55",
        s=60,
        linewidths=1.2,
        zorder=2,
        label="camera",
    )
    for cam_id, (cx, cy) in sorted(cams.items()):
        ax.annotate(
            str(cam_id),
            (cx, cy),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="0.45",
        )

    trail_truth, = ax.plot(
        [], [], color="0.35", linewidth=1.5, zorder=3, label="truth trail"
    )
    trail_pred = None
    if xs_pred is not None:
        trail_pred, = ax.plot(
            [],
            [],
            color="tab:cyan",
            linewidth=1.2,
            alpha=args.pred_alpha,
            zorder=3,
            label="speed-pred trail",
        )
    arrow_artist: list[FancyArrowPatch] = []
    # Proxy artists so the dual-vehicle colors stay in the shared legend.
    ax.plot([], [], color="tab:red", linewidth=2.0, alpha=args.truth_alpha, label="truth vehicle")
    if xs_pred is not None:
        ax.plot(
            [],
            [],
            color="tab:cyan",
            linewidth=2.0,
            alpha=args.pred_alpha,
            label="speed-pred vehicle",
        )
    tp_scatter = ax.scatter(
        [], [], c="tab:blue", s=70, zorder=5, label="predicted TP (x)"
    )
    ghost_scatter = ax.scatter(
        [], [], c="tab:orange", s=90, marker="o", zorder=6, label="ghost FP (x)"
    )
    miss_scatter = ax.scatter(
        [],
        [],
        c="tab:red",
        s=90,
        marker="x",
        linewidths=2.0,
        zorder=6,
        label="missed FN (y)",
    )
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize=8)
    print(
        f"Classified detections: TP={len(tps)} ghost={len(ghosts)} miss={len(misses)}; "
        f"pre-plotted {len(cams)} cameras"
    )
    if ghosts:
        print("  ghosts:", [(g["camera_id"], g["frame"]) for g in ghosts])
    if misses:
        print("  misses:", [(m["camera_id"], m["frame"]) for m in misses])

    arrow_len = float(args.arrow_length)

    def clear_arrow() -> None:
        while arrow_artist:
            arrow_artist.pop().remove()

    def add_arrow(tip_x: float, tip_y: float, heading_rad: float, color: str, alpha: float, z: int) -> None:
        base_x = tip_x - arrow_len * np.cos(heading_rad)
        base_y = tip_y - arrow_len * np.sin(heading_rad)
        patch = FancyArrowPatch(
            (base_x, base_y),
            (tip_x, tip_y),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color=color,
            alpha=alpha,
            zorder=z,
        )
        ax.add_patch(patch)
        arrow_artist.append(patch)

    def draw_frame(step: int) -> None:
        idx = int(indices[step])
        frame = int(frames[idx])
        clear_arrow()

        trail_truth.set_data(xs[: idx + 1], ys[: idx + 1])
        if trail_pred is not None and xs_pred is not None and ys_pred is not None:
            trail_pred.set_data(xs_pred[: idx + 1], ys_pred[: idx + 1])

        heading = np.deg2rad(yaw[idx])
        # Draw predicted first (under), then truth on top with higher opacity.
        if xs_pred is not None and ys_pred is not None:
            add_arrow(
                float(xs_pred[idx]),
                float(ys_pred[idx]),
                heading,
                color="tab:cyan",
                alpha=args.pred_alpha,
                z=4,
            )
        add_arrow(
            float(xs[idx]),
            float(ys[idx]),
            heading,
            color="tab:red",
            alpha=args.truth_alpha,
            z=5,
        )

        tp_scatter.set_offsets(_offsets(tps, frame))
        ghost_scatter.set_offsets(_offsets(ghosts, frame))
        miss_scatter.set_offsets(_offsets(misses, frame))
        title_bits = [
            f"{run_dir.name}",
            f"frame {frame}",
            f"TP={sum(e['frame'] <= frame for e in tps)}",
            f"ghost={sum(e['frame'] <= frame for e in ghosts)}",
            f"miss={sum(e['frame'] <= frame for e in misses)}",
        ]
        if speed_pred is not None and speed_truth is not None:
            title_bits.append(
                f"v_true={speed_truth[idx]:.2f} v_pred={speed_pred[idx]:.2f} m/s"
            )
        title.set_text("  |  ".join(title_bits))

    anim = FuncAnimation(
        fig,
        draw_frame,
        frames=len(indices),
        interval=1000 / max(args.fps, 1),
        blit=False,
    )
    print(
        f"Writing {out_path} ({len(indices)} frames, stride={args.stride})..."
    )
    anim.save(out_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
