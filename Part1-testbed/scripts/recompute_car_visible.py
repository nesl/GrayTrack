#!/usr/bin/env python3
"""
Recompute car_visible in-place for recorded CARLA truth CSVs.

For each run under ROOT:
  <run>/params.json          — camera pos/rot, optional vehicle_bounding_box
  <run>/y/camera_*_truth.csv — rewritten car_visible column only
  <run>/videos/camera_*.mp4  — optional, used to estimate pose/video lag

Visibility = intersection (including edge/point touch) between:
  1) camera FOV projected onto z=0 via the CARLA pinhole model
  2) vehicle XY footprint (flat oriented box from BoundingBox extent/location)

Older recordings tick the world without draining camera queues between
waypoints, so CSV poses lead the video. When videos are present we estimate
that lag (same for every camera in a run) and evaluate the footprint at the
pose that produced each video frame.

Skips camera_overhead_truth.csv in every folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import util


DEFAULT_ROOT = "/media/ubuntu/samsung_scratch/carla"
OVERHEAD_NAME = "camera_overhead_truth.csv"


def _camera_id_from_truth_name(name: str):
    # camera_1_truth.csv -> "1", camera_16_truth.csv -> "16"
    if not (name.startswith("camera_") and name.endswith("_truth.csv")):
        return None
    mid = name[len("camera_") : -len("_truth.csv")]
    return mid


def _load_params(params_path: Path):
    with open(params_path, "r") as f:
        return json.load(f)


def _vehicle_bb_from_params(params: dict):
    bb = params.get("vehicle_bounding_box") or {}
    return {
        "extent_x": float(bb.get("extent_x", util.PRIUS_EXTENT_X)),
        "extent_y": float(bb.get("extent_y", util.PRIUS_EXTENT_Y)),
        "offset_x": float(bb.get("offset_x", util.PRIUS_BBOX_OFFSET_X)),
        "offset_y": float(bb.get("offset_y", util.PRIUS_BBOX_OFFSET_Y)),
    }


def _load_fov_polygons(params: dict):
    camera_params = params.get("camera_params") or {}
    fov_by_id = {}
    for cam_id, cfg in camera_params.items():
        if cam_id == "overhead":
            continue
        pos = tuple(cfg["pos"])
        rot = tuple(cfg["rot"])
        fov_by_id[str(cam_id)] = util.camera_frustum_polygon_at_z(
            pos, rot, ground_z=0.0
        )
    return fov_by_id


def _project_point(Xw, camera_pos, camera_rot):
    M = np.array(util._camera_transform(camera_pos, camera_rot).get_matrix())
    R, origin = M[:3, :3], M[:3, 3]
    d = R.T @ (Xw - origin)
    if d[0] <= 1e-6:
        return None
    focal = util.WIDTH / (2.0 * np.tan(np.deg2rad(util.FOV) / 2.0))
    return np.array(
        [
            focal * (d[1] / d[0]) + util.WIDTH / 2.0,
            focal * (-d[2] / d[0]) + util.HEIGHT / 2.0,
        ]
    )


def _dark_red_car_mask(frame_bgr):
    """Largest dark-red blob in the upper 65% of the frame (road, not plaza)."""
    import cv2

    b, g, r = cv2.split(frame_bgr)
    m = (
        (r.astype(np.int16) - g > 50)
        & (r.astype(np.int16) - b > 50)
        & (r > 60)
        & (r < 180)
        & (g < 100)
        & (b < 100)
    )
    m[int(util.HEIGHT * 0.65) :, :] = False
    num, labels, stats, cents = cv2.connectedComponentsWithStats(
        m.astype(np.uint8), 8
    )
    if num <= 1:
        return None, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))
    if areas[i - 1] < 200:
        return None, None
    return labels == i, cents[i]


def _estimate_pose_video_lag_for_camera(
    video_path: Path,
    rows,
    vis_idx,
    camera_pos,
    camera_rot,
    vehicle_bb: dict,
    max_lag: int = 40,
):
    """Lag for one camera from mid-visibility OBB↔blob alignment. 0 if unknown."""
    import cv2

    if len(vis_idx) < 30:
        return 0

    lo, hi = vis_idx[0], vis_idx[-1]
    sample_frames = sorted(
        {
            int(round(lo + 0.35 * (hi - lo))),
            int(round(lo + 0.50 * (hi - lo))),
            int(round(lo + 0.65 * (hi - lo))),
        }
    )
    ex = vehicle_bb["extent_x"]
    ey = vehicle_bb["extent_y"]
    ox = vehicle_bb["offset_x"]
    oy = vehicle_bb["offset_y"]

    votes = []
    for fi in sample_frames:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            continue
        blob, cent = _dark_red_car_mask(fr)
        if blob is None:
            continue
        best = None
        for K in range(0, max_lag + 1):
            j = max(0, fi - K)
            row = rows[j]
            poly = util.vehicle_footprint_polygon(
                float(row["x"]),
                float(row["y"]),
                float(row["theta2"]),
                extent_x=ex,
                extent_y=ey,
                offset_x=ox,
                offset_y=oy,
            )
            uvs = []
            for wx, wy in poly:
                p = _project_point(np.array([wx, wy, 0.4]), camera_pos, camera_rot)
                if p is None:
                    uvs = None
                    break
                uvs.append(p)
            if uvs is None:
                continue
            uvs = np.asarray(uvs)
            pm = np.zeros((util.HEIGHT, util.WIDTH), np.uint8)
            cv2.fillPoly(pm, [uvs.astype(np.int32)], 1)
            pm = pm.astype(bool)
            inter = int((pm & blob).sum())
            union = int((pm | blob).sum())
            iou = inter / max(1, union)
            pc = uvs.mean(axis=0)
            dist = float(np.hypot(pc[0] - cent[0], pc[1] - cent[1]))
            sc = (iou, -dist)
            if best is None or sc > best[0]:
                best = (sc, K)
        if best is not None and best[0][0] >= 0.05:
            votes.append(best[1])

    if len(votes) < 2:
        return 0
    return int(np.median(votes))


def _estimate_pose_video_lag(
    run_dir: Path,
    params: dict,
    vehicle_bb: dict,
    max_lag: int = 40,
):
    """
    Backward-compat helper: median lag across cameras with a clear visible span.
    Prefer per-camera estimation via process_run.
    """
    camera_params = params.get("camera_params") or {}
    y_dir = run_dir / "y"
    videos_dir = run_dir / "videos"
    if not videos_dir.is_dir():
        return 0
    lags = []
    for csv_path in sorted(y_dir.glob("camera_*_truth.csv")):
        if csv_path.name == OVERHEAD_NAME:
            continue
        cam_id = _camera_id_from_truth_name(csv_path.name)
        if cam_id is None or cam_id not in camera_params:
            continue
        video_path = videos_dir / f"camera_{cam_id}.mp4"
        if not video_path.is_file():
            continue
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        vis_idx = [
            i
            for i, r in enumerate(rows)
            if r.get("car_visible", "").strip().lower() in ("true", "1", "yes")
        ]
        cfg = camera_params[cam_id]
        lag = _estimate_pose_video_lag_for_camera(
            video_path,
            rows,
            vis_idx,
            tuple(cfg["pos"]),
            tuple(cfg["rot"]),
            vehicle_bb,
            max_lag=max_lag,
        )
        if lag > 0:
            lags.append(lag)
    if not lags:
        return 0
    return int(np.median(lags))


def _recompute_csv(
    csv_path: Path,
    fov_poly,
    vehicle_bb: dict,
    pose_lag_frames: int = 0,
    dry_run: bool = False,
):
    """
    Rewrite only the car_visible column, preserving every other field's
    exact original text (no float round-trip).
    """
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{csv_path}: empty file")
        rows = list(reader)

    try:
        vis_idx = header.index("car_visible")
        x_idx = header.index("x")
        y_idx = header.index("y")
        yaw_idx = header.index("theta2")
    except ValueError as e:
        raise KeyError(f"{csv_path}: missing required column ({e})") from e

    fov_aabb = (
        float(fov_poly[:, 0].min()),
        float(fov_poly[:, 1].min()),
        float(fov_poly[:, 0].max()),
        float(fov_poly[:, 1].max()),
    )

    n_true_old = 0
    n_true_new = 0
    n_flip = 0
    new_flags = []
    for i, row in enumerate(rows):
        old_val = row[vis_idx].strip().lower() in ("true", "1", "yes")
        j = max(0, i - pose_lag_frames)
        src = rows[j]
        new_val = util.is_car_visible_in_fov(
            float(src[x_idx]),
            float(src[y_idx]),
            float(src[yaw_idx]),
            fov_poly,
            fov_aabb=fov_aabb,
            extent_x=vehicle_bb["extent_x"],
            extent_y=vehicle_bb["extent_y"],
            offset_x=vehicle_bb["offset_x"],
            offset_y=vehicle_bb["offset_y"],
        )
        n_true_old += int(old_val)
        n_true_new += int(new_val)
        n_flip += int(old_val != new_val)
        new_flags.append(new_val)

    if not dry_run and n_flip > 0:
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=csv_path.stem + ".",
            suffix=".csv.tmp",
            dir=str(csv_path.parent),
        )
        os.close(tmp_fd)
        try:
            with open(tmp_name, "w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(header)
                for row, flag in zip(rows, new_flags):
                    row[vis_idx] = "True" if flag else "False"
                    writer.writerow(row)
            os.replace(tmp_name, csv_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    return {
        "rows": len(rows),
        "true_old": n_true_old,
        "true_new": n_true_new,
        "flipped": n_flip,
        "wrote": (not dry_run) and n_flip > 0,
    }


def process_run(
    run_dir: Path,
    dry_run: bool = False,
    align_video: bool = True,
    pose_lag_frames: int | None = None,
):
    params_path = run_dir / "params.json"
    y_dir = run_dir / "y"
    videos_dir = run_dir / "videos"
    if not params_path.is_file() or not y_dir.is_dir():
        return None

    params = _load_params(params_path)
    vehicle_bb = _vehicle_bb_from_params(params)
    fov_by_id = _load_fov_polygons(params)
    camera_params = params.get("camera_params") or {}

    # First pass: per-camera lag from video alignment (or forced lag).
    cam_meta = []
    for csv_path in sorted(y_dir.glob("camera_*_truth.csv")):
        if csv_path.name == OVERHEAD_NAME:
            continue
        cam_id = _camera_id_from_truth_name(csv_path.name)
        if cam_id is None or cam_id not in fov_by_id:
            if cam_id is not None:
                print(f"  SKIP {csv_path.name}: no camera params for id={cam_id}")
            continue
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        vis_idx = [
            i
            for i, r in enumerate(rows)
            if r.get("car_visible", "").strip().lower() in ("true", "1", "yes")
        ]
        mid = int(0.5 * (vis_idx[0] + vis_idx[-1])) if vis_idx else 0
        if pose_lag_frames is not None:
            lag = pose_lag_frames
        elif align_video and videos_dir.is_dir():
            video_path = videos_dir / f"camera_{cam_id}.mp4"
            if video_path.is_file() and cam_id in camera_params:
                cfg = camera_params[cam_id]
                lag = _estimate_pose_video_lag_for_camera(
                    video_path,
                    rows,
                    vis_idx,
                    tuple(cfg["pos"]),
                    tuple(cfg["rot"]),
                    vehicle_bb,
                )
            else:
                lag = 0
        else:
            lag = 0
        cam_meta.append(
            {
                "csv_path": csv_path,
                "cam_id": cam_id,
                "mid": mid,
                "lag": lag,
                "n_vis": len(vis_idx),
            }
        )

    # Fill gaps: lag grows over the run, so interpolate from cameras that aligned.
    anchors = [(m["mid"], m["lag"]) for m in cam_meta if m["lag"] > 0 and m["n_vis"] >= 30]
    if anchors and pose_lag_frames is None and align_video:
        anchors.sort()
        xs = np.array([a[0] for a in anchors], dtype=float)
        ys = np.array([a[1] for a in anchors], dtype=float)
        for m in cam_meta:
            if m["lag"] == 0 and m["n_vis"] >= 30:
                m["lag"] = int(np.clip(round(float(np.interp(m["mid"], xs, ys))), 0, 60))
                m["lag_src"] = "interp"
            else:
                m["lag_src"] = "video" if m["lag"] > 0 else "none"
    else:
        for m in cam_meta:
            m["lag_src"] = "forced" if pose_lag_frames is not None else (
                "video" if m["lag"] > 0 else "none"
            )

    results = []
    for m in cam_meta:
        stats = _recompute_csv(
            m["csv_path"],
            fov_by_id[m["cam_id"]],
            vehicle_bb,
            pose_lag_frames=m["lag"],
            dry_run=dry_run,
        )
        stats["camera"] = m["cam_id"]
        stats["path"] = str(m["csv_path"])
        stats["pose_lag_frames"] = m["lag"]
        results.append(stats)
        action = "dry-run" if dry_run else ("wrote" if stats["wrote"] else "unchanged")
        print(
            f"  camera_{m['cam_id']}: rows={stats['rows']} "
            f"true {stats['true_old']}->{stats['true_new']} "
            f"flipped={stats['flipped']} lag={m['lag']}({m['lag_src']}) [{action}]"
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Recompute car_visible via FOV∩vehicle-footprint (in place)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help=f"Dataset root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Optional single run folder name or path under root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats without writing CSVs",
    )
    parser.add_argument(
        "--no-align-video",
        action="store_true",
        help="Do not estimate pose/video lag from videos (use raw CSV pose)",
    )
    parser.add_argument(
        "--pose-lag-frames",
        type=int,
        default=None,
        help="Force a pose-leads-video lag (skips auto estimate)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")

    if args.run:
        run_path = Path(args.run)
        if not run_path.is_absolute():
            run_path = root / args.run
        runs = [run_path]
    else:
        runs = sorted(p for p in root.iterdir() if p.is_dir())

    total_csv = 0
    total_flip = 0
    total_wrote = 0
    for run_dir in runs:
        print(f"\n=== {run_dir.name} ===")
        results = process_run(
            run_dir,
            dry_run=args.dry_run,
            align_video=not args.no_align_video,
            pose_lag_frames=args.pose_lag_frames,
        )
        if results is None:
            print("  SKIP (missing params.json or y/)")
            continue
        total_csv += len(results)
        total_flip += sum(r["flipped"] for r in results)
        total_wrote += sum(1 for r in results if r["wrote"])

    print(
        f"\nDone. csvs={total_csv} flipped_rows={total_flip} "
        f"files_written={total_wrote}"
        + (" (dry-run)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
