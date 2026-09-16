"""Aggregate X infer CSV rows (packets) into one row per predicted video frame."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Hardcoded data roots (X = model output).
XY_INFER_DATA_DIR = Path("/media/ubuntu/research/carla_data_aug_combined_x_infer")

# Video frame truth CSVs (one row per frame: x,y,z,theta1,...).
XY_VIDEO_FRAME_TRUTH_DIR = Path("/media/ubuntu/research/carla_data_aug_infer_truth")
XY_VIDEO_FRAME_TRUTH_INTERPOLATED_DIR = (
    XY_VIDEO_FRAME_TRUTH_DIR.parent / f"{XY_VIDEO_FRAME_TRUTH_DIR.name}_interpolated"
)

# Sibling directory: <parent>/<stem>_preprocessed/ (e.g. xy_infer_data_preprocessed).
XY_INFER_DATA_PREPROCESSED_DIR = XY_INFER_DATA_DIR.parent / f"{XY_INFER_DATA_DIR.name}_preprocessed"

# Truth interpolation (stretch/shrink row count to match preprocessed X frames).
INTERP_NUMERIC_KIND = "linear"  # "linear" -> np.interp along row index
INTERP_CAR_VISIBLE = "nearest"  # "nearest" | "round" (round: linear then >= 0.5)

TRUTH_COLUMNS = [
    "x",
    "y",
    "z",
    "theta1",
    "theta2",
    "theta3",
    "vx",
    "vy",
    "vz",
    "speed",
    "car_visible",
]
TRUTH_NUMERIC_COLUMNS = [c for c in TRUTH_COLUMNS if c != "car_visible"]


def _base_stem(stem: str) -> str:
    for suffix in ("_truth", "_rtp_marker_pred"):
        while stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _pred_marker_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes"]) | (pd.to_numeric(series, errors="coerce") == 1)


def segment_indices(pred_marker: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive packet index ranges [start, end] for each predicted frame.

    Rows with pred_marker True close the current frame (RTP M bit semantics).
    Any trailing packets after the last marker form one additional frame.
    """
    n = len(pred_marker)
    if n == 0:
        return []
    if not pred_marker.any():
        return [(0, n - 1)]

    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(n):
        if pred_marker[i]:
            ranges.append((start, i))
            start = i + 1
    if start < n:
        ranges.append((start, n - 1))
    return ranges


def aggregate_frame_packets(df: pd.DataFrame, start: int, end: int, video_frame_index: int) -> dict:
    w = df.iloc[start : end + 1]
    fl = w["frame.len"].astype(float)
    tr = w["frame.time_relative"].astype(float)
    fn = w["frame.number"]
    rs = w["rtp.seq"].astype(float)
    rt = w["rtp.timestamp"].astype(float)
    te = w["frame.time_epoch"].astype(float)

    t0, t1 = float(tr.iloc[0]), float(tr.iloc[-1])
    return {
        "video_frame_index": video_frame_index,
        "packet_count": len(w),
        "combined_frame_len_bytes": int(fl.sum()),
        "frame_time_relative_first": t0,
        "frame_time_relative_last": t1,
        "frame_time_span": t1 - t0,
        "frame_number_first": fn.iloc[0],
        "frame_number_last": fn.iloc[-1],
        "rtp_seq_first": rs.iloc[0],
        "rtp_seq_last": rs.iloc[-1],
        "rtp_timestamp_first": rt.iloc[0],
        "rtp_timestamp_last": rt.iloc[-1],
        "frame_time_epoch_first": float(te.iloc[0]),
        "frame_time_epoch_last": float(te.iloc[-1]),
    }


def interpolate_truth_rows(df_truth: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """Resample truth rows to ``target_n`` via index-space interpolation."""
    n = len(df_truth)
    if n == target_n:
        return df_truth[TRUTH_COLUMNS].copy()

    x_old = np.arange(n, dtype=float)
    x_new = np.linspace(0, n - 1, target_n)

    out: dict[str, np.ndarray] = {}
    for col in TRUTH_NUMERIC_COLUMNS:
        y = df_truth[col].astype(float).to_numpy()
        if INTERP_NUMERIC_KIND == "linear":
            out[col] = np.interp(x_new, x_old, y)
        else:
            raise ValueError(f"unsupported INTERP_NUMERIC_KIND: {INTERP_NUMERIC_KIND}")

    cv = df_truth["car_visible"].to_numpy()
    if INTERP_CAR_VISIBLE == "nearest":
        idx = np.clip(np.round(x_new).astype(int), 0, n - 1)
        out["car_visible"] = cv[idx]
    elif INTERP_CAR_VISIBLE == "round":
        out["car_visible"] = (np.interp(x_new, x_old, cv.astype(float)) >= 0.5).astype(cv.dtype)
    else:
        raise ValueError(f"unsupported INTERP_CAR_VISIBLE: {INTERP_CAR_VISIBLE}")

    return pd.DataFrame(out, columns=TRUTH_COLUMNS)


def preprocess_x_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_pred_m"] = _pred_marker_bool(df["pred_marker"]).to_numpy()
    df = df.sort_values(["frame.time_relative", "frame.number"], kind="mergesort").reset_index(drop=True)
    pred = df["_pred_m"].to_numpy()
    spans = segment_indices(pred)
    rows = [aggregate_frame_packets(df, a, b, k) for k, (a, b) in enumerate(spans)]
    return pd.DataFrame(rows)


def _process_one_file(
    x_path: Path,
    truth_path: Path,
    preprocessed_dir: Path,
    truth_interp_dir: Path,
) -> str:
    df_x = pd.read_csv(x_path)
    df_out = preprocess_x_dataframe(df_x)
    out_path = preprocessed_dir / x_path.name
    df_out.to_csv(out_path, index=False)

    n_processed = len(df_out)
    n_pred_m = int(_pred_marker_bool(df_x["pred_marker"]).sum())
    df_truth = pd.read_csv(truth_path)
    n_truth = len(df_truth)
    diff = n_processed - n_truth
    df_truth_interp = interpolate_truth_rows(df_truth, n_processed)
    truth_interp_path = truth_interp_dir / truth_path.name
    df_truth_interp.to_csv(truth_interp_path, index=False)
    pct_err = 100.0 * abs(diff) / n_truth
    return (
        f"{out_path.name} diff={diff:+d} err={pct_err:.2f}% pred_m={n_pred_m} "
        f"truth={n_truth} -> {truth_interp_path.name}"
    )


def main() -> None:
    XY_INFER_DATA_PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    XY_VIDEO_FRAME_TRUTH_INTERPOLATED_DIR.mkdir(parents=True, exist_ok=True)

    x_files = sorted(XY_INFER_DATA_DIR.glob("*_rtp_marker_pred.csv"))
    if not x_files:
        x_files = sorted(XY_INFER_DATA_DIR.glob("*.csv"))

    truth_files = sorted(XY_VIDEO_FRAME_TRUTH_DIR.glob("*.csv"))
    truth_by_base = {_base_stem(p.stem): p for p in truth_files}

    jobs = [
        (x_path, truth_by_base[_base_stem(x_path.stem)])
        for x_path in x_files
        if x_path.parent.resolve() != XY_INFER_DATA_PREPROCESSED_DIR.resolve()
    ]
    max_workers = int(os.environ.get("PREPROCESS_WORKERS", os.cpu_count() or 1))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_one_file,
                x_path,
                truth_path,
                XY_INFER_DATA_PREPROCESSED_DIR,
                XY_VIDEO_FRAME_TRUTH_INTERPOLATED_DIR,
            ): x_path
            for x_path, truth_path in jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocess"):
            tqdm.write(fut.result())

    print(f"done {len(jobs)} -> {XY_INFER_DATA_PREPROCESSED_DIR}")


if __name__ == "__main__":
    main()
