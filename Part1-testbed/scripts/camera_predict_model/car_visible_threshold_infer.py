"""Threshold car-visible detection on stage-2 preprocessed X (no ML)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from car_visible_lstm_model import metrics_from_binary_pred, pr_auc_score

X_DATA_DIR = Path("/media/ubuntu/research/carla_data_aug_combined_x_infer_no_warmup")
Y_DATA_DIR = Path("/media/ubuntu/research/carla_data_aug_infer_truth_no_warmup")
Y_SUFFIX = "_truth.csv"

MIN_FRAMES = 1024

# Fixed z-score threshold; None uses median + MAD_K * MAD per sequence.
THRESHOLD: float | None = None
MAD_K = 6.0

LOCAL_BASELINE_WINDOW = 301
LOCAL_BASELINE_QUANTILE = 0.20
RESIDUAL_MEDIAN_WINDOW = 7
EVIDENCE_WINDOW = 21
MIN_ACTIVE_FRAC = 0.45
MIN_VISIBLE_RUN = 15
MAX_GAP_FILL = 8


def rolling_quantile(signal: np.ndarray, window: int, quantile: float) -> np.ndarray:
    s = pd.Series(signal)
    return (
        s.rolling(window=window, center=True, min_periods=1)
        .quantile(quantile)
        .to_numpy()
    )


def rolling_median(signal: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(signal)
    return (
        s.rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )


def robust_zscore(signal: np.ndarray) -> np.ndarray:
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med)))
    scale = 1.4826 * mad + 1e-9
    return (signal - med) / scale


def frame_size_residual_signal(
    df: pd.DataFrame,
    local_baseline_window: int,
    local_baseline_quantile: float,
    residual_median_window: int,
) -> np.ndarray:
    size = pd.to_numeric(df["combined_frame_len_bytes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    log_size = np.log1p(np.clip(size, 0.0, None))
    baseline = rolling_quantile(log_size, local_baseline_window, local_baseline_quantile)
    residual = log_size - baseline
    residual = rolling_median(residual, residual_median_window)
    return robust_zscore(residual)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mean = float(signal.mean())
    std = float(signal.std()) + 1e-9
    return (signal - mean) / std


def resolve_threshold(signal: np.ndarray, mad_k: float, fixed: float | None) -> float:
    if fixed is not None:
        return float(fixed)
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med)))
    return med + mad_k * (1.4826 * mad + 1e-9)


def rolling_active_fraction(active: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(active.astype(float))
    return (
        s.rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0 or len(mask) == 0:
        return mask

    out = mask.copy()
    n = len(out)
    i = 0

    while i < n:
        if out[i]:
            i += 1
            continue

        start = i
        while i < n and not out[i]:
            i += 1
        end = i

        gap_len = end - start
        left_true = start > 0 and out[start - 1]
        right_true = end < n and out[end]

        if left_true and right_true and gap_len <= max_gap:
            out[start:end] = True

    return out


def remove_short_true_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    if min_run <= 1 or len(mask) == 0:
        return mask

    out = mask.copy()
    n = len(out)
    i = 0

    while i < n:
        if not out[i]:
            i += 1
            continue

        start = i
        while i < n and out[i]:
            i += 1
        end = i

        run_len = end - start
        if run_len < min_run:
            out[start:end] = False

    return out


def sustained_threshold_prediction(
    signal: np.ndarray,
    threshold: float,
    evidence_window: int,
    min_active_frac: float,
    min_visible_run: int,
    max_gap_fill: int,
) -> tuple[np.ndarray, np.ndarray]:
    point_active = signal >= threshold
    active_frac = rolling_active_fraction(point_active, evidence_window)
    pred = active_frac >= min_active_frac
    pred = fill_short_false_gaps(pred, max_gap_fill)
    pred = remove_short_true_runs(pred, min_visible_run)
    return pred.astype(bool), active_frac


def infer_one(
    x_csv: Path,
    y_csv: Path | None,
    out_path: Path,
    threshold: float | None,
    mad_k: float,
    local_baseline_window: int,
    local_baseline_quantile: float,
    residual_median_window: int,
    evidence_window: int,
    min_active_frac: float,
    min_visible_run: int,
    max_gap_fill: int,
) -> None:
    x_df = pd.read_csv(x_csv)
    n_frames = len(x_df)
    if n_frames < MIN_FRAMES:
        print(f"Skipping {x_csv.name}: {n_frames} frames (< {MIN_FRAMES})")
        return

    signal = frame_size_residual_signal(
        x_df,
        local_baseline_window,
        local_baseline_quantile,
        residual_median_window,
    )
    thr = resolve_threshold(signal, mad_k, threshold)
    pred, active_frac = sustained_threshold_prediction(
        signal,
        thr,
        evidence_window,
        min_active_frac,
        min_visible_run,
        max_gap_fill,
    )

    # Probability-like score for plotting/PR-AUC only. This is not ML.
    # It combines signal strength and local persistence.
    score = np.maximum(signal - thr, 0.0) * active_frac
    score_norm = normalize_signal(score)
    probs = 1.0 / (1.0 + np.exp(-score_norm))

    df = x_df.copy()
    df["car_visible_prob"] = probs
    df["car_visible"] = pred
    df.to_csv(out_path, index=False)

    print(f"{x_csv.name} -> {out_path} (threshold={thr:.4f})")
    if y_csv is not None:
        target = pd.read_csv(y_csv, usecols=["car_visible"])["car_visible"].astype(bool).to_numpy()
        pred_arr = df["car_visible"].to_numpy()
        m = metrics_from_binary_pred(pred_arr, target)
        pr_auc = pr_auc_score(probs, target)
        print(
            f"  F1={m['f1']:.3f} Acc={m['acc']:.3f} "
            f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} PR-AUC={pr_auc:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--x-csv", type=Path)
    group.add_argument("--x-dir", type=Path, default=X_DATA_DIR)
    parser.add_argument("--y-csv", type=Path, default=None)
    parser.add_argument("--y-dir", type=Path, default=Y_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed z-score threshold; default uses THRESHOLD or median + MAD_K * MAD",
    )
    parser.add_argument("--mad-k", type=float, default=MAD_K)
    parser.add_argument("--local-baseline-window", type=int, default=LOCAL_BASELINE_WINDOW)
    parser.add_argument("--local-baseline-quantile", type=float, default=LOCAL_BASELINE_QUANTILE)
    parser.add_argument("--residual-median-window", type=int, default=RESIDUAL_MEDIAN_WINDOW)
    parser.add_argument("--evidence-window", type=int, default=EVIDENCE_WINDOW)
    parser.add_argument("--min-active-frac", type=float, default=MIN_ACTIVE_FRAC)
    parser.add_argument("--min-visible-run", type=int, default=MIN_VISIBLE_RUN)
    parser.add_argument("--max-gap-fill", type=int, default=MAX_GAP_FILL)
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else THRESHOLD

    infer_kwargs = {
        "threshold": threshold,
        "mad_k": args.mad_k,
        "local_baseline_window": args.local_baseline_window,
        "local_baseline_quantile": args.local_baseline_quantile,
        "residual_median_window": args.residual_median_window,
        "evidence_window": args.evidence_window,
        "min_active_frac": args.min_active_frac,
        "min_visible_run": args.min_visible_run,
        "max_gap_fill": args.max_gap_fill,
    }

    if args.x_csv is not None:
        infer_one(args.x_csv, args.y_csv, args.output, **infer_kwargs)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    for x_path in sorted(args.x_dir.rglob("*.csv")):
        rel_path = x_path.relative_to(args.x_dir)
        y_path = args.y_dir / rel_path.parent / f"{x_path.stem}{Y_SUFFIX}"
        out_path = args.output / rel_path.parent / f"{x_path.stem}_car_visible_pred.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        infer_one(
            x_path,
            y_path if y_path.is_file() else None,
            out_path,
            **infer_kwargs,
        )


if __name__ == "__main__":
    main()
