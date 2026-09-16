#!/usr/bin/env python3
"""Frame-size error: DNN PacketSegformer vs fixed-period (1/FPS) baseline.

Aggregates packet sizes between end-of-frame markers (true / pred), pairs each
true frame to the nearest predicted frame by end-time, and reports:

  non-normalized: MAE / RMSE in bytes
  normalized:     MAE / mean(true), RMSE / mean(true), MAPE

In-memory full-dataset infer — does not write prediction CSVs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fixed_period_packet_segmentation import predict_fixed_period_markers
from infer_rtp_marker_from_csv import (
    apply_normalization,
    load_csv_for_inference,
    load_model,
    load_or_compute_training_normalization_stats,
    predict_in_windows,
)

DATA_ROOT = Path("/media/ubuntu/research/xy_combined_data_2")
CHECKPOINT_PATH = Path(__file__).resolve().parent / "best_packetsegformer_v2.pkl"
STATS_CACHE_PATH = Path(__file__).resolve().parent / "train_norm_stats_xy_combined_data_2.pt"
PREDICTION_THRESHOLD = 0.3
WINDOW_SIZE = 250
WINDOW_STRIDE = 25
FPS = 20.0


def frame_sizes_from_markers(
    sizes: np.ndarray, times: np.ndarray, markers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum packet sizes in each [prev_end+1 .. end] segment (marker = frame end)."""
    ends = np.flatnonzero(markers.astype(bool))
    out_sz, out_t = [], []
    start = 0
    for e in ends:
        if e >= start:
            out_sz.append(float(sizes[start : e + 1].sum()))
            out_t.append(float(times[e]))
        start = e + 1
    return np.asarray(out_t, dtype=np.float64), np.asarray(out_sz, dtype=np.float64)


def paired_errors(
    true_t: np.ndarray, true_sz: np.ndarray, pred_t: np.ndarray, pred_sz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each true frame, error vs nearest-by-end-time predicted frame."""
    if true_sz.size == 0 or pred_sz.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    j = np.searchsorted(pred_t, true_t)
    j0 = np.clip(j - 1, 0, len(pred_t) - 1)
    j1 = np.clip(j, 0, len(pred_t) - 1)
    pick = np.where(np.abs(pred_t[j0] - true_t) <= np.abs(pred_t[j1] - true_t), j0, j1)
    return pred_sz[pick] - true_sz, true_sz


def summarize(err: np.ndarray, true_ref: np.ndarray) -> dict[str, float]:
    abs_e = np.abs(err)
    mae = float(abs_e.mean())
    rmse = float(np.sqrt(np.mean(err * err)))
    mean_true = float(np.mean(true_ref)) if true_ref.size else 1.0
    mape = float(np.mean(abs_e / np.maximum(true_ref, 1.0)))
    return {
        "n": float(err.size),
        "mae_bytes": mae,
        "rmse_bytes": rmse,
        "nmae": mae / max(mean_true, 1.0),
        "nrmse": rmse / max(mean_true, 1.0),
        "mape": mape,
        "mean_true_bytes": mean_true,
    }


def print_block(name: str, s: dict[str, float]) -> None:
    print(
        f"{name}: n_frames={s['n']:.0f} mean_true={s['mean_true_bytes']:.1f} B | "
        f"MAE={s['mae_bytes']:.1f} B  RMSE={s['rmse_bytes']:.1f} B | "
        f"NMAE={s['nmae']:.4f}  NRMSE={s['nrmse']:.4f}  MAPE={s['mape']:.4f}"
    )


def main() -> None:
    assert CHECKPOINT_PATH.is_file(), CHECKPOINT_PATH
    assert DATA_ROOT.is_dir(), DATA_ROOT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_or_compute_training_normalization_stats(DATA_ROOT, STATS_CACHE_PATH)
    csv_paths = sorted(p for p in DATA_ROOT.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    print(f"device={device} files={len(csv_paths)} thr={PREDICTION_THRESHOLD} fps={FPS:g}")
    print(f"checkpoint={CHECKPOINT_PATH.name}")

    model = None
    model_max_len = 0
    dnn_err: list[np.ndarray] = []
    dnn_true: list[np.ndarray] = []
    fps_err: list[np.ndarray] = []
    fps_true: list[np.ndarray] = []

    for csv_path in tqdm(csv_paths, desc="frame-size"):
        df_eval, x, mask = load_csv_for_inference(csv_path)
        sizes = df_eval["frame.len"].to_numpy(dtype=np.float64)
        times = df_eval["frame.time_relative"].to_numpy(dtype=np.float64)
        truth = df_eval["rtp.marker.original"].to_numpy(dtype=bool)
        true_t, true_sz = frame_sizes_from_markers(sizes, times, truth)

        # --- DNN ---
        seq_len = x.shape[1]
        if model is None or seq_len > model_max_len:
            model_max_len = max(model_max_len, seq_len)
            model = load_model(CHECKPOINT_PATH, device, model_max_len)
        x_n = apply_normalization(x, mask, mean, std).to(device)
        prob = predict_in_windows(model, x_n, mask.to(device), device, WINDOW_SIZE, WINDOW_STRIDE)
        dnn_t, dnn_sz = frame_sizes_from_markers(sizes, times, prob > PREDICTION_THRESHOLD)
        e, t_ref = paired_errors(true_t, true_sz, dnn_t, dnn_sz)
        if e.size:
            dnn_err.append(e)
            dnn_true.append(t_ref)

        # --- 1/FPS baseline (same packet rows as DNN eval) ---
        fps_pred = predict_fixed_period_markers(df_eval["frame.time_relative"], FPS).to_numpy()
        fps_t, fps_sz = frame_sizes_from_markers(sizes, times, fps_pred)
        e, t_ref = paired_errors(true_t, true_sz, fps_t, fps_sz)
        if e.size:
            fps_err.append(e)
            fps_true.append(t_ref)

    dnn_e = np.concatenate(dnn_err)
    dnn_t = np.concatenate(dnn_true)
    fps_e = np.concatenate(fps_err)
    fps_t = np.concatenate(fps_true)

    s_dnn = summarize(dnn_e, dnn_t)
    s_fps = summarize(fps_e, fps_t)
    print_block("DNN", s_dnn)
    print_block("1/FPS", s_fps)
    print(
        f"Δ (1/FPS − DNN): MAE={s_fps['mae_bytes'] - s_dnn['mae_bytes']:.1f} B  "
        f"NMAE={s_fps['nmae'] - s_dnn['nmae']:.4f}  MAPE={s_fps['mape'] - s_dnn['mape']:.4f}"
    )


if __name__ == "__main__":
    main()
