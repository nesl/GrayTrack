#!/usr/bin/env python3
"""Boundary timing error for PacketSegformer (mean |Δ| packets).

For each true RTP marker, distance (in packet index) to the nearest predicted
marker. Reports mean/median |Δ| and exact-match rate.

In-memory full-dataset infer — does not write prediction CSVs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

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


def nearest_abs_deltas(true_idx: np.ndarray, pred_idx: np.ndarray) -> np.ndarray:
    """Packet-index |Δ| from each true boundary to the nearest predicted boundary."""
    if true_idx.size == 0 or pred_idx.size == 0:
        return np.array([], dtype=np.int64)
    j = np.searchsorted(pred_idx, true_idx)
    j0 = np.clip(j - 1, 0, len(pred_idx) - 1)
    j1 = np.clip(j, 0, len(pred_idx) - 1)
    d0 = np.abs(pred_idx[j0] - true_idx)
    d1 = np.abs(pred_idx[j1] - true_idx)
    return np.minimum(d0, d1)


def main() -> None:
    assert CHECKPOINT_PATH.is_file(), CHECKPOINT_PATH
    assert DATA_ROOT.is_dir(), DATA_ROOT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_or_compute_training_normalization_stats(DATA_ROOT, STATS_CACHE_PATH)
    csv_paths = sorted(p for p in DATA_ROOT.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    print(f"device={device} files={len(csv_paths)} thr={PREDICTION_THRESHOLD}")
    print(f"checkpoint={CHECKPOINT_PATH.name}")

    model = None
    model_max_len = 0
    abs_chunks: list[np.ndarray] = []
    n_true = n_pred = 0

    for csv_path in tqdm(csv_paths, desc="timing"):
        df_eval, x, mask = load_csv_for_inference(csv_path)
        seq_len = x.shape[1]
        if model is None or seq_len > model_max_len:
            model_max_len = max(model_max_len, seq_len)
            model = load_model(CHECKPOINT_PATH, device, model_max_len)

        x_n = apply_normalization(x, mask, mean, std).to(device)
        prob = predict_in_windows(model, x_n, mask.to(device), device, WINDOW_SIZE, WINDOW_STRIDE)
        pred = prob > PREDICTION_THRESHOLD
        truth = df_eval["rtp.marker.original"].to_numpy(dtype=bool)

        true_idx = np.flatnonzero(truth)
        pred_idx = np.flatnonzero(pred)
        n_true += int(true_idx.size)
        n_pred += int(pred_idx.size)
        d = nearest_abs_deltas(true_idx, pred_idx)
        if d.size:
            abs_chunks.append(d)

    if not abs_chunks:
        raise SystemExit("no true/pred boundary pairs found")

    d = np.concatenate(abs_chunks)
    print(
        f"boundary timing |Δ| packets: mean={d.mean():.3f} median={np.median(d):.3f} "
        f"p90={np.percentile(d, 90):.3f} p95={np.percentile(d, 95):.3f} max={d.max()}"
    )
    print(f"exact match (Δ=0): {(d == 0).mean():.4f}  within±1: {(d <= 1).mean():.4f}")
    print(f"n_true={n_true} n_pred={n_pred} n_scored={d.size}")


if __name__ == "__main__":
    main()
