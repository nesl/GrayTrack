from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from infer_rtp_marker_from_csv import (
    apply_normalization,
    load_csv_for_inference,
    load_model,
    load_or_compute_training_normalization_stats,
    predict_in_windows,
)

# FIXME: the first row of the CSV is missing - off by one error

# Hardcoded paths and inference settings (match predict_rtp_marker_from_csv.py)
DATA_ROOT = Path("/media/ubuntu/research/carla_data_aug_combined")
OUTPUT_DIR = Path("/media/ubuntu/research/carla_data_aug_combined_x_infer")

CHECKPOINT_PATH = Path(
    "/home/ubuntu/GrayAssets_CityScale/scripts/mininet/frame_boundary_model/best_packetsegformer_carla_data_aug.pkl"
)
STATS_CACHE_PATH = Path(
    "/home/ubuntu/GrayAssets_CityScale/scripts/mininet/frame_boundary_model/train_norm_stats_carla_data_aug.pt"
)

PREDICTION_THRESHOLD = 0.3
WINDOW_SIZE = 250
WINDOW_STRIDE = 25

METADATA_COLUMNS = [
    "frame.number",
    "frame.time_epoch",
    "frame.time_relative",
    "frame.len",
    "rtp.seq",
    "rtp.timestamp",
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_or_compute_training_normalization_stats(DATA_ROOT, STATS_CACHE_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(p for p in DATA_ROOT.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    model = None
    model_max_len = 0

    for csv_path in tqdm(csv_paths, desc="Inference"):
        df_eval, x, mask = load_csv_for_inference(csv_path)
        seq_len = x.shape[1]
        if model is None or seq_len > model_max_len:
            model_max_len = max(model_max_len, seq_len)
            model = load_model(CHECKPOINT_PATH, device, model_max_len)

        x = apply_normalization(x, mask, mean, std).to(device)
        mask = mask.to(device)
        prob = predict_in_windows(model, x, mask, device, WINDOW_SIZE, WINDOW_STRIDE)
        pred = prob > PREDICTION_THRESHOLD

        out = df_eval[METADATA_COLUMNS].copy()
        out["rtp.marker.original"] = df_eval["rtp.marker.original"].astype(int)
        out["prob"] = prob
        out["pred_marker"] = pred.astype(int)
        out_path = OUTPUT_DIR / f"{csv_path.stem}_rtp_marker_pred.csv"
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
