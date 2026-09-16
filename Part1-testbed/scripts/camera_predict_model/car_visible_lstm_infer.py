"""Run car-visible LSTM inference on X CSV(s) in a directory.

Writes per-frame car_visible_prob and speed predictions. Thresholding,
hysteresis, and event evaluation are handled by eval_car_visible_events.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from car_visible_lstm_model import (
    MODEL_CHECKPOINT,
    NORM_CHECKPOINT,
    SPEED_COLS,
    X_DATA_DIR,
    TrainConfig,
    load_car_visible_lstm,
    load_norm_checkpoint,
    load_x_sequence,
    predict_car_visible_and_speed,
)

def infer_one(
    model: torch.nn.Module,
    x_csv: Path,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_speed_mean: torch.Tensor,
    target_speed_std: torch.Tensor,
    device: torch.device,
    out_path: Path,
    min_frames: int,
) -> None:
    x, mask = load_x_sequence(x_csv)

    n_frames = x.shape[1]
    if n_frames < min_frames:
        print(f"Skipping {x_csv.name}: {n_frames} frames (< {min_frames})")
        return

    _, prob, speed_pred = predict_car_visible_and_speed(
        model,
        x,
        mask,
        mean,
        std,
        target_speed_mean,
        target_speed_std,
        device,
    )

    df = pd.read_csv(x_csv).iloc[:n_frames].copy()
    df["car_visible_prob"] = prob.squeeze(0).numpy()
    speed_np = speed_pred.squeeze(0).numpy()
    for idx, col in enumerate(SPEED_COLS):
        df[f"{col}_pred"] = speed_np[:, idx]
    df.to_csv(out_path, index=False)

    print(f"{x_csv.name} -> {out_path} ({n_frames} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-checkpoint", type=Path, default=MODEL_CHECKPOINT)
    parser.add_argument("--norm-checkpoint", type=Path, default=NORM_CHECKPOINT)
    parser.add_argument("--x-dir", type=Path, default=X_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/media/ubuntu/research/carla_data_aug_car_visible_pred_7"),
    )
    args = parser.parse_args()

    config = TrainConfig()
    norm_state = load_norm_checkpoint(args.norm_checkpoint)
    mean = norm_state["mean"].float()
    std = norm_state["std"].float()
    target_speed_mean = norm_state["target_speed_mean"].float()
    target_speed_std = norm_state["target_speed_std"].float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_car_visible_lstm(args.model_checkpoint, config, device)

    args.output.mkdir(parents=True, exist_ok=True)
    for x_path in sorted(args.x_dir.glob("*.csv")):
        out_path = args.output / f"{x_path.stem}_car_visible_pred.csv"
        infer_one(
            model,
            x_path,
            mean,
            std,
            target_speed_mean,
            target_speed_std,
            device,
            out_path,
            config.min_frames,
        )


if __name__ == "__main__":
    main()
