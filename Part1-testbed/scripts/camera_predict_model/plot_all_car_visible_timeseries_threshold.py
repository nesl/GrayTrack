from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = Path(
    "/home/ubuntu/GrayAssets_CityScale/scripts/camera_predict_model/outputs_thresold_experiment_plots_3"
)

# Chart 1 (pred): car_visible inference output — per-frame X columns plus car_visible_prob / car_visible.
CAR_VISIBLE_PRED_DIR = Path(
    "/home/ubuntu/GrayAssets_CityScale/scripts/camera_predict_model/outputs_thresold_experiment_4"
)

# Stage 2 outputs (same as car_visible_lstm_model.py).
CAR_VISIBLE_CARLA_TRUTH_DIR = Path("/media/ubuntu/research/carla_data_aug_infer_truth_no_warmup")
RAW_FRAME_SIZE_DIR = Path("/media/ubuntu/research/carla_data_aug_combined_x_infer_no_warmup")
PREPROCESSED_FRAME_SIZE_DIR = Path("/media/ubuntu/research/carla_data_aug_combined_x_infer_no_warmup")

CAR_VISIBLE_PRED_SUFFIX = "_car_visible_pred.csv"
CAR_VISIBLE_CARLA_TRUTH_SUFFIX = "_truth.csv"

CAR_VISIBLE_TRUTH_COLOR = "#2ca02c"
CAR_VISIBLE_TRUTH_ALPHA = 0.18


def shade_car_visible_regions(
    ax,
    frames,
    car_visible,
    *,
    color=CAR_VISIBLE_TRUTH_COLOR,
    alpha=CAR_VISIBLE_TRUTH_ALPHA,
    label="car visible (truth)",
):
    """Shade x-axis spans where car_visible is true (step-hold: value at frame i until frame i+1)."""
    frames = np.asarray(frames, dtype=float)
    visible = np.asarray(car_visible).astype(bool)
    if len(frames) == 0:
        return False

    frame_step = float(np.median(np.diff(frames))) if len(frames) > 1 else 1.0
    labeled = False
    i = 0
    while i < len(frames):
        if not visible[i]:
            i += 1
            continue
        start = frames[i]
        j = i + 1
        while j < len(frames) and visible[j]:
            j += 1
        end = frames[j] if j < len(frames) else frames[-1] + frame_step
        ax.axvspan(
            start,
            end,
            color=color,
            alpha=alpha,
            label=label if not labeled else None,
            zorder=0,
        )
        labeled = True
        i = j
    return labeled


def plot_one(pred_csv: Path) -> None:
    stem = pred_csv.name[: -len(CAR_VISIBLE_PRED_SUFFIX)]
    car_visible_carla_truth_csv = CAR_VISIBLE_CARLA_TRUTH_DIR / f"{stem}{CAR_VISIBLE_CARLA_TRUTH_SUFFIX}"
    raw_frame_size_csv = RAW_FRAME_SIZE_DIR / f"{stem}.csv"
    preprocessed_frame_size_csv = PREPROCESSED_FRAME_SIZE_DIR / f"{stem}.csv"

    df_carla = pd.read_csv(car_visible_carla_truth_csv, usecols=["car_visible"])
    df_pred = pd.read_csv(pred_csv, usecols=["frame_number_first", "car_visible"])
    df_raw_frame_size = pd.read_csv(
        raw_frame_size_csv, usecols=["frame_number_first", "combined_frame_len_bytes"]
    )
    df_preprocessed_frame_size = pd.read_csv(
        preprocessed_frame_size_csv, usecols=["frame_number_first", "combined_frame_len_bytes"]
    )

    frame = df_pred["frame_number_first"]
    carla_car_visible = df_carla["car_visible"].astype(int)
    pred_car_visible = df_pred["car_visible"].astype(int)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

    ax1.step(frame, carla_car_visible, where="post", label="truth")
    ax1.step(frame, pred_car_visible, where="post", label="pred")
    ax1.set_xlabel("frame_number_first")
    ax1.set_ylabel("car_visible")
    ax1.legend()
    ax1.set_title("car_visible time series (Carla label vs prediction)")
    ax1.set_ylim(-0.08, 1.08)

    shade_car_visible_regions(ax2, frame, carla_car_visible)
    ax2.plot(
        df_raw_frame_size["frame_number_first"],
        df_raw_frame_size["combined_frame_len_bytes"],
        linewidth=0.8,
        zorder=1,
    )
    ax2.set_xlabel("frame_number_first")
    ax2.set_ylabel("combined_frame_len_bytes")
    ax2.set_title("raw frame size time series (network input)")
    if shade_car_visible_regions(ax2, frame, carla_car_visible):
        ax2.legend(loc="upper right")

    shade_car_visible_regions(ax3, frame, carla_car_visible)
    ax3.plot(
        df_preprocessed_frame_size["frame_number_first"],
        df_preprocessed_frame_size["combined_frame_len_bytes"],
        linewidth=0.8,
        zorder=1,
    )
    ax3.set_xlabel("frame_number_first")
    ax3.set_ylabel("combined_frame_len_bytes")
    ax3.set_title("preprocessed frame size time series (I-frame spikes removed)")
    if shade_car_visible_regions(ax3, frame, carla_car_visible):
        ax3.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_csvs = sorted(CAR_VISIBLE_PRED_DIR.glob(f"*{CAR_VISIBLE_PRED_SUFFIX}"))
    for pred_csv in tqdm(pred_csvs):
        plot_one(pred_csv)


if __name__ == "__main__":
    main()
