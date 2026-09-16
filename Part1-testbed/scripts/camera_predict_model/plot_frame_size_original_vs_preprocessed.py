"""Plot original vs warmup-skipped / I-frame-interpolated frame sizes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess_stage_2_warmup_interpolate import (
    OUT_X_DATA_DIR,
    WARMUP_FRAMES,
    X_DATA_DIR,
)

CSV_PREFIX = "2026_07_28_16_07_15_126_camera_1_20260818_122811_434597926_rtp_marker_pred"

ORIGINAL_X_CSV = X_DATA_DIR / f"{CSV_PREFIX}.csv"
PREPROCESSED_X_CSV = OUT_X_DATA_DIR / f"{CSV_PREFIX}.csv"


def load_frame_size(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["frame_number_first", "combined_frame_len_bytes"])


def zero_pad_preprocessed(
    original: pd.DataFrame, preprocessed: pd.DataFrame, warmup_frames: int
) -> np.ndarray:
    n_orig = len(original)
    n_pre = len(preprocessed)
    expected_pre = n_orig - warmup_frames
    if n_pre != expected_pre:
        raise ValueError(
            f"preprocessed row count {n_pre} != expected {expected_pre} "
            f"(original {n_orig} - warmup {warmup_frames})"
        )
    padded = np.zeros(n_orig, dtype=np.int64)
    padded[warmup_frames:] = preprocessed["combined_frame_len_bytes"].to_numpy()
    return padded


def main() -> None:
    df_orig = load_frame_size(ORIGINAL_X_CSV)
    df_pre = load_frame_size(PREPROCESSED_X_CSV)

    frame = df_orig["frame_number_first"]
    orig_size = df_orig["combined_frame_len_bytes"]
    padded_pre_size = zero_pad_preprocessed(df_orig, df_pre, WARMUP_FRAMES)

    y_min = min(orig_size.min(), padded_pre_size.min())
    y_max = max(orig_size.max(), padded_pre_size.max())
    ylim = (y_min, y_max)

    fig1, ax1 = plt.subplots(figsize=(16, 4))
    ax1.plot(frame, orig_size, linewidth=0.8)
    ax1.set_xlabel("frame_number_first")
    ax1.set_ylabel("combined_frame_len_bytes")
    ax1.set_title("original frame size")
    ax1.set_ylim(ylim)

    fig2, ax2 = plt.subplots(figsize=(16, 4))
    ax2.plot(frame, padded_pre_size, linewidth=0.8)
    ax2.set_xlabel("frame_number_first")
    ax2.set_ylabel("combined_frame_len_bytes")
    ax2.set_title(f"preprocessed frame size (0-padded warmup: {WARMUP_FRAMES} frames)")
    ax2.set_ylim(ylim)

    plt.show()


if __name__ == "__main__":
    main()
