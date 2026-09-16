"""Remove warmup rows from paired X/Y CSVs and interpolate I-frame spikes in X."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Inputs: stage 1 outputs.
X_DATA_DIR = Path("/media/ubuntu/research/carla_data_aug_combined_x_infer_preprocessed")
Y_DATA_DIR = Path("/media/ubuntu/research/carla_data_aug_infer_truth_interpolated")

# Outputs: warmup removed (+ speed zeroed when car not visible).
OUT_X_DATA_DIR = Path("/media/ubuntu/research/carla_data_aug_combined_x_infer_no_warmup")
OUT_Y_DATA_DIR = Path("/media/ubuntu/research/carla_data_aug_infer_truth_no_warmup")

WARMUP_FRAMES = 256
ENABLE_IFRAME_INTERPOLATION = False
IFRAME_MAD_K = 8.0
IFRAME_QUANTILE = 0.995
IFRAME_TAIL_FRAMES = 0

REQUIRED_X_COLUMNS = ("combined_frame_len_bytes", "video_frame_index")
SPEED_VELOCITY_COLS = ("vx", "vy", "vz", "speed")


def _zero_speed_when_not_visible(y_df: pd.DataFrame) -> int:
    """Set speed/velocity to 0 on frames where the car is not visible."""
    # pandas reads True/False CSV values as bool
    not_visible = ~y_df["car_visible"]
    y_df.loc[not_visible, list(SPEED_VELOCITY_COLS)] = 0
    return int(not_visible.sum())


def y_path_for_x(x_path: Path, rel_path: Path) -> Path:
    # e.g. foo_rtp_marker_pred.csv -> foo_rtp_marker_pred_truth.csv
    return Y_DATA_DIR / rel_path.parent / f"{x_path.stem}_truth.csv"


def out_y_path_for_x(x_path: Path, rel_path: Path) -> Path:
    return OUT_Y_DATA_DIR / rel_path.parent / f"{x_path.stem}_truth.csv"


def _expand_iframe_mask(mask: np.ndarray, tail_frames: int) -> np.ndarray:
    if tail_frames <= 0 or not mask.any():
        return mask
    expanded = mask.copy()
    spike_indices = np.flatnonzero(mask)
    for idx in spike_indices:
        end = min(idx + tail_frames + 1, len(expanded))
        expanded[idx + 1 : end] = True
    return expanded


def _detect_iframe_mask(x_df: pd.DataFrame) -> np.ndarray:
    s = np.log1p(x_df["combined_frame_len_bytes"].astype(float))
    med = s.median()
    mad = np.median(np.abs(s - med)) + 1e-9
    q_hi = x_df["combined_frame_len_bytes"].quantile(IFRAME_QUANTILE)
    iframe_mask = (s > med + IFRAME_MAD_K * mad) | (
        x_df["combined_frame_len_bytes"] > q_hi
    )
    return _expand_iframe_mask(iframe_mask.to_numpy(), IFRAME_TAIL_FRAMES)


def _interpolate_iframe_spikes(x_df: pd.DataFrame, iframe_mask: np.ndarray) -> int:
    n_iframes = int(iframe_mask.sum())
    if n_iframes == 0:
        return 0
    x_df["combined_frame_len_bytes"] = (
        x_df["combined_frame_len_bytes"]
        .astype(float)
        .mask(iframe_mask, np.nan)
        .interpolate(method="linear", limit_direction="both")
    )
    x_df["combined_frame_len_bytes"] = (
        x_df["combined_frame_len_bytes"]
        .clip(lower=0)
        .round()
        .astype("int64")
    )
    return n_iframes


def process_pair(x_path: Path) -> tuple[bool, str, dict[str, int] | None]:
    rel_path = x_path.relative_to(X_DATA_DIR)
    y_path = y_path_for_x(x_path, rel_path)
    out_x_path = OUT_X_DATA_DIR / rel_path
    out_y_path = out_y_path_for_x(x_path, rel_path)

    if not y_path.is_file():
        raise FileNotFoundError(f"missing Y pair for {rel_path}: {y_path}")

    try:
        x_df = pd.read_csv(x_path)
        y_df = pd.read_csv(y_path)
    except pd.errors.EmptyDataError:
        return False, f"warning: empty file for {rel_path}", None
    except Exception as exc:
        return False, f"warning: failed to read {rel_path}: {exc}", None

    if x_df.empty or y_df.empty:
        return False, f"warning: empty dataframe for {rel_path}", None

    missing_cols = [c for c in REQUIRED_X_COLUMNS if c not in x_df.columns]
    if missing_cols:
        return (
            False,
            f"warning: missing X columns {missing_cols} in {rel_path}",
            None,
        )

    orig_x_rows = len(x_df)
    orig_y_rows = len(y_df)

    if orig_x_rows != orig_y_rows:
        return (
            False,
            f"warning: row count mismatch for {rel_path} "
            f"(X={orig_x_rows}, Y={orig_y_rows})",
            None,
        )

    x_columns = list(x_df.columns)
    y_columns = list(y_df.columns)

    warmup_removed = min(WARMUP_FRAMES, orig_x_rows, orig_y_rows)
    x_df = x_df.iloc[WARMUP_FRAMES:].copy()
    y_df = y_df.iloc[WARMUP_FRAMES:].copy()

    if x_df.empty or y_df.empty:
        return (
            False,
            f"warning: no rows left after warmup removal for {rel_path}",
            None,
        )

    n_iframes = 0
    if ENABLE_IFRAME_INTERPOLATION:
        iframe_mask = _detect_iframe_mask(x_df)
        n_iframes = _interpolate_iframe_spikes(x_df, iframe_mask)

    n_speed_zeroed = _zero_speed_when_not_visible(y_df)

    x_df = x_df[x_columns]
    y_df = y_df[y_columns]

    out_x_path.parent.mkdir(parents=True, exist_ok=True)
    out_y_path.parent.mkdir(parents=True, exist_ok=True)
    x_df.to_csv(out_x_path, index=False)
    y_df.to_csv(out_y_path, index=False)

    rows_after_warmup = len(x_df)
    summary = (
        f"{rel_path}\n"
        f"  original X rows: {orig_x_rows}\n"
        f"  original Y rows: {orig_y_rows}\n"
        f"  rows after warmup removal: {rows_after_warmup}\n"
        f"  warmup rows removed: {warmup_removed}\n"
        f"  I-frame rows interpolated in X: {n_iframes}\n"
        f"  speed/velocity zeroed (not visible): {n_speed_zeroed}\n"
        f"  X output: {out_x_path}\n"
        f"  Y output: {out_y_path}"
    )
    stats = {
        "warmup_removed_x": warmup_removed,
        "warmup_removed_y": warmup_removed,
        "iframes_interpolated": n_iframes,
        "speed_zeroed": n_speed_zeroed,
    }
    return True, summary, stats


def main() -> None:
    OUT_X_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_Y_DATA_DIR.mkdir(parents=True, exist_ok=True)

    x_files = sorted(X_DATA_DIR.rglob("*.csv"))
    total_found = len(x_files)
    total_processed = 0
    total_skipped = 0
    total_warmup_x = 0
    total_warmup_y = 0
    total_iframes = 0
    total_speed_zeroed = 0

    for x_path in x_files:
        ok, message, stats = process_pair(x_path)
        print(message)
        if ok and stats is not None:
            total_processed += 1
            total_warmup_x += stats["warmup_removed_x"]
            total_warmup_y += stats["warmup_removed_y"]
            total_iframes += stats["iframes_interpolated"]
            total_speed_zeroed += stats["speed_zeroed"]
        else:
            total_skipped += 1

    print()
    print(f"total pairs found: {total_found}")
    print(f"total pairs processed: {total_processed}")
    print(f"total pairs skipped: {total_skipped}")
    print(f"total warmup rows removed from X: {total_warmup_x}")
    print(f"total warmup rows removed from Y: {total_warmup_y}")
    print(f"total I-frame rows interpolated in X: {total_iframes}")
    print(f"total frames with speed/velocity zeroed (not visible): {total_speed_zeroed}")


if __name__ == "__main__":
    main()
