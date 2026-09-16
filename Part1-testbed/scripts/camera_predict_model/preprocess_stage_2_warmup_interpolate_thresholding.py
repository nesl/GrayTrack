"""Remove warmup rows from paired X/Y CSVs and interpolate I-frame spikes in X."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

X_DATA_DIR = Path("/media/ubuntu/research/xy_combined_data_x_truth_2_preprocessed")
XY_VIDEO_FRAME_TRUTH_DIR = Path("/media/ubuntu/research/xy_infer_truth_data_2_interpolated")

OUT_X_DATA_DIR = Path("/media/ubuntu/research/xy_combined_data_x_truth_5_no_warmup_no_iframes")
OUT_Y_TRUTH_DIR = Path("/media/ubuntu/research/xy_infer_truth_data_5_no_warmup_no_iframes")

WARMUP_FRAMES = 256
IFRAME_MAD_K = 8.0
IFRAME_QUANTILE = 0.995
IFRAME_HEAD_FRAMES = 1
IFRAME_TAIL_FRAMES = 2
POST_IFRAME_SPIKE_WINDOW = 9
POST_IFRAME_SPIKE_MAD_K = 6.0
POST_IFRAME_SPIKE_MAX_RUN = 3

REQUIRED_X_COLUMNS = ("combined_frame_len_bytes", "video_frame_index")
Y_SUFFIX = "_rtp_marker_pred_truth.csv"


def y_path_for_x(x_path: Path, rel_path: Path) -> Path:
    return XY_VIDEO_FRAME_TRUTH_DIR / rel_path.parent / f"{x_path.stem}{Y_SUFFIX}"


def out_y_path_for_x(x_path: Path, rel_path: Path) -> Path:
    return OUT_Y_TRUTH_DIR / rel_path.parent / f"{x_path.stem}{Y_SUFFIX}"


def _expand_iframe_mask(mask: np.ndarray, tail_frames: int) -> np.ndarray:
    if tail_frames <= 0 or not mask.any():
        return mask
    expanded = mask.copy()
    spike_indices = np.flatnonzero(mask)
    for idx in spike_indices:
        end = min(idx + tail_frames + 1, len(expanded))
        expanded[idx + 1 : end] = True
    return expanded


def _base_iframe_mask(x_df: pd.DataFrame) -> np.ndarray:
    s = np.log1p(x_df["combined_frame_len_bytes"].astype(float))
    med = s.median()
    mad = np.median(np.abs(s - med)) + 1e-9
    q_hi = x_df["combined_frame_len_bytes"].quantile(IFRAME_QUANTILE)
    return ((s > med + IFRAME_MAD_K * mad) | (
        x_df["combined_frame_len_bytes"] > q_hi
    )).to_numpy()


def _detect_iframe_mask(x_df: pd.DataFrame) -> np.ndarray:
    return _expand_iframe_mask(_base_iframe_mask(x_df), IFRAME_TAIL_FRAMES)


def _head_omit_mask(iframe_mask: np.ndarray, head_frames: int) -> np.ndarray:
    if head_frames <= 0 or not iframe_mask.any():
        return np.zeros_like(iframe_mask, dtype=bool)
    omit_mask = np.zeros_like(iframe_mask, dtype=bool)
    iframe_indices = np.flatnonzero(iframe_mask)
    for idx in iframe_indices:
        start = max(0, idx - head_frames)
        omit_mask[start:idx] = True
    return omit_mask


def _interpolate_iframe_spikes(x_df: pd.DataFrame, iframe_mask: np.ndarray) -> int:
    n_iframes = int(iframe_mask.sum())
    if n_iframes == 0:
        return 0
    original = pd.to_numeric(x_df["combined_frame_len_bytes"], errors="coerce").astype(float)
    interpolated = (
        original.mask(iframe_mask, np.nan).interpolate(
            method="linear", limit_direction="both"
        )
    )
    # If interpolation still has gaps (for example, every row was masked), fall
    # back to the original values where possible and then to zero.
    interpolated = interpolated.where(np.isfinite(interpolated), original)
    interpolated = interpolated.fillna(0.0)
    x_df["combined_frame_len_bytes"] = (
        interpolated
        .clip(lower=0)
        .round()
        .astype("int64")
    )
    return n_iframes


def _keep_narrow_runs(mask: np.ndarray, max_run: int) -> np.ndarray:
    if max_run <= 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    padded = np.concatenate(([False], mask, [False])).astype(int)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends):
        if end - start <= max_run:
            out[start:end] = True
    return out


def _detect_post_iframe_spike_mask(x_df: pd.DataFrame) -> np.ndarray:
    signal = pd.Series(
        np.log1p(x_df["combined_frame_len_bytes"].astype(float)),
        index=x_df.index,
    )
    rolling_median = signal.rolling(
        window=POST_IFRAME_SPIKE_WINDOW, center=True, min_periods=1
    ).median()
    abs_dev = (signal - rolling_median).abs()
    rolling_mad = abs_dev.rolling(
        window=POST_IFRAME_SPIKE_WINDOW, center=True, min_periods=1
    ).median()
    robust_sigma = 1.4826 * rolling_mad
    spike_raw = (signal > rolling_median + POST_IFRAME_SPIKE_MAD_K * robust_sigma).to_numpy()
    return _keep_narrow_runs(spike_raw, POST_IFRAME_SPIKE_MAX_RUN)


def _interpolate_post_iframe_spikes(
    x_df: pd.DataFrame, post_iframe_spike_mask: np.ndarray
) -> int:
    n_spikes = int(post_iframe_spike_mask.sum())
    if n_spikes == 0:
        return 0
    original = pd.to_numeric(x_df["combined_frame_len_bytes"], errors="coerce").astype(float)
    interpolated = (
        original.mask(post_iframe_spike_mask, np.nan).interpolate(
            method="linear", limit_direction="both"
        )
    )
    interpolated = interpolated.where(np.isfinite(interpolated), original)
    interpolated = interpolated.fillna(0.0)
    x_df["combined_frame_len_bytes"] = (
        interpolated
        .clip(lower=0)
        .round()
        .astype("int64")
    )
    return n_spikes


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

    iframe_head_mask = _head_omit_mask(_base_iframe_mask(x_df), IFRAME_HEAD_FRAMES)
    iframe_head_removed = int(iframe_head_mask.sum())
    if iframe_head_removed:
        x_df = x_df.loc[~iframe_head_mask].copy()
        y_df = y_df.loc[~iframe_head_mask].copy()

    if x_df.empty or y_df.empty:
        return (
            False,
            f"warning: no rows left after I-frame head removal for {rel_path}",
            None,
        )

    iframe_mask = _detect_iframe_mask(x_df)
    n_iframes = _interpolate_iframe_spikes(x_df, iframe_mask)

    post_iframe_spike_mask = _detect_post_iframe_spike_mask(x_df)
    n_post_iframe_spikes = _interpolate_post_iframe_spikes(x_df, post_iframe_spike_mask)

    x_df = x_df[x_columns]
    y_df = y_df[y_columns]

    out_x_path.parent.mkdir(parents=True, exist_ok=True)
    out_y_path.parent.mkdir(parents=True, exist_ok=True)
    x_df.to_csv(out_x_path, index=False)
    y_df.to_csv(out_y_path, index=False)

    rows_after_processing = len(x_df)
    summary = (
        f"{rel_path}\n"
        f"  original X rows: {orig_x_rows}\n"
        f"  original Y rows: {orig_y_rows}\n"
        f"  rows after processing: {rows_after_processing}\n"
        f"  warmup rows removed: {warmup_removed}\n"
        f"  I-frame head rows removed: {iframe_head_removed}\n"
        f"  I-frame rows interpolated in X: {n_iframes}\n"
        f"  narrow post-I-frame/codec spikes interpolated in X: {n_post_iframe_spikes}\n"
        f"  X output: {out_x_path}\n"
        f"  Y output: {out_y_path}"
    )
    stats = {
        "warmup_removed_x": warmup_removed,
        "warmup_removed_y": warmup_removed,
        "iframe_head_removed_x": iframe_head_removed,
        "iframe_head_removed_y": iframe_head_removed,
        "iframes_interpolated": n_iframes,
        "post_iframe_spikes_interpolated": n_post_iframe_spikes,
    }
    return True, summary, stats


def main() -> None:
    OUT_X_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_Y_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    x_files = sorted(X_DATA_DIR.rglob("*.csv"))
    total_found = len(x_files)
    total_processed = 0
    total_skipped = 0
    total_warmup_x = 0
    total_warmup_y = 0
    total_iframe_head_x = 0
    total_iframe_head_y = 0
    total_iframes = 0
    total_post_iframe_spikes = 0

    for x_path in x_files:
        ok, message, stats = process_pair(x_path)
        print(message)
        if ok and stats is not None:
            total_processed += 1
            total_warmup_x += stats["warmup_removed_x"]
            total_warmup_y += stats["warmup_removed_y"]
            total_iframe_head_x += stats["iframe_head_removed_x"]
            total_iframe_head_y += stats["iframe_head_removed_y"]
            total_iframes += stats["iframes_interpolated"]
            total_post_iframe_spikes += stats["post_iframe_spikes_interpolated"]
        else:
            total_skipped += 1

    print()
    print(f"total pairs found: {total_found}")
    print(f"total pairs processed: {total_processed}")
    print(f"total pairs skipped: {total_skipped}")
    print(f"total warmup rows removed from X: {total_warmup_x}")
    print(f"total warmup rows removed from Y: {total_warmup_y}")
    print(f"total I-frame head rows removed from X: {total_iframe_head_x}")
    print(f"total I-frame head rows removed from Y: {total_iframe_head_y}")
    print(f"total I-frame rows interpolated in X: {total_iframes}")
    print(
        "total narrow post-I-frame/codec spikes interpolated in X: "
        f"{total_post_iframe_spikes}"
    )


if __name__ == "__main__":
    main()
