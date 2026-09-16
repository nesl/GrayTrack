"""Export passage detections (x) alongside the untouched per-frame truth CSVs (y_truth).

`x/<sequence>.csv` holds one row per detected vehicle passage with two columns:
the video frame the passage is centred on and the detector's confidence for it.
`y_truth/<sequence>_rtp_marker_pred_truth.csv` is copied byte-for-byte from the
truth build and stays at one row per video frame, so x and y row counts differ by
design: x is event-level, y_truth is frame-level.

y_truth carries no frame column of its own; its row N is video frame
`first_frame + N`, and `first_frame` is recorded per sequence in the index CSV.

Detection parameters must match the run being exported. Pass `--camera-hyperparams`
with the `camera_hyperparams.csv` written by `eval_car_visible_events.py` to reuse
its per-sequence tuned values, or set `--merge-gap-s` / `--min-event-duration-s`.

Example:
  python export_passage_events.py \
    --camera-hyperparams /media/ubuntu/research/carla_data_aug_event_eval_7/camera_hyperparams.csv
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from car_visible_lstm_model import Y_DATA_DIR, Y_SUFFIX
from eval_car_visible_events import (
    DEFAULT_PRED_DIR,
    PRED_SUFFIX,
    binarize_scores,
    discover_pred_paths,
    filter_pred_paths,
    load_camera_series,
    parse_camera_ids,
    pred_stem,
    resolve_truth_path,
)
from passage_events import PassageEvent, events_from_binary_series

X_SUBDIR = "x"
Y_TRUTH_SUBDIR = "y_truth"
INDEX_FILENAME = "passage_events_index.csv"

FRAME_COLUMN = "frame_number"
CONFIDENCE_COLUMN = "confidence"
X_COLUMNS = [FRAME_COLUMN, CONFIDENCE_COLUMN]

DEFAULT_FRAME_COL = "video_frame_index"
DEFAULT_OUTPUT_DIR = Path("/media/ubuntu/research/passage_events_export_7")
DEFAULT_CAMERA_HYPERPARAMS = Path(
    "/media/ubuntu/research/carla_data_aug_event_eval_7/camera_hyperparams.csv"
)


@dataclass(frozen=True)
class DetectParams:
    """The subset of eval hyperparameters that changes where passages start and end."""

    threshold: float | None = None
    smooth_alpha: float = 1.0
    hyst_on_thr: float = 1.0
    hyst_off_thr: float = 1.0
    merge_gap_s: float = 1.0
    min_duration_s: float = 0.0
    representative: str = "midpoint"
    binarize_from_scores: bool = False


def load_frame_numbers(pred_path: Path, frame_col: str, n_rows: int) -> np.ndarray:
    frames = pd.to_numeric(
        pd.read_csv(pred_path, usecols=[frame_col])[frame_col], errors="coerce"
    ).to_numpy()[:n_rows]
    return frames.astype(np.int64)


def detection_mask(series, params: DetectParams) -> np.ndarray:
    if not params.binarize_from_scores:
        return series.active_fallback
    return binarize_scores(
        series.scores,
        params.threshold,
        params.smooth_alpha,
        params.hyst_on_thr,
        params.hyst_off_thr,
    )


def detect_passages(series, params: DetectParams) -> list[PassageEvent]:
    return events_from_binary_series(
        series.timestamps,
        detection_mask(series, params),
        series.camera_id,
        end_timestamps=series.end_timestamps,
        scores=series.scores,
        merge_gap_s=params.merge_gap_s,
        min_duration_s=params.min_duration_s,
        representative=params.representative,
    )


def detection_rows(
    events: list[PassageEvent], frames: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        row_index = event.representative_index
        rows.append(
            {
                FRAME_COLUMN: int(frames[row_index]),
                CONFIDENCE_COLUMN: "" if event.peak_score is None else event.peak_score,
            }
        )
    return rows


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "yes")


def params_from_row(row: pd.Series, fallback: DetectParams) -> DetectParams:
    threshold = row.get("threshold", "")
    return DetectParams(
        threshold=None if pd.isna(threshold) or str(threshold).strip() == "" else float(threshold),
        smooth_alpha=float(row.get("smooth_alpha", fallback.smooth_alpha)),
        hyst_on_thr=float(row.get("hyst_on_thr", fallback.hyst_on_thr)),
        hyst_off_thr=float(row.get("hyst_off_thr", fallback.hyst_off_thr)),
        merge_gap_s=float(row.get("merge_gap_s", fallback.merge_gap_s)),
        min_duration_s=float(row.get("min_duration_s", fallback.min_duration_s)),
        representative=str(row.get("representative", fallback.representative)),
        binarize_from_scores=as_bool(
            row.get("binarize_from_scores", fallback.binarize_from_scores)
        ),
    )


def load_camera_hyperparams(
    path: Path, pred_paths: list[Path], fallback: DetectParams
) -> list[DetectParams]:
    """Pair saved hyperparameters to pred files by row order, as the eval wrote them."""
    df = pd.read_csv(path)
    assert len(df) == len(pred_paths)
    return [params_from_row(row, fallback) for _, row in df.iterrows()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pred-csv", type=Path)
    group.add_argument("--pred-dir", type=Path, default=None)
    parser.add_argument("--truth-csv", type=Path, default=None)
    parser.add_argument("--truth-dir", type=Path, default=Y_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--y-suffix", type=str, default=Y_SUFFIX)
    parser.add_argument("--pred-glob", type=str, default=f"*{PRED_SUFFIX}")
    parser.add_argument("--include-cameras", type=str, default=None)
    parser.add_argument("--exclude-cameras", type=str, default=None)
    parser.add_argument("--timestamp-col", type=str, default=None)
    parser.add_argument("--frame-col", type=str, default=DEFAULT_FRAME_COL)
    parser.add_argument("--active-col", type=str, default="car_visible")
    parser.add_argument("--truth-col", type=str, default="car_visible")
    parser.add_argument("--peak-score-col", type=str, default="car_visible_prob")
    parser.add_argument("--max-length-mismatch", type=float, default=0.05)
    parser.add_argument(
        "--camera-hyperparams",
        type=Path,
        default=DEFAULT_CAMERA_HYPERPARAMS,
        help="camera_hyperparams.csv from eval_car_visible_events.py; paired to pred files by row order",
    )
    parser.add_argument("--binarize-from-scores", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--smooth-alpha", type=float, default=1.0)
    parser.add_argument("--hyst-on-thr", type=float, default=0.90)
    parser.add_argument("--hyst-off-thr", type=float, default=0.60)
    parser.add_argument("--merge-gap-s", type=float, default=1.0)
    parser.add_argument("--min-event-duration-s", type=float, default=0.0)
    parser.add_argument("--representative", choices=("peak", "midpoint"), default="midpoint")
    return parser.parse_args()


def resolve_pred_paths(args: argparse.Namespace) -> list[Path]:
    if args.pred_csv is not None:
        return [args.pred_csv]
    args.pred_dir = args.pred_dir or DEFAULT_PRED_DIR
    pred_paths = discover_pred_paths(args.pred_dir, args.pred_glob)
    n_found = len(pred_paths)
    pred_paths = filter_pred_paths(
        pred_paths,
        parse_camera_ids(args.include_cameras),
        parse_camera_ids(args.exclude_cameras),
    )
    if len(pred_paths) != n_found:
        print(f"Camera filter: kept {len(pred_paths)} of {n_found} sequences")
    return pred_paths


def main() -> None:
    args = parse_args()
    pred_paths = resolve_pred_paths(args)

    fixed_params = DetectParams(
        threshold=args.threshold,
        smooth_alpha=args.smooth_alpha,
        hyst_on_thr=args.hyst_on_thr,
        hyst_off_thr=args.hyst_off_thr,
        merge_gap_s=args.merge_gap_s,
        min_duration_s=args.min_event_duration_s,
        representative=args.representative,
        binarize_from_scores=args.binarize_from_scores,
    )
    saved_params = load_camera_hyperparams(
        args.camera_hyperparams, pred_paths, fixed_params
    )

    x_dir = args.output_dir / X_SUBDIR
    y_dir = args.output_dir / Y_TRUTH_SUBDIR
    x_dir.mkdir(parents=True, exist_ok=True)
    y_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    total_detections = 0

    for i, pred_path in enumerate(pred_paths):
        truth_path = (
            args.truth_csv
            if args.pred_csv is not None
            else resolve_truth_path(pred_path, args.truth_dir, args.y_suffix)
        )
        series = load_camera_series(
            pred_path,
            truth_path,
            timestamp_col=args.timestamp_col,
            active_col=args.active_col,
            truth_col=args.truth_col,
            peak_score_col=args.peak_score_col,
            max_length_mismatch=args.max_length_mismatch,
        )
        params = saved_params[i]
        frames = load_frame_numbers(pred_path, args.frame_col, series.timestamps.size)
        events = detect_passages(series, params)

        sequence = pred_stem(pred_path)
        x_path = x_dir / f"{sequence}.csv"
        pd.DataFrame(detection_rows(events, frames), columns=X_COLUMNS).to_csv(
            x_path, index=False
        )
        # Verbatim copy: the truth build is the reference, so it is never rewritten.
        y_path = y_dir / truth_path.name
        shutil.copy2(truth_path, y_path)

        total_detections += len(events)
        index_rows.append(
            {
                "sequence": sequence,
                "camera_id": series.camera_id,
                "x_csv": str(x_path.relative_to(args.output_dir)),
                "y_truth_csv": str(y_path.relative_to(args.output_dir)),
                "num_detections": len(events),
                "num_truth_frames": series.timestamps.size,
                "frame_col": args.frame_col,
                # y_truth has no frame column, so its row N is frame first_frame + N.
                "first_frame": int(frames[0]),
                "merge_gap_s": params.merge_gap_s,
                "min_duration_s": params.min_duration_s,
                "representative": params.representative,
                "binarize_from_scores": params.binarize_from_scores,
                "threshold": "" if params.threshold is None else params.threshold,
            }
        )
        print(f"{sequence}: {len(events)} detections -> {x_path.name} / {y_path.name}")

    index_path = args.output_dir / INDEX_FILENAME
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    print(f"\nExported {len(index_rows)} sequences, {total_detections} detections")
    print(f"  {x_dir}")
    print(f"  {y_dir}")
    print(f"  {index_path}")


if __name__ == "__main__":
    main()
