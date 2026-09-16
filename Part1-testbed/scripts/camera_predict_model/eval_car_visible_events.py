"""Event-level evaluation for car_visible / vehicle passage detection.

Per camera (default): sweep hyperparameters on truth, pick best F1, then report
event metrics averaged across cameras.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from car_visible_lstm_model import Y_DATA_DIR, Y_SUFFIX, postprocess_probs
from passage_events import (
    camera_id_from_path,
    classification_metrics,
    end_timestamp_for_row,
    events_from_binary_series,
    match_events_one_to_one,
    resolve_timestamp_column,
    summarize_timing_errors,
    timing_error,
)

PRED_SUFFIX = "_car_visible_pred.csv"
DEFAULT_PRED_DIR = Path("/media/ubuntu/research/carla_data_aug_car_visible_pred_7")
DEFAULT_EVENT_OUTPUT_DIR = Path("/media/ubuntu/research/carla_data_aug_event_eval_7")

DEFAULT_THRESHOLDS = (0.5, 0.7, 0.9, 0.95)
DEFAULT_HYST_PAIRS = ((0.9, 0.6), (0.8, 0.5), (0.7, 0.4))
DEFAULT_MERGE_GAPS = (0.5, 1.0, 2.0)
DEFAULT_MATCH_TOLERANCES = (0.5, 1.0, 2.0)
# Minimum passage duration, the time-domain analogue of MIN_VISIBLE_RUN (15 frames @ 20 FPS).
DEFAULT_MIN_DURATIONS = (0.0, 1.0, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class CameraSeries:
    camera_id: str
    timestamps: np.ndarray
    end_timestamps: np.ndarray
    scores: np.ndarray | None
    active_fallback: np.ndarray
    gt_active: np.ndarray


@dataclass(frozen=True)
class Hyperparams:
    threshold: float | None
    smooth_alpha: float
    hyst_on_thr: float
    hyst_off_thr: float
    merge_gap_s: float
    match_tolerance_s: float
    binarize_from_scores: bool
    min_duration_s: float = 0.0
    # "peak" uses the peak-score time for predictions but the midpoint for ground truth
    # (which has no scores); "midpoint" keeps both sides on the same reference point.
    representative: str = "peak"


def pred_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_car_visible_pred"):
        return stem[: -len("_car_visible_pred")]
    return stem


def resolve_truth_path(pred_path: Path, truth_dir: Path, y_suffix: str) -> Path:
    return truth_dir / f"{pred_stem(pred_path)}{y_suffix}"


def binarize_scores(
    scores: np.ndarray,
    threshold: float | None,
    smooth_alpha: float,
    hyst_on_thr: float,
    hyst_off_thr: float,
) -> np.ndarray:
    if hyst_on_thr != hyst_off_thr:
        return postprocess_probs(scores, smooth_alpha, hyst_on_thr, hyst_off_thr)
    thr = 0.5 if threshold is None else threshold
    return scores >= thr


def load_camera_series(
    pred_path: Path,
    truth_path: Path,
    *,
    timestamp_col: str | None,
    active_col: str,
    truth_col: str,
    peak_score_col: str | None,
    max_length_mismatch: float = 0.05,
) -> CameraSeries:
    df_pred = pd.read_csv(pred_path)
    df_truth = pd.read_csv(truth_path, usecols=[truth_col])
    n = min(len(df_pred), len(df_truth))
    if n == 0:
        raise ValueError(f"empty sequence for {pred_path.name}")
    if len(df_pred) != len(df_truth):
        # Rows are paired by index, so a large mismatch means pred and truth came from
        # different dataset builds; truncating would silently time-shift ground truth.
        drift = abs(len(df_pred) - len(df_truth)) / max(len(df_truth), 1)
        severity = "ERROR" if drift > max_length_mismatch else "warning"
        print(
            f"{severity}: length mismatch for {pred_path.name}: "
            f"pred={len(df_pred)} truth={len(df_truth)} ({100 * drift:.1f}%); using {n}"
        )
        if drift > max_length_mismatch:
            raise ValueError(
                f"{pred_path.name}: pred/truth row counts differ by {100 * drift:.1f}% "
                f"(limit {100 * max_length_mismatch:.1f}%). The truth directory is probably "
                f"from a different dataset build than the predictions. "
                f"Raise --max-length-mismatch to override."
            )

    ts_col = resolve_timestamp_column(df_pred, timestamp_col)
    timestamps = pd.to_numeric(df_pred[ts_col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    end_timestamps = np.array(
        [end_timestamp_for_row(df_pred, i, ts_col) for i in range(n)],
        dtype=np.float64,
    )
    gt_active = df_truth[truth_col].iloc[:n].astype(bool).to_numpy()

    scores: np.ndarray | None = None
    if peak_score_col and peak_score_col in df_pred.columns:
        scores = pd.to_numeric(df_pred[peak_score_col].iloc[:n], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float64
        )

    if active_col in df_pred.columns:
        active_fallback = df_pred[active_col].iloc[:n].astype(bool).to_numpy()
    elif scores is not None:
        active_fallback = scores >= 0.5
    else:
        raise ValueError(
            f"{pred_path.name}: missing {active_col!r} column and no {peak_score_col!r} scores"
        )

    return CameraSeries(
        camera_id=camera_id_from_path(pred_path),
        timestamps=timestamps,
        end_timestamps=end_timestamps,
        scores=scores,
        active_fallback=active_fallback,
        gt_active=gt_active,
    )


def observation_time_s(series: CameraSeries) -> float:
    """Camera time the sequence covers, so false events can be reported per camera-minute."""
    if series.timestamps.size == 0:
        return 0.0
    span = float(np.nanmax(series.end_timestamps) - np.nanmin(series.timestamps))
    return span if span > 0.0 else 0.0


def fp_rate_per_minute(fp: int, obs_time_s: float) -> float:
    if obs_time_s <= 0.0:
        return float("nan")
    return fp / (obs_time_s / 60.0)


def pred_active_for_params(series: CameraSeries, params: Hyperparams) -> np.ndarray:
    if params.binarize_from_scores:
        if series.scores is None:
            raise ValueError(f"camera {series.camera_id}: scores required for binarize_from_scores")
        return binarize_scores(
            series.scores,
            params.threshold,
            params.smooth_alpha,
            params.hyst_on_thr,
            params.hyst_off_thr,
        )
    return series.active_fallback


def evaluate_series(
    series: CameraSeries,
    params: Hyperparams,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], list[float]]:
    pred_active = pred_active_for_params(series, params)
    scores = series.scores if params.binarize_from_scores else series.scores

    pred_events = events_from_binary_series(
        series.timestamps,
        pred_active,
        series.camera_id,
        end_timestamps=series.end_timestamps,
        scores=scores,
        merge_gap_s=params.merge_gap_s,
        min_duration_s=params.min_duration_s,
        representative=params.representative,
    )
    gt_events = events_from_binary_series(
        series.timestamps,
        series.gt_active,
        series.camera_id,
        end_timestamps=series.end_timestamps,
        scores=None,
        merge_gap_s=params.merge_gap_s,
        representative=params.representative,
    )

    pairs, unmatched_pred, unmatched_gt = match_events_one_to_one(
        pred_events, gt_events, params.match_tolerance_s
    )
    matched_pred = {pi for pi, _ in pairs}
    pair_by_pred = {pi: gi for pi, gi in pairs}

    pred_rows: list[dict[str, object]] = []
    timing_errors: list[float] = []
    for pi, event in enumerate(pred_events):
        if pi in matched_pred:
            gi = pair_by_pred[pi]
            gt_event = gt_events[gi]
            err = timing_error(event, gt_event)
            timing_errors.append(err)
            pred_rows.append(
                {
                    "camera_id": event.camera_id,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "representative_timestamp": event.representative_timestamp,
                    "peak_score": event.peak_score,
                    "matched": True,
                    "matched_gt_timestamp": gt_event.representative_timestamp,
                    "timing_error": err,
                }
            )
        else:
            pred_rows.append(
                {
                    "camera_id": event.camera_id,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "representative_timestamp": event.representative_timestamp,
                    "peak_score": event.peak_score,
                    "matched": False,
                    "matched_gt_timestamp": "",
                    "timing_error": "",
                }
            )

    gt_rows: list[dict[str, object]] = []
    matched_gt = {gi for _, gi in pairs}
    for gi, event in enumerate(gt_events):
        if gi not in matched_gt:
            gt_rows.append(
                {
                    "camera_id": event.camera_id,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "representative_timestamp": event.representative_timestamp,
                    "matched": False,
                }
            )

    tp = len(pairs)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    cls = classification_metrics(tp, fp, fn)
    timing_stats = summarize_timing_errors(np.asarray(timing_errors, dtype=np.float64))
    num_gt = len(gt_events)
    obs_time_s = observation_time_s(series)
    metrics = {
        "camera_id": series.camera_id,
        "num_gt_events": num_gt,
        "num_pred_events": len(pred_events),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "ghost_rate": (fp / num_gt) if num_gt > 0 else float("nan"),
        **cls,
        **timing_stats,
        "observation_time_s": obs_time_s,
        "fp_per_camera_min": fp_rate_per_minute(fp, obs_time_s),
    }
    return pred_rows, gt_rows, metrics, timing_errors


def iter_hyperparam_grid(
    *,
    thresholds: tuple[float, ...],
    hyst_pairs: tuple[tuple[float, float], ...],
    smooth_alpha: float,
    merge_gaps: tuple[float, ...],
    match_tolerances: tuple[float, ...],
    min_durations: tuple[float, ...],
    representative: str,
    can_binarize_from_scores: bool,
    force_binarize_from_scores: bool,
) -> list[Hyperparams]:
    combos: list[Hyperparams] = []
    binarize_modes = (True,) if force_binarize_from_scores else ((False, True) if can_binarize_from_scores else (False,))

    for binarize_from_scores in binarize_modes:
        bin_opts: list[tuple[float | None, float, float, float]] = [(None, smooth_alpha, 1.0, 1.0)]
        if binarize_from_scores:
            bin_opts = [(thr, smooth_alpha, thr, thr) for thr in thresholds]
            bin_opts.extend((None, smooth_alpha, on, off) for on, off in hyst_pairs)

        for threshold, alpha, hyst_on, hyst_off in bin_opts:
            for merge_gap_s in merge_gaps:
                for match_tolerance_s in match_tolerances:
                    for min_duration_s in min_durations:
                        combos.append(
                            Hyperparams(
                                threshold=threshold,
                                smooth_alpha=alpha,
                                hyst_on_thr=hyst_on,
                                hyst_off_thr=hyst_off,
                                merge_gap_s=merge_gap_s,
                                match_tolerance_s=match_tolerance_s,
                                binarize_from_scores=binarize_from_scores,
                                min_duration_s=min_duration_s,
                                representative=representative,
                            )
                        )
    return combos


def sweep_camera(
    series: CameraSeries,
    grid: list[Hyperparams],
) -> tuple[list[dict[str, object]], Hyperparams, dict[str, object], int]:
    """Evaluate every hyperparameter combo for one camera; return sweep rows + best."""
    sweep_rows: list[dict[str, object]] = []
    best_params = grid[0]
    best_metrics: dict[str, object] = {}
    best_idx = 0
    best_f1 = -1.0
    best_tp = -1
    best_fp = 10**9

    for idx, params in enumerate(grid):
        _, _, metrics, _ = evaluate_series(series, params)
        sweep_rows.append(
            {
                "camera_id": series.camera_id,
                **hyperparams_to_row(params),
                "num_gt_events": metrics["num_gt_events"],
                "num_pred_events": metrics["num_pred_events"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "ghost_rate": metrics["ghost_rate"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mean_abs_timing_error": metrics["mean_abs_timing_error"],
                "is_best": False,
            }
        )
        f1 = float(metrics["f1"])
        tp = int(metrics["TP"])
        fp = int(metrics["FP"])
        if (
            f1 > best_f1
            or (f1 == best_f1 and tp > best_tp)
            or (f1 == best_f1 and tp == best_tp and fp < best_fp)
        ):
            best_f1 = f1
            best_tp = tp
            best_fp = fp
            best_params = params
            best_metrics = metrics
            best_idx = idx

    sweep_rows[best_idx]["is_best"] = True
    return sweep_rows, best_params, best_metrics, best_idx


def tune_camera(
    series: CameraSeries,
    grid: list[Hyperparams],
) -> tuple[Hyperparams, dict[str, object], list[dict[str, object]]]:
    sweep_rows, best_params, best_metrics, _ = sweep_camera(series, grid)
    return best_params, best_metrics, sweep_rows


def hyperparams_to_row(params: Hyperparams) -> dict[str, object]:
    return {
        "threshold": params.threshold if params.threshold is not None else "",
        "smooth_alpha": params.smooth_alpha,
        "hyst_on_thr": params.hyst_on_thr,
        "hyst_off_thr": params.hyst_off_thr,
        "merge_gap_s": params.merge_gap_s,
        "match_tolerance_s": params.match_tolerance_s,
        "min_duration_s": params.min_duration_s,
        "representative": params.representative,
        "binarize_from_scores": params.binarize_from_scores,
    }


EMPTY_METRIC_COLS = (
    "num_gt_events",
    "num_pred_events",
    "TP",
    "FP",
    "FN",
    "ghost_rate",
    "precision",
    "recall",
    "f1",
    "mean_timing_error",
    "mean_abs_timing_error",
    "median_abs_timing_error",
    "timing_jitter",
    "timing_rmse",
    "observation_time_s",
    "fp_per_camera_min",
)


def empty_metrics_row(label: str) -> dict[str, object]:
    return {"camera_id": label, **{col: float("nan") for col in EMPTY_METRIC_COLS}}


def pooled_metrics(
    rows: list[dict[str, object]],
    timing_errors: np.ndarray | None = None,
    label: str = "POOLED",
) -> dict[str, object]:
    """Micro-average: sum counts across cameras, then derive rates.

    Timing stats come from the pooled per-event errors, not from averaging the
    per-sequence summaries: averaging is exact for the mean and mean-square but not
    for the median, and a per-sequence standard deviation is ~0 when a sequence has
    a single matched event.
    """
    if not rows:
        return empty_metrics_row(label)

    tp = sum(int(r["TP"]) for r in rows)
    fp = sum(int(r["FP"]) for r in rows)
    fn = sum(int(r["FN"]) for r in rows)
    num_gt = sum(int(r["num_gt_events"]) for r in rows)
    obs_time_s = sum(float(r["observation_time_s"]) for r in rows)
    errors = (
        np.asarray([], dtype=np.float64)
        if timing_errors is None
        else np.asarray(timing_errors, dtype=np.float64)
    )
    return {
        "camera_id": label,
        "num_gt_events": num_gt,
        "num_pred_events": sum(int(r["num_pred_events"]) for r in rows),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "ghost_rate": (fp / num_gt) if num_gt > 0 else float("nan"),
        **classification_metrics(tp, fp, fn),
        **summarize_timing_errors(errors),
        "observation_time_s": obs_time_s,
        "fp_per_camera_min": fp_rate_per_minute(fp, obs_time_s),
    }


def print_pooled_summary(pooled: dict[str, object], errors: np.ndarray) -> None:
    """Report the pooled numbers a tracking simulation needs to be parameterized."""
    obs_s = float(pooled["observation_time_s"])
    obs_min = obs_s / 60.0
    fp = int(pooled["FP"])
    print(
        f"\nObservation time: {obs_s:.1f} s = {obs_min:.2f} camera-min"
    )
    print(
        f"False events: {fp} -> {float(pooled['fp_per_camera_min']):.5f} FP/camera-min"
    )
    if errors.size == 0:
        print("Timing error: no matched events")
        return
    print(
        f"Timing error over {errors.size} matched events: "
        f"mean={float(pooled['mean_timing_error']):+.4f}s "
        f"std={float(pooled['timing_jitter']):.4f}s "
        f"mean|d|={float(pooled['mean_abs_timing_error']):.4f}s "
        f"median|d|={float(pooled['median_abs_timing_error']):.4f}s "
        f"RMSE={float(pooled['timing_rmse']):.4f}s"
    )


def average_metrics(rows: list[dict[str, object]], label: str = "MEAN") -> dict[str, object]:
    if not rows:
        return empty_metrics_row(label)

    def mean_col(col: str) -> float:
        vals = [float(r[col]) for r in rows if r.get(col) == r.get(col)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "camera_id": label,
        "num_gt_events": mean_col("num_gt_events"),
        "num_pred_events": mean_col("num_pred_events"),
        "TP": mean_col("TP"),
        "FP": mean_col("FP"),
        "FN": mean_col("FN"),
        "ghost_rate": mean_col("ghost_rate"),
        "precision": mean_col("precision"),
        "recall": mean_col("recall"),
        "f1": mean_col("f1"),
        "mean_timing_error": mean_col("mean_timing_error"),
        "mean_abs_timing_error": mean_col("mean_abs_timing_error"),
        "median_abs_timing_error": mean_col("median_abs_timing_error"),
        "timing_jitter": mean_col("timing_jitter"),
        "timing_rmse": mean_col("timing_rmse"),
        "observation_time_s": mean_col("observation_time_s"),
        "fp_per_camera_min": mean_col("fp_per_camera_min"),
    }


def discover_pred_paths(pred_dir: Path, pred_glob: str) -> list[Path]:
    return sorted(pred_dir.rglob(pred_glob) if "**" in pred_glob else pred_dir.glob(pred_glob))


def parse_camera_ids(text: str | None) -> set[str]:
    if not text:
        return set()
    return {c.strip() for c in text.split(",") if c.strip()}


def filter_pred_paths(
    paths: list[Path], include: set[str], exclude: set[str]
) -> list[Path]:
    if not include and not exclude:
        return paths
    kept = []
    for p in paths:
        cam = camera_id_from_path(p)
        if include and cam not in include:
            continue
        if cam in exclude:
            continue
        kept.append(p)
    return kept


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def parse_hyst_pairs(text: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        on_s, off_s = chunk.split(",")
        pairs.append((float(on_s.strip()), float(off_s.strip())))
    return tuple(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pred-csv", type=Path)
    group.add_argument("--pred-dir", type=Path, default=None)
    parser.add_argument("--truth-csv", type=Path, default=None)
    parser.add_argument("--truth-dir", type=Path, default=Y_DATA_DIR)
    parser.add_argument("--y-suffix", type=str, default=Y_SUFFIX)
    parser.add_argument("--pred-glob", type=str, default=f"*{PRED_SUFFIX}")
    parser.add_argument("--include-cameras", type=str, default=None, help="Comma-separated camera ids to keep")
    parser.add_argument("--exclude-cameras", type=str, default=None, help="Comma-separated camera ids to drop")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVENT_OUTPUT_DIR)
    parser.add_argument("--timestamp-col", type=str, default=None)
    parser.add_argument("--active-col", type=str, default="car_visible")
    parser.add_argument("--truth-col", type=str, default="car_visible")
    parser.add_argument("--peak-score-col", type=str, default="car_visible_prob")
    parser.add_argument(
        "--max-length-mismatch",
        type=float,
        default=0.05,
        help="Fail if pred/truth row counts differ by more than this fraction (mispaired dataset build)",
    )
    parser.add_argument("--no-tune-per-camera", action="store_true")
    parser.add_argument("--binarize-from-scores", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--smooth-alpha", type=float, default=1.0)
    parser.add_argument("--hyst-on-thr", type=float, default=0.90)
    parser.add_argument("--hyst-off-thr", type=float, default=0.60)
    parser.add_argument("--merge-gap-s", type=float, default=1.0)
    parser.add_argument("--match-tolerance-s", type=float, default=1.0)
    parser.add_argument("--min-event-duration-s", type=float, default=0.0)
    parser.add_argument(
        "--representative",
        choices=("peak", "midpoint"),
        default="midpoint",
        help="Reference point for matching/timing. 'midpoint' keeps predictions and ground "
        "truth on the same basis; 'peak' uses peak-score time for predictions only",
    )
    parser.add_argument("--tune-thresholds", type=str, default=",".join(str(x) for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--tune-hyst-pairs", type=str, default=";".join(f"{a},{b}" for a, b in DEFAULT_HYST_PAIRS))
    parser.add_argument("--tune-merge-gaps", type=str, default=",".join(str(x) for x in DEFAULT_MERGE_GAPS))
    parser.add_argument(
        "--tune-match-tolerances",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MATCH_TOLERANCES),
    )
    parser.add_argument(
        "--tune-min-durations",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MIN_DURATIONS),
    )
    args = parser.parse_args()

    if args.pred_csv is not None:
        pred_paths = [args.pred_csv]
    else:
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

    tune_per_camera = not args.no_tune_per_camera
    sample_cols = pd.read_csv(pred_paths[0], nrows=0).columns
    has_precomputed_active = args.active_col in sample_cols
    can_binarize_from_scores = bool(args.peak_score_col and args.peak_score_col in sample_cols)
    grid = iter_hyperparam_grid(
        thresholds=parse_float_tuple(args.tune_thresholds),
        hyst_pairs=parse_hyst_pairs(args.tune_hyst_pairs),
        smooth_alpha=args.smooth_alpha,
        merge_gaps=parse_float_tuple(args.tune_merge_gaps),
        match_tolerances=parse_float_tuple(args.tune_match_tolerances),
        min_durations=parse_float_tuple(args.tune_min_durations),
        representative=args.representative,
        can_binarize_from_scores=can_binarize_from_scores,
        force_binarize_from_scores=args.binarize_from_scores or not has_precomputed_active,
    )
    fixed_params = Hyperparams(
        threshold=args.threshold,
        smooth_alpha=args.smooth_alpha,
        hyst_on_thr=args.hyst_on_thr,
        hyst_off_thr=args.hyst_off_thr,
        merge_gap_s=args.merge_gap_s,
        match_tolerance_s=args.match_tolerance_s,
        binarize_from_scores=args.binarize_from_scores,
        min_duration_s=args.min_event_duration_s,
        representative=args.representative,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_pred_rows: list[dict[str, object]] = []
    all_gt_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    hyperparam_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    pooled_timing_errors: list[float] = []

    print(f"Hyperparameter grid: {len(grid)} combos per camera")

    for pred_path in pred_paths:
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

        if tune_per_camera:
            print(f"Sweeping {len(grid)} combos for camera {series.camera_id} ({pred_path.name})...")
            best_params, best_metrics, camera_sweep = tune_camera(series, grid)
            sweep_rows.extend(camera_sweep)
            params = best_params
            metrics = best_metrics
            pred_rows, gt_rows, _, timing_errors = evaluate_series(series, params)
        else:
            params = fixed_params
            pred_rows, gt_rows, metrics, timing_errors = evaluate_series(series, params)

        pooled_timing_errors.extend(timing_errors)
        all_pred_rows.extend(pred_rows)
        all_gt_rows.extend(gt_rows)
        metrics_rows.append(metrics)
        hyperparam_rows.append({"camera_id": series.camera_id, **hyperparams_to_row(params), "f1": metrics["f1"]})
        print(
            f"{pred_path.name}: tuned={tune_per_camera} "
            f"thr={params.threshold} hyst=({params.hyst_on_thr},{params.hyst_off_thr}) "
            f"merge={params.merge_gap_s:g}s match={params.match_tolerance_s:g}s "
            f"mindur={params.min_duration_s:g}s "
            f"GT={metrics['num_gt_events']} pred={metrics['num_pred_events']} "
            f"TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
        )

    camera_rows = list(metrics_rows)
    with_gt = [r for r in camera_rows if int(r["num_gt_events"]) > 0]
    pooled_errors = np.asarray(pooled_timing_errors, dtype=np.float64)
    metrics_rows.append(average_metrics(camera_rows))
    metrics_rows.append(average_metrics(with_gt, label="MEAN_GT_ONLY"))
    pooled = pooled_metrics(camera_rows, pooled_errors)
    metrics_rows.append(pooled)
    pd.DataFrame(all_pred_rows).to_csv(args.output_dir / "predicted_events.csv", index=False)
    pd.DataFrame(all_gt_rows).to_csv(args.output_dir / "unmatched_gt_events.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "event_metrics.csv", index=False)
    pd.DataFrame(hyperparam_rows).to_csv(args.output_dir / "camera_hyperparams.csv", index=False)
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(args.output_dir / "hyperparam_sweep.csv", index=False)
    for row in metrics_rows[-3:]:
        print(
            f"{row['camera_id']}: P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1']:.3f} (TP={row['TP']:g} FP={row['FP']:g} FN={row['FN']:g})"
        )
    print_pooled_summary(pooled, pooled_errors)
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
