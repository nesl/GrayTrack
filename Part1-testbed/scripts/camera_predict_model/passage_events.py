"""Passage-event extraction and one-to-one timestamp matching for car_visible eval."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COL_CANDIDATES = (
    "frame_time_relative_first",
    "frame.time_relative",
    "frame_time_relative",
    "timestamp",
)


@dataclass(frozen=True)
class PassageEvent:
    camera_id: str
    start_time: float
    end_time: float
    representative_timestamp: float
    peak_score: float | None = None
    # Rows in the caller's input order, so an event can be traced back to the
    # per-frame CSV it came from. Filled in by attach_row_indices.
    start_index: int | None = None
    end_index: int | None = None
    representative_index: int | None = None


def camera_id_from_path(path: Path) -> str:
    match = re.search(r"_camera_(\d+)_", path.stem)
    if match:
        return match.group(1)
    return path.stem


def resolve_timestamp_column(df: pd.DataFrame, timestamp_col: str | None = None) -> str:
    if timestamp_col is not None:
        if timestamp_col not in df.columns:
            raise ValueError(f"timestamp column {timestamp_col!r} not in dataframe")
        return timestamp_col
    for col in TIME_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"no timestamp column found; expected one of {TIME_COL_CANDIDATES}, got {list(df.columns)}"
    )


def end_timestamp_for_row(df: pd.DataFrame, row_idx: int, start_col: str) -> float:
    if "frame_time_relative_last" in df.columns:
        return float(df["frame_time_relative_last"].iloc[row_idx])
    return float(df[start_col].iloc[row_idx])


def contiguous_true_segments(active: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges [start, end] for each contiguous True run."""
    active = np.asarray(active, dtype=bool)
    if active.size == 0:
        return []

    segments: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, val in enumerate(active):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            segments.append((start, i - 1))
            in_run = False
    if in_run:
        segments.append((start, active.size - 1))
    return segments


def segment_to_event(
    start_idx: int,
    end_idx: int,
    timestamps: np.ndarray,
    end_timestamps: np.ndarray,
    scores: np.ndarray | None,
    camera_id: str,
    representative: str = "peak",
) -> PassageEvent:
    start_time = float(timestamps[start_idx])
    end_time = float(end_timestamps[end_idx])
    peak_score: float | None = None
    peak_time: float | None = None
    if scores is not None:
        peak_idx = start_idx + int(np.argmax(scores[start_idx : end_idx + 1]))
        peak_score = float(scores[peak_idx])
        peak_time = float(timestamps[peak_idx])

    if representative == "peak" and peak_time is not None:
        representative_timestamp = peak_time
    elif representative in ("peak", "midpoint"):
        representative_timestamp = 0.5 * (start_time + end_time)
    else:
        raise ValueError(f"unknown representative mode: {representative!r}")

    return PassageEvent(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        representative_timestamp=representative_timestamp,
        peak_score=peak_score,
    )


def merge_events_by_gap(
    events: list[PassageEvent], merge_gap_s: float, representative: str = "peak"
) -> list[PassageEvent]:
    if merge_gap_s <= 0.0 or len(events) <= 1:
        return events

    merged: list[PassageEvent] = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        gap = event.start_time - prev.end_time
        if gap < merge_gap_s:
            if representative == "midpoint":
                rep_ts = 0.5 * (prev.start_time + max(prev.end_time, event.end_time))
                scores = [s for s in (prev.peak_score, event.peak_score) if s is not None]
                peak_score = max(scores) if scores else None
            elif prev.peak_score is not None and event.peak_score is not None:
                if event.peak_score > prev.peak_score:
                    rep_ts = event.representative_timestamp
                    peak_score = event.peak_score
                else:
                    rep_ts = prev.representative_timestamp
                    peak_score = prev.peak_score
            elif prev.peak_score is not None:
                rep_ts = prev.representative_timestamp
                peak_score = prev.peak_score
            elif event.peak_score is not None:
                rep_ts = event.representative_timestamp
                peak_score = event.peak_score
            else:
                rep_ts = 0.5 * (prev.start_time + event.end_time)
                peak_score = None
            merged[-1] = PassageEvent(
                camera_id=prev.camera_id,
                start_time=prev.start_time,
                end_time=max(prev.end_time, event.end_time),
                representative_timestamp=rep_ts,
                peak_score=peak_score,
            )
        else:
            merged.append(event)
    return merged


def drop_short_events(events: list[PassageEvent], min_duration_s: float) -> list[PassageEvent]:
    """Drop blips shorter than a real passage (analogue of MIN_VISIBLE_RUN in frames)."""
    if min_duration_s <= 0.0:
        return events
    return [e for e in events if (e.end_time - e.start_time) >= min_duration_s]


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def attach_row_indices(
    events: list[PassageEvent],
    timestamps: np.ndarray,
    end_timestamps: np.ndarray,
    order: np.ndarray,
) -> list[PassageEvent]:
    """Tag each event with the input rows its start/end/representative times came from.

    Runs after merging and filtering so the lookup can never perturb event timing.
    `timestamps`/`end_timestamps` are sorted; `order` maps a sorted position back to
    the row the caller passed in.
    """
    if not events or timestamps.size == 0:
        return events
    return [
        replace(
            event,
            start_index=int(order[nearest_index(timestamps, event.start_time)]),
            end_index=int(order[nearest_index(end_timestamps, event.end_time)]),
            representative_index=int(
                order[nearest_index(timestamps, event.representative_timestamp)]
            ),
        )
        for event in events
    ]


def events_from_binary_series(
    timestamps: np.ndarray,
    active: np.ndarray,
    camera_id: str,
    *,
    end_timestamps: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    merge_gap_s: float = 1.0,
    min_duration_s: float = 0.0,
    representative: str = "peak",
) -> list[PassageEvent]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if timestamps.shape[0] != active.shape[0]:
        raise ValueError(
            f"timestamps and active must have same length, got {timestamps.shape[0]} vs {active.shape[0]}"
        )
    if scores is not None and scores.shape[0] != active.shape[0]:
        raise ValueError(
            f"scores and active must have same length, got {scores.shape[0]} vs {active.shape[0]}"
        )

    end_ts = (
        np.asarray(end_timestamps, dtype=np.float64)
        if end_timestamps is not None
        else timestamps
    )
    order = np.argsort(timestamps, kind="mergesort")
    timestamps = timestamps[order]
    end_ts = end_ts[order]
    active = active[order]
    if scores is not None:
        scores = scores[order]

    raw_events = [
        segment_to_event(start, end, timestamps, end_ts, scores, camera_id, representative)
        for start, end in contiguous_true_segments(active)
    ]
    merged = merge_events_by_gap(raw_events, merge_gap_s, representative)
    kept = drop_short_events(merged, min_duration_s)
    return attach_row_indices(kept, timestamps, end_ts, order)


def match_events_one_to_one(
    pred_events: list[PassageEvent],
    gt_events: list[PassageEvent],
    tolerance_s: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy one-to-one matching by smallest |Δ representative timestamp| within tolerance."""
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(pred_events):
        for gi, gt in enumerate(gt_events):
            delta = abs(pred.representative_timestamp - gt.representative_timestamp)
            if delta <= tolerance_s:
                candidates.append((delta, pi, gi))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        pairs.append((pi, gi))

    unmatched_pred = [i for i in range(len(pred_events)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_events)) if i not in matched_gt]
    return pairs, unmatched_pred, unmatched_gt


def timing_error(pred: PassageEvent, gt: PassageEvent) -> float:
    return pred.representative_timestamp - gt.representative_timestamp


def summarize_timing_errors(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {
            "mean_timing_error": float("nan"),
            "mean_abs_timing_error": float("nan"),
            "median_abs_timing_error": float("nan"),
            "timing_jitter": float("nan"),
            "timing_rmse": float("nan"),
        }
    abs_err = np.abs(errors)
    return {
        "mean_timing_error": float(errors.mean()),
        "mean_abs_timing_error": float(abs_err.mean()),
        "median_abs_timing_error": float(np.median(abs_err)),
        "timing_jitter": float(errors.std(ddof=0)),
        "timing_rmse": float(np.sqrt(np.mean(errors**2))),
    }


def classification_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def events_to_rows(events: list[PassageEvent]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        rows.append(
            {
                "camera_id": event.camera_id,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "representative_timestamp": event.representative_timestamp,
                "peak_score": event.peak_score,
                "start_index": event.start_index,
                "end_index": event.end_index,
                "representative_index": event.representative_index,
            }
        )
    return rows
