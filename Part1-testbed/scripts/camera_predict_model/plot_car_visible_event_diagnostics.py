"""Per-camera event stats + best/median/worst passage timelines.

Reads an eval_car_visible_events.py output directory, aggregates sequence-level
metrics by physical camera, and plots prediction vs ground-truth timelines for
the best / median / worst sequence of each camera.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_car_visible_events import (
    PRED_SUFFIX,
    discover_pred_paths,
    load_camera_series,
    pred_stem,
    resolve_truth_path,
)
from car_visible_lstm_model import Y_DATA_DIR, Y_SUFFIX
from passage_events import events_from_binary_series

AGG_LABELS = ("MEAN", "MEAN_GT_ONLY", "POOLED", "ALL")

PRED_COLOR = "#d62728"
GT_COLOR = "#2ca02c"

DEFAULT_PRED_DIR = Path("/media/ubuntu/research/carla_data_aug_car_visible_pred_7")
DEFAULT_EVAL_DIR = Path("/media/ubuntu/research/carla_data_aug_event_eval_7")


def load_sequence_metrics(eval_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(eval_dir / "event_metrics.csv")
    return df[~df.camera_id.astype(str).isin(AGG_LABELS)].copy()


def attach_sequence_paths(metrics: pd.DataFrame, pred_dir: Path, pred_glob: str) -> pd.DataFrame:
    """event_metrics rows are in pred-path order, so zip them back together."""
    paths = discover_pred_paths(pred_dir, pred_glob)
    assert len(paths) == len(metrics)
    out = metrics.reset_index(drop=True).copy()
    out["pred_path"] = [str(p) for p in paths]
    out["sequence"] = [pred_stem(p) for p in paths]
    return out


def per_camera_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby("camera_id", as_index=False).agg(
        num_sequences=("f1", "size"),
        num_sequences_with_gt=("num_gt_events", lambda s: int((s > 0).sum())),
        gt_events=("num_gt_events", "sum"),
        pred_events=("num_pred_events", "sum"),
        TP=("TP", "sum"),
        FP=("FP", "sum"),
        FN=("FN", "sum"),
        mean_f1=("f1", "mean"),
        median_f1=("f1", "median"),
        mean_abs_timing_error=("mean_abs_timing_error", "mean"),
        observation_time_s=("observation_time_s", "sum"),
    )
    tp, fp, fn = grouped.TP, grouped.FP, grouped.FN
    obs_min = grouped.observation_time_s / 60.0
    grouped["fp_per_camera_min"] = np.where(obs_min > 0, fp / obs_min, np.nan)
    grouped["precision"] = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    grouped["recall"] = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    denom = grouped.precision + grouped.recall
    grouped["f1"] = np.where(denom > 0, 2 * grouped.precision * grouped.recall / denom, 0.0)
    grouped["ghost_rate"] = np.where(grouped.gt_events > 0, fp / grouped.gt_events, np.nan)
    return grouped.sort_values("f1", ascending=False).reset_index(drop=True)


def pick_best_median_worst(group: pd.DataFrame) -> dict[str, pd.Series]:
    """Rank a camera's sequences by F1; ties broken by fewer false positives."""
    ranked = group.sort_values(["f1", "FP"], ascending=[False, True]).reset_index(drop=True)
    return {
        "best": ranked.iloc[0],
        "median": ranked.iloc[len(ranked) // 2],
        "worst": ranked.iloc[-1],
    }


def plot_timeline(
    ax: plt.Axes,
    row: pd.Series,
    label: str,
    *,
    truth_dir: Path,
    y_suffix: str,
    active_col: str,
    truth_col: str,
    peak_score_col: str,
    merge_gap_s: float,
    min_duration_s: float,
) -> None:
    pred_path = Path(row.pred_path)
    truth_path = resolve_truth_path(pred_path, truth_dir, y_suffix)
    series = load_camera_series(
        pred_path,
        truth_path,
        timestamp_col=None,
        active_col=active_col,
        truth_col=truth_col,
        peak_score_col=peak_score_col,
    )

    t = series.timestamps
    ax.fill_between(t, 0, series.gt_active.astype(float), step="post", color=GT_COLOR, alpha=0.35, lw=0)
    ax.step(t, series.gt_active.astype(float), where="post", color=GT_COLOR, lw=1.0, label="ground truth")
    ax.step(t, series.active_fallback.astype(float) * 0.92, where="post", color=PRED_COLOR, lw=1.0, label="prediction")

    if series.scores is not None:
        ax.plot(t, series.scores, color="#1f77b4", lw=0.6, alpha=0.55, label="score")

    pred_events = events_from_binary_series(
        t,
        series.active_fallback,
        series.camera_id,
        end_timestamps=series.end_timestamps,
        scores=series.scores,
        merge_gap_s=merge_gap_s,
        min_duration_s=min_duration_s,
    )
    for event in pred_events:
        ax.axvline(event.representative_timestamp, color=PRED_COLOR, ls=":", lw=0.8, alpha=0.8)

    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1])
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(
        f"{label} — camera {row.camera_id} — F1={row.f1:.2f} "
        f"(GT={int(row.num_gt_events)} pred={int(row.num_pred_events)} "
        f"TP={int(row.TP)} FP={int(row.FP)} FN={int(row.FN)})",
        fontsize=9,
    )
    ax.grid(alpha=0.25, lw=0.4)


def plot_camera(
    camera_id: object,
    group: pd.DataFrame,
    out_path: Path,
    **kwargs: object,
) -> None:
    picks = pick_best_median_worst(group)
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharey=True)
    for ax, (label, row) in zip(axes, picks.items()):
        plot_timeline(ax, row, label, **kwargs)  # type: ignore[arg-type]
    axes[-1].set_xlabel("time (s, frame_time_relative)")
    handles = [
        mpatches.Patch(color=GT_COLOR, alpha=0.5, label="ground-truth passage"),
        mpatches.Patch(color=PRED_COLOR, label="predicted car_visible"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(f"Camera {camera_id} — best / median / worst sequence", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_overview(stats: pd.DataFrame, metrics: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    order = stats.sort_values("f1", ascending=False)
    axes[0].bar(range(len(order)), order.f1, color="#1f77b4")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order.camera_id.astype(str), rotation=90, fontsize=7)
    axes[0].set_ylabel("pooled F1")
    axes[0].set_title("Per-camera event F1")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(range(len(order)), order.ghost_rate, color="#d62728")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order.camera_id.astype(str), rotation=90, fontsize=7)
    axes[1].set_ylabel("ghost rate (FP / GT)")
    axes[1].set_title("Per-camera ghost rate")
    axes[1].grid(alpha=0.3, axis="y")

    with_gt = metrics[metrics.num_gt_events > 0]
    axes[2].hist(with_gt.f1, bins=20, color="#2ca02c", edgecolor="white")
    axes[2].set_xlabel("sequence F1 (sequences with ground truth)")
    axes[2].set_ylabel("sequences")
    axes[2].set_title(f"F1 distribution (n={len(with_gt)})")
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--truth-dir", type=Path, default=Y_DATA_DIR)
    parser.add_argument("--pred-glob", type=str, default=f"*{PRED_SUFFIX}")
    parser.add_argument("--y-suffix", type=str, default=Y_SUFFIX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--active-col", type=str, default="car_visible")
    parser.add_argument("--truth-col", type=str, default="car_visible")
    parser.add_argument("--peak-score-col", type=str, default="car_visible_prob")
    parser.add_argument("--merge-gap-s", type=float, default=1.0)
    parser.add_argument("--min-event-duration-s", type=float, default=0.0)
    parser.add_argument("--max-cameras", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or (args.eval_dir / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = attach_sequence_paths(
        load_sequence_metrics(args.eval_dir), args.pred_dir, args.pred_glob
    )
    stats = per_camera_stats(metrics)
    stats.to_csv(out_dir / "per_camera_stats.csv", index=False)
    metrics.drop(columns=["pred_path"]).to_csv(out_dir / "per_sequence_stats.csv", index=False)

    print(stats.to_string(index=False))
    plot_overview(stats, metrics, out_dir / "per_camera_overview.png")

    plot_kwargs = {
        "truth_dir": args.truth_dir,
        "y_suffix": args.y_suffix,
        "active_col": args.active_col,
        "truth_col": args.truth_col,
        "peak_score_col": args.peak_score_col,
        "merge_gap_s": args.merge_gap_s,
        "min_duration_s": args.min_event_duration_s,
    }

    camera_ids = list(stats.camera_id)
    if args.max_cameras is not None:
        camera_ids = camera_ids[: args.max_cameras]

    for camera_id in camera_ids:
        group = metrics[metrics.camera_id == camera_id]
        if group.empty:
            continue
        plot_camera(camera_id, group, out_dir / f"camera_{camera_id}_timelines.png", **plot_kwargs)

    print(f"\nWrote {len(camera_ids)} camera plots + stats to {out_dir}")


if __name__ == "__main__":
    main()
