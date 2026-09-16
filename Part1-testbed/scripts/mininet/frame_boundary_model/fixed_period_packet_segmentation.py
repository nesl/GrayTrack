#!/usr/bin/env python3
"""Naive fixed-period packet segmentation baseline (1/FPS).

Marks the last packet in each 1/FPS time bin as a frame boundary, then
scores against rtp.marker ground truth (TP/FP/FN, P/R/F1, Acc).

Standalone: only needs pandas (+ stdlib).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TIME_COL = "frame.time_relative"
TRUTH_COLS = ("rtp.marker", "rtp.marker.original")


def marker_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes"]) | (pd.to_numeric(series, errors="coerce") == 1)


def truth_column(columns: pd.Index) -> str:
    for name in TRUTH_COLS:
        if name in columns:
            return name
    raise ValueError(f"CSV must contain one of: {', '.join(TRUTH_COLS)}")


def predict_fixed_period_markers(time_s: pd.Series, fps: float) -> pd.Series:
    """Last packet in each floor((t-t0)*fps) bin is a predicted boundary."""
    t = pd.to_numeric(time_s, errors="coerce")
    if t.isna().any():
        raise ValueError(f"Found NaN in {TIME_COL}")
    if len(t) == 0:
        return pd.Series(dtype=bool)

    period = 1.0 / float(fps)
    t0 = float(t.iloc[0])
    # Bin index for each packet; force monotonic non-decreasing bins by time order.
    order = t.argsort(kind="mergesort")
    t_sorted = t.iloc[order]
    bins = ((t_sorted - t0) / period).astype(int)

    pred_sorted = pd.Series(False, index=t_sorted.index)
    # Last index in each bin (in time order) is the boundary.
    last_in_bin = ~bins.duplicated(keep="last")
    pred_sorted.loc[last_in_bin] = True
    # Always mark the final packet.
    pred_sorted.iloc[-1] = True

    pred = pd.Series(False, index=t.index)
    pred.iloc[order] = pred_sorted.to_numpy()
    return pred


def metrics(pred: pd.Series, truth: pd.Series) -> dict[str, float]:
    pred_b = pred.astype(bool).to_numpy()
    truth_b = truth.astype(bool).to_numpy()
    tp = float((pred_b & truth_b).sum())
    fp = float((pred_b & ~truth_b).sum())
    fn = float((~pred_b & truth_b).sum())
    tn = float((~pred_b & ~truth_b).sum())
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
        "n": float(len(pred_b)),
        "n_pred": float(pred_b.sum()),
        "n_true": float(truth_b.sum()),
    }


def eval_one(path: Path, fps: float, write_out: Path | None) -> dict[str, float]:
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"{path}: missing column {TIME_COL}")
    truth_col = truth_column(df.columns)
    truth = marker_bool(df[truth_col])
    pred = predict_fixed_period_markers(df[TIME_COL], fps=fps)
    stats = metrics(pred, truth)

    if write_out is not None:
        out = df.copy()
        if "rtp.marker.original" not in out.columns:
            out["rtp.marker.original"] = truth.astype(int)
        out["pred_marker"] = pred.astype(int)
        out["pred_method"] = f"fixed_period_1/{fps:g}fps"
        write_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(write_out, index=False)

    return stats


def collect_inputs(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob("*.csv") if p.is_file())
    raise SystemExit(f"not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="One packet CSV, or a directory of CSVs",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Assumed video FPS (default: 20)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write CSVs with pred_marker column",
    )
    args = parser.parse_args()

    paths = collect_inputs(args.input)
    if not paths:
        raise SystemExit(f"no CSV files under {args.input}")

    totals = {k: 0.0 for k in ("tp", "fp", "fn", "tn", "n", "n_pred", "n_true")}
    print(f"files={len(paths)} fps={args.fps:g} period={1.0 / args.fps:.6f}s")

    for path in paths:
        out_path = None
        if args.output_dir is not None:
            out_path = args.output_dir.expanduser().resolve() / f"{path.stem}_fixed_period_pred.csv"
        stats = eval_one(path, fps=args.fps, write_out=out_path)
        for k in totals:
            totals[k] += stats[k]
        print(
            f"{path.name}: F1={stats['f1']:.3f} Acc={stats['acc']:.3f} "
            f"P={stats['precision']:.3f} R={stats['recall']:.3f} "
            f"TP={stats['tp']:.0f} FP={stats['fp']:.0f} FN={stats['fn']:.0f} "
            f"pred={stats['n_pred']:.0f} true={stats['n_true']:.0f}"
        )

    eps = 1e-12
    tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    print(
        f"TOTAL: F1={f1:.3f} Acc={acc:.3f} P={precision:.3f} R={recall:.3f} "
        f"TP={tp:.0f} FP={fp:.0f} FN={fn:.0f} "
        f"pred={totals['n_pred']:.0f} true={totals['n_true']:.0f} rows={totals['n']:.0f}"
    )


if __name__ == "__main__":
    main()
