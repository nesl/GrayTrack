"""Accuracy of pred_marker vs RTP marker ground truth in an infer CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_TRUTH_COLUMNS = ("rtp.marker", "rtp.marker.original")


def _marker_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes"]) | (pd.to_numeric(series, errors="coerce") == 1)


def _truth_column(columns: pd.Index) -> str:
    for name in _TRUTH_COLUMNS:
        if name in columns:
            return name
    raise ValueError(f"CSV must contain one of: {', '.join(_TRUTH_COLUMNS)}")


def eval_accuracy(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    if "pred_marker" not in df.columns:
        raise ValueError("CSV must contain column: pred_marker")

    truth_col = _truth_column(df.columns)
    truth = _marker_bool(df[truth_col])
    pred = _marker_bool(df["pred_marker"])

    n = len(df)
    n_correct = int((truth == pred).sum())
    accuracy_pct = 100.0 * n_correct / n if n else 0.0

    truth_counts = truth.value_counts().sort_index()
    pred_counts = pred.value_counts().sort_index()

    return {
        "path": path,
        "truth_col": truth_col,
        "n_rows": n,
        "n_correct": n_correct,
        "n_incorrect": n - n_correct,
        "accuracy_pct": accuracy_pct,
        "truth_counts": truth_counts,
        "pred_counts": pred_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input CSV path")
    args = parser.parse_args()

    path = args.csv.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"file does not exist: {path}")

    try:
        result = eval_accuracy(path)
    except ValueError as exc:
        raise SystemExit(exc) from None

    truth_col = result["truth_col"]
    print(f"file: {result['path']}")
    print(f"truth column: {truth_col}")
    print(f"rows: {result['n_rows']}")
    print(f"correct: {result['n_correct']}")
    print(f"incorrect: {result['n_incorrect']}")
    print(f"accuracy: {result['accuracy_pct']:.2f}%")
    print()
    print(f"{truth_col} counts (0=False, 1=True):")
    for val, count in result["truth_counts"].items():
        print(f"  {int(val)}: {count}")
    print("pred_marker counts (0=False, 1=True):")
    for val, count in result["pred_counts"].items():
        print(f"  {int(val)}: {count}")


if __name__ == "__main__":
    main()
