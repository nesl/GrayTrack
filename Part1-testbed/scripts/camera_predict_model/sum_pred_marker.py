"""Sum marker column in a CSV (pred_marker > rtp.marker > is_frame_boundary)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_MARKER_COLUMNS = ("pred_marker", "rtp.marker", "is_frame_boundary")


def _marker_column(path: Path) -> str:
    columns = pd.read_csv(path, nrows=0).columns
    for name in _MARKER_COLUMNS:
        if name in columns:
            return name
    raise ValueError(f"CSV must contain one of: {', '.join(_MARKER_COLUMNS)}")


def sum_pred_marker(path: Path) -> int:
    col = _marker_column(path)
    series = pd.read_csv(path, usecols=[col])[col]
    s = series.astype(str).str.strip().str.lower()
    truthy = s.isin(["1", "true", "t", "yes"]) | (pd.to_numeric(series, errors="coerce") == 1)
    return int(truthy.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input CSV path")
    args = parser.parse_args()

    path = args.csv.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"file does not exist: {path}")

    try:
        print(sum_pred_marker(path))
    except ValueError as exc:
        raise SystemExit(exc) from None


if __name__ == "__main__":
    main()
