"""Normalize packet CSVs under xy_combined_data to a fixed header layout.

Output columns (in order):
  frame.number,frame.time_epoch,frame.time_relative,frame.len,rtp.seq,
  rtp.timestamp,rtp.marker.original,prob,pred_marker

- ``rtp.marker.original`` is taken from the input if present; otherwise it is
  copied from ``rtp.marker``.
- ``prob`` is set to 0 for every row.
- ``pred_marker`` copies ``rtp.marker`` when that column exists; otherwise it
  matches ``rtp.marker.original`` (same values as the resolved original marker).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_DIR = Path("/media/ubuntu/research/xy_combined_data_2/")

OUTPUT_COLUMNS = [
    "frame.number",
    "frame.time_epoch",
    "frame.time_relative",
    "frame.len",
    "rtp.seq",
    "rtp.timestamp",
    "rtp.marker.original",
    "prob",
    "pred_marker",
]

BASE_COLUMNS = [
    "frame.number",
    "frame.time_epoch",
    "frame.time_relative",
    "frame.len",
    "rtp.seq",
    "rtp.timestamp",
]


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = _strip_columns(df)
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if "rtp.marker.original" in df.columns:
        orig = df["rtp.marker.original"]
    elif "rtp.marker" in df.columns:
        orig = df["rtp.marker"]
    else:
        raise ValueError("need rtp.marker or rtp.marker.original")

    if "rtp.marker" in df.columns:
        pred_src = df["rtp.marker"]
    else:
        pred_src = orig

    out = pd.DataFrame({c: df[c] for c in BASE_COLUMNS})
    out["rtp.marker.original"] = orig
    out["prob"] = 0
    out["pred_marker"] = pred_src
    return out[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of CSV files to normalize (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write CSVs here with the same filenames. Default: <input-dir>_header_normalized",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each CSV in --input-dir instead of writing to --output-dir.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist or is not a directory: {input_dir}")

    if args.in_place:
        output_dir = input_dir
    elif args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_header_normalized"

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        print(f"No *.csv files under {input_dir}")
        return

    ok = 0
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            out = normalize_dataframe(df)
        except Exception as e:
            print(f"SKIP {path.name}: {e}")
            continue
        dest = output_dir / path.name
        out.to_csv(dest, index=False)
        print(f"wrote {dest} ({len(out)} rows)")
        ok += 1

    print(f"\nDone. Normalized {ok} / {len(csv_paths)} file(s) -> {output_dir}")


if __name__ == "__main__":
    main()
