"""Compare pred_marker in infer CSVs to matching truth CSVs (aligned by frame.number)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INFER_DIR = Path("/media/ubuntu/research/xy_combined_data_x_infer_2/")
DEFAULT_TRUTH_DIR = Path("/media/ubuntu/research/xy_combined_data_x_truth_2/")

PRED_SUFFIX = "_rtp_marker_pred"


def truth_path_for_infer(infer_path: Path, truth_dir: Path) -> Path:
    stem = infer_path.stem
    if stem.endswith(PRED_SUFFIX):
        stem = stem[: -len(PRED_SUFFIX)]
    session, _, rest = stem.partition("_camera_")
    cam_num = rest.split("_", 1)[0]
    return sorted(truth_dir.glob(f"{session}_camera_{cam_num}_*.csv"))[0]


def count_pred_differences(infer_path: Path, truth_path: Path) -> tuple[int, int]:
    """Return (n_different, n_compared)."""
    df_i = pd.read_csv(infer_path, usecols=["frame.number", "pred_marker"])
    df_t = pd.read_csv(truth_path, usecols=["frame.number", "pred_marker"])

    merged = df_i.merge(
        df_t,
        on="frame.number",
        how="inner",
        suffixes=("_infer", "_truth"),
    )
    n_compared = len(merged)
    n_different = int((merged["pred_marker_infer"] != merged["pred_marker_truth"]).sum())
    return n_different, n_compared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--infer-dir",
        type=Path,
        default=DEFAULT_INFER_DIR,
        help=f"Directory of infer CSVs (default: {DEFAULT_INFER_DIR})",
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=DEFAULT_TRUTH_DIR,
        help=f"Directory of truth CSVs (default: {DEFAULT_TRUTH_DIR})",
    )
    args = parser.parse_args()

    infer_dir = args.infer_dir.expanduser().resolve()
    truth_dir = args.truth_dir.expanduser().resolve()

    if not infer_dir.is_dir():
        raise SystemExit(f"infer directory does not exist: {infer_dir}")
    if not truth_dir.is_dir():
        raise SystemExit(f"truth directory does not exist: {truth_dir}")

    infer_paths = sorted(infer_dir.glob("*.csv"))
    if not infer_paths:
        print(f"No *.csv files under {infer_dir}")
        return

    total_diff = 0

    for infer_path in infer_paths:
        truth_path = truth_path_for_infer(infer_path, truth_dir)
        try:
            n_diff, n_compared = count_pred_differences(infer_path, truth_path)
        except Exception:
            continue

        pct = (100.0 * n_diff / n_compared) if n_compared else 0.0
        print(f"{infer_path.name} {n_diff} {pct:.2f}%")
        total_diff += n_diff

    print(total_diff)


if __name__ == "__main__":
    main()
