"""Copy the held-out test-split X + truth CSVs into /media/ubuntu/research.

Splits at the **capture-run** level (not per camera CSV) with the same 80/10/10
fractions and seed=42 as training, so every test run keeps all of its cameras.
Only sequences with >= min_frames and a paired truth file are copied.

Writes:
  /media/ubuntu/research/carla_data_aug_x_infer_test/
  /media/ubuntu/research/carla_data_aug_truth_test/
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from car_visible_lstm_model import TrainConfig, X_DATA_DIR, Y_DATA_DIR, Y_SUFFIX

DEFAULT_X_OUT = Path("/media/ubuntu/research/carla_data_aug_x_infer_test")
DEFAULT_TRUTH_OUT = Path("/media/ubuntu/research/carla_data_aug_truth_test")
DEFAULT_PRED_OUT = Path("/media/ubuntu/research/carla_data_aug_car_visible_pred_test")
EXPECTED_CAMERAS = 11


def run_key(stem: str) -> str:
    return stem.split("_camera_", 1)[0]


def list_kept_by_run(
    x_dir: Path, truth_dir: Path, min_frames: int
) -> dict[str, list[Path]]:
    by_run: dict[str, list[Path]] = defaultdict(list)
    for x_path in tqdm(sorted(x_dir.glob("*.csv")), desc="Scanning sequences"):
        y_path = truth_dir / f"{x_path.stem}{Y_SUFFIX}"
        if not y_path.is_file():
            continue
        n = len(pd.read_csv(x_path, usecols=[0]))
        if n < min_frames:
            continue
        by_run[run_key(x_path.stem)].append(x_path)
    return dict(by_run)


def split_test_runs(runs: list[str], config: TrainConfig) -> list[str]:
    n = len(runs)
    n_train = max(1, int(round(config.train_split * n)))
    n_val = max(1, int(round(config.val_split * n)))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train = max(1, n_train - (1 - n_test))
        n_test = n - n_train - n_val
    _, _, test_subset = random_split(
        runs,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.split_seed),
    )
    return sorted(test_subset)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-dir", type=Path, default=X_DATA_DIR)
    parser.add_argument("--truth-dir", type=Path, default=Y_DATA_DIR)
    parser.add_argument("--x-out", type=Path, default=DEFAULT_X_OUT)
    parser.add_argument("--truth-out", type=Path, default=DEFAULT_TRUTH_OUT)
    parser.add_argument(
        "--pred-out",
        type=Path,
        default=DEFAULT_PRED_OUT,
        help="Also wipe this infer output dir so stale preds cannot mix with the new split",
    )
    args = parser.parse_args()

    config = TrainConfig()
    by_run = list_kept_by_run(args.x_dir, args.truth_dir, config.min_frames)
    if not by_run:
        raise SystemExit(f"No sequences with >= {config.min_frames} frames under {args.x_dir}")

    runs = sorted(by_run)
    test_runs = split_test_runs(runs, config)
    reset_dir(args.x_out)
    reset_dir(args.truth_out)
    reset_dir(args.pred_out)

    n_copied = 0
    incomplete = []
    for run in tqdm(test_runs, desc="Copying test runs"):
        paths = by_run[run]
        if len(paths) != EXPECTED_CAMERAS:
            incomplete.append((run, len(paths)))
        for x_src in paths:
            y_src = args.truth_dir / f"{x_src.stem}{Y_SUFFIX}"
            shutil.copy2(x_src, args.x_out / x_src.name)
            shutil.copy2(y_src, args.truth_out / y_src.name)
            n_copied += 1

    print(
        f"Split runs (seed={config.split_seed}): "
        f"total={len(runs)} test={len(test_runs)} "
        f"({len(test_runs) / len(runs):.1%})"
    )
    print(f"Copied {n_copied} camera CSVs across {len(test_runs)} runs")
    if incomplete:
        print(f"Runs with != {EXPECTED_CAMERAS} cameras after min_frames filter: {len(incomplete)}")
        for run, n in incomplete[:10]:
            print(f"  {run}: {n} cameras")
    print(f"X out:     {args.x_out}")
    print(f"Truth out: {args.truth_out}")
    print(f"Cleared pred out for fresh infer: {args.pred_out}")


if __name__ == "__main__":
    main()
