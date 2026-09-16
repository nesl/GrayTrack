from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Stage 2 outputs (paired per-frame X + truth).
X_DATA_DIR = Path("/media/ubuntu/research/carla_data_aug_combined_x_infer_no_warmup")
Y_DATA_DIR = Path("/media/ubuntu/research/carla_data_aug_infer_truth_no_warmup")

# Checkpoints (written next to this script).
SCRIPT_DIR = Path(__file__).resolve().parent
NORM_CHECKPOINT = SCRIPT_DIR / "car_visible_lstm_norm.pt"
MODEL_CHECKPOINT = SCRIPT_DIR / "best_car_visible_lstm.pt"

# Example inference (writes car_visible_prob; binarize in eval_car_visible_events.py):
#   python car_visible_lstm_infer.py \
#     --model-checkpoint best_car_visible_lstm.pt \
#     --norm-checkpoint car_visible_lstm_norm.pt \
#     --x-dir /path/to/x_csvs \
#     --output /media/ubuntu/research/carla_data_aug_car_visible_pred

# Raw CSV columns read for feature construction (combined_frame_len_bytes is not fed raw).
# video_frame_index stays in the CSV for alignment but is not a model feature.
X_LOAD_COLS = ["combined_frame_len_bytes", "frame_time_span"]
FRAME_SIZE_RAW_COL = "combined_frame_len_bytes"
FRAME_SIZE_EWM_SPAN = 32

FEATURE_COLS = [
    "combined_frame_len_bytes_log",
    "combined_frame_len_bytes_residual",
    "combined_frame_len_bytes_ratio",
    "frame_time_span",
]
SPEED_COLS = ["speed"]
TRUTH_COLS = ["car_visible", *SPEED_COLS]
Y_SUFFIX = "_truth.csv"  # X is *_rtp_marker_pred.csv -> Y is *_rtp_marker_pred_truth.csv


def derive_frame_size_features(frame_len_bytes: np.ndarray | pd.Series) -> dict[str, np.ndarray]:
    """Causal derived frame-size features from raw combined_frame_len_bytes."""
    bytes_ = pd.Series(frame_len_bytes, dtype=np.float64)
    log_ = np.log1p(bytes_)
    baseline = bytes_.ewm(span=FRAME_SIZE_EWM_SPAN, adjust=False).mean()
    return {
        "combined_frame_len_bytes_log": log_.to_numpy(dtype=np.float32),
        "combined_frame_len_bytes_residual": (bytes_ - baseline).to_numpy(dtype=np.float32),
        "combined_frame_len_bytes_ratio": (bytes_ / (baseline + 1e-9)).to_numpy(dtype=np.float32),
    }


def build_feature_matrix(df_x: pd.DataFrame) -> np.ndarray:
    derived = derive_frame_size_features(df_x[FRAME_SIZE_RAW_COL])
    cols = [
        derived["combined_frame_len_bytes_log"],
        derived["combined_frame_len_bytes_residual"],
        derived["combined_frame_len_bytes_ratio"],
        df_x["frame_time_span"].to_numpy(dtype=np.float32),
    ]
    return np.column_stack(cols)


class CarVisibleLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        p_drop: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(out_dim // 2, 1),
        )
        self.speed_head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h, _ = self.lstm(x)
        logit = self.head(h).squeeze(-1)
        speed = self.speed_head(h)
        assert speed.shape[-1] == 1, f"Expected regression output dim 1, got {speed.shape}"
        return {"logit": logit, "speed": speed}


def build_window_index(
    mask: torch.Tensor,
    window_size: int,
    stride: int,
) -> list[tuple[int, int]]:
    window_size = int(window_size)
    stride = int(stride)
    index: list[tuple[int, int]] = []
    n_seq, _ = mask.shape
    for seq_id in range(n_seq):
        n_valid = int(mask[seq_id].sum().item())
        if n_valid == 0:
            continue
        starts = list(range(0, n_valid, stride))
        last_start = max(0, n_valid - window_size)
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        for st in sorted(set(starts)):
            index.append((seq_id, st))
    return index


def window_label_is_all_zero(
    y: torch.Tensor,
    mask: torch.Tensor,
    seq_id: int,
    start: int,
    window_size: int,
) -> bool:
    n_valid = int(mask[seq_id].sum().item())
    n_win = max(0, min(window_size, n_valid - start))
    if n_win == 0:
        return True
    segment = y[seq_id, start : start + n_win]
    return not bool(segment.any().item())


def compute_zero_windows_to_drop(n_zero: int, n_nonzero: int) -> int:
    """How many all-zero windows to remove so zero / non-zero window counts match."""
    if n_zero == 0 or n_nonzero == 0:
        return 0
    n_drop = n_zero - n_nonzero
    if n_drop <= 0:
        return 0
    return min(n_drop, n_zero)


def uniform_drop_ranks(n_pool: int, n_drop: int) -> set[int]:
    """Pick ranks in [0, n_pool) to drop, spaced uniformly across the pool."""
    if n_drop <= 0:
        return set()
    if n_drop >= n_pool:
        return set(range(n_pool))
    ranks = np.linspace(0, n_pool - 1, num=n_drop, dtype=int)
    return {int(r) for r in ranks}


def balance_window_index(
    index: list[tuple[int, int]],
    y: torch.Tensor,
    mask: torch.Tensor,
    window_size: int,
) -> tuple[list[tuple[int, int]], dict[str, int | float]]:
    zero_positions: list[int] = []

    for i, (seq_id, start) in enumerate(
        tqdm(index, desc="Classifying windows", dynamic_ncols=True)
    ):
        if window_label_is_all_zero(y, mask, seq_id, start, window_size):
            zero_positions.append(i)

    n_zero_start = len(zero_positions)
    n_nonzero_start = len(index) - n_zero_start
    n_dropped = compute_zero_windows_to_drop(n_zero_start, n_nonzero_start)

    drop_global: set[int] = set()
    if n_dropped > 0:
        ranks_to_drop = uniform_drop_ranks(len(zero_positions), n_dropped)
        drop_global = {zero_positions[r] for r in ranks_to_drop}

    balanced = [win for i, win in enumerate(index) if i not in drop_global]
    n_zero_remaining = n_zero_start - len(drop_global)
    total_final = len(balanced)
    p_zero_final = n_zero_remaining / total_final if total_final else 0.0
    p_nonzero_final = (total_final - n_zero_remaining) / total_final if total_final else 0.0

    stats: dict[str, int | float] = {
        "n_zero_start": n_zero_start,
        "n_nonzero_start": n_nonzero_start,
        "n_dropped": n_dropped,
        "n_remaining": total_final,
        "n_zero_remaining": n_zero_remaining,
        "n_nonzero_remaining": n_nonzero_start,
        "p_zero_final": p_zero_final,
        "p_nonzero_final": p_nonzero_final,
    }
    return balanced, stats


def print_window_balance_stats(stats: dict[str, int | float], tolerance: float = 0.05) -> None:
    print(
        "Window balance: "
        f"all-zero start={stats['n_zero_start']}, "
        f"non-zero start={stats['n_nonzero_start']}, "
        f"dropped={stats['n_dropped']}, "
        f"remaining={stats['n_remaining']} "
        f"(zero={stats['n_zero_remaining']}, non-zero={stats['n_nonzero_remaining']})"
    )
    print(
        f"  Final distribution: p(all-zero)={stats['p_zero_final']:.3f}, "
        f"p(>=1 one)={stats['p_nonzero_final']:.3f} "
        f"(target ~0.50 +/- {tolerance:.2f})"
    )


class WindowedCarVisibleDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y_visible: torch.Tensor,
        y_speed: torch.Tensor,
        mask: torch.Tensor,
        window_size: int = 16,
        stride: int = 2,
        index: list[tuple[int, int]] | None = None,
    ):
        assert x.dim() == 3 and x.size(-1) == len(FEATURE_COLS)
        assert y_visible.dim() == 2 and mask.dim() == 2 and x.shape[:2] == y_visible.shape == mask.shape
        assert y_speed.dim() == 3 and y_speed.shape == (*x.shape[:2], 1)
        self.x = x
        self.y_visible = y_visible
        self.y_speed = y_speed
        self.mask = mask
        self.window_size = int(window_size)
        self.stride = int(stride)
        if index is not None:
            self.index = list(index)
        else:
            self.index = build_window_index(mask, self.window_size, self.stride)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_id, start = self.index[idx]
        w = self.window_size
        n_valid = int(self.mask[seq_id].sum().item())
        n_win = max(0, min(w, n_valid - start))

        xw = torch.zeros((w, len(FEATURE_COLS)), dtype=self.x.dtype, device=self.x.device)
        yw_visible = torch.zeros((w,), dtype=torch.bool, device=self.y_visible.device)
        yw_speed = torch.zeros(
            (w, 1),
            dtype=self.y_speed.dtype,
            device=self.y_speed.device,
        )
        mw = torch.zeros((w,), dtype=torch.bool, device=self.mask.device)

        if n_win > 0:
            xw[:n_win] = self.x[seq_id, start : start + n_win]
            yw_visible[:n_win] = self.y_visible[seq_id, start : start + n_win]
            yw_speed[:n_win] = self.y_speed[seq_id, start : start + n_win]
            mw[:n_win] = True
        return xw, yw_visible, yw_speed, mw


def visibility_bce_loss(
    logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: float = 1.0
) -> torch.Tensor:
    logits = logit.masked_select(mask)
    targets = target.masked_select(mask).float()
    if logits.numel() == 0:
        return torch.zeros((), device=logit.device)
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.as_tensor(pos_weight, device=logit.device)
    )


def speed_regression_loss(
    pred_speed: torch.Tensor,
    target_speed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    assert pred_speed.shape == target_speed.shape, (
        f"Predicted/target speed shapes must match, got {pred_speed.shape} vs {target_speed.shape}"
    )
    pred = pred_speed.masked_select(mask.unsqueeze(-1))
    target = target_speed.masked_select(mask.unsqueeze(-1))
    if pred.numel() == 0:
        return torch.zeros((), device=pred_speed.device)
    return F.smooth_l1_loss(pred, target)


@torch.no_grad()
def visibility_metrics(
    logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, thr: float = 0.9
) -> dict[str, float]:
    valid_prob = torch.sigmoid(logit).masked_select(mask).cpu().numpy()
    valid_target = target.masked_select(mask).cpu().numpy().astype(np.bool_)
    return threshold_metric_from_probs(valid_prob, valid_target, thr)


def metrics_from_binary_pred(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(np.bool_)
    tgt = target.astype(np.bool_)
    tp = float(np.logical_and(pred, tgt).sum())
    fp = float(np.logical_and(pred, np.logical_not(tgt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), tgt).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(tgt)).sum())
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    return {
        "f1": float(f1),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "pred_visible": float(tp + fp),
        "true_visible": float(tp + fn),
        "false_visible": float(fp),
    }


def threshold_metric_from_probs(
    prob: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, float]:
    return metrics_from_binary_pred(prob > float(threshold), target)


def postprocess_probs(
    probs: np.ndarray,
    smooth_alpha: float = 1.0,
    hyst_on_thr: float = 0.90,
    hyst_off_thr: float = 0.60,
) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return np.array([], dtype=np.bool_)

    if smooth_alpha != 1.0:
        smoothed = np.empty_like(probs)
        smoothed[0] = probs[0]
        for t in range(1, probs.size):
            smoothed[t] = smooth_alpha * probs[t] + (1.0 - smooth_alpha) * smoothed[t - 1]
    else:
        smoothed = probs

    if hyst_on_thr == hyst_off_thr:
        return smoothed > float(hyst_on_thr)

    pred = np.zeros(probs.size, dtype=np.bool_)
    state = False
    for t in range(probs.size):
        if not state:
            if smoothed[t] >= hyst_on_thr:
                state = True
        elif smoothed[t] <= hyst_off_thr:
            state = False
        pred[t] = state
    return pred


def threshold_metric_sweep_from_probs(
    prob: np.ndarray, target: np.ndarray, thresholds: tuple[float, ...]
) -> dict[float, dict[str, float]]:
    return {
        float(threshold): threshold_metric_from_probs(prob, target, float(threshold))
        for threshold in thresholds
    }


def pr_auc_score(prob: np.ndarray, target: np.ndarray) -> float:
    if prob.size == 0 or target.size == 0:
        return 0.0
    y_true = target.astype(np.bool_)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(-prob, kind="mergesort")
    y_sorted = y_true[order].astype(np.float64)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    # NumPy < 2.0 compatibility (Python 3.7 envs often use older NumPy).
    return float(np.trapz(precision, recall))


def speed_metrics(
    pred_speed: torch.Tensor,
    target_speed: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    assert pred_speed.shape == target_speed.shape, (
        f"Predicted/target speed shapes must match, got {pred_speed.shape} vs {target_speed.shape}"
    )
    pred = pred_speed.masked_select(mask.unsqueeze(-1))
    target = target_speed.masked_select(mask.unsqueeze(-1))
    if pred.numel() == 0:
        return {"speed_mae": 0.0, "speed_rmse": 0.0}

    abs_err = (pred - target).abs()
    sq_err = (pred - target) ** 2
    return {
        "speed_mae": float(abs_err.mean().item()),
        "speed_rmse": float(torch.sqrt(sq_err.mean()).item()),
    }


def speed_metrics_from_flat(
    pred_speed: torch.Tensor,
    target_speed: torch.Tensor,
) -> dict[str, float]:
    assert pred_speed.shape == target_speed.shape, (
        f"Predicted/target speed shapes must match, got {pred_speed.shape} vs {target_speed.shape}"
    )
    if pred_speed.numel() == 0:
        return {"speed_mae": 0.0, "speed_rmse": 0.0}

    abs_err = (pred_speed - target_speed).abs()
    sq_err = (pred_speed - target_speed) ** 2
    return {
        "speed_mae": float(abs_err.mean().item()),
        "speed_rmse": float(torch.sqrt(sq_err.mean()).item()),
    }


class RunningSpeedMetrics:
    """Online MAE/RMSE accumulation without storing per-frame predictions."""

    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred_speed: torch.Tensor, target_speed: torch.Tensor) -> None:
        assert pred_speed.shape == target_speed.shape, (
            f"Predicted/target speed shapes must match, got {pred_speed.shape} vs {target_speed.shape}"
        )
        if pred_speed.numel() == 0:
            return
        pred = pred_speed.detach().cpu().double().reshape(-1)
        target = target_speed.detach().cpu().double().reshape(-1)
        self.abs_sum += float((pred - target).abs().sum().item())
        self.sq_sum += float(((pred - target) ** 2).sum().item())
        self.count += int(pred.numel())

    def finalize(self) -> dict[str, float]:
        if self.count == 0:
            return {"speed_mae": 0.0, "speed_rmse": 0.0}

        abs_mean = self.abs_sum / self.count
        sq_mean = self.sq_sum / self.count
        return {
            "speed_mae": float(abs_mean),
            "speed_rmse": float(np.sqrt(sq_mean)),
        }


def _trim_to_active(
    xb: torch.Tensor,
    yb_visible: torch.Tensor,
    yb_speed: torch.Tensor,
    mb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    true_t = max(int(mb.sum(dim=1).max().item()), 1)
    return xb[:, :true_t], yb_visible[:, :true_t], yb_speed[:, :true_t], mb[:, :true_t]


def run_epoch(
    model: CarVisibleLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    pos_weight: float,
    speed_weight: float,
    target_speed_mean: torch.Tensor,
    target_speed_std: torch.Tensor,
    thr: float = 0.9,
    metric_thresholds: tuple[float, ...] | None = None,
    epoch_idx: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train(mode=train_mode)
    total_loss = 0.0
    total_visibility_loss = 0.0
    total_speed_loss = 0.0
    all_prob: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_pred_speed: list[torch.Tensor] = []
    all_target_speed: list[torch.Tensor] = []
    running_speed = RunningSpeedMetrics()
    n_batches = 0

    phase = "Train" if train_mode else "Test "
    epoch_tag = ""
    if epoch_idx is not None and total_epochs is not None:
        epoch_tag = f" [{epoch_idx:03d}/{total_epochs}]"
    loader_iter = loader
    if show_progress:
        loader_iter = tqdm(loader, desc=f"{phase}{epoch_tag}", leave=False, dynamic_ncols=True)

    target_speed_mean = target_speed_mean.to(device)
    target_speed_std = target_speed_std.to(device)

    for xb, yb_visible, yb_speed, mb in loader_iter:
        xb = xb.to(device)
        yb_visible = yb_visible.to(device)
        yb_speed = yb_speed.to(device)
        mb = mb.to(device)
        xb, yb_visible, yb_speed, mb = _trim_to_active(xb, yb_visible, yb_speed, mb)
        assert yb_speed.shape[-1] == 1, f"Expected speed target dim 1, got {yb_speed.shape}"
        assert torch.isfinite(yb_speed).all(), "All speed targets must be finite."
        yb_speed_norm = (yb_speed - target_speed_mean) / target_speed_std

        with torch.set_grad_enabled(train_mode):
            out = model(xb, mb)
            visibility_loss = visibility_bce_loss(
                out["logit"], yb_visible, mb, pos_weight=pos_weight
            )
            speed_loss = speed_regression_loss(
                out["speed"], yb_speed_norm, mb
            )
            loss = visibility_loss + speed_weight * speed_loss
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        pred_speed_full = out["speed"].detach() * target_speed_std + target_speed_mean
        assert pred_speed_full.shape == yb_speed.shape, (
            f"Predicted/target speed shapes must match, got {pred_speed_full.shape} vs {yb_speed.shape}"
        )
        pred_speed = pred_speed_full.masked_select(mb.unsqueeze(-1))
        target_speed = yb_speed.detach().masked_select(mb.unsqueeze(-1))
        running_speed.update(pred_speed, target_speed)

        if not train_mode:
            valid_prob = torch.sigmoid(out["logit"].detach()).masked_select(mb).cpu().numpy()
            valid_target = yb_visible.masked_select(mb).cpu().numpy().astype(np.bool_)
            all_prob.append(valid_prob)
            all_target.append(valid_target)
            all_pred_speed.append(pred_speed.cpu())
            all_target_speed.append(target_speed.cpu())

        total_loss += float(loss.item())
        total_visibility_loss += float(visibility_loss.item())
        total_speed_loss += float(speed_loss.item())
        n_batches += 1

        if show_progress and hasattr(loader_iter, "set_postfix"):
            loader_iter.set_postfix(
                loss=f"{(total_loss / max(1, n_batches)):.4f}",
            )

    denom = max(1, n_batches)
    speed_stats = running_speed.finalize()
    if train_mode:
        return {
            "loss": total_loss / denom,
            "visibility_loss": total_visibility_loss / denom,
            "speed_loss": total_speed_loss / denom,
            "f1": 0.0,
            "acc": 0.0,
            "pr_auc": 0.0,
            "pred_visible": 0.0,
            "true_visible": 0.0,
            "false_visible": 0.0,
            "threshold_metrics": {},
            **speed_stats,
        }

    all_prob_concat = np.concatenate(all_prob) if all_prob else np.array([], dtype=np.float64)
    all_target_concat = np.concatenate(all_target) if all_target else np.array([], dtype=np.bool_)
    pred_speed_cat = torch.cat(all_pred_speed, dim=0) if all_pred_speed else torch.zeros((0,))
    target_speed_cat = torch.cat(all_target_speed, dim=0) if all_target_speed else torch.zeros((0,))
    pr_auc = pr_auc_score(
        all_prob_concat,
        all_target_concat,
    )
    thresholds = metric_thresholds if metric_thresholds is not None else (thr,)
    threshold_metrics = threshold_metric_sweep_from_probs(
        all_prob_concat, all_target_concat, thresholds
    )

    selected = threshold_metrics.get(float(thr), threshold_metrics[thresholds[0]])
    speed_stats = speed_metrics_from_flat(pred_speed_cat, target_speed_cat)
    return {
        "loss": total_loss / denom,
        "visibility_loss": total_visibility_loss / denom,
        "speed_loss": total_speed_loss / denom,
        "f1": selected["f1"],
        "acc": selected["acc"],
        "pr_auc": pr_auc,
        "pred_visible": selected["pred_visible"],
        "true_visible": selected["true_visible"],
        "false_visible": selected["false_visible"],
        "threshold_metrics": threshold_metrics,
        **speed_stats,
    }


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    drop: float = 0.1
    lr: float = 1e-3
    epochs: int = 150
    batch_size: int = 512
    window_size: int = 512 # best run
    stride: int = 1 # best run
    # Random window split fractions (must sum to 1.0). Track best-by-val; report final metrics on test.
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    split_seed: int = 42
    early_stop_patience: int = 15
    stop_at_epoch: int | None = None
    metric_threshold: float = 0.9
    metric_threshold_sweep: tuple[float, ...] = (0.5, 0.7, 0.9)
    min_frames: int = 100
    speed_weight: float = 1.0


def y_path_for_x(x_path: Path) -> Path:
    return Y_DATA_DIR / f"{x_path.stem}{Y_SUFFIX}"


def load_x_sequence(x_csv: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    df_x = pd.read_csv(x_csv, usecols=X_LOAD_COLS)
    x = build_feature_matrix(df_x)
    x_t = torch.from_numpy(x).unsqueeze(0)
    mask_t = torch.ones((1, x_t.shape[1]), dtype=torch.bool)
    return x_t, mask_t


def _parse_bool_series(series: pd.Series) -> pd.Series:
    bool_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(bool_map)


def load_truth_targets(y_csv: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    y_csv = Path(y_csv)
    header_cols = pd.read_csv(y_csv, nrows=0).columns.tolist()
    assert "speed" in header_cols, f"'speed' must exist in truth CSV: {y_csv}"
    df_y = pd.read_csv(y_csv, usecols=["car_visible", "speed"])
    visible = _parse_bool_series(df_y["car_visible"])
    speed = pd.to_numeric(df_y["speed"], errors="coerce")
    valid_rows = visible.notna() & speed.notna()
    df_clean = pd.DataFrame({"car_visible": visible, "speed": speed})
    df_clean = df_clean.loc[valid_rows].reset_index(drop=True)
    y_visible = df_clean["car_visible"].astype(bool).to_numpy()
    y_speed = df_clean[["speed"]].to_numpy(dtype=np.float32)
    assert y_speed.ndim == 2 and y_speed.shape[1] == 1, f"Expected speed shape [frames, 1], got {y_speed.shape}"
    assert np.isfinite(y_speed).all(), "All speed targets must be finite."
    return torch.from_numpy(y_visible), torch.from_numpy(y_speed)


def load_single_sequence(
    x_csv: str | Path,
    y_csv: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_t, mask_t = load_x_sequence(x_csv)
    y_visible_t, y_speed_t = load_truth_targets(y_csv)
    n_frames = min(x_t.shape[1], y_visible_t.shape[0], y_speed_t.shape[0])
    x_t = x_t[:, :n_frames]
    mask_t = mask_t[:, :n_frames]
    y_visible_t = y_visible_t[:n_frames].unsqueeze(0)
    y_speed_t = y_speed_t[:n_frames].unsqueeze(0)
    assert y_speed_t.shape == (*x_t.shape[:2], 1), (
        f"Expected speed tensor shape {(*x_t.shape[:2], 1)}, got {y_speed_t.shape}"
    )
    return x_t, y_visible_t, y_speed_t, mask_t


def load_norm_checkpoint(path: str | Path) -> dict:
    norm_state = torch.load(Path(path), map_location="cpu")
    feature_cols = norm_state.get("feature_cols", FEATURE_COLS)
    if list(feature_cols) != FEATURE_COLS:
        raise ValueError(
            "Feature columns mismatch between checkpoint and script. "
            f"Checkpoint: {feature_cols} | Script: {FEATURE_COLS}"
        )
    if "target_speed_mean" not in norm_state:
        norm_state["target_speed_mean"] = torch.zeros((), dtype=torch.float32)
    if "target_speed_std" not in norm_state:
        norm_state["target_speed_std"] = torch.ones((), dtype=torch.float32)
    return norm_state


def build_car_visible_lstm(config: TrainConfig, device: torch.device) -> CarVisibleLSTM:
    return CarVisibleLSTM(
        input_dim=len(FEATURE_COLS),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        p_drop=config.drop,
        bidirectional=config.bidirectional,
    ).to(device)


def load_car_visible_lstm(
    model_path: str | Path,
    config: TrainConfig,
    device: torch.device,
) -> CarVisibleLSTM:
    model = build_car_visible_lstm(config, device)
    model.load_state_dict(torch.load(Path(model_path), map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_car_visible_and_speed(
    model: CarVisibleLSTM,
    x: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_speed_mean: torch.Tensor,
    target_speed_std: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_norm = apply_normalization(x, mask, mean, std)
    out = model(x_norm.to(device), mask.to(device))
    logit = out["logit"].cpu()
    speed = (out["speed"].cpu() * target_speed_std.cpu()) + target_speed_mean.cpu()
    assert speed.shape[-1] == 1, f"Expected speed prediction dim 1, got {speed.shape}"
    return logit, torch.sigmoid(logit), speed


def frame_metrics_from_logits(
    logit: torch.Tensor,
    target: torch.Tensor | None,
    mask: torch.Tensor,
    pos_weight: float,
    metric_threshold: float,
    metric_threshold_sweep: tuple[float, ...],
    smooth_alpha: float = 1.0,
    hyst_on_thr: float = 0.90,
    hyst_off_thr: float = 0.60,
) -> dict[str, float | dict[float, dict[str, float]] | bool]:
    valid_prob = torch.sigmoid(logit).masked_select(mask).cpu().numpy()
    loss = 0.0
    if target is not None:
        loss = float(
            visibility_bce_loss(logit, target, mask, pos_weight=pos_weight).item()
        )
        valid_target = target.masked_select(mask).cpu().numpy().astype(np.bool_)
    else:
        valid_target = np.array([], dtype=np.bool_)

    pr_auc = pr_auc_score(valid_prob, valid_target) if target is not None else 0.0
    use_postprocess = hyst_on_thr != hyst_off_thr

    threshold_metrics: dict[float, dict[str, float]] = {}
    if target is not None:
        threshold_metrics = threshold_metric_sweep_from_probs(
            valid_prob, valid_target, metric_threshold_sweep
        )

    postprocess_metrics: dict[str, float] = {}
    if use_postprocess and target is not None:
        postprocess_metrics = metrics_from_binary_pred(
            postprocess_probs(valid_prob, smooth_alpha, hyst_on_thr, hyst_off_thr),
            valid_target,
        )

    if use_postprocess and postprocess_metrics:
        selected = postprocess_metrics
    elif threshold_metrics:
        selected = threshold_metrics.get(float(metric_threshold), {})
    else:
        selected = {}

    return {
        "loss": loss,
        "f1": float(selected.get("f1", 0.0)),
        "acc": float(selected.get("acc", 0.0)),
        "pr_auc": pr_auc,
        "pred_visible": float(selected.get("pred_visible", 0.0)),
        "true_visible": float(selected.get("true_visible", 0.0)),
        "false_visible": float(selected.get("false_visible", 0.0)),
        "threshold_metrics": threshold_metrics,
        "use_postprocess": use_postprocess,
        "postprocess_metrics": postprocess_metrics,
    }


def print_frame_metrics(
    stats: dict[str, float | dict[float, dict[str, float]] | bool],
    metric_threshold_sweep: tuple[float, ...],
    prefix: str = "Inference",
    smooth_alpha: float = 1.0,
    hyst_on_thr: float = 0.90,
    hyst_off_thr: float = 0.60,
) -> None:
    print(f"  {prefix}  loss={stats['loss']:.4f} PR-AUC={stats['pr_auc']:.3f}")

    threshold_metrics = stats.get("threshold_metrics")
    if isinstance(threshold_metrics, dict) and threshold_metrics:
        print("  pre-postprocess (threshold):")
        for threshold in metric_threshold_sweep:
            m = threshold_metrics.get(float(threshold))
            if m is None:
                continue
            print(
                f"    {prefix}@thr={threshold:.2f} F1={m['f1']:.3f} Acc={m['acc']:.3f} "
                f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} "
                f"pred_vis={m['pred_visible']:.0f} true_vis={m['true_visible']:.0f} "
                f"false_vis={m['false_visible']:.0f}"
            )

    if stats.get("use_postprocess"):
        print(
            f"  post-postprocess: smooth_alpha={smooth_alpha:.3f} "
            f"hyst_on_thr={hyst_on_thr:.2f} hyst_off_thr={hyst_off_thr:.2f}"
        )
        m = stats.get("postprocess_metrics")
        if isinstance(m, dict) and m:
            print(
                f"    {prefix} F1={m['f1']:.3f} Acc={m['acc']:.3f} "
                f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} "
                f"pred_vis={m['pred_visible']:.0f} true_vis={m['true_visible']:.0f} "
                f"false_vis={m['false_visible']:.0f}"
            )


def load_sequences_from_dirs(
    x_dir: Path = X_DATA_DIR,
    y_dir: Path = Y_DATA_DIR,
    min_frames: int = TrainConfig().min_frames,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_paths = sorted(x_dir.glob("*.csv"))
    if not x_paths:
        raise ValueError(f"No CSV files found under {x_dir}")

    x_list: list[torch.Tensor] = []
    y_visible_list: list[torch.Tensor] = []
    y_speed_list: list[torch.Tensor] = []
    m_list: list[torch.Tensor] = []
    max_len = 0

    for x_path in tqdm(x_paths, desc="Loading CSV pairs"):
        y_path = y_path_for_x(x_path)
        x_i, y_visible_i, y_speed_i, m_i = load_single_sequence(x_path, y_path)
        x_i = x_i.squeeze(0)
        y_visible_i = y_visible_i.squeeze(0)
        y_speed_i = y_speed_i.squeeze(0)
        m_i = m_i.squeeze(0)
        if int(m_i.sum().item()) < min_frames:
            continue
        x_list.append(x_i)
        y_visible_list.append(y_visible_i)
        y_speed_list.append(y_speed_i)
        m_list.append(m_i)
        max_len = max(max_len, int(x_i.shape[0]))

    n_seq = len(x_list)
    print(f"Kept {n_seq}/{len(x_paths)} sequences with >= {min_frames} frames")
    if n_seq == 0:
        raise ValueError(f"No sequences left after filtering with >= {min_frames} frames")

    n_feat = len(FEATURE_COLS)
    x = torch.zeros((n_seq, max_len, n_feat), dtype=torch.float32)
    y_visible = torch.zeros((n_seq, max_len), dtype=torch.bool)
    y_speed = torch.zeros((n_seq, max_len, 1), dtype=torch.float32)
    mask = torch.zeros((n_seq, max_len), dtype=torch.bool)

    for i in range(n_seq):
        n = x_list[i].shape[0]
        x[i, :n] = x_list[i]
        y_visible[i, :n] = y_visible_list[i]
        y_speed[i, :n] = y_speed_list[i]
        mask[i, :n] = m_list[i]

    assert y_speed.shape == (*x.shape[:2], 1), f"Expected speed tensor shape {(*x.shape[:2], 1)}, got {y_speed.shape}"
    return x, y_visible, y_speed, mask


def compute_normalization(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask3 = mask.unsqueeze(-1)
    valid = mask3.to(x.dtype)
    cnt = valid.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (x * valid).sum(dim=(0, 1)) / cnt
    sum_sq = (((x - mean) * valid) ** 2).sum(dim=(0, 1))
    std = (sum_sq / cnt).clamp_min(1e-12).sqrt()
    return mean, std


def compute_target_speed_normalization_from_train_windows(
    y_speed: torch.Tensor,
    dataset: WindowedCarVisibleDataset,
    train_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_speeds: list[torch.Tensor] = []
    for ds_idx in train_indices:
        seq_id, start = dataset.index[ds_idx]
        n_valid = int(dataset.mask[seq_id].sum().item())
        n_win = max(0, min(dataset.window_size, n_valid - start))
        if n_win == 0:
            continue
        valid_speeds.append(y_speed[seq_id, start : start + n_win].reshape(-1))

    if valid_speeds:
        train_speed = torch.cat(valid_speeds, dim=0).float()
    else:
        train_speed = torch.zeros((1,), dtype=torch.float32)
    assert torch.isfinite(train_speed).all(), "All speed targets must be finite."
    mean = train_speed.mean()
    std = train_speed.std(unbiased=False).clamp_min(1e-12)
    return mean, std


def apply_normalization(x: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mask3 = mask.unsqueeze(-1)
    x_norm = (x - mean) / std
    return torch.where(mask3, x_norm, x)


def compute_pos_weight_from_windows(
    y: torch.Tensor,
    mask: torch.Tensor,
    window_index: list[tuple[int, int]],
    window_size: int,
) -> float:
    n_pos = 0.0
    n_neg = 0.0
    window_size = int(window_size)
    for seq_id, start in tqdm(window_index, desc="Computing pos_weight", dynamic_ncols=True):
        n_valid = int(mask[seq_id].sum().item())
        n_win = max(0, min(window_size, n_valid - start))
        if n_win == 0:
            continue
        segment = y[seq_id, start : start + n_win]
        n_pos += float(segment.sum().item())
        n_neg += float(n_win - segment.sum().item())
    if n_pos < 1.0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def save_norm_checkpoint(
    path: Path,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_speed_mean: torch.Tensor,
    target_speed_std: torch.Tensor,
    feature_cols: list[str],
    pos_weight: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean.cpu(),
            "std": std.cpu(),
            "target_speed_mean": target_speed_mean.cpu(),
            "target_speed_std": target_speed_std.cpu(),
            "feature_cols": feature_cols,
            "target_cols": TRUTH_COLS,
            "pos_weight": pos_weight,
        },
        path,
    )


def train_car_visible_lstm(
    config: TrainConfig = TrainConfig(),
    norm_out: Path = NORM_CHECKPOINT,
    model_out: Path = MODEL_CHECKPOINT,
) -> CarVisibleLSTM:
    x, y_visible, y_speed, mask = load_sequences_from_dirs(min_frames=config.min_frames)

    mean, std = compute_normalization(x, mask)
    x = apply_normalization(x, mask, mean, std)

    print("CUDA is available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_car_visible_lstm(config, device)

    all_window_index = build_window_index(mask, config.window_size, config.stride)
    print(f"Precomputed {len(all_window_index)} training windows before balancing")
    balanced_index, balance_stats = balance_window_index(
        all_window_index,
        y_visible,
        mask,
        window_size=config.window_size,
    )
    print_window_balance_stats(balance_stats, tolerance=0.05)

    pos_weight = compute_pos_weight_from_windows(
        y_visible, mask, balanced_index, window_size=config.window_size
    )
    ds = WindowedCarVisibleDataset(
        x,
        y_visible,
        y_speed,
        mask,
        window_size=config.window_size,
        stride=config.stride,
        index=balanced_index,
    )
    n_total = len(ds)
    print(
        "Training hyperparameters: "
        f"window_size={config.window_size}, stride={config.stride}, batch_size={config.batch_size}, "
        f"epochs={config.epochs}, "
        f"train/val/test={config.train_split:.2f}/{config.val_split:.2f}/{config.test_split:.2f}, "
        f"split_seed={config.split_seed}, lr={config.lr:.2e}, "
        f"hidden_dim={config.hidden_dim}, num_layers={config.num_layers}, drop={config.drop}, "
        f"bidirectional={config.bidirectional}, metric_threshold={config.metric_threshold}, "
        f"metric_threshold_sweep={config.metric_threshold_sweep}, "
        f"early_stop_patience={config.early_stop_patience}, stop_at_epoch={config.stop_at_epoch}, "
        f"min_frames={config.min_frames}, pos_weight={pos_weight:.2f}, "
        f"speed_weight={config.speed_weight:.2f}"
    )
    print(f"Total windows available for training: {n_total}")
    split_sum = float(config.train_split + config.val_split + config.test_split)
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test splits must sum to 1.0, got {split_sum:.6f} "
            f"({config.train_split}, {config.val_split}, {config.test_split})"
        )
    n_train = max(1, int(round(config.train_split * n_total)))
    n_val = max(1, int(round(config.val_split * n_total)))
    n_test = n_total - n_train - n_val
    if n_test < 1:
        # Steal from train so every split is non-empty.
        n_train = max(1, n_train - (1 - n_test))
        n_test = n_total - n_train - n_val
    if n_train + n_val + n_test != n_total or min(n_train, n_val, n_test) < 1:
        raise ValueError(
            f"Failed to build non-empty train/val/test split from n_total={n_total}: "
            f"n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )
    train_ds, val_ds, test_ds = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.split_seed),
    )
    print(
        f"Split windows (seed={config.split_seed}): "
        f"train={n_train} ({n_train / n_total:.1%}), "
        f"val={n_val} ({n_val / n_total:.1%}), "
        f"test={n_test} ({n_test / n_total:.1%})"
    )
    train_indices = list(train_ds.indices) if hasattr(train_ds, "indices") else list(range(len(train_ds)))
    target_speed_mean, target_speed_std = compute_target_speed_normalization_from_train_windows(
        y_speed, ds, train_indices
    )
    save_norm_checkpoint(
        norm_out,
        mean,
        std,
        target_speed_mean,
        target_speed_std,
        FEATURE_COLS,
        pos_weight,
    )
    print(
        f"Saved normalization checkpoint to {norm_out} (pos_weight={pos_weight:.2f}, "
        f"target_speed_mean={float(target_speed_mean):.4f}, target_speed_std={float(target_speed_std):.4f})"
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    best_val_speed_rmse = float("inf")
    epochs_since_last_best = 0
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    max_epochs = config.epochs
    if config.stop_at_epoch is not None:
        max_epochs = min(config.epochs, config.stop_at_epoch)

    for epoch in range(1, max_epochs + 1):
        print(f"EPOCH [{epoch:03d}/{max_epochs}]")
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_mode=True,
            pos_weight=pos_weight,
            speed_weight=config.speed_weight,
            target_speed_mean=target_speed_mean,
            target_speed_std=target_speed_std,
            thr=config.metric_threshold,
            epoch_idx=epoch,
            total_epochs=max_epochs,
            show_progress=True,
        )
        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            train_mode=False,
            pos_weight=pos_weight,
            speed_weight=config.speed_weight,
            target_speed_mean=target_speed_mean,
            target_speed_std=target_speed_std,
            thr=config.metric_threshold,
            metric_thresholds=config.metric_threshold_sweep,
            epoch_idx=epoch,
            total_epochs=max_epochs,
            show_progress=True,
        )
        print(
            f"  Train loss={train_stats['loss']:.4f} "
            f"(vis={train_stats['visibility_loss']:.4f}, speed_loss={train_stats['speed_loss']:.4f}) "
            f"speed_MAE={train_stats['speed_mae']:.3f} "
            f"speed_RMSE={train_stats['speed_rmse']:.3f}"
        )
        print(
            f"  Val   loss={val_stats['loss']:.4f} "
            f"(vis={val_stats['visibility_loss']:.4f}, speed_loss={val_stats['speed_loss']:.4f}) "
            f"PR-AUC={val_stats['pr_auc']:.3f} speed_MAE={val_stats['speed_mae']:.3f} "
            f"speed_RMSE={val_stats['speed_rmse']:.3f}"
        )
        for threshold in config.metric_threshold_sweep:
            m = val_stats["threshold_metrics"][float(threshold)]
            print(
                f"    Val@thr={threshold:.2f} F1={m['f1']:.3f} Acc={m['acc']:.3f} "
                f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} "
                f"pred_vis={m['pred_visible']:.0f} true_vis={m['true_visible']:.0f} "
                f"false_vis={m['false_visible']:.0f}"
            )

        if val_stats["speed_rmse"] < best_val_speed_rmse:
            best_val_speed_rmse = val_stats["speed_rmse"]
            torch.save(model.state_dict(), model_out)
            print(
                f"  Saved new best checkpoint to {model_out} "
                f"(val speed_RMSE={best_val_speed_rmse:.4f}, val F1={val_stats['f1']:.4f})"
            )
            epochs_since_last_best = 0
        else:
            epochs_since_last_best += 1

        if epochs_since_last_best >= config.early_stop_patience:
            print(
                f"Stopping early after {config.early_stop_patience} epochs "
                "without val speed_RMSE improvement."
            )
            break

        if config.stop_at_epoch is not None and epoch >= config.stop_at_epoch:
            torch.save(model.state_dict(), model_out)
            print(
                f"Stopping at epoch {config.stop_at_epoch}; saved checkpoint to {model_out} "
                f"(val speed_RMSE={val_stats['speed_rmse']:.4f}, val F1={val_stats['f1']:.4f})"
            )
            break

    # Final metrics: reload best-by-val checkpoint and evaluate once on held-out test.
    model.load_state_dict(torch.load(model_out, map_location=device))
    test_stats = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=optimizer,
        device=device,
        train_mode=False,
        pos_weight=pos_weight,
        speed_weight=config.speed_weight,
        target_speed_mean=target_speed_mean,
        target_speed_std=target_speed_std,
        thr=config.metric_threshold,
        metric_thresholds=config.metric_threshold_sweep,
        show_progress=True,
    )
    print(
        f"  Test  (best-by-val) loss={test_stats['loss']:.4f} "
        f"(vis={test_stats['visibility_loss']:.4f}, speed_loss={test_stats['speed_loss']:.4f}) "
        f"PR-AUC={test_stats['pr_auc']:.3f} speed_MAE={test_stats['speed_mae']:.3f} "
        f"speed_RMSE={test_stats['speed_rmse']:.3f}"
    )
    for threshold in config.metric_threshold_sweep:
        m = test_stats["threshold_metrics"][float(threshold)]
        print(
            f"    Test@thr={threshold:.2f} F1={m['f1']:.3f} Acc={m['acc']:.3f} "
            f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} "
            f"pred_vis={m['pred_visible']:.0f} true_vis={m['true_visible']:.0f} "
            f"false_vis={m['false_visible']:.0f}"
        )

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train car-visible LSTM model.")
    parser.add_argument("--window", type=int, default=TrainConfig.window_size)
    parser.add_argument("--stride", type=int, default=TrainConfig.stride)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--hidden", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--dropout", type=float, default=TrainConfig.drop)
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Enable bidirectional LSTM (default: disabled).",
    )
    parser.add_argument(
        "--stop-at-epoch",
        type=int,
        default=None,
        metavar="N",
        help="Stop training after epoch N and save the checkpoint (default: train until epochs or patience).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        window_size=args.window,
        stride=args.stride,
        batch_size=args.batch_size,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        drop=args.dropout,
        bidirectional=args.bidirectional,
        stop_at_epoch=args.stop_at_epoch,
    )
    train_car_visible_lstm(config=config)
