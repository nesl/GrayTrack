from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from frame_boundary_model import PacketSegformer, SinusoidalPositionalEncoding, load_sequences_from_root

# FIXME: we should be using per-sample mean rather than global mean/std.

# Hardcode these paths as requested.
CSV_PATH = Path("/media/ubuntu/research/tmp/2026_03_04_11_16_26_119_camera_12_20260505_134142_977824248.csv")
CHECKPOINT_PATH = Path("/home/ubuntu/GrayAssets_CityScale/scripts/mininet/frame_boundary_model/best_packetsegformer.pkl")
TRAIN_DATA_ROOT = Path("/media/ubuntu/research/tmp")
STATS_CACHE_PATH = Path("/home/ubuntu/GrayAssets_CityScale/scripts/mininet/frame_boundary_model/train_norm_stats.pt")
PREDICTION_THRESHOLD = 0.3
WINDOW_SIZE = 250
WINDOW_STRIDE = 25

REQUIRED_COLUMNS = [
    "frame.number",
    "frame.time_epoch",
    "frame.time_relative",
    "frame.len",
    "rtp.seq",
    "rtp.timestamp",
    "rtp.marker",
]


def marker_series_to_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "t", "yes"])
    )


def load_csv_for_inference(csv_path: Path) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS).copy()
    for col in ["frame.time_relative", "frame.len"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["frame.time_relative", "frame.len"]).sort_values(
        "frame.time_relative", kind="mergesort"
    ).reset_index(drop=True)

    original_marker = marker_series_to_bool(df["rtp.marker"]).to_numpy(dtype=bool)
    if original_marker.size and not original_marker[-1]:
        original_marker[-1] = True
    time_rel = df["frame.time_relative"].to_numpy(dtype=np.float32)
    frame_len = df["frame.len"].to_numpy(dtype=np.float32)

    if len(df) < 2:
        raise ValueError("CSV must contain at least 2 rows after cleaning to compute dt features.")

    # Match training data construction: x[i] = [dt_i, frame_len_{i+1}], y[i] = marker_{i+1}.
    dt = time_rel[1:] - time_rel[:-1]
    x_np = np.stack([dt, frame_len[1:]], axis=1).astype(np.float32)
    y_np = original_marker[1:]

    x = torch.from_numpy(x_np).unsqueeze(0)
    mask = torch.ones((1, x.shape[1]), dtype=torch.bool)
    df_eval = df.iloc[1:].reset_index(drop=True).copy()
    df_eval["rtp.marker.original"] = y_np
    return df_eval, x, mask


def compute_training_normalization_stats(data_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    # Exact training-time normalization pipeline from frame_boundary_model.py.
    x_train, _, mask_train = load_sequences_from_root(data_root)
    mask3 = mask_train.unsqueeze(-1)
    valid = mask3.to(x_train.dtype)
    cnt = valid.sum(dim=(0, 1)).clamp_min(1.0)
    sum_x = (x_train * valid).sum(dim=(0, 1))
    mean = sum_x / cnt
    sum_sq = (((x_train - mean) * valid) ** 2).sum(dim=(0, 1))
    std = (sum_sq / cnt).clamp_min(1e-12).sqrt()
    return mean, std


def load_or_compute_training_normalization_stats(data_root: Path, cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        mean = payload["mean"]
        std = payload["std"]
        return mean, std

    mean, std = compute_training_normalization_stats(data_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"mean": mean.detach().cpu(), "std": std.detach().cpu(), "data_root": str(data_root)},
        cache_path,
    )
    return mean, std


def apply_normalization(x: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mask3 = mask.unsqueeze(-1)
    x_centered = (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    return torch.where(mask3, x_centered, x)


def load_model(checkpoint_path: Path, device: torch.device, seq_len: int) -> PacketSegformer:
    model = PacketSegformer(d_model=16, nhead=1, num_layers=4, dim_ff=16, p_drop=0.1).to(device)
    # Training used window size 250, but inference may be longer. Replace pos-enc buffer length.
    model.pos_enc = SinusoidalPositionalEncoding(d_model=16, max_len=max(250, seq_len)).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_in_windows(
    model: PacketSegformer,
    x: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    window_size: int,
    stride: int,
    window_batch_size: int = 64,
) -> np.ndarray:
    # x, mask are single-sequence tensors with shape [1, T, ...] and [1, T].
    t_total = int(mask.sum().item())
    if t_total <= 0:
        return np.zeros((0,), dtype=np.float32)

    starts = list(range(0, t_total, stride))
    last_start = max(0, t_total - window_size)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    starts = sorted(set(starts))

    prob_sum = np.zeros((t_total,), dtype=np.float64)
    prob_count = np.zeros((t_total,), dtype=np.int64)

    for batch_idx in range(0, len(starts), window_batch_size):
        batch_starts = starts[batch_idx : batch_idx + window_batch_size]
        b = len(batch_starts)
        xw = torch.full((b, window_size, 2), -1.0, dtype=x.dtype, device=device)
        mw = torch.zeros((b, window_size), dtype=torch.bool, device=device)
        n_wins = [max(0, min(window_size, t_total - st)) for st in batch_starts]

        for i, (st, n_win) in enumerate(zip(batch_starts, n_wins)):
            if n_win <= 0:
                continue
            xw[i, :n_win] = x[:, st : st + n_win]
            mw[i, :n_win] = True

        out = model(xw, mw)
        prob_batch = torch.sigmoid(out["b_logit"]).detach().cpu().numpy()

        for st, n_win, prob_w in zip(batch_starts, n_wins, prob_batch):
            if n_win <= 0:
                continue
            prob_sum[st : st + n_win] += prob_w[:n_win]
            prob_count[st : st + n_win] += 1

    prob_count = np.clip(prob_count, a_min=1, a_max=None)
    return (prob_sum / prob_count).astype(np.float32)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not TRAIN_DATA_ROOT.exists():
        raise FileNotFoundError(f"Training data root not found: {TRAIN_DATA_ROOT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_eval, x, mask = load_csv_for_inference(CSV_PATH)
    mean, std = load_or_compute_training_normalization_stats(TRAIN_DATA_ROOT, STATS_CACHE_PATH)
    x = apply_normalization(x, mask, mean, std)
    model = load_model(CHECKPOINT_PATH, device, seq_len=x.shape[1])

    x = x.to(device)
    mask = mask.to(device)

    prob = predict_in_windows(
        model=model,
        x=x,
        mask=mask,
        device=device,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
    )

    pred_marker = prob > PREDICTION_THRESHOLD
    orig_marker = df_eval["rtp.marker.original"].to_numpy(dtype=bool)

    print(f"Using CSV: {CSV_PATH}")
    print(f"Using checkpoint: {CHECKPOINT_PATH}")
    print(f"Normalization source root: {TRAIN_DATA_ROOT}")
    print(f"Normalization cache: {STATS_CACHE_PATH}")
    print(f"Window size: {WINDOW_SIZE}, stride: {WINDOW_STRIDE}")
    print(f"Rows evaluated: {len(df_eval)}")
    print(f"Prediction threshold: {PREDICTION_THRESHOLD}")
    print("-" * 80)

    matches = 0
    for i in range(len(df_eval)):
        original = bool(orig_marker[i])
        predicted = bool(pred_marker[i])
        if original == predicted:
            matches += 1
        print(
            f"idx={i:06d} frame.number={df_eval.at[i, 'frame.number']} "
            f"orig={int(original)} pred={int(predicted)} prob={prob[i]:.6f}"
        )

    total = len(df_eval)
    accuracy = matches / total if total else 0.0
    mismatches = total - matches

    print("-" * 80)
    print(f"Final comparison: matches={matches}, mismatches={mismatches}, accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
