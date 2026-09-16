from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset, random_split


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 250):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len, :]


class PacketSegformer(nn.Module):
    def __init__(self, d_model: int = 16, nhead: int = 1, num_layers: int = 4, dim_ff: int = 16, p_drop: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(5, d_model), nn.ReLU(), nn.Dropout(p_drop))
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        enc_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=p_drop,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=num_layers)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model // 2, 1),
        )

    def _prep_inputs(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        x_clipped = torch.where(mask.unsqueeze(-1), x, torch.tensor(0.0, device=device))
        t = x_clipped[:, :, 0]
        dt = torch.where(mask, torch.diff(t, dim=1, prepend=t[:, :1]), torch.zeros_like(t))
        dt = dt.clamp(min=1e-9)
        logdt = dt.log()
        pos_frac = (torch.arange(seq_len, device=device)[None, :].expand(batch_size, seq_len) / max(seq_len - 1, 1)).to(
            x.dtype
        )
        return torch.stack([t, x_clipped[:, :, 1], dt, logdt, pos_frac], dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _, seq_len, _ = x.shape
        feats = self._prep_inputs(x, mask)
        h = self.in_proj(feats)
        h = h + self.pos_enc(seq_len).to(h.dtype).to(h.device)
        z = self.encoder(h, src_key_padding_mask=~mask)
        b_logit = self.boundary_head(z).squeeze(-1)
        return {"b_logit": b_logit}


class WindowedPacketSeqDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, window_size: int = 250, stride: int = 25):
        assert x.dim() == 3 and x.size(-1) == 2
        assert y.dim() == 2 and mask.dim() == 2 and x.shape[:2] == y.shape == mask.shape
        self.x = x
        self.y = y
        self.mask = mask
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.index: list[tuple[int, int]] = []

        n_seq, _ = y.shape
        for seq_id in range(n_seq):
            n_valid = int(self.mask[seq_id].sum().item())
            if n_valid == 0:
                continue
            starts = list(range(0, n_valid, self.stride))
            last_start = max(0, n_valid - self.window_size)
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            for st in sorted(set(starts)):
                self.index.append((seq_id, st))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_id, start = self.index[idx]
        w = self.window_size
        n_valid = int(self.mask[seq_id].sum().item())
        n_win = max(0, min(w, n_valid - start))

        xw = torch.full((w, 2), -1.0, dtype=self.x.dtype, device=self.x.device)
        yw = torch.zeros((w,), dtype=torch.bool, device=self.y.device)
        mw = torch.zeros((w,), dtype=torch.bool, device=self.mask.device)

        if n_win > 0:
            xw[:n_win] = self.x[seq_id, start : start + n_win]
            yw[:n_win] = self.y[seq_id, start : start + n_win]
            mw[:n_win] = True
        return xw, yw, mw


def boundary_bce_loss(b_logit: torch.Tensor, b_target: torch.Tensor, mask: torch.Tensor, pos_weight: float = 18.0) -> torch.Tensor:
    logits = b_logit.masked_select(mask)
    targets = b_target.masked_select(mask).float()
    if logits.numel() == 0:
        return torch.zeros((), device=b_logit.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.as_tensor(pos_weight, device=b_logit.device))


def count_loss(b_logit: torch.Tensor, b_target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(b_logit) * mask.float()
    pred_count = p.sum(dim=1)
    true_count = b_target.float().mul(mask).sum(dim=1)
    return F.smooth_l1_loss(pred_count, true_count, beta=delta)


def l1_prob_loss(b_logit: torch.Tensor, mask: torch.Tensor, lam: float = 1e-1) -> torch.Tensor:
    p = torch.sigmoid(b_logit) * mask.float()
    return lam * p.sum() / mask.sum().clamp_min(1)


def boundary_loss_augmented(
    b_logit: torch.Tensor,
    b_target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 18.0,
    beta_count: float = 1.0,
    lam_spar: float = 1e-1,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = boundary_bce_loss(b_logit, b_target, mask, pos_weight=pos_weight)
    cnt = count_loss(b_logit, b_target, mask)
    spar = l1_prob_loss(b_logit, mask, lam=lam_spar)
    loss = bce + beta_count * cnt + spar
    return loss, {"bce": float(bce.item()), "count": float(cnt.item()), "sparsity": float(spar.item())}


@torch.no_grad()
def boundary_metrics(b_logit: torch.Tensor, b_target: torch.Tensor, mask: torch.Tensor, thr: float = 0.3) -> dict[str, float]:
    prob = torch.sigmoid(b_logit)
    pred = (prob > thr) & mask
    tgt = b_target & mask
    tp = (pred & tgt).sum().float()
    fp = (pred & ~tgt).sum().float()
    fn = (~pred & tgt).sum().float()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = ((pred == tgt) & mask).sum().float() / mask.sum().clamp_min(1)
    return {
        "f1": float(f1.item()),
        "acc": float(acc.item()),
        "pred_boundaries": int(pred.sum().item()),
        "true_boundaries": int(tgt.sum().item()),
    }


def _trim_to_active(xb: torch.Tensor, yb: torch.Tensor, mb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    true_t = max(int(mb.sum(dim=1).max().item()), 1)
    return xb[:, :true_t], yb[:, :true_t], mb[:, :true_t]


def run_epoch(
    model: PacketSegformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    bce_pos_wt: float,
    count_beta: float,
    thr: float = 0.3,
) -> dict[str, float]:
    model.train(mode=train_mode)
    total_loss = total_bce = total_cnt = total_spar = 0.0
    all_f1: list[float] = []
    all_acc: list[float] = []
    pred_count = true_count = 0
    n_batches = 0

    for xb, yb, mb in loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        xb, yb, mb = _trim_to_active(xb, yb, mb)

        with torch.set_grad_enabled(train_mode):
            out = model(xb, mb)
            loss, parts = boundary_loss_augmented(
                out["b_logit"],
                yb,
                mb,
                pos_weight=bce_pos_wt,
                beta_count=count_beta,
            )
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        m = boundary_metrics(out["b_logit"].detach(), yb, mb, thr=thr)
        all_f1.append(m["f1"])
        all_acc.append(m["acc"])
        pred_count += m["pred_boundaries"]
        true_count += m["true_boundaries"]

        total_loss += float(loss.item())
        total_bce += parts["bce"]
        total_cnt += parts["count"]
        total_spar += parts["sparsity"]
        n_batches += 1

    denom = max(1, n_batches)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "count": total_cnt / denom,
        "sparsity": total_spar / denom,
        "f1": float(np.mean(all_f1)) if all_f1 else 0.0,
        "acc": float(np.mean(all_acc)) if all_acc else 0.0,
        "pred_boundaries": float(pred_count),
        "true_boundaries": float(true_count),
    }


@dataclass
class TrainConfig:
    d_model: int = 16
    n_head: int = 1
    n_layers: int = 4
    ff_dim: int = 16
    drop: float = 0.1
    bce_pos_wt: float = 18.0
    count_beta: float = 1.0
    # Match the notebook run that reached ~99% Acc (was wrongly 3e-5 in the script).
    lr: float = 3.5e-4
    epochs: int = 100
    # Notebook used 48000; 4096 OOMs on 8GB. 1024 fits and worked for prior runs.
    batch_size: int = 1024
    window_size: int = 250
    stride: int = 25
    # Random window split fractions (must sum to 1.0). Track best-by-val; report final metrics on test.
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    # None = run all epochs (no early stop). Notebook hit Acc>=0.99 around epoch 16 and kept improving to 100.
    early_stop_patience: Optional[int] = None
    metric_threshold: float = 0.3
    split_seed: int = 42


def load_single_sequence(
    x_csv: str | Path,
    x_timestamp_col: str = "frame.time_relative",
    x_size_col: str = "frame.len",
    y_timestamp_col: str = "frame.time_relative",
    y_marker_col: str = "rtp.marker",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pd.read_csv(x_csv, usecols=[x_timestamp_col, x_size_col, y_marker_col]).copy()
    df[x_timestamp_col] = pd.to_numeric(df[x_timestamp_col], errors="coerce")
    df[x_size_col] = pd.to_numeric(df[x_size_col], errors="coerce")
    df = df.dropna(subset=[x_timestamp_col, x_size_col]).sort_values(x_timestamp_col, kind="mergesort").reset_index(drop=True)
    df[y_marker_col] = df[y_marker_col].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes"])

    abs_t = df[x_timestamp_col].to_numpy(dtype=np.float32)
    pkt_sz = df[x_size_col].to_numpy(dtype=np.float32)
    marker = df[y_marker_col].to_numpy(dtype=bool)
    if marker.size and not marker[-1]:
        marker[-1] = True

    dt = abs_t[1:] - abs_t[:-1]
    x = np.stack([dt, pkt_sz[1:]], axis=1)
    y = marker[1:]

    x_t = torch.from_numpy(x).unsqueeze(0)
    y_t = torch.from_numpy(y).unsqueeze(0)
    mask_t = torch.ones((1, x_t.shape[1]), dtype=torch.bool)
    return x_t, y_t, mask_t


def load_sequences_from_root(
    data_root: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # MAX_CSV_FILES = 20
    root = Path(data_root).expanduser()
    paths = sorted(root.rglob("*.csv"))
    # paths = paths[:MAX_CSV_FILES]
    if not paths:
        raise ValueError(f"No CSV files found under {data_root}")
    print(f"Found {len(paths)} CSV files under {root}")

    x_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    m_list: list[torch.Tensor] = []
    max_len = 0

    for p in tqdm(paths, desc="Loading CSVs"):
        x_i, y_i, m_i = load_single_sequence(p)
        x_i = x_i.squeeze(0)
        y_i = y_i.squeeze(0)
        m_i = m_i.squeeze(0)
        num_frames_i = int((y_i & m_i).sum().item())
        if num_frames_i < 150:
            continue
        x_list.append(x_i)
        y_list.append(y_i)
        m_list.append(m_i)
        max_len = max(max_len, int(x_i.shape[0]))

    n_seq = len(x_list)
    print(f"Kept {n_seq}/{len(paths)} sequences with num_frames >= 150")
    if n_seq == 0:
        raise ValueError("No sequences left after filtering with num_frames >= 150")
    x = torch.full((n_seq, max_len, 2), -1.0, dtype=torch.float32)
    y = torch.zeros((n_seq, max_len), dtype=torch.bool)
    mask = torch.zeros((n_seq, max_len), dtype=torch.bool)

    for i in range(n_seq):
        n = x_list[i].shape[0]
        x[i, :n] = x_list[i]
        y[i, :n] = y_list[i]
        mask[i, :n] = m_list[i]

    return x, y, mask


def train_from_csv_root(
    data_root: str | Path,
    model_out: str | Path = "best_packetsegformer_v2.pkl",
    config: TrainConfig = TrainConfig(),
) -> PacketSegformer:
    x, y, mask = load_sequences_from_root(data_root)

    mask3 = mask.unsqueeze(-1)
    valid = mask3.to(x.dtype)
    cnt = valid.sum(dim=(0, 1)).clamp_min(1.0)
    sum_x = (x * valid).sum(dim=(0, 1))
    mean = sum_x / cnt
    sum_sq = (((x - mean) * valid) ** 2).sum(dim=(0, 1))
    std = (sum_sq / cnt).clamp_min(1e-12).sqrt()
    x_centered = (x - mean) / std
    x = torch.where(mask3, x_centered, x)

    print("CUDA is available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PacketSegformer(
        d_model=config.d_model,
        nhead=config.n_head,
        num_layers=config.n_layers,
        dim_ff=config.ff_dim,
        p_drop=config.drop,
    ).to(device)

    ds = WindowedPacketSeqDataset(x, y, mask, window_size=config.window_size, stride=config.stride)
    n_total = len(ds)
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
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    best_val_acc = -1.0
    epochs_since_last_best = 0
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Training hyperparameters: lr={config.lr:.2e}, batch_size={config.batch_size}, "
        f"epochs={config.epochs}, early_stop_patience={config.early_stop_patience}, "
        f"window={config.window_size}, stride={config.stride}, thr={config.metric_threshold}"
    )

    for epoch in range(1, config.epochs + 1):
        print(f"EPOCH [{epoch:03d}/{config.epochs}]")
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_mode=True,
            bce_pos_wt=config.bce_pos_wt,
            count_beta=config.count_beta,
            thr=config.metric_threshold,
        )
        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            train_mode=False,
            bce_pos_wt=config.bce_pos_wt,
            count_beta=config.count_beta,
            thr=config.metric_threshold,
        )
        print(
            f"  Train loss={train_stats['loss']:.4f} bce={train_stats['bce']:.4f} count={train_stats['count']:.4f} "
            f"F1={train_stats['f1']:.3f} Acc={train_stats['acc']:.3f}"
        )
        print(
            f"  Val   loss={val_stats['loss']:.4f} bce={val_stats['bce']:.4f} count={val_stats['count']:.4f} "
            f"F1={val_stats['f1']:.3f} Acc={val_stats['acc']:.3f}"
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            torch.save(model.state_dict(), model_out)
            print(f"  Saved new best checkpoint to {model_out} (val Acc={best_val_acc:.4f})")
            epochs_since_last_best = 0
        else:
            epochs_since_last_best += 1

        if (
            config.early_stop_patience is not None
            and epochs_since_last_best >= config.early_stop_patience
        ):
            print(
                f"Stopping early after {config.early_stop_patience} epochs "
                "without val accuracy improvement."
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
        bce_pos_wt=config.bce_pos_wt,
        count_beta=config.count_beta,
        thr=config.metric_threshold,
    )
    print(
        f"  Test  (best-by-val) loss={test_stats['loss']:.4f} bce={test_stats['bce']:.4f} "
        f"count={test_stats['count']:.4f} F1={test_stats['f1']:.3f} Acc={test_stats['acc']:.3f} "
        f"| #pred={test_stats['pred_boundaries']:.0f} vs #true={test_stats['true_boundaries']:.0f}"
    )
    return model


if __name__ == "__main__":
    DATA_ROOT = "/media/ubuntu/research/carla_data_aug_combined"
    train_from_csv_root(
        DATA_ROOT,
        model_out="best_packetsegformer_carla_data_aug.pkl",
    )
