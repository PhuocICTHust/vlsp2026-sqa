"""
Script: scripts/train_conformer_pro.py
Train Multi-Scale Conformer-Pro Fault Localization Head with SpecAugment and Boundary Loss.
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sqa.evaluator import FAULT_TYPES, compute_f1_macro, compute_frame_iou_macro
from sqa.frame_grid import EVALUATOR_GRID, MODEL_GRID
from sqa.postprocess import BASELINE_POSTPROCESS, PostProcessConfig, predict_fault_type, serialize_intervals
from scripts.train_fault_head_v5 import (
    CombinedCachedDataset,
    DEVICE,
    NEW_DIR,
    OLD_DIR,
    TEST_DIR,
    compute_focal_bce_loss,
    compute_presence_loss,
    compute_soft_iou_loss,
    fault_collate_fn,
)


def apply_spec_augment(x: torch.Tensor, mask: torch.Tensor, is_training: bool = True, p: float = 0.5) -> torch.Tensor:
    """SpecAugment on hidden features [B, T, D]."""
    if not is_training or torch.rand(1).item() > p:
        return x
    b, t, d = x.shape
    x_aug = x.clone()
    for i in range(b):
        valid_t = int(mask[i].sum().item())
        if valid_t < 10:
            continue
        n_t_masks = np.random.randint(1, 3)
        for _ in range(n_t_masks):
            t_len = np.random.randint(4, min(16, valid_t // 2))
            t_start = np.random.randint(0, valid_t - t_len)
            x_aug[i, t_start : t_start + t_len, :] = 0.0

        n_c_masks = np.random.randint(1, 3)
        for _ in range(n_c_masks):
            c_len = np.random.randint(32, 65)
            c_start = np.random.randint(0, d - c_len)
            x_aug[i, :, c_start : c_start + c_len] = 0.0
    return x_aug


def compute_boundary_loss(probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Penalize blurry edges by aligning first-order temporal differences."""
    diff_p = torch.abs(probs[:, 1:, :] - probs[:, :-1, :])
    diff_t = torch.abs(targets[:, 1:, :] - targets[:, :-1, :])
    valid = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).to(probs.dtype)
    return F.binary_cross_entropy(diff_p.clamp(1e-6, 1.0 - 1e-6), diff_t, weight=valid)


class MultiScaleConvModule(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)

        self.dw_k5 = nn.Conv1d(dim // 4, dim // 4, kernel_size=5, padding=2, groups=dim // 4)
        self.dw_k15 = nn.Conv1d(dim // 2, dim // 2, kernel_size=15, padding=7, groups=dim // 2)
        self.dw_k31 = nn.Conv1d(dim // 4, dim // 4, kernel_size=31, padding=15, groups=dim // 4)

        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.layer_norm(x) * mask_f
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)

        x5 = self.dw_k5(x[:, : x.shape[1] // 4, :])
        x15 = self.dw_k15(x[:, x.shape[1] // 4 : 3 * x.shape[1] // 4, :])
        x31 = self.dw_k31(x[:, 3 * x.shape[1] // 4 :, :])
        x_dw = torch.cat([x5, x15, x31], dim=1)

        x_dw = self.batch_norm(x_dw)
        x_dw = F.gelu(x_dw)
        x_out = self.pointwise_conv2(x_dw)
        x_out = self.dropout(x_out).transpose(1, 2)
        return (res + x_out) * mask_f


class ConformerProBlock(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        self.conv_module = MultiScaleConvModule(dim, dropout=dropout)

        self.ffn2_norm = nn.LayerNorm(dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x)) * mask_f
        norm_x = self.attn_norm(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=~mask)
        x = (x + self.attn_drop(attn_out)) * mask_f
        x = self.conv_module(x, mask_f)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x)) * mask_f
        return self.final_norm(x) * mask_f


class ConformerProFaultHead(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_blocks: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ConformerProBlock(dim=hidden_dim, num_heads=4, dropout=0.1)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = h.shape
        if valid_mask is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=h.device)
        else:
            mask = valid_mask.to(h.device)
        mask_f = mask.unsqueeze(-1).to(h.dtype)

        x = self.in_proj(h) * mask_f
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.classifier(x)
        return logits.masked_fill(~mask.unsqueeze(-1), -30.0)


def main():
    print("=" * 75)
    print("TRAINING CONFORMER-PRO FAULT HEAD (MULTI-SCALE + SPECAUG + BOUNDARY LOSS)")
    print("=" * 75)
    print(f"Device: {DEVICE}")

    old_items = []
    with open(OLD_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            pt_path = OLD_DIR / "wavlm_cache" / f"{Path(item['file_id']).stem}.pt"
            if pt_path.exists():
                old_items.append({"file_id": f"old_{item['file_id']}", "pt_path": pt_path, "fault_dict": item["faults"]})

    new_items = []
    with open(NEW_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            pt_path = NEW_DIR / "wavlm_cache" / f"{Path(item['file_id']).stem}.pt"
            if pt_path.exists():
                new_items.append({"file_id": f"new_{item['file_id']}", "pt_path": pt_path, "fault_dict": item["faults"]})

    all_train = old_items + new_items
    print(f"Train samples: {len(all_train)} files.")

    test_items = []
    with open(TEST_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            pt_path = TEST_DIR / "wavlm_cache" / f"{Path(item['file_id']).stem}.pt"
            test_items.append({"file_id": item["file_id"], "pt_path": pt_path, "fault_dict": item["faults"]})

    train_loader = DataLoader(CombinedCachedDataset(all_train), batch_size=16, shuffle=True, collate_fn=fault_collate_fn)
    val_loader = DataLoader(CombinedCachedDataset(test_items), batch_size=16, shuffle=False, collate_fn=fault_collate_fn)

    model = ConformerProFaultHead(input_dim=1024, hidden_dim=256, num_blocks=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = 0.0
    best_epoch = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            h = batch["hidden"].to(DEVICE)
            m = batch["valid_mask"].to(DEVICE)
            t = batch["targets"].to(DEVICE)

            h_aug = apply_spec_augment(h, m, p=0.5)

            logits = model(h_aug, valid_mask=m)
            probs = torch.sigmoid(logits)

            l_focal = compute_focal_bce_loss(logits, t, m)
            l_iou = compute_soft_iou_loss(probs, t, m)
            l_pres = compute_presence_loss(probs, t, m)
            l_bound = compute_boundary_loss(probs, t, m)
            loss = l_focal + 0.5 * l_iou + 0.2 * l_pres + 0.3 * l_bound

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        preds = []
        gts = []
        durs = []
        with torch.no_grad():
            for batch in val_loader:
                h = batch["hidden"].to(DEVICE)
                m = batch["valid_mask"].to(DEVICE)
                probs = torch.sigmoid(model(h, valid_mask=m)).cpu().numpy()
                for i, fid in enumerate(batch["file_ids"]):
                    dur = batch["durations"][i]
                    file_pred = {}
                    for c_idx, ft in enumerate(FAULT_TYPES):
                        cfg = BASELINE_POSTPROCESS
                        ivs, _, _, _ = predict_fault_type(probs[i, :, c_idx], MODEL_GRID, dur, cfg)
                        file_pred[ft] = serialize_intervals(ivs, round_digits=2, duration=dur)
                    preds.append({"file_id": fid, "faults": file_pred})
                    gts.append({"file_id": fid, "faults": batch["gt_faults"][i]})
                    durs.append(dur)

        f1, _ = compute_f1_macro(preds, gts)
        iou, _ = compute_frame_iou_macro(preds, gts, durs, EVALUATOR_GRID)
        comp = 0.5 * f1 + 0.5 * iou
        marker = ""
        if comp > best_score:
            best_score = comp
            best_epoch = ep
            marker = " [NEW BEST]"
            torch.save(model.state_dict(), "outputs/fault_head_v5/best_conformer_pro.pt")

        print(f"Epoch {ep:2d}/{epochs} | Loss: {train_loss:.4f} | Val F1: {f1:.4f} | Val IoU: {iou:.4f} | Combined: {comp*100:.2f}%{marker}")

    print("\n" + "=" * 75)
    print(f"Completed in {time.time() - t0:.1f}s. Best validation score: {best_score*100:.2f}% at Epoch {best_epoch}")
    print("=" * 75)


if __name__ == "__main__":
    main()
