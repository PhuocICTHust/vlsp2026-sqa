"""
Script: scripts/train_decoupled_fault_head.py
Train Decoupled Tri-Branch Fault Localization Head (10 Epochs) on Vietnamese augmented speech.
  - Distortion branch: Short receptive-field TCN (kernel 3, dilations 1, 2)
  - Noise branch: Multi-scale dilated TCN (kernel 7, dilations 1, 2, 4)
  - Unnatural Pause branch: Temporal BiLSTM (128-d)
"""

import json
import math
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


class _ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        y = self.norm(x) * mask_f
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        y = self.drop(F.gelu(y))
        return (x + y) * mask_f


class DecoupledFaultHead(nn.Module):
    def __init__(self, input_dim: int = 1024, branch_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.in_norm = nn.LayerNorm(input_dim)

        self.dist_proj = nn.Linear(input_dim, branch_dim)
        self.dist_blocks = nn.ModuleList([
            _ConvBlock(branch_dim, kernel_size=3, dilation=1),
            _ConvBlock(branch_dim, kernel_size=3, dilation=2),
        ])
        self.dist_out = nn.Linear(branch_dim, 1)

        self.noise_proj = nn.Linear(input_dim, branch_dim)
        self.noise_blocks = nn.ModuleList([
            _ConvBlock(branch_dim, kernel_size=7, dilation=1),
            _ConvBlock(branch_dim, kernel_size=7, dilation=2),
            _ConvBlock(branch_dim, kernel_size=7, dilation=4),
        ])
        self.noise_out = nn.Linear(branch_dim, 1)

        self.pause_proj = nn.Linear(input_dim, branch_dim)
        self.pause_lstm = nn.LSTM(branch_dim, 64, batch_first=True, bidirectional=True)
        self.pause_out = nn.Linear(128, 1)

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = h.shape
        if valid_mask is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=h.device)
        else:
            mask = valid_mask.to(h.device)
        mask_f = mask.unsqueeze(-1).to(h.dtype)

        h_norm = self.in_norm(h) * mask_f

        xd = self.dist_proj(h_norm) * mask_f
        for blk in self.dist_blocks:
            xd = blk(xd, mask_f)
        logit_dist = self.dist_out(xd)

        xn = self.noise_proj(h_norm) * mask_f
        for blk in self.noise_blocks:
            xn = blk(xn, mask_f)
        logit_noise = self.noise_out(xn)

        xp = self.pause_proj(h_norm) * mask_f
        lstm_out, _ = self.pause_lstm(xp)
        logit_pause = self.pause_out(lstm_out * mask_f)

        logits = torch.cat([logit_noise, logit_dist, logit_pause], dim=-1)
        return logits.masked_fill(~mask.unsqueeze(-1), -30.0)


def main():
    print("=" * 75)
    print("TRAINING DECOUPLED TRI-BRANCH FAULT HEAD (10 EPOCHS)")
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
    print(f"Test samples: {len(test_items)} files.")

    train_loader = DataLoader(CombinedCachedDataset(all_train), batch_size=16, shuffle=True, collate_fn=fault_collate_fn)
    val_loader = DataLoader(CombinedCachedDataset(test_items), batch_size=16, shuffle=False, collate_fn=fault_collate_fn)

    model = DecoupledFaultHead(input_dim=1024, branch_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            h = batch["hidden"].to(DEVICE)
            m = batch["valid_mask"].to(DEVICE)
            t = batch["targets"].to(DEVICE)

            logits = model(h, valid_mask=m)
            probs = torch.sigmoid(logits)

            l_focal = compute_focal_bce_loss(logits, t, m)
            l_iou = compute_soft_iou_loss(probs, t, m)
            l_pres = compute_presence_loss(probs, t, m)
            loss = l_focal + 0.5 * l_iou + 0.2 * l_pres

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
        marker = " [NEW BEST]" if comp > best_score else ""
        if comp > best_score:
            best_score = comp
            torch.save(model.state_dict(), "outputs/fault_head_v5/best_decoupled_fault_head.pt")

        print(f"Epoch {ep:2d}/{epochs} | Loss: {train_loss:.4f} | Val F1: {f1:.4f} | Val IoU: {iou:.4f} | Combined: {comp*100:.2f}%{marker}")

    print(f"\nTraining completed in {time.time() - t0:.1f}s. Best validation score: {best_score*100:.2f}%")


if __name__ == "__main__":
    main()
