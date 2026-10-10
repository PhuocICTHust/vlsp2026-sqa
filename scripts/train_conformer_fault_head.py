"""
Experiment 3: Conformer Frame Decoder Head for Fault Localization.
Thay thế TCN bằng Conformer Block (Feed-Forward + Multi-Head Self-Attention + Depthwise Conv + Feed-Forward).
Huấn luyện trên 2,000 files tiếng Việt có sẵn WavLM cache.
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sqa.evaluator import FAULT_TYPES, compute_f1_macro, compute_frame_iou_macro
from sqa.frame_grid import MODEL_GRID, EVALUATOR_GRID
from sqa.postprocess import BASELINE_POSTPROCESS, PostProcessConfig, predict_fault_type, serialize_intervals
from scripts.train_fault_head_v5 import (
    CombinedCachedDataset,
    fault_collate_fn,
    compute_focal_bce_loss,
    compute_soft_iou_loss,
    compute_presence_loss,
    OLD_DIR,
    NEW_DIR,
    TEST_DIR,
    DEVICE,
)

class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.layer_norm(x) * mask_f
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # [B, D, T]
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.gelu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return (res + x) * mask_f

class ConformerBlock(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4, conv_kernel: int = 15, dropout: float = 0.1):
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

        self.conv_module = ConformerConvModule(dim, kernel_size=conv_kernel, dropout=dropout)

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

class ConformerFaultHead(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_blocks: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ConformerBlock(dim=hidden_dim, num_heads=4, conv_kernel=15, dropout=0.1)
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
    print("THỬ NGHIỆM CONFORMER FRAME DECODER FAULT HEAD")
    print("=" * 75)
    print(f"Device: {DEVICE}")

    old_items = []
    with open(OLD_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            pt_path = OLD_DIR / "wavlm_cache" / f"{Path(item['file_id']).stem}.pt"
            if pt_path.exists():
                old_items.append({"file_id": f"old_{item['file_id']}", "pt_path": pt_path, "fault_dict": item["faults"]})

    new_items = []
    with open(NEW_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            pt_path = NEW_DIR / "wavlm_cache" / f"{Path(item['file_id']).stem}.pt"
            if pt_path.exists():
                new_items.append({"file_id": f"new_{item['file_id']}", "pt_path": pt_path, "fault_dict": item["faults"]})

    all_train = old_items + new_items
    print(f"Train samples: {len(all_train)} files.")

    test_items = []
    with open(TEST_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            pt_path = TEST_DIR / "wavlm_cache" / f"{Path(item['file_id']).stem}.pt"
            test_items.append({"file_id": item["file_id"], "pt_path": pt_path, "fault_dict": item["faults"]})

    train_loader = DataLoader(CombinedCachedDataset(all_train), batch_size=16, shuffle=True, collate_fn=fault_collate_fn)
    val_loader = DataLoader(CombinedCachedDataset(test_items), batch_size=16, shuffle=False, collate_fn=fault_collate_fn)

    model = ConformerFaultHead(input_dim=1024, hidden_dim=256, num_blocks=2).to(DEVICE)
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
        marker = " [★ NEW BEST]" if comp > best_score else ""
        if comp > best_score:
            best_score = comp
            torch.save(model.state_dict(), "outputs/fault_head_v5/best_conformer_fault_head.pt")

        print(f"Epoch {ep:2d}/{epochs} | Loss: {train_loss:.4f} | Val F1: {f1:.4f} | Val IoU: {iou:.4f} | Combined: {comp*100:.2f}%{marker}")

    print(f"\nHoàn tất trong {time.time() - t0:.1f}s. Điểm cao nhất Conformer Head: {best_score*100:.2f}%")

if __name__ == "__main__":
    main()
