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
from torch.utils.data import DataLoader, Dataset

from sqa.evaluator import FAULT_TYPES, compute_f1_macro, compute_frame_iou_macro
from sqa.frame_grid import MODEL_GRID, EVALUATOR_GRID, intervals_to_model_frame_targets
from sqa.postprocess import BASELINE_POSTPROCESS, PostProcessConfig, predict_fault_type, serialize_intervals
from scripts.train_fault_head_v5 import (
    compute_focal_bce_loss,
    compute_soft_iou_loss,
    compute_presence_loss,
    OLD_DIR,
    NEW_DIR,
    TEST_DIR,
    DEVICE,
)
from scripts.train_conformer_fault_head import ConformerConvModule, ConformerBlock

class DualBackboneConformerHead(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256, num_blocks: int = 2, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            ConformerBlock(dim=hidden_dim, num_heads=4, conv_kernel=15, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = h.shape
        if valid_mask is None:
            mask = torch.ones((B, T), dtype=torch.bool, device=h.device)
        else:
            mask = valid_mask
        mask_f = mask.unsqueeze(-1).float()
        x = self.in_proj(h) * mask_f
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.classifier(x)
        return logits.masked_fill(~mask.unsqueeze(-1), -30.0)

class DualCachedDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wavlm_pt = torch.load(item["wavlm_path"], map_location="cpu")
        xlsr_pt = torch.load(item["xlsr_path"], map_location="cpu")
        
        hw = wavlm_pt["hidden"]
        hx = xlsr_pt["hidden"]
        min_t = min(hw.shape[0], hx.shape[0])
        fused = torch.cat([hw[:min_t], hx[:min_t]], dim=-1)
        
        dur = float(wavlm_pt.get("duration", 0.0))
        if dur <= 0.0:
            dur = float(item.get("duration", 0.0))
        if dur <= 0.0:
            dur = float(min_t * MODEL_GRID.frame_hop_sec)
        valid_mask = torch.ones(min_t, dtype=torch.bool)
        
        target_frames = np.zeros((min_t, len(FAULT_TYPES)), dtype=np.float32)
        for i, ft in enumerate(FAULT_TYPES):
            intervals = item["fault_dict"].get(ft, [])
            target_frames[:, i] = intervals_to_model_frame_targets(intervals, min_t, MODEL_GRID)
        
        return {
            "hidden": fused,
            "valid_mask": valid_mask,
            "targets": torch.from_numpy(target_frames),
            "file_id": item["file_id"],
            "duration": dur,
            "gt_faults": item["fault_dict"],
        }

def dual_collate_fn(batch):
    max_len = max(b["hidden"].shape[0] for b in batch)
    feat_dim = batch[0]["hidden"].shape[1]
    b_size = len(batch)
    
    padded_h = torch.zeros((b_size, max_len, feat_dim), dtype=torch.float32)
    padded_m = torch.zeros((b_size, max_len), dtype=torch.bool)
    padded_t = torch.zeros((b_size, max_len, 3), dtype=torch.float32)
    
    for i, b in enumerate(batch):
        l = b["hidden"].shape[0]
        padded_h[i, :l] = b["hidden"]
        padded_m[i, :l] = b["valid_mask"]
        padded_t[i, :l] = b["targets"]
        
    return {
        "hidden": padded_h,
        "valid_mask": padded_m,
        "targets": padded_t,
        "file_ids": [b["file_id"] for b in batch],
        "durations": [b["duration"] for b in batch],
        "gt_faults": [b["gt_faults"] for b in batch],
    }

def main():
    print("=" * 80, flush=True)
    print("HUẤN LUYỆN DUAL-BACKBONE CONFORMER HEAD (WavLM-Large + XLS-R-300M)", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {DEVICE}", flush=True)

    train_items = []
    for d_name, p_dir in [("old", OLD_DIR), ("new", NEW_DIR)]:
        l_path = p_dir / "labels.jsonl"
        with open(l_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                it = json.loads(line)
                stem = Path(it["file_id"]).stem
                wp = p_dir / "wavlm_cache" / f"{stem}.pt"
                xp = p_dir / "xlsr_cache" / f"{stem}.pt"
                if wp.exists() and xp.exists():
                    train_items.append({
                        "file_id": f"{d_name}_{it['file_id']}",
                        "wavlm_path": wp,
                        "xlsr_path": xp,
                        "fault_dict": it["faults"],
                        "duration": it.get("duration", 0.0),
                    })

    print(f"Loaded {len(train_items)} dual-cached training files.", flush=True)

    test_items = []
    with open(TEST_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            it = json.loads(line)
            stem = Path(it["file_id"]).stem
            wp = TEST_DIR / "wavlm_cache" / f"{stem}.pt"
            xp = TEST_DIR / "xlsr_cache" / f"{stem}.pt"
            if wp.exists() and xp.exists():
                test_items.append({
                    "file_id": it["file_id"],
                    "wavlm_path": wp,
                    "xlsr_path": xp,
                    "fault_dict": it["faults"],
                    "duration": it.get("duration", 0.0),
                })
    print(f"Loaded {len(test_items)} dual-cached validation files.", flush=True)

    train_loader = DataLoader(DualCachedDataset(train_items), batch_size=16, shuffle=True, collate_fn=dual_collate_fn)
    val_loader = DataLoader(DualCachedDataset(test_items), batch_size=16, shuffle=False, collate_fn=dual_collate_fn)

    model = DualBackboneConformerHead(input_dim=2048, hidden_dim=256, num_blocks=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = 0.0
    best_f1 = 0.0
    best_iou = 0.0
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
        preds, gts, durs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                h = batch["hidden"].to(DEVICE)
                m = batch["valid_mask"].to(DEVICE)
                probs = torch.sigmoid(model(h, valid_mask=m)).cpu().numpy()
                for i, fid in enumerate(batch["file_ids"]):
                    dur = batch["durations"][i]
                    file_pred = {}
                    for c_idx, ft in enumerate(FAULT_TYPES):
                        ivs, _, _, _ = predict_fault_type(probs[i, :, c_idx], MODEL_GRID, dur, BASELINE_POSTPROCESS)
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
            best_f1 = f1
            best_iou = iou
            marker = " [★ NEW BEST DUAL-BACKBONE]"
            torch.save(model.state_dict(), ROOT / "outputs" / "fault_head_v5" / "best_dual_backbone_conformer.pt")

        print(f"Epoch {ep:2d}/{epochs} | Loss: {train_loss:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | Fault Avg: {comp*100:.2f}%{marker}", flush=True)

    print(f"\n★ Hoàn tất huấn luyện Dual-Backbone Conformer trong {time.time() - t0:.1f}s!", flush=True)
    print(f"  Best Val F1:  {best_f1:.4f}", flush=True)
    print(f"  Best Val IoU: {best_iou:.4f}", flush=True)
    print(f"  Best Fault Avg (Uncalibrated): {best_score*100:.2f}%", flush=True)

if __name__ == "__main__":
    main()
