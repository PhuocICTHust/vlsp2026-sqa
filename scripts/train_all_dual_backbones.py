"""
Script: scripts/train_all_dual_backbones.py
Train remaining Dual-Backbone architectures (Conformer-Pro and Decoupled Tri-Branch)
on fused 2048-dim WavLM + XLS-R representations.
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
    DEVICE,
    NEW_DIR,
    OLD_DIR,
    TEST_DIR,
    compute_focal_bce_loss,
    compute_presence_loss,
    compute_soft_iou_loss,
)
from scripts.train_dual_backbone_conformer import DualCachedDataset, dual_collate_fn
from scripts.train_conformer_pro import ConformerProFaultHead, apply_spec_augment, compute_boundary_loss
from scripts.train_decoupled_fault_head import DecoupledFaultHead


def load_datasets():
    train_items = []
    for d_name, p_dir in [("old", OLD_DIR), ("new", NEW_DIR)]:
        l_path = p_dir / "labels.jsonl"
        with open(l_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
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
                    })

    test_items = []
    with open(TEST_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
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
                })

    return train_items, test_items


def train_dual_conformer_pro(train_loader, val_loader):
    print("\n" + "=" * 70, flush=True)
    print("1. TRAINING DUAL-BACKBONE CONFORMER-PRO (MULTI-SCALE + BOUNDARY LOSS)", flush=True)
    print("=" * 70, flush=True)

    model = ConformerProFaultHead(input_dim=2048, hidden_dim=256, num_blocks=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            h = batch["hidden"].to(DEVICE)
            m = batch["valid_mask"].to(DEVICE)
            t = batch["targets"].to(DEVICE)

            h_aug = apply_spec_augment(h, m, is_training=True, p=0.4)
            logits = model(h_aug, valid_mask=m)
            probs = torch.sigmoid(logits)

            l_focal = compute_focal_bce_loss(logits, t, m)
            l_iou = compute_soft_iou_loss(probs, t, m)
            l_pres = compute_presence_loss(probs, t, m)
            l_bnd = compute_boundary_loss(probs, t, m)
            loss = l_focal + 0.5 * l_iou + 0.2 * l_pres + 0.1 * l_bnd

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
            marker = " [NEW BEST]"
            torch.save(model.state_dict(), ROOT / "outputs" / "fault_head_v5" / "best_dual_conformer_pro.pt")

        print(f"  Epoch {ep:2d}/{epochs} | Loss: {train_loss:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | Fault: {comp*100:.2f}%{marker}", flush=True)

    print(f"Dual-Backbone Conformer-Pro Best: {best_score*100:.2f}%", flush=True)
    return best_score


def train_dual_decoupled(train_loader, val_loader):
    print("\n" + "=" * 70, flush=True)
    print("2. TRAINING DUAL-BACKBONE DECOUPLED TRI-BRANCH (TCN + BiLSTM)", flush=True)
    print("=" * 70, flush=True)

    model = DecoupledFaultHead(input_dim=2048, branch_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = 0.0
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
            marker = " [NEW BEST]"
            torch.save(model.state_dict(), ROOT / "outputs" / "fault_head_v5" / "best_dual_decoupled.pt")

        print(f"  Epoch {ep:2d}/{epochs} | Loss: {train_loss:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | Fault: {comp*100:.2f}%{marker}", flush=True)

    print(f"Dual-Backbone Decoupled Best: {best_score*100:.2f}%", flush=True)
    return best_score


def main():
    print("=" * 70, flush=True)
    print("TRAINING ALL DUAL-BACKBONE ARCHITECTURES (2048-DIMENSIONAL)", flush=True)
    print("=" * 70, flush=True)

    train_items, test_items = load_datasets()
    print(f"Train samples: {len(train_items)} | Val samples: {len(test_items)}", flush=True)

    train_loader = DataLoader(DualCachedDataset(train_items), batch_size=16, shuffle=True, collate_fn=dual_collate_fn)
    val_loader = DataLoader(DualCachedDataset(test_items), batch_size=16, shuffle=False, collate_fn=dual_collate_fn)

    t0 = time.time()
    train_dual_conformer_pro(train_loader, val_loader)
    train_dual_decoupled(train_loader, val_loader)
    print(f"\nTraining completed for both architectures in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
