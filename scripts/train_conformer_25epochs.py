"""
Train Conformer Frame Decoder Fault Head for 25 Epochs.
Tìm điểm bão hòa thực sự của Conformer trên 2,000 files tiếng Việt có WavLM cache.
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
from torch.utils.data import DataLoader

from sqa.evaluator import FAULT_TYPES, compute_f1_macro, compute_frame_iou_macro
from sqa.frame_grid import MODEL_GRID, EVALUATOR_GRID
from sqa.postprocess import BASELINE_POSTPROCESS, predict_fault_type, serialize_intervals
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
from scripts.train_conformer_fault_head import ConformerFaultHead

def main():
    print("=" * 75)
    print("HUẤN LUYỆN CONFORMER FRAME DECODER 25 EPOCHS (TÌM ĐIỂM BÃO HÒA)")
    print("=" * 75)
    print(f"Device: {DEVICE}")

    # Load items
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
    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)

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
        marker = ""
        if comp > best_score:
            best_score = comp
            best_epoch = ep
            marker = " [★ NEW BEST]"
            torch.save(model.state_dict(), "outputs/fault_head_v5/best_conformer_25ep.pt")

        print(f"Epoch {ep:2d}/{epochs} | Loss: {train_loss:.4f} | Val F1: {f1:.4f} | Val IoU: {iou:.4f} | Combined: {comp*100:.2f}%{marker}")

    print("\n" + "=" * 75)
    print(f"Hoàn tất 25 epochs trong {time.time() - t0:.1f}s.")
    print(f"Điểm cao nhất: {best_score*100:.2f}% tại Epoch {best_epoch}!")
    print("=" * 75)

if __name__ == "__main__":
    main()
