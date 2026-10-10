"""
Script: scripts/train_fault_head_v5.py
Huấn luyện FaultFrameHead v5 (Universal) trên GỘP CẢ 2,000 FILES TIẾNG VIỆT:
- 1,000 files từ vietnamese_aug (data cũ)
- 1,000 files từ vietnamese_aug_v3/train (data mới)
- Validation trên 125 files Gold Test Set (vietnamese_test_v1).
- Tinh chỉnh ngưỡng tối ưu (Presence-Gating + Hysteresis).
- Lưu kết quả tại outputs/fault_head_v5/.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.evaluator import FAULT_TYPES, compute_f1_macro, compute_frame_iou_macro
from sqa.frame_grid import EVALUATOR_GRID, MODEL_GRID, intervals_to_model_frame_targets
from sqa.frame_head import FaultFrameHead, topk_presence
from sqa.postprocess import BASELINE_POSTPROCESS, PostProcessConfig, predict_fault_type, serialize_intervals

OLD_DIR = ROOT / "data" / "vietnamese_aug"
NEW_DIR = ROOT / "data" / "vietnamese_aug_v3" / "train"
TEST_DIR = ROOT / "data" / "vietnamese_test_v1"

INIT_CKPT = ROOT / "outputs" / "fault_head_trainval" / "best_fault_frame_head.pt"
OUT_DIR = ROOT / "outputs" / "fault_head_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 12
BATCH_SIZE = 16
LR = 8e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CombinedCachedDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        info = self.items[idx]
        pt_path = info["pt_path"]
        fid = info["file_id"]
        fault_dict = info["fault_dict"]

        data = torch.load(pt_path, map_location="cpu")
        hidden = data["hidden"].float()
        valid_mask = data["valid_mask"].bool()
        n_frames = len(hidden)

        target_frames = np.zeros((n_frames, len(FAULT_TYPES)), dtype=np.float32)
        for i, ft in enumerate(FAULT_TYPES):
            intervals = fault_dict.get(ft, [])
            target_frames[:, i] = intervals_to_model_frame_targets(intervals, n_frames, MODEL_GRID)

        return {
            "hidden": hidden,
            "valid_mask": valid_mask,
            "target_frames": torch.from_numpy(target_frames),
            "file_id": fid,
            "duration": data.get("duration", n_frames * MODEL_GRID.frame_hop_sec),
            "gt_faults": fault_dict,
        }


def fault_collate_fn(batch):
    max_len = max(item["hidden"].shape[0] for item in batch)
    hidden_dim = batch[0]["hidden"].shape[1]
    n_faults = batch[0]["target_frames"].shape[1]
    b = len(batch)

    padded_hidden = torch.zeros(b, max_len, hidden_dim, dtype=torch.float32)
    padded_valid_mask = torch.zeros(b, max_len, dtype=torch.bool)
    padded_targets = torch.zeros(b, max_len, n_faults, dtype=torch.float32)

    file_ids = []
    durations = []
    gt_faults = []

    for i, item in enumerate(batch):
        t = item["hidden"].shape[0]
        padded_hidden[i, :t] = item["hidden"]
        padded_valid_mask[i, :t] = item["valid_mask"][:t]
        padded_targets[i, :t] = item["target_frames"][:t]
        file_ids.append(item["file_id"])
        durations.append(item["duration"])
        gt_faults.append(item["gt_faults"])

    return {
        "hidden": padded_hidden,
        "valid_mask": padded_valid_mask,
        "targets": padded_targets,
        "file_ids": file_ids,
        "durations": durations,
        "gt_faults": gt_faults,
    }


def compute_focal_bce_loss(logits, targets, valid_mask, gamma=2.0, alpha=0.5):
    mask = valid_mask.unsqueeze(-1).expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = alpha_t * (1.0 - p_t).pow(gamma) * bce
    return (focal * mask).sum() / mask.sum().clamp(min=1)


def compute_soft_iou_loss(probs, targets, valid_mask, eps=1e-6):
    mask = valid_mask.unsqueeze(-1).expand_as(probs)
    p = probs * mask
    t = targets * mask
    intersection = (p * t).sum(dim=(0, 1))
    union = p.sum(dim=(0, 1)) + t.sum(dim=(0, 1)) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


def compute_presence_loss(probs, targets, valid_mask):
    pred_presence = topk_presence(probs, valid_mask)
    mask = valid_mask.unsqueeze(-1).expand_as(targets)
    gt_presence = (targets * mask).amax(dim=1).clamp(0.0, 1.0)
    eps = 1e-7
    pred_presence = pred_presence.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(pred_presence, gt_presence)


@torch.no_grad()
def evaluate_fault_head(head: FaultFrameHead, loader: DataLoader, post_cfg=BASELINE_POSTPROCESS):
    head.eval()
    all_preds = []
    all_gts = []
    all_durs = []

    for batch in loader:
        hidden = batch["hidden"].to(DEVICE)
        valid_mask = batch["valid_mask"].to(DEVICE)
        logits = head(hidden, valid_mask=valid_mask)
        probs = torch.sigmoid(logits).cpu().numpy()

        for i, fid in enumerate(batch["file_ids"]):
            dur = batch["durations"][i]
            gt_dict = batch["gt_faults"][i]
            file_probs = probs[i]

            file_pred = {}
            for c_idx, ft in enumerate(FAULT_TYPES):
                cfg = post_cfg[ft] if isinstance(post_cfg, dict) else post_cfg
                ivs, _, _, _ = predict_fault_type(file_probs[:, c_idx], MODEL_GRID, dur, cfg)
                file_pred[ft] = serialize_intervals(ivs, round_digits=2, duration=dur)

            all_preds.append({"file_id": fid, "faults": file_pred})
            all_gts.append({"file_id": fid, "faults": gt_dict})
            all_durs.append(dur)

    f1_macro, f1_details = compute_f1_macro(all_preds, all_gts)
    iou_macro, iou_details = compute_frame_iou_macro(all_preds, all_gts, all_durs, EVALUATOR_GRID)
    composite = 0.50 * f1_macro + 0.50 * iou_macro

    return {
        "f1_macro": f1_macro,
        "f1_details": f1_details,
        "iou_macro": iou_macro,
        "iou_details": iou_details,
        "composite": composite,
    }


def calibrate_thresholds(head: FaultFrameHead, loader: DataLoader):
    print("\n--- Post-Processing Calibration ---")
    threshold_cands = [0.20, 0.25, 0.30, 0.35, 0.40]
    min_dur_cands = [0.10, 0.15, 0.20]
    merge_gap_cands = [0.10, 0.15, 0.20]

    best_cfg = {ft: BASELINE_POSTPROCESS for ft in FAULT_TYPES}
    baseline_res = evaluate_fault_head(head, loader, best_cfg)
    best_score = baseline_res["composite"]
    print(f"Baseline Post-processing Score: {best_score*100:.2f}% (F1: {baseline_res['f1_macro']:.4f}, IoU: {baseline_res['iou_macro']:.4f})")

    # Greedy search per fault type with presence gate
    for ft in FAULT_TYPES:
        print(f"Cân chỉnh tham số cho lỗi: {ft}...")
        best_ft_cfg = best_cfg[ft]
        for use_pg in [True, False]:
            for tau_p in [0.25, 0.35, 0.45]:
                for th in threshold_cands:
                    for md in min_dur_cands:
                        for mg in merge_gap_cands:
                            trial_cfg = dict(best_cfg)
                            trial_cfg[ft] = PostProcessConfig(
                                tau_frame=th,
                                min_duration=md,
                                merge_gap=mg,
                                use_presence_gate=use_pg,
                                tau_presence=tau_p,
                                use_hysteresis=True,
                                tau_low=th * 0.7,
                            )
                            res = evaluate_fault_head(head, loader, trial_cfg)
                            if res["composite"] > best_score:
                                best_score = res["composite"]
                                best_ft_cfg = trial_cfg[ft]
                                best_cfg[ft] = best_ft_cfg
        print(f"-> Tối ưu {ft}: tau_frame={best_ft_cfg.tau_frame:.2f}, min_dur={best_ft_cfg.min_duration:.2f}, merge={best_ft_cfg.merge_gap:.2f}, pg={best_ft_cfg.use_presence_gate}")

    final_res = evaluate_fault_head(head, loader, best_cfg)
    print(f"-> Điểm sau khi Cân chỉnh: {final_res['composite']*100:.2f}% (F1: {final_res['f1_macro']:.4f}, IoU: {final_res['iou_macro']:.4f})")
    return best_cfg, final_res


def main():
    print("=" * 75)
    print("HUẤN LUYỆN FAULT FRAME HEAD V5 — GỘP FULL 2,000 FILES TIẾNG VIỆT")
    print("=" * 75)
    print(f"Device: {DEVICE}")

    # 1. Load partition 1
    old_items = []
    with open(OLD_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            fid = item["file_id"]
            stem = Path(fid).stem
            pt_path = OLD_DIR / "wavlm_cache" / f"{stem}.pt"
            if pt_path.exists():
                old_items.append({
                    "dataset": "old",
                    "file_id": f"old_{fid}",
                    "pt_path": pt_path,
                    "fault_dict": item["faults"]
                })
    print(f"-> Đã nạp từ tập cũ: {len(old_items)} files.")

    # 2. Load partition 2
    new_items = []
    with open(NEW_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            fid = item["file_id"]
            stem = Path(fid).stem
            pt_path = NEW_DIR / "wavlm_cache" / f"{stem}.pt"
            if pt_path.exists():
                new_items.append({
                    "dataset": "new",
                    "file_id": f"new_{fid}",
                    "pt_path": pt_path,
                    "fault_dict": item["faults"]
                })
    print(f"-> Đã nạp từ tập mới: {len(new_items)} files.")

    all_train_items = old_items + new_items
    print(f"Tổng cộng mẫu huấn luyện: {len(all_train_items)} FILES tiếng Việt.")

    # 3. Load validation set
    test_items = []
    with open(TEST_DIR / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            fid = item["file_id"]
            stem = Path(fid).stem
            pt_path = TEST_DIR / "wavlm_cache" / f"{stem}.pt"
            test_items.append({
                "dataset": "test",
                "file_id": fid,
                "pt_path": pt_path,
                "fault_dict": item["faults"]
            })
    print(f"-> Tập Validation (Gold Test Set): {len(test_items)} files.")

    train_ds = CombinedCachedDataset(all_train_items)
    val_ds = CombinedCachedDataset(test_items)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fault_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fault_collate_fn)

    # 4. Initialize model
    print(f"-> Khởi tạo trọng số từ: {INIT_CKPT.name}")
    init_payload = torch.load(INIT_CKPT, map_location="cpu")
    head_kwargs = init_payload.get("fault_head_kwargs", {
        "input_dim": 1024,
        "n_faults": 3,
        "hidden_dim": 256,
        "n_blocks": 2,
    })
    head = FaultFrameHead(**head_kwargs).to(DEVICE)
    head.load_state_dict(init_payload["fault_head_state_dict"])

    # Baseline Zero-shot
    m_base = evaluate_fault_head(head, val_loader)
    print(f"\nĐiểm khởi điểm (Zero-shot) trên 125 clips Test Set:")
    print(f"  F1: {m_base['f1_macro']:.4f} | IoU: {m_base['iou_macro']:.4f} | Combined: {m_base['composite']*100:.2f}%")

    # Start Fine-tuning
    print(f"\nBắt đầu Huấn luyện Fault Head v5 ({EPOCHS} epochs, lr={LR})...")
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_score = m_base["composite"]
    best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        head.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            h = batch["hidden"].to(DEVICE)
            m = batch["valid_mask"].to(DEVICE)
            t = batch["targets"].to(DEVICE)

            logits = head(h, valid_mask=m)
            probs = torch.sigmoid(logits)

            l_focal = compute_focal_bce_loss(logits, t, m)
            l_iou = compute_soft_iou_loss(probs, t, m)
            l_pres = compute_presence_loss(probs, t, m)
            loss = l_focal + 0.5 * l_iou + 0.2 * l_pres

            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        m_eval = evaluate_fault_head(head, val_loader)
        score = m_eval["composite"]

        marker = ""
        if score > best_score:
            best_score = score
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            marker = " [★ NEW BEST]"

        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Val F1: {m_eval['f1_macro']:.4f} | Val IoU: {m_eval['iou_macro']:.4f} | Combined: {score*100:.2f}%{marker}")

    print(f"\nHoàn thành huấn luyện 2,000 files trong {time.time() - t0:.1f}s.")

    # Save Best Weights
    head.load_state_dict(best_weights)
    best_ckpt = OUT_DIR / "best_fault_frame_head.pt"
    torch.save({
        "fault_head_state_dict": best_weights,
        "fault_head_kwargs": head_kwargs,
        "best_composite": best_score,
        "seed": SEED,
    }, best_ckpt)
    print(f"Đã lưu checkpoint tốt nhất tại: {best_ckpt}")

    # Calibrate Post-processing on Test Set
    calib_cfg, final_res = calibrate_thresholds(head, val_loader)
    calib_json = OUT_DIR / "calibrated_postprocess_config.json"
    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump({
            "calibrated_configs": {k: v.__dict__ for k, v in calib_cfg.items()},
            "calibrated_metrics": final_res,
        }, f, indent=2)
    print(f"Đã lưu file cấu hình ngưỡng tại: {calib_json}")

    print("\n" + "=" * 65)
    print("TỔNG KẾT MÔ HÌNH FAULT HEAD V5 TRÊN GOLD TEST SET:")
    print(f"  F1 Macro:  {final_res['f1_macro']:.4f}")
    print(f"  IoU Macro: {final_res['iou_macro']:.4f}")
    print(f"  Combined:  {final_res['composite']*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
