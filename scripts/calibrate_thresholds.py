"""
Threshold & Post-Processing Calibration for FaultFrameHead.

Grid searches post-processing parameters per fault type:
- tau_frame (frame probability threshold)
- merge_gap (merge adjacent intervals closer than this gap in seconds)
- min_duration (discard intervals shorter than this duration in seconds)
- use_presence_gate & tau_presence (clip-level presence gating)

Objective:
Maximize Composite Fault Score = 0.50 * F1_macro + 0.50 * IoU_macro
(Which accounts for 60% of total VLSP 2026 SQA competition score).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.evaluator import FAULT_TYPES, compute_f1_macro, compute_frame_iou_macro
from sqa.frame_grid import EVALUATOR_GRID, MODEL_GRID
from sqa.frame_head import FaultFrameHead
from sqa.postprocess import (
    BASELINE_POSTPROCESS,
    PostProcessConfig,
    predict_fault_type,
    serialize_intervals,
)

# Reuse dataset loading helpers from train_fault_frame_head
from scripts.train_fault_frame_head import (
    CACHE_DIR,
    CachedFaultDataset,
    build_annotation_lookup,
    fault_collate_fn,
    get_group,
    get_source,
    make_source_balanced_split,
)

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
LABEL_V2_PATH = ROOT / "data" / "qualispeech" / "label_v2" / "parsed_labels.jsonl"
REVIEWED_JSON_PATH = ROOT / "label_studio" / "reviewed_labels_for_kien.json"
CHECKPOINT_PATH = ROOT / "outputs" / "fault_head" / "best_fault_frame_head.pt"
OUT_CONFIG_PATH = ROOT / "outputs" / "fault_head" / "calibrated_postprocess_config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_predictions(model, loader, device):
    """Run model inference once and cache raw probabilities in CPU memory."""
    model.eval()
    all_probs = []
    all_lengths = []
    all_gt = []
    all_file_ids = []

    print("Extracting raw model probabilities on validation set...")
    with torch.no_grad():
        for batch in loader:
            hidden = batch["hidden"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            logits = model(hidden, valid_mask=valid_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            lengths = valid_mask.sum(dim=1).cpu().numpy()
            for b in range(len(lengths)):
                n_valid = int(lengths[b])
                all_probs.append(probs[b, :n_valid, :])  # [T, 3]
                all_lengths.append(n_valid)
                all_gt.append(batch["gt_faults"][b])
                all_file_ids.append(batch["file_ids"][b])

    print(f"Collected predictions for {len(all_probs)} validation clips.")
    return all_probs, all_lengths, all_gt, all_file_ids


def evaluate_config_for_fault(
    fault_idx: int,
    fault_type: str,
    cfg: PostProcessConfig,
    all_probs: list[np.ndarray],
    all_gt: list[dict],
) -> tuple[float, float, float]:
    """Evaluate F1 and IoU for a single fault type under a candidate PostProcessConfig."""
    pred_entries = []
    gt_entries = []

    for idx, probs in enumerate(all_probs):
        f_probs = probs[:, fault_idx]
        duration = len(f_probs) * MODEL_GRID.frame_hop_sec + MODEL_GRID.offset_sec
        ivs, _, _, _ = predict_fault_type(
            frame_probs=f_probs,
            grid=MODEL_GRID,
            duration=duration,
            config=cfg,
        )
        norm_ivs = serialize_intervals(ivs, round_digits=2, duration=duration)

        pred_entries.append({
            "id": f"clip_{idx}",
            "faults": {fault_type: norm_ivs},
            "_duration": duration,
        })
        gt_entries.append({
            "id": f"clip_{idx}",
            "faults": {fault_type: all_gt[idx].get(fault_type, [])},
            "_duration": duration,
        })

    durations = [e["_duration"] for e in pred_entries]
    f1, _ = compute_f1_macro(pred_entries, gt_entries, fault_types=[fault_type])
    iou, _ = compute_frame_iou_macro(
        pred_entries, gt_entries, durations=durations, grid=EVALUATOR_GRID, fault_types=[fault_type]
    )
    composite = 0.5 * f1 + 0.5 * iou
    return f1, iou, composite


def grid_search_fault(
    fault_idx: int,
    fault_type: str,
    all_probs: list[np.ndarray],
    all_gt: list[dict],
) -> tuple[PostProcessConfig, dict]:
    """Run fine-grained grid search for one fault type."""
    print(f"\n--- GRID SEARCH: {fault_type.upper()} ---")

    # Baseline metrics
    base_f1, base_iou, base_comp = evaluate_config_for_fault(
        fault_idx, fault_type, BASELINE_POSTPROCESS, all_probs, all_gt
    )
    print(f"  Baseline: F1={base_f1:.4f} | IoU={base_iou:.4f} | Score={base_comp:.4f}")

    # Parameter grid tailored for speech fault dynamics
    if fault_type == "unnatural_pause":
        # Pauses typically span at least 150-250ms; small gaps (<250ms) should be merged
        tau_frame_list = [0.35, 0.40, 0.45, 0.50, 0.55]
        merge_gap_list = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
        min_duration_list = [0.0, 0.10, 0.15, 0.20, 0.25]
        presence_opts = [(False, 0.5), (True, 0.4), (True, 0.5)]
    elif fault_type == "noise":
        # Noise can be continuous or transient
        tau_frame_list = [0.35, 0.40, 0.45, 0.50, 0.55]
        merge_gap_list = [0.0, 0.05, 0.10, 0.15, 0.20]
        min_duration_list = [0.0, 0.05, 0.10, 0.15]
        presence_opts = [(False, 0.5), (True, 0.35), (True, 0.45)]
    else:  # distortion
        tau_frame_list = [0.35, 0.40, 0.45, 0.50, 0.55]
        merge_gap_list = [0.0, 0.05, 0.10, 0.15]
        min_duration_list = [0.0, 0.05, 0.10, 0.15]
        presence_opts = [(False, 0.5), (True, 0.4), (True, 0.5)]

    best_cfg = BASELINE_POSTPROCESS
    best_comp = base_comp
    best_metrics = {"f1": base_f1, "iou": base_iou, "score": base_comp}

    total_candidates = (
        len(tau_frame_list) * len(merge_gap_list) * len(min_duration_list) * len(presence_opts)
    )
    print(f"  Searching {total_candidates} candidate configurations...")

    count = 0
    for tau, gap, m_dur, (use_p, tau_p) in itertools.product(
        tau_frame_list, merge_gap_list, min_duration_list, presence_opts
    ):
        cfg = PostProcessConfig(
            tau_frame=tau,
            merge_gap=gap,
            min_duration=m_dur,
            use_presence_gate=use_p,
            tau_presence=tau_p,
        )
        f1, iou, comp = evaluate_config_for_fault(fault_idx, fault_type, cfg, all_probs, all_gt)
        count += 1

        if comp > best_comp:
            best_comp = comp
            best_cfg = cfg
            best_metrics = {"f1": f1, "iou": iou, "score": comp}

    print(
        f"  BEST {fault_type}: Score={best_metrics['score']:.4f} "
        f"(+{best_metrics['score'] - base_comp:+.4f}) | "
        f"F1={best_metrics['f1']:.4f} | IoU={best_metrics['iou']:.4f}"
    )
    print(
        f"  Optimal parameters: tau_frame={best_cfg.tau_frame}, "
        f"merge_gap={best_cfg.merge_gap}s, min_duration={best_cfg.min_duration}s, "
        f"presence_gate={best_cfg.use_presence_gate} (tau_p={best_cfg.tau_presence})"
    )
    return best_cfg, best_metrics


def main():
    print("=" * 70)
    print("FAULT HEAD THRESHOLD & POST-PROCESSING CALIBRATION")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device:     {DEVICE}")

    # Load validation dataset
    df = pd.read_csv(CSV_PATH)
    df["sample_id"] = df["id"].astype(str).str.strip().str.replace(r"\.wav$", "", regex=True)
    df["target"] = pd.to_numeric(df["Overall quality"], errors="raise").astype(float)
    df["group"] = df["sample_id"].map(get_group)
    df["source"] = df["group"].map(get_source)

    annotation_lookup = build_annotation_lookup(df, REVIEWED_JSON_PATH, label_v2_path=LABEL_V2_PATH)
    _, val_df = make_source_balanced_split(df)
    val_dataset = CachedFaultDataset(val_df, CACHE_DIR, annotation_lookup)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=fault_collate_fn,
    )

    # Load model
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu")
    kwargs = payload.get("fault_head_kwargs", {})
    model = FaultFrameHead(**kwargs).to(DEVICE)
    model.load_state_dict(payload["fault_head_state_dict"])

    # Collect raw probabilities
    all_probs, all_lengths, all_gt, all_file_ids = collect_predictions(model, val_loader, DEVICE)

    # Baseline evaluation across all 3 faults
    base_pred_entries = []
    gt_entries = []
    for idx, probs in enumerate(all_probs):
        duration = len(probs) * MODEL_GRID.frame_hop_sec + MODEL_GRID.offset_sec
        clip_faults = {}
        for f_idx, ft in enumerate(FAULT_TYPES):
            ivs, _, _, _ = predict_fault_type(
                probs[:, f_idx], MODEL_GRID, duration, BASELINE_POSTPROCESS
            )
            clip_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=duration)
        base_pred_entries.append({"id": f"clip_{idx}", "faults": clip_faults, "_duration": duration})
        gt_entries.append({"id": f"clip_{idx}", "faults": all_gt[idx], "_duration": duration})

    base_durations = [e["_duration"] for e in base_pred_entries]
    base_f1, _ = compute_f1_macro(base_pred_entries, gt_entries, fault_types=FAULT_TYPES)
    base_iou, _ = compute_frame_iou_macro(
        base_pred_entries, gt_entries, durations=base_durations, grid=EVALUATOR_GRID, fault_types=FAULT_TYPES
    )
    base_composite = 0.5 * base_f1 + 0.5 * base_iou

    print("\n" + "=" * 70)
    print("CURRENT BASELINE PERFORMANCE (BEFORE CALIBRATION)")
    print("=" * 70)
    print(f"F1 macro:       {base_f1:.4f}")
    print(f"IoU macro:      {base_iou:.4f}")
    print(f"Composite score:{base_composite:.4f}")

    # Optimize each fault type independently
    calibrated_configs = {}
    calibrated_metrics = {}

    for f_idx, ft in enumerate(FAULT_TYPES):
        best_cfg, metrics = grid_search_fault(f_idx, ft, all_probs, all_gt)
        calibrated_configs[ft] = best_cfg
        calibrated_metrics[ft] = metrics

    # Overall evaluation with calibrated configurations
    cal_pred_entries = []
    for idx, probs in enumerate(all_probs):
        duration = len(probs) * MODEL_GRID.frame_hop_sec + MODEL_GRID.offset_sec
        clip_faults = {}
        for f_idx, ft in enumerate(FAULT_TYPES):
            cfg = calibrated_configs[ft]
            ivs, _, _, _ = predict_fault_type(probs[:, f_idx], MODEL_GRID, duration, cfg)
            clip_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=duration)
        cal_pred_entries.append({"id": f"clip_{idx}", "faults": clip_faults, "_duration": duration})

    cal_durations = [e["_duration"] for e in cal_pred_entries]
    cal_f1, _ = compute_f1_macro(cal_pred_entries, gt_entries, fault_types=FAULT_TYPES)
    cal_iou, _ = compute_frame_iou_macro(
        cal_pred_entries, gt_entries, durations=cal_durations, grid=EVALUATOR_GRID, fault_types=FAULT_TYPES
    )
    cal_composite = 0.5 * cal_f1 + 0.5 * cal_iou

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS (AFTER OPTIMIZATION)")
    print("=" * 70)
    print(f"Baseline Composite:   {base_composite:.4f}")
    print(f"Calibrated Composite: {cal_composite:.4f}  (+{cal_composite - base_composite:+.4f})")
    print(f"Calibrated F1 macro:  {cal_f1:.4f}  (+{cal_f1 - base_f1:+.4f})")
    print(f"Calibrated IoU macro: {cal_iou:.4f}  (+{cal_iou - base_iou:+.4f})")

    # Save to JSON
    saved_payload = {
        "calibrated_configs": {ft: cfg.to_dict() for ft, cfg in calibrated_configs.items()},
        "baseline_metrics": {"f1_macro": base_f1, "iou_macro": base_iou, "composite": base_composite},
        "calibrated_metrics": {"f1_macro": cal_f1, "iou_macro": cal_iou, "composite": cal_composite},
    }
    OUT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(saved_payload, f, indent=2)

    print(f"\nSaved calibrated configuration to: {OUT_CONFIG_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
