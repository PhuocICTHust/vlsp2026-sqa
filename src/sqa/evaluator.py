from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

from .frame_grid import EvaluatorGrid, time_to_frame_end, time_to_frame_start

FAULT_TYPES = ("noise", "distortion", "unnatural_pause")


@dataclass(frozen=True)
class MetricPolicy:

    iou_aggregation: str = "filewise_mean"
    empty_gt_handling: str = "exclude"
    gt_positive_empty_pred: str = "zero"
    pcc_transform: str = "none"
    epsilon: float = 1e-8


DEFAULT_POLICY = MetricPolicy()


def intervals_to_frame_mask(intervals, duration: float, grid: EvaluatorGrid) -> np.ndarray:
    n_frames = int(np.ceil(duration / grid.frame_hop_sec))
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in intervals:
        s = max(0, time_to_frame_start(start, grid.frame_hop_sec, grid.offset_sec))
        e = min(n_frames, time_to_frame_end(end, grid.frame_hop_sec, grid.offset_sec))
        if e > s:
            mask[s:e] = True
    return mask


def frame_iou(pred_intervals, gt_intervals, duration: float, grid: EvaluatorGrid, eps: float = 1e-8) -> float:
    p = intervals_to_frame_mask(pred_intervals, duration, grid)
    g = intervals_to_frame_mask(gt_intervals, duration, grid)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return inter / (union + eps)


def compute_pcc(pred_scores, gt_scores, strict: bool = False) -> float:
    pred = np.asarray(pred_scores, dtype=float)
    gt = np.asarray(gt_scores, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"compute_pcc: shape lệch nhau pred={pred.shape} vs gt={gt.shape}")

    finite_ok = np.isfinite(pred).all() and np.isfinite(gt).all()
    if not finite_ok:
        if strict:
            raise ValueError("compute_pcc: pred/gt chứa NaN hoặc Inf — kiểm tra model, không phải lỗi evaluator.")
        return 0.0
    if len(pred) < 2 or np.std(pred) == 0 or np.std(gt) == 0:
        if strict:
            raise ValueError("compute_pcc: <2 mẫu hoặc prediction/gt là hằng số, PCC không xác định.")
        return 0.0
    r, _ = pearsonr(pred, gt)
    return float(r) if np.isfinite(r) else 0.0


def compute_f1_macro(preds, gts, fault_types=FAULT_TYPES) -> tuple[float, dict]:
    per_type = {}
    for ft in fault_types:
        y_true = [int(len(gt["faults"][ft]) > 0) for gt in gts]
        y_pred = [int(len(pred["faults"][ft]) > 0) for pred in preds]
        per_type[ft] = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return float(np.mean(list(per_type.values()))), per_type


def compute_frame_iou_macro(preds, gts, durations, grid: EvaluatorGrid, fault_types=FAULT_TYPES) -> tuple[float, dict]:
    per_type = {}
    for ft in fault_types:
        ious = []
        for pred, gt, dur in zip(preds, gts, durations):
            gt_iv = gt["faults"][ft]
            if len(gt_iv) == 0:
                continue
            ious.append(frame_iou(pred["faults"][ft], gt_iv, dur, grid))
        per_type[ft] = float(np.mean(ious)) if ious else 0.0
    return float(np.mean(list(per_type.values()))), per_type


def bucket_iou_by_duration(
    preds, gts, durations, grid: EvaluatorGrid, fault_type,
    buckets=((0, 0.1), (0.1, 0.5), (0.5, 1), (1, 3), (3, 999)),
):
    result = {b: [] for b in buckets}
    for pred, gt, dur in zip(preds, gts, durations):
        gt_iv = gt["faults"][fault_type]
        if not gt_iv:
            continue
        gt_len = sum(e - s for s, e in gt_iv)
        for lo, hi in buckets:
            if lo <= gt_len < hi:
                result[(lo, hi)].append(frame_iou(pred["faults"][fault_type], gt_iv, dur, grid))
                break
    return {b: (float(np.mean(v)) if v else None) for b, v in result.items()}


def final_score(pcc: float, f1_macro: float, iou_macro: float) -> float:
    return 0.40 * pcc + 0.30 * f1_macro + 0.30 * iou_macro