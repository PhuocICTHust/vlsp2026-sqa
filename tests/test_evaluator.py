import math

import numpy as np
import pytest

from sqa.evaluator import (
    compute_f1_macro,
    compute_frame_iou_macro,
    compute_pcc,
    final_score,
    frame_iou,
    intervals_to_frame_mask,
)
from sqa.frame_grid import EvaluatorGrid

GRID = EvaluatorGrid(frame_hop_sec=0.02, offset_sec=0.0)



def test_mask_exact_frame_0():
    mask = intervals_to_frame_mask([[0.00, 0.02]], duration=0.10, grid=GRID)
    assert mask.tolist() == [True, False, False, False, False]


def test_mask_exact_frame_1_does_not_leak_into_frame_0():
    mask = intervals_to_frame_mask([[0.02, 0.04]], duration=0.10, grid=GRID)
    assert mask.tolist() == [False, True, False, False, False]


def test_mask_straddling_boundary_marks_both_frames():
    # [0.019, 0.021] chạm cả frame 0 và frame 1 -> floor/ceil cố tình bao trọn cả 2
    mask = intervals_to_frame_mask([[0.019, 0.021]], duration=0.10, grid=GRID)
    assert mask.tolist() == [True, True, False, False, False]


def test_mask_short_interval_within_one_frame():
    mask = intervals_to_frame_mask([[0.005, 0.015]], duration=0.10, grid=GRID)
    assert mask.tolist() == [True, False, False, False, False]


def test_mask_interval_beyond_duration_is_clamped_not_crashing():
    mask = intervals_to_frame_mask([[0.08, 0.50]], duration=0.10, grid=GRID)
    assert mask.tolist() == [False, False, False, False, True]



def test_iou_perfect_match_is_one():
    assert math.isclose(frame_iou([[0.0, 0.5]], [[0.0, 0.5]], duration=1.0, grid=GRID), 1.0)


def test_iou_pred_covers_gt_entirely():
    # pred rộng gấp đôi gt -> IoU = |gt|/|pred|
    iou = frame_iou([[0.0, 1.0]], [[0.0, 0.5]], duration=1.0, grid=GRID)
    assert math.isclose(iou, 0.5, abs_tol=0.02)


def test_iou_gt_positive_pred_empty_is_zero():
    assert frame_iou([], [[0.0, 0.5]], duration=1.0, grid=GRID) == 0.0


def test_iou_both_empty_is_zero_by_policy():
    assert frame_iou([], [], duration=1.0, grid=GRID) == 0.0


def test_iou_overlapping_multi_interval_union_correct():
    pred = [[0.0, 0.3], [0.2, 0.6]]  # chồng lấn -> union thật là [0.0, 0.6]
    gt = [[0.0, 0.6]]
    assert math.isclose(frame_iou(pred, gt, duration=1.0, grid=GRID), 1.0)


def test_gt_negative_file_excluded_from_macro():
    preds = [{"faults": {"noise": [[0.0, 0.5]], "distortion": [], "unnatural_pause": []}}]
    gts = [{"faults": {"noise": [], "distortion": [], "unnatural_pause": []}}]  # GT không có lỗi nào
    macro, per_type = compute_frame_iou_macro(preds, gts, durations=[1.0], grid=GRID)
    # không có file GT dương nào cho loại nào -> tất cả per_type = 0.0 (không lỗi, chỉ là rỗng)
    assert per_type["noise"] == 0.0


def test_imbalanced_positive_counts_per_type_still_macro_averaged_correctly():
    preds = [
        {"faults": {"noise": [[0.0, 1.0]], "distortion": [], "unnatural_pause": []}} for _ in range(5)
    ] + [{"faults": {"noise": [], "distortion": [], "unnatural_pause": [[0.0, 1.0]]}}]
    gts = [
        {"faults": {"noise": [[0.0, 1.0]], "distortion": [], "unnatural_pause": []}} for _ in range(5)
    ] + [{"faults": {"noise": [], "distortion": [], "unnatural_pause": [[0.0, 1.0]]}}]
    durations = [1.0] * 6
    macro, per_type = compute_frame_iou_macro(preds, gts, durations, grid=GRID)
    assert math.isclose(per_type["noise"], 1.0)
    assert math.isclose(per_type["unnatural_pause"], 1.0)
    assert per_type["distortion"] == 0.0  # không có GT dương nào cho distortion



def test_pcc_basic():
    pred = [1, 2, 3, 4, 5]
    gt = [1, 2, 3, 4, 5]
    assert math.isclose(compute_pcc(pred, gt), 1.0)


def test_pcc_negative_not_transformed():
    pred = [1, 2, 3, 4, 5]
    gt = [5, 4, 3, 2, 1]
    assert compute_pcc(pred, gt) < 0  # KHÔNG bị clip về 0/dương


def test_pcc_constant_prediction_returns_zero_non_strict():
    assert compute_pcc([3, 3, 3], [1, 2, 3]) == 0.0


def test_pcc_constant_prediction_raises_in_strict_mode():
    with pytest.raises(ValueError):
        compute_pcc([3, 3, 3], [1, 2, 3], strict=True)


def test_pcc_nan_input_non_strict_returns_zero():
    assert compute_pcc([1, 2, float("nan")], [1, 2, 3]) == 0.0


def test_pcc_nan_input_strict_raises():
    with pytest.raises(ValueError):
        compute_pcc([1, 2, float("nan")], [1, 2, 3], strict=True)


def test_pcc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_pcc([1, 2, 3], [1, 2])


def test_pcc_too_few_samples_returns_zero():
    assert compute_pcc([1.0], [2.0]) == 0.0


# ---------- compute_f1_macro ----------

def test_f1_macro_perfect_prediction():
    preds = [
        {"faults": {"noise": [[0, 1]], "distortion": [], "unnatural_pause": []}},
        {"faults": {"noise": [], "distortion": [[0, 1]], "unnatural_pause": []}},
        {"faults": {"noise": [], "distortion": [], "unnatural_pause": [[0, 1]]}},
    ]
    gts = [
        {"faults": {"noise": [[0, 1]], "distortion": [], "unnatural_pause": []}},
        {"faults": {"noise": [], "distortion": [[0, 1]], "unnatural_pause": []}},
        {"faults": {"noise": [], "distortion": [], "unnatural_pause": [[0, 1]]}},
    ]
    macro, per_type = compute_f1_macro(preds, gts)
    assert math.isclose(macro, 1.0)
    assert all(math.isclose(v, 1.0) for v in per_type.values())


def test_f1_macro_excludes_no_fault_class_uses_positive_class_only():
    # 2 file no-fault đúng cả 2 phía -> không được tính là success thổi phồng F1
    preds = [{"faults": {"noise": [], "distortion": [], "unnatural_pause": []}} for _ in range(2)]
    gts = [{"faults": {"noise": [], "distortion": [], "unnatural_pause": []}} for _ in range(2)]
    macro, per_type = compute_f1_macro(preds, gts)
    assert per_type["noise"] == 0.0  # zero_division=0, không giả vờ perfect vì toàn negative



def test_final_score_formula():
    assert math.isclose(final_score(1.0, 1.0, 1.0), 1.0)
    assert math.isclose(final_score(0.0, 0.0, 0.0), 0.0)
    assert math.isclose(final_score(0.5, 0.5, 0.5), 0.5)
    assert math.isclose(final_score(1.0, 0.0, 0.0), 0.40)
