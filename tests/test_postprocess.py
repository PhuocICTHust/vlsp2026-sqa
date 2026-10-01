import numpy as np
import pytest

from sqa.frame_grid import ModelFrameGrid
from sqa.postprocess import (
    BASELINE_POSTPROCESS,
    PostProcessConfig,
    frame_probs_to_intervals,
    normalize_intervals,
    predict_fault_type,
    presence_from_frames,
    serialize_intervals,
)

GRID = ModelFrameGrid(frame_hop_sec=0.02, offset_sec=0.0)



def test_normalize_merges_overlap_and_touching():
    iv = [[0.0, 0.5], [0.5, 1.0], [0.8, 1.2]]
    out = normalize_intervals(iv, duration=5.0)
    assert out == [[0.0, 1.2]]


def test_normalize_does_not_merge_gap_by_default():
    iv = [[0.0, 0.5], [0.6, 1.0]]
    out = normalize_intervals(iv, duration=5.0, merge_gap=0.0)
    assert out == [[0.0, 0.5], [0.6, 1.0]]


@pytest.mark.parametrize(
    "gap, expect_merged",
    [
        (0.0, True),
        (0.05, True),
        (0.08, False),
        (0.081, False),
    ],
)
def test_merge_gap_boundary_semantics_is_strict_less_than(gap, expect_merged):
    iv = [[0.0, 1.0], [1.0 + gap, 2.0]]
    out = normalize_intervals(iv, duration=5.0, merge_gap=0.08)
    if expect_merged:
        assert len(out) == 1
    else:
        assert len(out) == 2


def test_normalize_clips_to_duration():
    out = normalize_intervals([[-1.0, 0.5], [4.5, 10.0]], duration=5.0)
    assert out == [[0.0, 0.5], [4.5, 5.0]]


def test_normalize_filters_min_duration():
    out = normalize_intervals([[0.0, 0.05]], duration=5.0, min_duration=0.10)
    assert out == []


def test_normalize_default_round_digits_none_keeps_full_precision():
    out = normalize_intervals([[0.123456, 0.654321]], duration=5.0)
    assert out == [[0.123456, 0.654321]]


def test_serialize_intervals_rounds_only_at_this_step():
    out = serialize_intervals([[0.123456, 0.654321]], round_digits=2)
    assert out == [[0.12, 0.65]]



@pytest.mark.parametrize("t", [0, 1, 2, 3, 10])
def test_presence_from_frames_various_short_lengths(t):
    probs = np.linspace(0.0, 1.0, num=t)
    p = presence_from_frames(probs)
    if t == 0:
        assert p == 0.0
    else:
        assert 0.0 <= p <= 1.0


def test_presence_from_frames_respects_valid_len_ignores_padding():
    real = np.full(10, 0.1)
    padding = np.full(10, 0.99)
    probs = np.concatenate([real, padding])
    p_with_valid_len = presence_from_frames(probs, valid_len=10)
    p_without = presence_from_frames(probs)
    assert p_with_valid_len < 0.5  # không bị pha bởi padding
    assert p_without > p_with_valid_len  # nếu không cắt valid_len sẽ bị lệch cao hơn hẳn



def test_frame_probs_to_intervals_uses_real_duration_not_estimate():
    probs = np.array([0.9, 0.9, 0.1, 0.1])  # 2 frame đầu vượt threshold
    out = frame_probs_to_intervals(probs, GRID, duration=0.03, threshold=0.5, min_duration=0.0, merge_gap=0.0)
    # interval thô là [0.0, 0.04] nhưng duration thật chỉ 0.03 -> phải bị clip
    assert out == [[0.0, 0.03]]



def test_baseline_never_uses_presence_gate():
    # xác suất thấp đều nhưng có 1 frame vượt threshold trực tiếp
    probs = np.array([0.1, 0.1, 0.6, 0.1])
    intervals, p_presence, branch, raw = predict_fault_type(
        probs, GRID, duration=0.08, tau_frame=0.5, min_duration=0.0, config=BASELINE_POSTPROCESS
    )
    assert branch in ("direct", "no_evidence")  # KHÔNG BAO GIỜ "presence_low" khi baseline
    assert intervals == [[0.04, 0.06]]


def test_baseline_is_deterministic():
    probs = np.array([0.1, 0.7, 0.7, 0.2, 0.05])
    out1 = predict_fault_type(probs, GRID, duration=0.10, config=BASELINE_POSTPROCESS)
    out2 = predict_fault_type(probs, GRID, duration=0.10, config=BASELINE_POSTPROCESS)
    assert out1 == out2


def test_baseline_returns_empty_when_no_frame_crosses_threshold():
    probs = np.full(5, 0.05)
    intervals, _p, branch, raw = predict_fault_type(probs, GRID, duration=0.10, config=BASELINE_POSTPROCESS)
    assert intervals == []
    assert branch == "no_evidence"
    assert raw is False


def test_whole_file_fallback_never_happens():
    # Toàn bộ config bật, xác suất cực thấp everywhere -> vẫn KHÔNG được đoán [[0, duration]]
    full_config = PostProcessConfig(
        use_presence_gate=True, use_relax_threshold=True, use_relax_min_duration=True, use_peak_window=True
    )
    probs = np.full(20, 0.01)
    intervals, _p, branch, _raw = predict_fault_type(
        probs, GRID, duration=0.4, tau_presence=0.9, tau_peak_min=0.9, config=full_config
    )
    assert intervals != [[0.0, 0.4]]
    assert intervals == []



def test_peak_window_at_first_frame_stays_within_bounds():
    probs = np.array([0.9, 0.05, 0.05, 0.05])
    cfg = PostProcessConfig(use_presence_gate=False, use_peak_window=True)
    intervals, _p, branch, _raw = predict_fault_type(
        probs, GRID, duration=0.08, tau_frame=0.99, tau_peak_min=0.5, config=cfg
    )
    assert branch == "peak_window"
    assert intervals[0][0] >= 0.0
    assert intervals[0][1] <= 0.08


def test_peak_window_at_last_frame_stays_within_bounds():
    probs = np.array([0.05, 0.05, 0.05, 0.9])
    cfg = PostProcessConfig(use_presence_gate=False, use_peak_window=True)
    intervals, _p, branch, _raw = predict_fault_type(
        probs, GRID, duration=0.08, tau_frame=0.99, tau_peak_min=0.5, config=cfg
    )
    assert branch == "peak_window"
    assert intervals[0][1] <= 0.08


def test_peak_window_tiny_sequence_one_frame():
    probs = np.array([0.9])
    cfg = PostProcessConfig(use_presence_gate=False, use_peak_window=True)
    intervals, _p, branch, _raw = predict_fault_type(
        probs, GRID, duration=0.02, tau_frame=0.99, tau_peak_min=0.5, config=cfg
    )
    assert branch == "peak_window"
    assert 0.0 <= intervals[0][0] < intervals[0][1] <= 0.02
