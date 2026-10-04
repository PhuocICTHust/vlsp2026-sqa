"""GT -> frame -> interval -> IoU phải = 1 với frame head hoàn hảo."""
import numpy as np
import pytest

from sqa.evaluator import frame_iou
from sqa.frame_grid import (
    EVALUATOR_GRID,
    MODEL_GRID,
    SAMPLE_RATE,
    ModelFrameGrid,
    conv_output_frames,
    intervals_to_model_frame_targets,
)
from sqa.postprocess import frame_probs_to_intervals, serialize_intervals


def _oracle_iou(gt, duration, grid=MODEL_GRID, true_hop=0.02, round_digits=2):
    n = conv_output_frames(int(round(duration * SAMPLE_RATE)))
    truth_grid = ModelFrameGrid(frame_hop_sec=true_hop, offset_sec=0.0)
    probs = intervals_to_model_frame_targets(gt, n, truth_grid)
    pred = frame_probs_to_intervals(probs, grid, duration, threshold=0.5, min_duration=0.0, merge_gap=0.0)
    pred = serialize_intervals(pred, round_digits=round_digits, duration=duration)
    return frame_iou(pred, gt, duration, EVALUATOR_GRID)


CASES = [
    ([[0.50, 0.70]], 3.0),
    ([[2.00, 2.20]], 3.0),
    ([[9.50, 9.70]], 10.0),
    ([[19.60, 19.80]], 20.0),
    ([[29.50, 29.90]], 30.0),
    ([[0.00, 0.04]], 1.0),
    ([[0.96, 1.00]], 1.0),
    ([[4.10, 7.3337]], 7.3337),
    ([[0.58, 0.60]], 2.0),
    ([[0.20, 0.26], [0.30, 0.34], [1.14, 1.88]], 2.5),
    ([[0.00, 5.00]], 5.0),
]


@pytest.mark.parametrize("gt, duration", CASES)
def test_perfect_frame_head_roundtrips_to_iou_one(gt, duration):
    assert _oracle_iou(gt, duration) == pytest.approx(1.0, abs=1e-6)


def test_roundtrip_randomized():
    rng = np.random.default_rng(1337)
    for _ in range(300):
        duration = float(np.round(rng.uniform(0.5, 30.0), 3))
        n_iv = int(rng.integers(1, 4))
        starts = np.sort(rng.uniform(0, duration - 0.04, n_iv))
        gt = []
        for s in starts:
            s = round(float(s), 2)
            e = round(min(duration, s + float(rng.uniform(0.02, 2.0))), 2)
            if gt and s <= gt[-1][1] + 0.02:
                continue
            if e > s:
                gt.append([s, e])
        # GT chạm frame WavLM cuối -> quy ước kéo tới duration
        last_frame_start = (conv_output_frames(int(round(duration * SAMPLE_RATE))) - 1) * 0.02
        if gt and gt[-1][1] > last_frame_start + 1e-9:
            gt[-1][1] = duration
        assert _oracle_iou(gt, duration) == pytest.approx(1.0, abs=1e-6), (gt, duration)


def test_fitted_hop_drift_is_harmful():
    old = ModelFrameGrid(frame_hop_sec=0.020052601, offset_sec=0.015194489)
    assert _oracle_iou([[19.60, 19.80]], 20.0, grid=old) < 0.6
    assert _oracle_iou([[19.60, 19.80]], 20.0, grid=MODEL_GRID) == pytest.approx(1.0)


def test_tail_extension_tradeoff_is_at_most_one_evaluator_frame():
    iou = _oracle_iou([[0.60, 0.74]], 0.754)
    assert iou == pytest.approx(7 / 8)


def test_tail_extension_only_bridges_missing_wavlm_tail():
    probs = np.ones(10, dtype=np.float32)  # 0.2 s
    out = frame_probs_to_intervals(probs, MODEL_GRID, duration=5.0, min_duration=0.0, merge_gap=0.0)
    assert out == [[0.0, pytest.approx(0.2)]]
    out = frame_probs_to_intervals(probs, MODEL_GRID, duration=0.215, min_duration=0.0, merge_gap=0.0)
    assert out == [[0.0, 0.215]]


def test_serialize_keeps_duration_end_unrounded():
    out = serialize_intervals([[0.05, 1.204]], round_digits=2, duration=1.204)
    assert out == [[0.05, 1.204]]
    out = serialize_intervals([[0.0, 1.2049]], round_digits=2, duration=1.2049)
    assert out[0][1] <= 1.2049