import math
import numpy as np

from sqa.frame_grid import EvaluatorGrid, ModelFrameGrid


def test_frame_output_slot_uses_hop_only_not_receptive_field():
    grid = ModelFrameGrid(frame_hop_sec=0.02, offset_sec=0.0, frame_duration_sec=0.04)
    s, e = grid.frame_output_slot(5)
    assert math.isclose(s, 0.10)
    assert math.isclose(e, 0.12)  # 0.10 + hop (0.02), KHÔNG phải + frame_duration_sec (0.04)


def test_frame_receptive_field_uses_frame_duration():
    grid = ModelFrameGrid(frame_hop_sec=0.02, offset_sec=0.0, frame_duration_sec=0.04)
    s, e = grid.frame_receptive_field(5)
    assert math.isclose(s, 0.10)
    assert math.isclose(e, 0.14)  # 0.10 + frame_duration_sec (0.04)


def test_offset_shifts_frame_0():
    grid = ModelFrameGrid(frame_hop_sec=0.02, offset_sec=0.01)
    s, e = grid.frame_output_slot(0)
    assert math.isclose(s, 0.01)
    assert math.isclose(e, 0.03)


def test_n_frames_matches_evaluator_grid_convention():
    grid = ModelFrameGrid(frame_hop_sec=0.02, offset_sec=0.0)
    # 1.0s / 0.02s = đúng 50 frame
    assert grid.n_frames(1.0) == 50
    # 1.005s -> ceil(50.25) = 51
    assert grid.n_frames(1.005) == 51


def test_evaluator_grid_is_separate_instance_from_model_grid():
    eg = EvaluatorGrid()
    mg = ModelFrameGrid()
    assert eg is not mg
    assert not hasattr(eg, "frame_duration_sec")  # EvaluatorGrid không có khái niệm receptive field


from sqa.frame_grid import (  # noqa: E402
    MODEL_GRID,
    SAMPLE_RATE,
    WAVLM_HOP_SAMPLES,
    WAVLM_RECEPTIVE_FIELD_SAMPLES,
    conv_output_frames,
    conv_stack_geometry,
    intervals_to_model_frame_targets,
)


def test_wavlm_conv_geometry_is_320_hop_400_receptive_field():
    assert conv_stack_geometry() == (320, 400)
    assert (WAVLM_HOP_SAMPLES, WAVLM_RECEPTIVE_FIELD_SAMPLES) == (320, 400)


def test_model_grid_hop_is_exactly_20ms_and_offset_zero():
    assert MODEL_GRID.frame_hop_sec * SAMPLE_RATE == 320
    assert MODEL_GRID.frame_hop_sec == EvaluatorGrid().frame_hop_sec
    assert MODEL_GRID.offset_sec == 0.0
    assert MODEL_GRID.source == "theoretical_conv_stride_320"


def test_conv_output_frames_matches_hf_formula():
    def hf(n):
        for k, s in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
            n = (n - k) // s + 1
        return max(n, 0)

    for n in [399, 400, 401, 719, 720, 16_000, 16_001, 48_000, 160_000, 480_123]:
        assert conv_output_frames(n) == hf(n), n


def test_model_frame_targets_float_safe_on_rounded_boundaries():
    t = intervals_to_model_frame_targets([[0.58, 0.62]], n_frames=40, grid=MODEL_GRID)
    assert np.flatnonzero(t).tolist() == [29, 30]


def test_model_frame_targets_clip_and_ignore_degenerate():
    t = intervals_to_model_frame_targets([[0.9, 5.0], [0.3, 0.3], [0.5, 0.2]], n_frames=50, grid=MODEL_GRID)
    assert np.flatnonzero(t).tolist() == list(range(45, 50))