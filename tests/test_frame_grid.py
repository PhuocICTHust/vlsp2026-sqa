import math

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
