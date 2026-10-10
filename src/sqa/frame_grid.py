"""ModelFrameGrid: frame geometry and time-axis mappings."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
WAVLM_CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
WAVLM_CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)

FRAME_INDEX_EPS = 1e-6 


def conv_stack_geometry(kernels=WAVLM_CONV_KERNELS, strides=WAVLM_CONV_STRIDES) -> tuple[int, int]:
    """Compute total stride and receptive field in samples."""
    if len(kernels) != len(strides):
        raise ValueError(f"Length mismatch: {kernels} vs {strides}")
    rf, jump = 1, 1
    for k, s in zip(kernels, strides):
        rf += (k - 1) * jump
        jump *= s
    return jump, rf


WAVLM_HOP_SAMPLES, WAVLM_RECEPTIVE_FIELD_SAMPLES = conv_stack_geometry()


def conv_output_frames(
    num_samples: int,
    hop_samples: int = WAVLM_HOP_SAMPLES,
    receptive_field_samples: int = WAVLM_RECEPTIVE_FIELD_SAMPLES,
) -> int:
    """T = floor((N - 400) / 320) + 1."""
    if num_samples < receptive_field_samples:
        return 0
    return (num_samples - receptive_field_samples) // hop_samples + 1


def time_to_frame_start(t: float, hop: float, offset: float = 0.0) -> int:
    return int(math.floor((t - offset) / hop + FRAME_INDEX_EPS))


def time_to_frame_end(t: float, hop: float, offset: float = 0.0) -> int:
    return int(math.ceil((t - offset) / hop - FRAME_INDEX_EPS))


@dataclass(frozen=True)
class ModelFrameGrid:
    frame_hop_sec: float = 0.02
    offset_sec: float = 0.0
    frame_duration_sec: float = 0.02
    convention: str = "frame_start"
    source: str = "default_unverified"

    def frame_output_slot(self, i: int) -> tuple[float, float]:
        s = i * self.frame_hop_sec + self.offset_sec
        return s, s + self.frame_hop_sec

    def frame_receptive_field(self, i: int) -> tuple[float, float]:
        s = i * self.frame_hop_sec + self.offset_sec
        return s, s + self.frame_duration_sec

    def n_frames(self, duration_sec: float) -> int:
        return max(0, math.ceil((duration_sec - self.offset_sec) / self.frame_hop_sec))


@dataclass(frozen=True)
class EvaluatorGrid:
    frame_hop_sec: float = 0.02
    offset_sec: float = 0.0


def intervals_to_model_frame_targets(
    intervals,
    n_frames: int,
    grid: ModelFrameGrid,
) -> np.ndarray:
    """Convert time intervals (seconds) to frame targets [n_frames]."""
    target = np.zeros(n_frames, dtype=np.float32)
    for start, end in intervals:
        if not end > start:
            continue
        s = max(0, time_to_frame_start(start, grid.frame_hop_sec, grid.offset_sec))
        e = min(n_frames, time_to_frame_end(end, grid.frame_hop_sec, grid.offset_sec))
        if e > s:
            target[s:e] = 1.0
    return target


MODEL_GRID = ModelFrameGrid(
    frame_hop_sec=WAVLM_HOP_SAMPLES / SAMPLE_RATE,  # 320 sample = 20 ms, không fit
    offset_sec=0.0,
    frame_duration_sec=WAVLM_RECEPTIVE_FIELD_SAMPLES / SAMPLE_RATE,
    source="theoretical_conv_stride_320",
)

EVALUATOR_GRID = EvaluatorGrid()