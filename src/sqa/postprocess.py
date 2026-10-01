from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame_grid import ModelFrameGrid


@dataclass(frozen=True)
class PostProcessConfig:

    use_presence_gate: bool = False
    use_relax_threshold: bool = False
    use_relax_min_duration: bool = False
    use_peak_window: bool = False


BASELINE_POSTPROCESS = PostProcessConfig()


def normalize_intervals(
    intervals: list[list[float]],
    duration: float,
    merge_gap: float = 0.0,
    min_duration: float = 0.0,
    round_digits: int | None = None,
) -> list[list[float]]:
    iv = [[float(s), float(e)] for s, e in intervals if float(s) < float(e)]
    iv = [[max(0.0, s), min(duration, e)] for s, e in iv if max(0.0, s) < min(duration, e)]
    iv.sort(key=lambda x: x[0])

    merged: list[list[float]] = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        elif merged and merge_gap > 0 and (s - merged[-1][1]) < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    merged = [[s, e] for s, e in merged if (e - s) >= min_duration]
    if round_digits is None:
        return merged
    return [[round(s, round_digits), round(e, round_digits)] for s, e in merged]


def serialize_intervals(intervals: list[list[float]], round_digits: int = 2) -> list[list[float]]:
    return [[round(s, round_digits), round(e, round_digits)] for s, e in intervals]


def presence_from_frames(
    frame_probs: np.ndarray, valid_len: int | None = None, k_ratio: float = 0.05, min_k: int = 3
) -> float:
    probs = frame_probs if valid_len is None else frame_probs[:valid_len]
    t = len(probs)
    if t == 0:
        return 0.0
    k = min(t, max(min_k, int(round(k_ratio * t))))
    return float(np.sort(probs)[-k:].mean())


def frame_probs_to_intervals(
    probs: np.ndarray,
    grid: ModelFrameGrid,
    duration: float | None = None,
    threshold: float = 0.5,
    min_duration: float = 0.10,
    merge_gap: float = 0.08,
) -> list[list[float]]:
    if duration is None:
        duration = len(probs) * grid.frame_hop_sec + grid.offset_sec

    binary = probs > threshold
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(binary):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(binary)))

    raw = [
        [s * grid.frame_hop_sec + grid.offset_sec, e * grid.frame_hop_sec + grid.offset_sec]
        for s, e in runs
    ]
    return normalize_intervals(raw, duration=duration, merge_gap=merge_gap, min_duration=min_duration)


def predict_fault_type(
    frame_probs: np.ndarray,
    grid: ModelFrameGrid,
    duration: float,
    tau_presence: float = 0.5,
    tau_frame: float = 0.5,
    tau_low: float = 0.2,
    min_duration: float = 0.10,
    merge_gap: float = 0.08,
    tau_peak_min: float = 0.3,
    valid_len: int | None = None,
    config: PostProcessConfig = BASELINE_POSTPROCESS,
) -> tuple[list[list[float]], float, str, bool]:
    probs = frame_probs if valid_len is None else frame_probs[:valid_len]
    if len(probs) == 0:
        return [], 0.0, "empty_audio", False

    raw_frame_presence = bool((probs > tau_frame).any())
    p_presence = presence_from_frames(probs)

    if config.use_presence_gate and p_presence < tau_presence:
        return [], p_presence, "presence_low", raw_frame_presence

    def _clip(intervals: list[list[float]]) -> list[list[float]]:
        return [[max(0.0, s), min(duration, e)] for s, e in intervals if max(0.0, s) < min(duration, e)]

    peak_prob = float(probs.max())
    relaxed_configs = [{"name": "direct", "threshold": tau_frame, "min_duration": min_duration}]
    if config.use_relax_threshold:
        relaxed_configs.append(
            {"name": "relax_threshold", "threshold": max(0.5 * peak_prob, tau_low), "min_duration": min_duration}
        )
    if config.use_relax_min_duration:
        relaxed_configs.append(
            {"name": "relax_min_dur", "threshold": max(0.5 * peak_prob, tau_low), "min_duration": 0.0}
        )

    for cfg in relaxed_configs:
        intervals = _clip(
            frame_probs_to_intervals(probs, grid, duration, cfg["threshold"], cfg["min_duration"], merge_gap)
        )
        if intervals:
            return intervals, p_presence, cfg["name"], raw_frame_presence

    if config.use_peak_window and peak_prob >= tau_peak_min:
        peak_idx = int(probs.argmax())
        half_win_frames = int(round(0.15 / grid.frame_hop_sec))
        s = max(0, peak_idx - half_win_frames)
        e = min(len(probs), peak_idx + half_win_frames)
        interval = _clip([[s * grid.frame_hop_sec + grid.offset_sec, e * grid.frame_hop_sec + grid.offset_sec]])
        if interval:
            return interval, p_presence, "peak_window", raw_frame_presence

    return [], p_presence, "no_evidence", raw_frame_presence
