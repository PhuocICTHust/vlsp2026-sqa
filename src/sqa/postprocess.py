from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from .frame_grid import ModelFrameGrid


@dataclass(frozen=True)
class PostProcessConfig:
    """Post-processing configuration for predicted fault intervals."""
    tau_frame: float = 0.5
    min_duration: float = 0.0
    merge_gap: float = 0.0
    use_presence_gate: bool = False
    tau_presence: float = 0.5
    presence_k_ratio: float = 0.05
    presence_min_k: int = 3
    use_relax_threshold: bool = False
    use_relax_min_duration: bool = False
    tau_low: float = 0.2
    use_peak_window: bool = False
    tau_peak_min: float = 0.3
    peak_half_window_sec: float = 0.15
    use_hysteresis: bool = False
    legacy_competition_mode: bool = False

    def __post_init__(self):
        for name in ("tau_frame", "tau_presence", "tau_low", "tau_peak_min"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} phải trong [0, 1], nhận {v}")
        for name in ("min_duration", "merge_gap", "peak_half_window_sec", "presence_k_ratio"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} không được âm, nhận {getattr(self, name)}")
        if self.presence_min_k < 1:
            raise ValueError(f"presence_min_k phải >= 1, nhận {self.presence_min_k}")

    @property
    def is_baseline(self) -> bool:
        return (
            self.min_duration == 0.0
            and self.merge_gap == 0.0
            and not (
                self.use_presence_gate
                or self.use_relax_threshold
                or self.use_relax_min_duration
                or self.use_peak_window
                or self.use_hysteresis
            )
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "is_baseline": self.is_baseline}


BASELINE_POSTPROCESS = PostProcessConfig()


def resolve_config(
    config: PostProcessConfig | Mapping[str, PostProcessConfig], fault_type: str
) -> PostProcessConfig:
    if isinstance(config, PostProcessConfig):
        return config
    if fault_type not in config:
        raise KeyError(f"thiếu PostProcessConfig cho fault type '{fault_type}'")
    return config[fault_type]


def normalize_intervals(
    intervals: list[list[float]],
    duration: float,
    merge_gap: float = 0.0,
    min_duration: float = 0.0,
    round_digits: int | None = None,
    legacy_competition_mode: bool = False,
) -> list[list[float]]:
    iv = [[float(s), float(e)] for s, e in intervals if float(s) < float(e)]
    iv = [[max(0.0, s), min(duration, e)] for s, e in iv if max(0.0, s) < min(duration, e)]
    iv.sort(key=lambda x: x[0])

    merged: list[list[float]] = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        elif merged and merge_gap > 0 and (
            (s - merged[-1][1]) < merge_gap if legacy_competition_mode else round(s - merged[-1][1], 6) < merge_gap
        ):
            merged[-1][1] = e
        else:
            merged.append([s, e])

    if legacy_competition_mode:
        merged = [[s, e] for s, e in merged if (e - s) >= min_duration]
    else:
        merged = [[s, e] for s, e in merged if (e - s) >= (min_duration - 1e-6)]

    if round_digits is None:
        return merged
    return [[round(s, round_digits), round(e, round_digits)] for s, e in merged]


def serialize_intervals(
    intervals: list[list[float]], round_digits: int = 2, duration: float | None = None
) -> list[list[float]]:
    """Serialize interval timestamps to list of floats."""
    out = []
    for s, e in intervals:
        s_out = round(s, round_digits)
        if duration is not None and e >= duration - 1e-9:
            e_out = float(duration)
        else:
            e_out = round(e, round_digits)
            if duration is not None:
                e_out = min(e_out, float(duration))
        out.append([max(0.0, s_out), e_out])
    return out


def presence_from_frames(
    frame_probs: np.ndarray, valid_len: int | None = None, k_ratio: float = 0.05, min_k: int = 3
) -> float:
    probs = frame_probs if valid_len is None else frame_probs[:valid_len]
    t = len(probs)
    if t == 0:
        return 0.0
    k = min(t, max(min_k, int(round(k_ratio * t))))
    return float(np.sort(probs)[-k:].mean())


def hysteresis_threshold(probs: np.ndarray, tau_high: float, tau_low: float) -> np.ndarray:
    """
    Two-threshold hysteresis:
    Keep connected segments above tau_low if and only if they contain at least one point > tau_high.
    """
    above_low = probs > tau_low
    above_high = probs > tau_high
    out = np.zeros_like(above_low, dtype=bool)
    n = len(probs)
    i = 0
    while i < n:
        if above_low[i]:
            start = i
            has_high = False
            while i < n and above_low[i]:
                if above_high[i]:
                    has_high = True
                i += 1
            if has_high:
                out[start:i] = True
        else:
            i += 1
    return out


def frame_probs_to_intervals(
    probs: np.ndarray,
    grid: ModelFrameGrid,
    duration: float | None = None,
    threshold: float = 0.5,
    min_duration: float = 0.0,
    merge_gap: float = 0.0,
    tau_low: float | None = None,
    legacy_competition_mode: bool = False,
) -> list[list[float]]:
    if duration is None:
        duration = len(probs) * grid.frame_hop_sec + grid.offset_sec

    if tau_low is not None and 0.0 <= tau_low < threshold:
        binary = hysteresis_threshold(probs, tau_high=threshold, tau_low=tau_low)
    else:
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

    # Extend tail interval to file duration if within boundary
    n = len(binary)
    tail_end = n * grid.frame_hop_sec + grid.offset_sec
    extend_tail = 0.0 <= duration - tail_end < 2 * grid.frame_hop_sec
    raw = [
        [
            s * grid.frame_hop_sec + grid.offset_sec,
            duration if (e == n and extend_tail) else e * grid.frame_hop_sec + grid.offset_sec,
        ]
        for s, e in runs
    ]
    return normalize_intervals(
        raw,
        duration=duration,
        merge_gap=merge_gap,
        min_duration=min_duration,
        legacy_competition_mode=legacy_competition_mode,
    )


def predict_fault_type(
    frame_probs: np.ndarray,
    grid: ModelFrameGrid,
    duration: float,
    config: PostProcessConfig = BASELINE_POSTPROCESS,
    valid_len: int | None = None,
) -> tuple[list[list[float]], float, str, bool]:
    probs = frame_probs if valid_len is None else frame_probs[:valid_len]
    if len(probs) == 0:
        return [], 0.0, "empty_audio", False

    raw_frame_presence = bool((probs > config.tau_frame).any())
    p_presence = presence_from_frames(probs, k_ratio=config.presence_k_ratio, min_k=config.presence_min_k)

    if config.use_presence_gate and p_presence < config.tau_presence:
        return [], p_presence, "presence_low", raw_frame_presence

    def _clip(intervals: list[list[float]]) -> list[list[float]]:
        return [[max(0.0, s), min(duration, e)] for s, e in intervals if max(0.0, s) < min(duration, e)]

    peak_prob = float(probs.max())
    relaxed_tau = max(0.5 * peak_prob, config.tau_low)
    attempts = [("direct", config.tau_frame, config.min_duration)]
    if config.use_relax_threshold:
        attempts.append(("relax_threshold", relaxed_tau, config.min_duration))
    if config.use_relax_min_duration:
        attempts.append(("relax_min_dur", relaxed_tau, 0.0))

    hyst_tau_low = config.tau_low if config.use_hysteresis else None

    for name, threshold, min_duration in attempts:
        intervals = _clip(
            frame_probs_to_intervals(
                probs,
                grid,
                duration,
                threshold=threshold,
                min_duration=min_duration,
                merge_gap=config.merge_gap,
                tau_low=hyst_tau_low,
                legacy_competition_mode=config.legacy_competition_mode,
            )
        )
        if intervals:
            return intervals, p_presence, name, raw_frame_presence

    if config.use_peak_window and peak_prob >= config.tau_peak_min:
        peak_idx = int(probs.argmax())
        half_win_frames = int(round(config.peak_half_window_sec / grid.frame_hop_sec))
        s = max(0, peak_idx - half_win_frames)
        e = min(len(probs), peak_idx + half_win_frames)
        interval = _clip([[s * grid.frame_hop_sec + grid.offset_sec, e * grid.frame_hop_sec + grid.offset_sec]])
        if interval:
            return interval, p_presence, "peak_window", raw_frame_presence

    return [], p_presence, "no_evidence", raw_frame_presence