from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .evaluator import FAULT_TYPES
from .frame_grid import MODEL_GRID, ModelFrameGrid
from .model import SQAModel
from collections.abc import Mapping

from .postprocess import (
    BASELINE_POSTPROCESS,
    PostProcessConfig,
    predict_fault_type,
    resolve_config,
    serialize_intervals,
)

TARGET_SAMPLE_RATE = 16_000


def load_audio_mono_16k(path: str | Path) -> tuple[np.ndarray, float]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # convert to mono
    if sr != TARGET_SAMPLE_RATE:
        duration = len(data) / sr
        n_target = int(round(duration * TARGET_SAMPLE_RATE))
        x_old = np.linspace(0, 1, num=len(data), endpoint=False)
        x_new = np.linspace(0, 1, num=n_target, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
        sr = TARGET_SAMPLE_RATE
    duration = len(data) / sr
    return data, duration


def build_submission_entry(
    file_id: str,
    wav: np.ndarray,
    duration: float,
    model: SQAModel,
    grid: ModelFrameGrid = MODEL_GRID,
    postprocess_config: PostProcessConfig | Mapping[str, PostProcessConfig] = BASELINE_POSTPROCESS,
    device: str = "cpu",
) -> dict:
    model.eval()
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(device)  # [1, num_samples]

    with torch.no_grad():
        quality, fault_probs, _ = model(wav_t)

    quality_score = float(quality.squeeze(0).cpu().item())
    fault_probs_np = fault_probs.squeeze(0).cpu().numpy()  # [T, 3]

    faults = {}
    for i, ft in enumerate(FAULT_TYPES):
        intervals, _p_presence, _branch, _raw = predict_fault_type(
            fault_probs_np[:, i], grid, duration, config=resolve_config(postprocess_config, ft)
        )
        faults[ft] = serialize_intervals(intervals, round_digits=2, duration=duration)

    return {"file_id": file_id, "quality_score": round(quality_score, 3), "faults": faults}


def run_on_folder(
    audio_dir: str | Path,
    model: SQAModel,
    grid: ModelFrameGrid = MODEL_GRID,
    postprocess_config: PostProcessConfig | Mapping[str, PostProcessConfig] = BASELINE_POSTPROCESS,
    device: str = "cpu",
) -> tuple[list[dict], dict]:
    audio_dir = Path(audio_dir)
    entries, duration_lookup = [], {}
    for wav_path in sorted(audio_dir.glob("*.wav")):
        file_id = wav_path.name
        wav, duration = load_audio_mono_16k(wav_path)
        entry = build_submission_entry(file_id, wav, duration, model, grid, postprocess_config, device)
        entries.append(entry)
        duration_lookup[file_id] = duration
    return entries, duration_lookup