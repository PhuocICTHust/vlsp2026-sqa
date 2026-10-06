"""Kiểm tra alignment WavLM: hop lấy từ config, chỉ đo offset và drift.

Chạy: python scripts/run_alignment_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from sqa.frame_grid import MODEL_GRID, SAMPLE_RATE, conv_output_frames, conv_stack_geometry
from sqa.model import WavLMBackbone

BURST_SEC = 0.01


def make_transient(duration_sec: float, transient_at: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    x = np.zeros(int(round(duration_sec * sr)), dtype=np.float32)
    start = int(round(transient_at * sr))
    end = min(len(x), start + int(BURST_SEC * sr))
    rng = np.random.default_rng(0)
    x[start:end] = 0.8 * rng.standard_normal(end - start).astype(np.float32)
    return x


def run_case(backbone: WavLMBackbone, duration_sec: float, transient_at: float, hop: float) -> dict:
    x = make_transient(duration_sec, transient_at)
    with torch.no_grad():
        h = backbone(torch.from_numpy(x).unsqueeze(0)).squeeze(0)
        h_base = backbone(torch.zeros(1, len(x))).squeeze(0)
    activation = (h - h_base).norm(dim=-1).numpy()
    peak_idx = int(activation.argmax())
    center = transient_at + BURST_SEC / 2
    above = np.flatnonzero(activation >= 0.5 * activation.max())
    return {
        "duration_sec": duration_sec,
        "transient_at": transient_at,
        "T": len(activation),
        "T_expected": conv_output_frames(len(x)),
        "peak_frame_idx": peak_idx,
        "residual_ms": round((center - peak_idx * hop) * 1000.0, 2),
        "span50_frames": int(above[-1] - above[0] + 1) if len(above) else 0,
    }


def main():
    backbone = WavLMBackbone(freeze=True)
    backbone.eval()
    cfg = backbone.backbone.config
    hop_samples, rf_samples = conv_stack_geometry(tuple(cfg.conv_kernel), tuple(cfg.conv_stride))
    hop = hop_samples / SAMPLE_RATE
    print(f"conv_kernel={cfg.conv_kernel} conv_stride={cfg.conv_stride}")
    print(f"=> hop = {hop_samples} sample = {hop * 1000:.3f} ms, receptive field (conv) = {rf_samples} sample")
    if abs(hop - MODEL_GRID.frame_hop_sec) > 1e-12:
        raise SystemExit(f"MODEL_GRID.frame_hop_sec={MODEL_GRID.frame_hop_sec} lệch hop lý thuyết {hop}")

    cases = []
    for duration in (1.0, 2.0, 3.0):
        for t in (0.10, 0.50, 0.90, 1.50, 2.00):
            if t + BURST_SEC < duration:
                cases.append(run_case(backbone, duration, t, hop))
    for duration, t in ((10.0, 9.5), (20.0, 10.0), (20.0, 19.5)):
        cases.append(run_case(backbone, duration, t, hop))

    header = list(cases[0].keys())
    print("\t".join(header))
    for c in cases:
        print("\t".join(str(c[k]) for k in header))

    bad_t = [c for c in cases if c["T"] != c["T_expected"]]
    res = np.array([c["residual_ms"] for c in cases])
    pos = np.array([c["transient_at"] for c in cases])
    slope_ms_per_10s = np.polyfit(pos, res, 1)[0] * 10.0

    print(f"\nT == conv_output_frames(N): {'PASS' if not bad_t else f'FAIL ({len(bad_t)} case)'}")
    print(f"residual: mean={res.mean():.2f} ms, std={res.std():.2f} ms, "
          f"min={res.min():.2f}, max={res.max():.2f}")
    print(f"drift: {slope_ms_per_10s:+.2f} ms / 10 s (kỳ vọng ~0)")


if __name__ == "__main__":
    main()