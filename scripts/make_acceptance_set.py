from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000


def _tone(duration_sec: float, freq: float = 220.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _add_noise(x: np.ndarray, start: float, end: float, sr: int = SR, level: float = 0.15) -> np.ndarray:
    out = x.copy()
    s, e = int(start * sr), min(len(x), int(end * sr))
    rng = np.random.default_rng(0)
    out[s:e] += (level * rng.standard_normal(e - s)).astype(np.float32)
    return out


def _add_clipping(x: np.ndarray, start: float, end: float, sr: int = SR, ceiling: float = 0.05) -> np.ndarray:
    out = x.copy()
    s, e = int(start * sr), min(len(x), int(end * sr))
    out[s:e] = np.clip(out[s:e] * 8.0, -ceiling, ceiling) * (0.3 / ceiling)
    return out


def _add_silence_gap(x: np.ndarray, start: float, end: float, sr: int = SR) -> np.ndarray:
    out = x.copy()
    s, e = int(start * sr), min(len(x), int(end * sr))
    out[s:e] = 0.0
    return out


def build_acceptance_set(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    # 1. Không có lỗi nào
    x = _tone(1.5)
    sf.write(out_dir / "01_no_fault.wav", x, SR)
    manifest["01_no_fault.wav"] = {"expected_case": "no_fault", "duration": 1.5}

    # 2-4. Mỗi loại lỗi riêng lẻ
    x = _add_noise(_tone(1.5), 0.5, 1.0)
    sf.write(out_dir / "02_only_noise.wav", x, SR)
    manifest["02_only_noise.wav"] = {"expected_case": "noise_only", "duration": 1.5}

    x = _add_clipping(_tone(1.5), 0.5, 1.0)
    sf.write(out_dir / "03_only_distortion.wav", x, SR)
    manifest["03_only_distortion.wav"] = {"expected_case": "distortion_only", "duration": 1.5}

    x = _add_silence_gap(_tone(1.5), 0.5, 0.9)
    sf.write(out_dir / "04_only_unnatural_pause.wav", x, SR)
    manifest["04_only_unnatural_pause.wav"] = {"expected_case": "unnatural_pause_only", "duration": 1.5}

    # 5. Nhiều loại lỗi cùng lúc
    x = _tone(2.0)
    x = _add_noise(x, 0.3, 0.7)
    x = _add_clipping(x, 1.0, 1.3)
    x = _add_silence_gap(x, 1.6, 1.8)
    sf.write(out_dir / "05_multi_fault.wav", x, SR)
    manifest["05_multi_fault.wav"] = {"expected_case": "multi_fault", "duration": 2.0}

    # 6. Nhiều interval cho CÙNG 1 loại lỗi
    x = _tone(2.0)
    x = _add_noise(x, 0.2, 0.5)
    x = _add_noise(x, 1.2, 1.5)
    sf.write(out_dir / "06_multi_interval_same_type.wav", x, SR)
    manifest["06_multi_interval_same_type.wav"] = {"expected_case": "multi_interval_noise", "duration": 2.0}

    # 7a-c. Độ dài audio khác nhau — lộ sớm vấn đề duration nếu có
    for i, dur in enumerate([0.5, 3.0, 6.0], start=1):
        x = _add_noise(_tone(dur), dur * 0.3, dur * 0.6)
        fname = f"07{chr(96 + i)}_duration_{dur:g}s.wav"
        sf.write(out_dir / fname, x, SR)
        manifest[fname] = {"expected_case": "duration_variety", "duration": dur}

    manifest_path = out_dir / "manifest.json"
    import json

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/acceptance_set")
    args = parser.parse_args()
    m = build_acceptance_set(Path(args.out))
    print(f"Đã sinh {len(m)} file audio vào {args.out}/")
    for name in m:
        print(" -", name)
