"""
Extract WavLM-Large feature cache for Vietnamese Augmentation dataset (1,000 audio files).
Saves to data/vietnamese_aug/wavlm_cache/<stem>.pt
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WavLMModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.model import normalize_waveform

WAV_DIR = ROOT / "data" / "vietnamese_aug" / "wav"
OUT_CACHE = ROOT / "data" / "vietnamese_aug" / "wavlm_cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio_mono(path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, float]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    duration = float(len(wav)) / float(target_sr)
    tensor = torch.from_numpy(np.clip(wav, -1.0, 1.0).astype(np.float32))
    return tensor, duration


def main():
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(WAV_DIR.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files in {WAV_DIR}")
    print(f"Target cache folder: {OUT_CACHE}")
    print(f"Device: {DEVICE}")

    pending = [p for p in wav_files if not (OUT_CACHE / f"{p.stem}.pt").exists()]
    print(f"Already cached: {len(wav_files) - len(pending)} | Pending: {len(pending)}")
    if not pending:
        print("All files already cached!")
        return

    print("Loading WavLM-Large backbone...")
    backbone = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad = False

    t0 = time.time()
    with torch.no_grad():
        for i, wav_path in enumerate(pending, start=1):
            wav, dur = load_audio_mono(wav_path)
            wav_gpu = wav.to(DEVICE)[None]
            mask = torch.ones_like(wav_gpu, dtype=torch.long)
            wav_norm = normalize_waveform(wav_gpu, mask)
            hidden = backbone(wav_norm, attention_mask=mask).last_hidden_state[0].cpu().float()
            valid_mask = torch.ones(hidden.shape[0], dtype=torch.bool)

            out_pt = OUT_CACHE / f"{wav_path.stem}.pt"
            torch.save(
                {
                    "hidden": hidden,
                    "valid_mask": valid_mask,
                    "target": 3.0,
                    "file_id": wav_path.name,
                    "duration": dur,
                },
                out_pt,
            )

            if i % 100 == 0 or i == len(pending):
                elapsed = time.time() - t0
                speed = i / elapsed
                rem = (len(pending) - i) / speed if speed > 0 else 0
                print(f"Extracted {i:4d}/{len(pending)} files ({speed:.1f} files/s, ~{rem:.0f}s remaining)...")

    print(f"Finished caching all {len(wav_files)} files in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
