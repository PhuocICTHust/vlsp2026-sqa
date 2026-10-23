import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
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
    tensor = torch.from_numpy(wav).float()
    return tensor, duration

def extract_folder(wav_dir: Path, out_dir: Path, model: AutoModel):
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(wav_dir.glob("*.wav"))
    pending = [p for p in wav_files if not (out_dir / f"{p.stem}.pt").exists()]
    print(f"\n[{wav_dir.parent.name}] Total: {len(wav_files)} | Pending: {len(pending)}", flush=True)
    if not pending:
        print("  All files already cached!", flush=True)
        return

    t0 = time.time()
    with torch.no_grad():
        for i, wav_path in enumerate(pending, start=1):
            wav, dur = load_audio_mono(wav_path)
            wav_gpu = wav.to(DEVICE).unsqueeze(0)
            hidden = model(wav_gpu).last_hidden_state[0].cpu().float()
            
            out_pt = out_dir / f"{wav_path.stem}.pt"
            torch.save(
                {
                    "hidden": hidden,
                    "duration": dur,
                    "file_id": wav_path.name,
                },
                out_pt,
            )
            if i % 100 == 0 or i == len(pending):
                elapsed = time.time() - t0
                spd = i / elapsed if elapsed > 0 else 0
                rem = (len(pending) - i) / spd if spd > 0 else 0
                print(f"  Cached {i:4d}/{len(pending)} ({spd:.1f} fps, ~{rem:.0f}s left)", flush=True)
    print(f"  Done in {time.time() - t0:.1f}s!", flush=True)

def main():
    print("=" * 65, flush=True)
    print("TRÍCH XUẤT XLS-R FEATURE CACHE CHO 2,125 AUDIO TIẾNG VIỆT", flush=True)
    print("=" * 65, flush=True)
    print(f"Device: {DEVICE}", flush=True)

    print("Loading facebook/wav2vec2-xls-r-300m...", flush=True)
    model = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m").to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    extract_folder(ROOT / "data" / "vietnamese_test_v1" / "wav", ROOT / "data" / "vietnamese_test_v1" / "xlsr_cache", model)

    extract_folder(ROOT / "data" / "vietnamese_aug_v3" / "train" / "wav", ROOT / "data" / "vietnamese_aug_v3" / "train" / "xlsr_cache", model)

    extract_folder(ROOT / "data" / "vietnamese_aug" / "wav", ROOT / "data" / "vietnamese_aug" / "xlsr_cache", model)

    print("\nXLS-R feature extraction completed successfully!", flush=True)

if __name__ == "__main__":
    main()
