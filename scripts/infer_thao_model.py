"""
Inference script for Thảo's wav2vec2-xls-r-300m-lora model on QualiSpeech test.csv.
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import Wav2Vec2Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "qualispeech" / "test.csv"
WAV_DIR = ROOT / "data" / "qualispeech" / "wav"
OUT_CSV = ROOT / "outputs" / "thao_xlsr_test_preds.csv"


class AttentionPool(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.proj = nn.Linear(h, 1)

    def forward(self, h, m=None):
        s = self.proj(h).squeeze(-1)
        if m is not None:
            s = s.masked_fill(~m, float("-inf"))
        return (F.softmax(s, dim=1).unsqueeze(-1) * h).sum(dim=1)


class QualityHead(nn.Module):
    def __init__(self, h, mlp_hidden=256, dropout=0.1):
        super().__init__()
        self.pool = AttentionPool(h)
        layers = [
            nn.Linear(h, mlp_hidden),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, h, m=None):
        return torch.sigmoid(self.mlp(self.pool(h, m)).squeeze(-1)) * 4.0 + 1.0


def load_audio(path: Path, sr_out: int = 16000) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if sr != sr_out:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, sr_out)
        wav = resample_poly(wav, sr_out // g, sr // g).astype(np.float32)
    return torch.from_numpy(np.clip(wav, -1.0, 1.0).astype(np.float32))


def main():
    print("=" * 60)
    print("INFERENCE: THẢO WAV2VEC2-XLS-R-300M ON TEST.CSV")
    print("=" * 60)
    print("Device:", DEVICE)

    print("Downloading / locating checkpoint from Hugging Face...")
    model_dir = Path(snapshot_download("vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-lora"))
    print("Model dir:", model_dir)

    cfg = json.loads((model_dir / "sqa_config.json").read_text())
    print("Loaded sqa_config.json:", cfg["base_model"])

    backbone = Wav2Vec2Model.from_pretrained(model_dir / "backbone").eval().to(DEVICE)
    head = QualityHead(cfg["hidden_size"], cfg["head"]["mlp_hidden"], cfg["head"].get("dropout", 0.1)).eval().to(DEVICE)
    head.load_state_dict(load_file(model_dir / "quality_head.safetensors"))

    df_test = pd.read_csv(TEST_CSV)
    print(f"Total test samples: {len(df_test)}")

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for i, (_, row) in enumerate(df_test.iterrows(), start=1):
            wav_name = str(row["id"]).strip()
            if not wav_name.endswith(".wav"):
                wav_name += ".wav"
            wav_path = WAV_DIR / wav_name
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing audio: {wav_path}")

            wav = load_audio(wav_path).to(DEVICE)[None]  # [1, T]
            # Normalization as configured in Thao's model
            wav = (wav - wav.mean(dim=1, keepdim=True)) / torch.sqrt(
                wav.var(dim=1, unbiased=False, keepdim=True) + 1e-7
            )
            mask = torch.ones_like(wav, dtype=torch.long)
            h = backbone(wav, attention_mask=mask).last_hidden_state
            v_mask = torch.ones(h.shape[:2], dtype=torch.bool, device=DEVICE)
            pred = float(head(h.float(), v_mask).item())

            all_preds.append(pred)
            all_ids.append(wav_name)
            all_targets.append(float(row["Overall quality"]))

            if i % 200 == 0 or i == len(df_test):
                print(f"Processed {i:4d}/{len(df_test)} clips...")

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    pcc = float(np.corrcoef(preds_arr, targets_arr)[0, 1])
    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    vx, vy = np.var(preds_arr), np.var(targets_arr)
    mx, my = np.mean(preds_arr), np.mean(targets_arr)
    cov = np.mean((preds_arr - mx) * (targets_arr - my))
    ccc = float(2.0 * cov / (vx + vy + (mx - my) ** 2 + 1e-8))

    print()
    print("=" * 60)
    print(f"THẢO MODEL METRICS ON TEST.CSV ({len(df_test)} clips):")
    print(f"  PCC: {pcc:.4f}")
    print(f"  CCC: {ccc:.4f}")
    print(f"  MSE: {mse:.4f}")
    print("=" * 60)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"id": all_ids, "ground_truth": targets_arr, "prediction": preds_arr})
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved predictions to: {OUT_CSV}")


if __name__ == "__main__":
    main()
