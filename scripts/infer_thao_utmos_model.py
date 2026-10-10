"""
Inference script for Thảo's new model:
vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval
Evaluates on QualiSpeech test.csv (1,852 samples).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import transformers
from transformers import Wav2Vec2Model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "qualispeech" / "test.csv"
WAV_DIR = ROOT / "data" / "qualispeech" / "wav"
OUT_CSV = ROOT / "outputs" / "thao_xlsr_utmos_test_preds.csv"
HF_REPO = "vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval"


class UtmosNisqaQualityHead(nn.Module):
    def __init__(self, h, bilstm_hidden=128, mlp_hidden=256, dropout=0.0, **_):
        super().__init__()
        self.bilstm = nn.LSTM(h, bilstm_hidden, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(2 * bilstm_hidden, 1)
        layers = [
            nn.Linear(4 * bilstm_hidden, mlp_hidden),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, h, m=None):
        out = self.bilstm(h)[0]
        s = self.attn_proj(out).squeeze(-1)
        if m is not None:
            s = s.masked_fill(~m, float("-inf"))
        w = F.softmax(s, dim=1).unsqueeze(-1)
        mean = (w * out).sum(dim=1)
        std = torch.sqrt((w * (out - mean.unsqueeze(1)).pow(2)).sum(dim=1).clamp_min(1e-7))
        return torch.sigmoid(self.mlp(torch.cat([mean, std], dim=-1)).squeeze(-1)) * 4.0 + 1.0


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
    print("=" * 70)
    print("INFERENCE ON QUALISPEECH TEST SET (THẢO XLS-R UTMOS-NISQA TRAINVAL)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    model_dir = Path(snapshot_download(HF_REPO))
    cfg = json.loads((model_dir / "sqa_config.json").read_text())
    print(f"Model config: hidden_size={cfg['hidden_size']}, head={cfg['head']}")

    backbone = Wav2Vec2Model.from_pretrained(model_dir / "backbone").eval().to(DEVICE)
    head_kwargs = dict(cfg["head"])
    head_kwargs.pop("type", None)
    head = UtmosNisqaQualityHead(cfg["hidden_size"], **head_kwargs).eval().to(DEVICE)
    head.load_state_dict(load_file(model_dir / "quality_head.safetensors"))

    df = pd.read_csv(TEST_CSV)
    print(f"Loaded test CSV with {len(df)} samples.")

    predictions = []
    targets = []
    ids = []

    print("\nStarting inference...")
    with torch.no_grad():
        for i, row in enumerate(df.itertuples(), start=1):
            fid = row.id
            target = float(row._8) if hasattr(row, "_8") else float(row.Overall_quality)
            wav_path = WAV_DIR / fid

            if not wav_path.exists():
                print(f"Warning: Missing {wav_path}")
                continue

            wav = load_audio(wav_path).to(DEVICE)[None]
            # Standard audio normalization
            wav = (wav - wav.mean(dim=1, keepdim=True)) / torch.sqrt(
                wav.var(dim=1, unbiased=False, keepdim=True) + 1e-7
            )
            mask = torch.ones_like(wav, dtype=torch.long)
            h = backbone(wav, attention_mask=mask).last_hidden_state
            vmask = torch.ones(h.shape[:2], dtype=torch.bool, device=DEVICE)
            pred = float(head(h.float(), vmask).item())

            ids.append(fid)
            predictions.append(pred)
            targets.append(target)

            if i % 200 == 0 or i == len(df):
                print(f"Processed {i:4d}/{len(df)} clips...")

    preds_arr = np.array(predictions)
    targets_arr = np.array(targets)

    pcc = float(np.corrcoef(preds_arr, targets_arr)[0, 1])
    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    vx, vy = np.var(preds_arr), np.var(targets_arr)
    mx, my = np.mean(preds_arr), np.mean(targets_arr)
    cov = np.mean((preds_arr - mx) * (targets_arr - my))
    ccc = float(2.0 * cov / (vx + vy + (mx - my) ** 2 + 1e-8))

    print("\n" + "=" * 50)
    print("THẢO XLS-R UTMOS-NISQA DEV RESULTS:")
    print(f"  PCC:  {pcc:.4f}")
    print(f"  CCC:  {ccc:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print("=" * 50)

    out_df = pd.DataFrame({"id": ids, "target": targets, "prediction": predictions})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved predictions to: {OUT_CSV}")


if __name__ == "__main__":
    main()
