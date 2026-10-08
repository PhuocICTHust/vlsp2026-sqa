"""
Build complete VLSP 2026 SQA Final Submission JSON using Tri-Ensemble.

Ensemble Configuration:
- Quality Score (MOS):
  0.315 * WavLM-Large LoRA UtmosNisqa
  + 0.395 * Frozen WavLM UtmosNisqa
  + 0.290 * wav2vec2-XLS-R-300M LoRA (Thảo model)
- Fault Localization:
  SOTA Multi-scale TCN + BiLSTM Context with Calibrated Hysteresis Post-processing.

Output:
submission.json validated against official competition constraints.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import Wav2Vec2Model, WavLMModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqa.evaluator import FAULT_TYPES
from sqa.frame_grid import MODEL_GRID
from sqa.frame_head import FaultFrameHead
from sqa.lora import apply_lora_to_backbone
from sqa.model import UtmosNisqaQualityHead, normalize_waveform
from sqa.postprocess import BASELINE_POSTPROCESS, PostProcessConfig, predict_fault_type, serialize_intervals
from sqa.validator import validate_submission

# Default Paths
DEFAULT_LORA_CKPT = ROOT / "checkpoints" / "wavlm_utmos_nisqa_lora_trainval_best.pt"
DEFAULT_FROZEN_CKPT = ROOT / "outputs" / "cached_trainval_utmos_tuned" / "best_quality_head.pt"
DEFAULT_FAULT_CKPT = ROOT / "outputs" / "fault_head_trainval" / "best_fault_frame_head.pt"
DEFAULT_CALIB_CONFIG = ROOT / "outputs" / "fault_head_trainval" / "calibrated_postprocess_config.json"
THAO_HF_REPO = "vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-lora"


# ----------------------------------------------------------------------
# AUDIO HELPER
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# THAO MODEL WRAPPER
# ----------------------------------------------------------------------
class ThaoAttentionPool(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.proj = nn.Linear(h, 1)

    def forward(self, h, m=None):
        s = self.proj(h).squeeze(-1)
        if m is not None:
            s = s.masked_fill(~m, float("-inf"))
        return (F.softmax(s, dim=1).unsqueeze(-1) * h).sum(dim=1)


class ThaoQualityHead(nn.Module):
    def __init__(self, h, mlp_hidden=256, dropout=0.1):
        super().__init__()
        self.pool = ThaoAttentionPool(h)
        layers = [
            nn.Linear(h, mlp_hidden),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, h, m=None):
        return torch.sigmoid(self.mlp(self.pool(h, m)).squeeze(-1)) * 4.0 + 1.0


def load_thao_model(device: torch.device):
    print("Loading Thảo wav2vec2-XLS-R model from Hugging Face...")
    model_dir = Path(snapshot_download(THAO_HF_REPO))
    cfg = json.loads((model_dir / "sqa_config.json").read_text())
    backbone = Wav2Vec2Model.from_pretrained(model_dir / "backbone").eval().to(device)
    head = ThaoQualityHead(cfg["hidden_size"], cfg["head"]["mlp_hidden"], cfg["head"].get("dropout", 0.1)).eval().to(device)
    head.load_state_dict(load_file(model_dir / "quality_head.safetensors"))
    return backbone, head


# ----------------------------------------------------------------------
# MAIN SUBMISSION BUILDER
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build complete VLSP 2026 SQA final submission JSON with Tri-Ensemble.")
    parser.add_argument("--audio-dir", type=str, required=True, help="Folder containing WAV files.")
    parser.add_argument("--out-json", type=str, default="submission.json", help="Output submission JSON path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora-ckpt", type=str, default=str(DEFAULT_LORA_CKPT))
    parser.add_argument("--frozen-ckpt", type=str, default=str(DEFAULT_FROZEN_CKPT))
    parser.add_argument("--fault-ckpt", type=str, default=str(DEFAULT_FAULT_CKPT))
    parser.add_argument("--calib-config", type=str, default=str(DEFAULT_CALIB_CONFIG))
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    device = torch.device(args.device)

    print("=" * 70)
    print("VLSP 2026 SQA FINAL SUBMISSION BUILDER (TRI-ENSEMBLE)")
    print("=" * 70)
    print(f"Audio folder:     {audio_dir}")
    print(f"Device:           {device}")
    print(f"Output JSON:      {args.out_json}")

    # 1. Load Calibrated Post-Processing Config
    post_cfg = BASELINE_POSTPROCESS
    calib_path = Path(args.calib_config)
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        if "calibrated_configs" in raw_cfg:
            post_cfg = {
                k: PostProcessConfig(**{param: val for param, val in v.items() if param != "is_baseline"})
                for k, v in raw_cfg["calibrated_configs"].items()
            }
            print(f"Loaded calibrated post-processing config from: {calib_path.name}")

    # 2. Load Fault Frame Head
    fault_ckpt = Path(args.fault_ckpt)
    fault_payload = torch.load(fault_ckpt, map_location="cpu")
    fault_head = FaultFrameHead(**fault_payload.get("fault_head_kwargs", {})).eval().to(device)
    fault_head.load_state_dict(fault_payload["fault_head_state_dict"])
    print(f"Loaded FaultFrameHead from: {fault_ckpt.name}")

    # 3. Load Model 1: WavLM-Large LoRA + UtmosNisqa
    print("Loading Model 1 (WavLM-Large LoRA + UtmosNisqa)...")
    lora_ckpt = Path(args.lora_ckpt)
    lora_payload = torch.load(lora_ckpt, map_location="cpu")
    backbone_wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(device)
    for p in backbone_wavlm.parameters():
        p.requires_grad = False
    apply_lora_to_backbone(backbone_wavlm, lora_payload["lora_state_dict"])
    head_lora = UtmosNisqaQualityHead(hidden_size=1024, bilstm_hidden=128, dropout=0.0).eval().to(device)
    head_lora.load_state_dict(lora_payload["quality_head_state_dict"])

    # 4. Load Model 2: Frozen WavLM + UtmosNisqa
    print("Loading Model 2 (Frozen WavLM + UtmosNisqa)...")
    frozen_ckpt = Path(args.frozen_ckpt)
    frozen_payload = torch.load(frozen_ckpt, map_location="cpu")
    head_frozen = UtmosNisqaQualityHead(hidden_size=1024, bilstm_hidden=128, dropout=0.0).eval().to(device)
    head_frozen.load_state_dict(frozen_payload["model_state_dict"])

    # 5. Load Model 3: Thảo's wav2vec2-XLS-R
    backbone_thao, head_thao = load_thao_model(device)

    print("\nAll models successfully loaded into memory!")
    print("Tri-Ensemble Weights: LoRA=31.5%, Frozen=39.5%, XLS-R=29.0%")

    # Discover audio files
    wav_files = sorted(audio_dir.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found in {audio_dir}")
    print(f"\nProcessing {len(wav_files)} audio files...")

    entries = []
    duration_lookup = {}

    with torch.no_grad():
        for i, wav_path in enumerate(wav_files, start=1):
            file_id = wav_path.name
            wav, duration = load_audio_mono(wav_path)
            duration_lookup[file_id] = duration

            wav_gpu = wav.to(device)[None]  # [1, T_wav]
            attention_mask = torch.ones_like(wav_gpu, dtype=torch.long)

            # WavLM features (used for FaultHead and Model 2)
            # WavLM normalized input for LoRA
            wav_norm = normalize_waveform(wav_gpu, attention_mask)
            wavlm_hidden = backbone_wavlm(wav_norm, attention_mask=attention_mask).last_hidden_state  # [1, T_frames, 1024]
            valid_mask = torch.ones(wavlm_hidden.shape[:2], dtype=torch.bool, device=device)

            # Fault predictions
            fault_logits = fault_head(wavlm_hidden, valid_mask=valid_mask)
            fault_probs = torch.sigmoid(fault_logits)[0].cpu().numpy()  # [T_frames, 3]

            clip_faults = {}
            for f_idx, ft in enumerate(FAULT_TYPES):
                cfg = post_cfg[ft] if isinstance(post_cfg, dict) else post_cfg
                ivs, _, _, _ = predict_fault_type(fault_probs[:, f_idx], MODEL_GRID, duration, cfg)
                clip_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=duration)

            # Model 1 prediction (LoRA)
            p1 = float(head_lora(wavlm_hidden, valid_mask).item())

            # Model 2 prediction (Frozen Head)
            p2 = float(head_frozen(wavlm_hidden, valid_mask).item())

            # Model 3 prediction (Thảo XLS-R)
            wav_thao = (wav_gpu - wav_gpu.mean(dim=1, keepdim=True)) / torch.sqrt(
                wav_gpu.var(dim=1, unbiased=False, keepdim=True) + 1e-7
            )
            mask_thao = torch.ones_like(wav_thao, dtype=torch.long)
            h_thao = backbone_thao(wav_thao, attention_mask=mask_thao).last_hidden_state
            p3 = float(head_thao(h_thao.float(), torch.ones(h_thao.shape[:2], dtype=torch.bool, device=device)).item())

            # Tri-Ensemble Blending
            p_ensemble = 0.315 * p1 + 0.395 * p2 + 0.290 * p3
            p_clamped = round(float(np.clip(p_ensemble, 1.0, 5.0)), 4)

            entry = {
                "file_id": file_id,
                "quality_score": p_clamped,
                "faults": clip_faults,
            }
            entries.append(entry)

            if i % 100 == 0 or i == len(wav_files):
                print(f"Processed {i:4d}/{len(wav_files)} files...")

    # Validate submission format
    print("\nValidating submission JSON against official VLSP 2026 constraints...")
    errors = validate_submission(entries, duration_lookup, strict=False)
    if errors:
        print(f"FAILED: Found {len(errors)} validation errors:")
        for err in errors[:10]:
            print("  ", err)
        raise AssertionError("Submission failed validation constraints!")

    print("VALIDATION PASS: 100% compliant with competition requirements!")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFinal submission successfully written to: {out_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
