"""
Build UPGRADED VLSP 2026 SQA Submission JSON using SOTA Tri-Ensemble v2.

Upgraded Tri-Ensemble Configuration:
- Quality Score (MOS):
  0.247 * WavLM-Large LoRA UtmosNisqa
  + 0.364 * Frozen WavLM UtmosNisqa
  + 0.389 * wav2vec2-XLS-R-300M UtmosNisqa (NEW Thảo model: vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval)
- Fault Localization:
  SOTA Multi-scale TCN + BiLSTM Context with Calibrated Hysteresis Post-processing.

Output:
submission_v2.json -> anhtraiAI_2.zip
"""
from __future__ import annotations

import argparse
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
import transformers

try:
    import transformers.utils.import_utils as _iu
    _iu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

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
THAO_NEW_HF_REPO = "vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval"


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
# NEW THAO MODEL HEAD
# ----------------------------------------------------------------------
class ThaoUtmosNisqaHead(nn.Module):
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


def load_thao_new_model(device: torch.device):
    print("Loading NEW Thảo wav2vec2-XLS-R-300M UtmosNisqa model from Hugging Face...")
    model_dir = Path(snapshot_download(THAO_NEW_HF_REPO))
    cfg = json.loads((model_dir / "sqa_config.json").read_text())
    backbone = Wav2Vec2Model.from_pretrained(model_dir / "backbone").eval().to(device)
    head_kw = dict(cfg["head"])
    head_kw.pop("type", None)
    head = ThaoUtmosNisqaHead(cfg["hidden_size"], **head_kw).eval().to(device)
    head.load_state_dict(load_file(model_dir / "quality_head.safetensors"))
    return backbone, head


# ----------------------------------------------------------------------
# MAIN SUBMISSION BUILDER
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build complete VLSP 2026 SQA final submission JSON with UPGRADED Tri-Ensemble.")
    parser.add_argument("--audio-dir", type=str, default="data/private_test/audio", help="Folder containing WAV files.")
    parser.add_argument("--out-json", type=str, default="submissions/submission_v2.json", help="Output submission JSON path.")
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
    print("VLSP 2026 SQA FINAL SUBMISSION BUILDER (UPGRADED TRI-ENSEMBLE V2)")
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

    # Discover audio files
    wav_files = sorted(audio_dir.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found in {audio_dir}")
    print(f"\nDiscovered {len(wav_files)} audio files in {audio_dir}")

    print("Loading audio waveforms into memory...")
    audios = {}
    for p in wav_files:
        wav, dur = load_audio_mono(p)
        audios[p.name] = (wav, dur)
    print(f"Loaded {len(audios)} audio clips.")

    # ------------------------------------------------------------------
    # PASS 1: Frozen WavLM -> Fault Localization & Model 2 (Frozen MOS)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PASS 1: Frozen WavLM (Fault Localization + Frozen Head MOS)")
    print("=" * 60)
    fault_ckpt = Path(args.fault_ckpt)
    fault_payload = torch.load(fault_ckpt, map_location="cpu")
    fault_head = FaultFrameHead(**fault_payload.get("fault_head_kwargs", {})).eval().to(device)
    fault_head.load_state_dict(fault_payload["fault_head_state_dict"])

    frozen_ckpt = Path(args.frozen_ckpt)
    frozen_payload = torch.load(frozen_ckpt, map_location="cpu")
    backbone_frozen = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(device)
    for p in backbone_frozen.parameters():
        p.requires_grad = False
    kw_frozen = frozen_payload.get("quality_head_kwargs", {"bilstm_hidden": 128, "dropout": 0.0, "use_fault_features": False})
    kw_frozen.setdefault("use_fault_features", False)
    head_frozen = UtmosNisqaQualityHead(hidden_size=1024, **kw_frozen).eval().to(device)
    head_frozen.load_state_dict(frozen_payload["model_state_dict"])

    preds_frozen = {}
    faults_all = {}
    with torch.no_grad():
        for i, (fname, (wav, dur)) in enumerate(audios.items(), start=1):
            wav_gpu = wav.to(device)[None]
            mask = torch.ones_like(wav_gpu, dtype=torch.long)
            wav_norm = normalize_waveform(wav_gpu, mask)
            hidden = backbone_frozen(wav_norm, attention_mask=mask).last_hidden_state
            vmask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)

            p2 = float(head_frozen(hidden, vmask).item())
            preds_frozen[fname] = p2

            f_logits = fault_head(hidden, valid_mask=vmask)
            f_probs = torch.sigmoid(f_logits)[0].cpu().numpy()
            clip_faults = {}
            for f_idx, ft in enumerate(FAULT_TYPES):
                cfg = post_cfg[ft] if isinstance(post_cfg, dict) else post_cfg
                ivs, _, _, _ = predict_fault_type(f_probs[:, f_idx], MODEL_GRID, dur, cfg)
                clip_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=dur)
            faults_all[fname] = clip_faults

            if i % 100 == 0 or i == len(audios):
                print(f"Pass 1: Processed {i:3d}/{len(audios)} files...")

    del backbone_frozen, fault_head, head_frozen
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PASS 2: WavLM-Large LoRA -> Model 1 (LoRA MOS)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PASS 2: WavLM-Large LoRA (Model 1 MOS)")
    print("=" * 60)
    lora_ckpt = Path(args.lora_ckpt)
    lora_payload = torch.load(lora_ckpt, map_location="cpu")
    backbone_lora = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(device)
    for p in backbone_lora.parameters():
        p.requires_grad = False
    apply_lora_to_backbone(backbone_lora, lora_payload["lora_state_dict"])
    kw_lora = lora_payload.get("quality_head_kwargs", {"bilstm_hidden": 128, "dropout": 0.0, "use_fault_features": False})
    kw_lora.setdefault("use_fault_features", False)
    head_lora = UtmosNisqaQualityHead(hidden_size=1024, **kw_lora).eval().to(device)
    head_lora.load_state_dict(lora_payload["quality_head_state_dict"])

    preds_lora = {}
    with torch.no_grad():
        for i, (fname, (wav, dur)) in enumerate(audios.items(), start=1):
            wav_gpu = wav.to(device)[None]
            mask = torch.ones_like(wav_gpu, dtype=torch.long)
            wav_norm = normalize_waveform(wav_gpu, mask)
            hidden = backbone_lora(wav_norm, attention_mask=mask).last_hidden_state
            vmask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)

            p1 = float(head_lora(hidden, vmask).item())
            preds_lora[fname] = p1

            if i % 100 == 0 or i == len(audios):
                print(f"Pass 2: Processed {i:3d}/{len(audios)} files...")

    del backbone_lora, head_lora
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PASS 3: wav2vec2-XLS-R-300M UtmosNisqa MOS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PASS 3: NEW wav2vec2-XLS-R-300M UtmosNisqa (Thảo Model MOS)")
    print("=" * 60)
    backbone_thao, head_thao = load_thao_new_model(device)

    preds_thao = {}
    with torch.no_grad():
        for i, (fname, (wav, dur)) in enumerate(audios.items(), start=1):
            wav_gpu = wav.to(device)[None]
            wav_thao = (wav_gpu - wav_gpu.mean(dim=1, keepdim=True)) / torch.sqrt(
                wav_gpu.var(dim=1, unbiased=False, keepdim=True) + 1e-7
            )
            mask_thao = torch.ones_like(wav_thao, dtype=torch.long)
            h_thao = backbone_thao(wav_thao, attention_mask=mask_thao).last_hidden_state
            vmask_thao = torch.ones(h_thao.shape[:2], dtype=torch.bool, device=device)
            p3 = float(head_thao(h_thao.float(), vmask_thao).item())
            preds_thao[fname] = p3

            if i % 100 == 0 or i == len(audios):
                print(f"Pass 3: Processed {i:3d}/{len(audios)} files...")

    del backbone_thao, head_thao
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # ENSEMBLE COMBINATION V2
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMBINING UPGRADED TRI-ENSEMBLE V2 PREDICTIONS")
    print("=" * 60)
    print("Weights: 24.7% WavLM LoRA + 36.4% WavLM Frozen + 38.9% Thảo XLS-R Utmos")

    entries = []
    duration_lookup = {fname: dur for fname, (wav, dur) in audios.items()}

    for fname in audios.keys():
        p1 = preds_lora[fname]
        p2 = preds_frozen[fname]
        p3 = preds_thao[fname]
        p_ensemble = 0.247 * p1 + 0.364 * p2 + 0.389 * p3
        p_clamped = round(float(np.clip(p_ensemble, 1.0, 5.0)), 4)

        entries.append({
            "file_id": fname,
            "quality_score": p_clamped,
            "faults": faults_all[fname],
        })

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
    print(f"\nUpgraded submission successfully written to: {out_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
