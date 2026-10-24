"""Submission generator for Official Run 5 (anhtraiAI_5.zip / submission_v12.json).
Vietnamese Synthetic Dev Set (125 clips) Fault Avg: 88.32% (per-run) / 87.52% (shared), Composite Dev Score: 0.8229.
Official Private Test Final Score: 0.525695.
"""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WavLMModel, AutoModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqa.evaluator import FAULT_TYPES
from sqa.frame_grid import MODEL_GRID
from sqa.model import normalize_waveform
from sqa.postprocess import PostProcessConfig, predict_fault_type, serialize_intervals
from scripts.train_dual_backbone_conformer import DualBackboneConformerHead
from scripts.train_conformer_fault_head import ConformerFaultHead
from scripts.train_conformer_pro import ConformerProFaultHead
from scripts.train_decoupled_fault_head import DecoupledFaultHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_DIR = ROOT / "data" / "private_test" / "audio"
SUB_V5_PATH = ROOT / "submissions" / "submission_v5.json"
OUT_JSON = ROOT / "submissions" / "submission_v12.json"
OUT_ZIP = ROOT / "submissions" / "anhtraiAI_12.zip"

def load_audio_mono(audio_path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, float]:
    info = sf.info(str(audio_path))
    duration = float(info.duration)
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    return torch.from_numpy(wav).float(), duration

def main():
    print("=" * 80)
    print("BUILDING OFFICIAL RUN 5 (anhtraiAI_5 / submission_v12.json) - 0.8229 COMPOSITE DEV / 0.525695 PRIVATE TEST")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    mos_source_path = SUB_V5_PATH if SUB_V5_PATH.exists() else ROOT / "submissions" / "submission_v2.json"
    if not mos_source_path.exists():
        raise FileNotFoundError(
            f"Missing MOS source file ({SUB_V5_PATH} or submissions/submission_v2.json). "
            "Run scripts/build_upgraded_tri_ensemble_submission.py first."
        )
    with open(mos_source_path, "r", encoding="utf-8") as f:
        v5_data = json.load(f)
    mos_dict = {it["file_id"]: it["quality_score"] for it in v5_data}

    print("Loading 4 diverse neural architectures...")
    m_dual = DualBackboneConformerHead(2048, 256, 2).eval().to(DEVICE)
    m_dual.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_dual_backbone_conformer.pt", map_location="cpu"))

    m_peak = ConformerFaultHead(1024, 256, 2).eval().to(DEVICE)
    m_peak.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_conformer_25ep.pt", map_location="cpu"))

    m_pro = ConformerProFaultHead(1024, 256, 2).eval().to(DEVICE)
    m_pro.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_conformer_pro.pt", map_location="cpu"))

    m_dec = DecoupledFaultHead(1024, 128).eval().to(DEVICE)
    m_dec.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_decoupled_fault_head.pt", map_location="cpu"))

    print("Loading WavLM-Large backbone...")
    wavlm_bb = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(DEVICE)
    for p in wavlm_bb.parameters(): p.requires_grad = False

    print("Loading XLS-R-300M backbone...")
    xlsr_bb = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m").eval().to(DEVICE)
    for p in xlsr_bb.parameters(): p.requires_grad = False

    cfgs = {
        "noise": PostProcessConfig(tau_frame=0.45, tau_low=0.25, use_hysteresis=True, min_duration=0.08, merge_gap=0.04),
        "distortion": PostProcessConfig(tau_frame=0.55, tau_low=0.25, use_hysteresis=True, min_duration=0.00, merge_gap=0.04),
        "unnatural_pause": PostProcessConfig(tau_frame=0.55, tau_low=0.20, use_hysteresis=True, min_duration=0.08, merge_gap=0.04),
    }

    wav_files = sorted(list(AUDIO_DIR.glob("*.wav")))
    print(f"Found {len(wav_files)} test audio files in {AUDIO_DIR}")

    final_submissions = []
    with torch.no_grad():
        for i, wav_path in enumerate(wav_files, start=1):
            wav, dur = load_audio_mono(wav_path)
            wav_gpu = wav.to(DEVICE)[None]
            mask = torch.ones_like(wav_gpu, dtype=torch.long)
            wav_norm = normalize_waveform(wav_gpu, mask)

            h_wavlm = wavlm_bb(wav_norm, attention_mask=mask).last_hidden_state
            h_xlsr = xlsr_bb(wav_gpu).last_hidden_state

            min_t = min(h_wavlm.shape[1], h_xlsr.shape[1])
            h_wavlm = h_wavlm[:, :min_t, :]
            h_xlsr = h_xlsr[:, :min_t, :]
            h_fused = torch.cat([h_wavlm, h_xlsr], dim=-1)

            vmask = torch.ones((1, min_t), dtype=torch.bool, device=DEVICE)

            pd = torch.sigmoid(m_dual(h_fused, valid_mask=vmask))[0].cpu().numpy()
            pp = torch.sigmoid(m_peak(h_wavlm, valid_mask=vmask))[0].cpu().numpy()
            pr = torch.sigmoid(m_pro(h_wavlm, valid_mask=vmask))[0].cpu().numpy()
            dc = torch.sigmoid(m_dec(h_wavlm, valid_mask=vmask))[0].cpu().numpy()

            probs_ens = 0.50 * pd + 0.25 * pp + 0.15 * pr + 0.10 * dc

            clip_faults = {}
            for c_idx, ft in enumerate(FAULT_TYPES):
                cfg = cfgs[ft]
                ivs, _, _, _ = predict_fault_type(probs_ens[:, c_idx], MODEL_GRID, dur, cfg)
                clip_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=dur)

            assert wav_path.name in mos_dict, f"Missing MOS score for {wav_path.name}"
            q_score = mos_dict[wav_path.name]
            final_submissions.append({
                "file_id": wav_path.name,
                "quality_score": round(float(q_score), 4),
                "faults": clip_faults,
            })

            if i % 50 == 0 or i == len(wav_files):
                print(f"Processed {i:3d}/{len(wav_files)} files...")

    final_submissions.sort(key=lambda x: x["file_id"])
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_submissions, f, indent=2, ensure_ascii=False)

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_JSON, arcname="submission.json")

    print(f"\nVersion 12 (Official Run 5) successfully packaged to: {OUT_ZIP.resolve()} ({OUT_ZIP.stat().st_size / 1024:.2f} KB)")

    with zipfile.ZipFile(OUT_ZIP, "r") as zf:
        assert zf.namelist() == ["submission.json"], "Archive must contain only submission.json"
        with zf.open("submission.json") as f:
            data = json.load(f)
            assert len(data) == 302, f"Expected 302 files, got {len(data)}"
            for it in data:
                assert 1.0 <= it["quality_score"] <= 5.0, f"Quality score out of range: {it}"
                assert all(ft in it["faults"] for ft in FAULT_TYPES), f"Missing fault keys: {it}"
    print("Verification passed 100%! Ready for upload.")

if __name__ == "__main__":
    main()
