"""
Script: scripts/compare_single_vs_ensemble.py
Reproduces Table 5 (tab:single_vs_ens) of paper_lncs.tex:
Controlled ablation comparing standalone single neural architectures against
multi-model ensembling on the Official Private Test Set (302 clips)
under identical static post-processing thresholds.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import soundfile as sf
import torch
from transformers import WavLMModel

from scripts.train_conformer_fault_head import ConformerFaultHead
from scripts.train_conformer_pro import ConformerProFaultHead
from scripts.train_decoupled_fault_head import DecoupledFaultHead
from sqa.evaluator import (
    FAULT_TYPES,
    compute_f1_macro,
    compute_frame_iou_macro,
    final_score,
)
from sqa.frame_grid import EVALUATOR_GRID, MODEL_GRID
from sqa.model import normalize_waveform
from sqa.postprocess import PostProcessConfig, predict_fault_type, serialize_intervals

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GT_PATH = ROOT / "data" / "private_test_ground_truth.json"
AUDIO_DIR = ROOT / "data" / "private_test" / "audio"
OFFICIAL_PCC_MOS = 0.703139


def main():
    print("=" * 85)
    print("TABLE 5: STANDALONE SINGLE MODELS VS ENSEMBLE ON OFFICIAL PRIVATE TEST SET")
    print("=" * 85)

    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_list = json.load(f)

    wav_files = [AUDIO_DIR / x["file_id"] for x in gt_list]
    durations = [sf.info(str(p)).duration for p in wav_files]

    print("Loading models...")
    m_peak = ConformerFaultHead(1024, 256, 2).eval().to(DEVICE)
    m_peak.load_state_dict(
        torch.load(ROOT / "outputs" / "fault_head_v5" / "best_conformer_25ep.pt", map_location="cpu")
    )

    m_pro = ConformerProFaultHead(1024, 256, 2).eval().to(DEVICE)
    m_pro.load_state_dict(
        torch.load(ROOT / "outputs" / "fault_head_v5" / "best_conformer_pro.pt", map_location="cpu")
    )

    m_dec = DecoupledFaultHead(1024, 128).eval().to(DEVICE)
    m_dec.load_state_dict(
        torch.load(ROOT / "outputs" / "fault_head_v5" / "best_decoupled_fault_head.pt", map_location="cpu")
    )

    print("Loading WavLM-Large backbone...")
    wavlm_bb = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(DEVICE)
    for p in wavlm_bb.parameters():
        p.requires_grad = False

    cached_wavlm = []
    print(f"Extracting WavLM features for {len(wav_files)} clips on {DEVICE}...")
    with torch.no_grad():
        for i, p in enumerate(wav_files, 1):
            wav, sr = sf.read(str(p), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            dur = len(wav) / sr
            wav_t = torch.from_numpy(wav).float().to(DEVICE).unsqueeze(0)
            mask = torch.ones_like(wav_t, dtype=torch.long)
            wav_norm = normalize_waveform(wav_t, mask)
            hidden = wavlm_bb(wav_norm, attention_mask=mask).last_hidden_state
            cached_wavlm.append((hidden, dur))
            if i % 100 == 0 or i == len(wav_files):
                print(f"   [{i:3d}/{len(wav_files)}] clips extracted")

    cfgs = {
        "noise": PostProcessConfig(tau_frame=0.55, tau_low=0.25, use_hysteresis=True, min_duration=0.0, merge_gap=0.04),
        "distortion": PostProcessConfig(tau_frame=0.55, tau_low=0.25, use_hysteresis=True, min_duration=0.0, merge_gap=0.04),
        "unnatural_pause": PostProcessConfig(tau_frame=0.45, tau_low=0.20, use_hysteresis=True, min_duration=0.08, merge_gap=0.04),
    }

    def evaluate_predictions(probs_list):
        preds = []
        for (hidden, dur), p in zip(cached_wavlm, probs_list):
            clip_faults = {}
            for c_idx, ft in enumerate(FAULT_TYPES):
                ivs, _, _, _ = predict_fault_type(p[:, c_idx], MODEL_GRID, dur, cfgs[ft])
                clip_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=dur)
            preds.append({"file_id": "", "quality_score": 3.50, "faults": clip_faults})

        f1_m, f1_dict = compute_f1_macro(preds, gt_list)
        iou_m, iou_dict = compute_frame_iou_macro(preds, gt_list, durations, EVALUATOR_GRID)
        score = final_score(OFFICIAL_PCC_MOS, f1_m, iou_m)
        return {
            "score": score,
            "f1_m": f1_m,
            "f1_dict": f1_dict,
            "iou_m": iou_m,
            "iou_dict": iou_dict,
        }

    # Generate probabilities
    print("\nGenerating model probabilities...")
    probs_peak, probs_pro, probs_dec = [], [], []
    with torch.no_grad():
        for hidden, dur in cached_wavlm:
            vmask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=DEVICE)
            probs_peak.append(torch.sigmoid(m_peak(hidden, valid_mask=vmask))[0].cpu().numpy())
            probs_pro.append(torch.sigmoid(m_pro(hidden, valid_mask=vmask))[0].cpu().numpy())
            probs_dec.append(torch.sigmoid(m_dec(hidden, valid_mask=vmask))[0].cpu().numpy())

    probs_ens = [
        0.60 * p1 + 0.20 * p2 + 0.20 * p3
        for p1, p2, p3 in zip(probs_peak, probs_pro, probs_dec)
    ]

    models = [
        ("Single Conformer Peak (Alone)", probs_peak),
        ("Single Conformer Pro (Alone)", probs_pro),
        ("Single Decoupled Tri-Branch (Alone)", probs_dec),
        ("Ensemble Run 3 (60% Peak + 20% Pro + 20% Dec)", probs_ens),
    ]

    print("\n" + "=" * 85)
    print(f"{'Model Architecture':<42} | {'F1_macro':<9} | {'IoU_macro':<10} | {'Final Score':<11}")
    print("-" * 85)
    for name, p_list in models:
        res = evaluate_predictions(p_list)
        print(f"{name:<42} | {res['f1_m']:<9.4f} | {res['iou_m']:<10.4f} | {res['score']:<11.6f}")


if __name__ == "__main__":
    main()
