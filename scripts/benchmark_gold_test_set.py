"""
Script: scripts/benchmark_gold_test_set.py
Reproduces the Vietnamese Synthetic Dev Set (125 clips, data/vietnamese_test_v1) fault localization
benchmarks reported in Table 2 (tab:dev_results) and Table 6 (tab:loss_attribution) of paper_lncs.tex:
  1. Shared-threshold controlled comparison across architectures (40.39%, 78.65%, 83.70%, 85.35%, 87.01%, 87.52%).
  2. Exact per-run submission builder post-processing configurations (v2, v6, v10, v11, v12) paired with
     QualiSpeech Test PCC_MOS = 0.7325 (Composite Dev Scores: 0.5300, 0.7955, 0.8064, 0.8180, 0.8229).
"""

import json
import sys
import time
from math import gcd
from pathlib import Path
from typing import Dict, List

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
from scipy.signal import resample_poly
from transformers import AutoModel, WavLMModel

from scripts.train_conformer_fault_head import ConformerFaultHead
from scripts.train_conformer_pro import ConformerProFaultHead
from scripts.train_decoupled_fault_head import DecoupledFaultHead
from scripts.train_dual_backbone_conformer import DualBackboneConformerHead
from sqa.evaluator import FAULT_TYPES, compute_f1_macro, compute_frame_iou_macro
from sqa.frame_grid import EVALUATOR_GRID, MODEL_GRID
from sqa.frame_head import FaultFrameHead
from sqa.model import normalize_waveform
from sqa.postprocess import PostProcessConfig, predict_fault_type, serialize_intervals

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = ROOT / "data" / "vietnamese_test_v1"
WAV_DIR = TEST_DIR / "wav"
LABELS_PATH = TEST_DIR / "labels.jsonl"
QUALISPEECH_DEV_PCC_MOS = 0.7325


def load_audio_mono(path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, float, float]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    duration = float(len(wav)) / float(target_sr)
    rms = float(np.sqrt(np.mean(wav ** 2) + 1e-12))
    rms_dbfs = float(20.0 * np.log10(rms + 1e-9))
    tensor = torch.from_numpy(wav.astype(np.float32))
    return tensor, duration, rms_dbfs


def extract_features(wav_files: List[Path]) -> Dict[str, dict]:
    print(f"-> Extracting WavLM-Large (normalized) and XLS-R-300M features for {len(wav_files)} clips on {DEVICE}...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(DEVICE)
    for p in wavlm.parameters():
        p.requires_grad = False

    xlsr_cache_dir = TEST_DIR / "xlsr_cache"
    xlsr_model = None
    if not xlsr_cache_dir.exists() or len(list(xlsr_cache_dir.glob("*.pt"))) < len(wav_files):
        xlsr_model = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m").eval().to(DEVICE)
        for p in xlsr_model.parameters():
            p.requires_grad = False

    feats = {}
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(wav_files, 1):
            wav, dur, rms_dbfs = load_audio_mono(p)
            wav_gpu = wav.unsqueeze(0).to(DEVICE)
            mask = torch.ones_like(wav_gpu, dtype=torch.long)
            wav_norm = normalize_waveform(wav_gpu, mask)
            h_wavlm = wavlm(wav_norm, attention_mask=mask).last_hidden_state.squeeze(0).cpu()

            cache_pt = xlsr_cache_dir / f"{p.stem}.pt"
            if cache_pt.exists():
                h_xlsr = torch.load(cache_pt, map_location="cpu")["hidden"].float()
            else:
                h_xlsr = xlsr_model(wav_gpu).last_hidden_state.squeeze(0).cpu().float()

            feats[p.name] = {
                "h_wavlm": h_wavlm,
                "h_xlsr": h_xlsr,
                "duration": dur,
                "rms_dbfs": rms_dbfs,
            }
            if i % 25 == 0 or i == len(wav_files):
                print(f"   [{i:3d}/{len(wav_files)}] extracted ({time.time() - t0:.1f}s)")

    del wavlm
    if xlsr_model is not None:
        del xlsr_model
    torch.cuda.empty_cache()
    return feats


def evaluate_probs(
    probs_by_file: Dict[str, np.ndarray],
    gts: List[dict],
    feats: Dict[str, dict],
    cfgs: Dict[str, PostProcessConfig],
) -> dict:
    ordered_preds = []
    durations = []
    for g in gts:
        fid = g["file_id"]
        dur = feats[fid]["duration"]
        durations.append(dur)
        probs = probs_by_file[fid]
        file_faults = {}
        for c_idx, ft in enumerate(FAULT_TYPES):
            ivs, _, _, _ = predict_fault_type(probs[:, c_idx], MODEL_GRID, dur, cfgs[ft])
            file_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=dur)
        ordered_preds.append({"file_id": fid, "faults": file_faults, "duration": dur})

    f1_macro, f1_details = compute_f1_macro(ordered_preds, gts)
    iou_macro, iou_details = compute_frame_iou_macro(ordered_preds, gts, durations, EVALUATOR_GRID)
    fault_avg = 0.5 * (f1_macro + iou_macro)
    composite_dev = 0.40 * QUALISPEECH_DEV_PCC_MOS + 0.30 * f1_macro + 0.30 * iou_macro
    return {
        "f1_macro": f1_macro,
        "f1_details": f1_details,
        "iou_macro": iou_macro,
        "iou_details": iou_details,
        "fault_avg": fault_avg,
        "composite_dev": composite_dev,
    }


def main():
    print("=" * 95)
    print("BENCHMARK ON VIETNAMESE SYNTHETIC DEV SET (125 CLIPS - data/vietnamese_test_v1)")
    print("=" * 95)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        gts = [json.loads(line) for line in f if line.strip()]
    wav_files = sorted(WAV_DIR.glob("*.wav"))
    assert len(wav_files) == len(gts), f"Mismatch: {len(wav_files)} wavs vs {len(gts)} labels"

    feats = extract_features(wav_files)

    # Load all model heads
    payload_v1 = torch.load(ROOT / "outputs" / "fault_head_trainval" / "best_fault_frame_head.pt", map_location="cpu")
    m_v1 = FaultFrameHead(**payload_v1["fault_head_kwargs"]).eval().to(DEVICE)
    m_v1.load_state_dict(payload_v1["fault_head_state_dict"])

    m_dec = DecoupledFaultHead(1024, 128).eval().to(DEVICE)
    m_dec.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_decoupled_fault_head.pt", map_location="cpu"))

    m_conf10 = ConformerFaultHead(1024, 256, 2).eval().to(DEVICE)
    m_conf10.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_conformer_fault_head.pt", map_location="cpu"))

    m_peak = ConformerFaultHead(1024, 256, 2).eval().to(DEVICE)
    m_peak.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_conformer_25ep.pt", map_location="cpu"))

    m_pro = ConformerProFaultHead(1024, 256, 2).eval().to(DEVICE)
    m_pro.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_conformer_pro.pt", map_location="cpu"))

    m_dual = DualBackboneConformerHead(2048, 256, 2).eval().to(DEVICE)
    m_dual.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_dual_backbone_conformer.pt", map_location="cpu"))

    probs_v1, probs_dec, probs_r2, probs_r3, probs_r4, probs_r5 = {}, {}, {}, {}, {}, {}

    with torch.no_grad():
        for fid, item in feats.items():
            hw = item["h_wavlm"].unsqueeze(0).to(DEVICE)
            hx = item["h_xlsr"].unsqueeze(0).to(DEVICE)
            min_t = min(hw.shape[1], hx.shape[1])
            hw_t = hw[:, :min_t, :]
            hx_t = hx[:, :min_t, :]
            hf_t = torch.cat([hw_t, hx_t], dim=-1)
            vmask = torch.ones((1, min_t), dtype=torch.bool, device=DEVICE)

            p_v1 = torch.sigmoid(m_v1(hw_t, valid_mask=vmask))[0].cpu().numpy()
            p_dec = torch.sigmoid(m_dec(hw_t, valid_mask=vmask))[0].cpu().numpy()
            p_c10 = torch.sigmoid(m_conf10(hw_t, valid_mask=vmask))[0].cpu().numpy()
            p_peak = torch.sigmoid(m_peak(hw_t, valid_mask=vmask))[0].cpu().numpy()
            p_pro = torch.sigmoid(m_pro(hw_t, valid_mask=vmask))[0].cpu().numpy()
            p_dual = torch.sigmoid(m_dual(hf_t, valid_mask=vmask))[0].cpu().numpy()

            probs_v1[fid] = p_v1
            probs_dec[fid] = p_dec
            probs_r2[fid] = 0.70 * p_c10 + 0.30 * p_dec
            probs_r3[fid] = 0.60 * p_peak + 0.20 * p_pro + 0.20 * p_dec
            probs_r4[fid] = 0.60 * p_dual + 0.40 * p_peak
            probs_r5[fid] = 0.50 * p_dual + 0.25 * p_peak + 0.15 * p_pro + 0.10 * p_dec

    # 1. Shared Static Thresholds (Table 2 column 'Shared-Threshold Fault Avg' & Table 6 Dev column)
    with open(ROOT / "outputs" / "fault_head_trainval" / "calibrated_postprocess_config.json", "r", encoding="utf-8") as f:
        raw_v1_cfg = json.load(f)
    cal_dict = raw_v1_cfg.get("calibrated_configs", raw_v1_cfg)
    cfg_v1 = {
        ft: PostProcessConfig(**{k: v for k, v in cal_dict[ft].items() if k != "is_baseline"})
        for ft in FAULT_TYPES
    }

    shared_vi_cfg = {
        "noise": PostProcessConfig(tau_frame=0.45, tau_low=0.25, use_hysteresis=True, min_duration=0.08, merge_gap=0.04),
        "distortion": PostProcessConfig(tau_frame=0.55, tau_low=0.25, use_hysteresis=True, min_duration=0.00, merge_gap=0.04),
        "unnatural_pause": PostProcessConfig(tau_frame=0.45, tau_low=0.20, use_hysteresis=True, min_duration=0.08, merge_gap=0.04),
    }

    shared_runs = [
        ("Run 1 (v2): English Baseline FaultFrameHead", probs_v1, shared_vi_cfg),
        ("Standalone Decoupled Tri-Branch Head", probs_dec, shared_vi_cfg),
        ("Run 2 (v6): 0.70 Conformer(10ep) + 0.30 Decoupled", probs_r2, shared_vi_cfg),
        ("Run 3 (v10): 0.60 Peak(25ep) + 0.20 Pro + 0.20 Dec", probs_r3, shared_vi_cfg),
        ("Run 4 (v11): 0.60 Dual-Backbone + 0.40 Peak(25ep)", probs_r4, shared_vi_cfg),
        ("Run 5 (v12): 0.50 Dual + 0.25 Peak + 0.15 Pro + 0.10 Dec", probs_r5, shared_vi_cfg),
    ]

    print("\n" + "=" * 95)
    print("1. SHARED-THRESHOLD CONTROLLED COMPARISON ON VIETNAMESE DEV SET (125 CLIPS)")
    print("=" * 95)
    print(f"{'System Configuration':<56} | {'F1_mac':<8} | {'IoU_mac':<8} | {'Fault Avg':<9}")
    print("-" * 95)
    for label, pmap, cfg in shared_runs:
        res = evaluate_probs(pmap, gts, feats, cfg)
        print(f"{label:<56} | {res['f1_macro']:.4f}   | {res['iou_macro']:.4f}   | {res['fault_avg']*100:6.2f}%")

    # 2. Exact Per-Run Submission Builder Configurations (Table 2 Per-Run Calibrated Columns)
    # RMS statistics for adaptive runs
    all_rms = [feats[g["file_id"]]["rms_dbfs"] for g in gts]
    mean_rms = float(np.mean(all_rms))
    std_rms = float(np.std(all_rms))

    def evaluate_adaptive(probs_by_file, builder_cfgs):
        ordered_preds = []
        durations = []
        for g in gts:
            fid = g["file_id"]
            dur = feats[fid]["duration"]
            durations.append(dur)
            probs = probs_by_file[fid]
            norm_rms = (feats[fid]["rms_dbfs"] - mean_rms) / (std_rms + 1e-6)
            file_faults = {}
            for c_idx, ft in enumerate(FAULT_TYPES):
                bc = builder_cfgs[ft]
                adj_tau = np.clip(bc["tf"] + bc.get("beta", 0.0) * norm_rms, 0.15, 0.70)
                adj_low = np.clip(bc["tl"] + bc.get("beta", 0.0) * 0.5 * norm_rms, 0.10, 0.50)
                pcfg = PostProcessConfig(
                    tau_frame=float(adj_tau),
                    tau_low=float(adj_low),
                    use_hysteresis=True,
                    min_duration=bc["md"],
                    merge_gap=0.04,
                )
                ivs, _, _, _ = predict_fault_type(probs[:, c_idx], MODEL_GRID, dur, pcfg)
                file_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=dur)
            ordered_preds.append({"file_id": fid, "faults": file_faults, "duration": dur})

        f1_macro, f1_details = compute_f1_macro(ordered_preds, gts)
        iou_macro, iou_details = compute_frame_iou_macro(ordered_preds, gts, durations, EVALUATOR_GRID)
        fault_avg = 0.5 * (f1_macro + iou_macro)
        composite_dev = 0.40 * QUALISPEECH_DEV_PCC_MOS + 0.30 * f1_macro + 0.30 * iou_macro
        return {
            "f1_macro": f1_macro,
            "f1_details": f1_details,
            "iou_macro": iou_macro,
            "iou_details": iou_details,
            "fault_avg": fault_avg,
            "composite_dev": composite_dev,
        }

    builder_cfgs_v6 = {
        "noise": {"tf": 0.55, "tl": 0.15, "md": 0.12, "beta": +0.02},
        "distortion": {"tf": 0.50, "tl": 0.25, "md": 0.08, "beta": -0.02},
        "unnatural_pause": {"tf": 0.50, "tl": 0.15, "md": 0.08, "beta": 0.00},
    }
    builder_cfgs_v10 = {
        "noise": {"tf": 0.55, "tl": 0.25, "md": 0.00, "beta": -0.02},
        "distortion": {"tf": 0.55, "tl": 0.25, "md": 0.00, "beta": 0.00},
        "unnatural_pause": {"tf": 0.45, "tl": 0.20, "md": 0.08, "beta": -0.02},
    }
    cfg_r4 = {
        "noise": PostProcessConfig(tau_frame=0.50, tau_low=0.20, use_hysteresis=True, min_duration=0.08, merge_gap=0.04),
        "distortion": PostProcessConfig(tau_frame=0.55, tau_low=0.25, use_hysteresis=True, min_duration=0.00, merge_gap=0.04),
        "unnatural_pause": PostProcessConfig(tau_frame=0.40, tau_low=0.15, use_hysteresis=True, min_duration=0.04, merge_gap=0.04),
    }
    cfg_r5 = {
        "noise": PostProcessConfig(tau_frame=0.45, tau_low=0.25, use_hysteresis=True, min_duration=0.08, merge_gap=0.04),
        "distortion": PostProcessConfig(tau_frame=0.55, tau_low=0.25, use_hysteresis=True, min_duration=0.00, merge_gap=0.04),
        "unnatural_pause": PostProcessConfig(tau_frame=0.55, tau_low=0.20, use_hysteresis=True, min_duration=0.08, merge_gap=0.04),
    }

    print("\n" + "=" * 95)
    print("2. PER-RUN CALIBRATED SUBMISSION BUILDER CONFIGS (PAIRED WITH QUALISPEECH PCC = 0.7325)")
    print("=" * 95)
    res_r1 = evaluate_probs(probs_v1, gts, feats, cfg_v1)
    res_r2 = evaluate_adaptive(probs_r2, builder_cfgs_v6)
    res_r3 = evaluate_adaptive(probs_r3, builder_cfgs_v10)
    res_r4 = evaluate_probs(probs_r4, gts, feats, cfg_r4)
    res_r5 = evaluate_probs(probs_r5, gts, feats, cfg_r5)

    for label, res in [
        ("Run 1 (anhtraiAI_1 / v2)", res_r1),
        ("Run 2 (anhtraiAI_2 / v6)", res_r2),
        ("Run 3 (anhtraiAI_3 / v10)", res_r3),
        ("Run 4 (anhtraiAI_4 / v11)", res_r4),
        ("Run 5 (anhtraiAI_5 / v12)", res_r5),
    ]:
        f1d, ioud = res["f1_details"], res["iou_details"]
        print(
            f"{label:<28} | F1={res['f1_macro']:.4f} (N:{f1d['noise']:.4f}, D:{f1d['distortion']:.4f}, P:{f1d['unnatural_pause']:.4f}) "
            f"| IoU={res['iou_macro']:.4f} (N:{ioud['noise']:.4f}, D:{ioud['distortion']:.4f}, P:{ioud['unnatural_pause']:.4f}) "
            f"| FaultAvg={res['fault_avg']*100:.2f}% | CompDev={res['composite_dev']:.4f}"
        )


if __name__ == "__main__":
    main()
