"""
Script: scripts/benchmark_gold_test_set.py
Chạy đánh giá Benchmark các Fault Model trên tập Gold Test Set 125 clips tiếng Việt mới (data/vietnamese_test_v1).
Đo đạc chính xác:
- F1-Macro (Clip-level Presence)
- Frame IoU-Macro (20ms grid, chuẩn BTC)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import WavLMModel

from sqa.evaluator import (
    FAULT_TYPES,
    compute_f1_macro,
    compute_frame_iou_macro,
)
from sqa.frame_grid import EVALUATOR_GRID, MODEL_GRID
from sqa.frame_head import FaultFrameHead
from sqa.postprocess import BASELINE_POSTPROCESS, PostProcessConfig, predict_fault_type, serialize_intervals

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = ROOT / "data" / "vietnamese_test_v1"
WAV_DIR = TEST_DIR / "wav"
LABELS_PATH = TEST_DIR / "labels.jsonl"

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

def extract_features(wav_files: List[Path], backbone: WavLMModel) -> Dict[str, tuple[torch.Tensor, float]]:
    features = {}
    print(f"-> Đang trích xuất đặc trưng WavLM cho {len(wav_files)} test clips trên GPU...")
    t0 = time.time()
    for i, p in enumerate(wav_files, 1):
        wav, dur = load_audio_mono(p)
        wav_t = wav.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = backbone(wav_t)
            hidden = out.last_hidden_state.squeeze(0).cpu() # [T, 1024]
        features[p.name] = (hidden, dur)
        if i % 25 == 0 or i == len(wav_files):
            print(f"   [{i}/{len(wav_files)}] hoàn tất ({time.time() - t0:.1f}s)")
    return features

def load_model_and_config(ckpt_path: Path, config_path: Path) -> tuple[FaultFrameHead, dict]:
    print(f"-> Đang load checkpoint: {ckpt_path.name}")
    payload = torch.load(ckpt_path, map_location="cpu")
    head_kwargs = payload.get("fault_head_kwargs", {
        "input_dim": 1024,
        "n_faults": 3,
        "hidden_dim": 256,
        "n_blocks": 2,
    })
    head = FaultFrameHead(**head_kwargs).to(DEVICE)
    head.load_state_dict(payload["fault_head_state_dict"])
    head.eval()

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        if "calibrated_configs" in raw_cfg:
            configs = {
                k: PostProcessConfig(**{param: val for param, val in v.items() if param != "is_baseline"})
                for k, v in raw_cfg["calibrated_configs"].items()
            }
        else:
            configs = {
                ft: PostProcessConfig.from_dict(raw_cfg.get(ft, {}))
                for ft in FAULT_TYPES
            }
    else:
        configs = {ft: BASELINE_POSTPROCESS for ft in FAULT_TYPES}

    return head, configs

def run_inference(head: FaultFrameHead, configs: dict, features: Dict[str, tuple[torch.Tensor, float]]) -> List[dict]:
    preds = []
    for fid, (hidden, dur) in features.items():
        h = hidden.unsqueeze(0).to(DEVICE)
        vmask = torch.ones(h.shape[:2], dtype=torch.bool, device=DEVICE)
        with torch.no_grad():
            logits = head(h, valid_mask=vmask).squeeze(0) # [T, 3]
            probs = torch.sigmoid(logits).cpu().numpy()

        file_faults = {}
        for c_idx, ft in enumerate(FAULT_TYPES):
            p_series = probs[:, c_idx]
            cfg = configs[ft]
            ivs, _, _, _ = predict_fault_type(p_series, MODEL_GRID, dur, cfg)
            file_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=dur)

        preds.append({
            "file_id": fid,
            "faults": file_faults,
            "duration": dur
        })
    return preds

def main():
    print("=" * 70)
    print("BENCHMARK TRÊN TẬP TEST SET TIẾNG VIỆT MỚI (125 CLIPS - GOLD LABELS)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Test Set Folder: {TEST_DIR}")
    
    # 1. Load Ground Truth
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        gts = [json.loads(line) for line in f if line.strip()]
    gt_map = {item["file_id"]: item for item in gts}
    print(f"-> Đã nạp {len(gts)} Ground Truth records.")

    wav_files = sorted(WAV_DIR.glob("*.wav"))
    assert len(wav_files) == len(gts), f"Số lượng file không khớp: {len(wav_files)} wav vs {len(gts)} gt"

    # 2. Extract WavLM Features
    print("-> Khởi tạo WavLM-Large backbone...")
    backbone = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad = False

    features = extract_features(wav_files, backbone)
    del backbone
    torch.cuda.empty_cache()

    durations = [features[g["file_id"]][1] for g in gts]

    # Models to benchmark:
    models_to_test = [
        (
            "Model v1/v2 Fault Head (QualiSpeech SOTA Baseline)",
            ROOT / "outputs" / "fault_head_trainval" / "best_fault_frame_head.pt",
            ROOT / "outputs" / "fault_head_trainval" / "calibrated_postprocess_config.json"
        ),
        (
            "Model v3 Fault Head (Vietnamese 1,000 Adapted)",
            ROOT / "outputs" / "fault_head_vietnamese" / "best_fault_frame_head.pt",
            ROOT / "outputs" / "fault_head_vietnamese" / "calibrated_postprocess_config.json"
        )
    ]

    results_table = []

    for name, ckpt, cfg_path in models_to_test:
        print("\n" + "-" * 70)
        print(f"Đang đánh giá: {name}")
        head, configs = load_model_and_config(ckpt, cfg_path)
        raw_preds = run_inference(head, configs, features)
        
        # Sort predictions matching gts order
        pred_map = {p["file_id"]: p for p in raw_preds}
        ordered_preds = [pred_map[g["file_id"]] for g in gts]

        # Compute Metrics
        f1_macro, f1_details = compute_f1_macro(ordered_preds, gts)
        iou_macro, iou_details = compute_frame_iou_macro(ordered_preds, gts, durations, EVALUATOR_GRID)
        combined_fault_score = 0.5 * f1_macro + 0.5 * iou_macro

        results_table.append({
            "name": name,
            "f1_macro": f1_macro,
            "f1_details": f1_details,
            "iou_macro": iou_macro,
            "iou_details": iou_details,
            "combined": combined_fault_score
        })

        print(f"   * F1-Macro:  {f1_macro:.4f} (Noise: {f1_details['noise']:.4f}, Dist: {f1_details['distortion']:.4f}, Pause: {f1_details['unnatural_pause']:.4f})")
        print(f"   * IoU-Macro: {iou_macro:.4f} (Noise: {iou_details['noise']:.4f}, Dist: {iou_details['distortion']:.4f}, Pause: {iou_details['unnatural_pause']:.4f})")
        print(f"   * Combined Fault Score (0.5*F1 + 0.5*IoU): {combined_fault_score*100:.2f}%")

    print("\n" + "=" * 70)
    print("BẢNG SO SÁNH TRỰC TIẾP TRÊN GOLD TEST SET TIẾNG VIỆT (125 CLIPS)")
    print("=" * 70)
    print(f"{'Model':<45} | {'F1-Macro':<9} | {'IoU-Macro':<9} | {'Combined':<9}")
    print("-" * 75)
    for res in results_table:
        print(f"{res['name']:<45} | {res['f1_macro']:<9.4f} | {res['iou_macro']:<9.4f} | {res['combined']*100:<8.2f}%")

if __name__ == "__main__":
    main()
