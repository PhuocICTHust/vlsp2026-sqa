from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqa.frame_grid import MODEL_GRID
from sqa.model import SQAModel
from sqa.pipeline import run_on_folder
from sqa.postprocess import BASELINE_POSTPROCESS
from sqa.validator import validate_submission


def main():
    parser = argparse.ArgumentParser(description="Build complete VLSP 2026 SQA submission JSON.")
    parser.add_argument("--audio-dir", type=str, required=True, help="Folder containing WAV files to infer.")
    parser.add_argument("--out-json", type=str, default="submission.json", help="Output JSON path.")
    parser.add_argument(
        "--fault-ckpt",
        type=str,
        default="outputs/fault_head/best_fault_frame_head.pt",
        help="Path to trained FaultFrameHead checkpoint.",
    )
    if (ROOT / "checkpoints/wavlm_manual_lora_trainval_best.pt").exists():
        default_quality = "checkpoints/wavlm_manual_lora_trainval_best.pt"
    elif (ROOT / "checkpoints/wavlm_manual_lora_full_best.pt").exists():
        default_quality = "checkpoints/wavlm_manual_lora_full_best.pt"
    else:
        default_quality = "outputs/cached_full/best_quality_head.pt"
    parser.add_argument(
        "--quality-ckpt",
        type=str,
        default=default_quality,
        help="Path to trained QualityHead or full model checkpoint.",
    )
    parser.add_argument(
        "--postprocess-config",
        type=str,
        default="outputs/fault_head/calibrated_postprocess_config.json" if (ROOT / "outputs/fault_head/calibrated_postprocess_config.json").exists() else None,
        help="Path to calibrated_postprocess_config.json.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    fault_ckpt_path = ROOT / args.fault_ckpt
    quality_ckpt_path = ROOT / args.quality_ckpt

    print("=" * 70)
    print("VLSP 2026 SQA FULL SUBMISSION PIPELINE")
    print("=" * 70)
    print(f"Audio directory: {audio_dir}")
    print(f"Output JSON:     {args.out_json}")
    print(f"Device:          {args.device}")

    # Initialize complete SQA model
    fault_kwargs = None
    if fault_ckpt_path.exists():
        fault_payload = torch.load(fault_ckpt_path, map_location="cpu")
        fault_kwargs = fault_payload.get("fault_head_kwargs", {})
        print(f"Loading FaultFrameHead from: {fault_ckpt_path.name}")
    else:
        print(f"Warning: Fault checkpoint {fault_ckpt_path} not found. Using untrained FaultFrameHead.")

    # Detect QualityHead type from checkpoint
    q_type = "standard"
    q_kwargs = None
    q_payload = None
    if quality_ckpt_path.exists():
        q_payload = torch.load(quality_ckpt_path, map_location="cpu")
        if isinstance(q_payload, dict):
            if "quality_head_type" in q_payload:
                q_type = q_payload["quality_head_type"]
                q_kwargs = q_payload.get("quality_head_kwargs", {})
            elif "quality_head_state_dict" in q_payload and any(k.startswith("bilstm.") for k in q_payload["quality_head_state_dict"]):
                q_type = "utmos_nisqa"
                q_kwargs = q_payload.get("quality_head_kwargs", {})

    print(f"Initializing SQAModel with QualityHead type: {q_type}")
    model = SQAModel(
        model_name="microsoft/wavlm-large",
        freeze_backbone=True,
        fault_head_kwargs=fault_kwargs,
        quality_head_kwargs=q_kwargs,
        quality_head_type=q_type,
    ).to(args.device)

    # Load weights
    if fault_ckpt_path.exists():
        model.fault_head.load_state_dict(fault_payload["fault_head_state_dict"])
        print("-> FaultFrameHead weights loaded successfully.")

    if q_payload is not None:
        if "quality_head_state_dict" in q_payload:
            model.quality_head.load_state_dict(q_payload["quality_head_state_dict"])
            print(f"-> QualityHead ({q_type}) weights loaded from full/LoRA checkpoint.")
            if "lora_state_dict" in q_payload:
                from sqa.lora import apply_lora_to_backbone
                apply_lora_to_backbone(model.backbone.backbone, q_payload["lora_state_dict"])
                print("-> WavLM LoRA weights attached and loaded into backbone.")
            if q_payload.get("normalize_wav", False):
                model.normalize_wav = True
                print("-> Audio waveform normalization enabled for model.")
        elif isinstance(q_payload, dict):
            # Direct QualityHead state_dict (from cached_full)
            model.quality_head.load_state_dict(q_payload)
            print("-> QualityHead weights loaded from cached_full checkpoint.")
    else:
        print(f"Warning: Quality checkpoint {quality_ckpt_path} not found.")

    model.eval()

    # Postprocess configuration
    from sqa.postprocess import PostProcessConfig
    post_cfg = BASELINE_POSTPROCESS
    if args.postprocess_config and (ROOT / args.postprocess_config).exists():
        cfg_file = ROOT / args.postprocess_config
        with open(cfg_file, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        if "calibrated_configs" in raw_cfg:
            post_cfg = {
                k: PostProcessConfig(**{param: val for param, val in v.items() if param != "is_baseline"})
                for k, v in raw_cfg["calibrated_configs"].items()
            }
            print(f"Loaded calibrated postprocessing configs from: {cfg_file.name}")
        else:
            post_cfg = PostProcessConfig(**{param: val for param, val in raw_cfg.items() if param != "is_baseline"})
            print(f"Loaded single postprocessing config from: {cfg_file.name}")

    # Run inference
    print(f"\nRunning inference on folder: {audio_dir} ...")
    entries, duration_lookup = run_on_folder(
        audio_dir=audio_dir,
        model=model,
        grid=MODEL_GRID,
        postprocess_config=post_cfg,
        device=args.device,
    )
    print(f"Generated predictions for {len(entries)} files.")

    # Validate output format
    print("\nValidating submission against official competition constraints...")
    errors = validate_submission(entries, duration_lookup, strict=False)
    if errors:
        print(f"FAILED: Found {len(errors)} validation errors:")
        for e in errors[:10]:
            print("  ", e)
        raise AssertionError("Submission JSON failed validation constraints!")

    print("VALIDATION PASS: 100% compliant with competition requirements.")

    # Save to disk
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved final submission file to: {out_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
