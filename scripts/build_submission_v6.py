"""Submission generator for Official Run 2 (anhtraiAI_2.zip / submission_v6.json).
Vietnamese Synthetic Dev Set (125 clips) Fault Avg: 83.75% (per-run calibrated) / 83.70% (shared threshold).
"""

import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WavLMModel

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
from sqa.model import normalize_waveform
from sqa.postprocess import PostProcessConfig, predict_fault_type, serialize_intervals
from scripts.train_conformer_fault_head import ConformerFaultHead
from scripts.train_decoupled_fault_head import DecoupledFaultHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_DIR = ROOT / "data" / "private_test" / "audio"
SUB_V5_PATH = ROOT / "submissions" / "submission_v5.json"
OUT_JSON = ROOT / "submissions" / "submission_v6.json"
OUT_ZIP = ROOT / "submissions" / "anhtraiAI_6.zip"


def load_audio_mono(audio_path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, float, float]:
    info = sf.info(str(audio_path))
    duration = float(info.duration)
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        import resampy
        wav = resampy.resample(wav, sr, target_sr)
    
    rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
    rms_dbfs = 20.0 * np.log10(rms + 1e-9)
    return torch.from_numpy(wav).float(), duration, float(rms_dbfs)


def main():
    print("=" * 75)
    print("BUILDING OFFICIAL RUN 2 (anhtraiAI_2 / submission_v6.json) - 83.75% DEV FAULT AVG")
    print("=" * 75)
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
    print(f"Loaded {len(mos_dict)} quality scores from SOTA Tri-Ensemble ({mos_source_path.name}).")

    cfgs = {
        "noise": {"tf": 0.55, "tl": 0.15, "md": 0.12, "beta": +0.02},
        "distortion": {"tf": 0.50, "tl": 0.25, "md": 0.08, "beta": -0.02},
        "unnatural_pause": {"tf": 0.50, "tl": 0.15, "md": 0.08, "beta": 0.00},
    }
    print("Super-Ensemble Configurations:")
    for ft, c in cfgs.items():
        print(f"  [{ft.upper()}]: tau_frame={c['tf']}, tau_low={c['tl']}, min_dur={c['md']}, beta={c['beta']:+.2f}")

    model_conf = ConformerFaultHead(1024, 256, 2).eval().to(DEVICE)
    model_conf.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_conformer_fault_head.pt", map_location="cpu"))

    model_dec = DecoupledFaultHead(1024, 128).eval().to(DEVICE)
    model_dec.load_state_dict(torch.load(ROOT / "outputs" / "fault_head_v5" / "best_decoupled_fault_head.pt", map_location="cpu"))

    print("\nLoading WavLM-Large backbone...")
    backbone = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad = False

    wav_files = sorted(list(AUDIO_DIR.glob("*.wav")))
    print(f"Processing {len(wav_files)} Private Test clips...")

    all_rms = []
    audio_cache = {}
    for p in wav_files:
        wav, dur, rms_val = load_audio_mono(p)
        audio_cache[p.name] = (wav, dur, rms_val)
        all_rms.append(rms_val)
    mean_rms = float(np.mean(all_rms))
    std_rms = float(np.std(all_rms))
    print(f"Dataset RMS stats: mean={mean_rms:.2f} dBFS, std={std_rms:.2f} dBFS")

    final_submissions = []

    with torch.no_grad():
        for i, (fname, (wav, dur, rms_val)) in enumerate(audio_cache.items(), start=1):
            wav_gpu = wav.to(DEVICE)[None]
            mask = torch.ones_like(wav_gpu, dtype=torch.long)
            wav_norm = normalize_waveform(wav_gpu, mask)
            hidden = backbone(wav_norm, attention_mask=mask).last_hidden_state
            vmask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=DEVICE)

            prob_conf = torch.sigmoid(model_conf(hidden, valid_mask=vmask))[0].cpu().numpy()
            prob_dec = torch.sigmoid(model_dec(hidden, valid_mask=vmask))[0].cpu().numpy()

            prob_ens = 0.70 * prob_conf + 0.30 * prob_dec

            norm_rms = (rms_val - mean_rms) / (std_rms + 1e-6)

            clip_faults = {}
            for c_idx, ft in enumerate(FAULT_TYPES):
                c = cfgs[ft]
                adj_tau = np.clip(c["tf"] + c["beta"] * norm_rms, 0.15, 0.70)
                adj_low = np.clip(c["tl"] + c["beta"] * 0.5 * norm_rms, 0.10, 0.50)

                post_cfg = PostProcessConfig(
                    tau_frame=float(adj_tau),
                    tau_low=float(adj_low),
                    use_hysteresis=True,
                    min_duration=c["md"],
                    merge_gap=0.04,
                )
                ivs, _, _, _ = predict_fault_type(prob_ens[:, c_idx], MODEL_GRID, dur, post_cfg)
                clip_faults[ft] = serialize_intervals(ivs, round_digits=2, duration=dur)

            assert fname in mos_dict, f"Missing MOS score for {fname}"
            q_score = mos_dict[fname]
            final_submissions.append({
                "file_id": fname,
                "quality_score": round(float(q_score), 4),
                "faults": clip_faults,
            })

            if i % 50 == 0 or i == len(wav_files):
                print(f"Processed {i:3d}/{len(wav_files)} files...")

    final_submissions.sort(key=lambda x: x["file_id"])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_submissions, f, indent=2, ensure_ascii=False)
    print(f"\nSaved submission JSON: {OUT_JSON.resolve()}")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_JSON, arcname="submission.json")
    print(f"Packaged ZIP archive: {OUT_ZIP.resolve()} ({OUT_ZIP.stat().st_size / 1024:.2f} KB)")

    print("\n" + "=" * 75)
    print("VERIFICATION & SUMMARY OF VERSION 6 (OFFICIAL RUN 2)")
    print("=" * 75)
    with zipfile.ZipFile(OUT_ZIP, "r") as zf:
        namelist = zf.namelist()
        print(f"Zip internal files: {namelist}")
        assert namelist == ["submission.json"], "Archive must contain only submission.json"
        with zf.open("submission.json") as zf_entry:
            parsed = json.load(zf_entry)
            assert len(parsed) == 302, f"Expected 302 entries, got {len(parsed)}"
            for it in parsed:
                assert "file_id" in it and "quality_score" in it and "faults" in it
                assert 1.0 <= it["quality_score"] <= 5.0
                for ft in FAULT_TYPES:
                    assert ft in it["faults"]
    print("All verification checks passed 100% successfully!")


if __name__ == "__main__":
    main()
