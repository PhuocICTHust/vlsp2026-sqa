# A Cross-Backbone Dual-Stream Framework for Vietnamese Speech Quality Assessment and Temporal Fault Localization

Official repository of team **anhtraiAI** (HuSTek Lab & SoICT, Hanoi University of Science and Technology) for the **VLSP 2026 Speech Quality Assessment** shared task.

This repository provides code, configurations, and evaluation tools for no-reference speech quality assessment, jointly addressing **utterance-level perceptual quality estimation (MOS)** and **millisecond-level temporal fault localization** (`noise`, `distortion`, `unnatural_pause`).

---

## 1. Task Description & Competition Metric

Given a degraded 16 kHz mono speech recording without a clean reference signal, the system predicts:

1. **Perceptual Quality Score ($Q \in [1.0, 5.0]$)**: Utterance-level Mean Opinion Score (MOS).
2. **Temporal Fault Localization**: Presence and time intervals $[t_{\text{start}}, t_{\text{end}}]$ in seconds for three fault categories:
   * `noise`: Background noise, babble, music, or environmental interference.
   * `distortion`: Clipping, codec/vocoder artifacts, crackle, or packet loss.
   * `unnatural_pause`: Abnormal silent intervals or mid-clause speech disfluencies.

### Official Evaluation Formula
$$\text{Final Score} = 0.40 \times \text{PCC}_{\text{MOS}} + 0.30 \times \text{F1}_{\text{macro}} + 0.30 \times \text{IoU}_{\text{macro}}$$

* $\text{PCC}_{\text{MOS}}$: Pearson Correlation Coefficient between predicted and ground-truth MOS across all test utterances.
* $\text{F1}_{\text{macro}}$: Unweighted macro-averaged binary clip-level F1 score across the three fault classes.
* $\text{IoU}_{\text{macro}}$: Unweighted macro-averaged frame-level Intersection-over-Union on a 20 ms grid, evaluated conditionally on ground-truth positive recordings for each fault class.

---

## 2. System Architecture

```text
                             Raw Audio Waveform (16 kHz Mono)
                                        │
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
              Microsoft WavLM-Large            Meta wav2vec2-XLS-R-300M
           (1024-d Acoustic Latents)         (1024-d Multilingual Latents)
                       │                                 │
                       ├────────────────┬────────────────┤
                       ▼                                 ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
│      SLSQP Tri-Ensemble MOS Head      │ │   Temporal Fault Localization Heads   │
│         (40% of Final Score)          │ │         (60% of Final Score)          │
├───────────────────────────────────────┤ ├───────────────────────────────────────┤
│ • BiLSTM (d=128) + Attention Stat Pool│ │ • Dual Conformer (2048-d Fused, k=15) │
│ • SLSQP Convex Weights (QualiSpeech): │ │ • WavLM Conformer & Peak (1024-d,k=15)│
│   - 24.7% WavLM-Large LoRA (r=16,α=32)│ │ • Conformer-Pro (k ∈ {5, 15, 31})     │
│   - 36.4% Frozen WavLM-Large ASP      │ │ • Decoupled Tri-Branch (TCN + BiLSTM) │
│   - 38.9% XLS-R-300M ASP              │ │ • Loss: Focal + 0.5 SoftIoU + 0.2 Pres│
└──────────────────┬────────────────────┘ └──────────────────┬────────────────────┘
                   ▼                                         ▼
        Predicted Quality Score                  Per-Class Two-Tier Hysteresis
              [1.0, 5.0]                        [τ_high, τ_low, g_merge, d_min]
```

### Core Components
1. **SLSQP-Blended Tri-Ensemble MOS Head**: Combines three 1024-d Attention-weighted Statistics Pooling (ASP) models trained on English `QualiSpeech` (`trainval`, 12,725 clips): (i) `WavLM-Large` fine-tuned with LoRA ($r=16, \alpha=32$) and a composite MSE+CCC+PCC+Ranking objective, (ii) frozen `WavLM-Large` ASP, and (iii) `wav2vec2-XLS-R-300M` fine-tuned with LoRA ($r=16, \alpha=32$) and an ASP head (publicly hosted at [`vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval`](https://huggingface.co/vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval)). Convex weights ($0.247, 0.364, 0.389$) fit on the 1,852-clip `QualiSpeech` test split achieve $\text{PCC}_{\text{MOS}} = 0.7325$ on `QualiSpeech` test and $0.7031$ zero-shot on the Vietnamese Private Test.
2. **Cross-Backbone Dual Foundation Latents (2048-dim)**: Concatenating frame-level `WavLM-Large` (1024-d) and `XLS-R-300M` (1024-d) representations allows the Conformer decoder to jointly model physical acoustic distortions and Vietnamese tonal phonetic transitions.
3. **Complementary Localization Decoders**: Combines standard `Conformer` ($k=15$, 10-epoch and 25-epoch Peak checkpoints), `Conformer-Pro` (parallel depthwise convolutions $k \in \{5, 15, 31\}$), `DualBackboneConformer` (2048-d), and `DecoupledFaultHead` (dedicated $k=3$ CNN for distortion, dilated $k=7$ TCN for noise, and BiLSTM for pause).
4. **Programmatic Vietnamese Degradation Engine**: Synthesizes 2,000 training utterances (`vietnamese_aug` + `vietnamese_aug_v3/train`) and a 125-clip Vietnamese development benchmark (`vietnamese_test_v1`) from clean Vietnamese speech (`VIVOS`, `Common Voice`, and TTS) with exact programmatic interval labels and 422 ($21.1\%$) clean negative anchors.

---

## 3. Official Competition Submissions & Results (Private Test: 302 Clips)

All five official submissions share the same Tri-Ensemble MOS predictions ($\text{PCC}_{\text{MOS}} = 0.703139$, stored in `submissions/submission_v2.json` and `submissions/submission_v5.json`).

### Summary of Official Runs

| Official Run | Builder Script / JSON | Fault Localization Architecture & Blending Weights | Post-Processing Mode | Dev Score (125 VN / 1,852 QS) | Official Private Test Score |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Run 1 (`anhtraiAI_1.zip`)** | `build_upgraded_tri_ensemble_submission.py` (`submission_v2.json`) | `1.00 FaultFrameHead` (English QualiSpeech Dilated TCN + BiLSTM) | Static (`calibrated_postprocess_config.json`) | `0.5300` | `0.520512` |
| **Run 2 (`anhtraiAI_2.zip`)** | `build_submission_v6.py` (`submission_v6.json`) | `0.70 Conformer (10ep) + 0.30 Decoupled Tri-Branch` | RMS-Adaptive Hysteresis | `0.7955` | `0.518264` |
| **Run 3 (`anhtraiAI_3.zip`)** | `build_submission_v10.py` (`submission_v10.json`) | `0.60 Conformer Peak (25ep) + 0.20 Conformer-Pro + 0.20 Decoupled` | RMS-Adaptive Hysteresis | `0.8064` | `0.527468` |
| **Run 4 (`anhtraiAI_4.zip`)** | `build_submission_v11.py` (`submission_v11.json`) | **`0.60 Dual Conformer (2048-d) + 0.40 Conformer Peak (25ep)`** | **Static Per-Class Hysteresis** | `0.8180` | **`0.530785` (Best)** |
| **Run 5 (`anhtraiAI_5.zip`)** | `build_submission_v12.py` (`submission_v12.json`) | `0.50 Dual + 0.25 Peak + 0.15 Conformer-Pro + 0.10 Decoupled` | Static Per-Class Hysteresis | `0.8229` | `0.524251` |

### Detailed Metric Breakdown on Private Test (Paper Table 3)

| Submission Run | Noise F1 | Distortion F1 | Pause F1 | $\mathbf{F1}_{\text{macro}}$ | Noise IoU | Distortion IoU | Pause IoU | $\mathbf{IoU}_{\text{macro}}$ | Final Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run 1 (`anhtraiAI_1`) | 0.6477 | 0.7823 | 0.0528 | 0.4942 | 0.5227 | 0.2421 | **0.1450** | 0.3033 | 0.520512 |
| Run 2 (`anhtraiAI_2`) | 0.6718 | 0.7732 | 0.0244 | 0.4898 | **0.5797** | 0.2978 | 0.0233 | 0.3002 | 0.518264 |
| Run 3 (`anhtraiAI_3`) | **0.7546** | **0.8252** | 0.0000 | **0.5266** | 0.5363 | **0.3460** | 0.0000 | 0.2941 | 0.527468 |
| **Run 4 (`anhtraiAI_4`)** | 0.7110 | 0.7787 | **0.0822** | 0.5239 | 0.5642 | 0.3333 | 0.0259 | **0.3078** | **`0.530785`** |
| Run 5 (`anhtraiAI_5`) | 0.7086 | 0.7843 | 0.0333 | 0.5087 | 0.5627 | 0.3273 | 0.0137 | 0.3012 | 0.524251 |

---

## 4. Repository Structure

```text
vlsp2026-sqa/
├── scripts/
│   ├── build_upgraded_tri_ensemble_submission.py  # Builder for Official Run 1 (anhtraiAI_1 / v2)
│   ├── build_submission_v6.py                     # Builder for Official Run 2 (anhtraiAI_2 / v6)
│   ├── build_submission_v10.py                    # Builder for Official Run 3 (anhtraiAI_3 / v10)
│   ├── build_submission_v11.py                    # Builder for Official Run 4 (anhtraiAI_4 / v11 - Best)
│   ├── build_submission_v12.py                    # Builder for Official Run 5 (anhtraiAI_5 / v12)
│   ├── package_submission.py                      # Schema validation & zip packager
│   │
│   ├── augment_engine.py                          # Programmatic Vietnamese audio degradation engine
│   ├── build_vi_test_sources.py                   # Clean Vietnamese source selector (VIVOS/CV/TTS)
│   ├── report_aug_stats.py                        # Dataset statistics validator (2,000 clips, durations, class ratios)
│   ├── train_conformer_fault_head.py              # 10-epoch Conformer fault decoder
│   ├── train_conformer_25epochs.py                # 25-epoch Conformer Peak fault decoder
│   ├── train_conformer_pro.py                     # Multi-Scale Conformer-Pro (k=5,15,31) decoder
│   ├── train_decoupled_fault_head.py              # Decoupled Tri-Branch (TCN + BiLSTM) decoder
│   ├── train_dual_backbone_conformer.py           # Cross-Backbone Dual Conformer (2048-d) decoder
│   ├── run_train_xlsr.sh                          # Training recipe for XLS-R-300M LoRA MOS model
│   │
│   ├── cache_vietnamese_aug.py                    # WavLM-Large feature caching (Round 1)
│   ├── cache_vietnamese_aug_v3.py                 # WavLM-Large feature caching (Round 2 & Dev)
│   ├── cache_xlsr_features.py                     # Meta XLS-R-300M feature caching
│   ├── infer_thao_utmos_model.py                  # Inference for XLS-R-300M ASP MOS model
│   ├── compute_ensemble_new_thao.py               # SLSQP convex blending for Tri-Ensemble MOS
│   ├── checkpoint_manifest.py                     # Checkpoint manifest generator and SHA256 integrity verifier
│   ├── benchmark_gold_test_set.py                 # Vietnamese Dev (125 clips) benchmark script (shared & per-run)
│   ├── evaluate_official_submissions.py           # Private Test (302 clips) official benchmark (Tables 3, 4, 6, 7)
│   └── compare_single_vs_ensemble.py              # Single-model vs. ensemble ablation script (Table 5)
│
├── src/
│   └── sqa/
│       ├── dataset.py                             # Waveform loader, padding, and batch collation
│       ├── evaluator.py                           # PCC, F1-macro, and 20ms Frame IoU evaluation
│       ├── frame_grid.py                          # Sample-to-frame temporal grid mapping (20ms stride)
│       ├── frame_head.py                          # Baseline FaultFrameHead (Dilated TCN + BiLSTM)
│       ├── lora.py                                # Low-Rank Adaptation (LoRA) injection
│       ├── model.py                               # UtmosNisqaQualityHead & MOS loss functions
│       ├── postprocess.py                         # Two-tier hysteresis & interval normalization
│       └── validator.py                           # Official JSON submission validator
│
├── checkpoints_manifest.json                      # SHA256 hashes of pre-trained checkpoints
├── pyproject.toml
└── README.md
```

---

## 5. Installation & Reproducibility Workflow

### Setup Instructions
```bash
git clone https://github.com/PhuocICTHust/vlsp2026-sqa.git
cd vlsp2026-sqa

conda create -n vlsp2026-sqa python=3.11 -y
conda activate vlsp2026-sqa

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft soundfile scipy pandas numpy scikit-learn tqdm safetensors huggingface_hub resampy matplotlib
pip install -e .
```

### Pre-trained Model Checkpoints
To reproduce the official submissions (Runs 2–5) without re-running long training routines, download the pre-trained model weights archive (~36.1 MB):

* **Google Drive**: [Download `checkpoints_official_runs_2_to_5.zip`](https://drive.google.com/file/d/17cx71iaNdjlEGXDqKkhjve8r8KFOGuke/view?usp=sharing)
* **Hugging Face Hub (MOS XLS-R model)**: [`vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval`](https://huggingface.co/vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval)

After downloading, extract the archive into `outputs/fault_head_v5/` and verify checksums:
```bash
# Create target directory and unpack
mkdir -p outputs/fault_head_v5
unzip checkpoints_official_runs_2_to_5.zip -d outputs/fault_head_v5/

# Verify SHA256 integrity for the fault localization checkpoints
python scripts/checkpoint_manifest.py --verify --group runs_2_to_5_fault_heads
```

The archive contains:
* `best_conformer_fault_head.pt` (10-epoch Conformer, used in Run 2)
* `best_conformer_25ep.pt` (25-epoch Conformer Peak, used in Runs 3, 4, 5)
* `best_conformer_pro.pt` (Multi-scale Conformer-Pro, used in Runs 3, 5)
* `best_decoupled_fault_head.pt` (Decoupled Tri-Branch, used in Runs 2, 3, 5)
* `best_dual_backbone_conformer.pt` (Cross-Backbone Dual Conformer, used in Runs 4, 5)

### Reproducing Feature Caching, Training, and Official Submissions
```bash
# 1. Cache WavLM-Large (1024-d) and XLS-R-300M (1024-d) frame features
python scripts/cache_vietnamese_aug.py
python scripts/cache_vietnamese_aug_v3.py
python scripts/cache_xlsr_features.py

# 2. Train temporal fault localization heads (optional if checkpoints are downloaded)
python scripts/train_conformer_fault_head.py
python scripts/train_conformer_25epochs.py
python scripts/train_conformer_pro.py
python scripts/train_decoupled_fault_head.py
python scripts/train_dual_backbone_conformer.py

# 3. Evaluate MOS Tri-Ensemble weights on QualiSpeech Test (1,852 clips)
python scripts/compute_ensemble_new_thao.py

# 4. Evaluate fault localization on Vietnamese Dev Benchmark (125 clips, Table 2)
python scripts/benchmark_gold_test_set.py

# 5. Evaluate all official submissions on Private Test Set (302 clips, Tables 3, 4, 6). Table 7 (post-competition oracle diagnostics) is printed from reported values: the experiments that produced it are not included in this repository.
python scripts/evaluate_official_submissions.py

# 6. Compare single models vs. ensemble on Private Test Set (Table 5)
python scripts/compare_single_vs_ensemble.py

# 7. Build Best Official Submission (Run 4 / anhtraiAI_4)
python scripts/build_submission_v11.py
python scripts/package_submission.py --team-name anhtraiAI --attempt 4 --submission-json submissions/submission_v11.json
```

---

## 6. License & Data Terms

* **Code License**: The source code in this repository is licensed under **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike 4.0 International), consistent with the upstream QualiSpeech dataset license.
* **Dataset Redistribution Policy**: In compliance with the QualiSpeech (CC BY-NC-SA 4.0) and BVCC licensing terms, no raw audio files or BVCC-derived annotations are redistributed in this repository. Users must obtain QualiSpeech directly from the official Hugging Face page (`https://huggingface.co/datasets/tsinghua-ee/QualiSpeech`).

---

## 7. Authors & Citation

* **Team**: `anhtraiAI` — Phuoc Duc Pham (Corresponding author: `phuoc.pd235988@sis.hust.edu.vn`), Thao Do Minh Vu, Dat Tien Nguyen, Hieu Trung Tran, Kien Hoang Le, Hoang Duc Nguyen, Nhat Tran Minh Nguyen, Dien Huy Nguyen, Thang Duc Nguyen.
* **Affiliation**: HuSTek Lab & School of Information and Communication Technology (SoICT), Hanoi University of Science and Technology (HUST), Hanoi, Vietnam.

```bibtex
@inproceedings{pham2026crossbackbone,
  title     = {A Cross-Backbone Dual-Stream Framework for Vietnamese Speech Quality Assessment and Temporal Fault Localization},
  author    = {Pham, Phuoc Duc and Vu, Thao Do Minh and Nguyen, Dat Tien and Tran, Hieu Trung and Le, Kien Hoang and Nguyen, Hoang Duc and Nguyen, Nhat Tran Minh and Nguyen, Dien Huy and Nguyen, Thang Duc},
  booktitle = {Proceedings of the 13th International Workshop on Vietnamese Language and Speech Processing (VLSP 2026)},
  year      = {2026}
}
```

