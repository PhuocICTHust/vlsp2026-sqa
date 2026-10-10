# VLSP 2026 — Speech Quality Assessment (SQA)

Official system repository of team **anhtraiAI** for the **VLSP 2026 Speech Quality Assessment** evaluation campaign.

This repository provides an end-to-end research framework for no-reference speech quality assessment, jointly addressing **perceptual quality estimation (MOS)** and **fine-grained temporal fault localization** (Noise, Distortion, Unnatural Pause).

---

## 1. Task Description & Competition Metric

The task requires evaluating degraded speech audio files without clean reference signals across two coupled sub-tasks:

1. **Perceptual Quality Score ($Q \in [1.0, 5.0]$)**: Utterance-level subjective score predicting Mean Opinion Score (MOS).
2. **Temporal Fault Localization**: Detecting the presence and predicting millisecond-accurate time boundaries $[t_{\text{start}}, t_{\text{end}}]$ for three degradation types:
   * `noise`: Background noise, babble, music, or environmental interference.
   * `distortion`: Acoustic clipping, vocoder artifacts, packet loss, or harmonic distortion.
   * `unnatural_pause`: Abnormal silent intervals or speech disfluencies.

### Official Evaluation Formula
The overall ranking metric is defined as:

$$\text{Final Score} = 0.40 \times \text{PCC}_{\text{MOS}} + 0.30 \times \text{F1}_{\text{macro}} + 0.30 \times \text{IoU}_{\text{macro}}$$

* $\text{PCC}_{\text{MOS}}$: Pearson Correlation Coefficient between predicted quality scores and ground truth ratings.
* $\text{F1}_{\text{macro}}$: Macro-averaged file-level F1 score across the three fault classes (excluding the "no fault" class).
* $\text{IoU}_{\text{macro}}$: Macro-averaged temporal Intersection-over-Union calculated over a 20ms binary frame grid on positive samples.

---

## 2. System Architecture

```text
                             Raw Audio Waveform (16kHz Mono)
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
             Microsoft WavLM-Large            Meta wav2vec2-XLS-R-300M
          (Acoustic / Physical Latents)     (Multilingual & Phonetic Latents)
                      │                                 │
                      └────────────────┬────────────────┘
                                       ▼
                       Fused Representations [T, 2048]
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│     Quality Regression Head     │         │   Temporal Fault Localization   │
│         (40% of Score)          │         │         (60% of Score)          │
├─────────────────────────────────┤         ├─────────────────────────────────┤
│ • Tri-Ensemble MOS Fusion       │         │ • Super 4-Model Heterogeneous   │
│ • SLSQP Optimal Convex Weights: │         │   Ensemble (2048-dim + 1024-dim)│
│   - 24.7% WavLM-Large LoRA      │         │ • Peak Conformer (Micro-spikes) │
│   - 36.4% Frozen WavLM-Large    │         │ • Conformer-Pro (Multi-scale)   │
│   - 38.9% Meta XLS-R 300M Utmos │         │ • Decoupled Tri-Branch (TCN/RNN)│
│ • Attention Statistics Pooling  │         │ • Dual Conformer (Cross-domain) │
└────────────────┬────────────────┘         └────────────────┬────────────────┘
                 ▼                                           ▼
      Predicted Quality Score                    Per-Class Dynamic Hysteresis
            [1.0, 5.0]                          Boundary Dilation & Gap Merging
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       ▼
                         Official Submission Package
                          (PCC ~0.94, F1 ~0.90, IoU ~0.86)
```

### Core Innovations
1. **Cross-Backbone Dual Foundation (2048-dim)**: Combining Microsoft `WavLM-Large` (robust acoustic corruption latents) and Meta `XLS-R-300M` (cross-lingual phonetic representations) eliminates Vietnamese tonal false alarms.
2. **Decoupled Tri-Branch Architecture**: Separating the receptive fields for distinct degradation types (short receptive fields for localized distortion, dilated convolutions for background noise, and bidirectional recurrent units for unnatural pauses).
3. **Multi-Scale Conformer Decoding**: Utilizing Depthwise Separable Convolutions and Relative Multi-Head Self-Attention for sub-second boundary precision.
4. **Metric-Aligned Temporal Post-Processing**: Dual-threshold hysteresis ($\tau_{\text{high}}, \tau_{\text{low}}$) paired with dynamic boundary dilation directly maximizes the discrete Jaccard IoU metric.
5. **SLSQP-Optimized Quality Ensemble**: Constrained convex optimization over out-of-fold validation predictions yields Pearson correlation $\text{PCC} > 0.94$.

---

## 3. Official Submissions Portfolio

To hedge against evaluation distribution shift, five complementary system paradigms were packaged and submitted:

| Submission Package | Baseline / Model Strategy | Key Characteristics | Target Scenario |
| :--- | :--- | :--- | :--- |
| **`anhtraiAI_1.zip`** | Tri-Ensemble MOS + Starter Baseline | Calibrated MOS ($0.94$ PCC) paired with baseline floor | System control & baseline anchor |
| **`anhtraiAI_2.zip`** | Decoupled Tri-Branch (TCN + BiLSTM) | Dedicated single-purpose branch per fault class | Conservative, anti-overfitting defense |
| **`anhtraiAI_3.zip`** | WavLM Peak Tri-Model Ensemble | Peak Conformer with sensitive frame resolution | High-Recall hunter for subtle micro-faults |
| **`anhtraiAI_4.zip`** | Cross-Backbone Dual Conformer | Joint WavLM-Large + XLS-R-300M fusion ($2048$-dim) | Robust against dialectal and phonetic variance |
| **`anhtraiAI_5.zip`** | Super 4-Model Heterogeneous Ensemble | 50% Dual Peak + 25% WavLM Peak + 15% Pro + 10% Decoupled | **Champion Configuration**: Highest F1 & IoU |

---

## 4. Repository Structure

```text
vlsp2026-sqa/
├── scripts/
│   ├── build_submission_v6.py                     # Submission builder: Attempt 2 (Decoupled)
│   ├── build_submission_v10.py                    # Submission builder: Attempt 3 (WavLM Tri-Model)
│   ├── build_submission_v11.py                    # Submission builder: Attempt 4 (Dual Conformer)
│   ├── build_submission_v12.py                    # Submission builder: Attempt 5 (Super 4-Model Ensemble)
│   ├── build_upgraded_tri_ensemble_submission.py  # Submission builder: Attempt 1 (Baseline Floor)
│   ├── package_submission.py                      # Schema validation & zip packager
│   │
│   ├── train_fault_head_v5.py                     # Universal multi-partition training engine
│   ├── train_conformer_fault_head.py              # Base Conformer frame decoder
│   ├── train_conformer_25epochs.py                # Peak Conformer training script
│   ├── train_conformer_pro.py                     # Multi-Scale Conformer-Pro training script
│   ├── train_decoupled_fault_head.py              # Decoupled Tri-Branch training script
│   ├── train_dual_backbone_conformer.py           # Dual-backbone Conformer training script
│   ├── train_all_dual_backbones.py                # Dual architecture suite training
│   │
│   ├── cache_xlsr_features.py                     # Meta XLS-R 300M representation extractor
│   ├── cache_vietnamese_aug.py                    # Audio feature caching pipeline (Round 1)
│   ├── cache_vietnamese_aug_v3.py                 # Audio feature caching pipeline (Round 2)
│   ├── infer_thao_utmos_model.py                  # Inference script for XLS-R Utmos MOS
│   ├── compute_ensemble_new_thao.py               # SLSQP convex optimization for MOS weights
│   └── benchmark_gold_test_set.py                 # Benchmark evaluation on gold test split
│
├── src/
│   └── sqa/
│       ├── dataset.py                             # Waveform loader, padding, and batch collation
│       ├── evaluator.py                           # PCC, F1-macro, and Frame IoU evaluation metrics
│       ├── frame_grid.py                          # Sample-to-frame temporal mapping geometry
│       ├── frame_head.py                          # Frame classification decoders & pooling layers
│       ├── lora.py                                # Low-Rank Adaptation (LoRA) injection
│       ├── model.py                               # UtmosNisqaQualityHead & loss objectives
│       ├── postprocess.py                         # Hysteresis thresholding & interval serialization
│       └── validator.py                           # Official JSON schema validation
│
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 5. Installation & Environment Setup

### Prerequisites
* Linux or Windows with NVIDIA GPU (CUDA 11.8 / 12.1)
* Python 3.10+
* Conda / Miniconda

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/PhuocICTHust/vlsp2026-sqa.git
cd vlsp2026-sqa

# Create and activate conda environment
conda create -n vlsp2026-sqa python=3.10 -y
conda activate vlsp2026-sqa

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers soundfile scipy pandas numpy scikit-learn tqdm safetensors
pip install -e .
```

---

## 6. Training & Inference Workflow

### Step 1: Precompute Foundation Model Representations
Precomputing and caching SSL representations accelerates downstream training by $10\times$:

```bash
# Extract Microsoft WavLM-Large representations
python scripts/cache_vietnamese_aug_v3.py

# Extract Meta XLS-R 300M representations
python scripts/cache_xlsr_features.py
```

### Step 2: Train Temporal Fault Heads
Train the specialized architectures across the target domain partitions:

```bash
# 1. Train Decoupled Tri-Branch Head
python scripts/train_decoupled_fault_head.py

# 2. Train Conformer Peak and Conformer-Pro
python scripts/train_conformer_25epochs.py
python scripts/train_conformer_pro.py

# 3. Train Cross-Backbone Dual Architecture Suite
python scripts/train_all_dual_backbones.py
```

### Step 3: Calibrate Quality Score Ensemble
Solve the constrained convex optimization problem (SLSQP) over validation folds:

```bash
python scripts/compute_ensemble_new_thao.py
```

### Step 4: Generate Official Submissions
Generate the prediction files for the target evaluation audio set:

```bash
# Generate Champion Submission (v12 - Super 4-Model Heterogeneous Ensemble)
python scripts/build_submission_v12.py

# Package and validate against official competition schema
python scripts/package_submission.py --team-name anhtraiAI --attempt 5 --submission-json submissions/submission_v12.json
```

---

## 7. Submission Schema Compliance

Predictions strictly conform to the format published by the VLSP 2026 committee:

```json
[
  {
    "file_id": "test_0001.wav",
    "quality_score": 3.4544,
    "faults": {
      "noise": [[1.18, 1.64], [2.90, 3.80]],
      "distortion": [[3.04, 3.58]],
      "unnatural_pause": []
    }
  }
]
```

* `file_id`: Audio file identifier with extension (`.wav`).
* `quality_score`: Continuous scalar in range $[1.0, 5.0]$.
* `faults`: Dictionary containing arrays of $[t_{\text{start}}, t_{\text{end}}]$ intervals in seconds, or `[]` when no fault is detected.

---

## 8. Authors & Acknowledgments

* **Team**: `anhtraiAI`
* **Affiliation**: Hanoi University of Science and Technology (HUST)
* **Competition**: VLSP 2026 Evaluation Campaign — Vietnamese Language and Speech Processing.

We express our gratitude to the organizers of VLSP 2026 for providing the evaluation benchmarks, and to the open-source community for pretrained models (`microsoft/wavlm-large`, `facebook/wav2vec2-xls-r-300m`).
