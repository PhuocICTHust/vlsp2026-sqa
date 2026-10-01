# VLSP 2026 — Speech Quality Assessment

A research-oriented speech quality assessment system developed for **VLSP 2026**.

The project focuses on predicting the perceptual quality of speech audio from raw waveforms using a pretrained **WavLM** speech representation model and lightweight task-specific prediction heads. The current system is designed around continuous speech-quality scoring, with support for additional speech fault analysis.

## Overview

Speech quality assessment aims to estimate how natural, clear, and perceptually acceptable a speech signal is.

This project explores a neural architecture that combines:

* **WavLM** for pretrained speech representations
* **Attention pooling** for temporal feature aggregation
* **Quality regression** for predicting a speech-quality score on a 1–5 scale
* **Concordance Correlation Coefficient (CCC)** together with MSE for training
* Audio-level preprocessing and frame-level masking for variable-length speech
* Feature caching for efficient repeated experimentation
* Evaluation and validation utilities for reproducible experiments

The primary model operates directly on speech waveforms and does not require manually engineered acoustic features as its main input.

---

## Architecture

```text
                    Raw waveform
                         │
                         ▼
                 ┌───────────────┐
                 │     WavLM     │
                 │  pretrained   │
                 │ representation│
                 └───────┬───────┘
                         │
                         ▼
                Frame-level features
                         │
                         ▼
                 ┌───────────────┐
                 │ Attention Pool│
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │   MLP Head    │
                 └───────┬───────┘
                         │
                         ▼
                  Quality Score
                     [1, 5]
```

The WavLM backbone provides contextualized frame-level representations. Attention pooling aggregates these representations into an utterance-level representation before the quality regression head predicts the final score.

Variable-length inputs are handled through padding and attention masks. The sample-level waveform mask is converted to the corresponding frame-level mask before temporal pooling.

---

## Model Components

### WavLM Backbone

The project uses the pretrained:

```text
microsoft/wavlm-large
```

The backbone can be frozen while training the downstream quality head, reducing the number of trainable parameters and computational requirements.

### Attention Pooling

An attention-based pooling layer converts frame-level WavLM representations into a single utterance-level representation.

This allows the model to assign different weights to different temporal regions of an utterance instead of using simple mean pooling.

### Quality Head

The quality head consists of:

```text
Attention Pooling
      ↓
Linear
      ↓
ReLU
      ↓
Linear
      ↓
Sigmoid → [1, 5]
```

The output is constrained to the 1–5 quality-score range.

### Training Objective

The quality model combines Mean Squared Error and Concordance Correlation Coefficient:

```text
L = 0.5 × MSE + 0.5 × (1 − CCC)
```

MSE measures numerical prediction error, while CCC evaluates agreement between predicted and reference quality scores.

---

## Dataset

The project currently uses **QualiSpeech** for model development and experimentation.

QualiSpeech provides speech samples annotated with multiple perceptual quality dimensions, including:

* Speed
* Naturalness
* Background noise
* Distortion
* Listening effort
* Continuity
* Overall quality
* Feeling of voice
* Unnatural pause

The primary regression target used by this project is:

```text
Overall quality
```

with scores ranging from **1 to 5**.

The dataset contains more samples than the redistributable audio archive. Depending on the available data package, the local audio set may therefore cover only a subset of the metadata.

### Dataset Source

QualiSpeech is available through Hugging Face:

`tsinghua-ee/QualiSpeech`

The dataset is distributed under its respective licensing terms. Users should consult the original dataset repository before downloading or redistributing the data.

---

## Repository Structure

```text
vlsp2026-sqa/
│
├── data/
│   ├── acceptance_set/
│   └── qualispeech/
│
├── scripts/
│   ├── make_acceptance_set.py
│   ├── run_alignment_test.py
│   ├── run_milestone_a.py
│   ├── run_training_smoke_test.py
│   ├── run_quality_head_overfit.py
│   ├── run_real_tiny_overfit.py
│   └── ...
│
├── src/
│   └── sqa/
│       ├── cache.py
│       ├── dataset.py
│       ├── evaluator.py
│       ├── frame_grid.py
│       ├── model.py
│       ├── pipeline.py
│       ├── postprocess.py
│       └── validator.py
│
├── tests/
│
├── pyproject.toml
└── README.md
```

### Core Modules

| Module           | Description                                                                          |
| ---------------- | ------------------------------------------------------------------------------------ |
| `model.py`       | WavLM backbone, attention pooling, quality head, loss functions and model definition |
| `dataset.py`     | Dataset loading, waveform processing and batching                                    |
| `frame_grid.py`  | Mapping between waveform time and model frames                                       |
| `cache.py`       | Feature caching utilities                                                            |
| `pipeline.py`    | Model inference and processing pipeline                                              |
| `postprocess.py` | Prediction post-processing                                                           |
| `evaluator.py`   | Evaluation utilities and metrics                                                     |
| `validator.py`   | Prediction and output validation                                                     |

---

## Installation

Create a Python environment and install the project dependencies.

Example:

```bash
conda create -n vlsp2026-sqa python=3.10
conda activate vlsp2026-sqa
```

Install PyTorch with a CUDA build appropriate for the local GPU, followed by the project dependencies.

For example:

```bash
pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Dataset Preparation

Place the QualiSpeech metadata and audio archive under:

```text
data/qualispeech/
```

For example:

```text
data/qualispeech/
├── train.csv
└── wav_part1.zip
```

The dataset loader can read WAV files directly from the archive, avoiding the need to extract the complete archive before experimentation.

---

## Running the Project

### Tests

Run the test suite with:

```bash
python -m pytest -q
```

### Alignment Verification

The temporal alignment between the waveform and WavLM frame representation can be inspected with:

```bash
python scripts/run_alignment_test.py
```

### Acceptance Pipeline

Generate the synthetic acceptance set:

```bash
python scripts/make_acceptance_set.py --out data/acceptance_set
```

Run the acceptance pipeline:

```bash
python scripts/run_milestone_a.py
```

### Training Smoke Test

The training smoke test verifies the complete forward/backward path on variable-length audio:

```bash
python scripts/run_training_smoke_test.py
```

### Quality Head Overfit Test

The isolated quality head can be tested independently of WavLM:

```bash
python scripts/run_quality_head_overfit.py
```

### Real-data Tiny Overfit

A small subset of QualiSpeech can be used to validate the complete real-data training path:

```bash
python scripts/run_real_tiny_overfit.py
```

These scripts are development and validation utilities rather than final benchmark commands.

---

## Evaluation

The project uses regression-oriented metrics for speech quality prediction.

### Mean Squared Error

Measures the squared difference between predicted and reference quality scores:

```text
MSE = mean((y_pred − y_true)²)
```

### Mean Absolute Error

Measures the average absolute prediction error:

```text
MAE = mean(|y_pred − y_true|)
```

### Concordance Correlation Coefficient

CCC evaluates agreement between predictions and reference scores while considering both correlation and systematic differences in location and scale.

CCC is particularly useful for continuous subjective quality scores where agreement with human ratings is more important than correlation alone.

---

## Frame Alignment

The WavLM representation operates on a temporal frame grid rather than individual waveform samples.

The project therefore maintains an explicit model-frame definition containing:

* frame hop
* frame offset
* frame duration
* source of the temporal calibration

This allows frame-level predictions or temporal annotations to be mapped consistently between waveform time and model representation time.

---

## Feature Caching

Feature extraction from a large pretrained speech model can be substantially more expensive than training a lightweight prediction head.

The project therefore provides a feature-cache layer for storing intermediate representations.

Cache keys incorporate information associated with the representation, including:

* dataset version
* file identifier
* backbone checkpoint
* preprocessing configuration

This prevents incompatible representations from being silently reused across experiments.

---

## Validation

The repository includes validation utilities for checking model outputs and evaluation inputs.

Validation covers cases such as:

* valid score ranges
* malformed predictions
* missing predictions
* out-of-range temporal intervals
* variable-duration audio
* empty prediction outputs

The synthetic acceptance set is intended to validate software contracts and pipeline behavior. It is **not a substitute for a real speech-quality benchmark**.

---

## Current Development Status

The repository currently contains the core infrastructure for:

* WavLM-based speech representation
* Quality regression
* Variable-length waveform processing
* Attention-based temporal pooling
* MSE + CCC training
* Frame/time alignment
* Feature caching
* Evaluation and validation
* Real-data training smoke tests

The model-development stage is focused on validating the quality regression pipeline on QualiSpeech before large-scale training and benchmark evaluation.

---

## Reproducibility

Experiments should record at least:

* model checkpoint
* dataset version
* preprocessing configuration
* training configuration
* random seed
* optimizer and learning rate
* batch size
* evaluation split
* feature-cache version

This is particularly important when comparing experiments involving cached WavLM representations.

---

## License

This repository contains research code. Dataset files and pretrained model weights may be subject to separate licenses and terms.

Please refer to the original licenses of:

* QualiSpeech
* WavLM
* Hugging Face Transformers
* PyTorch

before redistributing datasets, model weights, or derived artifacts.

---

## Acknowledgements

This project builds upon open-source speech representation and machine learning research, including:

* Microsoft WavLM
* Hugging Face Transformers
* PyTorch
* QualiSpeech

The project is developed in the context of **VLSP 2026 Speech Quality Assessment**.
