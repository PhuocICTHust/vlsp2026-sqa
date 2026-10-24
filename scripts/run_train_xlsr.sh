#!/usr/bin/env bash
# Reconstruction of the XLS-R-300M LoRA MOS model used in the Tri-Ensemble
# (Hugging Face: vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval).
#
# Hyper-parameters below are taken from that repository's sqa_config.json / README:
#   backbone facebook/wav2vec2-xls-r-300m, LoRA r=16 alpha=32 on q_proj/v_proj,
#   UTMOS-style head (BiLSTM 128 + attention mean/std pooling + MLP 256),
#   loss 0.4 MSE + 0.3 CCC + 0.3 PCC + 0.1 rank (margin 0.1),
#   train = QualiSpeech train.csv + val.csv, epoch selected on test.csv (best epoch 8).
# Batch size / learning rates were not recorded in the config and fall back to the defaults of
# scripts/train_lora_common.py -- adjust them if you need an exact re-run.
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs

"$PYTHON_BIN" scripts/train_lora_common.py \
    --backbone wav2vec2 \
    --model-name facebook/wav2vec2-xls-r-300m \
    --train-csv data/qualispeech/train.csv data/qualispeech/val.csv \
    --val-csv data/qualispeech/test.csv \
    --select-on-test \
    --quality-head-type utmos_nisqa \
    --bilstm-hidden 128 \
    --lora-r 16 \
    --lora-alpha 32 \
    --epochs 8 \
    --w-mse 0.4 \
    --w-ccc 0.3 \
    --w-pcc 0.3 \
    --w-rank 0.1 \
    --rank-margin 0.1 \
    --output-dir outputs/lora_runs/B4b_xlsr_utmos_trainval 2>&1 | tee outputs/train_xlsr_utmos.log
