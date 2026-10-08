#!/bin/bash
set -e
cd /home/svs/audio_exp
mkdir -p outputs checkpoints

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting UtmosNisqaQualityHead WavLM LoRA Training on Train+Val with Test as Dev..."
/home/svs/miniforge3/envs/svs/bin/python scripts/train_wavlm_lora.py \
    --train-csv data/qualispeech/train_val.csv \
    --val-csv data/qualispeech/test.csv \
    --checkpoint-path checkpoints/wavlm_utmos_nisqa_lora_trainval_best.pt \
    --quality-head-type utmos_nisqa \
    --bilstm-hidden 128 \
    --batch-size 2 \
    --grad-accum 8 \
    --num-workers 4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --epochs 8 \
    --w-mse 0.4 \
    --w-ccc 0.3 \
    --w-pcc 0.3 \
    --w-rank 0.1 \
    --rank-margin 0.1 \
    --normalize-wav \
    --save-preds 2>&1 | tee outputs/train_utmos_nisqa.log
