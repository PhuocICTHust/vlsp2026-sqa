#!/usr/bin/env bash
set -e
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "======================================================================"
echo "WAITING FOR ACTIVE HUBERT LORA TRAINING TO COMPLETE..."
echo "======================================================================"

while tmux has-session -t hubert_train 2>/dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] hubert_train is still running. Waiting 30 seconds..."
    sleep 30
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] HuBERT LoRA training finished!"

echo ""
echo "======================================================================"
echo "STEP 1: EXTRACTING & CACHING HUBERT-LARGE FEATURES"
echo "======================================================================"
"$PYTHON_BIN" -u scripts/cache_hubert_large.py \
    --model-name facebook/hubert-large-ll60k \
    --csv-path data/qualispeech/train_full.csv \
    --wav-dir data/qualispeech/wav \
    --cache-dir data/qualispeech/hubert_large_cache \
    --batch-size 4 \
    --num-workers 0 \
    --normalize-wav 2>&1 | tee outputs/cache_hubert_large.log

echo ""
echo "======================================================================"
echo "STEP 2: TRAINING FROZEN HUBERT + UTMOS-NISQA HEAD"
echo "======================================================================"
mkdir -p outputs/cached_trainval_hubert_utmos

"$PYTHON_BIN" -u scripts/train_wavlm_frozen.py \
    --train-csv data/qualispeech/train_val.csv \
    --val-csv data/qualispeech/test.csv \
    --cache-dir data/qualispeech/hubert_large_cache \
    --out-dir outputs/cached_trainval_hubert_utmos \
    --quality-head-type utmos_nisqa \
    --bilstm-hidden 128 \
    --batch-size 32 \
    --epochs 20 \
    --lr 2e-4 \
    --w-mse 0.4 \
    --w-ccc 0.3 \
    --w-pcc 0.3 \
    --w-rank 0.1 \
    --rank-margin 0.1 2>&1 | tee outputs/train_hubert_frozen.log

echo ""
echo "======================================================================"
echo "ALL PIPELINE TASKS COMPLETED SUCCESSFULLY!"
echo "======================================================================"
