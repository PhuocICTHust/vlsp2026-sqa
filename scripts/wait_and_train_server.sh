#!/bin/bash
# Autonomous GPU watcher & training trigger on server
# Usage: Run inside tmux session `svs_train` on server.

TARGET_FREE_MB=18000
POLL_INTERVAL=30

echo "======================================================================"
echo "SVS SQA: AUTONOMOUS GPU WATCHER & TRAINING TRIGGER"
echo "======================================================================"
echo "Waiting until GPU 0 Free VRAM >= ${TARGET_FREE_MB} MiB..."

while true; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    TIMESTAMP=$(date +'%Y-%m-%d %H:%M:%S')
    echo "[${TIMESTAMP}] GPU Free VRAM: ${FREE_MB} MiB / 24564 MiB"

    if [ "$FREE_MB" -ge "$TARGET_FREE_MB" ]; then
        echo ""
        echo "======================================================================"
        echo "GPU IS READY (${FREE_MB} MiB free)! STARTING TRAINING NOW..."
        echo "======================================================================"
        
        python scripts/train_wavlm_lora.py \
            --train-csv data/qualispeech/train_val.csv \
            --val-csv data/qualispeech/test.csv \
            --checkpoint-path checkpoints/wavlm_manual_lora_trainval_best.pt \
            --batch-size 4 \
            --grad-accum 4 \
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
            --save-preds
            
        echo "======================================================================"
        echo "TRAINING COMPLETE!"
        echo "======================================================================"
        break
    fi

    sleep $POLL_INTERVAL
done
