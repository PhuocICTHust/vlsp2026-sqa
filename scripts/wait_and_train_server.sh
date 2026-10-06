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
            --train-csv data/qualispeech/train.csv \
            --val-csv data/qualispeech/val.csv \
            --test-csv data/qualispeech/test.csv \
            --batch-size 8 \
            --grad-accum 2 \
            --num-workers 4 \
            --lora-r 16 \
            --lora-alpha 32 \
            --epochs 10
            
        echo "======================================================================"
        echo "TRAINING COMPLETE!"
        echo "======================================================================"
        break
    fi

    sleep $POLL_INTERVAL
done
