"""
Script: scripts/auto_train_pipeline.py
End-to-end training pipeline orchestrator for VLSP 2026 SQA.
Monitors feature caching, trains fault localization frame heads, and triggers downstream training.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

CACHE_DIR = ROOT / "data" / "qualispeech" / "wavlm_cache"
TOTAL_SAMPLES = 14577

SSH_HOST = os.environ.get("SQA_SSH_HOST", "user@remote-gpu-host")
SSH_KEY = os.environ.get("SQA_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))

SSH_CMD = [
    "ssh",
    "-i",
    SSH_KEY,
    "-o",
    "ConnectTimeout=10",
    SSH_HOST,
]


def wait_for_cache():
    print("=" * 70)
    print("PHASE 1: WAITING FOR CACHE EXTRACTION (Target: 14,577 .pt files)")
    print("=" * 70)
    while True:
        if CACHE_DIR.exists():
            count = len(list(CACHE_DIR.glob("*.pt")))
            pct = (count / TOTAL_SAMPLES) * 100
            print(f"[{time.strftime('%H:%M:%S')}] Cached: {count}/{TOTAL_SAMPLES} ({pct:.1f}%)")
            if count >= TOTAL_SAMPLES:
                print("\n-> Cache extraction 100% COMPLETE!")
                break
        else:
            print("Waiting for cache directory...")
        time.sleep(15)


def run_fault_head_training():
    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING FAULT FRAME HEAD (20 Epochs)")
    print("=" * 70)
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "train_fault_head_v5.py"),
    ]
    print("Running command:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError("train_fault_head_v5.py failed!")
    print("\n-> FaultFrameHead training completed successfully!")


def monitor_and_train_server(target_free_mb=18000, poll_seconds=30):
    print("\n" + "=" * 70)
    print(f"PHASE 3: MONITORING REMOTE GPU HOST ({SSH_HOST}) UNTIL FREE >= {target_free_mb} MB")
    print("=" * 70)

    while True:
        try:
            res = subprocess.run(
                SSH_CMD + ["nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if res.returncode == 0:
                free_mb = int(res.stdout.strip().split("\n")[0])
                print(f"[{time.strftime('%H:%M:%S')}] Server GPU Free VRAM: {free_mb} MiB")

                if free_mb >= target_free_mb:
                    print(f"\n-> Server GPU is FREE ({free_mb} MiB)! Starting WavLM LoRA Training in tmux svs_train...")
                    train_cmd = (
                        "tmux send-keys -t svs_train "
                        "'python scripts/train_wavlm_lora.py "
                        "--train-csv data/qualispeech/train.csv "
                        "--val-csv data/qualispeech/val.csv "
                        "--test-csv data/qualispeech/test.csv "
                        "--batch-size 8 "
                        "--grad-accum 2 "
                        "--num-workers 4 "
                        "--lora-r 16 "
                        "--lora-alpha 32 "
                        "--epochs 10' Enter"
                    )
                    subprocess.run(SSH_CMD + [train_cmd], check=True)
                    print("-> Training command SENT to tmux session svs_train!")
                    break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Check error: {e}")

        time.sleep(poll_seconds)


def main():
    print("STARTING PIPELINE ORCHESTRATOR")
    wait_for_cache()
    run_fault_head_training()
    monitor_and_train_server()
    print("\nALL PHASES COMPLETED!")


if __name__ == "__main__":
    main()
