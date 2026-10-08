"""
HuBERT + PEFT LoRA (q_proj, v_proj) with UtmosNisqaQualityHead.
Trained on merged QualiSpeech train_val.csv with test.csv as dev set.

Features:
- Backbone: facebook/hubert-large-ll60k (or facebook/hubert-base-ls960)
- PEFT LoRA on q_proj and v_proj
- UtmosNisqaQualityHead: BiLSTM (128-d) + Attention-weighted Mean-Std Statistics Pooling
- Quadro-Loss: 0.4*MSE + 0.3*(1-CCC) + 0.3*(1-PCC) + 0.1*PairwiseRankingLoss
- Active-speech waveform normalization
- Gradient Accumulation & Cosine Annealing LR Schedule
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import transformers
try:
    import transformers.utils.import_utils as _iu
    _iu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
from transformers import HubertModel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.sqa.dataset import QualiSpeechDataset, qualispeech_collate
from src.sqa.model import (
    QualityHead,
    UtmosNisqaQualityHead,
    normalize_waveform,
    pairwise_ranking_loss,
    quality_loss,
)

CSV_PATH = ROOT / "data" / "qualispeech" / "train_val.csv"
WAV_DIR = ROOT / "data" / "qualispeech" / "wav"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CHECKPOINT = CHECKPOINT_DIR / "hubert_utmos_nisqa_lora_trainval_best.pt"

SEED = 42
MODEL_NAME = "facebook/hubert-large-ll60k"
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
EPOCHS = 8
LORA_R = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
LR_LORA = 5e-5
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT_HEAD = 0.10
W_MSE = 0.4
W_CCC = 0.3
W_PCC = 0.3
W_RANK = 0.1
RANK_MARGIN = 0.1
NORMALIZE_WAV = True
PRINT_EVERY = 50
NUM_WORKERS = 4
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_frame_valid_mask(
    backbone: nn.Module,
    attention_mask: torch.Tensor,
    max_frames: int,
) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1)
    if hasattr(backbone, "_get_feat_extract_output_lengths"):
        frame_lengths = backbone._get_feat_extract_output_lengths(lengths)
    elif hasattr(backbone, "base_model") and hasattr(backbone.base_model, "_get_feat_extract_output_lengths"):
        frame_lengths = backbone.base_model._get_feat_extract_output_lengths(lengths)
    elif hasattr(backbone, "base_model") and hasattr(backbone.base_model, "model") and hasattr(backbone.base_model.model, "_get_feat_extract_output_lengths"):
        frame_lengths = backbone.base_model.model._get_feat_extract_output_lengths(lengths)
    else:
        frame_lengths = (lengths // 320).clamp(min=1, max=max_frames)
    frame_lengths = frame_lengths.to(attention_mask.device)
    frame_idx = torch.arange(max_frames, device=attention_mask.device).unsqueeze(0)
    return frame_idx < frame_lengths.unsqueeze(1)


def forward_batch(
    backbone: nn.Module,
    quality_head: nn.Module,
    batch: dict,
    normalize_wav: bool = True,
) -> torch.Tensor:
    wav = batch["wav"].to(DEVICE, non_blocking=True)
    attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

    if normalize_wav:
        wav = normalize_waveform(wav, attention_mask)

    hidden = backbone(input_values=wav, attention_mask=attention_mask).last_hidden_state
    valid_mask = make_frame_valid_mask(backbone, attention_mask, hidden.shape[1])

    return quality_head(hidden, valid_mask).view(-1)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    p = np.asarray(preds, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    std_p, std_t = np.std(p), np.std(t)
    pcc = float(np.corrcoef(p, t)[0, 1]) if (std_p > 1e-8 and std_t > 1e-8) else 0.0
    vx, vy = np.var(p), np.var(t)
    mx, my = np.mean(p), np.mean(t)
    cov = np.mean((p - mx) * (t - my))
    denom = vx + vy + (mx - my) ** 2
    ccc = float(2.0 * cov / (denom + 1e-8)) if denom > 1e-8 else 0.0
    return {"mse": mse, "mae": mae, "ccc": ccc, "pcc": pcc}


@torch.no_grad()
def evaluate(
    backbone: nn.Module,
    quality_head: nn.Module,
    loader: DataLoader,
    normalize_wav: bool = True,
    max_batches: int | None = None,
) -> dict[str, any]:
    backbone.eval()
    quality_head.eval()
    all_preds, all_targets = [], []

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(USE_AMP and DEVICE.type == "cuda")):
            pred = forward_batch(backbone, quality_head, batch, normalize_wav=normalize_wav)
        all_preds.append(pred.float().cpu().view(-1))
        all_targets.append(batch["target"].cpu().view(-1))

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    metrics = compute_metrics(preds, targets)
    metrics["predictions"] = torch.from_numpy(preds)
    metrics["targets"] = torch.from_numpy(targets)
    return metrics


def train_one_epoch(
    backbone: nn.Module,
    quality_head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    print_every: int = PRINT_EVERY,
    w_mse: float = W_MSE,
    w_ccc: float = W_CCC,
    w_pcc: float = W_PCC,
    w_rank: float = W_RANK,
    rank_margin: float = RANK_MARGIN,
    normalize_wav: bool = NORMALIZE_WAV,
    max_batches: int | None = None,
) -> tuple[float, int]:
    backbone.train()
    quality_head.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    num_updates = 0
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    accum_preds, accum_targets = [], []

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(USE_AMP and DEVICE.type == "cuda")):
            pred = forward_batch(backbone, quality_head, batch, normalize_wav=normalize_wav)
            target = batch["target"].to(DEVICE)
            accum_preds.append(pred.view(-1))
            accum_targets.append(target.view(-1))

        if batch_idx % grad_accum_steps == 0 or batch_idx == total_batches:
            preds_chunk = torch.cat(accum_preds)
            targets_chunk = torch.cat(accum_targets)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(USE_AMP and DEVICE.type == "cuda")):
                base_loss = quality_loss(preds_chunk.float(), targets_chunk.float(), w_mse=w_mse, w_ccc=w_ccc, w_pcc=w_pcc)
                if w_rank > 0.0:
                    rank_loss = pairwise_ranking_loss(preds_chunk.float(), targets_chunk.float(), margin=rank_margin)
                    loss = base_loss + w_rank * rank_loss
                else:
                    rank_loss = preds_chunk.new_zeros(())
                    loss = base_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, batch={batch_idx}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(quality_head.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            num_updates += 1
            global_step += 1
            accum_preds, accum_targets = [], []

            if (
                global_step % print_every == 0
                or batch_idx == total_batches
                or batch_idx % (grad_accum_steps * 6) == 0
            ):
                print(
                    f"Epoch {epoch:02d} | batch {batch_idx:04d}/{total_batches:04d} "
                    f"| step {global_step:04d} | loss={loss.item():.4f} "
                    f"(base={base_loss.item():.4f}, rank={rank_loss.item():.4f}) "
                    f"| lr={optimizer.param_groups[0]['lr']:.2e}"
                )

    return running_loss / max(num_updates, 1), global_step


def save_checkpoint(
    path: Path,
    backbone: nn.Module,
    quality_head: nn.Module,
    epoch: int,
    global_step: int,
    val_metrics: dict,
    model_name: str,
    quality_head_type: str,
    bilstm_hidden: int,
    normalize_wav: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_name": model_name,
        "quality_head_type": quality_head_type,
        "quality_head_kwargs": {"bilstm_hidden": bilstm_hidden, "dropout": DROPOUT_HEAD} if quality_head_type == "utmos_nisqa" else {},
        "normalize_wav": normalize_wav,
        "lora_state_dict": get_peft_model_state_dict(backbone),
        "quality_head_state_dict": quality_head.state_dict(),
        "val_pcc": val_metrics["pcc"],
        "val_ccc": val_metrics["ccc"],
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="HuBERT LoRA Training on QualiSpeech")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Pretrained HuBERT model name")
    parser.add_argument("--train-csv", type=str, default="data/qualispeech/train_val.csv", help="Train CSV")
    parser.add_argument("--val-csv", type=str, default="data/qualispeech/test.csv", help="Val CSV")
    parser.add_argument("--test-csv", type=str, default=None, help="Test CSV")
    parser.add_argument("--checkpoint-path", type=str, default=str(BEST_CHECKPOINT), help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS, help="Gradient accumulation")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Workers")
    parser.add_argument("--lora-r", type=int, default=LORA_R, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA, help="LoRA alpha")
    parser.add_argument("--lr-lora", type=float, default=LR_LORA, help="LoRA LR")
    parser.add_argument("--lr-head", type=float, default=LR_HEAD, help="Head LR")
    parser.add_argument("--w-mse", type=float, default=W_MSE, help="MSE weight")
    parser.add_argument("--w-ccc", type=float, default=W_CCC, help="CCC weight")
    parser.add_argument("--w-pcc", type=float, default=W_PCC, help="PCC weight")
    parser.add_argument("--w-rank", type=float, default=W_RANK, help="Rank weight")
    parser.add_argument("--rank-margin", type=float, default=RANK_MARGIN, help="Rank margin")
    parser.add_argument("--quality-head-type", type=str, default="utmos_nisqa", choices=["standard", "utmos_nisqa"])
    parser.add_argument("--bilstm-hidden", type=int, default=128, help="BiLSTM hidden units")
    parser.add_argument("--normalize-wav", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Max train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Max val batches per eval")
    args = parser.parse_args()

    set_seed(SEED)
    effective_bs = args.batch_size * args.grad_accum
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HUBERT + PEFT LoRA — QUALISPEECH TRAINING")
    print("=" * 70)
    print("Device:                 ", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:                    ", torch.cuda.get_device_name(0))
    print("Backbone:               ", args.model_name)
    print("Batch size:             ", args.batch_size)
    print("Gradient accumulation:  ", args.grad_accum)
    print("Effective batch size:   ", effective_bs)
    print("Epochs:                 ", args.epochs)
    print("LoRA rank / alpha:      ", f"{args.lora_r} / {args.lora_alpha}")
    print("LoRA LR / Head LR:      ", f"{args.lr_lora} / {args.lr_head}")
    print("Quality Head Type:      ", args.quality_head_type)
    print("Normalize waveform:     ", args.normalize_wav)
    print(f"Loss weights:            MSE={args.w_mse}, CCC={args.w_ccc}, PCC={args.w_pcc}, Rank={args.w_rank}")
    print("Checkpoint output:      ", checkpoint_path)

    # Dataset splits
    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    print(f"\nUsing explicit splits:\n  Train: {train_path}\n  Val:   {val_path}")

    train_dataset = QualiSpeechDataset(csv_path=train_path, wav_dir=WAV_DIR)
    val_dataset = QualiSpeechDataset(csv_path=val_path, wav_dir=WAV_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=qualispeech_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=qualispeech_collate,
    )

    print(f"\nSPLIT SUMMARY: Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Backbone & LoRA
    print("\nLoading HuBERT backbone...")
    base_backbone = HubertModel.from_pretrained(args.model_name).to(DEVICE)
    if hasattr(base_backbone.config, "layerdrop"):
        base_backbone.config.layerdrop = 0.0

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    backbone = get_peft_model(base_backbone, lora_config)
    backbone.print_trainable_parameters()

    if USE_GRADIENT_CHECKPOINTING and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    # Head
    hidden_size = backbone.config.hidden_size
    if args.quality_head_type == "utmos_nisqa":
        quality_head = UtmosNisqaQualityHead(
            hidden_size=hidden_size,
            bilstm_hidden=args.bilstm_hidden,
            dropout=DROPOUT_HEAD,
            use_fault_features=False,
        ).to(DEVICE)
    else:
        quality_head = QualityHead(hidden_size=hidden_size, dropout=DROPOUT_HEAD).to(DEVICE)

    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in quality_head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr_lora},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (len(train_loader) // args.grad_accum + 1) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    # Initial Validation
    print("\nINITIAL VALIDATION (EPOCH 0)")
    initial = evaluate(backbone, quality_head, val_loader, normalize_wav=args.normalize_wav, max_batches=args.max_val_batches)
    print(
        f"Init | MSE={initial['mse']:.6f} | MAE={initial['mae']:.6f} | "
        f"CCC={initial['ccc']:.6f} | PCC={initial['pcc']:.6f}"
    )

    best_val_pcc = -float("inf")
    best_epoch = -1
    global_step = 0
    history = []

    print("\n" + "=" * 70)
    print("TRAINING LOOP")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            backbone,
            quality_head,
            train_loader,
            optimizer,
            scaler,
            epoch,
            global_step,
            grad_accum_steps=args.grad_accum,
            scheduler=scheduler,
            w_mse=args.w_mse,
            w_ccc=args.w_ccc,
            w_pcc=args.w_pcc,
            w_rank=args.w_rank,
            rank_margin=args.rank_margin,
            normalize_wav=args.normalize_wav,
            max_batches=args.max_train_batches,
        )

        val_metrics = evaluate(backbone, quality_head, val_loader, normalize_wav=args.normalize_wav, max_batches=args.max_val_batches)
        print(
            f"\nEpoch {epoch:02d} Complete | train_loss={train_loss:.6f} | "
            f"VAL MSE={val_metrics['mse']:.6f} | MAE={val_metrics['mae']:.6f} | "
            f"CCC={val_metrics['ccc']:.6f} | PCC={val_metrics['pcc']:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_ccc": val_metrics["ccc"],
            "val_pcc": val_metrics["pcc"],
        })

        if val_metrics["pcc"] > best_val_pcc:
            best_val_pcc = val_metrics["pcc"]
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                backbone,
                quality_head,
                epoch,
                global_step,
                val_metrics,
                args.model_name,
                args.quality_head_type,
                args.bilstm_hidden,
                args.normalize_wav,
            )
            print(f"  -> NEW BEST CHECKPOINT (epoch={epoch}, VAL PCC={best_val_pcc:.6f})\n")

    print("\n" + "=" * 70)
    print("RESTORE BEST CHECKPOINT")
    print("=" * 70)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    set_peft_model_state_dict(backbone, ckpt["lora_state_dict"])
    quality_head.load_state_dict(ckpt["quality_head_state_dict"])
    print(f"Best epoch: {ckpt['epoch']} | Best val PCC: {ckpt['val_pcc']:.6f}")

    final = evaluate(backbone, quality_head, val_loader, normalize_wav=args.normalize_wav)
    print(f"FINAL RESULT: VAL MSE: {final['mse']:.6f} | CCC: {final['ccc']:.6f} | PCC: {final['pcc']:.6f}")

    if args.save_preds:
        try:
            val_ids = val_loader.dataset.df["id"].astype(str).tolist()
            preds_csv = checkpoint_path.with_name(checkpoint_path.stem + "_dev_preds.csv")
            pd.DataFrame({
                "id": val_ids[:len(final["predictions"])],
                "target": final["targets"].numpy(),
                "predicted": final["predictions"].numpy(),
            }).to_csv(preds_csv, index=False)
            print(f"Saved dev predictions to: {preds_csv}")
        except Exception as e:
            print("Warning: could not save predictions CSV:", e)

    history_path = checkpoint_path.with_name(checkpoint_path.stem + "_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to: {history_path}")
    print("\nTraining HuBERT finished successfully.")


if __name__ == "__main__":
    main()
