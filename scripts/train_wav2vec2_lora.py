"""wav2vec 2.0 + PEFT LoRA (q_proj, v_proj) for overall-quality regression.

Mirror of train_hubert_lora.py; only the backbone and its input handling differ.
Wav2Vec2Attention calls self.q_proj(x) / self.v_proj(x) as nn.Linear modules,
so PEFT LoRA wraps them directly (unlike WavLM, which needs the manual patch).

Usage:
    python scripts/train_wav2vec2_lora.py
    python scripts/train_wav2vec2_lora.py --model-name facebook/wav2vec2-xls-r-300m
    python scripts/train_wav2vec2_lora.py --no-spec-augment
"""
from __future__ import annotations

import argparse

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


# PROJECT PATH

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.sqa.dataset import QualiSpeechDataset, qualispeech_collate
from src.sqa.hub import resolve_model
from src.sqa.model import (
    QualityHead,
    quality_loss,
    concordance_correlation_coefficient,
    pearson_correlation,
)


# PATHS

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
WAV_DIR = ROOT / "data" / "qualispeech" / "wav"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = CHECKPOINT_DIR / "wav2vec2_lora_full_best.pt"  # overridden in main() from --model-name


# CONFIG

SEED = 42

# Default: same size (95M, 12 x 768) and pre-training data (LibriSpeech 960 h)
# as facebook/hubert-base-ls960, so the backbone is the only variable.
# Large / multilingual option: facebook/wav2vec2-xls-r-300m (24 x 1024, incl. Vietnamese).
MODEL_NAME = "facebook/wav2vec2-base"

# RTX 3060 6 GB: 1 sample per GPU step, accumulated over 4 steps for stable VRAM & correlation.
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4

EPOCHS = 10
VAL_RATIO = 0.20

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LR_LORA = 5e-5
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4

DROPOUT_HEAD = 0.10

PRINT_EVERY = 50
EVAL_EVERY = 1

NUM_WORKERS = 0

USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# SEED

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# GROUP & SOURCE BALANCED SPLIT (MATCHES BASELINES)

def get_group(sample_id: str) -> str:
    sample_id = str(sample_id).strip()
    if "-" not in sample_id:
        raise ValueError(f"Cannot infer group from sample id: {sample_id}")
    return sample_id.split("-", 1)[0]


def get_source(group: str) -> str:
    group = str(group).strip().lower()
    if group == "sysnists":
        return "nisqa_sim"
    if group == "sysnistl":
        return "nisqa_live"
    if group == "sys0giga":
        return "gigaspeech"
    if group.startswith("sys") and group.endswith("tts"):
        middle = group[3:-3]
        if middle.isdigit():
            return "recent_tts"
    return "bvcc"


def make_source_balanced_split(df: pd.DataFrame):
    """Source-balanced split matching the cached full baselines (8446 train, 2112 val)."""
    rng = np.random.RandomState(SEED)
    train_parts = []
    val_parts = []

    source_order = [
        "bvcc",
        "nisqa_sim",
        "gigaspeech",
        "nisqa_live",
        "recent_tts",
    ]

    for source in source_order:
        source_df = df[df["source"] == source].copy()
        source_train_parts = []
        source_val_parts = []

        for target_value in sorted(source_df["target"].unique()):
            class_df = source_df[source_df["target"] == target_value].copy()
            indices = np.arange(len(class_df))
            rng.shuffle(indices)

            n_val = int(round(len(class_df) * VAL_RATIO))
            if len(class_df) >= 2:
                n_val = max(1, min(n_val, len(class_df) - 1))
            else:
                n_val = 0

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            source_val_parts.append(class_df.iloc[val_indices])
            source_train_parts.append(class_df.iloc[train_indices])

        source_train = pd.concat(source_train_parts, ignore_index=True)
        source_val = pd.concat(source_val_parts, ignore_index=True)

        source_train = source_train.sample(
            frac=1.0, random_state=SEED
        ).reset_index(drop=True)
        source_val = source_val.sample(
            frac=1.0, random_state=SEED
        ).reset_index(drop=True)

        train_parts.append(source_train)
        val_parts.append(source_val)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    train_df = train_df.sample(
        frac=1.0, random_state=SEED
    ).reset_index(drop=True)
    val_df = val_df.sample(
        frac=1.0, random_state=SEED
    ).reset_index(drop=True)

    return train_df, val_df


# INPUT PREPROCESSING (wav2vec2-specific)

# Filled in main() from the checkpoint's preprocessor_config.json.
INPUT_CFG = {"do_normalize": True, "pass_mask": False}


def load_input_cfg(model_name: str, config) -> dict:
    """wav2vec2 checkpoints differ in preprocessing:
    - do_normalize: most expect per-utterance zero-mean / unit-variance input.
    - return_attention_mask: False for feat_extract_norm == "group" models
      (e.g. wav2vec2-base), which were pre-trained without a mask; HF says to
      zero-pad and NOT pass attention_mask. True for "layer" models (XLS-R, large-lv60).
    """
    try:
        fe = AutoFeatureExtractor.from_pretrained(resolve_model(model_name))
        return {
            "do_normalize": bool(getattr(fe, "do_normalize", True)),
            "pass_mask": bool(getattr(fe, "return_attention_mask", False)),
        }
    except Exception as exc:
        print(f"[warn] feature extractor not loaded ({exc}); using config fallback")
        return {
            "do_normalize": True,
            "pass_mask": getattr(config, "feat_extract_norm", "group") == "layer",
        }


def prepare_inputs(wav: torch.Tensor, attention_mask: torch.Tensor):
    if INPUT_CFG["do_normalize"]:
        m = attention_mask.to(wav.dtype)
        n = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (wav * m).sum(dim=1, keepdim=True) / n
        var = (((wav - mean) * m) ** 2).sum(dim=1, keepdim=True) / n
        wav = (wav - mean) / torch.sqrt(var + 1e-7) * m  # padding stays 0
    mask_arg = attention_mask.long() if INPUT_CFG["pass_mask"] else None
    return wav, mask_arg


# FRAME MASK

def make_frame_valid_mask(
    backbone,
    attention_mask: torch.Tensor,
    max_frames: int,
) -> torch.Tensor:
    base_model = backbone.get_base_model()
    sample_lengths = attention_mask.long().sum(dim=1)
    frame_lengths = (
        base_model._get_feat_extract_output_lengths(sample_lengths)
        .to(attention_mask.device)
    )
    frame_idx = torch.arange(
        max_frames,
        device=attention_mask.device,
    ).unsqueeze(0)
    return frame_idx < frame_lengths.unsqueeze(1)


# FORWARD

def forward_batch(
    backbone,
    quality_head,
    batch,
):
    wav = batch["wav"].to(DEVICE, dtype=torch.float32)
    attention_mask = batch["attention_mask"].to(DEVICE)

    model_wav, model_mask = prepare_inputs(wav, attention_mask)
    output = backbone(
        input_values=model_wav,
        attention_mask=model_mask,
    )
    hidden = output.last_hidden_state
    frame_mask = make_frame_valid_mask(
        backbone,
        attention_mask,
        hidden.shape[1],
    )
    pred = quality_head(hidden, frame_mask)
    return pred


# EVALUATION

@torch.no_grad()
def evaluate(
    backbone,
    quality_head,
    loader,
):
    backbone.eval()
    quality_head.eval()

    predictions = []
    targets = []

    for batch in loader:
        pred = forward_batch(backbone, quality_head, batch)
        target = batch["target"].to(DEVICE)

        predictions.append(pred.float().cpu())
        targets.append(target.float().cpu())

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    ccc = concordance_correlation_coefficient(predictions, targets).item()
    pcc = pearson_correlation(predictions, targets).item()

    return {
        "predictions": predictions,
        "targets": targets,
        "mse": mse,
        "mae": mae,
        "ccc": ccc,
        "pcc": pcc,
    }


# TRAINING ONE EPOCH WITH MICRO-BATCH PREDICTION ACCUMULATION

def train_one_epoch(
    backbone,
    quality_head,
    loader,
    optimizer,
    scaler,
    epoch,
    global_step,
    scheduler=None,
):
    backbone.train()
    quality_head.train()

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    accum_preds = []
    accum_targets = []
    num_updates = 0

    use_amp = USE_AMP and DEVICE.type == "cuda"

    for batch_idx, batch in enumerate(loader, start=1):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = forward_batch(backbone, quality_head, batch)
            target = batch["target"].to(DEVICE)

            accum_preds.append(pred)
            accum_targets.append(target)

        if batch_idx % GRAD_ACCUM_STEPS == 0 or batch_idx == len(loader):
            preds_chunk = torch.cat(accum_preds)
            targets_chunk = torch.cat(accum_targets)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                # Hybrid loss: 0.4 MSE + 0.3 (1-CCC) + 0.3 (1-PCC)
                # Gives strong correlation and ranking supervision across the accumulated batch.
                loss = quality_loss(
                    preds_chunk.float(),
                    targets_chunk.float(),
                    w_mse=0.4,
                    w_ccc=0.3,
                    w_pcc=0.3,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_idx}: {loss.item()}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in list(backbone.parameters()) + list(quality_head.parameters())
                    if p.requires_grad
                ],
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            global_step += 1
            running_loss += loss.item()
            num_updates += 1

            accum_preds = []
            accum_targets = []

            if (
                num_updates == 1
                or num_updates % (PRINT_EVERY // GRAD_ACCUM_STEPS or 1) == 0
                or batch_idx == len(loader)
            ):
                print(
                    f"Epoch {epoch:02d} "
                    f"| batch {batch_idx:04d}/{len(loader):04d} "
                    f"| step {global_step:04d} "
                    f"| chunk_loss={loss.item():.6f} "
                    f"| lr_lora={optimizer.param_groups[0]['lr']:.2e}"
                )

    return running_loss / max(num_updates, 1), global_step


# MAIN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--no-spec-augment",
        action="store_true",
        help="Set mask_time_prob = mask_feature_prob = 0 (HF masks hidden states in train mode).",
    )
    return parser.parse_args()


def main():
    global MODEL_NAME, EPOCHS, BEST_CHECKPOINT, INPUT_CFG
    args = parse_args()
    MODEL_NAME = args.model_name
    EPOCHS = args.epochs
    tag = MODEL_NAME.split("/")[-1].replace(".", "_")
    BEST_CHECKPOINT = CHECKPOINT_DIR / f"{tag}_lora_full_best.pt"

    set_seed(SEED)

    print("=" * 70)
    print("wav2vec 2.0 + PEFT LoRA — FULL DATA TRAINING")
    print("=" * 70)
    print("Device:                 ", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:                    ", torch.cuda.get_device_name(0))
    print("Backbone:               ", MODEL_NAME)
    print("Batch size:             ", BATCH_SIZE)
    print("Gradient accumulation:  ", GRAD_ACCUM_STEPS)
    print("Effective batch size:   ", BATCH_SIZE * GRAD_ACCUM_STEPS)
    print("Epochs:                 ", EPOCHS)
    print("Validation ratio:       ", VAL_RATIO)
    print("LoRA rank / alpha:      ", f"{LORA_R} / {LORA_ALPHA}")
    print("LoRA LR / Head LR:      ", f"{LR_LORA} / {LR_HEAD}")
    print("Weight decay:           ", WEIGHT_DECAY)
    print("Head dropout:           ", DROPOUT_HEAD)
    print("AMP:                    ", USE_AMP)
    print("Gradient checkpointing: ", USE_GRADIENT_CHECKPOINTING)

    # DATA PREPARATION (SOURCE-BALANCED)
    df = pd.read_csv(CSV_PATH)
    df["sample_id"] = (
        df["id"].astype(str).str.strip().str.replace(r"\.wav$", "", regex=True)
    )
    df["target"] = pd.to_numeric(df["Overall quality"], errors="raise").astype(float)
    df["group"] = df["sample_id"].map(get_group)
    df["source"] = df["group"].map(get_source)

    train_df, val_df = make_source_balanced_split(df)
    if len(train_df) != 8446 or len(val_df) != 2112:
        print(f"Warning: split count is {len(train_df)} / {len(val_df)} (expected 8446 / 2112).")

    full_dataset = QualiSpeechDataset(
        csv_path=CSV_PATH,
        wav_dir=WAV_DIR,
        indices=list(range(len(df))),
    )

    id_to_idx = {str(fid): i for i, fid in enumerate(full_dataset.df["id"].astype(str))}
    train_indices = [id_to_idx[x] for x in train_df["id"].astype(str)]
    val_indices = [id_to_idx[x] for x in val_df["id"].astype(str)]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=qualispeech_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=qualispeech_collate,
    )

    print()
    print("SPLIT SUMMARY")
    print(f"Train samples: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"Val samples:   {len(val_dataset)}, Batches: {len(val_loader)}")

    # BACKBONE & LORA
    # CTC-fine-tuned checkpoints (e.g. *-960h) also load: lm_head is dropped.
    backbone = Wav2Vec2Model.from_pretrained(resolve_model(MODEL_NAME)).to(DEVICE)
    if hasattr(backbone.config, "layerdrop"):
        backbone.config.layerdrop = 0.0
    if args.no_spec_augment:
        backbone.config.mask_time_prob = 0.0
        backbone.config.mask_feature_prob = 0.0

    INPUT_CFG = load_input_cfg(MODEL_NAME, backbone.config)
    print("Input cfg:              ", INPUT_CFG)
    print("feat_extract_norm:      ", backbone.config.feat_extract_norm)
    print("mask_time_prob:         ", backbone.config.mask_time_prob)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    if hasattr(backbone, "enable_input_require_grads"):
        try:
            backbone.enable_input_require_grads()
        except NotImplementedError:
            pass  # older transformers: speech models have no input embeddings

    if USE_GRADIENT_CHECKPOINTING:
        if hasattr(backbone, "gradient_checkpointing_enable"):
            # Non-reentrant: LoRA grads flow even if the encoder input has
            # requires_grad=False (e.g. after freeze_feature_encoder()).
            try:
                backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                backbone.gradient_checkpointing_enable()

    # QUALITY HEAD & OPTIMIZER
    hidden_size = backbone.config.hidden_size
    quality_head = QualityHead(
        hidden_size=hidden_size,
        dropout=DROPOUT_HEAD,
    ).to(DEVICE)

    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in quality_head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": LR_LORA},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS + 1) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and DEVICE.type == "cuda"),
    )

    # INITIAL VALIDATION
    print()
    print("INITIAL VALIDATION (EPOCH 0)")
    initial = evaluate(backbone, quality_head, val_loader)
    print(
        f"Init | MSE={initial['mse']:.6f} | "
        f"MAE={initial['mae']:.6f} | "
        f"CCC={initial['ccc']:.6f} | "
        f"PCC={initial['pcc']:.6f}"
    )

    best_val_pcc = -float("inf")
    best_epoch = -1
    global_step = 0

    print()
    print("=" * 70)
    print("TRAINING LOOP")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss, global_step = train_one_epoch(
            backbone,
            quality_head,
            train_loader,
            optimizer,
            scaler,
            epoch,
            global_step,
            scheduler=scheduler,
        )

        val_result = evaluate(backbone, quality_head, val_loader)

        print(
            f"\nEpoch {epoch:02d} Complete | "
            f"train_loss={train_loss:.6f} | "
            f"VAL MSE={val_result['mse']:.6f} | "
            f"MAE={val_result['mae']:.6f} | "
            f"CCC={val_result['ccc']:.6f} | "
            f"PCC={val_result['pcc']:.6f}"
        )

        # Checkpoint selection on PCC (aligns directly with competition evaluation).
        if val_result["pcc"] > best_val_pcc:
            best_val_pcc = val_result["pcc"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_name": MODEL_NAME,
                    "hidden_size": hidden_size,
                    "input_cfg": INPUT_CFG,
                    "no_spec_augment": args.no_spec_augment,
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "backbone_lora_state_dict": get_peft_model_state_dict(backbone),
                    "quality_head_state_dict": quality_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mse": val_result["mse"],
                    "val_mae": val_result["mae"],
                    "val_ccc": val_result["ccc"],
                    "val_pcc": val_result["pcc"],
                },
                BEST_CHECKPOINT,
            )
            print(f"  -> NEW BEST CHECKPOINT (epoch={epoch}, VAL PCC={best_val_pcc:.6f})")

    # RESTORE BEST CHECKPOINT
    print()
    print("=" * 70)
    print("RESTORE BEST CHECKPOINT")
    print("=" * 70)

    ckpt = torch.load(BEST_CHECKPOINT, map_location=DEVICE)
    set_peft_model_state_dict(backbone, ckpt["backbone_lora_state_dict"])
    quality_head.load_state_dict(ckpt["quality_head_state_dict"])

    print("Best epoch:      ", ckpt["epoch"])
    print("Best global step:", ckpt["global_step"])
    print("Best val PCC:    ", f"{ckpt['val_pcc']:.6f}")
    print("Checkpoint:      ", BEST_CHECKPOINT)

    final = evaluate(backbone, quality_head, val_loader)
    print()
    print("=" * 70)
    print("FINAL RESULT — BEST VALIDATION CHECKPOINT")
    print("=" * 70)
    print(f"VAL MSE: {final['mse']:.8f}")
    print(f"VAL MAE: {final['mae']:.8f}")
    print(f"VAL CCC: {final['ccc']:.8f}")
    print(f"VAL PCC: {final['pcc']:.8f}")
    print()
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
