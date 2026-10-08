from __future__ import annotations

import argparse
import json
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import WavLMModel


# ======================================================================
# PROJECT PATH
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.sqa.dataset import QualiSpeechDataset, qualispeech_collate
from src.sqa.model import (
    QualityHead,
    UtmosNisqaQualityHead,
    quality_loss,
    concordance_correlation_coefficient,
    pearson_correlation,
    pairwise_ranking_loss,
    normalize_waveform,
)


# ======================================================================
# PATHS
# ======================================================================

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
WAV_DIR = ROOT / "data" / "qualispeech" / "wav"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = CHECKPOINT_DIR / "wavlm_manual_lora_full_best.pt"


# ======================================================================
# CONFIG
# ======================================================================

SEED = 42

MODEL_NAME = "microsoft/wavlm-large"

# Default: 1 sample per forward, accumulated over 16 steps (effective batch size 16)
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16

EPOCHS = 8
VAL_RATIO = 0.20

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

LR_LORA = 5e-5
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4

DROPOUT_HEAD = 0.10

# Loss weights
W_MSE = 0.4
W_CCC = 0.3
W_PCC = 0.3
W_RANK = 0.1
RANK_MARGIN = 0.1
NORMALIZE_WAV = True

PRINT_EVERY = 50
EVAL_EVERY = 1

NUM_WORKERS = 0

USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True

EXPECTED_WAVLM_LAYERS = 24
EXPECTED_LORA_TENSORS = EXPECTED_WAVLM_LAYERS * 2 * 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# SEED
# ======================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# SOURCE-BALANCED SPLIT (MATCHES BASELINES)
# ======================================================================

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


def make_source_balanced_split(df: pd.DataFrame, val_ratio: float = VAL_RATIO):
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

            n_val = int(round(len(class_df) * val_ratio))
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


# ======================================================================
# DISABLE LAYERDROP
# ======================================================================

def disable_layerdrop(backbone) -> None:
    if hasattr(backbone.config, "layerdrop"):
        backbone.config.layerdrop = 0.0
    if hasattr(backbone.config, "encoder_layerdrop"):
        backbone.config.encoder_layerdrop = 0.0
    if hasattr(backbone, "encoder"):
        if hasattr(backbone.encoder, "layerdrop"):
            backbone.encoder.layerdrop = 0.0
    print("LayerDrop: disabled")


# ======================================================================
# MANUAL LoRA
# ======================================================================

class ManualLoRA(nn.Module):
    """
    W' = W + scaling * B @ A
    A: [r, in_features]
    B: [out_features, r]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0.")

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)


# ======================================================================
# ATTACH MANUAL LoRA
# ======================================================================

def attach_manual_lora(backbone, r: int = LORA_R, alpha: float = LORA_ALPHA):
    layers = backbone.encoder.layers
    if len(layers) != EXPECTED_WAVLM_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_WAVLM_LAYERS} WavLM layers, found {len(layers)}."
        )

    target_names = []
    for layer_idx, layer in enumerate(layers):
        attention = layer.attention

        q_proj = attention.q_proj
        v_proj = attention.v_proj

        q_proj.weight.requires_grad = False
        if q_proj.bias is not None:
            q_proj.bias.requires_grad = False

        v_proj.weight.requires_grad = False
        if v_proj.bias is not None:
            v_proj.bias.requires_grad = False

        q_lora = ManualLoRA(
            in_features=q_proj.in_features,
            out_features=q_proj.out_features,
            r=r,
            alpha=alpha,
        ).to(
            device=q_proj.weight.device,
            dtype=q_proj.weight.dtype,
        )

        v_lora = ManualLoRA(
            in_features=v_proj.in_features,
            out_features=v_proj.out_features,
            r=r,
            alpha=alpha,
        ).to(
            device=v_proj.weight.device,
            dtype=v_proj.weight.dtype,
        )

        attention.add_module("manual_lora_q", q_lora)
        attention.add_module("manual_lora_v", v_lora)

        target_names.extend(
            [
                f"encoder.layers.{layer_idx}.attention.q_proj",
                f"encoder.layers.{layer_idx}.attention.v_proj",
            ]
        )

    print()
    print("=" * 70)
    print("MANUAL LoRA ATTACHMENT")
    print("=" * 70)
    print("Encoder layers:     ", len(layers))
    print("LoRA target modules:", len(target_names))
    print("Rank / Alpha:       ", f"{LORA_R} / {LORA_ALPHA}")
    return target_names


# ======================================================================
# PATCHED WAVLM ATTENTION
# ======================================================================

def manual_lora_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.Tensor,
    gated_position_bias: torch.FloatTensor,
):
    query = key = value = hidden_states.transpose(0, 1)
    key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None

    bias_k = None
    bias_v = None
    add_zero_attn = False

    q_weight = self.q_proj.weight + self.manual_lora_q.delta_weight()
    k_weight = self.k_proj.weight
    v_weight = self.v_proj.weight + self.manual_lora_v.delta_weight()

    attn_output, attn_weights = F.multi_head_attention_forward(
        query,
        key,
        value,
        self.embed_dim,
        self.num_heads,
        torch.empty([0], device=hidden_states.device, dtype=hidden_states.dtype),
        torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
        bias_k,
        bias_v,
        add_zero_attn,
        self.dropout,
        self.out_proj.weight,
        self.out_proj.bias,
        self.training,
        key_padding_mask,
        need_weights=False,
        attn_mask=gated_position_bias,
        use_separate_proj_weight=True,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
    )

    attn_output = attn_output.transpose(0, 1)
    if attn_weights is not None:
        attn_weights = attn_weights[:, None].broadcast_to(
            attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
        )
    return attn_output, attn_weights


def patch_wavlm_attention(backbone) -> None:
    patched = 0
    for layer in backbone.encoder.layers:
        attention = layer.attention
        if not hasattr(attention, "manual_lora_q"):
            raise RuntimeError(
                "Manual LoRA was not attached before attention patch."
            )
        attention.torch_multi_head_self_attention = types.MethodType(
            manual_lora_attention_forward,
            attention,
        )
        patched += 1
    print("WavLM attention layers patched:", patched)


def freeze_backbone(backbone) -> None:
    for param in backbone.parameters():
        param.requires_grad = False
    print("All WavLM base parameters frozen.")


# ======================================================================
# STATE DICT HELPERS (LIGHTWEIGHT CHECKPOINTS)
# ======================================================================

def extract_lora_state_dict(backbone) -> dict[str, torch.Tensor]:
    """Extract only the manual LoRA parameters (~3 MB instead of 1.27 GB)."""
    state_dict = {}
    for name, param in backbone.named_parameters():
        if "manual_lora_q" in name or "manual_lora_v" in name:
            state_dict[name] = param.data.clone().cpu()
    return state_dict


def load_lora_state_dict(backbone, lora_dict: dict[str, torch.Tensor]) -> None:
    """Load manual LoRA parameters back into the backbone."""
    backbone_params = dict(backbone.named_parameters())
    for name, tensor in lora_dict.items():
        if name in backbone_params:
            backbone_params[name].data.copy_(tensor.to(backbone_params[name].device))
        else:
            raise KeyError(f"Parameter {name} not found in backbone.")


# ======================================================================
# FRAME MASK & FORWARD
# ======================================================================

def make_frame_valid_mask(
    backbone,
    attention_mask,
    max_frames,
):
    lengths = attention_mask.long().sum(dim=1)
    frame_lengths = (
        backbone._get_feat_extract_output_lengths(lengths).to(
            attention_mask.device
        )
    )
    frame_idx = torch.arange(
        max_frames,
        device=attention_mask.device,
    ).unsqueeze(0)
    return frame_idx < frame_lengths.unsqueeze(1)


def forward_batch(backbone, quality_head, batch, normalize_wav: bool = True):
    wav = batch["wav"].to(DEVICE, dtype=torch.float32)
    attention_mask = batch["attention_mask"].to(DEVICE)

    if normalize_wav:
        wav = normalize_waveform(wav, attention_mask)

    output = backbone(
        input_values=wav,
        attention_mask=attention_mask.long(),
    )
    hidden = output.last_hidden_state
    valid_mask = make_frame_valid_mask(
        backbone,
        attention_mask,
        hidden.shape[1],
    )
    pred = quality_head(hidden, valid_mask)
    return pred


# ======================================================================
# EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate(
    backbone,
    quality_head,
    loader,
    normalize_wav: bool = True,
    max_batches: int | None = None,
):
    backbone.eval()
    quality_head.eval()

    predictions = []
    targets = []

    use_amp = USE_AMP and DEVICE.type == "cuda"
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = forward_batch(backbone, quality_head, batch, normalize_wav=normalize_wav)
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


# ======================================================================
# TRAINING ONE EPOCH WITH MICRO-BATCH PREDICTION ACCUMULATION
# ======================================================================

def train_one_epoch(
    backbone,
    quality_head,
    loader,
    optimizer,
    scaler,
    epoch,
    global_step,
    scheduler=None,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    w_mse: float = W_MSE,
    w_ccc: float = W_CCC,
    w_pcc: float = W_PCC,
    w_rank: float = W_RANK,
    rank_margin: float = RANK_MARGIN,
    normalize_wav: bool = NORMALIZE_WAV,
    max_batches: int | None = None,
):
    backbone.train()
    quality_head.train()

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    accum_preds = []
    accum_targets = []
    num_updates = 0

    use_amp = USE_AMP and DEVICE.type == "cuda"
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = forward_batch(backbone, quality_head, batch, normalize_wav=normalize_wav)
            target = batch["target"].to(DEVICE)

            accum_preds.append(pred)
            accum_targets.append(target)

        if batch_idx % grad_accum_steps == 0 or batch_idx == total_batches:
            preds_chunk = torch.cat(accum_preds)
            targets_chunk = torch.cat(accum_targets)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                # Base loss: w_mse MSE + w_ccc (1-CCC) + w_pcc (1-PCC)
                base_loss = quality_loss(
                    preds_chunk.float(),
                    targets_chunk.float(),
                    w_mse=w_mse,
                    w_ccc=w_ccc,
                    w_pcc=w_pcc,
                )

                if w_rank > 0.0:
                    rank_loss = pairwise_ranking_loss(
                        preds_chunk.float(),
                        targets_chunk.float(),
                        margin=rank_margin,
                    )
                    loss = base_loss + w_rank * rank_loss
                else:
                    rank_loss = preds_chunk.new_zeros(())
                    loss = base_loss

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

            print_interval = max(1, PRINT_EVERY // grad_accum_steps)
            if (
                num_updates == 1
                or num_updates % print_interval == 0
                or batch_idx == total_batches
            ):
                print(
                    f"Epoch {epoch:02d} "
                    f"| batch {batch_idx:04d}/{total_batches:04d} "
                    f"| step {global_step:04d} "
                    f"| loss={loss.item():.4f} "
                    f"(base={base_loss.item():.4f}, rank={rank_loss.item():.4f}) "
                    f"| lr={optimizer.param_groups[0]['lr']:.2e}"
                )

    return running_loss / max(num_updates, 1), global_step


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="WavLM LoRA Full Data Training")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per GPU forward pass")
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader worker processes")
    parser.add_argument("--lora-r", type=int, default=LORA_R, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA, help="LoRA alpha scaling factor")
    parser.add_argument("--lr-lora", type=float, default=LR_LORA, help="Learning rate for LoRA weights")
    parser.add_argument("--lr-head", type=float, default=LR_HEAD, help="Learning rate for Quality Head")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--csv-path", type=str, default=str(CSV_PATH), help="Path to CSV dataset (if auto-splitting)")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Validation ratio (if auto-splitting)")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to explicit train CSV (e.g. data/qualispeech/train_val.csv)")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to explicit val CSV (e.g. data/qualispeech/test.csv)")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to explicit test CSV (e.g. data/qualispeech/test.csv)")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to save best checkpoint")
    parser.add_argument("--w-mse", type=float, default=W_MSE, help="Weight for MSE loss")
    parser.add_argument("--w-ccc", type=float, default=W_CCC, help="Weight for (1-CCC) loss")
    parser.add_argument("--w-pcc", type=float, default=W_PCC, help="Weight for (1-PCC) loss")
    parser.add_argument("--w-rank", type=float, default=W_RANK, help="Weight for Pairwise Ranking Loss")
    parser.add_argument("--rank-margin", type=float, default=RANK_MARGIN, help="Margin for Pairwise Ranking Loss")
    parser.add_argument("--normalize-wav", action=argparse.BooleanOptionalAction, default=NORMALIZE_WAV, help="Normalize waveform (zero-mean unit-variance on active speech)")
    parser.add_argument("--save-preds", action="store_true", help="Save predictions CSV on val/dev set")
    parser.add_argument("--save-history", type=str, default=None, help="JSON path to save training history")
    parser.add_argument("--quality-head-type", type=str, default="standard", choices=["standard", "utmos_nisqa"], help="Quality head type: standard or utmos_nisqa")
    parser.add_argument("--bilstm-hidden", type=int, default=128, help="BiLSTM hidden units for utmos_nisqa quality head")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Max train batches per epoch (for smoke testing)")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Max val batches per evaluation (for smoke testing)")
    args = parser.parse_args()

    set_seed(args.seed)

    effective_bs = args.batch_size * args.grad_accum

    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    elif args.train_csv and "train_val" in str(args.train_csv):
        checkpoint_path = CHECKPOINT_DIR / "wavlm_manual_lora_trainval_best.pt"
    else:
        checkpoint_path = BEST_CHECKPOINT
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WAVLM-LARGE + MANUAL LoRA — FULL DATA TRAINING")
    print("=" * 70)
    print("Device:                 ", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:                    ", torch.cuda.get_device_name(0))
    print("Backbone:               ", MODEL_NAME)
    print("Batch size:             ", args.batch_size)
    print("Gradient accumulation:  ", args.grad_accum)
    print("Effective batch size:   ", effective_bs)
    print("Epochs:                 ", args.epochs)
    print("Num workers:            ", args.num_workers)
    print("Validation ratio:       ", VAL_RATIO)
    print("LoRA rank / alpha:      ", f"{args.lora_r} / {args.lora_alpha}")
    print("LoRA LR / Head LR:      ", f"{args.lr_lora} / {args.lr_head}")
    print("Weight decay:           ", WEIGHT_DECAY)
    print("Head dropout:           ", DROPOUT_HEAD)
    print("Normalize waveform:     ", args.normalize_wav)
    print(f"Loss weights:            MSE={args.w_mse}, CCC={args.w_ccc}, PCC={args.w_pcc}, Rank={args.w_rank} (margin={args.rank_margin})")
    print("Checkpoint output:      ", checkpoint_path)
    print("AMP:                    ", USE_AMP)
    print("Gradient checkpointing: ", USE_GRADIENT_CHECKPOINTING)

    # ------------------------------------------------------------------
    # DATA PREPARATION (SOURCE-BALANCED)
    # ------------------------------------------------------------------
    test_loader = None
    if args.train_csv and args.val_csv:
        train_path = Path(args.train_csv)
        val_path = Path(args.val_csv)
        print(f"\nUsing explicit splits:\n  Train: {train_path}\n  Val:   {val_path}")

        train_dataset = QualiSpeechDataset(csv_path=train_path, wav_dir=WAV_DIR)
        val_dataset = QualiSpeechDataset(csv_path=val_path, wav_dir=WAV_DIR)

        if args.test_csv:
            test_path = Path(args.test_csv)
            print(f"  Test:  {test_path}")
            test_dataset = QualiSpeechDataset(csv_path=test_path, wav_dir=WAV_DIR)
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(DEVICE.type == "cuda"),
                collate_fn=qualispeech_collate,
            )
    else:
        csv_path = Path(args.csv_path)
        df = pd.read_csv(csv_path)
        df["sample_id"] = (
            df["id"].astype(str).str.strip().str.replace(r"\.wav$", "", regex=True)
        )
        df["target"] = pd.to_numeric(df["Overall quality"], errors="raise").astype(float)
        df["group"] = df["sample_id"].map(get_group)
        df["source"] = df["group"].map(get_source)

        train_df, val_df = make_source_balanced_split(df, val_ratio=args.val_ratio)
        if csv_path.name == "train.csv" and (len(train_df) != 8446 or len(val_df) != 2112):
            print(f"Warning: split count is {len(train_df)} / {len(val_df)} (expected 8446 / 2112).")
        else:
            print(f"Dataset: {csv_path.name} | Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")

        full_dataset = QualiSpeechDataset(
            csv_path=csv_path,
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
        batch_size=args.batch_size,
        shuffle=True,
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

    print()
    print("SPLIT SUMMARY")
    print(f"Train samples: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"Val samples:   {len(val_dataset)}, Batches: {len(val_loader)}")
    if test_loader is not None:
        print(f"Test samples:  {len(test_loader.dataset)}, Batches: {len(test_loader)}")

    # ------------------------------------------------------------------
    # LOAD WAVLM & ATTACH MANUAL LoRA
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("LOAD WAVLM-LARGE")
    print("=" * 70)

    backbone = WavLMModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE)
    freeze_backbone(backbone)
    disable_layerdrop(backbone)

    target_names = attach_manual_lora(backbone, r=args.lora_r, alpha=args.lora_alpha)
    patch_wavlm_attention(backbone)

    if USE_GRADIENT_CHECKPOINTING:
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
            print("Gradient checkpointing: enabled")

    # ------------------------------------------------------------------
    # QUALITY HEAD & OPTIMIZER
    # ------------------------------------------------------------------
    hidden_size = backbone.config.hidden_size
    if args.quality_head_type == "utmos_nisqa":
        quality_head = UtmosNisqaQualityHead(
            hidden_size=hidden_size,
            bilstm_hidden=args.bilstm_hidden,
            dropout=DROPOUT_HEAD,
            use_fault_features=False,
        ).to(DEVICE)
    else:
        quality_head = QualityHead(
            hidden_size=hidden_size,
            dropout=DROPOUT_HEAD,
        ).to(DEVICE)

    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in quality_head.parameters() if p.requires_grad]

    print(f"Quality Head Type:                {args.quality_head_type}")
    print(f"LoRA trainable parameters:        {sum(p.numel() for p in lora_params):,}")
    print(f"QualityHead trainable parameters: {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr_lora},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    effective_train_batches = len(train_loader) if args.max_train_batches is None else min(len(train_loader), args.max_train_batches)
    total_steps = (effective_train_batches // args.grad_accum + 1) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and DEVICE.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # INITIAL VALIDATION
    # ------------------------------------------------------------------
    print()
    print("INITIAL VALIDATION (EPOCH 0)")
    initial = evaluate(
        backbone,
        quality_head,
        val_loader,
        normalize_wav=args.normalize_wav,
        max_batches=args.max_val_batches,
    )
    print(
        f"Init | MSE={initial['mse']:.6f} | "
        f"MAE={initial['mae']:.6f} | "
        f"CCC={initial['ccc']:.6f} | "
        f"PCC={initial['pcc']:.6f}"
    )

    best_val_pcc = -float("inf")
    best_epoch = -1
    global_step = 0
    history = []

    print()
    print("=" * 70)
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
            scheduler=scheduler,
            grad_accum_steps=args.grad_accum,
            w_mse=args.w_mse,
            w_ccc=args.w_ccc,
            w_pcc=args.w_pcc,
            w_rank=args.w_rank,
            rank_margin=args.rank_margin,
            normalize_wav=args.normalize_wav,
            max_batches=args.max_train_batches,
        )

        val_result = evaluate(
            backbone,
            quality_head,
            val_loader,
            normalize_wav=args.normalize_wav,
            max_batches=args.max_val_batches,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_result["mse"],
            "val_mae": val_result["mae"],
            "val_ccc": val_result["ccc"],
            "val_pcc": val_result["pcc"],
        }
        history.append(epoch_record)

        print(
            f"\nEpoch {epoch:02d} Complete | "
            f"train_loss={train_loss:.6f} | "
            f"VAL MSE={val_result['mse']:.6f} | "
            f"MAE={val_result['mae']:.6f} | "
            f"CCC={val_result['ccc']:.6f} | "
            f"PCC={val_result['pcc']:.6f}"
        )

        # Checkpoint selection on PCC
        if val_result["pcc"] > best_val_pcc:
            best_val_pcc = val_result["pcc"]
            best_epoch = epoch

            # Save lightweight checkpoint (~4 MB instead of 1.27 GB)
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_name": MODEL_NAME,
                    "hidden_size": hidden_size,
                    "quality_head_type": args.quality_head_type,
                    "quality_head_kwargs": {
                        "bilstm_hidden": args.bilstm_hidden,
                        "dropout": DROPOUT_HEAD,
                        "use_fault_features": False,
                    } if args.quality_head_type == "utmos_nisqa" else {},
                    "normalize_wav": args.normalize_wav,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "w_mse": args.w_mse,
                    "w_ccc": args.w_ccc,
                    "w_pcc": args.w_pcc,
                    "w_rank": args.w_rank,
                    "rank_margin": args.rank_margin,
                    "lora_state_dict": extract_lora_state_dict(backbone),
                    "quality_head_state_dict": quality_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mse": val_result["mse"],
                    "val_mae": val_result["mae"],
                    "val_ccc": val_result["ccc"],
                    "val_pcc": val_result["pcc"],
                },
                checkpoint_path,
            )
            print(f"  -> NEW BEST CHECKPOINT (epoch={epoch}, VAL PCC={best_val_pcc:.6f})")

    # ------------------------------------------------------------------
    # RESTORE BEST CHECKPOINT
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("RESTORE BEST CHECKPOINT")
    print("=" * 70)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    load_lora_state_dict(backbone, ckpt["lora_state_dict"])
    quality_head.load_state_dict(ckpt["quality_head_state_dict"])

    print("Best epoch:      ", ckpt["epoch"])
    print("Best global step:", ckpt["global_step"])
    print("Best val PCC:    ", f"{ckpt['val_pcc']:.6f}")
    print("Checkpoint:      ", checkpoint_path)

    final = evaluate(
        backbone,
        quality_head,
        val_loader,
        normalize_wav=args.normalize_wav,
        max_batches=args.max_val_batches,
    )
    print()
    print("=" * 70)
    print("FINAL RESULT — BEST VALIDATION CHECKPOINT")
    print("=" * 70)
    print(f"VAL MSE: {final['mse']:.8f}")
    print(f"VAL MAE: {final['mae']:.8f}")
    print(f"VAL CCC: {final['ccc']:.8f}")
    print(f"VAL PCC: {final['pcc']:.8f}")

    if args.save_preds:
        try:
            if hasattr(val_loader.dataset, "df"):
                val_ids = val_loader.dataset.df["id"].astype(str).tolist()
            elif hasattr(val_loader.dataset, "dataset") and hasattr(val_loader.dataset, "indices"):
                val_ids = val_loader.dataset.dataset.df.iloc[val_loader.dataset.indices]["id"].astype(str).tolist()
            else:
                val_ids = [f"sample_{i}" for i in range(len(final["predictions"]))]

            df_preds = pd.DataFrame({
                "id": val_ids[:len(final["predictions"])],
                "target": final["targets"].numpy(),
                "predicted": final["predictions"].numpy(),
            })
            preds_csv = checkpoint_path.with_name(checkpoint_path.stem + "_dev_preds.csv")
            df_preds.to_csv(preds_csv, index=False)
            print(f"Saved dev predictions to: {preds_csv}")
        except Exception as e:
            print(f"Warning: could not save predictions CSV: {e}")

    history_path = Path(args.save_history) if args.save_history else checkpoint_path.with_name(checkpoint_path.stem + "_history.json")
    try:
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to: {history_path}")
    except Exception as e:
        print(f"Warning: could not save history JSON: {e}")

    if test_loader is not None:
        print()
        print("=" * 70)
        print("FINAL BLIND TEST EVALUATION")
        print("=" * 70)
        test_final = evaluate(
            backbone,
            quality_head,
            test_loader,
            normalize_wav=args.normalize_wav,
            max_batches=args.max_val_batches,
        )
        print(f"TEST MSE: {test_final['mse']:.8f}")
        print(f"TEST MAE: {test_final['mae']:.8f}")
        print(f"TEST CCC: {test_final['ccc']:.8f}")
        print(f"TEST PCC: {test_final['pcc']:.8f}")

    print()
    print("Training finished successfully.")


if __name__ == "__main__":
    main()

