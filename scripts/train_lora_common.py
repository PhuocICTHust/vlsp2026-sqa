from __future__ import annotations

import argparse
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
from transformers import HubertModel, WavLMModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.dataset import QualiSpeechDataset, qualispeech_collate
from sqa.model import (
    QualityHead,
    quality_loss,
    concordance_correlation_coefficient,
    pearson_correlation,
)

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
WAV_DIR = ROOT / "data" / "qualispeech" / "wav"

SEED = 42
VAL_RATIO = 0.20
EPOCHS = 10

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
NUM_WORKERS = 0
PIN_MEMORY = True

HEAD_LR = 5e-4
LORA_LR = 5e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

DROPOUT_HEAD = 0.10

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

USE_AMP = True
AMP_DTYPE = torch.float16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def prepare_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["sample_id"] = (
        df["id"].astype(str).str.strip().str.replace(
            r"\.wav$", "", regex=True
        )
    )
    df["target"] = pd.to_numeric(
        df["Overall quality"], errors="raise"
    ).astype(float)
    df["group"] = df["sample_id"].map(get_group)
    df["source"] = df["group"].map(get_source)
    return df


# ======================================================================
# WAVLM MANUAL LORA SUPPORT
# ======================================================================

class ManualLoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: float):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)


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
        need_weights=True,
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


def attach_wavlm_manual_lora(backbone):
    for param in backbone.parameters():
        param.requires_grad = False

    if hasattr(backbone.config, "layerdrop"):
        backbone.config.layerdrop = 0.0

    for layer in backbone.encoder.layers:
        attention = layer.attention
        q_proj = attention.q_proj
        v_proj = attention.v_proj

        q_lora = ManualLoRA(q_proj.in_features, q_proj.out_features, LORA_R, LORA_ALPHA).to(
            device=q_proj.weight.device, dtype=q_proj.weight.dtype
        )
        v_lora = ManualLoRA(v_proj.in_features, v_proj.out_features, LORA_R, LORA_ALPHA).to(
            device=v_proj.weight.device, dtype=v_proj.weight.dtype
        )
        attention.add_module("manual_lora_q", q_lora)
        attention.add_module("manual_lora_v", v_lora)
        attention.torch_multi_head_self_attention = types.MethodType(
            manual_lora_attention_forward, attention
        )


def get_base_model_for_mask(backbone):
    if hasattr(backbone, "base_model") and hasattr(backbone.base_model, "model"):
        return backbone.base_model.model
    return backbone


def frame_valid_mask(backbone, attention_mask: torch.Tensor) -> torch.Tensor:
    base = get_base_model_for_mask(backbone)
    lengths = attention_mask.long().sum(dim=1)
    frame_lengths = base._get_feat_extract_output_lengths(lengths).to(
        attention_mask.device
    )
    max_frames = int(
        base._get_feat_extract_output_lengths(
            torch.tensor([attention_mask.shape[1]], device=attention_mask.device)
        ).item()
    )
    frame_idx = torch.arange(max_frames, device=attention_mask.device).unsqueeze(0)
    return frame_idx < frame_lengths.unsqueeze(1)


def evaluate(backbone, quality_head, loader, device, amp_enabled):
    backbone.eval()
    quality_head.eval()

    all_pred = []
    all_target = []

    with torch.no_grad():
        for batch in loader:
            wav = batch["wav"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=AMP_DTYPE,
                enabled=amp_enabled,
            ):
                out = backbone(wav, attention_mask=attention_mask)
                hidden = out.last_hidden_state
                valid_mask = frame_valid_mask(backbone, attention_mask)
                pred = quality_head(hidden, valid_mask)

            all_pred.append(pred.float().cpu())
            all_target.append(target.float().cpu())

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)

    mse = torch.mean((pred - target).square()).item()
    mae = torch.mean((pred - target).abs()).item()
    pcc = pearson_correlation(pred, target).item()
    ccc_val = concordance_correlation_coefficient(pred, target).item()

    return {
        "mse": mse,
        "mae": mae,
        "ccc": ccc_val,
        "pcc": pcc,
    }


def build_backbone(backbone_name: str, device: torch.device):
    if backbone_name == "wavlm":
        model_name = "microsoft/wavlm-large"
        backbone = WavLMModel.from_pretrained(model_name).to(device)
        expected_hidden = 1024
        attach_wavlm_manual_lora(backbone)
    elif backbone_name == "hubert":
        model_name = "facebook/hubert-base-ls960"
        backbone = HubertModel.from_pretrained(model_name).to(device)
        expected_hidden = 768
        if hasattr(backbone.config, "layerdrop"):
            backbone.config.layerdrop = 0.0

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        backbone = get_peft_model(backbone, lora_config)
    else:
        raise ValueError(backbone_name)

    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()

    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    backbone.train()
    return backbone, model_name, expected_hidden


def count_params(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def save_checkpoint(path, backbone, quality_head, optimizer, scaler, epoch, metrics, args):
    if args.backbone == "hubert":
        lora_state = get_peft_model_state_dict(backbone)
    else:
        lora_state = {
            n: p.data.clone().cpu()
            for n, p in backbone.named_parameters()
            if "manual_lora_q" in n or "manual_lora_v" in n
        }

    checkpoint = {
        "lora_state_dict": lora_state,
        "quality_head_state_dict": quality_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "backbone": args.backbone,
        "model_name": args.model_name,
        "hidden_size": args.hidden_size,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "seed": SEED,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, backbone, quality_head, device, backbone_name):
    ckpt = torch.load(path, map_location=device)
    if backbone_name == "hubert":
        set_peft_model_state_dict(backbone, ckpt["lora_state_dict"])
    else:
        backbone_params = dict(backbone.named_parameters())
        for n, t in ckpt["lora_state_dict"].items():
            if n in backbone_params:
                backbone_params[n].data.copy_(t.to(backbone_params[n].device))
    quality_head.load_state_dict(ckpt["quality_head_state_dict"])
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        choices=["wavlm", "hubert"],
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=GRAD_ACCUM_STEPS,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = USE_AMP and device.type == "cuda"

    if args.output_dir is None:
        out_dir = ROOT / "outputs" / f"{args.backbone}_lora"
    else:
        out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LoRA RAW-AUDIO QUALITY TRAINING (COMMON PIPELINE)")
    print("=" * 70)
    print(f"Backbone:         {args.backbone}")
    print(f"Device:           {device}")
    if device.type == "cuda":
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Grad accum:       {args.grad_accum}")
    print(f"Effective batch:  {args.batch_size * args.grad_accum}")
    print(f"AMP:              {amp_enabled}")
    print(f"LoRA LR / Head LR:{LORA_LR} / {HEAD_LR}")

    df = prepare_dataframe()
    train_df, val_df = make_source_balanced_split(df)

    dataset = QualiSpeechDataset(
        csv_path=CSV_PATH,
        wav_dir=WAV_DIR,
        indices=list(range(len(df))),
    )

    id_to_index = {
        str(file_id): i
        for i, file_id in enumerate(dataset.df["id"].astype(str))
    }
    train_indices = [id_to_index[x] for x in train_df["id"].astype(str)]
    val_indices = [id_to_index[x] for x in val_df["id"].astype(str)]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        collate_fn=qualispeech_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        collate_fn=qualispeech_collate,
    )

    backbone, model_name, hidden_size = build_backbone(args.backbone, device)
    args.model_name = model_name
    args.hidden_size = hidden_size

    quality_head = QualityHead(hidden_size=hidden_size, dropout=DROPOUT_HEAD).to(device)

    lora_trainable, lora_total = count_params(backbone)
    head_trainable, head_total = count_params(quality_head)

    print()
    print("MODEL AUDIT")
    print(f"Model name:       {model_name}")
    print(f"Hidden size:      {hidden_size}")
    print(f"LoRA trainable:   {lora_trainable:,}")
    print(f"Head trainable:   {head_trainable:,}")
    print(f"TOTAL TRAINABLE:  {lora_trainable + head_trainable:,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LORA_LR},
            {"params": quality_head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (len(train_loader) // args.grad_accum + 1) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print()
    print("INITIAL VALIDATION")
    initial = evaluate(backbone, quality_head, val_loader, device, amp_enabled)
    print(
        f"Init | MSE={initial['mse']:.6f} | "
        f"MAE={initial['mae']:.6f} | "
        f"CCC={initial['ccc']:.6f} | "
        f"PCC={initial['pcc']:.6f}"
    )

    best_pcc = -math.inf
    best_epoch = -1
    history = []
    best_path = out_dir / "best_lora.pt"

    print()
    print("=" * 70)
    print("TRAINING LOOP")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        quality_head.train()

        running_loss = 0.0
        num_updates = 0
        optimizer.zero_grad(set_to_none=True)

        accum_preds = []
        accum_targets = []

        for step, batch in enumerate(train_loader, start=1):
            wav = batch["wav"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=AMP_DTYPE,
                enabled=amp_enabled,
            ):
                out = backbone(wav, attention_mask=attention_mask)
                hidden = out.last_hidden_state
                valid_mask = frame_valid_mask(backbone, attention_mask)
                pred = quality_head(hidden, valid_mask)

                accum_preds.append(pred)
                accum_targets.append(target)

            if step % args.grad_accum == 0 or step == len(train_loader):
                preds_chunk = torch.cat(accum_preds)
                targets_chunk = torch.cat(accum_targets)

                with torch.autocast(
                    device_type=device.type,
                    dtype=AMP_DTYPE,
                    enabled=amp_enabled,
                ):
                    loss = quality_loss(
                        preds_chunk.float(),
                        targets_chunk.float(),
                        w_mse=0.4,
                        w_ccc=0.3,
                        w_pcc=0.3,
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}")

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in list(backbone.parameters()) + list(quality_head.parameters()) if p.requires_grad],
                    MAX_GRAD_NORM,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()

                running_loss += loss.item()
                num_updates += 1
                accum_preds = []
                accum_targets = []

        train_loss = running_loss / max(num_updates, 1)
        val_metrics = evaluate(backbone, quality_head, val_loader, device, amp_enabled)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_MSE={val_metrics['mse']:.6f} | "
            f"val_CCC={val_metrics['ccc']:.6f} | "
            f"val_PCC={val_metrics['pcc']:.6f}"
        )

        if val_metrics["pcc"] > best_pcc:
            best_pcc = val_metrics["pcc"]
            best_epoch = epoch
            save_checkpoint(
                best_path,
                backbone,
                quality_head,
                optimizer,
                scaler,
                epoch,
                val_metrics,
                args,
            )
            print(f"  -> NEW BEST checkpoint (epoch {epoch}, PCC={best_pcc:.6f})")

        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    print()
    print("=" * 70)
    print("LOAD BEST CHECKPOINT")
    print("=" * 70)

    load_checkpoint(best_path, backbone, quality_head, device, args.backbone)
    final = evaluate(backbone, quality_head, val_loader, device, amp_enabled)

    print(f"Best epoch:   {best_epoch}")
    print(f"Best val PCC: {final['pcc']:.8f}")
    print(f"Best val MSE: {final['mse']:.8f}")
    print(f"Best val CCC: {final['ccc']:.8f}")
    print(f"Checkpoint:   {best_path}")


if __name__ == "__main__":
    main()
