from __future__ import annotations

import argparse
import json
import math
import random
import platform
import shutil
import sys
import time
import types
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import AutoFeatureExtractor, HubertModel, Wav2Vec2Model, WavLMModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.dataset import QualiSpeechDataset, qualispeech_collate
from sqa.hub import resolve_model
try:  # optional helper (only needed where an old torchao breaks new peft, e.g. Kaggle)
    from sqa.peft_compat import disable_incompatible_torchao
    disable_incompatible_torchao()
except ImportError:
    pass
from sqa.model import (
    QualityHead,
    UtmosNisqaQualityHead,
    pairwise_ranking_loss,
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

# Backbones whose attention calls q_proj(x)/v_proj(x) as nn.Linear modules,
# so PEFT LoRA can wrap them directly. WavLM is NOT in this set: its attention
# reads q_proj.weight inside F.multi_head_attention_forward, so it needs the
# manual LoRA patch below.
PEFT_BACKBONES = {"hubert", "wav2vec2"}

DEFAULT_MODEL_NAMES = {
    "wavlm": "microsoft/wavlm-large",
    "hubert": "facebook/hubert-base-ls960",
    # Same size (95M, 12 layers, 768-d) and pre-training data (LibriSpeech 960 h)
    # as hubert-base-ls960 -> clean backbone comparison. For a multilingual
    "wav2vec2": "facebook/wav2vec2-base",
}

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


def report_subset(file_id: str) -> str:
    """Fine-grained subset name for per-subset metrics (not used for splitting)."""
    group = str(file_id).split("-", 1)[0].lower()
    if "giga" in group:
        return "gigaspeech"
    if group.startswith("sysnis"):
        return {
            "sysnists": "nisqa_train_sim", "sysnistl": "nisqa_train_live",
            "sysnisvs": "nisqa_val_sim", "sysnisvl": "nisqa_val_live",
            "sysnistf": "nisqa_test_for", "sysnistp": "nisqa_test_p501",
        }.get(group, group)
    if group.startswith("sys") and group.endswith("tts") and group[3:-3].isdigit():
        return "recent_tts"
    return "bvcc"


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


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


# WAVLM MANUAL LORA SUPPORT

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


def attach_wavlm_manual_lora(backbone, r: int = LORA_R, alpha: float = LORA_ALPHA):
    for param in backbone.parameters():
        param.requires_grad = False

    if hasattr(backbone.config, "layerdrop"):
        backbone.config.layerdrop = 0.0

    for layer in backbone.encoder.layers:
        attention = layer.attention
        q_proj = attention.q_proj
        v_proj = attention.v_proj

        q_lora = ManualLoRA(q_proj.in_features, q_proj.out_features, r, alpha).to(
            device=q_proj.weight.device, dtype=q_proj.weight.dtype
        )
        v_lora = ManualLoRA(v_proj.in_features, v_proj.out_features, r, alpha).to(
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


def prepare_inputs(wav: torch.Tensor, attention_mask: torch.Tensor, input_cfg: dict):
    """Apply the backbone's expected input preprocessing.

    - do_normalize: per-utterance zero-mean / unit-variance over the valid
      (non-padded) samples, padding kept at 0 (same as HF
      Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm).
    - pass_mask: models with feat_extract_norm == "group" (e.g. wav2vec2-base)
      were pre-trained without an attention mask; HF recommends zero-padding
      and NOT passing the mask. The frame mask for pooling is still used.
    """
    if input_cfg.get("do_normalize", False):
        m = attention_mask.to(wav.dtype)
        n = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (wav * m).sum(dim=1, keepdim=True) / n
        var = (((wav - mean) * m) ** 2).sum(dim=1, keepdim=True) / n
        wav = (wav - mean) / torch.sqrt(var + 1e-7) * m
    mask_arg = attention_mask.long() if input_cfg.get("pass_mask", True) else None
    return wav, mask_arg


def evaluate(backbone, quality_head, loader, device, amp_enabled, input_cfg=None, return_predictions=False,
             desc=None, log_every=500):
    input_cfg = input_cfg or {}
    all_ids = []
    t0 = time.time()
    if desc:
        print(f"[{desc}] evaluating {len(loader.dataset)} files ...", flush=True)
    backbone.eval()
    quality_head.eval()

    all_pred = []
    all_target = []

    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            if desc and i % log_every == 0:
                print(f"[{desc}] {i}/{len(loader)} batches | {time.time() - t0:.0f}s", flush=True)
            wav = batch["wav"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=AMP_DTYPE,
                enabled=amp_enabled,
            ):
                model_wav, model_mask = prepare_inputs(wav, attention_mask, input_cfg)
                out = backbone(model_wav, attention_mask=model_mask)
                hidden = out.last_hidden_state
                valid_mask = frame_valid_mask(backbone, attention_mask)
                pred = quality_head(hidden, valid_mask)

            all_pred.append(pred.float().cpu())
            all_target.append(target.float().cpu())
            all_ids.extend(batch["file_id"])

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)

    mse = torch.mean((pred - target).square()).item()
    mae = torch.mean((pred - target).abs()).item()
    pcc = pearson_correlation(pred, target).item()
    ccc_val = concordance_correlation_coefficient(pred, target).item()

    if desc:
        print(f"[{desc}] done in {time.time() - t0:.0f}s", flush=True)
    metrics = {
        "mse": mse,
        "mae": mae,
        "ccc": ccc_val,
        "pcc": pcc,
    }
    if return_predictions:
        preds = pd.DataFrame({
            "file_id": all_ids,
            "pred": pred.numpy(),
            "target": target.numpy(),
        })
        return metrics, preds
    return metrics


def subset_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    preds = preds.assign(subset=preds["file_id"].map(report_subset))
    for name, g in [("ALL", preds), *preds.groupby("subset")]:
        pcc = float(np.corrcoef(g["pred"], g["target"])[0, 1]) if len(g) > 2 and g["target"].std() > 0 else float("nan")
        rows.append({
            "subset": name, "n": len(g), "pcc": pcc,
            "mse": float(((g["pred"] - g["target"]) ** 2).mean()),
            "mae": float((g["pred"] - g["target"]).abs().mean()),
            "pred_mean": float(g["pred"].mean()), "pred_std": float(g["pred"].std()),
            "target_mean": float(g["target"].mean()), "target_std": float(g["target"].std()),
        })
    return pd.DataFrame(rows)


def _wav2vec2_input_cfg(model_name: str, config) -> dict:
    """Read the checkpoint's own preprocessing; fall back to config if offline."""
    try:
        fe = AutoFeatureExtractor.from_pretrained(resolve_model(model_name))
        return {
            "do_normalize": bool(getattr(fe, "do_normalize", True)),
            "pass_mask": bool(getattr(fe, "return_attention_mask", False)),
        }
    except Exception as exc:  # no preprocessor_config.json / offline
        print(f"[warn] feature extractor not loaded ({exc}); using config fallback")
        return {
            "do_normalize": True,
            "pass_mask": getattr(config, "feat_extract_norm", "group") == "layer",
        }


def build_backbone(
    backbone_name: str,
    device: torch.device,
    model_name: str | None = None,
    no_spec_augment: bool = False,
    lora_r: int = LORA_R,
    lora_alpha: float = LORA_ALPHA,
    gradient_checkpointing: bool = True,
):
    model_name = model_name or DEFAULT_MODEL_NAMES[backbone_name]
    # Kept identical to the previous WavLM/HuBERT runs (raw waveform + mask).
    input_cfg = {"do_normalize": False, "pass_mask": True}

    if backbone_name == "wavlm":
        backbone = WavLMModel.from_pretrained(resolve_model(model_name)).to(device)
        attach_wavlm_manual_lora(backbone, lora_r, lora_alpha)
    elif backbone_name in PEFT_BACKBONES:
        model_cls = HubertModel if backbone_name == "hubert" else Wav2Vec2Model
        backbone = model_cls.from_pretrained(resolve_model(model_name)).to(device)
        if hasattr(backbone.config, "layerdrop"):
            backbone.config.layerdrop = 0.0
        if backbone_name == "wav2vec2":
            input_cfg = _wav2vec2_input_cfg(model_name, backbone.config)
        if no_spec_augment:
            # SpecAugment-style time/feature masking is active in train() mode
            # whenever mask_time_prob > 0 (0.05 in most checkpoints).
            backbone.config.mask_time_prob = 0.0
            backbone.config.mask_feature_prob = 0.0

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
            # No task_type: the generic PeftModel passes input_values through unchanged.
            # (task_type="FEATURE_EXTRACTION" renames the first argument to input_ids, which wav2vec2/HuBERT reject.)
        )
        backbone = get_peft_model(backbone, lora_config)
    else:
        raise ValueError(backbone_name)

    if hasattr(backbone, "enable_input_require_grads"):
        try:
            backbone.enable_input_require_grads()
        except NotImplementedError:
            # Older transformers: speech models have no input embeddings.
            # Not needed here: the conv feature encoder sets requires_grad on its
            # input in train mode, and we use non-reentrant checkpointing.
            pass

    if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        try:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            backbone.gradient_checkpointing_enable()

    hidden_size = int(get_base_model_for_mask(backbone).config.hidden_size)
    backbone.train()
    return backbone, model_name, hidden_size, input_cfg


QUALITY_HEAD_TYPES = ("standard", "utmos_nisqa")


def build_quality_head(head_type: str, hidden_size: int, bilstm_hidden: int = 128):
    """standard: attention pool + MLP. utmos_nisqa: BiLSTM + attention mean/std pooling + MLP (src/sqa/model.py),
    without fault features, exactly as in scripts/train_wavlm_lora.py --quality-head-type utmos_nisqa."""
    if head_type == "utmos_nisqa":
        kwargs = {"bilstm_hidden": bilstm_hidden, "dropout": DROPOUT_HEAD, "use_fault_features": False}
        return UtmosNisqaQualityHead(hidden_size=hidden_size, **kwargs), kwargs
    if head_type == "standard":
        return QualityHead(hidden_size=hidden_size, dropout=DROPOUT_HEAD), {"dropout": DROPOUT_HEAD}
    raise ValueError(f"unknown quality head type {head_type!r}")


def read_csvs(paths) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def make_dataset(csv_paths, wav_dir):
    """One CSV -> QualiSpeechDataset; several (e.g. train.csv val.csv) -> concatenation."""
    parts = [QualiSpeechDataset(csv_path=p, wav_dir=wav_dir) for p in csv_paths]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def count_params(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def save_checkpoint(path, backbone, quality_head, optimizer, scaler, epoch, metrics, args, extra=None):
    if args.backbone in PEFT_BACKBONES:
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
        "input_cfg": args.input_cfg,
        "no_spec_augment": args.no_spec_augment,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quality_head_type": args.quality_head_type,
        "quality_head_kwargs": args.quality_head_kwargs,
        "loss": {"w_mse": args.w_mse, "w_ccc": args.w_ccc, "w_pcc": args.w_pcc,
                 "w_rank": args.w_rank, "rank_margin": args.rank_margin},
        "seed": SEED,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        **(extra or {}),
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)  # atomic: a crash during saving never leaves a half-written checkpoint


def load_checkpoint(path, backbone, quality_head, device, backbone_name):
    # Our own checkpoints; last.pt also holds RNG states (numpy arrays), so weights_only must be False.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if backbone_name in PEFT_BACKBONES:
        set_peft_model_state_dict(backbone, ckpt["lora_state_dict"])
    else:
        backbone_params = dict(backbone.named_parameters())
        for n, t in ckpt["lora_state_dict"].items():
            if n in backbone_params:
                backbone_params[n].data.copy_(t.to(backbone_params[n].device))
    quality_head.load_state_dict(ckpt["quality_head_state_dict"])
    return ckpt


class _Tee:
    """Write everything printed (stdout + stderr) to the console and to train.log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _pkg(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        choices=["wavlm", "hubert", "wav2vec2"],
        required=True,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HF checkpoint id; default per backbone (see DEFAULT_MODEL_NAMES).",
    )
    parser.add_argument(
        "--no-spec-augment",
        action="store_true",
        help="Disable HF time/feature masking during training (HuBERT/wav2vec2).",
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
    parser.add_argument("--train-csv", type=str, nargs="+", default=[str(CSV_PATH)],
                        help="Training CSV(s); several are concatenated (e.g. train.csv val.csv). "
                             "Without --val-csv a single CSV is split 80/20 internally.")
    parser.add_argument("--val-csv", type=str, default=None,
                        help="Official validation CSV: train on all of --train-csv, select the epoch on this.")
    parser.add_argument("--test-csv", type=str, default=None,
                        help="Evaluated once with the best checkpoint after training (never used for selection).")
    parser.add_argument("--wav-dir", type=str, default=str(WAV_DIR),
                        help="Flat folder with the .wav files named as the CSV 'id' column.")
    parser.add_argument("--quality-head-type", choices=QUALITY_HEAD_TYPES, default="standard",
                        help="standard = attention pool + MLP; utmos_nisqa = BiLSTM + mean/std pooling + MLP.")
    parser.add_argument("--bilstm-hidden", type=int, default=128, help="BiLSTM size of the utmos_nisqa head.")
    parser.add_argument("--w-mse", type=float, default=0.4)
    parser.add_argument("--w-ccc", type=float, default=0.3)
    parser.add_argument("--w-pcc", type=float, default=0.3)
    parser.add_argument("--w-rank", type=float, default=0.0,
                        help="Pairwise ranking loss weight inside each accumulated chunk (WavLM utmos run: 0.1).")
    parser.add_argument("--rank-margin", type=float, default=0.1)
    parser.add_argument("--select-on-test", action="store_true",
                        help="Allow --val-csv to be the test CSV (epoch chosen on test; test numbers become "
                             "optimistic). Needed for the train+val -> test-as-dev recipe.")
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--lr-lora", type=float, default=LORA_LR)
    parser.add_argument("--lr-head", type=float, default=HEAD_LR)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Faster on large-VRAM GPUs; results unchanged.")
    parser.add_argument("--log-every", type=int, default=500,
                        help="Print training progress every N batches.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the run in --output-dir (from last.pt, else best_lora.pt). With a larger "
                             "--epochs than before, the extra epochs use a fresh cosine schedule starting at "
                             "--lr-lora/--lr-head (pass smaller values for gentler fine-tuning).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an output dir that already has best_lora.pt.")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping: stop after N epochs without val PCC improvement (0 = off).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum val PCC increase that counts as an improvement.",
    )
    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = USE_AMP and device.type == "cuda"

    if args.output_dir is None:
        out_dir = ROOT / "outputs" / f"{args.backbone}_lora"
    else:
        out_dir = ROOT / args.output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.resume:
        if not ((out_dir / "last.pt").exists() or (out_dir / "best_lora.pt").exists()):
            sys.exit(f"ERROR: --resume but {out_dir} has neither last.pt nor best_lora.pt")
    elif (out_dir / "best_lora.pt").exists() and not args.overwrite:
        sys.exit(f"ERROR: {out_dir} already has best_lora.pt – use another --output-dir, --resume or --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "train.log", "a" if args.resume else "w", encoding="utf-8")
    if args.resume:
        log_file.write(f"\n{'#' * 70}\n# RESUME {stamp}\n{'#' * 70}\n")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    t_start = time.time()
    run_config = {
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "args": {k: v for k, v in vars(args).items()},
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "packages": {n: _pkg(n) for n in ["torch", "transformers", "peft", "torchao", "numpy", "pandas"]},
    }
    config_name = f"run_config_resume_{stamp}.json" if args.resume else "run_config.json"
    (out_dir / config_name).write_text(json.dumps(run_config, indent=2, default=str))

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
    print(f"LoRA r / alpha:   {args.lora_r} / {args.lora_alpha}")
    print(f"LoRA LR / Head LR:{args.lr_lora} / {args.lr_head}")
    print(f"Num workers:      {args.num_workers}")
    print(f"Grad ckpt:        {not args.no_gradient_checkpointing}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        collate_fn=qualispeech_collate,
        persistent_workers=args.num_workers > 0,
    )
    train_csvs = [resolve_path(p) for p in args.train_csv]
    train_csv = train_csvs[0]
    wav_dir = resolve_path(args.wav_dir)
    test_loader = None

    # leakage guard: no file may be in training and in the selection / test set
    train_ids = set(read_csvs(train_csvs)["id"].astype(str))
    for flag, path in [("--val-csv", args.val_csv), ("--test-csv", args.test_csv)]:
        if path:
            overlap = train_ids & set(pd.read_csv(resolve_path(path))["id"].astype(str))
            if overlap:
                sys.exit(f"ERROR: {len(overlap)} files of {flag} are also in --train-csv (e.g. {sorted(overlap)[:3]})")
    selection_on_test = bool(args.val_csv and args.test_csv
                             and resolve_path(args.val_csv).resolve() == resolve_path(args.test_csv).resolve())
    if selection_on_test and not args.select_on_test:
        sys.exit("ERROR: --val-csv is the same file as --test-csv, so the epoch would be chosen on test. "
                 "Pass --select-on-test if this is intended (test numbers are then optimistic).")
    if selection_on_test:
        print("WARNING: epoch selection on the TEST set – report test numbers as dev numbers, not held-out results.")

    if args.val_csv:
        # Official split: all training CSVs for training, val.csv for epoch selection.
        val_csv = resolve_path(args.val_csv)
        print(f"Split mode:       explicit (train={'+'.join(p.name for p in train_csvs)}, val={val_csv.name})")
        train_dataset = make_dataset(train_csvs, wav_dir)
        val_dataset = QualiSpeechDataset(csv_path=val_csv, wav_dir=wav_dir)
    else:
        if len(train_csvs) > 1:
            sys.exit("ERROR: several --train-csv files need --val-csv")
        print(f"Split mode:       internal 80/20 of {train_csv.name}")
        df = pd.read_csv(train_csv)
        df["sample_id"] = df["id"].astype(str).str.strip().str.replace(r"\.wav$", "", regex=True)
        df["target"] = pd.to_numeric(df["Overall quality"], errors="raise").astype(float)
        df["group"] = df["sample_id"].map(get_group)
        df["source"] = df["group"].map(get_source)
        train_df, val_df = make_source_balanced_split(df)

        dataset = QualiSpeechDataset(
            csv_path=train_csv,
            wav_dir=wav_dir,
            indices=list(range(len(df))),
        )
        id_to_index = {
            str(file_id): i
            for i, file_id in enumerate(dataset.df["id"].astype(str))
        }
        train_indices = [id_to_index[x] for x in train_df["id"].astype(str) if x in id_to_index]
        val_indices = [id_to_index[x] for x in val_df["id"].astype(str) if x in id_to_index]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

    if args.test_csv:
        test_csv = resolve_path(args.test_csv)
        test_dataset = QualiSpeechDataset(csv_path=test_csv, wav_dir=wav_dir)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print(f"Train / val / test files: {len(train_dataset)} / {len(val_dataset)} / "
          f"{len(test_loader.dataset) if test_loader is not None else 0}")

    print(f"Loading backbone {args.model_name or DEFAULT_MODEL_NAMES[args.backbone]} "
          "(first run downloads it from Hugging Face) ...", flush=True)
    t_load = time.time()
    backbone, model_name, hidden_size, input_cfg = build_backbone(
        args.backbone,
        device,
        model_name=args.model_name,
        no_spec_augment=args.no_spec_augment,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    args.model_name = model_name
    args.hidden_size = hidden_size
    args.input_cfg = input_cfg
    print(f"Backbone loaded in {time.time() - t_load:.0f}s", flush=True)
    if args.batch_size > 1 and not input_cfg.get("pass_mask", True):
        sys.exit(
            f"ERROR: {model_name} is used without an attention mask (feat_extract_norm=group), so a batch "
            f"of {args.batch_size} would pad files with zeros that leak into GroupNorm. "
            "Use --batch-size 1 and a larger --grad-accum."
        )

    quality_head, args.quality_head_kwargs = build_quality_head(args.quality_head_type, hidden_size, args.bilstm_hidden)
    quality_head = quality_head.to(device)

    lora_trainable, lora_total = count_params(backbone)
    head_trainable, head_total = count_params(quality_head)

    print()
    print("MODEL AUDIT")
    print(f"Model name:       {model_name}")
    print(f"Hidden size:      {hidden_size}")
    print(f"Input cfg:        {input_cfg}")
    print(f"SpecAugment off:  {args.no_spec_augment}")
    print(f"Quality head:     {args.quality_head_type} {args.quality_head_kwargs}")
    print(f"Loss:             {args.w_mse} MSE + {args.w_ccc} (1-CCC) + {args.w_pcc} (1-PCC)"
          f" + {args.w_rank} rank (margin {args.rank_margin})")
    print(f"LoRA trainable:   {lora_trainable:,}")
    print(f"Head trainable:   {head_trainable:,}")
    print(f"TOTAL TRAINABLE:  {lora_trainable + head_trainable:,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.lr_lora},
            {"params": quality_head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum + 1
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_pcc = -math.inf
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False
    history = []
    best_path = out_dir / "best_lora.pt"
    last_path = out_dir / "last.pt"
    start_epoch = 1
    resume_info = None

    if args.resume:
        source = last_path if last_path.exists() else best_path
        ckpt = torch.load(source, map_location=device, weights_only=False)
        for key in ("backbone", "model_name", "lora_r", "lora_alpha", "quality_head_type"):
            saved = ckpt.get(key, "standard" if key == "quality_head_type" else None)  # old runs: standard head
            if saved is not None and saved != getattr(args, key):
                sys.exit(f"ERROR: cannot resume – checkpoint {key}={saved!r} but this run has {getattr(args, key)!r}")
        load_checkpoint(source, backbone, quality_head, device, args.backbone)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        if source == last_path:
            history = ckpt["history"]
            best_pcc, best_epoch = ckpt["best_pcc"], ckpt["best_epoch"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            start_epoch = ckpt["epoch"] + 1
        else:
            print("[resume] no last.pt (run made before it was saved) – continuing from best_lora.pt, i.e. from "
                  f"the weights of the best epoch ({ckpt['epoch']}), not the last one.")
            hist_csv = out_dir / "training_history.csv"
            history = pd.read_csv(hist_csv).to_dict(orient="records") if hist_csv.exists() else []
            best_pcc, best_epoch = ckpt["metrics"]["pcc"], ckpt["epoch"]
            start_epoch = max(len(history), ckpt["epoch"]) + 1

        if start_epoch > args.epochs:
            sys.exit(f"ERROR: the run already has {start_epoch - 1} epochs; pass --epochs larger than that to train more")

        # keep the results of the previous phase for the paper; they are rewritten at the end
        archive = out_dir / f"previous_{stamp}"
        for name in ["final_metrics.json", "val_predictions.csv", "test_predictions.csv",
                     "training_history.csv", "run_config.json"]:
            if (out_dir / name).exists():
                archive.mkdir(exist_ok=True)
                shutil.copy2(out_dir / name, archive / name)

        exact = (source == last_path and ckpt.get("epochs_total") == args.epochs
                 and ckpt.get("scheduler_state_dict") is not None)
        if exact:
            # interrupted run with unchanged settings: continue the same LR schedule and RNG streams
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            rng = ckpt.get("rng") or {}
            if rng:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
            mode = "exact continuation of the same schedule"
        else:
            # extension: fresh cosine over the extra epochs, starting at the given learning rates
            for group, lr in zip(optimizer.param_groups, [args.lr_lora, args.lr_head]):
                group["lr"] = lr
                group["initial_lr"] = lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps_per_epoch * (args.epochs - start_epoch + 1), eta_min=1e-6
            )
            epochs_no_improve = 0
            mode = (f"extension: new cosine schedule over epochs {start_epoch}-{args.epochs}, "
                    f"LR {args.lr_lora} / {args.lr_head}")
        resume_info = {"from": str(source), "from_epoch": start_epoch - 1, "best_pcc_before": best_pcc,
                       "best_epoch_before": best_epoch, "mode": mode, "utc": stamp}
        print()
        print("RESUME")
        print(f"From:             {source.name} (epoch {start_epoch - 1}, best PCC {best_pcc:.6f} @ epoch {best_epoch})")
        print(f"Mode:             {mode}")
        print(f"Epochs:           {start_epoch} -> {args.epochs}")
    else:
        print()
        print("INITIAL VALIDATION")
        initial = evaluate(backbone, quality_head, val_loader, device, amp_enabled, input_cfg,
                           desc="val, before training", log_every=args.log_every)
        print(
            f"Init | MSE={initial['mse']:.6f} | "
            f"MAE={initial['mae']:.6f} | "
            f"CCC={initial['ccc']:.6f} | "
            f"PCC={initial['pcc']:.6f}"
        )

    print()
    print("=" * 70)
    print("TRAINING LOOP")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        backbone.train()
        quality_head.train()

        running_loss = 0.0
        num_updates = 0
        optimizer.zero_grad(set_to_none=True)

        accum_preds = []
        accum_targets = []
        t_epoch = time.time()
        recent_losses = []
        print(f"Epoch {epoch}/{args.epochs}: {len(train_loader)} batches", flush=True)

        for step, batch in enumerate(train_loader, start=1):
            wav = batch["wav"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=AMP_DTYPE,
                enabled=amp_enabled,
            ):
                model_wav, model_mask = prepare_inputs(wav, attention_mask, input_cfg)
                out = backbone(model_wav, attention_mask=model_mask)
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
                        w_mse=args.w_mse,
                        w_ccc=args.w_ccc,
                        w_pcc=args.w_pcc,
                    )
                    if args.w_rank > 0.0:
                        loss = loss + args.w_rank * pairwise_ranking_loss(
                            preds_chunk.float(), targets_chunk.float(), margin=args.rank_margin
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
                recent_losses.append(loss.item())
                num_updates += 1
                accum_preds = []
                accum_targets = []

            if step % args.log_every == 0:
                elapsed = time.time() - t_epoch
                rate = step / elapsed
                eta = (len(train_loader) - step) / rate
                avg = sum(recent_losses) / max(len(recent_losses), 1)
                print(f"  epoch {epoch} | {step}/{len(train_loader)} | loss {avg:.4f} | "
                      f"{rate * args.batch_size:.1f} files/s | epoch ETA {eta / 60:.1f} min", flush=True)
                recent_losses = []

        train_loss = running_loss / max(num_updates, 1)
        print(f"Epoch {epoch} training done in {(time.time() - t_epoch) / 60:.1f} min", flush=True)
        val_metrics = evaluate(backbone, quality_head, val_loader, device, amp_enabled, input_cfg, desc=f"val, epoch {epoch}", log_every=args.log_every)

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

        improved = val_metrics["pcc"] > best_pcc + args.min_delta
        if improved:
            epochs_no_improve = 0
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
        else:
            epochs_no_improve += 1
            print(f"  -> no improvement for {epochs_no_improve} epoch(s) (best PCC={best_pcc:.6f} @ epoch {best_epoch})")

        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        save_checkpoint(
            last_path, backbone, quality_head, optimizer, scaler, epoch, val_metrics, args,
            extra={
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_pcc": best_pcc,
                "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
                "epochs_total": args.epochs,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
        )

        if args.patience > 0 and epochs_no_improve >= args.patience:
            stopped_early = True
            print(f"EARLY STOPPING at epoch {epoch}: no val PCC improvement for {args.patience} epochs.")
            break

    print()
    print("=" * 70)
    print("LOAD BEST CHECKPOINT")
    print("=" * 70)

    load_checkpoint(best_path, backbone, quality_head, device, args.backbone)
    final, val_preds = evaluate(backbone, quality_head, val_loader, device, amp_enabled, input_cfg,
                                return_predictions=True)
    val_preds.to_csv(out_dir / "val_predictions.csv", index=False)
    results = {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "resume": resume_info,
        "quality_head_type": args.quality_head_type,
        "selection_on_test": selection_on_test,
        "val": final,
        "val_by_subset": subset_metrics(val_preds).to_dict(orient="records"),
    }
    if test_loader is not None:
        test_metrics, test_preds = evaluate(backbone, quality_head, test_loader, device, amp_enabled,
                                            input_cfg, return_predictions=True)
        test_preds.to_csv(out_dir / "test_predictions.csv", index=False)
        test_sub = subset_metrics(test_preds)
        results["test"] = test_metrics
        results["test_by_subset"] = test_sub.to_dict(orient="records")
        print()
        print("=" * 70)
        print("TEST EVALUATION (epoch was chosen on this set – optimistic)" if selection_on_test
              else "FINAL BLIND TEST EVALUATION (best checkpoint, evaluated once)")
        print("=" * 70)
        print(test_sub.to_string(index=False))
    results["wall_time_min"] = round((time.time() - t_start) / 60, 1)
    results["config"] = run_config
    (out_dir / "final_metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"Wall time:    {results['wall_time_min']} min")
    print(f"Results:      {out_dir / 'final_metrics.json'}")

    print(f"Best epoch:   {best_epoch}")
    print(f"Early stop:   {stopped_early} (patience={args.patience}, ran {len(history)}/{args.epochs} epochs)")
    print(f"Best val PCC: {final['pcc']:.8f}")
    print(f"Best val MSE: {final['mse']:.8f}")
    print(f"Best val CCC: {final['ccc']:.8f}")
    print(f"Checkpoint:   {best_path}")


if __name__ == "__main__":
    main()
