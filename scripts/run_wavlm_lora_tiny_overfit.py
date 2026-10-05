from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import WavLMModel


# ======================================================================
# PROJECT PATH
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sqa.dataset import (
    QualiSpeechDataset,
    qualispeech_collate,
)

from src.sqa.model import (
    QualityHead,
    quality_loss,
    concordance_correlation_coefficient,
)


# ======================================================================
# PATHS
# ======================================================================

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"

WAV_DIR = ROOT / "data" / "qualispeech" / "wav"

CHECKPOINT_DIR = ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "wavlm_manual_lora_tiny_best.pt"
)


# ======================================================================
# CONFIG
# ======================================================================

SEED = 42

MODEL_NAME = "microsoft/wavlm-large"

BATCH_SIZE = 2

STEPS = 1000

LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.0

LR_LORA = 1e-4

LR_HEAD = 1e-3

WEIGHT_DECAY = 0.0

PRINT_EVERY = 5

EVAL_EVERY = 5

TINY_COUNTS = {
    1: 3,
    2: 3,
    3: 4,
    4: 3,
    5: 3,
}

EXPECTED_TINY_SIZE = 16

MSE_THRESHOLD = 0.10

CCC_THRESHOLD = 0.95

EXPECTED_WAVLM_LAYERS = 24

# 24 layers × q/v × A/B
EXPECTED_LORA_TENSORS = (
    EXPECTED_WAVLM_LAYERS
    * 2
    * 2
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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
# SELECT TINY 16
# ======================================================================

def select_tiny_indices() -> list[int]:

    df = pd.read_csv(CSV_PATH)

    df["target_numeric"] = pd.to_numeric(
        df["Overall quality"],
        errors="raise",
    )

    selected_indices = []

    print()
    print("=" * 70)
    print("SELECT TINY OVERFIT SUBSET")
    print("=" * 70)

    for target_value, count in TINY_COUNTS.items():

        class_df = df[
            df["target_numeric"] == target_value
        ].copy()

        valid_indices = []

        for idx, row in class_df.iterrows():

            file_id = str(
                row["id"]
            ).strip()

            if file_id.lower().endswith(".wav"):
                wav_name = file_id
            else:
                wav_name = file_id + ".wav"

            if (WAV_DIR / wav_name).exists():
                valid_indices.append(idx)

        if len(valid_indices) < count:
            raise RuntimeError(
                f"Not enough WAV files for "
                f"target={target_value}: "
                f"need {count}, "
                f"found {len(valid_indices)}."
            )

        rng = np.random.RandomState(
            SEED + target_value
        )

        chosen = rng.choice(
            valid_indices,
            size=count,
            replace=False,
        )

        selected_indices.extend(
            int(x)
            for x in chosen
        )

    selected_indices = sorted(
        selected_indices
    )

    if len(selected_indices) != EXPECTED_TINY_SIZE:
        raise RuntimeError(
            f"Expected {EXPECTED_TINY_SIZE} "
            f"samples, got {len(selected_indices)}."
        )

    print(
        "Samples selected:",
        len(selected_indices),
    )

    print("Target distribution:")

    selected_targets = (
        df.loc[
            selected_indices,
            "target_numeric",
        ]
        .value_counts()
        .sort_index()
    )

    for target_value, count in selected_targets.items():

        print(
            f"  target={int(target_value)}: "
            f"{int(count)}"
        )

    return selected_indices


# ======================================================================
# DISABLE LAYERDROP
# ======================================================================

def disable_layerdrop(backbone) -> None:

    if hasattr(
        backbone.config,
        "layerdrop",
    ):
        backbone.config.layerdrop = 0.0

    if hasattr(
        backbone.config,
        "encoder_layerdrop",
    ):
        backbone.config.encoder_layerdrop = 0.0

    if hasattr(
        backbone,
        "encoder",
    ):
        if hasattr(
            backbone.encoder,
            "layerdrop",
        ):
            backbone.encoder.layerdrop = 0.0

    print("LayerDrop: disabled")


# ======================================================================
# MANUAL LoRA
# ======================================================================

class ManualLoRA(nn.Module):
    """
    LoRA:

        W' = W + scaling * B @ A

    A:
        [r, in_features]

    B:
        [out_features, r]

    Base W remains frozen.
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
            raise ValueError(
                "LoRA rank must be > 0."
            )

        self.r = r

        self.alpha = alpha

        self.scaling = alpha / r

        self.lora_A = nn.Parameter(
            torch.empty(
                r,
                in_features,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                r,
            )
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=np.sqrt(5),
        )

    def delta_weight(self) -> torch.Tensor:

        return (
            self.scaling
            * (
                self.lora_B
                @ self.lora_A
            )
        )


# ======================================================================
# ATTACH MANUAL LoRA
# ======================================================================

def attach_manual_lora(backbone):

    layers = backbone.encoder.layers

    if len(layers) != EXPECTED_WAVLM_LAYERS:
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_WAVLM_LAYERS} "
            f"WavLM layers, "
            f"found {len(layers)}."
        )

    target_names = []

    for layer_idx, layer in enumerate(layers):

        attention = layer.attention

        q_proj = attention.q_proj

        v_proj = attention.v_proj

        # --------------------------------------------------------------
        # Freeze original q/v
        # --------------------------------------------------------------

        q_proj.weight.requires_grad = False

        if q_proj.bias is not None:
            q_proj.bias.requires_grad = False

        v_proj.weight.requires_grad = False

        if v_proj.bias is not None:
            v_proj.bias.requires_grad = False

        # --------------------------------------------------------------
        # Manual LoRA modules
        # --------------------------------------------------------------

        q_lora = ManualLoRA(
            in_features=q_proj.in_features,
            out_features=q_proj.out_features,
            r=LORA_R,
            alpha=LORA_ALPHA,
        ).to(
            device=q_proj.weight.device,
            dtype=q_proj.weight.dtype,
        )

        v_lora = ManualLoRA(
            in_features=v_proj.in_features,
            out_features=v_proj.out_features,
            r=LORA_R,
            alpha=LORA_ALPHA,
        ).to(
            device=v_proj.weight.device,
            dtype=v_proj.weight.dtype,
        )

        attention.add_module(
            "manual_lora_q",
            q_lora,
        )

        attention.add_module(
            "manual_lora_v",
            v_lora,
        )

        target_names.extend(
            [
                f"encoder.layers.{layer_idx}"
                ".attention.q_proj",
                f"encoder.layers.{layer_idx}"
                ".attention.v_proj",
            ]
        )

    print()
    print("=" * 70)
    print("MANUAL LoRA ATTACHMENT")
    print("=" * 70)

    print(
        "Encoder layers:",
        len(layers),
    )

    print(
        "LoRA target modules:",
        len(target_names),
    )

    print(
        "Expected target modules:",
        EXPECTED_WAVLM_LAYERS * 2,
    )

    print(
        "LoRA trainable tensors:",
        EXPECTED_LORA_TENSORS,
    )

    print(
        "Rank:",
        LORA_R,
    )

    print(
        "Alpha:",
        LORA_ALPHA,
    )

    print(
        "Dropout:",
        LORA_DROPOUT,
    )

    for name in target_names[:4]:
        print("  ", name)

    print("  ...")

    return target_names


# ======================================================================
# ORIGINAL WAVLM ATTENTION + ONLY LoRA WEIGHT CHANGE
# ======================================================================

def manual_lora_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.Tensor,
    gated_position_bias: torch.FloatTensor,
):
    """
    Same WavLM attention path.

    Only q_proj.weight and v_proj.weight are replaced by:

        W + LoRA_delta

    Actual attention computation remains:

        F.multi_head_attention_forward()
    """

    # --------------------------------------------------------------
    # WavLM uses self-attention:
    # q = k = v = hidden_states
    # --------------------------------------------------------------

    query = key = value = (
        hidden_states.transpose(
            0,
            1,
        )
    )

    # --------------------------------------------------------------
    # WavLM attention mask
    # --------------------------------------------------------------

    key_padding_mask = None

    if attention_mask is not None:

        key_padding_mask = (
            attention_mask.ne(1)
        )

    bias_k = None

    bias_v = None

    add_zero_attn = False

    # --------------------------------------------------------------
    # Effective q/v weights
    # --------------------------------------------------------------

    q_weight = (
        self.q_proj.weight
        + self.manual_lora_q.delta_weight()
    )

    k_weight = (
        self.k_proj.weight
    )

    v_weight = (
        self.v_proj.weight
        + self.manual_lora_v.delta_weight()
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # Use PyTorch's ORIGINAL MHA implementation.
    # No custom QK^T / softmax / V implementation.
    # --------------------------------------------------------------

    attn_output, attn_weights = (
        F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,

            # in_proj_weight
            torch.empty(
                [0],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),

            # in_proj_bias
            torch.cat(
                (
                    self.q_proj.bias,
                    self.k_proj.bias,
                    self.v_proj.bias,
                )
            ),

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
    )

    # --------------------------------------------------------------
    # Same output layout as WavLM
    # --------------------------------------------------------------

    attn_output = (
        attn_output.transpose(
            0,
            1,
        )
    )

    if attn_weights is not None:

        attn_weights = (
            attn_weights[
                :,
                None,
            ].broadcast_to(
                attn_weights.shape[:1]
                + (self.num_heads,)
                + attn_weights.shape[1:]
            )
        )

    return (
        attn_output,
        attn_weights,
    )


# ======================================================================
# PATCH ATTENTION
# ======================================================================

def patch_wavlm_attention(backbone) -> None:

    layers = backbone.encoder.layers

    patched = 0

    for layer in layers:

        attention = layer.attention

        if not hasattr(
            attention,
            "manual_lora_q",
        ):
            raise RuntimeError(
                "Manual LoRA was not attached "
                "before attention patch."
            )

        attention.torch_multi_head_self_attention = (
            types.MethodType(
                manual_lora_attention_forward,
                attention,
            )
        )

        patched += 1

    print(
        "WavLM attention layers patched:",
        patched,
    )

    if patched != len(layers):
        raise RuntimeError(
            "Not all WavLM attention "
            "layers were patched."
        )


# ======================================================================
# FREEZE BACKBONE
# ======================================================================

def freeze_backbone(backbone) -> None:

    for param in backbone.parameters():

        param.requires_grad = False

    print(
        "All WavLM base parameters frozen."
    )


# ======================================================================
# PARAMETER AUDIT
# ======================================================================

def audit_trainable_parameters(
    backbone,
    quality_head,
):

    lora_params = []

    unexpected_backbone = []

    for name, param in backbone.named_parameters():

        if not param.requires_grad:
            continue

        if (
            "manual_lora_q" in name
            or "manual_lora_v" in name
        ):
            lora_params.append(name)
        else:
            unexpected_backbone.append(name)

    print()
    print("=" * 70)
    print("TRAINABLE PARAMETER AUDIT")
    print("=" * 70)

    print(
        "Manual LoRA tensors:",
        len(lora_params),
    )

    print(
        "Expected:",
        EXPECTED_LORA_TENSORS,
    )

    if len(lora_params) != EXPECTED_LORA_TENSORS:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_LORA_TENSORS} "
            f"LoRA tensors, "
            f"found {len(lora_params)}."
        )

    if unexpected_backbone:

        print(
            "Unexpected trainable WavLM parameters:"
        )

        for name in unexpected_backbone:
            print(
                "  ",
                name,
            )

        raise RuntimeError(
            "Unexpected WavLM "
            "parameters are trainable."
        )

    lora_count = sum(
        p.numel()
        for p in backbone.parameters()
        if p.requires_grad
    )

    head_count = sum(
        p.numel()
        for p in quality_head.parameters()
        if p.requires_grad
    )

    print(
        "Trainable LoRA parameters:",
        lora_count,
    )

    print(
        "Trainable QualityHead parameters:",
        head_count,
    )

    print(
        "Unexpected WavLM trainables: 0"
    )

    print(
        "Parameter audit: PASS"
    )


# ======================================================================
# FRAME MASK
# ======================================================================

def make_frame_valid_mask(
    backbone,
    attention_mask,
    max_frames,
):

    lengths = (
        attention_mask
        .long()
        .sum(dim=1)
    )

    frame_lengths = (
        backbone
        ._get_feat_extract_output_lengths(
            lengths
        )
        .to(
            attention_mask.device
        )
    )

    frame_idx = torch.arange(
        max_frames,
        device=attention_mask.device,
    ).unsqueeze(0)

    return (
        frame_idx
        < frame_lengths.unsqueeze(1)
    )


# ======================================================================
# EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate_tiny(
    backbone,
    quality_head,
    loader,
):

    backbone.eval()

    quality_head.eval()

    predictions = []

    targets = []

    file_ids = []

    for batch in loader:

        wav = batch["wav"].to(
            DEVICE,
            dtype=torch.float32,
        )

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        target = batch["target"].to(
            DEVICE
        )

        output = backbone(
            input_values=wav,
            attention_mask=attention_mask.long(),
        )

        hidden = (
            output.last_hidden_state
        )

        valid_mask = (
            make_frame_valid_mask(
                backbone,
                attention_mask,
                hidden.shape[1],
            )
        )

        pred = quality_head(
            hidden,
            valid_mask,
        )

        predictions.append(
            pred.float().cpu()
        )

        targets.append(
            target.float().cpu()
        )

        if "file_id" in batch:
            file_ids.extend(
                batch["file_id"]
            )

    predictions = torch.cat(
        predictions
    )

    targets = torch.cat(
        targets
    )

    mse = torch.mean(
        (
            predictions
            - targets
        ) ** 2
    ).item()

    mae = torch.mean(
        torch.abs(
            predictions
            - targets
        )
    ).item()

    ccc = (
        concordance_correlation_coefficient(
            predictions,
            targets,
        )
        .item()
    )

    return {
        "predictions": predictions,
        "targets": targets,
        "file_ids": file_ids,
        "mse": mse,
        "mae": mae,
        "ccc": ccc,
    }


# ======================================================================
# PRINT PREDICTIONS
# ======================================================================

def print_predictions(result):

    predictions = result["predictions"]

    targets = result["targets"]

    file_ids = result["file_ids"]

    print()

    print(
        f"{'file_id':45s} "
        f"{'target':>8s} "
        f"{'pred':>8s} "
        f"{'abs_err':>10s}"
    )

    print("-" * 80)

    for i in range(
        len(predictions)
    ):

        file_id = (
            file_ids[i]
            if i < len(file_ids)
            else "unknown"
        )

        target = targets[i].item()

        pred = predictions[i].item()

        print(
            f"{str(file_id):45s} "
            f"{target:8.3f} "
            f"{pred:8.3f} "
            f"{abs(pred-target):10.3f}"
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    set_seed(SEED)

    print("=" * 70)

    print(
        "WAVLM-LARGE + MANUAL LoRA "
        "TINY OVERFIT"
    )

    print("=" * 70)

    print(
        "Device:",
        DEVICE,
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Backbone:",
        MODEL_NAME,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Steps:",
        STEPS,
    )

    print(
        "LoRA rank:",
        LORA_R,
    )

    print(
        "LoRA alpha:",
        LORA_ALPHA,
    )

    print(
        "LoRA LR:",
        LR_LORA,
    )

    print(
        "QualityHead LR:",
        LR_HEAD,
    )

    # ==================================================================
    # DATA
    # ==================================================================

    tiny_indices = (
        select_tiny_indices()
    )

    dataset = QualiSpeechDataset(
        csv_path=CSV_PATH,
        wav_dir=WAV_DIR,
        indices=tiny_indices,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        collate_fn=qualispeech_collate,
    )

    eval_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        collate_fn=qualispeech_collate,
    )

    # ==================================================================
    # LOAD WAVLM
    # ==================================================================

    print()
    print("=" * 70)
    print("LOAD WAVLM-LARGE")
    print("=" * 70)

    backbone = (
        WavLMModel.from_pretrained(
            MODEL_NAME
        )
    )

    backbone = backbone.to(DEVICE)

    # ==================================================================
    # FREEZE
    # ==================================================================

    freeze_backbone(
        backbone
    )

    # ==================================================================
    # LAYERDROP
    # ==================================================================

    disable_layerdrop(
        backbone
    )

    # ==================================================================
    # MANUAL LoRA
    # ==================================================================

    target_names = (
        attach_manual_lora(
            backbone
        )
    )

    # ==================================================================
    # PATCH ATTENTION
    # ==================================================================

    patch_wavlm_attention(
        backbone
    )

    # ==================================================================
    # QUALITY HEAD
    # ==================================================================

    hidden_size = (
        backbone.config.hidden_size
    )

    print(
        "WavLM hidden size:",
        hidden_size,
    )

    quality_head = (
        QualityHead(
            hidden_size=hidden_size
        )
        .to(DEVICE)
    )

    # ==================================================================
    # PARAMETER AUDIT
    # ==================================================================

    audit_trainable_parameters(
        backbone,
        quality_head,
    )

    # ==================================================================
    # OPTIMIZER
    # ==================================================================

    lora_params = [
        p
        for p in backbone.parameters()
        if p.requires_grad
    ]

    head_params = [
        p
        for p in quality_head.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        [
            {
                "params": lora_params,
                "lr": LR_LORA,
            },
            {
                "params": head_params,
                "lr": LR_HEAD,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # ==================================================================
    # INITIAL EVALUATION
    # ==================================================================

    initial = evaluate_tiny(
        backbone,
        quality_head,
        eval_loader,
    )

    print()
    print("=" * 70)
    print("INITIAL FULL-16")
    print("=" * 70)

    print(
        f"MSE: {initial['mse']:.8f}"
    )

    print(
        f"MAE: {initial['mae']:.8f}"
    )

    print(
        f"CCC: {initial['ccc']:.8f}"
    )

    # ==================================================================
    # TRAIN
    # ==================================================================

    backbone.train()

    quality_head.train()

    loader_iter = iter(
        train_loader
    )

    best_mse = float("inf")

    best_ccc = -float("inf")

    best_step = -1

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for step in range(
        1,
        STEPS + 1,
    ):

        try:

            batch = next(
                loader_iter
            )

        except StopIteration:

            loader_iter = iter(
                train_loader
            )

            batch = next(
                loader_iter
            )

        wav = batch["wav"].to(
            DEVICE,
            dtype=torch.float32,
        )

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        target = batch["target"].to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------------------
        # FORWARD
        # --------------------------------------------------------------

        output = backbone(
            input_values=wav,
            attention_mask=attention_mask.long(),
        )

        hidden = (
            output.last_hidden_state
        )

        if step == 1:

            print()
            print("=" * 70)
            print("WAVLM GRADIENT DIAGNOSTIC")
            print("=" * 70)

            print(
                "hidden.requires_grad:",
                hidden.requires_grad,
            )

            print(
                "hidden.grad_fn:",
                hidden.grad_fn,
            )

        valid_mask = (
            make_frame_valid_mask(
                backbone,
                attention_mask,
                hidden.shape[1],
            )
        )

        pred = quality_head(
            hidden,
            valid_mask,
        )

        loss = quality_loss(
            pred,
            target,
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at "
                f"step {step}: "
                f"{loss.item()}"
            )

        # --------------------------------------------------------------
        # BACKWARD
        # --------------------------------------------------------------

        loss.backward()

        # --------------------------------------------------------------
        # LoRA GRADIENT DIAGNOSTIC
        # --------------------------------------------------------------

        missing_grad = []

        zero_grad_count = 0

        lora_tensor_count = 0

        for name, param in (
            backbone.named_parameters()
        ):

            if not param.requires_grad:
                continue

            if (
                "manual_lora_q" in name
                or "manual_lora_v" in name
            ):

                lora_tensor_count += 1

                if param.grad is None:

                    missing_grad.append(
                        name
                    )

                else:

                    if not torch.isfinite(
                        param.grad
                    ).all():

                        raise RuntimeError(
                            f"Non-finite "
                            f"gradient: "
                            f"{name}"
                        )

                    if (
                        param.grad.abs().max()
                        .item()
                        == 0.0
                    ):

                        zero_grad_count += 1

        if (
            lora_tensor_count
            != EXPECTED_LORA_TENSORS
        ):

            raise RuntimeError(
                f"Expected "
                f"{EXPECTED_LORA_TENSORS} "
                f"LoRA tensors, "
                f"found "
                f"{lora_tensor_count}."
            )

        if missing_grad:

            raise RuntimeError(
                "LoRA parameters with "
                "missing gradients at "
                f"step {step}:\n"
                + "\n".join(
                    missing_grad
                )
            )

        # --------------------------------------------------------------
        # QUALITY HEAD GRADIENT
        # --------------------------------------------------------------

        for name, param in (
            quality_head.named_parameters()
        ):

            if not param.requires_grad:
                continue

            if param.grad is None:

                raise RuntimeError(
                    f"QualityHead parameter "
                    f"has no gradient: "
                    f"{name}"
                )

            if not torch.isfinite(
                param.grad
            ).all():

                raise RuntimeError(
                    f"Non-finite "
                    f"QualityHead gradient: "
                    f"{name}"
                )

        if step == 1:

            print()
            print("=" * 70)
            print("FIRST LoRA GRADIENT")
            print("=" * 70)

            first_found = False

            for name, param in (
                backbone.named_parameters()
            ):

                if (
                    (
                        "manual_lora_q"
                        in name
                    )
                    or
                    (
                        "manual_lora_v"
                        in name
                    )
                ) and param.requires_grad:

                    print(
                        name
                    )

                    print(
                        "  grad is None:",
                        param.grad is None,
                    )

                    print(
                        "  grad norm:",
                        f"{param.grad.norm().item():.6e}",
                    )

                    first_found = True

                    break

            if not first_found:

                raise RuntimeError(
                    "No LoRA tensor found."
                )

            print(
                "LoRA tensors with exactly "
                "zero gradient:",
                zero_grad_count,
                "/",
                EXPECTED_LORA_TENSORS,
            )

            print(
                "Zero gradient at step 1 "
                "can be normal for LoRA-A "
                "because LoRA-B starts at zero."
            )

        # --------------------------------------------------------------
        # OPTIMIZER
        # --------------------------------------------------------------

        optimizer.step()

        # --------------------------------------------------------------
        # TRAIN LOG
        # --------------------------------------------------------------

        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == STEPS
        ):

            print(
                f"Step {step:03d}/{STEPS} "
                f"| train_loss="
                f"{loss.item():.8f}"
            )

        # --------------------------------------------------------------
        # FULL-16 EVALUATION
        # --------------------------------------------------------------

        if (
            step % EVAL_EVERY == 0
            or step == STEPS
        ):

            result = evaluate_tiny(
                backbone,
                quality_head,
                eval_loader,
            )

            mse = result["mse"]

            mae = result["mae"]

            ccc = result["ccc"]

            print(
                f"           FULL-16 "
                f"MSE={mse:.8f} "
                f"MAE={mae:.8f} "
                f"CCC={ccc:.8f}"
            )

            if mse < best_mse:

                best_mse = mse

                best_ccc = ccc

                best_step = step

                torch.save(
                    {
                        "step": step,
                        "model_name": MODEL_NAME,
                        "hidden_size": hidden_size,

                        "lora": {
                            "type": "manual",
                            "rank": LORA_R,
                            "alpha": LORA_ALPHA,
                            "dropout": LORA_DROPOUT,
                            "targets": target_names,
                        },

                        "backbone_state_dict":
                            backbone.state_dict(),

                        "quality_head_state_dict":
                            quality_head.state_dict(),

                        "optimizer_state_dict":
                            optimizer.state_dict(),

                        "loss": loss.item(),
                        "mse": mse,
                        "mae": mae,
                        "ccc": ccc,
                    },
                    BEST_CHECKPOINT,
                )

                print(
                    f"           -> NEW BEST "
                    f"(step={step}, "
                    f"MSE={mse:.8f})"
                )

            backbone.train()

            quality_head.train()

    # ==================================================================
    # RESTORE BEST
    # ==================================================================

    print()
    print("=" * 70)
    print("RESTORE BEST CHECKPOINT")
    print("=" * 70)

    if not BEST_CHECKPOINT.exists():

        raise RuntimeError(
            "Best checkpoint was not created."
        )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE,
    )

    backbone.load_state_dict(
        checkpoint[
            "backbone_state_dict"
        ]
    )

    quality_head.load_state_dict(
        checkpoint[
            "quality_head_state_dict"
        ]
    )

    print(
        "Best step:",
        checkpoint["step"],
    )

    print(
        "Checkpoint:",
        BEST_CHECKPOINT,
    )

    # ==================================================================
    # FINAL FULL-16
    # ==================================================================

    final = evaluate_tiny(
        backbone,
        quality_head,
        eval_loader,
    )

    print()
    print("=" * 70)
    print(
        "FINAL RESULT — BEST FULL-16"
    )
    print("=" * 70)

    print(
        f"MSE: "
        f"{final['mse']:.8f}"
    )

    print(
        f"MAE: "
        f"{final['mae']:.8f}"
    )

    print(
        f"CCC: "
        f"{final['ccc']:.8f}"
    )

    print_predictions(
        final
    )

    # ==================================================================
    # PASS
    # ==================================================================

    mse_pass = (
        np.isfinite(
            final["mse"]
        )
        and
        final["mse"] < MSE_THRESHOLD
    )

    ccc_pass = (
        np.isfinite(
            final["ccc"]
        )
        and
        final["ccc"] > CCC_THRESHOLD
    )

    finite_pass = bool(
        torch.isfinite(
            final["predictions"]
        ).all()
    )

    print()
    print("=" * 70)
    print("TINY OVERFIT CHECK")
    print("=" * 70)

    print(
        f"MSE < {MSE_THRESHOLD}:",
        "PASS" if mse_pass else "FAIL",
    )

    print(
        f"CCC > {CCC_THRESHOLD}:",
        "PASS" if ccc_pass else "FAIL",
    )

    print(
        "Finite predictions:",
        "PASS" if finite_pass else "FAIL",
    )

    if (
        mse_pass
        and ccc_pass
        and finite_pass
    ):

        print()
        print("=" * 70)

        print(
            ">>> WAVLM MANUAL LoRA "
            "TINY OVERFIT: PASS"
        )

        print("=" * 70)

    else:

        print()
        print("=" * 70)

        print(
            ">>> WAVLM MANUAL LoRA "
            "TINY OVERFIT: FAIL"
        )

        print("=" * 70)

        raise RuntimeError(
            "Tiny overfit thresholds "
            "were not satisfied."
        )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
