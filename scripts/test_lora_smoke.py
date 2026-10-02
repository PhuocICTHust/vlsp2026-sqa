from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import HubertModel, WavLMModel
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.dataset import QualiSpeechDataset, qualispeech_collate
from sqa.model import QualityHead, quality_loss


# ======================================================================
# PATHS / DEVICE
# ======================================================================

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
WAV_DIR = ROOT / "data" / "qualispeech" / "wav"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# BACKBONE
# ======================================================================

# Current test: HuBERT-base
MODEL_NAME = "facebook/hubert-base-ls960"

# To test WavLM instead, uncomment:
# MODEL_NAME = "microsoft/wavlm-large"


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("LoRA SMOKE TEST")
    print("=" * 70)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    #
    # Only 2 samples are used.
    # This test is NOT a training benchmark.
    # It only verifies:
    #
    #   WAV
    #    ↓
    #   Backbone
    #    ↓
    #   LoRA
    #    ↓
    #   QualityHead
    #    ↓
    #   Loss
    #    ↓
    #   Backward
    #
    # ------------------------------------------------------------------

    dataset = QualiSpeechDataset(
        csv_path=CSV_PATH,
        wav_dir=WAV_DIR,
        indices=[0, 1],
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=qualispeech_collate,
    )

    batch = next(iter(loader))

    wav = batch["wav"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    target = batch["target"].to(DEVICE)

    print("WAV shape:", tuple(wav.shape))
    print("Target:", target.tolist())

    # ------------------------------------------------------------------
    # Load backbone
    # ------------------------------------------------------------------

    if "hubert" in MODEL_NAME.lower():

        print("Backbone: HuBERT")

        backbone = HubertModel.from_pretrained(
            MODEL_NAME
        )

    else:

        print("Backbone: WavLM")

        backbone = WavLMModel.from_pretrained(
            MODEL_NAME
        )

    backbone = backbone.to(DEVICE)

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Disable LayerDrop for deterministic smoke testing.
    #
    # HuBERT can randomly skip encoder layers during training when
    # layerdrop > 0. If a layer is skipped, its LoRA parameters do not
    # participate in that forward pass and therefore receive no gradient.
    #
    # For the smoke test we want every LoRA layer to participate.
    # ------------------------------------------------------------------

    if hasattr(backbone.config, "layerdrop"):

        backbone.config.layerdrop = 0.0

        print("LayerDrop: disabled")

    # ------------------------------------------------------------------
    # LoRA configuration
    # ------------------------------------------------------------------

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    # ------------------------------------------------------------------
    # Apply LoRA
    # ------------------------------------------------------------------

    #
    # Do NOT manually freeze backbone before this.
    #
    # PEFT handles the freezing of pretrained parameters and leaves
    # LoRA parameters trainable.
    #

    backbone = get_peft_model(
        backbone,
        lora_config,
    )

    backbone.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Verify trainable parameters
    # ------------------------------------------------------------------

    trainable_names = [
        name
        for name, param in backbone.named_parameters()
        if param.requires_grad
    ]

    if not trainable_names:

        raise RuntimeError(
            "No trainable parameters found after applying LoRA."
        )

    # Every trainable parameter should be a LoRA parameter.
    non_lora_trainable = [
        name
        for name in trainable_names
        if "lora_" not in name
    ]

    if non_lora_trainable:

        raise RuntimeError(
            "Unexpected non-LoRA trainable parameters:\n"
            + "\n".join(
                non_lora_trainable[:20]
            )
        )

    print(
        "Trainable LoRA parameters:",
        len(trainable_names),
    )

    # ------------------------------------------------------------------
    # Enable input gradients
    # ------------------------------------------------------------------
    #
    # PEFT adapters need the forward graph to remain connected to the
    # trainable LoRA parameters.
    #
    # This does NOT unfreeze the pretrained backbone.
    #

    if hasattr(
        backbone,
        "enable_input_require_grads",
    ):

        backbone.enable_input_require_grads()

        print(
            "Input gradient support: enabled"
        )

    # ------------------------------------------------------------------
    # Quality Head
    # ------------------------------------------------------------------

    hidden_size = backbone.config.hidden_size

    print(
        "Hidden size:",
        hidden_size,
    )

    quality_head = QualityHead(
        hidden_size=hidden_size
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    lora_params = [
        param
        for param in backbone.parameters()
        if param.requires_grad
    ]

    head_params = list(
        quality_head.parameters()
    )

    if not lora_params:

        raise RuntimeError(
            "No LoRA parameters available for optimizer."
        )

    if not head_params:

        raise RuntimeError(
            "No QualityHead parameters available."
        )

    optimizer = torch.optim.Adam(
        [
            {
                "params": lora_params,
                "lr": 1e-4,
            },
            {
                "params": head_params,
                "lr": 1e-3,
            },
        ]
    )

    print(
        "LoRA optimizer LR:",
        1e-4,
    )

    print(
        "QualityHead optimizer LR:",
        1e-3,
    )

    # ------------------------------------------------------------------
    # Train mode
    # ------------------------------------------------------------------

    backbone.train()
    quality_head.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    # No torch.no_grad().
    #
    # No autocast here either.
    #
    # The smoke test is specifically checking the autograd graph.
    # FP32 makes the test easier to diagnose.
    # ------------------------------------------------------------------

    output = backbone(
        input_values=wav,
        attention_mask=attention_mask.long(),
    )

    hidden = output.last_hidden_state

    # ------------------------------------------------------------------
    # Verify hidden tensor
    # ------------------------------------------------------------------

    print(
        "Hidden shape:",
        tuple(hidden.shape),
    )

    print(
        "Hidden requires_grad:",
        hidden.requires_grad,
    )

    if not hidden.requires_grad:

        raise RuntimeError(
            "Hidden states do not require gradients. "
            "The LoRA computational graph is detached."
        )

    # ------------------------------------------------------------------
    # Frame mask
    # ------------------------------------------------------------------
    #
    # Approximate mask is sufficient for this smoke test.
    #
    # The real training script should use the exact frame-valid mask.
    # ------------------------------------------------------------------

    frame_mask = torch.ones(
        hidden.shape[:2],
        dtype=torch.bool,
        device=DEVICE,
    )

    # ------------------------------------------------------------------
    # Quality prediction
    # ------------------------------------------------------------------

    pred = quality_head(
        hidden,
        frame_mask,
    )

    # ------------------------------------------------------------------
    # Quality loss
    # ------------------------------------------------------------------

    loss = quality_loss(
        pred,
        target,
    )

    print(
        "Prediction:",
        pred.detach().float().tolist(),
    )

    print(
        "Loss:",
        float(loss.detach().float()),
    )

    print(
        "Loss requires_grad:",
        loss.requires_grad,
    )

    if not torch.isfinite(loss):

        raise RuntimeError(
            "Loss is not finite."
        )

    if not loss.requires_grad:

        raise RuntimeError(
            "Loss does not require gradients."
        )

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    print()
    print("Running backward...")

    loss.backward()

    print("Backward: completed")

    # ------------------------------------------------------------------
    # Verify LoRA gradients
    # ------------------------------------------------------------------

    print()
    print("GRADIENT CHECK")
    print("-" * 70)

    lora_gradients = []

    missing_lora_gradients = []

    for name, param in backbone.named_parameters():

        if "lora_" not in name:
            continue

        if not param.requires_grad:
            continue

        if param.grad is None:

            missing_lora_gradients.append(
                name
            )

            continue

        if not torch.isfinite(
            param.grad
        ).all():

            raise RuntimeError(
                f"Non-finite LoRA gradient: {name}"
            )

        grad_norm = (
            param.grad
            .detach()
            .float()
            .norm()
            .item()
        )

        print(
            f"{name:75s} "
            f"grad_norm={grad_norm:.6e}"
        )

        lora_gradients.append(name)

    # ------------------------------------------------------------------
    # Smoke test requires every LoRA parameter to receive a gradient.
    #
    # LayerDrop was disabled above specifically so that every encoder
    # layer participates in the forward pass.
    # ------------------------------------------------------------------

    if missing_lora_gradients:

        print()
        print(
            "Missing LoRA gradients:"
        )

        for name in missing_lora_gradients:

            print(
                "  ",
                name,
            )

        raise RuntimeError(
            "Some LoRA parameters did not receive gradients."
        )

    if not lora_gradients:

        raise RuntimeError(
            "No LoRA parameters with gradients found."
        )

    # ------------------------------------------------------------------
    # Verify QualityHead gradients
    # ------------------------------------------------------------------

    print()
    print("QUALITY HEAD GRADIENT CHECK")
    print("-" * 70)

    head_gradients = []

    missing_head_gradients = []

    for name, param in quality_head.named_parameters():

        if not param.requires_grad:
            continue

        if param.grad is None:

            missing_head_gradients.append(
                name
            )

            continue

        if not torch.isfinite(
            param.grad
        ).all():

            raise RuntimeError(
                f"Non-finite QualityHead gradient: {name}"
            )

        grad_norm = (
            param.grad
            .detach()
            .float()
            .norm()
            .item()
        )

        print(
            f"{name:40s} "
            f"grad_norm={grad_norm:.6e}"
        )

        head_gradients.append(name)

    if missing_head_gradients:

        print()
        print(
            "Missing QualityHead gradients:"
        )

        for name in missing_head_gradients:

            print(
                "  ",
                name,
            )

        raise RuntimeError(
            "Some QualityHead parameters did not receive gradients."
        )

    if not head_gradients:

        raise RuntimeError(
            "No QualityHead parameters with gradients found."
        )

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    print()
    print("Running optimizer.step()...")

    optimizer.step()

    print(
        "Optimizer step: completed"
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("LoRA SMOKE TEST SUMMARY")
    print("=" * 70)

    print(
        "Backbone:",
        MODEL_NAME,
    )

    print(
        "Hidden size:",
        hidden_size,
    )

    print(
        "LoRA trainable tensors:",
        len(lora_gradients),
    )

    print(
        "QualityHead tensors with gradients:",
        len(head_gradients),
    )

    print(
        "Loss:",
        float(loss.detach().float()),
    )

    print()
    print("=" * 70)
    print(">>> LoRA SMOKE TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
