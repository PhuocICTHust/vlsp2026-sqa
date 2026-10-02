from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import HubertModel
from peft import LoraConfig, get_peft_model


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.dataset import QualiSpeechDataset, qualispeech_collate
from sqa.model import QualityHead, quality_loss, concordance_correlation_coefficient


CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
WAV_DIR = ROOT / "data" / "qualispeech" / "wav"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "hubert_lora_tiny_best.pt"
)


# ======================================================================
# CONFIG
# ======================================================================

SEED = 42

MODEL_NAME = "facebook/hubert-base-ls960"

BATCH_SIZE = 4

STEPS = 100

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
# SELECT EXACT 16-SAMPLE TINY SET
# ======================================================================

def select_tiny_indices(
    csv_path: Path,
    wav_dir: Path,
) -> list[int]:

    df = pd.read_csv(csv_path)

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

        # --------------------------------------------------------------
        # Keep only samples whose WAV exists.
        # --------------------------------------------------------------

        valid_rows = []

        for idx, row in class_df.iterrows():

            file_id = str(row["id"]).strip()

            if file_id.lower().endswith(".wav"):
                wav_name = file_id
            else:
                wav_name = file_id + ".wav"

            wav_path = wav_dir / wav_name

            if wav_path.exists():
                valid_rows.append(idx)

        if len(valid_rows) < count:

            raise RuntimeError(
                f"Not enough WAV files for target={target_value}. "
                f"Need {count}, found {len(valid_rows)}."
            )

        # Deterministic selection.
        rng = np.random.RandomState(
            SEED + target_value
        )

        chosen = rng.choice(
            valid_rows,
            size=count,
            replace=False,
        )

        chosen = sorted(
            [int(x) for x in chosen]
        )

        selected_indices.extend(chosen)

        print(
            f"target={target_value}: "
            f"selected={len(chosen)}"
        )

    if len(selected_indices) != EXPECTED_TINY_SIZE:

        raise RuntimeError(
            f"Expected {EXPECTED_TINY_SIZE} samples, "
            f"got {len(selected_indices)}."
        )

    # Keep deterministic order.
    selected_indices = sorted(
        selected_indices
    )

    print()
    print(
        "Total selected:",
        len(selected_indices),
    )

    print(
        "Target distribution:"
    )

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

    print()

    return selected_indices


# ======================================================================
# EXACT FRAME MASK
# ======================================================================

def make_frame_valid_mask(
    backbone,
    attention_mask: torch.Tensor,
    max_frames: int,
) -> torch.Tensor:

    base_model = backbone.get_base_model()

    sample_lengths = (
        attention_mask
        .long()
        .sum(dim=1)
    )

    frame_lengths = (
        base_model
        ._get_feat_extract_output_lengths(
            sample_lengths
        )
        .to(attention_mask.device)
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
# EVALUATE ALL 16 SAMPLES
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
            DEVICE
        )

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        target = batch[
            "target"
        ].to(DEVICE)

        output = backbone(
            input_values=wav,
            attention_mask=attention_mask.long(),
        )

        hidden = output.last_hidden_state

        frame_mask = make_frame_valid_mask(
            backbone,
            attention_mask,
            hidden.shape[1],
        )

        pred = quality_head(
            hidden,
            frame_mask,
        )

        predictions.append(
            pred.float().cpu()
        )

        targets.append(
            target.float().cpu()
        )

        batch_file_ids = batch.get(
            "file_id",
            ["unknown"] * len(target),
        )

        file_ids.extend(
            batch_file_ids
        )

    predictions = torch.cat(
        predictions
    )

    targets = torch.cat(
        targets
    )

    mse = torch.mean(
        (predictions - targets) ** 2
    ).item()

    mae = torch.mean(
        torch.abs(predictions - targets)
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
# PRINT FULL-16 RESULTS
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

    for file_id, target, pred in zip(
        file_ids,
        targets.tolist(),
        predictions.tolist(),
    ):

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
    print("HuBERT + LoRA TINY OVERFIT")
    print("=" * 70)

    print("Device:", DEVICE)

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
        "Steps:",
        STEPS,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
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
    # SELECT DATA
    # ==================================================================

    tiny_indices = select_tiny_indices(
        CSV_PATH,
        WAV_DIR,
    )

    # ==================================================================
    # DATASET
    # ==================================================================

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
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=qualispeech_collate,
    )

    eval_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=qualispeech_collate,
    )

    # ==================================================================
    # LOAD HUBERT
    # ==================================================================

    print()
    print("Loading HuBERT...")

    backbone = HubertModel.from_pretrained(
        MODEL_NAME
    )

    backbone = backbone.to(DEVICE)

    # ==================================================================
    # DISABLE LAYERDROP
    # ==================================================================

    if hasattr(
        backbone.config,
        "layerdrop",
    ):

        backbone.config.layerdrop = 0.0

        print(
            "LayerDrop: disabled"
        )

    # ==================================================================
    # LoRA
    # ==================================================================

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

    backbone = get_peft_model(
        backbone,
        lora_config,
    )

    backbone.print_trainable_parameters()

    # ==================================================================
    # INPUT GRADIENT SUPPORT
    # ==================================================================

    if hasattr(
        backbone,
        "enable_input_require_grads",
    ):

        backbone.enable_input_require_grads()

    # ==================================================================
    # QUALITY HEAD
    # ==================================================================

    hidden_size = backbone.config.hidden_size

    quality_head = QualityHead(
        hidden_size=hidden_size
    ).to(DEVICE)

    # ==================================================================
    # OPTIMIZER
    # ==================================================================

    lora_params = [
        p
        for p in backbone.parameters()
        if p.requires_grad
    ]

    head_params = list(
        quality_head.parameters()
    )

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

    initial_result = evaluate_tiny(
        backbone,
        quality_head,
        eval_loader,
    )

    print()
    print("=" * 70)
    print("INITIAL FULL-16")
    print("=" * 70)

    print(
        f"MSE:  {initial_result['mse']:.8f}"
    )

    print(
        f"MAE:  {initial_result['mae']:.8f}"
    )

    print(
        f"CCC:  {initial_result['ccc']:.8f}"
    )

    # ==================================================================
    # TRAINING
    # ==================================================================

    backbone.train()
    quality_head.train()

    best_mse = float("inf")
    best_ccc = -float("inf")
    best_step = -1

    loader_iter = iter(train_loader)

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
            DEVICE
        )

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        target = batch[
            "target"
        ].to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------------------
        # Forward
        # --------------------------------------------------------------

        output = backbone(
            input_values=wav,
            attention_mask=attention_mask.long(),
        )

        hidden = output.last_hidden_state

        frame_mask = make_frame_valid_mask(
            backbone,
            attention_mask,
            hidden.shape[1],
        )

        pred = quality_head(
            hidden,
            frame_mask,
        )

        loss = quality_loss(
            pred,
            target,
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"{loss.item()}"
            )

        # --------------------------------------------------------------
        # Backward
        # --------------------------------------------------------------

        loss.backward()

        # --------------------------------------------------------------
        # Check gradients
        # --------------------------------------------------------------

        for name, param in backbone.named_parameters():

            if (
                "lora_" in name
                and param.requires_grad
            ):

                if param.grad is None:

                    raise RuntimeError(
                        f"No LoRA gradient at step "
                        f"{step}: {name}"
                    )

                if not torch.isfinite(
                    param.grad
                ).all():

                    raise RuntimeError(
                        f"Non-finite LoRA gradient "
                        f"at step {step}: {name}"
                    )

        # --------------------------------------------------------------
        # Optimizer
        # --------------------------------------------------------------

        optimizer.step()

        # --------------------------------------------------------------
        # Training log
        # --------------------------------------------------------------

        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == STEPS
        ):

            print(
                f"Step {step:03d}/{STEPS} "
                f"| train_loss={loss.item():.8f}"
            )

        # --------------------------------------------------------------
        # Full-16 evaluation
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
                f"           "
                f"FULL-16 MSE={mse:.8f} "
                f"MAE={mae:.8f} "
                f"CCC={ccc:.8f}"
            )

            # ----------------------------------------------------------
            # Best checkpoint = lowest full-16 MSE
            # ----------------------------------------------------------

            if mse < best_mse:

                best_mse = mse
                best_ccc = ccc
                best_step = step

                torch.save(
                    {
                        "step": step,
                        "backbone": MODEL_NAME,
                        "hidden_size": hidden_size,
                        "lora_config": {
                            "r": 8,
                            "lora_alpha": 16,
                            "lora_dropout": 0.05,
                            "target_modules": [
                                "q_proj",
                                "v_proj",
                            ],
                        },
                        "backbone_state_dict": (
                            backbone.state_dict()
                        ),
                        "quality_head_state_dict": (
                            quality_head.state_dict()
                        ),
                        "optimizer_state_dict": (
                            optimizer.state_dict()
                        ),
                        "mse": mse,
                        "mae": mae,
                        "ccc": ccc,
                    },
                    BEST_CHECKPOINT,
                )

                print(
                    f"           "
                    f"-> NEW BEST "
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

    best_step = checkpoint["step"]

    print(
        "Best step:",
        best_step,
    )

    print(
        "Checkpoint:",
        BEST_CHECKPOINT,
    )

    # ==================================================================
    # FINAL FULL-16 EVALUATION
    # ==================================================================

    final_result = evaluate_tiny(
        backbone,
        quality_head,
        eval_loader,
    )

    # ==================================================================
    # PRINT FINAL
    # ==================================================================

    print()
    print("=" * 70)
    print("FINAL RESULT — HuBERT LoRA BEST FULL-16 CHECKPOINT")
    print("=" * 70)

    print(
        "Backbone:",
        MODEL_NAME,
    )

    print(
        "Hidden:",
        hidden_size,
    )

    print(
        "Best step:",
        best_step,
    )

    print(
        f"Loss:    "
        f"{checkpoint.get('loss', float('nan')):.8f}"
    )

    print(
        f"MSE:     "
        f"{final_result['mse']:.8f}"
    )

    print(
        f"MAE:     "
        f"{final_result['mae']:.8f}"
    )

    print(
        f"CCC:     "
        f"{final_result['ccc']:.8f}"
    )

    print_predictions(
        final_result
    )

    # ==================================================================
    # PASS / FAIL
    # ==================================================================

    mse_pass = (
        np.isfinite(final_result["mse"])
        and final_result["mse"]
        < MSE_THRESHOLD
    )

    ccc_pass = (
        np.isfinite(final_result["ccc"])
        and final_result["ccc"]
        > CCC_THRESHOLD
    )

    finite_predictions = bool(
        torch.isfinite(
            final_result["predictions"]
        ).all()
    )

    print()
    print("=" * 70)
    print("TINY OVERFIT CHECK")
    print("=" * 70)

    print(
        f"MSE < {MSE_THRESHOLD}:",
        final_result["mse"],
        "PASS" if mse_pass else "FAIL",
    )

    print(
        f"CCC > {CCC_THRESHOLD}:",
        final_result["ccc"],
        "PASS" if ccc_pass else "FAIL",
    )

    print(
        "Finite predictions:",
        "PASS" if finite_predictions else "FAIL",
    )

    print()

    if (
        mse_pass
        and ccc_pass
        and finite_predictions
    ):

        print("=" * 70)
        print(">>> HuBERT LoRA TINY OVERFIT: PASS")
        print("=" * 70)

        print(
            "The raw-WAV HuBERT + LoRA + QualityHead "
            "training pipeline can overfit the full 16-sample set."
        )

    else:

        print("=" * 70)
        print(">>> HuBERT LoRA TINY OVERFIT: FAIL")
        print("=" * 70)

        raise RuntimeError(
            "Tiny overfit thresholds were not satisfied."
        )


if __name__ == "__main__":
    main()
