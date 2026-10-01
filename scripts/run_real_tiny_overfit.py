from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.optim import Adam

# ---------------------------------------------------------------------
# Make src/ importable when running:
#     python scripts/run_real_tiny_overfit.py
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.dataset import QualiSpeechDataset, qualispeech_collate
from sqa.model import SQAModel, quality_loss


CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
ZIP_PATH = ROOT / "data" / "qualispeech" / "wav_part1.zip"

# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------

CHECKPOINT_DIR = ROOT / "checkpoints"
BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "real_tiny_best.pt"
)

SEED = 42

# Tiny-overfit configuration.
SAMPLES_PER_CLASS = {
    1: 3,
    2: 3,
    3: 4,
    4: 3,
    5: 3,
}

BATCH_SIZE = 8
STEPS = 300
LR = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def build_tiny_indices(
    dataset: QualiSpeechDataset,
) -> list[int]:
    """
    Select a stratified tiny subset from the rows whose audio is
    actually available in wav_part1.zip.

    Returns indices relative to dataset.df.
    """
    selected = []

    for quality, count in SAMPLES_PER_CLASS.items():
        candidates = dataset.df.index[
            dataset.df["Overall quality"] == quality
        ].tolist()

        if len(candidates) < count:
            raise RuntimeError(
                f"Not enough samples for quality={quality}: "
                f"need {count}, found {len(candidates)}"
            )

        # Deterministic selection.
        selected.extend(candidates[:count])

    return selected


def ccc_value(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Diagnostic CCC matching the implementation used by model.py.
    """
    pred = pred.detach().float().view(-1)
    target = target.detach().float().view(-1)

    if pred.numel() < 2:
        return 0.0

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)

    covariance = (
        (pred - pred_mean)
        * (target - target_mean)
    ).mean()

    ccc = (
        2.0 * covariance
        / (
            pred_var
            + target_var
            + (pred_mean - target_mean).pow(2)
            + 1e-8
        )
    )

    return float(ccc.item())


def extract_quality_prediction(output):
    """
    Keep this tolerant to the current SQAModel return structure.

    Expected current possibilities:
        Tensor
        dict containing 'quality'
        tuple/list whose first item is quality
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in (
            "quality",
            "quality_score",
            "score",
        ):
            if key in output:
                return output[key]

        raise KeyError(
            "SQAModel output dict does not contain a recognized "
            "quality key. Found: "
            f"{list(output.keys())}"
        )

    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError(
                "SQAModel returned an empty tuple/list."
            )

        return output[0]

    raise TypeError(
        f"Unsupported SQAModel output type: {type(output)}"
    )


def evaluate_full_dataset(
    model,
    dataset,
):
    """
    Evaluate the ENTIRE tiny dataset.

    This dataset contains exactly 16 samples.

    Returns:
        mse
        mae
        ccc
        loss
        predictions
        targets
        file_ids
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_ids = []

    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=qualispeech_collate,
    )

    with torch.no_grad():
        for batch in eval_loader:
            wav = batch["wav"].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                DEVICE,
                non_blocking=True,
            )

            target = batch["target"].to(
                DEVICE,
                non_blocking=True,
            )

            output = model(
                wav,
                attention_mask=attention_mask,
            )

            pred = extract_quality_prediction(output)

            pred = pred.view(-1)
            target = target.view(-1)

            all_predictions.append(
                pred.cpu()
            )

            all_targets.append(
                target.cpu()
            )

            all_ids.extend(
                batch["file_id"]
            )

    predictions = torch.cat(
        all_predictions
    )

    targets = torch.cat(
        all_targets
    )

    if len(predictions) != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} predictions, "
            f"but got {len(predictions)}."
        )

    mse = torch.mean(
        (predictions - targets) ** 2
    ).item()

    mae = torch.mean(
        torch.abs(predictions - targets)
    ).item()

    ccc = ccc_value(
        predictions,
        targets,
    )

    loss = quality_loss(
        predictions,
        targets,
    ).item()

    return (
        mse,
        mae,
        ccc,
        loss,
        predictions,
        targets,
        all_ids,
    )


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 72)
    print("QualiSpeech REAL TINY OVERFIT")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"CSV:    {CSV_PATH}")
    print(f"ZIP:    {ZIP_PATH}")
    print(
        f"Best checkpoint: {BEST_CHECKPOINT_PATH}"
    )
    print()

    # ---------------------------------------------------------------
    # 1. Full available QualiSpeech dataset
    # ---------------------------------------------------------------

    full_dataset = QualiSpeechDataset(
        csv_path=CSV_PATH,
        zip_path=ZIP_PATH,
    )

    print(
        f"CSV rows:              "
        f"{full_dataset.total_csv_rows}"
    )

    print(
        f"Rows with ZIP audio:   "
        f"{full_dataset.available_rows}"
    )

    print()

    # ---------------------------------------------------------------
    # 2. Stratified 16-sample subset
    # ---------------------------------------------------------------

    tiny_indices = build_tiny_indices(
        full_dataset
    )

    tiny_df = full_dataset.df.iloc[
        tiny_indices
    ].copy()

    print("Tiny subset:")

    print(
        tiny_df[
            ["id", "Overall quality"]
        ].to_string(index=False)
    )

    print()

    print("Tiny target distribution:")

    print(
        tiny_df["Overall quality"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    # Re-create Dataset with only those rows.
    dataset = QualiSpeechDataset(
        csv_path=CSV_PATH,
        zip_path=ZIP_PATH,
        indices=tiny_indices,
    )

    if len(dataset) != 16:
        raise RuntimeError(
            f"Expected tiny dataset to contain 16 samples, "
            f"but got {len(dataset)}."
        )

    # ---------------------------------------------------------------
    # 3. DataLoader
    # ---------------------------------------------------------------

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=qualispeech_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # ---------------------------------------------------------------
    # 4. Model
    # ---------------------------------------------------------------

    print("Loading SQAModel...")

    model = SQAModel().to(DEVICE)

    # The current experiment intentionally trains the QualityHead
    # while keeping WavLM frozen.
    #
    # WavLMBackbone(freeze=True) already disables its parameters.
    # Explicitly freeze anything named backbone as an additional
    # safety check.
    for name, parameter in model.named_parameters():
        if "backbone" in name:
            parameter.requires_grad = False

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("fault_head.")
    ]

    print()
    print("Trainable parameters:")

    for name, parameter in trainable:
        print(
            f"  {name}: "
            f"{parameter.numel():,} parameters"
        )

    total_trainable = sum(
        parameter.numel()
        for _, parameter in trainable
    )

    print(
        f"Total trainable parameters: "
        f"{total_trainable:,}"
    )

    print()

    if total_trainable == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "QualityHead may have been frozen unexpectedly."
        )

    optimizer = Adam(
        [
            parameter
            for _, parameter in trainable
        ],
        lr=LR,
    )

    # ---------------------------------------------------------------
    # 5. Initial forward
    # ---------------------------------------------------------------

    model.eval()

    with torch.no_grad():
        first_batch = next(iter(loader))

        wav = first_batch["wav"].to(
            DEVICE,
            non_blocking=True,
        )

        attention_mask = first_batch[
            "attention_mask"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        target = first_batch["target"].to(
            DEVICE,
            non_blocking=True,
        )

        output = model(
            wav,
            attention_mask=attention_mask,
        )

        pred = extract_quality_prediction(
            output
        )

        pred = pred.view(-1)
        target = target.view(-1)

        initial_mse = torch.mean(
            (pred - target) ** 2
        ).item()

        initial_ccc = ccc_value(
            pred,
            target,
        )

        print(
            f"Initial batch: "
            f"B={wav.shape[0]}, "
            f"T={wav.shape[1]}, "
            f"loss-compatible"
        )

        print(
            f"Initial MSE={initial_mse:.6f}, "
            f"CCC={initial_ccc:.6f}"
        )

        print(
            "Initial predictions:",
            [
                round(float(x), 4)
                for x in pred.cpu()
            ],
        )

        print()

    # ---------------------------------------------------------------
    # 6. Tiny overfit
    # ---------------------------------------------------------------

    model.train()

    # IMPORTANT:
    #
    # Best checkpoint is selected by FULL-16 MSE,
    # NOT by the average training batch loss.
    #
    # This directly matches the acceptance criterion:
    #
    #   FULL 16 samples:
    #       MSE < 0.10
    #       CCC > 0.95
    #
    best_full_mse = float("inf")
    best_full_mae = float("inf")
    best_full_ccc = float("-inf")
    best_full_loss = float("inf")

    best_step = 0
    best_state = None

    for step in range(1, STEPS + 1):
        # -----------------------------------------------------------
        # Training
        # -----------------------------------------------------------

        model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_ccc = 0.0
        batches = 0

        for batch in loader:
            wav = batch["wav"].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            target = batch["target"].to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                wav,
                attention_mask=attention_mask,
            )

            pred = extract_quality_prediction(
                output
            )

            pred = pred.view(-1)
            target = target.view(-1)

            loss = quality_loss(
                pred,
                target,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step}: "
                    f"{loss.item()}"
                )

            loss.backward()

            # Detect broken gradients early.
            bad_gradients = []

            for name, parameter in trainable:
                # FaultHead is currently a stub and is NOT part of
                # quality_loss(). Therefore it is expected to have
                # no gradient in this experiment.
                if name.startswith(
                    "fault_head."
                ):
                    continue

                if parameter.grad is None:
                    bad_gradients.append(
                        f"{name}: None"
                    )

                elif not torch.isfinite(
                    parameter.grad
                ).all():
                    bad_gradients.append(
                        f"{name}: non-finite"
                    )

            if bad_gradients:
                raise RuntimeError(
                    "Invalid gradients:\n"
                    + "\n".join(
                        bad_gradients
                    )
                )

            optimizer.step()

            batch_mse = torch.mean(
                (
                    pred.detach()
                    - target.detach()
                ) ** 2
            ).item()

            batch_ccc = ccc_value(
                pred,
                target,
            )

            total_loss += float(
                loss.item()
            )

            total_mse += batch_mse
            total_ccc += batch_ccc
            batches += 1

        avg_loss = total_loss / batches
        avg_mse = total_mse / batches
        avg_ccc = total_ccc / batches

        # -----------------------------------------------------------
        # FULL 16-SAMPLE EVALUATION
        # -----------------------------------------------------------

        (
            full_mse,
            full_mae,
            full_ccc,
            full_loss,
            _,
            _,
            _,
        ) = evaluate_full_dataset(
            model,
            dataset,
        )

        # -----------------------------------------------------------
        # Track BEST FULL-16 checkpoint
        # -----------------------------------------------------------

        if full_mse < best_full_mse:
            best_full_mse = full_mse
            best_full_mae = full_mae
            best_full_ccc = full_ccc
            best_full_loss = full_loss
            best_step = step

            # Save only trainable parameters in memory.
            # WavLM is frozen.
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in trainable
            }

            # -------------------------------------------------------
            # Save BEST checkpoint to disk.
            # -------------------------------------------------------

            CHECKPOINT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "step": best_step,
                    "mse": best_full_mse,
                    "mae": best_full_mae,
                    "ccc": best_full_ccc,
                    "loss": best_full_loss,

                    # QualityHead only.
                    #
                    # This is the important part for the next
                    # cached-feature verification step.
                    "quality_head_state_dict": {
                        name: tensor.clone()
                        for name, tensor in best_state.items()
                        if name.startswith("quality_head.")
                    },

                    # Keep all trainable parameters as well,
                    # for reproducibility/debugging.
                    "trainable_state_dict": {
                        name: tensor.clone()
                        for name, tensor in best_state.items()
                    },

                    "seed": SEED,
                    "samples_per_class": SAMPLES_PER_CLASS,
                    "batch_size": BATCH_SIZE,
                    "steps": STEPS,
                    "lr": LR,
                },
                BEST_CHECKPOINT_PATH,
            )

            print(
                f"  -> NEW BEST FULL-16 CHECKPOINT: "
                f"step={best_step}, "
                f"MSE={best_full_mse:.6f}, "
                f"CCC={best_full_ccc:.6f}, "
                f"Loss={best_full_loss:.6f}"
            )

            print(
                f"     Saved checkpoint: "
                f"{BEST_CHECKPOINT_PATH}"
            )

        # -----------------------------------------------------------
        # Progress logging
        # -----------------------------------------------------------

        if (
            step == 1
            or step % 25 == 0
            or step == STEPS
        ):
            print(
                f"step {step:3d} | "
                f"train_loss={avg_loss:.6f} | "
                f"train_MSE={avg_mse:.6f} | "
                f"train_CCC={avg_ccc:.6f} | "
                f"full16_MSE={full_mse:.6f} | "
                f"full16_CCC={full_ccc:.6f}"
            )

    # ---------------------------------------------------------------
    # 7. Restore BEST FULL-16 checkpoint
    # ---------------------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint was saved during training."
        )

    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(
                best_state[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )

    print()
    print(
        f"Restored BEST FULL-16 checkpoint: "
        f"step={best_step}"
    )

    print(
        f"Checkpoint: "
        f"{BEST_CHECKPOINT_PATH}"
    )

    print(
        f"Best FULL-16 MSE:  "
        f"{best_full_mse:.6f}"
    )

    print(
        f"Best FULL-16 CCC:  "
        f"{best_full_ccc:.6f}"
    )

    # ---------------------------------------------------------------
    # 8. Final evaluation
    # ---------------------------------------------------------------

    (
        final_mse,
        final_mae,
        final_ccc,
        final_loss,
        predictions,
        targets,
        all_ids,
    ) = evaluate_full_dataset(
        model,
        dataset,
    )

    # ---------------------------------------------------------------
    # 9. Final result
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "FINAL RESULT — BEST FULL-16 CHECKPOINT"
    )
    print("=" * 72)

    print(
        f"Best step:  {best_step}"
    )

    print(
        f"Best loss:  {best_full_loss:.6f}"
    )

    print()

    print(
        f"Loss: {final_loss:.6f}"
    )

    print(
        f"MSE:  {final_mse:.6f}"
    )

    print(
        f"MAE:  {final_mae:.6f}"
    )

    print(
        f"CCC:  {final_ccc:.6f}"
    )

    print()

    print(
        f"{'file_id':40s} "
        f"{'target':>8s} "
        f"{'pred':>8s} "
        f"{'abs_err':>8s}"
    )

    print("-" * 72)

    for file_id, target, pred in zip(
        all_ids,
        targets.tolist(),
        predictions.tolist(),
    ):
        print(
            f"{file_id:40s} "
            f"{target:8.3f} "
            f"{pred:8.3f} "
            f"{abs(pred - target):8.3f}"
        )

    # ---------------------------------------------------------------
    # 10. Acceptance criteria
    # ---------------------------------------------------------------
    #
    # REAL TINY OVERFIT PASS requires:
    #
    #   FULL 16 samples:
    #       MSE < 0.10
    #       CCC > 0.95
    #
    # This is NOT a benchmark.
    #
    # It verifies:
    #
    #   ZIP -> WAV -> WavLM -> QualityHead
    #        -> loss -> backward
    #
    # can memorize the real tiny dataset.
    # ---------------------------------------------------------------

    passed = (
        final_mse < 0.10
        and final_ccc > 0.95
        and torch.isfinite(predictions).all()
    )

    print()

    if passed:
        print(
            ">>> REAL TINY OVERFIT: PASS"
        )

        print(
            "Full 16-sample evaluation satisfies:"
        )

        print(
            f"  MSE < 0.10  : "
            f"{final_mse:.6f}"
        )

        print(
            f"  CCC > 0.95  : "
            f"{final_ccc:.6f}"
        )

        print()

        print(
            "The current WavLM + QualityHead "
            "pipeline passes the real tiny overfit test."
        )

    else:
        print(
            ">>> REAL TINY OVERFIT: FAIL"
        )

        print(
            "Full 16-sample evaluation does not "
            "satisfy the acceptance criteria."
        )

        print(
            "Do NOT start full training yet."
        )

        print(
            "Inspect Dataset -> padding/mask -> "
            "model forward -> gradients -> loss."
        )

    print()

    dataset.close()
    full_dataset.close()

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

