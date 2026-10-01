from __future__ import annotations

import random

import numpy as np
import torch

from sqa.model import SQAModel, quality_loss


SEED = 42
NUM_SAMPLES = 8
BATCH_SIZE = 4
STEPS = 300
LR = 1e-3

# 1-second dummy audio.
AUDIO_LENGTH = 16000

# Fixed MOS targets in [1, 5].
TARGETS = torch.tensor(
    [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    dtype=torch.float32,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset():
    """
    Create a fixed synthetic dataset.

    The audio is generated once and reused for every training step.
    This is intentional: tiny-overfit tests memorization, not
    generalization.
    """
    generator = torch.Generator().manual_seed(SEED)

    wav = torch.randn(
        NUM_SAMPLES,
        AUDIO_LENGTH,
        generator=generator,
        dtype=torch.float32,
    )

    attention_mask = torch.ones(
        NUM_SAMPLES,
        AUDIO_LENGTH,
        dtype=torch.long,
    )

    return wav, attention_mask, TARGETS.clone()


def ccc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)

    covariance = (
        (pred - pred_mean)
        * (target - target_mean)
    ).mean()

    denominator = (
        pred_var
        + target_var
        + (pred_mean - target_mean).pow(2)
    )

    return (2.0 * covariance) / (denominator + 1e-8)


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def run_epoch(
    model,
    optimizer,
    wav,
    attention_mask,
    targets,
    step,
):
    model.train()

    indices = torch.tensor(
        [(step * BATCH_SIZE + i) % NUM_SAMPLES
         for i in range(BATCH_SIZE)]
    )

    batch_wav = wav[indices]
    batch_mask = attention_mask[indices]
    batch_target = targets[indices]

    optimizer.zero_grad(set_to_none=True)

    quality, _, _ = model(
        batch_wav,
        attention_mask=batch_mask,
    )

    loss = quality_loss(
        quality,
        batch_target,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at step {step}: {loss.item()}"
        )

    loss.backward()

    # Check gradients.
    grad_norm_sq = 0.0

    for param in model.parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            continue

        if not torch.isfinite(param.grad).all():
            raise RuntimeError(
                f"Non-finite gradient at step {step}"
            )

        grad_norm_sq += param.grad.detach().pow(2).sum().item()

    grad_norm = grad_norm_sq ** 0.5

    if grad_norm == 0.0:
        raise RuntimeError(
            f"Zero gradient at step {step}"
        )

    optimizer.step()

    with torch.no_grad():
        batch_mse = mse(quality, batch_target)
        batch_ccc = ccc(quality, batch_target)

    return (
        loss.item(),
        batch_mse.item(),
        batch_ccc.item(),
        grad_norm,
        quality.detach().cpu(),
    )


def evaluate(
    model,
    wav,
    attention_mask,
    targets,
):
    model.eval()

    with torch.no_grad():
        quality, _, _ = model(
            wav,
            attention_mask=attention_mask,
        )

        loss = quality_loss(
            quality,
            targets,
        )

        metric_mse = mse(
            quality,
            targets,
        )

        metric_ccc = ccc(
            quality,
            targets,
        )

    return (
        loss.item(),
        metric_mse.item(),
        metric_ccc.item(),
        quality.cpu(),
    )


def main():
    set_seed(SEED)

    print("=" * 70)
    print("TINY OVERFIT TEST")
    print("=" * 70)

    print(f"Samples     : {NUM_SAMPLES}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Steps       : {STEPS}")
    print(f"Learning rate: {LR}")
    print(f"Audio length: {AUDIO_LENGTH} samples")
    print()

    wav, attention_mask, targets = make_dataset()

    print("Creating SQAModel...")
    model = SQAModel()

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    trainable_count = sum(
        p.numel()
        for p in trainable_params
    )

    frozen_count = sum(
        p.numel()
        for p in model.parameters()
        if not p.requires_grad
    )

    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Frozen parameters   : {frozen_count:,}")
    print()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LR,
        weight_decay=0.0,
    )

    # ---------------------------------------------------------
    # Initial evaluation
    # ---------------------------------------------------------

    initial_loss, initial_mse, initial_ccc, initial_pred = evaluate(
        model,
        wav,
        attention_mask,
        targets,
    )

    print("Initial:")
    print(f"  loss = {initial_loss:.6f}")
    print(f"  MSE  = {initial_mse:.6f}")
    print(f"  CCC  = {initial_ccc:.6f}")
    print(
        "  pred =",
        [round(x, 3) for x in initial_pred.tolist()],
    )
    print()

    best_loss = initial_loss
    best_step = 0

    history = []

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print("Training...")

    for step in range(1, STEPS + 1):

        loss, batch_mse, batch_ccc, grad_norm, _ = run_epoch(
            model,
            optimizer,
            wav,
            attention_mask,
            targets,
            step,
        )

        history.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_step = step

        if step == 1 or step % 10 == 0:
            eval_loss, eval_mse, eval_ccc, eval_pred = evaluate(
                model,
                wav,
                attention_mask,
                targets,
            )

            print(
                f"step={step:03d} "
                f"loss={eval_loss:.6f} "
                f"MSE={eval_mse:.6f} "
                f"CCC={eval_ccc:.6f} "
                f"grad={grad_norm:.4e}"
            )

    # ---------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------

    final_loss, final_mse, final_ccc, final_pred = evaluate(
        model,
        wav,
        attention_mask,
        targets,
    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Initial loss : {initial_loss:.6f}")
    print(f"Final loss   : {final_loss:.6f}")
    print(f"Best loss    : {best_loss:.6f}")
    print(f"Best step    : {best_step}")
    print(f"Final MSE    : {final_mse:.6f}")
    print(f"Final CCC    : {final_ccc:.6f}")

    print()
    print("Targets:")
    print(
        " ",
        [round(x, 3) for x in targets.tolist()],
    )

    print("Predictions:")
    print(
        " ",
        [round(x, 3) for x in final_pred.tolist()],
    )

    print()

    # ---------------------------------------------------------
    # Acceptance criteria
    # ---------------------------------------------------------

    loss_reduced = final_loss < initial_loss * 0.20
    mse_good = final_mse < 0.10
    ccc_good = final_ccc > 0.90

    print("Checks:")
    print(
        f"  loss reduced >80% : "
        f"{'PASS' if loss_reduced else 'FAIL'}"
    )
    print(
        f"  final MSE < 0.10  : "
        f"{'PASS' if mse_good else 'FAIL'}"
    )
    print(
        f"  final CCC > 0.90  : "
        f"{'PASS' if ccc_good else 'FAIL'}"
    )

    if loss_reduced and mse_good and ccc_good:
        print()
        print(">>> TINY OVERFIT: PASS")
        print(
            "QualityHead + loss + backward + optimizer "
            "can memorize the fixed dataset."
        )
    else:
        print()
        print(">>> TINY OVERFIT: FAIL")
        print(
            "The model did not sufficiently memorize "
            "the fixed dataset."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()