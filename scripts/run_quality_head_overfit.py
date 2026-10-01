from __future__ import annotations

import random

import numpy as np
import torch

from sqa.model import QualityHead, quality_loss


SEED = 42
NUM_SAMPLES = 8
FRAMES = 49
HIDDEN_SIZE = 1024
STEPS = 500
LR = 1e-3


TARGETS = torch.tensor(
    [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    dtype=torch.float32,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ccc(pred, target):
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

    return (
        2.0 * covariance
        / (denominator + 1e-8)
    )


def main():
    set_seed(SEED)

    print("=" * 70)
    print("QUALITY HEAD TINY OVERFIT")
    print("=" * 70)

    # Deliberately create clearly different hidden representations.
    h = torch.randn(
        NUM_SAMPLES,
        FRAMES,
        HIDDEN_SIZE,
    )

    # Add a strong sample-specific signal.
    for i in range(NUM_SAMPLES):
        h[i] += float(i) * 0.5

    model = QualityHead(
        hidden_size=HIDDEN_SIZE,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    model.train()

    with torch.no_grad():
        pred = model(h)

        initial_loss = quality_loss(
            pred,
            TARGETS,
        ).item()

        initial_mse = (
            (pred - TARGETS) ** 2
        ).mean().item()

        initial_ccc = ccc(
            pred,
            TARGETS,
        ).item()

    print(f"Initial loss = {initial_loss:.6f}")
    print(f"Initial MSE  = {initial_mse:.6f}")
    print(f"Initial CCC  = {initial_ccc:.6f}")
    print(
        "Initial pred =",
        [round(x, 3) for x in pred.tolist()],
    )

    best_loss = float("inf")

    for step in range(1, STEPS + 1):

        optimizer.zero_grad(set_to_none=True)

        pred = model(h)

        loss = quality_loss(
            pred,
            TARGETS,
        )

        loss.backward()

        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if step == 1 or step % 50 == 0:
            with torch.no_grad():
                current_mse = (
                    (pred - TARGETS) ** 2
                ).mean().item()

                current_ccc = ccc(
                    pred,
                    TARGETS,
                ).item()

            print(
                f"step={step:03d} "
                f"loss={loss.item():.6f} "
                f"MSE={current_mse:.6f} "
                f"CCC={current_ccc:.6f}"
            )

    model.eval()

    with torch.no_grad():
        final_pred = model(h)

        final_loss = quality_loss(
            final_pred,
            TARGETS,
        ).item()

        final_mse = (
            (final_pred - TARGETS) ** 2
        ).mean().item()

        final_ccc = ccc(
            final_pred,
            TARGETS,
        ).item()

    print()
    print("=" * 70)
    print("FINAL")
    print("=" * 70)

    print(f"Best loss  = {best_loss:.6f}")
    print(f"Final loss = {final_loss:.6f}")
    print(f"Final MSE  = {final_mse:.6f}")
    print(f"Final CCC  = {final_ccc:.6f}")

    print()
    print("Targets:")
    print(
        [round(x, 3) for x in TARGETS.tolist()]
    )

    print("Predictions:")
    print(
        [round(x, 3) for x in final_pred.tolist()]
    )

    passed = (
        final_mse < 0.05
        and final_ccc > 0.95
    )

    print()

    if passed:
        print(">>> QUALITY HEAD OVERFIT: PASS")
    else:
        print(">>> QUALITY HEAD OVERFIT: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()