from pathlib import Path

import torch
import torch.nn.functional as F

from sqa.model import SQAModel, quality_loss


def make_batch(lengths, device="cpu"):
    max_len = max(lengths)

    wav = torch.zeros(
        len(lengths),
        max_len,
        dtype=torch.float32,
        device=device,
    )

    attention_mask = torch.zeros(
        len(lengths),
        max_len,
        dtype=torch.long,
        device=device,
    )

    for i, length in enumerate(lengths):
        wav[i, :length] = torch.randn(length, device=device)
        attention_mask[i, :length] = 1

    return wav, attention_mask


def run_case(model, optimizer, lengths, name):
    wav, attention_mask = make_batch(lengths)

    # Fake MOS targets in [1, 5]
    target = torch.tensor(
        [2.0 + i for i in range(len(lengths))],
        dtype=torch.float32,
    )

    optimizer.zero_grad(set_to_none=True)

    quality, fault_probs, h = model(
        wav,
        attention_mask=attention_mask,
    )

    loss = quality_loss(quality, target)

    assert quality.shape == target.shape, (
        f"{name}: quality shape {quality.shape} != target {target.shape}"
    )

    assert h.ndim == 3, f"{name}: hidden state should be [B,T,H]"
    assert fault_probs.shape[:2] == h.shape[:2], (
        f"{name}: fault_probs {fault_probs.shape}, h {h.shape}"
    )

    assert torch.isfinite(loss), f"{name}: loss is not finite"

    loss.backward()

    # Quality head must receive gradients.
    grad_found = False

    for param in model.quality_head.parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                raise RuntimeError(
                    f"{name}: non-finite gradient detected"
                )
            if param.grad.abs().sum() > 0:
                grad_found = True

    assert grad_found, f"{name}: QualityHead received no gradient"

    optimizer.step()

    print(
        f"[PASS] {name}: "
        f"B={len(lengths)}, "
        f"lengths={lengths}, "
        f"frames={h.shape[1]}, "
        f"loss={loss.item():.6f}"
    )


def main():
    torch.manual_seed(42)

    model = SQAModel()

    # WavLM frozen; only heads are optimized.
    trainable = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    assert len(trainable) > 0, "No trainable parameters"

    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-3,
    )

    # 1. Batch size = 1
    run_case(
        model,
        optimizer,
        [16000],
        "batch_size_1",
    )

    # 2. Batch size = 2, same length
    run_case(
        model,
        optimizer,
        [16000, 16000],
        "batch_size_2",
    )

    # 3. Variable-length batch
    run_case(
        model,
        optimizer,
        [12000, 16000],
        "variable_length",
    )

    print("\n>>> TRAINING SMOKE TEST: PASS")


if __name__ == "__main__":
    main()