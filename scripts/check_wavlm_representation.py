from __future__ import annotations

import random

import numpy as np
import torch

from sqa.model import SQAModel


SEED = 42
NUM_SAMPLES = 8
AUDIO_LENGTH = 16000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(SEED)

    print("=" * 70)
    print("WAVLM REPRESENTATION DIAGNOSTIC")
    print("=" * 70)

    wav = torch.randn(
        NUM_SAMPLES,
        AUDIO_LENGTH,
    )

    attention_mask = torch.ones(
        NUM_SAMPLES,
        AUDIO_LENGTH,
        dtype=torch.long,
    )

    model = SQAModel()
    model.eval()

    with torch.no_grad():
        h = model.backbone(
            wav,
            attention_mask,
        )

        valid_mask = model.backbone.frame_valid_mask(
            attention_mask,
        )

        pooled = model.quality_head.pool(
            h,
            valid_mask,
        )

    print()
    print("Hidden representation")
    print(f"  shape = {tuple(h.shape)}")

    # Mean/std across samples.
    h_sample_mean = h.mean(dim=(1, 2))
    h_sample_std = h.std(dim=(1, 2))

    print(
        f"  sample mean range = "
        f"{h_sample_mean.min().item():.6f} .. "
        f"{h_sample_mean.max().item():.6f}"
    )

    print(
        f"  sample std range  = "
        f"{h_sample_std.min().item():.6f} .. "
        f"{h_sample_std.max().item():.6f}"
    )

    print()
    print("Pooled representation")
    print(f"  shape = {tuple(pooled.shape)}")

    pooled_sample_std = pooled.std(dim=0)

    print(
        f"  mean feature std across samples = "
        f"{pooled_sample_std.mean().item():.6f}"
    )

    print(
        f"  max feature std across samples = "
        f"{pooled_sample_std.max().item():.6f}"
    )

    # Normalize each embedding and calculate pairwise cosine similarity.
    pooled_norm = torch.nn.functional.normalize(
        pooled,
        dim=1,
    )

    similarity = pooled_norm @ pooled_norm.T

    mask = ~torch.eye(
        NUM_SAMPLES,
        dtype=torch.bool,
    )

    off_diag = similarity[mask]

    print()
    print("Pairwise cosine similarity")
    print(
        f"  mean = {off_diag.mean().item():.6f}"
    )
    print(
        f"  min  = {off_diag.min().item():.6f}"
    )
    print(
        f"  max  = {off_diag.max().item():.6f}"
    )

    # Pairwise Euclidean distance.
    distance = torch.cdist(
        pooled,
        pooled,
        p=2,
    )

    off_diag_dist = distance[mask]

    print()
    print("Pairwise Euclidean distance")
    print(
        f"  mean = {off_diag_dist.mean().item():.6f}"
    )
    print(
        f"  min  = {off_diag_dist.min().item():.6f}"
    )
    print(
        f"  max  = {off_diag_dist.max().item():.6f}"
    )

    print()
    print("Pooled embeddings:")
    for i, emb in enumerate(pooled):
        print(
            f"  sample {i}: "
            f"mean={emb.mean().item():.6f}, "
            f"std={emb.std().item():.6f}, "
            f"norm={emb.norm().item():.6f}"
        )

    print()
    print(">>> DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()