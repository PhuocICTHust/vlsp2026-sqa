from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.dataset import QualiSpeechDataset, qualispeech_collate
from sqa.model import SQAModel


CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
ZIP_PATH = ROOT / "data" / "qualispeech" / "wav_part1.zip"

CACHE_ROOT = ROOT / "data" / "qualispeech" / "wavlm_cache"
CHECKPOINT_PATH = ROOT / "checkpoints" / "real_tiny_best.pt"

MODEL_NAME = "microsoft/wavlm-large"

BATCH_SIZE = 4
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAMPLES_PER_CLASS = {
    1: 3,
    2: 3,
    3: 4,
    4: 3,
    5: 3,
}


# ============================================================================
# HELPERS
# ============================================================================


def build_zip_index(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    index = {}

    for name in names:
        if name.endswith("/"):
            continue

        key = Path(name).name.lower()

        if key not in index:
            index[key] = name

    return index


def build_tiny_indices(
    df: pd.DataFrame,
    zip_index: dict[str, str],
) -> list[int]:

    selected = []

    for quality, count in SAMPLES_PER_CLASS.items():

        candidates = []

        for idx, row in df.iterrows():

            file_id = str(row["id"])
            key = Path(file_id).name.lower()

            if key not in zip_index:
                continue

            target = float(row["Overall quality"])

            if target == quality:
                candidates.append(idx)

        if len(candidates) < count:
            raise RuntimeError(
                f"Not enough samples for quality={quality}"
            )

        selected.extend(candidates[:count])

    return selected


def cache_path_for_file(file_id: str) -> Path:
    return CACHE_ROOT / f"{Path(file_id).stem}.pt"


def load_cache(file_id: str) -> dict:
    path = cache_path_for_file(file_id)

    if not path.exists():
        raise FileNotFoundError(path)

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    hidden = payload["hidden"]
    valid_mask = payload["valid_mask"]

    if hidden.ndim != 2:
        raise RuntimeError(
            f"{file_id}: hidden shape={tuple(hidden.shape)}"
        )

    if valid_mask.ndim != 1:
        raise RuntimeError(
            f"{file_id}: valid_mask shape={tuple(valid_mask.shape)}"
        )

    if hidden.shape[0] != valid_mask.shape[0]:
        raise RuntimeError(
            f"{file_id}: hidden T={hidden.shape[0]} "
            f"mask T={valid_mask.shape[0]}"
        )

    return payload


def load_best_quality_head(model: SQAModel) -> None:

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint["quality_head_state_dict"]

    model.load_state_dict(
        state,
        strict=False,
    )

    print(f"Checkpoint step: {checkpoint['step']}")
    print(f"Checkpoint MSE:  {checkpoint['mse']}")
    print(f"Checkpoint CCC:  {checkpoint['ccc']}")


def tensor_stats(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:

    diff = (a - b).abs()

    print()
    print(name)
    print("-" * 72)
    print(f"Shape A:              {tuple(a.shape)}")
    print(f"Shape B:              {tuple(b.shape)}")
    print(f"Max absolute diff:    {diff.max().item():.12f}")
    print(f"Mean absolute diff:   {diff.mean().item():.12f}")
    print(f"RMSE:                 {torch.sqrt((diff ** 2).mean()).item():.12f}")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    print("=" * 72)
    print("DIAGNOSTIC CACHED QUALITY EQUIVALENCE TEST")
    print("=" * 72)

    print(f"Device: {DEVICE}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(ZIP_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    # ------------------------------------------------------------------------
    # CSV + ZIP
    # ------------------------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    zip_index = build_zip_index(ZIP_PATH)

    tiny_indices = build_tiny_indices(
        df,
        zip_index,
    )

    work_df = df.iloc[tiny_indices].copy()

    print()
    print("Tiny subset:")
    print(
        work_df[
            ["id", "Overall quality"]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------------
    # Check cache
    # ------------------------------------------------------------------------

    missing = []

    for _, row in work_df.iterrows():

        file_id = str(row["id"])
        path = cache_path_for_file(file_id)

        if not path.exists():
            missing.append(file_id)

    if missing:
        raise RuntimeError(
            "Missing cache files:\n"
            + "\n".join(missing)
        )

    print()
    print(f"Cache files found: {len(work_df)}/{len(work_df)}")

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    dataset = QualiSpeechDataset(
        csv_path=CSV_PATH,
        zip_path=ZIP_PATH,
        indices=work_df.index.tolist(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=qualispeech_collate,
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    print()
    print("Loading SQAModel...")

    model = SQAModel(
        model_name=MODEL_NAME,
        freeze_backbone=True,
    ).to(DEVICE)

    model.eval()

    load_best_quality_head(model)

    # Explicitly verify QualityHead is in eval mode.
    model.quality_head.eval()

    # ------------------------------------------------------------------------
    # Run direct path
    # ------------------------------------------------------------------------

    direct_hidden = {}
    direct_pooled = {}
    direct_predictions = {}
    targets = {}

    print()
    print("=" * 72)
    print("PATH A — DIRECT WAV -> WavLM")
    print("=" * 72)

    with torch.no_grad():

        for batch in loader:

            wav = batch["wav"].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                DEVICE,
                non_blocking=True,
            )

            file_ids = batch["file_id"]

            # ---------------------------------------------------------------
            # WavLM only
            # ---------------------------------------------------------------

            hidden = model.backbone(
                wav,
                attention_mask=attention_mask,
            )

            valid_mask = model.backbone.frame_valid_mask(
                attention_mask
            )

            # ---------------------------------------------------------------
            # Process each sample independently
            # ---------------------------------------------------------------

            for i, file_id in enumerate(file_ids):

                file_id = str(file_id)

                valid = valid_mask[i].bool()

                h = hidden[i][valid]

                # AttentionPool with ONLY real frames.
                pooled = model.quality_head.pool(
                    h.unsqueeze(0),
                    valid_mask=None,
                )

                pred = model.quality_head.mlp(
                    pooled
                ).squeeze(-1)

                pred = (
                    torch.sigmoid(pred) * 4.0 + 1.0
                )

                direct_hidden[file_id] = h.detach().cpu()
                direct_pooled[file_id] = pooled.detach().cpu()
                direct_predictions[file_id] = pred.item()

                targets[file_id] = float(
                    batch["target"][i]
                )

            del wav
            del attention_mask
            del hidden
            del valid_mask

    # ------------------------------------------------------------------------
    # Compare against cache
    # ------------------------------------------------------------------------

    cached_predictions = {}

    hidden_max_diffs = []
    pooled_max_diffs = []

    rows = []

    print()
    print("=" * 72)
    print("PER-SAMPLE DIAGNOSTICS")
    print("=" * 72)

    for _, row in work_df.iterrows():

        file_id = str(row["id"])

        payload = load_cache(file_id)

        cached_h = payload["hidden"].float()

        # Cache already contains only real frames.
        cached_pooled = model.quality_head.pool(
            cached_h.unsqueeze(0).to(DEVICE),
            valid_mask=None,
        )

        cached_raw = model.quality_head.mlp(
            cached_pooled
        ).squeeze(-1)

        cached_pred = (
            torch.sigmoid(cached_raw) * 4.0 + 1.0
        )

        cached_predictions[file_id] = cached_pred.item()

        # ---------------------------------------------------------------
        # Hidden comparison
        # ---------------------------------------------------------------

        direct_h = direct_hidden[file_id].float()

        if direct_h.shape != cached_h.shape:
            raise RuntimeError(
                f"{file_id}: hidden shape mismatch: "
                f"direct={tuple(direct_h.shape)} "
                f"cached={tuple(cached_h.shape)}"
            )

        hidden_diff = (
            direct_h - cached_h
        ).abs()

        hidden_max = hidden_diff.max().item()
        hidden_mean = hidden_diff.mean().item()

        hidden_max_diffs.append(hidden_max)

        # ---------------------------------------------------------------
        # Pooled comparison
        # ---------------------------------------------------------------

        direct_p = direct_pooled[file_id].float()
        cached_p = cached_pooled.detach().cpu().float()

        pooled_diff = (
            direct_p - cached_p
        ).abs()

        pooled_max = pooled_diff.max().item()
        pooled_mean = pooled_diff.mean().item()

        pooled_max_diffs.append(pooled_max)

        # ---------------------------------------------------------------
        # Prediction comparison
        # ---------------------------------------------------------------

        direct_pred = direct_predictions[file_id]
        cached_pred_value = cached_predictions[file_id]

        pred_diff = abs(
            direct_pred - cached_pred_value
        )

        rows.append(
            {
                "file_id": file_id,
                "target": targets[file_id],
                "hidden_max_diff": hidden_max,
                "hidden_mean_diff": hidden_mean,
                "pooled_max_diff": pooled_max,
                "pooled_mean_diff": pooled_mean,
                "direct_pred": direct_pred,
                "cached_pred": cached_pred_value,
                "pred_diff": pred_diff,
            }
        )

        print()
        print(file_id)
        print(
            f"  hidden: "
            f"max={hidden_max:.12f} "
            f"mean={hidden_mean:.12f}"
        )
        print(
            f"  pooled: "
            f"max={pooled_max:.12f} "
            f"mean={pooled_mean:.12f}"
        )
        print(
            f"  prediction: "
            f"direct={direct_pred:.9f} "
            f"cached={cached_pred_value:.9f} "
            f"diff={pred_diff:.12f}"
        )

    result_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------------

    direct_values = np.array(
        [
            direct_predictions[str(row["id"])]
            for _, row in work_df.iterrows()
        ],
        dtype=np.float64,
    )

    cached_values = np.array(
        [
            cached_predictions[str(row["id"])]
            for _, row in work_df.iterrows()
        ],
        dtype=np.float64,
    )

    target_values = work_df[
        "Overall quality"
    ].astype(float).to_numpy()

    def mse(pred):
        return float(
            np.mean((pred - target_values) ** 2)
        )

    def mae(pred):
        return float(
            np.mean(np.abs(pred - target_values))
        )

    def pcc(pred):
        return float(
            np.corrcoef(
                pred,
                target_values,
            )[0, 1]
        )

    print()
    print("=" * 72)
    print("AGGREGATE")
    print("=" * 72)

    print(
        f"Direct : "
        f"MSE={mse(direct_values):.9f} "
        f"MAE={mae(direct_values):.9f} "
        f"PCC={pcc(direct_values):.9f}"
    )

    print(
        f"Cached : "
        f"MSE={mse(cached_values):.9f} "
        f"MAE={mae(cached_values):.9f} "
        f"PCC={pcc(cached_values):.9f}"
    )

    prediction_diff = np.abs(
        direct_values - cached_values
    )

    print()
    print("=" * 72)
    print("FINAL DIAGNOSTICS")
    print("=" * 72)

    print(
        f"Hidden max diff:       "
        f"{max(hidden_max_diffs):.12f}"
    )

    print(
        f"Hidden mean max diff:  "
        f"{np.mean(hidden_max_diffs):.12f}"
    )

    print(
        f"Pooled max diff:       "
        f"{max(pooled_max_diffs):.12f}"
    )

    print(
        f"Pooled mean max diff:  "
        f"{np.mean(pooled_max_diffs):.12f}"
    )

    print(
        f"Prediction max diff:   "
        f"{prediction_diff.max():.12f}"
    )

    print(
        f"Prediction mean diff:  "
        f"{prediction_diff.mean():.12f}"
    )

    print()
    print(
        "Interpretation:"
    )
    print(
        "  - If hidden diff is already non-zero, the difference "
        "originates at WavLM extraction."
    )
    print(
        "  - If hidden diff is ~0 but pooled diff is non-zero, "
        "the difference originates at AttentionPool."
    )
    print(
        "  - If hidden + pooled are ~0 but prediction differs, "
        "inspect QualityHead execution."
    )

    # ------------------------------------------------------------------------
    # Save diagnostic table
    # ------------------------------------------------------------------------

    output_path = ROOT / "diagnostic_cached_quality.csv"

    result_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Diagnostic CSV: {output_path}")

    dataset.close()


if __name__ == "__main__":
    main()
