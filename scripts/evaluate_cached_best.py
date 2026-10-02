from __future__ import annotations

import sys
import random
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.sqa.model import QualityHead


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
CACHE_DIR = ROOT / "data" / "qualispeech" / "wavlm_cache"

OUT_DIR = ROOT / "outputs" / "cached_full"

BEST_CKPT = OUT_DIR / "best_quality_head.pt"

METRICS_PATH = (
    OUT_DIR / "best_checkpoint_metrics_by_source.csv"
)

PREDICTIONS_PATH = (
    OUT_DIR / "best_checkpoint_val_predictions.csv"
)

SEED = 42

BATCH_SIZE = 8
VAL_RATIO = 0.20

NUM_WORKERS = 0
PIN_MEMORY = True

EXPECTED_SAMPLES = 10558
HIDDEN_SIZE = 1024


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# GROUP
# ============================================================

def get_group(sample_id: str) -> str:

    sample_id = str(sample_id).strip()

    if "-" not in sample_id:
        raise ValueError(
            f"Cannot extract group from ID: {sample_id}"
        )

    group = sample_id.split("-", 1)[0].strip()

    if not group:
        raise ValueError(
            f"Empty group extracted from ID: {sample_id}"
        )

    return group


# ============================================================
# SOURCE
# ============================================================

def get_source(group: str) -> str:

    group = str(group).strip().lower()

    if group == "sysnists":
        return "nisqa_sim"

    if group == "sysnistl":
        return "nisqa_live"

    if group == "sys0giga":
        return "gigaspeech"

    if (
        group.startswith("sys")
        and group.endswith("tts")
        and group[3:-3].isdigit()
    ):
        return "recent_tts"

    return "bvcc"


# ============================================================
# DATASET
# ============================================================

class CachedQualiSpeechDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Path,
    ):

        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sample_id = str(
            row["sample_id"]
        )

        path = (
            self.cache_dir
            / f"{sample_id}.pt"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Cache file not found: {path}"
            )

        item = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        hidden = item["hidden"].float()
        valid_mask = item["valid_mask"].bool()

        target_csv = float(
            row["target"]
        )

        target_cache = float(
            item["target"]
        )

        # ----------------------------------------------------
        # TARGET CHECK
        # ----------------------------------------------------

        if abs(
            target_csv - target_cache
        ) > 1e-6:

            raise RuntimeError(
                f"Target mismatch for {sample_id}: "
                f"CSV={target_csv}, "
                f"cache={target_cache}"
            )

        # ----------------------------------------------------
        # FILE ID CHECK
        # ----------------------------------------------------

        cache_file_id = str(
            item["file_id"]
        ).strip()

        cache_file_id = (
            cache_file_id
            .replace(".wav", "")
        )

        if cache_file_id != sample_id:

            raise RuntimeError(
                f"file_id mismatch for {sample_id}: "
                f"cache contains {cache_file_id}"
            )

        # ----------------------------------------------------
        # SHAPE CHECK
        # ----------------------------------------------------

        if hidden.ndim != 2:

            raise RuntimeError(
                f"{sample_id}: hidden must be [T,H], "
                f"got {hidden.shape}"
            )

        if hidden.shape[1] != HIDDEN_SIZE:

            raise RuntimeError(
                f"{sample_id}: expected hidden size "
                f"{HIDDEN_SIZE}, "
                f"got {hidden.shape[1]}"
            )

        if valid_mask.ndim != 1:

            raise RuntimeError(
                f"{sample_id}: valid_mask must be [T], "
                f"got {valid_mask.shape}"
            )

        if len(valid_mask) != len(hidden):

            raise RuntimeError(
                f"{sample_id}: hidden T={len(hidden)}, "
                f"mask T={len(valid_mask)}"
            )

        # ----------------------------------------------------
        # NUMERICAL CHECK
        # ----------------------------------------------------

        if not torch.isfinite(hidden).all():

            raise RuntimeError(
                f"{sample_id}: hidden contains NaN/Inf"
            )

        if not torch.isfinite(
            torch.tensor(target_csv)
        ):

            raise RuntimeError(
                f"{sample_id}: target is NaN/Inf"
            )

        return {
            "hidden": hidden,
            "valid_mask": valid_mask,
            "target": torch.tensor(
                target_csv,
                dtype=torch.float32,
            ),
            "sample_id": sample_id,
        }


# ============================================================
# COLLATE
# ============================================================

def cached_collate(batch):

    hidden_list = [
        x["hidden"]
        for x in batch
    ]

    mask_list = [
        x["valid_mask"]
        for x in batch
    ]

    targets = torch.stack(
        [
            x["target"]
            for x in batch
        ]
    )

    hidden = pad_sequence(
        hidden_list,
        batch_first=True,
        padding_value=0.0,
    )

    valid_mask = pad_sequence(
        mask_list,
        batch_first=True,
        padding_value=False,
    )

    return {
        "hidden": hidden,
        "valid_mask": valid_mask,
        "target": targets,
        "sample_id": [
            x["sample_id"]
            for x in batch
        ],
    }


# ============================================================
# EXACT TRAINING SPLIT
# ============================================================

def make_source_balanced_split(
    df: pd.DataFrame,
):

    # IMPORTANT:
    # This MUST match train_cached_full.py exactly.
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

        source_df = df[
            df["source"] == source
        ].copy()

        if len(source_df) == 0:

            raise RuntimeError(
                f"Source '{source}' has zero samples."
            )

        source_train_parts = []
        source_val_parts = []

        # ----------------------------------------------------
        # STRATIFY BY QUALITY
        # ----------------------------------------------------

        for target_value in sorted(
            source_df["target"].unique()
        ):

            class_df = source_df[
                source_df["target"] == target_value
            ].copy()

            indices = np.arange(
                len(class_df)
            )

            rng.shuffle(indices)

            n_val = int(
                round(
                    len(class_df)
                    * VAL_RATIO
                )
            )

            if len(class_df) >= 2:

                n_val = max(
                    1,
                    min(
                        n_val,
                        len(class_df) - 1,
                    ),
                )

            else:

                n_val = 0

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            source_val_parts.append(
                class_df.iloc[val_indices]
            )

            source_train_parts.append(
                class_df.iloc[train_indices]
            )

        source_train = pd.concat(
            source_train_parts,
            ignore_index=True,
        )

        source_val = pd.concat(
            source_val_parts,
            ignore_index=True,
        )

        # ----------------------------------------------------
        # EXACT PER-SOURCE SHUFFLE
        # ----------------------------------------------------

        source_train = (
            source_train
            .sample(
                frac=1.0,
                random_state=SEED,
            )
            .reset_index(drop=True)
        )

        source_val = (
            source_val
            .sample(
                frac=1.0,
                random_state=SEED,
            )
            .reset_index(drop=True)
        )

        train_parts.append(
            source_train
        )

        val_parts.append(
            source_val
        )

    # --------------------------------------------------------
    # COMBINE SOURCES
    # --------------------------------------------------------

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    val_df = pd.concat(
        val_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # EXACT FINAL SHUFFLE
    # --------------------------------------------------------

    train_df = (
        train_df
        .sample(
            frac=1.0,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .sample(
            frac=1.0,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(train_df) + len(val_df) != len(df):

        raise RuntimeError(
            "Train + validation size mismatch."
        )

    train_ids = set(
        train_df["sample_id"]
    )

    val_ids = set(
        val_df["sample_id"]
    )

    overlap = train_ids & val_ids

    if overlap:

        raise RuntimeError(
            f"Sample leakage detected: "
            f"{sorted(overlap)[:20]}"
        )

    return train_df, val_df


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
):

    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()

    mse = F.mse_loss(
        pred,
        target,
    ).item()

    mae = F.l1_loss(
        pred,
        target,
    ).item()

    # --------------------------------------------------------
    # CCC
    # --------------------------------------------------------

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_var = pred.var(
        unbiased=False
    )

    target_var = target.var(
        unbiased=False
    )

    covariance = (
        (pred - pred_mean)
        * (target - target_mean)
    ).mean()

    denominator = (
        pred_var
        + target_var
        + (
            pred_mean
            - target_mean
        ).pow(2)
    )

    ccc = (
        2.0
        * covariance
        / (denominator + 1e-8)
    ).item()

    # --------------------------------------------------------
    # PCC
    # --------------------------------------------------------

    pred_centered = (
        pred - pred_mean
    )

    target_centered = (
        target - target_mean
    )

    numerator = (
        pred_centered
        * target_centered
    ).sum()

    pcc_denominator = torch.sqrt(
        (
            pred_centered.pow(2).sum()
            * target_centered.pow(2).sum()
        )
        + 1e-12
    )

    pcc = (
        numerator
        / pcc_denominator
    ).item()

    return {
        "pcc": pcc,
        "mse": mse,
        "mae": mae,
        "ccc": ccc,
    }


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    loader,
    device,
):

    model.eval()

    all_pred = []
    all_target = []
    all_ids = []

    for batch in loader:

        hidden = batch[
            "hidden"
        ].to(
            device,
            non_blocking=True,
        )

        valid_mask = batch[
            "valid_mask"
        ].to(
            device,
            non_blocking=True,
        )

        target = batch[
            "target"
        ].to(
            device,
            non_blocking=True,
        )

        pred = model(
            hidden,
            valid_mask,
        )

        all_pred.append(
            pred.cpu()
        )

        all_target.append(
            target.cpu()
        )

        all_ids.extend(
            batch["sample_id"]
        )

    pred = torch.cat(
        all_pred,
        dim=0,
    )

    target = torch.cat(
        all_target,
        dim=0,
    )

    return (
        all_ids,
        pred,
        target,
    )


# ============================================================
# SOURCE METRICS
# ============================================================

def calculate_source_metrics(
    prediction_df: pd.DataFrame,
):

    rows = []

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    pred = torch.tensor(
        prediction_df["pred"].to_numpy(),
        dtype=torch.float32,
    )

    target = torch.tensor(
        prediction_df["target"].to_numpy(),
        dtype=torch.float32,
    )

    metrics = compute_metrics(
        pred,
        target,
    )

    rows.append(
        {
            "source": "OVERALL",
            "n": len(prediction_df),
            **metrics,
        }
    )

    # --------------------------------------------------------
    # PER SOURCE
    # --------------------------------------------------------

    for source in [
        "bvcc",
        "nisqa_sim",
        "gigaspeech",
        "nisqa_live",
        "recent_tts",
    ]:

        sub = prediction_df[
            prediction_df["source"] == source
        ]

        if len(sub) == 0:
            continue

        pred = torch.tensor(
            sub["pred"].to_numpy(),
            dtype=torch.float32,
        )

        target = torch.tensor(
            sub["target"].to_numpy(),
            dtype=torch.float32,
        )

        metrics = compute_metrics(
            pred,
            target,
        )

        rows.append(
            {
                "source": source,
                "n": len(sub),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("EVALUATE BEST CHECKPOINT — EXACT TRAINING SPLIT")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Training: DISABLED")
    print("Optimizer: DISABLED")
    print("Backward: DISABLED")

    # ========================================================
    # LOAD CSV
    # ========================================================

    print()
    print("=" * 70)
    print("LOAD DATASET")
    print("=" * 70)

    df = pd.read_csv(
        CSV_PATH
    )

    if len(df) != EXPECTED_SAMPLES:

        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} rows, "
            f"got {len(df)}"
        )

    df["sample_id"] = (
        df["id"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.wav$",
            "",
            regex=True,
        )
    )

    df["target"] = pd.to_numeric(
        df["Overall quality"],
        errors="raise",
    )

    # ========================================================
    # GROUP / SOURCE
    # ========================================================

    df["group"] = df[
        "sample_id"
    ].map(get_group)

    df["source"] = df[
        "group"
    ].map(get_source)

    # ========================================================
    # SOURCE AUDIT
    # ========================================================

    expected_source_counts = {
        "bvcc": 4546,
        "nisqa_sim": 2883,
        "gigaspeech": 2000,
        "nisqa_live": 529,
        "recent_tts": 600,
    }

    actual_source_counts = (
        df["source"]
        .value_counts()
        .to_dict()
    )

    print()
    print("=" * 70)
    print("SOURCE COMPOSITION")
    print("=" * 70)

    print(
        df["source"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if actual_source_counts != expected_source_counts:

        raise RuntimeError(
            "Source composition mismatch.\n"
            f"Expected: {expected_source_counts}\n"
            f"Actual:   {actual_source_counts}"
        )

    print()
    print("Source composition: PASS")

    # ========================================================
    # CACHE AUDIT
    # ========================================================

    cache_files = sorted(
        CACHE_DIR.glob("*.pt")
    )

    cache_ids = {
        p.stem
        for p in cache_files
    }

    csv_ids = set(
        df["sample_id"]
    )

    csv_only = csv_ids - cache_ids
    cache_only = cache_ids - csv_ids

    print()
    print("=" * 70)
    print("CACHE AUDIT")
    print("=" * 70)

    print(
        f"CSV IDs:       {len(csv_ids)}"
    )

    print(
        f"Cache files:   {len(cache_ids)}"
    )

    print(
        f"CSV only:      {len(csv_only)}"
    )

    print(
        f"Cache only:    {len(cache_only)}"
    )

    if csv_only or cache_only:

        raise RuntimeError(
            "CSV/cache mismatch."
        )

    print("CSV <-> CACHE: PASS")

    # ========================================================
    # EXACT SPLIT
    # ========================================================

    print()
    print("=" * 70)
    print("RECREATE EXACT TRAINING SPLIT")
    print("=" * 70)

    train_df, val_df = (
        make_source_balanced_split(df)
    )

    print(
        f"Train samples:      {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    print(
        f"Validation ratio:   "
        f"{len(val_df) / len(df):.4f}"
    )

    print(
        f"Sample overlap:     "
        f"{len(set(train_df['sample_id']) & set(val_df['sample_id']))}"
    )

    # ========================================================
    # SOURCE SPLIT AUDIT
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATION SOURCE DISTRIBUTION")
    print("=" * 70)

    print(
        val_df["source"]
        .value_counts()
        .reindex(
            [
                "bvcc",
                "nisqa_sim",
                "gigaspeech",
                "nisqa_live",
                "recent_tts",
            ],
            fill_value=0,
        )
        .to_string()
    )

    # ========================================================
    # DATASET / LOADER
    # ========================================================

    val_dataset = (
        CachedQualiSpeechDataset(
            val_df,
            CACHE_DIR,
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
        collate_fn=cached_collate,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOAD BEST CHECKPOINT")
    print("=" * 70)

    model = QualityHead(
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    checkpoint = torch.load(
        BEST_CKPT,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"Checkpoint: {BEST_CKPT}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint val PCC: "
        f"{checkpoint['val_pcc']:.8f}"
    )

    print(
        f"Checkpoint val MSE: "
        f"{checkpoint['val_mse']:.8f}"
    )

    print(
        f"Checkpoint val CCC: "
        f"{checkpoint['val_ccc']:.8f}"
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    print()
    print("=" * 70)
    print("BEST-CHECKPOINT INFERENCE")
    print("=" * 70)

    ids, pred, target = run_inference(
        model,
        val_loader,
        device,
    )

    # ========================================================
    # PREDICTION DATAFRAME
    # ========================================================

    prediction_df = pd.DataFrame(
        {
            "sample_id": ids,
            "target": target.numpy(),
            "pred": pred.numpy(),
        }
    )

    source_lookup = (
        val_df[
            [
                "sample_id",
                "source",
            ]
        ]
        .drop_duplicates("sample_id")
        .set_index("sample_id")[
            "source"
        ]
        .to_dict()
    )

    prediction_df["source"] = (
        prediction_df["sample_id"]
        .map(source_lookup)
    )

    if prediction_df["source"].isna().any():

        raise RuntimeError(
            "Some validation IDs could not "
            "be mapped to source."
        )

    # ========================================================
    # CRITICAL CHECK:
    # SAME VALIDATION SET SIZE
    # ========================================================

    if len(prediction_df) != len(val_df):

        raise RuntimeError(
            "Inference sample count does not "
            "match validation sample count."
        )

    if set(prediction_df["sample_id"]) != set(
        val_df["sample_id"]
    ):

        raise RuntimeError(
            "Inference IDs do not match "
            "recreated validation IDs."
        )

    # ========================================================
    # METRICS
    # ========================================================

    metrics_df = calculate_source_metrics(
        prediction_df
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL + PER-SOURCE METRICS")
    print("=" * 70)

    print(
        metrics_df[
            [
                "source",
                "n",
                "pcc",
                "mse",
                "mae",
                "ccc",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    # ========================================================
    # VERIFY OVERALL AGAINST CHECKPOINT
    # ========================================================

    overall = metrics_df[
        metrics_df["source"] == "OVERALL"
    ].iloc[0]

    print()
    print("=" * 70)
    print("CHECKPOINT CONSISTENCY CHECK")
    print("=" * 70)

    pcc_diff = abs(
        overall["pcc"]
        - float(checkpoint["val_pcc"])
    )

    mse_diff = abs(
        overall["mse"]
        - float(checkpoint["val_mse"])
    )

    ccc_diff = abs(
        overall["ccc"]
        - float(checkpoint["val_ccc"])
    )

    print(
        f"PCC checkpoint : "
        f"{checkpoint['val_pcc']:.10f}"
    )

    print(
        f"PCC evaluated  : "
        f"{overall['pcc']:.10f}"
    )

    print(
        f"PCC abs diff    : "
        f"{pcc_diff:.12f}"
    )

    print()

    print(
        f"MSE checkpoint : "
        f"{checkpoint['val_mse']:.10f}"
    )

    print(
        f"MSE evaluated  : "
        f"{overall['mse']:.10f}"
    )

    print(
        f"MSE abs diff    : "
        f"{mse_diff:.12f}"
    )

    print()

    print(
        f"CCC checkpoint : "
        f"{checkpoint['val_ccc']:.10f}"
    )

    print(
        f"CCC evaluated  : "
        f"{overall['ccc']:.10f}"
    )

    print(
        f"CCC abs diff    : "
        f"{ccc_diff:.12f}"
    )

    # Very small floating-point differences are acceptable.
    tolerance = 1e-6

    if (
        pcc_diff > tolerance
        or mse_diff > tolerance
        or ccc_diff > tolerance
    ):

        raise RuntimeError(
            "\nEvaluation does NOT reproduce "
            "the checkpoint metrics.\n"
            "The validation split or inference "
            "pipeline is inconsistent."
        )

    print()
    print(
        ">>> CHECKPOINT CONSISTENCY: PASS"
    )

    # ========================================================
    # SOURCE STATISTICS
    # ========================================================

    print()
    print("=" * 70)
    print("SOURCE TARGET / PREDICTION STATISTICS")
    print("=" * 70)

    for source in [
        "OVERALL",
        "bvcc",
        "nisqa_sim",
        "gigaspeech",
        "nisqa_live",
        "recent_tts",
    ]:

        if source == "OVERALL":

            sub = prediction_df

        else:

            sub = prediction_df[
                prediction_df["source"] == source
            ]

        print()
        print(
            f"[{source}]"
        )

        print(
            f"  n           = {len(sub)}"
        )

        print(
            f"  target_mean = "
            f"{sub['target'].mean():.8f}"
        )

        print(
            f"  target_std  = "
            f"{sub['target'].std(ddof=0):.8f}"
        )

        print(
            f"  pred_mean   = "
            f"{sub['pred'].mean():.8f}"
        )

        print(
            f"  pred_std    = "
            f"{sub['pred'].std(ddof=0):.8f}"
        )

        print(
            f"  pred_min    = "
            f"{sub['pred'].min():.8f}"
        )

        print(
            f"  pred_max    = "
            f"{sub['pred'].max():.8f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Validation samples: "
        f"{len(val_df)}"
    )

    print(
        f"Overall PCC: "
        f"{overall['pcc']:.8f}"
    )

    print(
        f"Overall MSE: "
        f"{overall['mse']:.8f}"
    )

    print(
        f"Overall MAE: "
        f"{overall['mae']:.8f}"
    )

    print(
        f"Overall CCC: "
        f"{overall['ccc']:.8f}"
    )

    print()
    print(
        f"Metrics: "
        f"{METRICS_PATH}"
    )

    print(
        f"Predictions: "
        f"{PREDICTIONS_PATH}"
    )

    print()
    print(
        ">>> BEST CHECKPOINT EVALUATION: COMPLETE"
    )


if __name__ == "__main__":
    main()
