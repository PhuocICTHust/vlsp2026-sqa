from __future__ import annotations

import sys
import random
from pathlib import Path

# PROJECT ROOT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.sqa.model import QualityHead, quality_loss


# CONFIG

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"

# IMPORTANT:
# HuBERT cache is completely separate from WavLM cache.

CACHE_DIR = ROOT / "data" / "qualispeech" / "hubert_cache"

# IMPORTANT:
# Separate output directory so WavLM results are NOT overwritten.

OUT_DIR = ROOT / "outputs" / "hubert_cached_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = OUT_DIR / "best_quality_head.pt"
HISTORY_PATH = OUT_DIR / "training_history.csv"

SEED = 42

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3

VAL_RATIO = 0.20

NUM_WORKERS = 0
PIN_MEMORY = True

EXPECTED_SAMPLES = 10558

# HuBERT-base hidden size
# facebook/hubert-base-ls960 -> 768

HIDDEN_SIZE = 768


# REPRODUCIBILITY

def seed_everything(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# GROUP

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


# SOURCE

def get_source(group: str):
    """
    Map QualiSpeech system/group IDs to source categories.

    Based on the official QualiSpeech dataset composition:

        sysnists          -> NISQA SIM
        sysnistl          -> NISQA LIVE
        sys0giga          -> GigaSpeech
        sysXXtts          -> Recent TTS
        everything else   -> BVCC

    Important:
    This split is intentionally SAMPLE-LEVEL within each source.
    We do NOT keep groups disjoint between train/validation.
    """

    group = str(group).strip().lower()

    if group == "sysnists":
        return "nisqa_sim"

    if group == "sysnistl":
        return "nisqa_live"

    if group == "sys0giga":
        return "gigaspeech"

    # Recent TTS systems:
    # sys01tts, sys02tts, ..., sys10tts
    if (
        group.startswith("sys")
        and group.endswith("tts")
        and group[3:-3].isdigit()
    ):
        return "recent_tts"

    # Remaining groups correspond to BVCC.
    return "bvcc"


# CACHE DISCOVERY

def discover_cache_ids(cache_dir: Path) -> set[str]:

    cache_files = sorted(
        cache_dir.glob("*.pt")
    )

    ids = {
        p.stem
        for p in cache_files
    }

    print(
        f"Cache directory: {cache_dir}"
    )

    print(
        f"Cache files: {len(cache_files)}"
    )

    return ids


# CSV <-> CACHE AUDIT

def audit_csv_and_cache(
    df: pd.DataFrame,
    cache_ids: set[str],
):

    csv_ids = set(df["sample_id"])

    csv_only = csv_ids - cache_ids
    cache_only = cache_ids - csv_ids
    intersection = csv_ids & cache_ids

    print()
    print("=" * 70)
    print("CSV <-> CACHE AUDIT")
    print("=" * 70)

    print(
        f"CSV rows:              {len(df)}"
    )

    print(
        f"Unique CSV IDs:        {len(csv_ids)}"
    )

    print(
        f"Cache files:           {len(cache_ids)}"
    )

    print(
        f"CSV ∩ CACHE:           {len(intersection)}"
    )

    print(
        f"CSV only:              {len(csv_only)}"
    )

    print(
        f"CACHE only:            {len(cache_only)}"
    )

    if csv_only:

        print("\nFirst CSV-only IDs:")

        for x in sorted(csv_only)[:20]:
            print("  ", x)

    if cache_only:

        print("\nFirst CACHE-only IDs:")

        for x in sorted(cache_only)[:20]:
            print("  ", x)

    if csv_only or cache_only:

        raise RuntimeError(
            "CSV/cache ID mismatch. "
            "Refusing to start full training."
        )

    if len(df) != EXPECTED_SAMPLES:

        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} CSV rows, "
            f"got {len(df)}."
        )

    if len(cache_ids) != EXPECTED_SAMPLES:

        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} cache files, "
            f"got {len(cache_ids)}."
        )

    print()
    print("CSV <-> CACHE: PASS")


# DATASET

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
            row["Overall quality"]
        )

        target_cache = float(
            item["target"]
        )

        # TARGET CHECK

        if abs(
            target_csv - target_cache
        ) > 1e-6:

            raise RuntimeError(
                f"Target mismatch for {sample_id}: "
                f"CSV={target_csv}, "
                f"cache={target_cache}"
            )

        # FILE ID CHECK

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

        # SHAPE CHECK

        if hidden.ndim != 2:

            raise RuntimeError(
                f"{sample_id}: "
                f"hidden must be [T,H], "
                f"got {hidden.shape}"
            )

        if hidden.shape[1] != HIDDEN_SIZE:

            raise RuntimeError(
                f"{sample_id}: "
                f"expected hidden size "
                f"{HIDDEN_SIZE}, "
                f"got {hidden.shape[1]}"
            )

        if valid_mask.ndim != 1:

            raise RuntimeError(
                f"{sample_id}: "
                f"valid_mask must be [T], "
                f"got {valid_mask.shape}"
            )

        if len(valid_mask) != len(hidden):

            raise RuntimeError(
                f"{sample_id}: "
                f"hidden T={len(hidden)}, "
                f"mask T={len(valid_mask)}"
            )

        # NUMERICAL CHECK

        if not torch.isfinite(hidden).all():

            raise RuntimeError(
                f"{sample_id}: "
                f"hidden contains NaN/Inf"
            )

        if not torch.isfinite(
            torch.tensor(target_csv)
        ):

            raise RuntimeError(
                f"{sample_id}: "
                f"target is NaN/Inf"
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


# COLLATE

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


# SOURCE-BALANCED IID SPLIT

def make_source_balanced_split(
    df: pd.DataFrame,
):

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

    print()
    print("=" * 70)
    print("SOURCE-BALANCED IID SPLIT")
    print("=" * 70)

    for source in source_order:

        source_df = df[
            df["source"] == source
        ].copy()

        if len(source_df) == 0:

            raise RuntimeError(
                f"Source '{source}' has "
                f"zero samples."
            )

        source_train_parts = []
        source_val_parts = []

        # STRATIFY BY QUALITY

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

            # Ensure both sets get samples whenever
            # a class has enough samples.
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
                class_df.iloc[
                    val_indices
                ]
            )

            source_train_parts.append(
                class_df.iloc[
                    train_indices
                ]
            )

        source_train = pd.concat(
            source_train_parts,
            ignore_index=True,
        )

        source_val = pd.concat(
            source_val_parts,
            ignore_index=True,
        )

        # Shuffle each source partition.
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

        print(
            f"{source:15s} | "
            f"total={len(source_df):4d} | "
            f"train={len(source_train):4d} | "
            f"val={len(source_val):4d}"
        )

    # COMBINE SOURCES

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    val_df = pd.concat(
        val_parts,
        ignore_index=True,
    )

    # Final shuffle.
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

    # BASIC VALIDATION

    if len(train_df) == 0:

        raise RuntimeError(
            "Train split is empty."
        )

    if len(val_df) == 0:

        raise RuntimeError(
            "Validation split is empty."
        )

    if (
        len(train_df)
        + len(val_df)
        != len(df)
    ):

        raise RuntimeError(
            "Train + validation size "
            "does not equal full dataset."
        )

    # ID LEAKAGE CHECK

    train_ids = set(
        train_df["sample_id"]
    )

    val_ids = set(
        val_df["sample_id"]
    )

    overlap = (
        train_ids & val_ids
    )

    if overlap:

        raise RuntimeError(
            "Sample leakage detected: "
            f"{sorted(overlap)[:20]}"
        )

    # SOURCE CHECK

    for source in source_order:

        train_count = int(
            (train_df["source"] == source).sum()
        )

        val_count = int(
            (val_df["source"] == source).sum()
        )

        if train_count == 0 or val_count == 0:

            raise RuntimeError(
                f"Source '{source}' does not "
                f"exist in both train and validation."
            )

    print()
    print(
        f"Total samples:      {len(df)}"
    )

    print(
        f"Train samples:      {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    print(
        f"Train ratio:        "
        f"{len(train_df) / len(df):.4f}"
    )

    print(
        f"Validation ratio:   "
        f"{len(val_df) / len(df):.4f}"
    )

    print(
        f"Sample overlap:     {len(overlap)}"
    )

    return (
        train_df,
        val_df,
    )


# SPLIT AUDIT

def print_split_audit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("SOURCE DISTRIBUTION")
    print("=" * 70)

    source_table = pd.DataFrame(
        {
            "train": train_df[
                "source"
            ].value_counts(),
            "val": val_df[
                "source"
            ].value_counts(),
        }
    ).fillna(0).astype(int)

    source_table["total"] = (
        source_table["train"]
        + source_table["val"]
    )

    source_table["val_ratio"] = (
        source_table["val"]
        / source_table["total"]
    )

    print(
        source_table.to_string()
    )

    # TARGET DISTRIBUTION

    print()
    print("=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    full_counts = (
        train_df["target"]
        .value_counts()
        .add(
            val_df["target"]
            .value_counts(),
            fill_value=0,
        )
        .sort_index()
        .astype(int)
    )

    train_counts = (
        train_df["target"]
        .value_counts()
        .sort_index()
        .reindex(
            full_counts.index,
            fill_value=0,
        )
    )

    val_counts = (
        val_df["target"]
        .value_counts()
        .sort_index()
        .reindex(
            full_counts.index,
            fill_value=0,
        )
    )

    target_table = pd.DataFrame(
        {
            "full": full_counts,
            "train": train_counts,
            "val": val_counts,
        }
    )

    # Correct full percentage calculation.
    target_table["full_%"] = (
        target_table["full"]
        / (len(train_df) + len(val_df))
    ) * 100

    target_table["train_%"] = (
        target_table["train"]
        / len(train_df)
    ) * 100

    target_table["val_%"] = (
        target_table["val"]
        / len(val_df)
    ) * 100

    print(
        target_table.round(2).to_string()
    )

    # TARGET BY SOURCE

    print()
    print("=" * 70)
    print("TARGET DISTRIBUTION BY SOURCE")
    print("=" * 70)

    for source in sorted(
        train_df["source"].unique()
    ):

        train_source = train_df[
            train_df["source"] == source
        ]

        val_source = val_df[
            val_df["source"] == source
        ]

        print()
        print(
            f"[{source}]"
        )

        print(
            f"  train n={len(train_source)}, "
            f"mean={train_source['target'].mean():.4f}"
        )

        print(
            f"  val   n={len(val_source)}, "
            f"mean={val_source['target'].mean():.4f}"
        )

        print("  train:")

        print(
            train_source["target"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("  val:")

        print(
            val_source["target"]
            .value_counts()
            .sort_index()
            .to_string()
        )


# METRICS

@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
):

    pred = (
        pred.reshape(-1)
        .float()
    )

    target = (
        target.reshape(-1)
        .float()
    )

    mse = F.mse_loss(
        pred,
        target,
    ).item()

    mae = F.l1_loss(
        pred,
        target,
    ).item()

    # CCC

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

    # PCC
    #
    # Official quality metric for the shared task.
    #

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
        "mse": mse,
        "mae": mae,
        "ccc": ccc,
        "pcc": pcc,
    }


# EVALUATION

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    all_pred = []
    all_target = []

    # FORWARD OVER ENTIRE DATASET

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
            pred.detach().cpu()
        )

        all_target.append(
            target.detach().cpu()
        )

    # GLOBAL CONCATENATION

    pred = torch.cat(
        all_pred,
        dim=0,
    )

    target = torch.cat(
        all_target,
        dim=0,
    )

    # GLOBAL LOSS

    loss = quality_loss(
        pred,
        target,
    ).item()

    # GLOBAL METRICS

    metrics = compute_metrics(
        pred,
        target,
    )

    metrics["loss"] = loss

    return metrics


# MAIN

def main():

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("FULL CACHED HuBERT TRAINING")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "HuBERT forward: DISABLED"
    )

    print(
        "Training: QualityHead ONLY"
    )

    print(
        "Backbone: facebook/hubert-base-ls960"
    )

    print(
        "Split: SOURCE-BALANCED IID 80/20"
    )

    print(
        "Best checkpoint metric: VAL PCC"
    )

    print(
        f"Expected samples: "
        f"{EXPECTED_SAMPLES}"
    )

    # BUILD DATASET

    print()
    print("=" * 70)
    print("BUILD DATASET")
    print("=" * 70)

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"CSV rows: {len(df)}"
    )

    # REQUIRED COLUMNS

    if "id" not in df.columns:

        raise RuntimeError(
            "CSV does not contain 'id' column."
        )

    if "Overall quality" not in df.columns:

        raise RuntimeError(
            "CSV does not contain "
            "'Overall quality' column."
        )

    # NORMALIZE SAMPLE ID

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

    # TARGET

    df["target"] = pd.to_numeric(
        df["Overall quality"],
        errors="raise",
    )

    # CACHE AUDIT

    cache_ids = discover_cache_ids(
        CACHE_DIR
    )

    audit_csv_and_cache(
        df,
        cache_ids,
    )

    # GROUP

    df["group"] = df[
        "sample_id"
    ].map(get_group)

    # SOURCE

    df["source"] = df[
        "group"
    ].map(get_source)

    # SOURCE COUNTS

    print()
    print("=" * 70)
    print("FULL SOURCE COMPOSITION")
    print("=" * 70)

    source_counts = (
        df["source"]
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
    )

    print(
        source_counts.to_string()
    )

    # EXPECTED OFFICIAL COMPOSITION

    expected_source_counts = {
        "bvcc": 4546,
        "nisqa_sim": 2883,
        "gigaspeech": 2000,
        "nisqa_live": 529,
        "recent_tts": 600,
    }

    for source, expected in (
        expected_source_counts.items()
    ):

        actual = int(
            source_counts[source]
        )

        if actual != expected:

            raise RuntimeError(
                f"Unexpected {source} count: "
                f"got {actual}, "
                f"expected {expected}"
            )

    print()
    print(
        "Source composition: PASS"
    )

    # FULL TARGET DISTRIBUTION

    print()
    print("=" * 70)
    print("FULL TARGET DISTRIBUTION")
    print("=" * 70)

    print(
        df["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # SOURCE-BALANCED IID SPLIT

    (
        train_df,
        val_df,
    ) = make_source_balanced_split(
        df
    )

    # SPLIT AUDIT

    print_split_audit(
        train_df,
        val_df,
    )

    # DATASETS

    train_dataset = (
        CachedQualiSpeechDataset(
            train_df,
            CACHE_DIR,
        )
    )

    val_dataset = (
        CachedQualiSpeechDataset(
            val_df,
            CACHE_DIR,
        )
    )

    # DATALOADERS

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
        collate_fn=cached_collate,
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

    # MODEL

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    model = QualityHead(
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    trainable_params = []

    for name, param in model.named_parameters():

        if param.requires_grad:

            trainable_params.append(
                (
                    name,
                    param.numel(),
                )
            )

    print(
        "Trainable parameters:"
    )

    for name, n in trainable_params:

        print(
            f"  {name}: {n:,}"
        )

    total_params = sum(
        n
        for _, n in trainable_params
    )

    print(
        f"\nTotal trainable parameters: "
        f"{total_params:,}"
    )

    # HuBERT 768 -> QualityHead
    #
    # AttentionPool:
    #   768 + 1 = 769
    #
    # MLP:
    #   768 * 256 + 256 = 196,864
    #   256 * 1 + 1     = 257
    #
    # Total:
    #   769 + 196,864 + 257 = 197,890

    expected_params = 197_890

    if total_params != expected_params:

        raise RuntimeError(
            "Unexpected QualityHead "
            f"parameter count: "
            f"{total_params}, "
            f"expected "
            f"{expected_params}"
        )

    # OPTIMIZER

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    # INITIAL VALIDATION

    initial_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    print()
    print("=" * 70)
    print("INITIAL VALIDATION")
    print("=" * 70)

    print(
        f"Loss: "
        f"{initial_metrics['loss']:.8f}"
    )

    print(
        f"MSE:  "
        f"{initial_metrics['mse']:.8f}"
    )

    print(
        f"MAE:  "
        f"{initial_metrics['mae']:.8f}"
    )

    print(
        f"CCC:  "
        f"{initial_metrics['ccc']:.8f}"
    )

    print(
        f"PCC:  "
        f"{initial_metrics['pcc']:.8f}"
    )

    # TRAINING

    best_val_pcc = -float("inf")
    best_epoch = -1

    history = []

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        running_n = 0

        for batch in train_loader:

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

            # FORWARD

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                hidden,
                valid_mask,
            )

            loss = quality_loss(
                pred,
                target,
            )

            # NUMERICAL CHECK

            if not torch.isfinite(loss):

                raise RuntimeError(
                    "Non-finite training "
                    f"loss at epoch {epoch}"
                )

            # BACKWARD

            loss.backward()

            optimizer.step()

            # STATS

            n = target.size(0)

            running_loss += (
                loss.item() * n
            )

            running_n += n

        train_loss = (
            running_loss
            / running_n
        )

        # VALIDATION

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_ccc": val_metrics["ccc"],
                "val_pcc": val_metrics["pcc"],
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_MSE={val_metrics['mse']:.6f} | "
            f"val_CCC={val_metrics['ccc']:.6f} | "
            f"val_PCC={val_metrics['pcc']:.6f}"
        )

        # BEST CHECKPOINT BY PCC

        if (
            np.isfinite(val_metrics["pcc"])
            and val_metrics["pcc"]
            > best_val_pcc
        ):

            best_val_pcc = (
                val_metrics["pcc"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_loss":
                        val_metrics["loss"],

                    "val_mse":
                        val_metrics["mse"],

                    "val_mae":
                        val_metrics["mae"],

                    "val_ccc":
                        val_metrics["ccc"],

                    "val_pcc":
                        val_metrics["pcc"],

                    "seed": SEED,
                    "hidden_size":
                        HIDDEN_SIZE,
                    "lr": LR,
                    "batch_size":
                        BATCH_SIZE,
                    "val_ratio":
                        VAL_RATIO,

                    "split_type":
                        "source_balanced_iid",

                    "backbone":
                        "facebook/hubert-base-ls960",

                    "source_counts": {
                        k: int(v)
                        for k, v
                        in source_counts.items()
                    },
                },
                BEST_CKPT,
            )

            print(
                f"  -> NEW BEST checkpoint "
                f"(epoch {epoch}, "
                f"PCC={val_metrics['pcc']:.6f})"
            )


    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )


    print()
    print("=" * 70)
    print("LOAD BEST CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        BEST_CKPT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best val PCC: "
        f"{checkpoint['val_pcc']:.8f}"
    )

    print(
        f"Best val loss: "
        f"{checkpoint['val_loss']:.8f}"
    )

    print(
        f"Checkpoint: "
        f"{BEST_CKPT}"
    )

    # FINAL EVALUATION

    final_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    print()
    print("=" * 70)
    print(
        "FINAL BEST-CHECKPOINT "
        "VALIDATION"
    )
    print("=" * 70)

    print(
        f"Loss: "
        f"{final_metrics['loss']:.8f}"
    )

    print(
        f"MSE:  "
        f"{final_metrics['mse']:.8f}"
    )

    print(
        f"MAE:  "
        f"{final_metrics['mae']:.8f}"
    )

    print(
        f"CCC:  "
        f"{final_metrics['ccc']:.8f}"
    )

    print(
        f"PCC:  "
        f"{final_metrics['pcc']:.8f}"
    )

    # SUMMARY

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Backbone:            "
        f"facebook/hubert-base-ls960"
    )

    print(
        f"Hidden size:         "
        f"{HIDDEN_SIZE}"
    )

    print(
        f"Total samples:       "
        f"{len(df)}"
    )

    print(
        f"Train samples:       "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples:  "
        f"{len(val_df)}"
    )

    print(
        f"Best epoch:          "
        f"{best_epoch}"
    )

    print(
        f"Best validation PCC: "
        f"{final_metrics['pcc']:.8f}"
    )

    print(
        f"Best validation MSE: "
        f"{final_metrics['mse']:.8f}"
    )

    print(
        f"Best validation CCC: "
        f"{final_metrics['ccc']:.8f}"
    )

    print()
    print("Files:")

    print(
        f"  Best checkpoint: "
        f"{BEST_CKPT}"
    )

    print(
        f"  History:         "
        f"{HISTORY_PATH}"
    )

    print()
    print(
        ">>> HuBERT SOURCE-BALANCED IID "
        "TRAINING: COMPLETE"
    )


if __name__ == "__main__":
    main()
