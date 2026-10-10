from __future__ import annotations

# IMPORTANT:
# Add project root to sys.path BEFORE importing src.*

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import HubertModel

from src.sqa.dataset import (
    QualiSpeechDataset,
    qualispeech_collate,
)


# CONFIG

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"

WAV_DIR = ROOT / "data" / "qualispeech" / "wav"

# IMPORTANT:
# HuBERT cache is completely separate from WavLM cache.
CACHE_ROOT = (
    ROOT
    / "data"
    / "qualispeech"
    / "hubert_cache"
)

MODEL_NAME = "facebook/hubert-base-ls960"

EXPECTED_HIDDEN_SIZE = 768

# First run tiny subset only.
# Change to False ONLY after tiny cache + tiny overfit pass.
CACHE_TINY_ONLY = False

BATCH_SIZE = 4

NUM_WORKERS = 0

SEED = 42

TARGET_COLUMN = "Overall quality"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# SEED


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# SAMPLE ID


def normalize_sample_id(value) -> str:
    """
    Convert:

        abc.wav -> abc
        abc     -> abc
    """

    value = str(value).strip()

    if value.lower().endswith(".wav"):
        value = value[:-4]

    return value


# CACHE PATH


def cache_path_for_file(sample_id: str) -> Path:
    """
    Cache:

        xxx.wav

    as:

        hubert_cache/xxx.pt
    """

    return CACHE_ROOT / f"{sample_id}.pt"


# ATOMIC SAVE


def atomic_torch_save(
    obj,
    path: Path,
):
    """
    Save to temporary file first, then atomically replace target.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:

        torch.save(
            obj,
            tmp_path,
        )

        os.replace(
            tmp_path,
            path,
        )

    finally:

        if tmp_path.exists():

            try:
                tmp_path.unlink()

            except OSError:
                pass


# VERIFY ONE CACHE FILE


def verify_cache_file(
    path: Path,
    expected_sample_id: str,
    expected_target: float,
) -> tuple[bool, str]:

    if not path.exists():
        return False, "missing"

    try:

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as exc:

        return False, f"load_error: {exc}"

    if not isinstance(obj, dict):
        return False, "not_dict"

    required_keys = {
        "hidden",
        "valid_mask",
        "target",
        "file_id",
    }

    missing_keys = (
        required_keys
        - set(obj.keys())
    )

    if missing_keys:

        return (
            False,
            f"missing_keys={sorted(missing_keys)}",
        )

    hidden = obj["hidden"]

    valid_mask = obj["valid_mask"]

    target = obj["target"]

    file_id = str(
        obj["file_id"]
    )

    # hidden

    if not isinstance(
        hidden,
        torch.Tensor,
    ):
        return False, "hidden_not_tensor"

    if hidden.ndim != 2:

        return (
            False,
            f"hidden_ndim={hidden.ndim}",
        )

    if hidden.shape[1] != EXPECTED_HIDDEN_SIZE:

        return (
            False,
            f"hidden_size={hidden.shape[1]} "
            f"expected={EXPECTED_HIDDEN_SIZE}",
        )

    if hidden.shape[0] <= 0:

        return False, "zero_frames"

    if not torch.isfinite(
        hidden
    ).all():

        return False, "hidden_nonfinite"

    # valid mask

    if not isinstance(
        valid_mask,
        torch.Tensor,
    ):
        return False, "valid_mask_not_tensor"

    if valid_mask.ndim != 1:

        return (
            False,
            f"valid_mask_ndim={valid_mask.ndim}",
        )

    if (
        valid_mask.shape[0]
        != hidden.shape[0]
    ):

        return (
            False,
            f"frame_mismatch "
            f"hidden={hidden.shape[0]} "
            f"mask={valid_mask.shape[0]}",
        )

    # file ID

    if file_id != expected_sample_id:

        return (
            False,
            f"file_id={file_id!r} "
            f"expected={expected_sample_id!r}",
        )

    # target

    try:

        cached_target = float(
            target
        )

    except Exception:

        return False, "target_not_numeric"

    if not np.isfinite(
        cached_target
    ):

        return False, "target_nonfinite"

    if (
        abs(
            cached_target
            - expected_target
        )
        > 1e-6
    ):

        return (
            False,
            f"target={cached_target} "
            f"expected={expected_target}",
        )

    return True, "ok"


# BUILD TINY SUBSET


def build_tiny_indices(
    df: pd.DataFrame,
) -> list[int]:

    """
    Exact tiny distribution used previously:

        quality 1 -> 3
        quality 2 -> 3
        quality 3 -> 4
        quality 4 -> 3
        quality 5 -> 3

    Total = 16.
    """

    rng = np.random.RandomState(
        SEED
    )

    desired_counts = {
        1.0: 3,
        2.0: 3,
        3.0: 4,
        4.0: 3,
        5.0: 3,
    }

    selected_indices = []

    for (
        target_value,
        count,
    ) in desired_counts.items():

        class_df = df[
            df["target"]
            == target_value
        ]

        if len(class_df) < count:

            raise RuntimeError(
                f"Not enough samples for "
                f"target={target_value}: "
                f"need {count}, "
                f"found {len(class_df)}"
            )

        candidates = (
            class_df.index
            .to_numpy(copy=True)
        )

        rng.shuffle(
            candidates
        )

        selected_indices.extend(
            candidates[:count].tolist()
        )

    return selected_indices


# WAV INDEX


def build_wav_index(
    wav_dir: Path,
) -> dict[str, Path]:

    wav_paths = {}

    for wav_path in wav_dir.glob(
        "*.wav"
    ):

        sample_id = normalize_sample_id(
            wav_path.stem
        )

        if sample_id in wav_paths:

            raise RuntimeError(
                f"Duplicate WAV sample ID: "
                f"{sample_id}"
            )

        wav_paths[
            sample_id
        ] = wav_path

    return wav_paths


# VERIFY WAV MAPPING


def verify_wav_mapping(
    df: pd.DataFrame,
    wav_paths: dict[str, Path],
):

    csv_ids = set(
        df["sample_id"]
        .astype(str)
    )

    wav_ids = set(
        wav_paths.keys()
    )

    csv_only = (
        csv_ids - wav_ids
    )

    wav_only = (
        wav_ids - csv_ids
    )

    print()
    print("WAV MAPPING AUDIT")

    print(
        f"CSV IDs:       {len(csv_ids)}"
    )

    print(
        f"WAV IDs:       {len(wav_ids)}"
    )

    print(
        f"CSV only:      {len(csv_only)}"
    )

    print(
        f"WAV only:      {len(wav_only)}"
    )

    if csv_only:

        examples = sorted(
            csv_only
        )[:10]

        raise RuntimeError(
            "CSV samples missing WAV files. "
            f"Examples: {examples}"
        )

    if wav_only:

        examples = sorted(
            wav_only
        )[:10]

        print(
            f"WARNING: {len(wav_only)} "
            f"WAV files are not in CSV."
        )

        print(
            f"Examples: {examples}"
        )

    print(
        "CSV <-> WAV mapping: PASS"
    )


# VERIFY DATASET MAPPING


def verify_pending_dataset_mapping(
    dataset,
    pending_df: pd.DataFrame,
):

    """
    Verify that QualiSpeechDataset preserves
    the expected sample order.

    The actual dataset may expose:

        dataset.df["id"]

    rather than:

        dataset.df["sample_id"]

    Therefore both are supported.
    """

    if len(dataset) != len(
        pending_df
    ):

        raise RuntimeError(
            f"Dataset length mismatch: "
            f"{len(dataset)} != "
            f"{len(pending_df)}"
        )

    expected_ids = (
        pending_df["sample_id"]
        .astype(str)
        .tolist()
    )

    # If dataset exposes .df

    if hasattr(
        dataset,
        "df",
    ):

        dataset_df = dataset.df

        if len(dataset_df) != len(
            pending_df
        ):

            raise RuntimeError(
                "Dataset internal dataframe "
                "length mismatch: "
                f"{len(dataset_df)} != "
                f"{len(pending_df)}"
            )

        # Dataset uses sample_id

        if (
            "sample_id"
            in dataset_df.columns
        ):

            actual_ids = (
                dataset_df[
                    "sample_id"
                ]
                .astype(str)
                .str.strip()
                .str.replace(
                    r"\.wav$",
                    "",
                    regex=True,
                )
                .tolist()
            )

        # Dataset uses original id

        elif (
            "id"
            in dataset_df.columns
        ):

            actual_ids = (
                dataset_df["id"]
                .astype(str)
                .str.strip()
                .str.replace(
                    r"\.wav$",
                    "",
                    regex=True,
                )
                .tolist()
            )

        else:

            print(
                "WARNING: dataset.df exists "
                "but has neither 'id' nor "
                "'sample_id'."
            )

            print(
                "Skipping internal ID check."
            )

            print(
                "Pending dataset mapping: "
                "PASS (length verified)"
            )

            return

        if actual_ids != expected_ids:

            raise RuntimeError(
                "Dataset mapping mismatch!\n"
                "Expected IDs and actual "
                "dataset IDs differ."
            )

        print(
            "Pending dataset mapping: PASS"
        )

        return

    # Dataset has no .df

    print(
        "WARNING: QualiSpeechDataset has "
        "no .df attribute."
    )

    print(
        "Skipping internal ID check."
    )

    print(
        "Pending dataset mapping: "
        "PASS (length verified)"
    )


# MAIN


def main():

    set_seed(
        SEED
    )

    print("=" * 72)
    print(
        "QualiSpeech HuBERT-base "
        "REPRESENTATION CACHE"
    )
    print("=" * 72)

    print(
        f"Device:        {DEVICE}"
    )

    print(
        f"CSV:           {CSV_PATH}"
    )

    print(
        f"WAV directory: {WAV_DIR}"
    )

    print(
        f"Cache:         {CACHE_ROOT}"
    )

    print(
        f"Model:         {MODEL_NAME}"
    )

    print(
        f"Tiny only:     {CACHE_TINY_ONLY}"
    )

    print()

    # PATH CHECK

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}"
        )

    if not WAV_DIR.exists():

        raise FileNotFoundError(
            f"WAV directory not found: "
            f"{WAV_DIR}"
        )

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"CSV rows:      {len(df)}"
    )

    if "id" not in df.columns:

        raise KeyError(
            "CSV must contain 'id'. "
            f"Columns: {df.columns.tolist()}"
        )

    if TARGET_COLUMN not in df.columns:

        raise KeyError(
            f"CSV must contain "
            f"{TARGET_COLUMN!r}"
        )

    df = df.copy()

    # Exact original CSV row index.
    df[
        "original_csv_index"
    ] = np.arange(
        len(df)
    )

    # Remove .wav for internal sample ID.
    df[
        "sample_id"
    ] = (
        df["id"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.wav$",
            "",
            regex=True,
        )
    )

    df[
        "target"
    ] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise",
    )

    # DUPLICATE ID CHECK

    if df[
        "sample_id"
    ].duplicated().any():

        duplicates = (
            df.loc[
                df[
                    "sample_id"
                ].duplicated(
                    keep=False
                ),
                "sample_id",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate sample IDs detected. "
            f"Examples: {duplicates[:20]}"
        )

    # INDEX WAV

    wav_paths = build_wav_index(
        WAV_DIR
    )

    print(
        f"WAV files:     {len(wav_paths)}"
    )

    verify_wav_mapping(
        df,
        wav_paths,
    )

    # SELECT DATA

    if CACHE_TINY_ONLY:

        tiny_indices = (
            build_tiny_indices(df)
        )

        work_df = (
            df.loc[
                tiny_indices
            ]
            .copy()
        )

        # Restore original CSV order.
        work_df = (
            work_df
            .sort_values(
                "original_csv_index"
            )
            .reset_index(
                drop=True
            )
        )

        print()
        print(
            "Tiny subset selected."
        )

        print(
            f"Samples selected: "
            f"{len(work_df)}"
        )

        target_counts = (
            work_df[
                "target"
            ]
            .value_counts()
            .sort_index()
        )

        print(
            "Tiny target distribution:"
        )

        for (
            target_value,
            count,
        ) in target_counts.items():

            print(
                f"  target="
                f"{target_value:.0f}: "
                f"{count}"
            )

    else:

        work_df = (
            df.copy()
            .reset_index(
                drop=True
            )
        )

        print()
        print(
            "FULL DATASET selected."
        )

        print(
            f"Samples selected: "
            f"{len(work_df)}"
        )

    # CHECK EXISTING CACHE

    print()
    print(
        "Checking existing HuBERT "
        "cache files..."
    )

    valid_existing = 0
    invalid_existing = 0

    pending_rows = []

    for _, row in work_df.iterrows():

        sample_id = str(
            row["sample_id"]
        )

        target = float(
            row["target"]
        )

        cache_path = (
            cache_path_for_file(
                sample_id
            )
        )

        if cache_path.exists():

            ok, reason = (
                verify_cache_file(
                    cache_path,
                    expected_sample_id=sample_id,
                    expected_target=target,
                )
            )

            if ok:

                valid_existing += 1

                continue

            invalid_existing += 1

            print(
                f"Invalid cache: "
                f"{cache_path.name} "
                f"({reason})"
            )

        pending_rows.append(
            row
        )

    pending_df = pd.DataFrame(
        pending_rows,
        columns=work_df.columns,
    )

    print()
    print(
        f"Valid existing cache: "
        f"{valid_existing}"
    )

    print(
        f"Invalid existing cache: "
        f"{invalid_existing}"
    )

    print(
        f"Need HuBERT extraction: "
        f"{len(pending_df)}"
    )

    # ALL CACHE ALREADY EXISTS

    if len(pending_df) == 0:

        print()
        print(
            "All requested HuBERT "
            "cache files are valid."
        )

        failed = []

        for _, row in work_df.iterrows():

            sample_id = str(
                row["sample_id"]
            )

            target = float(
                row["target"]
            )

            cache_path = (
                cache_path_for_file(
                    sample_id
                )
            )

            ok, reason = (
                verify_cache_file(
                    cache_path,
                    expected_sample_id=sample_id,
                    expected_target=target,
                )
            )

            if not ok:

                failed.append(
                    (
                        sample_id,
                        reason,
                    )
                )

        if failed:

            raise RuntimeError(
                "Final cache verification "
                "failed: "
                f"{failed[:10]}"
            )

        print()
        print(
            ">>> HuBERT CACHE PASS"
        )

        return

    # PENDING DATASET

    print()
    print(
        f"Pending dataset: "
        f"{len(pending_df)}"
    )

    print(
        "Original CSV indices preserved: YES"
    )

    # BUILD DATASET

    pending_dataset = (
        QualiSpeechDataset(
            csv_path=CSV_PATH,
            wav_dir=WAV_DIR,
            indices=(
                pending_df[
                    "original_csv_index"
                ]
                .tolist()
            ),
        )
    )

    verify_pending_dataset_mapping(
        pending_dataset,
        pending_df,
    )


    print()
    print(
        "Loading frozen HuBERT-base..."
    )

    model = (
        HubertModel
        .from_pretrained(
            MODEL_NAME
        )
        .to(DEVICE)
    )

    # CRITICAL:
    # Freeze BEFORE checking requires_grad.

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad = False

    model.eval()

    hidden_size = (
        model.config.hidden_size
    )

    print(
        f"HuBERT hidden size: "
        f"{hidden_size}"
    )

    if (
        hidden_size
        != EXPECTED_HIDDEN_SIZE
    ):

        raise RuntimeError(
            f"Unexpected HuBERT hidden size: "
            f"{hidden_size} != "
            f"{EXPECTED_HIDDEN_SIZE}"
        )

    trainable_parameters = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    if trainable_parameters:

        raise RuntimeError(
            "HuBERT is not fully frozen. "
            f"Trainable parameters: "
            f"{trainable_parameters[:20]}"
        )

    print(
        "HuBERT frozen: YES"
    )

    print(
        "Trainable parameters: 0"
    )

    # DATALOADER

    loader = DataLoader(
        pending_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=qualispeech_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # EXTRACTION

    print()
    print("=" * 72)
    print(
        "START HuBERT FEATURE EXTRACTION"
    )
    print("=" * 72)

    extracted = 0

    model.eval()

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            loader
        ):

            # WAV

            if "wav" in batch:

                wav = batch[
                    "wav"
                ]

            elif "waveform" in batch:

                wav = batch[
                    "waveform"
                ]

            else:

                raise KeyError(
                    "Batch does not contain "
                    "'wav' or 'waveform'. "
                    f"Batch keys: "
                    f"{list(batch.keys())}"
                )

            # ATTENTION MASK

            if (
                "attention_mask"
                not in batch
            ):

                raise KeyError(
                    "Batch does not contain "
                    "'attention_mask'. "
                    f"Batch keys: "
                    f"{list(batch.keys())}"
                )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
            )

            # BATCH SIZE

            batch_size_actual = (
                wav.shape[0]
            )

            start_idx = (
                batch_idx
                * BATCH_SIZE
            )

            batch_rows = (
                pending_df.iloc[
                    start_idx:
                    start_idx
                    + batch_size_actual
                ]
            )

            if (
                len(batch_rows)
                != batch_size_actual
            ):

                raise RuntimeError(
                    "Batch/metadata size mismatch."
                )

            # DEVICE

            wav = wav.to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = (
                attention_mask.to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            # HuBERT FORWARD

            outputs = model(
                input_values=wav,
                attention_mask=attention_mask,
            )

            hidden = (
                outputs.last_hidden_state
            )

            # HIDDEN SHAPE

            if hidden.ndim != 3:

                raise RuntimeError(
                    f"Expected hidden "
                    f"[B,T,H], got "
                    f"{tuple(hidden.shape)}"
                )

            if (
                hidden.shape[2]
                != EXPECTED_HIDDEN_SIZE
            ):

                raise RuntimeError(
                    f"Unexpected HuBERT "
                    f"hidden size: "
                    f"{hidden.shape[2]} != "
                    f"{EXPECTED_HIDDEN_SIZE}"
                )

            # VALID FRAME LENGTHS

            input_lengths = (
                attention_mask
                .long()
                .sum(dim=1)
            )

            frame_lengths = (
                model
                ._get_feat_extract_output_lengths(
                    input_lengths
                )
            )

            max_frames = (
                hidden.shape[1]
            )

            frame_indices = (
                torch.arange(
                    max_frames,
                    device=DEVICE,
                )
                .unsqueeze(0)
            )

            valid_mask = (
                frame_indices
                < frame_lengths
                .unsqueeze(1)
            )

            # EACH SAMPLE

            for (
                local_idx,
                (_, row),
            ) in enumerate(
                batch_rows.iterrows()
            ):

                sample_id = str(
                    row["sample_id"]
                )

                expected_target = float(
                    row["target"]
                )

                # Hidden

                sample_hidden = (
                    hidden[
                        local_idx
                    ]
                    .detach()
                    .cpu()
                )

                # Valid mask

                sample_valid_mask = (
                    valid_mask[
                        local_idx
                    ]
                    .detach()
                    .cpu()
                )

                # Keep only valid frames

                sample_hidden = (
                    sample_hidden[
                        sample_valid_mask
                    ]
                )

                # After filtering, all remaining frames are valid.
                sample_mask = torch.ones(
                    sample_hidden.shape[0],
                    dtype=torch.bool,
                )

                # Shape checks

                if (
                    sample_hidden.ndim
                    != 2
                ):

                    raise RuntimeError(
                        f"Unexpected hidden "
                        f"shape for "
                        f"{sample_id}: "
                        f"{sample_hidden.shape}"
                    )

                if (
                    sample_hidden.shape[1]
                    != EXPECTED_HIDDEN_SIZE
                ):

                    raise RuntimeError(
                        f"Unexpected hidden "
                        f"dimension for "
                        f"{sample_id}: "
                        f"{sample_hidden.shape}"
                    )

                if (
                    sample_hidden.shape[0]
                    == 0
                ):

                    raise RuntimeError(
                        f"HuBERT produced "
                        f"zero valid frames "
                        f"for {sample_id}"
                    )

                if not torch.isfinite(
                    sample_hidden
                ).all():

                    raise RuntimeError(
                        f"Non-finite HuBERT "
                        f"representation "
                        f"for {sample_id}"
                    )

                # CACHE OBJECT

                cache_obj = {
                    "hidden": (
                        sample_hidden
                        .float()
                    ),
                    "valid_mask": (
                        sample_mask
                    ),
                    "target": torch.tensor(
                        expected_target,
                        dtype=torch.float32,
                    ),
                    "file_id": sample_id,
                }

                cache_path = (
                    cache_path_for_file(
                        sample_id
                    )
                )

                atomic_torch_save(
                    cache_obj,
                    cache_path,
                )

                # IMMEDIATE VERIFICATION

                ok, reason = (
                    verify_cache_file(
                        cache_path,
                        expected_sample_id=sample_id,
                        expected_target=expected_target,
                    )
                )

                if not ok:

                    raise RuntimeError(
                        f"Cache verification "
                        f"failed for "
                        f"{sample_id}: "
                        f"{reason}"
                    )

                extracted += 1

                print(
                    f"[{extracted:>4}/"
                    f"{len(pending_df)}] "
                    f"{sample_id} | "
                    f"target="
                    f"{expected_target:.4f} | "
                    f"frames="
                    f"{sample_hidden.shape[0]} | "
                    f"hidden="
                    f"{sample_hidden.shape[1]}"
                )

    # FINAL VERIFICATION

    print()
    print("=" * 72)
    print(
        "FINAL HuBERT CACHE VERIFICATION"
    )
    print("=" * 72)

    failed = []

    frame_counts = []

    for _, row in work_df.iterrows():

        sample_id = str(
            row["sample_id"]
        )

        expected_target = float(
            row["target"]
        )

        cache_path = (
            cache_path_for_file(
                sample_id
            )
        )

        ok, reason = (
            verify_cache_file(
                cache_path,
                expected_sample_id=sample_id,
                expected_target=expected_target,
            )
        )

        if not ok:

            failed.append(
                (
                    sample_id,
                    reason,
                )
            )

            continue

        try:

            obj = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=False,
            )

            frame_counts.append(
                int(
                    obj[
                        "hidden"
                    ].shape[0]
                )
            )

        except Exception as exc:

            failed.append(
                (
                    sample_id,
                    f"reload_error={exc}",
                )
            )

    if failed:

        print(
            f"FAILED cache files: "
            f"{len(failed)}"
        )

        for (
            sample_id,
            reason,
        ) in failed[:20]:

            print(
                f"  {sample_id}: "
                f"{reason}"
            )

        raise RuntimeError(
            "HuBERT cache verification failed."
        )

    # SUMMARY

    total_cache_files = len(
        list(
            CACHE_ROOT.glob(
                "*.pt"
            )
        )
    )

    print()

    print(
        f"Requested samples:     "
        f"{len(work_df)}"
    )

    print(
        f"Newly extracted:       "
        f"{extracted}"
    )

    print(
        f"Valid requested cache: "
        f"{len(work_df)}"
    )

    print(
        f"Cache files on disk:   "
        f"{total_cache_files}"
    )

    if frame_counts:

        print(
            "Frames/sample:         "
            f"min={min(frame_counts)}, "
            f"max={max(frame_counts)}, "
            f"mean={np.mean(frame_counts):.2f}"
        )

    print(
        f"Hidden size:           "
        f"{EXPECTED_HIDDEN_SIZE}"
    )

    print()
    print("=" * 72)
    print(
        ">>> HuBERT CACHE PASS"
    )
    print("=" * 72)


# ENTRY POINT


if __name__ == "__main__":
    main()
