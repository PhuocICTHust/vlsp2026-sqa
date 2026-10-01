from __future__ import annotations

import sys
import zipfile
from pathlib import Path

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
from sqa.model import WavLMBackbone


CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
ZIP_PATH = ROOT / "data" / "qualispeech" / "wav_part1.zip"

CACHE_ROOT = ROOT / "data" / "qualispeech" / "wavlm_cache"

# ============================================================================
# CONFIG
# ============================================================================

SEED = 42

# IMPORTANT:
# First run = True.
# After tiny cache verification passes, change to False.
CACHE_TINY_ONLY = False

SAMPLES_PER_CLASS = {
    1: 3,
    2: 3,
    3: 4,
    4: 3,
    5: 3,
}

BATCH_SIZE = 4

# WavLM is frozen, so gradients are not needed.
# Keep num_workers=0 for maximum Windows compatibility.
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_NAME = "microsoft/wavlm-large"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# ZIP INDEX
# ============================================================================


def build_zip_index(zip_path: Path) -> dict[str, str]:
    """
    Build:
        normalized filename -> actual ZIP member name

    This lets us handle different path separators/casing inside the ZIP.
    """

    print()
    print("Indexing ZIP...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    index: dict[str, str] = {}

    for name in names:
        if name.endswith("/"):
            continue

        normalized = Path(name).name.lower()

        # If duplicate basenames exist, keep the first one.
        if normalized not in index:
            index[normalized] = name

    print(f"ZIP files indexed: {len(index)}")

    return index


# ============================================================================
# TINY SUBSET
# ============================================================================


def build_tiny_indices(
    df: pd.DataFrame,
    zip_index: dict[str, str],
) -> list[int]:
    """
    Select the same 16-sample stratified subset used in
    run_real_tiny_overfit.py.

    Distribution:
        quality 1 -> 3
        quality 2 -> 3
        quality 3 -> 4
        quality 4 -> 3
        quality 5 -> 3
    """

    required = SAMPLES_PER_CLASS

    selected: list[int] = []

    # Use the same column names expected by the current dataset.
    id_col = "id"
    target_col = "Overall quality"

    if id_col not in df.columns:
        raise KeyError(
            f"Missing column '{id_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    if target_col not in df.columns:
        raise KeyError(
            f"Missing column '{target_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    for quality, count in required.items():
        candidates = []

        for idx, row in df.iterrows():
            file_id = str(row[id_col])
            key = Path(file_id).name.lower()

            if key not in zip_index:
                continue

            try:
                target = float(row[target_col])
            except (TypeError, ValueError):
                continue

            if target == quality:
                candidates.append(idx)

        if len(candidates) < count:
            raise RuntimeError(
                f"Not enough samples for quality={quality}. "
                f"Required={count}, found={len(candidates)}."
            )

        selected.extend(candidates[:count])

    return selected


# ============================================================================
# CACHE PATH
# ============================================================================


def cache_path_for_file(file_id: str) -> Path:
    """
    One cache file per audio file.

    Example:
        sys01tts-0000017.wav
        ->
        wavlm_cache/sys01tts-0000017.pt
    """

    stem = Path(file_id).stem

    return CACHE_ROOT / f"{stem}.pt"


# ============================================================================
# SAVE CACHE
# ============================================================================


def save_cache(
    path: Path,
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    target: torch.Tensor,
    file_id: str,
) -> None:
    """
    Save CPU tensors only.

    This is intentional:
        - cache should not depend on GPU
        - easier to reload later
        - no unnecessary CUDA memory usage
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "hidden": hidden.detach().cpu().contiguous(),
        "valid_mask": valid_mask.detach().cpu().contiguous(),
        "target": float(target.item()),
        "file_id": str(file_id),
    }

    torch.save(payload, path)


# ============================================================================
# LOAD / VERIFY EXISTING CACHE
# ============================================================================


def verify_cache_file(
    path: Path,
    expected_hidden_size: int,
) -> dict:
    """
    Verify one cache file.

    Returns metadata useful for logging.
    """

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "hidden",
        "valid_mask",
        "target",
        "file_id",
    }

    missing = required_keys.difference(payload.keys())

    if missing:
        raise RuntimeError(
            f"Cache file {path} is missing keys: {sorted(missing)}"
        )

    hidden = payload["hidden"]
    valid_mask = payload["valid_mask"]

    if not isinstance(hidden, torch.Tensor):
        raise TypeError(
            f"{path}: hidden must be torch.Tensor"
        )

    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError(
            f"{path}: valid_mask must be torch.Tensor"
        )

    if hidden.ndim != 2:
        raise RuntimeError(
            f"{path}: hidden must have shape [T,H], "
            f"got {tuple(hidden.shape)}"
        )

    if hidden.shape[1] != expected_hidden_size:
        raise RuntimeError(
            f"{path}: expected hidden size "
            f"{expected_hidden_size}, "
            f"got {hidden.shape[1]}"
        )

    if valid_mask.ndim != 1:
        raise RuntimeError(
            f"{path}: valid_mask must have shape [T], "
            f"got {tuple(valid_mask.shape)}"
        )

    if valid_mask.shape[0] != hidden.shape[0]:
        raise RuntimeError(
            f"{path}: hidden T={hidden.shape[0]} but "
            f"valid_mask T={valid_mask.shape[0]}"
        )

    if not torch.isfinite(hidden).all():
        raise RuntimeError(
            f"{path}: hidden contains NaN or Inf"
        )

    return {
        "frames": int(hidden.shape[0]),
        "hidden_size": int(hidden.shape[1]),
        "valid_frames": int(valid_mask.bool().sum().item()),
        "target": float(payload["target"]),
        "file_id": str(payload["file_id"]),
    }


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    set_seed(SEED)

    print("=" * 72)
    print("QualiSpeech WavLM REPRESENTATION CACHE")
    print("=" * 72)

    print(f"Device:       {DEVICE}")
    print(f"CSV:          {CSV_PATH}")
    print(f"ZIP:          {ZIP_PATH}")
    print(f"Cache:        {CACHE_ROOT}")
    print(f"Model:        {MODEL_NAME}")
    print(f"Tiny only:    {CACHE_TINY_ONLY}")
    print()

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"ZIP not found:\n{ZIP_PATH}"
        )

    # ------------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    print(f"CSV rows: {len(df)}")

    # ------------------------------------------------------------------------
    # Build ZIP index
    # ------------------------------------------------------------------------

    zip_index = build_zip_index(ZIP_PATH)

    # ------------------------------------------------------------------------
    # Determine samples
    # ------------------------------------------------------------------------

    if CACHE_TINY_ONLY:
        tiny_indices = build_tiny_indices(df, zip_index)

        work_df = df.iloc[tiny_indices].copy()

        print()
        print("TINY CACHE SUBSET:")
        print(
            work_df[
                ["id", "Overall quality"]
            ].to_string(index=False)
        )

        print()
        print("Tiny target distribution:")
        print(
            work_df["Overall quality"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        valid_indices = []

        for idx, row in df.iterrows():

            file_id = str(row["id"])

            key = Path(file_id).name.lower()

            if key in zip_index:
                valid_indices.append(idx)

    # IMPORTANT:

    # QualiSpeechDataset expects positional indices in the

    # filtered dataframe, not the original CSV indices.

        work_df = (

        df.iloc[valid_indices]

        .copy()

        .reset_index(drop=True)

        )

        print()

        print(f"Samples with ZIP audio: {len(work_df)}")



# ------------------------------------------------------------------------
    # Build Dataset
    # ------------------------------------------------------------------------

    # We construct the dataset using exactly the selected rows.
    #
    # The dataset is responsible for:
    #   - reading audio from ZIP
    #   - resampling / preprocessing
    #   - constructing waveform attention mask
    #
    # The cache script does NOT duplicate audio preprocessing logic.
    dataset = QualiSpeechDataset(
        csv_path=CSV_PATH,
        zip_path=ZIP_PATH,
        indices=work_df.index.tolist(),
    )

    print()
    print(f"Dataset samples: {len(dataset)}")

    # ------------------------------------------------------------------------
    # Load frozen WavLM
    # ------------------------------------------------------------------------

    print()
    print("Loading frozen WavLM...")

    backbone = WavLMBackbone(
        model_name=MODEL_NAME,
        freeze=True,
    ).to(DEVICE)

    backbone.eval()

    hidden_size = backbone.hidden_size

    print(f"WavLM hidden size: {hidden_size}")

    # Verify every backbone parameter is frozen.
    trainable_backbone = [
        name
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    ]

    if trainable_backbone:
        raise RuntimeError(
            "WavLM is not fully frozen. "
            f"Trainable parameters found: {trainable_backbone[:10]}"
        )

    print("WavLM parameters: FROZEN")

    # ------------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=qualispeech_collate,
    )

    # ------------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------------

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    print()
    print("Starting WavLM feature extraction...")
    print()

    saved = 0
    skipped = 0
    total_frames = 0

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader, start=1):

            wav = batch["wav"].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                DEVICE,
                non_blocking=True,
            )

            target = batch["target"]

            file_ids = batch["file_id"]

            # ---------------------------------------------------------------
            # WavLM
            # ---------------------------------------------------------------

            hidden = backbone(
                wav,
                attention_mask=attention_mask,
            )

            # hidden:
            #   [B, T, H]
            #
            # Note:
            #   T can differ conceptually by audio length.
            #   The collate function pads the batch to a common T.
            #
            # We calculate the real frame-level mask to identify
            # non-padding WavLM frames.

            valid_mask = backbone.frame_valid_mask(
                attention_mask
            )

            if hidden.ndim != 3:
                raise RuntimeError(
                    f"Expected hidden [B,T,H], "
                    f"got {tuple(hidden.shape)}"
                )

            if hidden.shape[0] != len(file_ids):
                raise RuntimeError(
                    f"Batch mismatch: hidden B={hidden.shape[0]}, "
                    f"file_ids={len(file_ids)}"
                )

            if hidden.shape[1] != valid_mask.shape[1]:
                raise RuntimeError(
                    f"Frame mismatch: hidden T={hidden.shape[1]}, "
                    f"valid_mask T={valid_mask.shape[1]}"
                )

            if hidden.shape[2] != hidden_size:
                raise RuntimeError(
                    f"Hidden size mismatch: "
                    f"{hidden.shape[2]} != {hidden_size}"
                )

            # ---------------------------------------------------------------
            # Save each audio independently
            # ---------------------------------------------------------------

            for i, file_id in enumerate(file_ids):

                file_target = target[i]

                cache_path = cache_path_for_file(file_id)

                # Do not overwrite an already valid cache file.
                if cache_path.exists():
                    try:
                        verify_cache_file(
                            cache_path,
                            expected_hidden_size=hidden_size,
                        )
                        skipped += 1
                        continue

                    except Exception:
                        # Existing cache is invalid.
                        # Recompute and overwrite it.
                        print(
                            f"  Invalid cache -> recomputing: "
                            f"{cache_path.name}"
                        )

                # Keep only REAL WavLM frames.
                #
                # hidden[i] is padded to the maximum frame length of the batch.
                # valid_mask[i] has the same batch-level frame length.
                #
                # We remove padding BEFORE saving so that:
                #     hidden      = [T_real, H]
                #     valid_mask  = [T_real]
                #
                # After trimming, every remaining frame is valid.
                sample_valid = valid_mask[i].bool()

                sample_hidden = hidden[i][sample_valid]

                sample_mask = torch.ones(
                    sample_hidden.shape[0],
                    dtype=torch.bool,
                    device=sample_hidden.device,
                )

                if sample_hidden.ndim != 2:
                    raise RuntimeError(
                        f"{file_id}: expected [T,H], "
                        f"got {tuple(sample_hidden.shape)}"
                    )

                if sample_hidden.shape[0] == 0:
                    raise RuntimeError(
                        f"{file_id}: zero valid WavLM frames."
                    )

                save_cache(
                    path=cache_path,
                    hidden=sample_hidden,
                    valid_mask=sample_mask,
                    target=file_target,
                    file_id=file_id,
                )

                saved += 1
                total_frames += sample_hidden.shape[0]

                print(
                    f"[{saved + skipped:4d}/{len(dataset):4d}] "
                    f"{file_id:<36} "
                    f"hidden={tuple(sample_hidden.shape)} "
                    f"target={float(file_target):.1f}"
                )

            # ---------------------------------------------------------------
            # GPU memory cleanup
            # ---------------------------------------------------------------

            del wav
            del attention_mask
            del hidden
            del valid_mask

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("CACHE COMPLETE")
    print("=" * 72)

    print(f"Dataset samples:     {len(dataset)}")
    print(f"New cache files:     {saved}")
    print(f"Skipped valid cache:  {skipped}")
    print(f"Cache directory:     {CACHE_ROOT}")

    # Verify every expected cache exists.
    print()
    print("Verifying cache files...")

    verified = 0
    frames = 0

    for _, row in work_df.iterrows():

        file_id = str(row["id"])
        cache_path = cache_path_for_file(file_id)

        if not cache_path.exists():
            raise RuntimeError(
                f"Missing cache file:\n{cache_path}"
            )

        info = verify_cache_file(
            cache_path,
            expected_hidden_size=hidden_size,
        )

        # Target consistency check.
        expected_target = float(row["Overall quality"])

        if abs(info["target"] - expected_target) > 1e-6:
            raise RuntimeError(
                f"Target mismatch for {file_id}: "
                f"cache={info['target']} "
                f"CSV={expected_target}"
            )

        # ID consistency check.
        if info["file_id"] != file_id:
            raise RuntimeError(
                f"file_id mismatch for {file_id}: "
                f"cache={info['file_id']}"
            )

        verified += 1
        frames += info["frames"]

    print(f"Verified files:      {verified}")
    print(f"Total cached frames: {frames}")

    if verified != len(work_df):
        raise RuntimeError(
            f"Verification mismatch: "
            f"{verified} verified vs {len(work_df)} expected."
        )

    print()
    print(">>> WavLM CACHE PASS")
    print()
    print("Next step:")
    print("  1. Run a cached-feature tiny test.")
    print("  2. Compare cached QualityHead output against")
    print("     the original WAV -> WavLM -> QualityHead pipeline.")
    print("  3. Only after that, set CACHE_TINY_ONLY = False")
    print("     and cache all 6,942 samples.")

    dataset.close()


if __name__ == "__main__":
    main()
