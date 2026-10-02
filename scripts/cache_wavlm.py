from __future__ import annotations

import sys
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
WAV_DIR = ROOT / "data" / "qualispeech" / "wav"
CACHE_ROOT = ROOT / "data" / "qualispeech" / "wavlm_cache"


# ============================================================================
# CONFIG
# ============================================================================

SEED = 42

# True  -> only cache the 16-sample tiny subset
# False -> cache all training samples
CACHE_TINY_ONLY = False

SAMPLES_PER_CLASS = {
    1: 3,
    2: 3,
    3: 4,
    4: 3,
    5: 3,
}

BATCH_SIZE = 4

# Keep 0 for Windows compatibility.
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_NAME = "microsoft/wavlm-large"

# WavLM-Large hidden size.
EXPECTED_HIDDEN_SIZE = 1024


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# TINY SUBSET
# ============================================================================

def build_tiny_indices(
    df: pd.DataFrame,
    wav_files: set[str],
) -> pd.DataFrame:

    selected = []

    for quality, n in SAMPLES_PER_CLASS.items():

        candidates = df[
            (df["Overall quality"] == quality)
            & (df["id"].astype(str).isin(wav_files))
        ]

        selected.append(
            candidates.head(n)
        )

    tiny_df = (
        pd.concat(selected)
        .drop_duplicates(subset=["id"])
        .reset_index(drop=False)
    )

    if len(tiny_df) != 16:
        raise RuntimeError(
            f"Tiny dataset should contain 16 samples, "
            f"got {len(tiny_df)}"
        )

    return tiny_df


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
    target: torch.Tensor | float,
    file_id: str,
) -> None:
    """
    Atomically save one cache file.

    The cache is first written to a temporary file and then
    renamed into place. This prevents partially written .pt
    files from being treated as valid caches.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if isinstance(target, torch.Tensor):
        target_value = float(target.item())
    else:
        target_value = float(target)

    payload = {
        "hidden": hidden.detach().cpu().contiguous(),
        "valid_mask": valid_mask.detach().cpu().contiguous(),
        "target": target_value,
        "file_id": str(file_id),
    }

    tmp_path = path.with_suffix(".tmp.pt")

    # Remove stale temporary file if one exists.
    if tmp_path.exists():
        tmp_path.unlink()

    try:

        torch.save(
            payload,
            tmp_path,
        )

        # Make sure the temporary file really exists.
        if not tmp_path.exists():
            raise RuntimeError(
                f"torch.save() did not create temporary cache: "
                f"{tmp_path}"
            )

        # Atomic replacement on the same filesystem.
        tmp_path.replace(path)

    finally:

        # Cleanup if anything failed before replace().
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ============================================================================
# VERIFY EXISTING CACHE
# ============================================================================

def verify_cache_file(
    path: Path,
    expected_hidden_size: int,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Cache file does not exist: {path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"{path}: cache payload must be dict"
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
            f"Cache file {path} is missing keys: "
            f"{sorted(missing)}"
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

    if hidden.shape[0] == 0:
        raise RuntimeError(
            f"{path}: hidden contains zero frames"
        )

    if not torch.isfinite(hidden).all():
        raise RuntimeError(
            f"{path}: hidden contains NaN or Inf"
        )

    target = float(payload["target"])

    if not torch.isfinite(
        torch.tensor(target, dtype=torch.float32)
    ):
        raise RuntimeError(
            f"{path}: target is NaN or Inf"
        )

    file_id = str(payload["file_id"])

    if not file_id:
        raise RuntimeError(
            f"{path}: file_id is empty"
        )

    return {
        "frames": int(hidden.shape[0]),
        "hidden_size": int(hidden.shape[1]),
        "valid_frames": int(
            valid_mask.bool().sum().item()
        ),
        "target": target,
        "file_id": file_id,
    }


# ============================================================================
# VERIFY DATASET MAPPING
# ============================================================================

def verify_pending_dataset_mapping(
    df: pd.DataFrame,
    pending_df: pd.DataFrame,
) -> None:
    """
    Verify that pending_df still refers to the original train.csv rows.

    This is deliberately strict because QualiSpeechDataset receives
    indices into the ORIGINAL CSV.
    """

    for original_index, row in pending_df.iterrows():

        file_id = str(row["id"])

        if original_index not in df.index:
            raise RuntimeError(
                f"Invalid original CSV index: {original_index}"
            )

        original_row = df.loc[original_index]

        original_file_id = str(
            original_row["id"]
        )

        if original_file_id != file_id:
            raise RuntimeError(
                f"Dataset mapping error at CSV index "
                f"{original_index}: "
                f"pending_df id={file_id}, "
                f"original CSV id={original_file_id}"
            )

        pending_target = float(
            row["Overall quality"]
        )

        original_target = float(
            original_row["Overall quality"]
        )

        if abs(
            pending_target - original_target
        ) > 1e-6:

            raise RuntimeError(
                f"Target mapping error for {file_id}: "
                f"pending_df={pending_target}, "
                f"original CSV={original_target}"
            )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    set_seed(SEED)

    print("=" * 72)
    print("QualiSpeech WavLM REPRESENTATION CACHE")
    print("=" * 72)

    print(f"Device:        {DEVICE}")
    print(f"CSV:           {CSV_PATH}")
    print(f"WAV directory: {WAV_DIR}")
    print(f"Cache:         {CACHE_ROOT}")
    print(f"Model:         {MODEL_NAME}")
    print(f"Tiny only:     {CACHE_TINY_ONLY}")
    print()

    # ------------------------------------------------------------------------
    # Check paths
    # ------------------------------------------------------------------------

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    if not WAV_DIR.exists():
        raise FileNotFoundError(
            f"WAV directory not found:\n{WAV_DIR}"
        )

    # ------------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    if "id" not in df.columns:
        raise RuntimeError(
            "train.csv must contain column 'id'."
        )

    if "Overall quality" not in df.columns:
        raise RuntimeError(
            "train.csv must contain column "
            "'Overall quality'."
        )

    print(f"CSV rows:      {len(df)}")

    # ------------------------------------------------------------------------
    # Check duplicate IDs
    # ------------------------------------------------------------------------

    duplicate_ids = (
        df["id"]
        .astype(str)
        .duplicated()
    )

    if duplicate_ids.any():

        duplicates = (
            df.loc[duplicate_ids, "id"]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Duplicate audio IDs found in train.csv.\n"
            f"Examples: {duplicates[:10]}"
        )

    # ------------------------------------------------------------------------
    # Index WAV files
    # ------------------------------------------------------------------------

    wav_files = {
        p.name
        for p in WAV_DIR.glob("*.wav")
    }

    print(f"WAV files:     {len(wav_files)}")

    # ------------------------------------------------------------------------
    # Check audio coverage
    # ------------------------------------------------------------------------

    csv_ids = set(
        df["id"].astype(str)
    )

    missing_audio = sorted(
        csv_ids - wav_files
    )

    if missing_audio:

        raise RuntimeError(
            f"Missing {len(missing_audio)} audio files.\n"
            f"First examples: {missing_audio[:10]}"
        )

    extra_audio = sorted(
        wav_files - csv_ids
    )

    if extra_audio:

        print(
            f"WARNING: {len(extra_audio)} WAV files "
            f"are not referenced by train.csv."
        )

    # ------------------------------------------------------------------------
    # Determine samples
    # ------------------------------------------------------------------------

    if CACHE_TINY_ONLY:

        work_df = build_tiny_indices(
            df,
            wav_files,
        )

        # build_tiny_indices creates an explicit copy of the
        # original CSV index in column "index".
        #
        # Restore the original CSV index for Dataset usage.
        work_df = work_df.set_index(
            "index",
            drop=True,
        )

        print()
        print("Tiny subset selected.")

    else:

        # IMPORTANT:
        # Preserve the original train.csv index.
        work_df = df.copy()

    print(
        f"Samples selected: {len(work_df)}"
    )

    # ------------------------------------------------------------------------
    # Cache directory
    # ------------------------------------------------------------------------

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Determine existing valid caches
    # ------------------------------------------------------------------------

    valid_cache_ids = []
    pending_ids = []

    print()
    print("Checking existing cache files...")

    for file_id in work_df["id"].astype(str):

        cache_path = cache_path_for_file(
            file_id
        )

        if not cache_path.exists():

            pending_ids.append(file_id)
            continue

        try:

            info = verify_cache_file(
                cache_path,
                expected_hidden_size=EXPECTED_HIDDEN_SIZE,
            )

            # Verify file_id consistency.
            if info["file_id"] != file_id:

                raise RuntimeError(
                    f"file_id mismatch: "
                    f"cache={info['file_id']} "
                    f"expected={file_id}"
                )

            valid_cache_ids.append(
                file_id
            )

        except Exception as exc:

            print(
                f"Invalid cache -> recompute: "
                f"{cache_path.name}"
            )

            print(
                f"  Reason: {exc}"
            )

            pending_ids.append(
                file_id
            )

    print()
    print(
        f"Valid existing cache: "
        f"{len(valid_cache_ids)}"
    )

    print(
        f"Need WavLM extraction: "
        f"{len(pending_ids)}"
    )

    # ------------------------------------------------------------------------
    # If everything is cached
    # ------------------------------------------------------------------------

    if not pending_ids:

        print()
        print(
            "All requested cache files "
            "are already valid."
        )

        print(
            "Skipping WavLM extraction."
        )

        hidden_size = (
            EXPECTED_HIDDEN_SIZE
        )

    else:

        # --------------------------------------------------------------------
        # Build dataframe containing ONLY pending samples.
        #
        # IMPORTANT:
        # DO NOT reset_index().
        #
        # QualiSpeechDataset receives indices into the ORIGINAL train.csv.
        # --------------------------------------------------------------------

        pending_set = set(
            pending_ids
        )

        pending_df = work_df[
            work_df["id"]
            .astype(str)
            .isin(pending_set)
        ].copy()

        # CRITICAL:
        # Keep original train.csv indices.
        pending_indices = (
            pending_df.index.tolist()
        )

        # --------------------------------------------------------------------
        # Verify pending selection
        # --------------------------------------------------------------------

        pending_ids_from_df = set(
            pending_df["id"].astype(str)
        )

        if pending_ids_from_df != pending_set:

            missing = (
                pending_set
                - pending_ids_from_df
            )

            extra = (
                pending_ids_from_df
                - pending_set
            )

            raise RuntimeError(
                "Pending ID mismatch.\n"
                f"Missing IDs: "
                f"{sorted(missing)[:10]}\n"
                f"Extra IDs: "
                f"{sorted(extra)[:10]}"
            )

        # --------------------------------------------------------------------
        # Verify that these are ORIGINAL CSV indices.
        # --------------------------------------------------------------------

        verify_pending_dataset_mapping(
            df=df,
            pending_df=pending_df,
        )

        print()
        print(
            f"Pending dataset: "
            f"{len(pending_indices)}"
        )

        print(
            "Original CSV indices preserved: YES"
        )

        # --------------------------------------------------------------------
        # Dataset
        # --------------------------------------------------------------------

        dataset = QualiSpeechDataset(
            csv_path=CSV_PATH,
            wav_dir=WAV_DIR,
            indices=pending_indices,
        )

        # --------------------------------------------------------------------
        # Load frozen WavLM
        # --------------------------------------------------------------------

        print()
        print(
            "Loading frozen WavLM..."
        )

        backbone = WavLMBackbone(
            model_name=MODEL_NAME,
            freeze=True,
        ).to(DEVICE)

        backbone.eval()

        hidden_size = backbone.hidden_size

        print(
            f"WavLM hidden size: "
            f"{hidden_size}"
        )

        # --------------------------------------------------------------------
        # Verify WavLM frozen
        # --------------------------------------------------------------------

        trainable_backbone = [
            name
            for name, parameter
            in backbone.named_parameters()
            if parameter.requires_grad
        ]

        if trainable_backbone:

            raise RuntimeError(
                "WavLM is not fully frozen. "
                f"Trainable parameters found: "
                f"{trainable_backbone[:10]}"
            )

        print(
            "WavLM parameters: FROZEN"
        )

        # --------------------------------------------------------------------
        # DataLoader
        # --------------------------------------------------------------------

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=qualispeech_collate,
        )

        # --------------------------------------------------------------------
        # Feature extraction
        # --------------------------------------------------------------------

        print()
        print(
            "Starting WavLM feature extraction..."
        )

        print()

        saved = 0
        total_frames = 0

        with torch.no_grad():

            for batch_idx, batch in enumerate(
                loader,
                start=1,
            ):

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

                target = batch["target"]
                file_ids = batch["file_id"]

                # ------------------------------------------------------------
                # WavLM
                # ------------------------------------------------------------

                hidden = backbone(
                    wav,
                    attention_mask=attention_mask,
                )

                valid_mask = (
                    backbone.frame_valid_mask(
                        attention_mask
                    )
                )

                # ------------------------------------------------------------
                # Shape checks
                # ------------------------------------------------------------

                if hidden.ndim != 3:

                    raise RuntimeError(
                        f"Expected hidden [B,T,H], "
                        f"got {tuple(hidden.shape)}"
                    )

                if hidden.shape[0] != len(
                    file_ids
                ):

                    raise RuntimeError(
                        f"Batch mismatch: "
                        f"hidden B={hidden.shape[0]}, "
                        f"file_ids={len(file_ids)}"
                    )

                if hidden.shape[1] != (
                    valid_mask.shape[1]
                ):

                    raise RuntimeError(
                        f"Frame mismatch: "
                        f"hidden T={hidden.shape[1]}, "
                        f"valid_mask T="
                        f"{valid_mask.shape[1]}"
                    )

                if hidden.shape[2] != (
                    hidden_size
                ):

                    raise RuntimeError(
                        f"Hidden size mismatch: "
                        f"{hidden.shape[2]} != "
                        f"{hidden_size}"
                    )

                # ------------------------------------------------------------
                # Save each sample
                # ------------------------------------------------------------

                for i, file_id_raw in enumerate(
                    file_ids
                ):

                    file_id = str(
                        file_id_raw
                    )

                    file_target = target[i]

                    # --------------------------------------------------------
                    # CRITICAL DATASET MAPPING CHECK
                    # --------------------------------------------------------

                    matching_rows = df[
                        df["id"]
                        .astype(str)
                        == file_id
                    ]

                    if len(matching_rows) != 1:

                        raise RuntimeError(
                            f"Expected exactly one "
                            f"CSV row for {file_id}, "
                            f"found "
                            f"{len(matching_rows)}"
                        )

                    csv_target = float(
                        matching_rows.iloc[0][
                            "Overall quality"
                        ]
                    )

                    dataset_target = float(
                        file_target
                    )

                    if abs(
                        dataset_target
                        - csv_target
                    ) > 1e-6:

                        raise RuntimeError(
                            "Dataset mapping error "
                            f"for {file_id}: "
                            f"dataset target="
                            f"{dataset_target}, "
                            f"CSV target="
                            f"{csv_target}"
                        )

                    # --------------------------------------------------------
                    # Make sure this file was actually pending.
                    # --------------------------------------------------------

                    if file_id not in pending_set:

                        raise RuntimeError(
                            f"Unexpected file_id "
                            f"returned by Dataset: "
                            f"{file_id}"
                        )

                    # --------------------------------------------------------
                    # Valid frames
                    # --------------------------------------------------------

                    sample_valid = (
                        valid_mask[i].bool()
                    )

                    sample_hidden = (
                        hidden[i][sample_valid]
                    )

                    sample_mask = torch.ones(
                        sample_hidden.shape[0],
                        dtype=torch.bool,
                        device=sample_hidden.device,
                    )

                    # --------------------------------------------------------
                    # Shape checks
                    # --------------------------------------------------------

                    if sample_hidden.ndim != 2:

                        raise RuntimeError(
                            f"{file_id}: "
                            f"expected [T,H], "
                            f"got "
                            f"{tuple(sample_hidden.shape)}"
                        )

                    if sample_hidden.shape[0] == 0:

                        raise RuntimeError(
                            f"{file_id}: "
                            f"zero valid WavLM frames."
                        )

                    if sample_hidden.shape[1] != (
                        EXPECTED_HIDDEN_SIZE
                    ):

                        raise RuntimeError(
                            f"{file_id}: "
                            f"hidden size "
                            f"{sample_hidden.shape[1]} "
                            f"!="
                            f"{EXPECTED_HIDDEN_SIZE}"
                        )

                    # --------------------------------------------------------
                    # Save cache
                    #
                    # Target is explicitly taken from CSV after verifying
                    # Dataset mapping.
                    # --------------------------------------------------------

                    cache_path = (
                        cache_path_for_file(
                            file_id
                        )
                    )

                    save_cache(
                        path=cache_path,
                        hidden=sample_hidden,
                        valid_mask=sample_mask,
                        target=csv_target,
                        file_id=file_id,
                    )

                    # --------------------------------------------------------
                    # Verify the file immediately after saving.
                    # --------------------------------------------------------

                    saved_info = (
                        verify_cache_file(
                            cache_path,
                            expected_hidden_size=(
                                EXPECTED_HIDDEN_SIZE
                            ),
                        )
                    )

                    if (
                        saved_info["file_id"]
                        != file_id
                    ):

                        raise RuntimeError(
                            f"Post-save file_id "
                            f"mismatch for "
                            f"{file_id}: "
                            f"{saved_info['file_id']}"
                        )

                    if abs(
                        saved_info["target"]
                        - csv_target
                    ) > 1e-6:

                        raise RuntimeError(
                            f"Post-save target "
                            f"mismatch for "
                            f"{file_id}: "
                            f"cache="
                            f"{saved_info['target']} "
                            f"CSV={csv_target}"
                        )

                    saved += 1

                    total_frames += (
                        sample_hidden.shape[0]
                    )

                    print(
                        f"[{saved:4d}/"
                        f"{len(dataset):4d}] "
                        f"{file_id:<36} "
                        f"hidden="
                        f"{tuple(sample_hidden.shape)} "
                        f"target="
                        f"{csv_target:.1f}"
                    )

                # ------------------------------------------------------------
                # GPU memory cleanup
                # ------------------------------------------------------------

                del wav
                del attention_mask
                del hidden
                del valid_mask

                if DEVICE.type == "cuda":

                    torch.cuda.empty_cache()

        dataset.close()

        print()
        print(
            f"New cache files: "
            f"{saved}"
        )

        print(
            f"New cached frames: "
            f"{total_frames}"
        )

        # --------------------------------------------------------------------
        # Important:
        # Number of actually saved files must match pending IDs.
        # --------------------------------------------------------------------

        if saved != len(pending_ids):

            raise RuntimeError(
                "Extraction count mismatch: "
                f"saved={saved}, "
                f"pending={len(pending_ids)}"
            )

    # =========================================================================
    # FINAL VERIFICATION
    # =========================================================================

    print()
    print("=" * 72)
    print("FINAL CACHE VERIFICATION")
    print("=" * 72)

    verified = 0
    frames = 0

    for _, row in work_df.iterrows():

        file_id = str(
            row["id"]
        )

        cache_path = (
            cache_path_for_file(
                file_id
            )
        )

        if not cache_path.exists():

            raise RuntimeError(
                f"Missing cache file:\n"
                f"{cache_path}"
            )

        info = verify_cache_file(
            cache_path,
            expected_hidden_size=(
                hidden_size
            ),
        )

        # --------------------------------------------------------------------
        # File ID consistency
        # --------------------------------------------------------------------

        if info["file_id"] != file_id:

            raise RuntimeError(
                f"file_id mismatch for "
                f"{file_id}: "
                f"cache={info['file_id']}"
            )

        # --------------------------------------------------------------------
        # Target consistency
        # --------------------------------------------------------------------

        expected_target = float(
            row["Overall quality"]
        )

        if abs(
            info["target"]
            - expected_target
        ) > 1e-6:

            raise RuntimeError(
                f"Target mismatch for "
                f"{file_id}: "
                f"cache={info['target']} "
                f"CSV={expected_target}"
            )

        # --------------------------------------------------------------------
        # Hidden size
        # --------------------------------------------------------------------

        if info["hidden_size"] != (
            EXPECTED_HIDDEN_SIZE
        ):

            raise RuntimeError(
                f"Hidden size mismatch for "
                f"{file_id}: "
                f"{info['hidden_size']} "
                f"!="
                f"{EXPECTED_HIDDEN_SIZE}"
            )

        # --------------------------------------------------------------------
        # Valid frames
        # --------------------------------------------------------------------

        if info["valid_frames"] <= 0:

            raise RuntimeError(
                f"Cache contains zero "
                f"valid frames for "
                f"{file_id}"
            )

        verified += 1
        frames += info["frames"]

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print(
        f"Expected files:  "
        f"{len(work_df)}"
    )

    print(
        f"Verified files:  "
        f"{verified}"
    )

    print(
        f"Total frames:    "
        f"{frames}"
    )

    print(
        f"Cache directory: "
        f"{CACHE_ROOT}"
    )

    if verified != len(work_df):

        raise RuntimeError(
            "Verification mismatch: "
            f"{verified} verified vs "
            f"{len(work_df)} expected."
        )

    print()
    print("=" * 72)
    print(">>> WavLM CACHE PASS")
    print("=" * 72)

    print()
    print("Next step:")
    print(
        "  1. Run cached-feature tiny test."
    )

    print(
        "  2. Compare cached QualityHead output "
        "against original WAV -> WavLM -> QualityHead."
    )

    print(
        "  3. Then train on the full cached dataset."
    )


if __name__ == "__main__":
    main()