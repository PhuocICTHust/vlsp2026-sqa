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
# False -> cache all 10,558 training samples
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
        .reset_index(drop=True)
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
    target: torch.Tensor,
    file_id: str,
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "hidden": hidden.detach().cpu().contiguous(),
        "valid_mask": valid_mask.detach().cpu().contiguous(),
        "target": float(target.item()),
        "file_id": str(file_id),
    }

    torch.save(payload, path)


# ============================================================================
# VERIFY EXISTING CACHE
# ============================================================================

def verify_cache_file(
    path: Path,
    expected_hidden_size: int,
) -> dict:

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
    print(f"WAV directory:{WAV_DIR}")
    print(f"Cache:        {CACHE_ROOT}")
    print(f"Model:        {MODEL_NAME}")
    print(f"Tiny only:    {CACHE_TINY_ONLY}")
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

    print(f"CSV rows:     {len(df)}")

    # ------------------------------------------------------------------------
    # Index WAV files
    # ------------------------------------------------------------------------

    wav_files = {
        p.name
        for p in WAV_DIR.glob("*.wav")
    }

    print(f"WAV files:    {len(wav_files)}")

    # ------------------------------------------------------------------------
    # Check audio coverage
    # ------------------------------------------------------------------------

    missing_audio = sorted(
        set(df["id"].astype(str))
        - wav_files
    )

    if missing_audio:
        raise RuntimeError(
            f"Missing {len(missing_audio)} audio files.\n"
            f"First examples: {missing_audio[:10]}"
        )

    extra_audio = sorted(
        wav_files
        - set(df["id"].astype(str))
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

    else:

        work_df = (
            df.copy()
            .reset_index(drop=True)
        )

    print()
    print(f"Samples selected: {len(work_df)}")

    # ------------------------------------------------------------------------
    # Cache directory
    # ------------------------------------------------------------------------

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Determine which caches are already valid
    #
    # IMPORTANT:
    # We do this BEFORE loading WavLM / DataLoader so that
    # existing caches are NOT recomputed.
    # ------------------------------------------------------------------------

    valid_cache_ids = []
    pending_ids = []

    for file_id in work_df["id"].astype(str):

        cache_path = cache_path_for_file(file_id)

        if not cache_path.exists():

            pending_ids.append(file_id)
            continue

        try:

            info = verify_cache_file(
                cache_path,
                expected_hidden_size=EXPECTED_HIDDEN_SIZE,
            )

            # Verify file_id consistency too.
            if info["file_id"] != file_id:
                raise RuntimeError(
                    f"file_id mismatch: "
                    f"cache={info['file_id']} "
                    f"expected={file_id}"
                )

            valid_cache_ids.append(file_id)

        except Exception as exc:

            print(
                f"Invalid cache -> recompute: "
                f"{cache_path.name}"
            )
            print(f"  Reason: {exc}")

            pending_ids.append(file_id)

    print()
    print(f"Valid existing cache: {len(valid_cache_ids)}")
    print(f"Need WavLM extraction: {len(pending_ids)}")

    # ------------------------------------------------------------------------
    # If everything is already cached, no need to load WavLM.
    # ------------------------------------------------------------------------

    if not pending_ids:

        print()
        print("All requested cache files are already valid.")
        print("Skipping WavLM extraction.")

        hidden_size = EXPECTED_HIDDEN_SIZE

    else:

        # --------------------------------------------------------------------
        # Build dataframe containing ONLY pending samples
        # --------------------------------------------------------------------

        pending_set = set(pending_ids)

        pending_df = work_df[
            work_df["id"].astype(str).isin(pending_set)
        ].copy()

        pending_df = pending_df.reset_index(drop=True)

        # --------------------------------------------------------------------
        # Dataset
        # --------------------------------------------------------------------

        dataset = QualiSpeechDataset(
            csv_path=CSV_PATH,
            wav_dir=WAV_DIR,
            indices=pending_df.index.tolist(),
        )

        print()
        print(f"Pending dataset: {len(dataset)}")

        # --------------------------------------------------------------------
        # Load frozen WavLM
        # --------------------------------------------------------------------

        print()
        print("Loading frozen WavLM...")

        backbone = WavLMBackbone(
            model_name=MODEL_NAME,
            freeze=True,
        ).to(DEVICE)

        backbone.eval()

        hidden_size = backbone.hidden_size

        print(f"WavLM hidden size: {hidden_size}")

        # --------------------------------------------------------------------
        # Verify WavLM frozen
        # --------------------------------------------------------------------

        trainable_backbone = [
            name
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        ]

        if trainable_backbone:
            raise RuntimeError(
                "WavLM is not fully frozen. "
                f"Trainable parameters found: "
                f"{trainable_backbone[:10]}"
            )

        print("WavLM parameters: FROZEN")

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
        print("Starting WavLM feature extraction...")
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

                attention_mask = batch["attention_mask"].to(
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

                valid_mask = backbone.frame_valid_mask(
                    attention_mask
                )

                # ------------------------------------------------------------
                # Shape checks
                # ------------------------------------------------------------

                if hidden.ndim != 3:
                    raise RuntimeError(
                        f"Expected hidden [B,T,H], "
                        f"got {tuple(hidden.shape)}"
                    )

                if hidden.shape[0] != len(file_ids):
                    raise RuntimeError(
                        f"Batch mismatch: "
                        f"hidden B={hidden.shape[0]}, "
                        f"file_ids={len(file_ids)}"
                    )

                if hidden.shape[1] != valid_mask.shape[1]:
                    raise RuntimeError(
                        f"Frame mismatch: "
                        f"hidden T={hidden.shape[1]}, "
                        f"valid_mask T={valid_mask.shape[1]}"
                    )

                if hidden.shape[2] != hidden_size:
                    raise RuntimeError(
                        f"Hidden size mismatch: "
                        f"{hidden.shape[2]} != {hidden_size}"
                    )

                # ------------------------------------------------------------
                # Save each sample
                # ------------------------------------------------------------

                for i, file_id in enumerate(file_ids):

                    file_target = target[i]

                    cache_path = cache_path_for_file(
                        file_id
                    )

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
                            f"{file_id}: "
                            f"zero valid WavLM frames."
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
                        f"[{saved:4d}/{len(dataset):4d}] "
                        f"{file_id:<36} "
                        f"hidden={tuple(sample_hidden.shape)} "
                        f"target={float(file_target):.1f}"
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
        print(f"New cache files: {saved}")
        print(f"New cached frames: {total_frames}")

    # ------------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL CACHE VERIFICATION")
    print("=" * 72)

    verified = 0
    frames = 0

    for _, row in work_df.iterrows():

        file_id = str(row["id"])

        cache_path = cache_path_for_file(
            file_id
        )

        if not cache_path.exists():
            raise RuntimeError(
                f"Missing cache file:\n{cache_path}"
            )

        info = verify_cache_file(
            cache_path,
            expected_hidden_size=hidden_size,
        )

        # Target consistency
        expected_target = float(
            row["Overall quality"]
        )

        if abs(
            info["target"] - expected_target
        ) > 1e-6:

            raise RuntimeError(
                f"Target mismatch for {file_id}: "
                f"cache={info['target']} "
                f"CSV={expected_target}"
            )

        # File ID consistency
        if info["file_id"] != file_id:

            raise RuntimeError(
                f"file_id mismatch for {file_id}: "
                f"cache={info['file_id']}"
            )

        verified += 1
        frames += info["frames"]

    print(f"Expected files:  {len(work_df)}")
    print(f"Verified files:  {verified}")
    print(f"Total frames:    {frames}")
    print(f"Cache directory: {CACHE_ROOT}")

    if verified != len(work_df):
        raise RuntimeError(
            f"Verification mismatch: "
            f"{verified} verified vs "
            f"{len(work_df)} expected."
        )

    print()
    print("=" * 72)
    print(">>> WavLM CACHE PASS")
    print("=" * 72)

    print()
    print("Next step:")
    print("  1. Run cached-feature tiny test.")
    print("  2. Compare cached QualityHead output against")
    print("     original WAV -> WavLM -> QualityHead.")
    print("  3. Then train on the full cached dataset.")


if __name__ == "__main__":
    main()