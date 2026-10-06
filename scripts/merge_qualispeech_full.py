"""
Merge QualiSpeech train, val, and test splits into a unified full dataset.
- Combines train.csv, val.csv, and test.csv into train_full.csv (14,577 rows).
- Extracts and consolidates all 4,019 extra audio files into data/qualispeech/wav/.
- Generates qualispeech_extra_wav.zip containing newly added audio + CSVs for fast server sync.
"""

import os
import shutil
import zipfile
import pandas as pd
from pathlib import Path

BASE_DIR = Path("data/qualispeech")
WAV_DIR = BASE_DIR / "wav"
WAV_PART1_ZIP = BASE_DIR / "wav_part1.zip"
BVCC_WAV_DIR = BASE_DIR / "main" / "DATA" / "wav"

TRAIN_CSV = BASE_DIR / "train.csv"
VAL_CSV = BASE_DIR / "val.csv"
TEST_CSV = BASE_DIR / "test.csv"
FULL_CSV = BASE_DIR / "train_full.csv"

EXTRA_ZIP = Path("qualispeech_extra_wav.zip")


def merge_csvs():
    print("=" * 60)
    print("1. MERGING CSV FILES")
    print("=" * 60)
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    df_train["split_origin"] = "train"
    df_val["split_origin"] = "val"
    df_test["split_origin"] = "test"

    print(f"Train rows: {len(df_train)}")
    print(f"Val rows:   {len(df_val)}")
    print(f"Test rows:  {len(df_test)}")

    df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    print(f"Full rows:  {len(df_full)}")

    assert len(df_full) == len(df_train) + len(df_val) + len(df_test)
    assert df_full["id"].nunique() == len(df_full), "Found duplicate IDs across splits!"

    df_full.to_csv(FULL_CSV, index=False)
    print(f"Saved merged dataset -> {FULL_CSV}")
    return df_full


def consolidate_audio(df_full):
    print("\n" + "=" * 60)
    print("2. CONSOLIDATING AUDIO FILES")
    print("=" * 60)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    existing_wavs = set(os.listdir(WAV_DIR))
    print(f"Already existing in {WAV_DIR}: {len(existing_wavs)}")

    missing_ids = set(df_full["id"]) - existing_wavs
    print(f"Missing audio files to extract/copy: {len(missing_ids)}")

    if not missing_ids:
        print("All audio files already present!")
        return []

    # Map files in wav_part1.zip
    zip_extracted = 0
    if WAV_PART1_ZIP.exists():
        print(f"Extracting missing files from {WAV_PART1_ZIP}...")
        with zipfile.ZipFile(WAV_PART1_ZIP, "r") as z:
            zip_members = {
                os.path.basename(m): m
                for m in z.namelist()
                if m.lower().endswith(".wav")
            }
            for audio_id in list(missing_ids):
                if audio_id in zip_members:
                    member_path = zip_members[audio_id]
                    source = z.open(member_path)
                    target_path = WAV_DIR / audio_id
                    with open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    missing_ids.remove(audio_id)
                    zip_extracted += 1

    print(f"Extracted from zip: {zip_extracted}")
    print(f"Remaining missing: {len(missing_ids)}")

    # Copy from BVCC
    bvcc_copied = 0
    if BVCC_WAV_DIR.exists():
        print(f"Copying missing BVCC files from {BVCC_WAV_DIR}...")
        bvcc_files = set(os.listdir(BVCC_WAV_DIR))
        for audio_id in list(missing_ids):
            if audio_id in bvcc_files:
                src_path = BVCC_WAV_DIR / audio_id
                dst_path = WAV_DIR / audio_id
                shutil.copy2(src_path, dst_path)
                missing_ids.remove(audio_id)
                bvcc_copied += 1

    print(f"Copied from BVCC: {bvcc_copied}")
    print(f"Unresolved missing: {len(missing_ids)}")

    if missing_ids:
        raise RuntimeError(f"Could not find {len(missing_ids)} audio files!")

    current_wavs = set(os.listdir(WAV_DIR))
    newly_added = list(current_wavs - existing_wavs)
    print(f"Total audio files in {WAV_DIR}: {len(current_wavs)}")
    print(f"Newly added audio files: {len(newly_added)}")
    return newly_added


def create_extra_sync_zip(newly_added_ids):
    print("\n" + "=" * 60)
    print("3. PACKAGING qualispeech_extra_wav.zip FOR SERVER SYNC")
    print("=" * 60)
    if EXTRA_ZIP.exists():
        EXTRA_ZIP.unlink()

    print(f"Creating {EXTRA_ZIP} containing {len(newly_added_ids)} wav files + CSVs...")
    with zipfile.ZipFile(EXTRA_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # Add CSVs
        z.write(FULL_CSV, arcname="train_full.csv")
        z.write(VAL_CSV, arcname="val.csv")
        z.write(TEST_CSV, arcname="test.csv")

        # Add only new wavs
        for idx, audio_id in enumerate(newly_added_ids, start=1):
            file_path = WAV_DIR / audio_id
            z.write(file_path, arcname=f"wav/{audio_id}")
            if idx % 1000 == 0 or idx == len(newly_added_ids):
                print(f"  Zipped {idx}/{len(newly_added_ids)} wavs...")

    size_mb = EXTRA_ZIP.stat().st_size / (1024 * 1024)
    print(f"Finished! Package size: {size_mb:.2f} MB -> {EXTRA_ZIP}")


def main():
    df_full = merge_csvs()
    newly_added = consolidate_audio(df_full)
    if newly_added:
        create_extra_sync_zip(newly_added)
    print("\nAll done! Full dataset merged and verified.")


if __name__ == "__main__":
    main()
