"""Recompute the Vietnamese augmentation corpus statistics quoted in the paper (Sec. 5).

Reads ``metadata.csv`` + ``labels.jsonl`` of every generated split and prints:
  * clip counts / total duration,
  * per-class clip prevalence (noise / distortion / unnatural_pause / clean),
  * per-class frame prevalence (fraction of audio time covered by fault intervals),
  * parameter ranges actually used by the engine (noise SNR, distortion / pause lengths, noise mode).

Usage:
    python scripts/report_aug_stats.py
    python scripts/report_aug_stats.py --dirs data/vietnamese_aug data/vietnamese_aug_v3/train
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAULTS = ("noise", "distortion", "unnatural_pause")

TRAIN_DIRS = ["data/vietnamese_aug", "data/vietnamese_aug_v3/train"]
DEV_DIRS = ["data/vietnamese_test_v1"]


def load_split(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path / "metadata.csv")
    labels = {}
    with open(path / "labels.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                labels[row["file_id"]] = row["faults"]
    meta["faults"] = meta["file_id"].map(labels)
    if meta["faults"].isna().any():
        raise ValueError(f"{path}: {int(meta['faults'].isna().sum())} clips without labels")
    meta["round"] = path.as_posix().split("data/")[-1]
    return meta


def interval_lengths(df: pd.DataFrame, fault: str) -> np.ndarray:
    return np.array([e - s for fl in df["faults"] for s, e in fl[fault]], dtype=float)


def report(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    total_s = df["duration"].sum()
    print("=" * 78)
    print(f"{name}: {n} clips, {total_s:.1f} s total audio")
    present = {ft: df["faults"].map(lambda fl: len(fl[ft]) > 0) for ft in FAULTS}
    clean = ~(present["noise"] | present["distortion"] | present["unnatural_pause"])
    for ft in FAULTS:
        print(f"  clips with {ft:16s}: {int(present[ft].sum()):5d}  ({100 * present[ft].mean():5.1f}%)")
    print(f"  clean clips                  : {int(clean.sum()):5d}  ({100 * clean.mean():5.1f}%)")
    for ft in FAULTS:
        cover = sum(sum(e - s for s, e in fl[ft]) for fl in df["faults"])
        print(f"  frame prevalence {ft:16s}: {100 * cover / total_s:6.2f}% of audio time")
    for ft in FAULTS:
        ln = interval_lengths(df, ft)
        if len(ln):
            print(f"  {ft:16s} interval length s: min={ln.min():.2f} median={np.median(ln):.2f} max={ln.max():.2f} (n={len(ln)})")
    if "noise_snr_db" in df and df["noise_snr_db"].notna().any():
        snr = df["noise_snr_db"].dropna()
        print(f"  noise SNR dB: min={snr.min():.1f} max={snr.max():.1f}")
    if "noise_mode" in df and df["noise_mode"].notna().any():
        print("  noise_mode counts:", df["noise_mode"].value_counts().to_dict())
    if "distortion_kind" in df and df["distortion_kind"].notna().any():
        print("  distortion_kind counts:", df["distortion_kind"].value_counts().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None, help="Split directories (default: train rounds, then dev).")
    args = ap.parse_args()

    train_dirs = [ROOT / d for d in (args.dirs or TRAIN_DIRS)]
    parts = [load_split(d) for d in train_dirs]
    for df in parts:
        report(f"[{df['round'].iloc[0]}]", df)
    report("TRAIN (all rounds)", pd.concat(parts, ignore_index=True))

    if args.dirs is None:
        dev = [load_split(ROOT / d) for d in DEV_DIRS]
        report("DEV (vietnamese_test_v1)", pd.concat(dev, ignore_index=True))


if __name__ == "__main__":
    main()
