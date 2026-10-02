from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"

SEED = 42
VAL_RATIO = 0.20


def get_group(sample_id):
    return str(sample_id).split("-", 1)[0]


df = pd.read_csv(CSV_PATH)

df["sample_id"] = (
    df["id"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.wav$", "", regex=True)
)

df["target"] = pd.to_numeric(
    df["Overall quality"],
    errors="raise",
)

df["group"] = df["sample_id"].map(get_group)


# ============================================================
# REPRODUCE EXACT SAME SPLIT
# ============================================================

groups = df["group"].unique().tolist()

rng = np.random.RandomState(SEED)
rng.shuffle(groups)

group_sizes = df.groupby("group").size().to_dict()

target_val_size = round(len(df) * VAL_RATIO)

selected_groups = []
val_size = 0

for group in groups:
    if val_size >= target_val_size:
        break

    selected_groups.append(group)
    val_size += group_sizes[group]

val_groups = set(selected_groups)

df["split"] = np.where(
    df["group"].isin(val_groups),
    "val",
    "train",
)


# ============================================================
# GROUP STATISTICS
# ============================================================

group_stats = (
    df.groupby(["split", "group"])["target"]
    .agg(
        n="size",
        mean="mean",
        std="std",
        min="min",
        max="max",
    )
    .reset_index()
)

print("=" * 80)
print("GROUP TARGET STATISTICS")
print("=" * 80)

for split in ["train", "val"]:

    g = group_stats[group_stats["split"] == split].copy()

    print("\n" + "-" * 80)
    print(split.upper())
    print("-" * 80)

    print(f"Groups : {len(g)}")
    print(f"Samples: {g['n'].sum()}")

    print("\nGroup mean distribution:")
    print(g["mean"].describe().to_string())

    print("\nGroup std distribution:")
    print(g["std"].describe().to_string())


# ============================================================
# EXTREME GROUPS
# ============================================================

print("\n" + "=" * 80)
print("HIGHEST TARGET-MEAN GROUPS")
print("=" * 80)

print(
    group_stats
    .sort_values("mean", ascending=False)
    .head(25)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 80)
print("LOWEST TARGET-MEAN GROUPS")
print("=" * 80)

print(
    group_stats
    .sort_values("mean", ascending=True)
    .head(25)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# VALIDATION GROUPS ONLY
# ============================================================

val_stats = group_stats[
    group_stats["split"] == "val"
].copy()

print("\n" + "=" * 80)
print("VALIDATION GROUPS WITH EXTREME MEANS")
print("=" * 80)

print("\nHighest:")
print(
    val_stats
    .sort_values("mean", ascending=False)
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nLowest:")
print(
    val_stats
    .sort_values("mean", ascending=True)
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SPECIFIC BAD GROUPS
# ============================================================

interesting = [
    "sys3ad60",
    "sysba29c",
    "sysb1913",
    "syseba96",
    "sys9dead",
    "sys9349e",
    "sys85ef8",
    "sysf2d64",
    "sysa1387",
    "sysnistl",
]

print("\n" + "=" * 80)
print("INTERESTING VALIDATION GROUPS")
print("=" * 80)

print(
    val_stats[
        val_stats["group"].isin(interesting)
    ]
    .sort_values("mean", ascending=False)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)