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


# Same split as train_cached_full.py
groups = df["group"].unique().tolist()

rng = np.random.RandomState(SEED)
rng.shuffle(groups)

group_sizes = (
    df.groupby("group")
    .size()
    .to_dict()
)

target_val_size = round(
    len(df) * VAL_RATIO
)

selected_groups = []
val_size = 0

for group in groups:
    if val_size >= target_val_size:
        break

    selected_groups.append(group)
    val_size += group_sizes[group]

val_groups = set(selected_groups)

train_df = df[
    ~df["group"].isin(val_groups)
].copy()

val_df = df[
    df["group"].isin(val_groups)
].copy()


print("=" * 70)
print("SPLIT AUDIT")
print("=" * 70)

print(
    f"Train samples: {len(train_df)}"
)

print(
    f"Validation samples: {len(val_df)}"
)

print(
    f"Train groups: {train_df['group'].nunique()}"
)

print(
    f"Validation groups: {val_df['group'].nunique()}"
)

print()
print("=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

for name, part in [
    ("FULL", df),
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
]:

    counts = (
        part["target"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        part["target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print()
    print(name)

    print("Counts:")
    print(counts.to_string())

    print("Percent:")
    print(
        percentages
        .round(2)
        .to_string()
    )


print()
print("=" * 70)
print("SYSNISTL")
print("=" * 70)

sysnistl = val_df[
    val_df["group"] == "sysnistl"
]

print(
    f"Samples: {len(sysnistl)}"
)

print(
    sysnistl["target"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("=" * 70)
print("GROUP SIZES")
print("=" * 70)

print(
    val_df.groupby("group")
    .size()
    .sort_values(ascending=False)
    .head(20)
    .to_string()
)