from pathlib import Path
import sys

import torch
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sqa.model import QualityHead
from src.sqa.model import concordance_correlation_coefficient


CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
CACHE_DIR = ROOT / "data" / "qualispeech" / "wavlm_cache"

CHECKPOINT = ROOT / "outputs" / "cached_full" / "best_quality_head.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
VAL_RATIO = 0.20


def get_group(sample_id):
    return str(sample_id).split("-", 1)[0]


def compute_metrics(pred, target):
    pred = torch.tensor(pred, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    ccc = concordance_correlation_coefficient(pred, target).item()

    return mse, mae, ccc


# ============================================================
# LOAD CSV
# ============================================================

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

val_df = df[df["group"].isin(val_groups)].copy()


print("=" * 70)
print("VALIDATION SET")
print("=" * 70)
print(f"Samples: {len(val_df)}")
print(f"Groups : {val_df['group'].nunique()}")


# ============================================================
# LOAD MODEL
# ============================================================

model = QualityHead(hidden_size=1024).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print(f"\nLoaded checkpoint: {CHECKPOINT}")


# ============================================================
# EVALUATE EACH SAMPLE
# ============================================================

results = []

with torch.no_grad():

    for _, row in val_df.iterrows():

        sample_id = row["sample_id"]
        target = float(row["target"])

        cache_path = CACHE_DIR / f"{sample_id}.pt"

        if not cache_path.exists():
            print(f"WARNING: missing cache {sample_id}")
            continue

        item = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        hidden = (
            item["hidden"]
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )

        valid_mask = (
            item["valid_mask"]
            .bool()
            .unsqueeze(0)
            .to(DEVICE)
        )

        pred = model(
            hidden,
            valid_mask,
        )

        pred = float(pred.item())

        results.append(
            {
                "sample_id": sample_id,
                "group": row["group"],
                "target": target,
                "pred": pred,
            }
        )


pred_df = pd.DataFrame(results)


# ============================================================
# OVERALL VALIDATION
# ============================================================

mse, mae, ccc = compute_metrics(
    pred_df["pred"].values,
    pred_df["target"].values,
)

print("\n" + "=" * 70)
print("OVERALL VALIDATION")
print("=" * 70)

print(f"MSE : {mse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"CCC : {ccc:.6f}")


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

target = pred_df["target"]
pred = pred_df["pred"]

print("\nTARGET")
print(target.describe().to_string())

print("\nPREDICTION")
print(pred.describe().to_string())

print("\nPREDICTION RANGES")

print(f"< 2      : {(pred < 2).sum()}")
print(f"2 - <3   : {((pred >= 2) & (pred < 3)).sum()}")
print(f"3 - <4   : {((pred >= 3) & (pred < 4)).sum()}")
print(f">= 4     : {(pred >= 4).sum()}")

print("\nTARGET RANGES")

print(f"< 2      : {(target < 2).sum()}")
print(f"2 - <3   : {((target >= 2) & (target < 3)).sum()}")
print(f"3 - <4   : {((target >= 3) & (target < 4)).sum()}")
print(f">= 4     : {(target >= 4).sum()}")


# ============================================================
# GROUP METRICS
# ============================================================

rows = []

for group, g in pred_df.groupby("group"):

    if len(g) < 2:
        continue

    mse, mae, ccc = compute_metrics(
        g["pred"].values,
        g["target"].values,
    )

    target_mean = g["target"].mean()
    pred_mean = g["pred"].mean()

    rows.append(
        {
            "group": group,
            "n": len(g),
            "mse": mse,
            "mae": mae,
            "ccc": ccc,
            "target_mean": target_mean,
            "pred_mean": pred_mean,
            "mean_error": pred_mean - target_mean,
        }
    )


group_df = pd.DataFrame(rows)


# ============================================================
# WORST GROUPS
# ============================================================

print("\n" + "=" * 70)
print("WORST GROUPS BY MSE")
print("=" * 70)

print(
    group_df
    .sort_values("mse", ascending=False)
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SYSNISTL
# ============================================================

print("\n" + "=" * 70)
print("SYSNISTL")
print("=" * 70)

sys_df = group_df[
    group_df["group"] == "sysnistl"
]

print(
    sys_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# BEST GROUPS
# ============================================================

print("\n" + "=" * 70)
print("BEST GROUPS BY MSE")
print("=" * 70)

print(
    group_df
    .sort_values("mse")
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# GROUP-LEVEL CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("GROUP-LEVEL CORRELATIONS")
print("=" * 70)

corr_target_error = group_df["target_mean"].corr(
    group_df["mean_error"]
)

corr_target_pred = group_df["target_mean"].corr(
    group_df["pred_mean"]
)

corr_target_mse = group_df["target_mean"].corr(
    group_df["mse"]
)

print(
    f"corr(target_mean, mean_error): "
    f"{corr_target_error:.6f}"
)

print(
    f"corr(target_mean, pred_mean):  "
    f"{corr_target_pred:.6f}"
)

print(
    f"corr(target_mean, mse):         "
    f"{corr_target_mse:.6f}"
)


# ============================================================
# EXTREME GROUP ERRORS
# ============================================================

print("\n" + "=" * 70)
print("MOST UNDERPREDICTED GROUPS")
print("=" * 70)

print(
    group_df
    .sort_values("mean_error")
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 70)
print("MOST OVERPREDICTED GROUPS")
print("=" * 70)

print(
    group_df
    .sort_values("mean_error", ascending=False)
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)
