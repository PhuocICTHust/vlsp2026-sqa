#!/usr/bin/env python3
"""
QualiSpeech Dataset Audit for VLSP 2026 Speech Quality Assessment.

This script audits the metadata currently used by the repository and the
repository's split/group conventions.

Main checks
-----------
1. Dataset schema / missing values.
2. ID format / uniqueness.
3. Exact metadata-duplicate candidates (excluding derived audit columns).
4. Current repository group/source convention.
5. Two split implementations currently present in the repository:
   - group-disjoint split used by check_split.py / analyze_group_targets.py /
     eval_by_group.py
   - source-balanced IID split used by train_cached_full.py
   The script reports whether they are consistent or not.
6. Group-size and target-distribution statistics.
7. Conservative audit of the three fault-description columns.
8. Strict initial Unnatural pause categorization for parser handoff.
9. Raw annotation value-frequency exports for all fault-description columns.
10. Optional audio inventory, file metadata, duration and SHA-256 checks.
11. Machine-readable PASS / WARN / HIGH-PRIORITY findings in the report.

Important terminology
---------------------
- `group` is the repository's current prefix-before-hyphen heuristic.
- This script does NOT claim that `group` is a final semantic source_group_id.
- `time_range_candidate` is NOT a confirmed localized interval.
- Exact metadata duplicates are NOT proof of duplicate audio.
- Audio-level duplicate checking requires an audio archive/directory.

Usage
-----
python scripts/dataset_audit.py
python scripts/dataset_audit.py \
    --csv data/qualispeech/train.csv \
    --report reports/dataset_audit.md

Optional audio audit:
python scripts/dataset_audit.py \
    --csv data/qualispeech/train.csv \
    --wav-dir data/qualispeech/wav \
    --audio-hash

Requirements: pandas, numpy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import re
import sys
import wave
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "qualispeech" / "train.csv"
DEFAULT_REPORT = ROOT / "reports" / "dataset_audit.md"

SEED = 42
VAL_RATIO = 0.20
QUALITY_COLUMN = "Overall quality"
ID_COLUMN = "id"

FAULT_COLUMNS = {
    "noise": "Noise Description",
    "distortion": "Distortion description",
    "unnatural_pause": "Unnatural pause",
}

REQUIRED_COLUMNS = {
    ID_COLUMN,
    QUALITY_COLUMN,
    *FAULT_COLUMNS.values(),
}

# These strings are deliberately conservative and are used for audit only.
EXPLICIT_ABSENT_PATTERNS = {
    "noise": {
        "not noticeable",
        "not noticeable.",
        "none",
        "no noise",
        "no noticeable noise",
        "no background noise",
    },
    "distortion": {
        "not distorted",
        "not distorted.",
        "none",
        "no distortion",
    },
    "unnatural_pause": {
        "very smooth",
        "very smooth.",
        "none",
        "no unnatural pause",
        "no pause",
    },
}

# Obvious cross-field phrases worth flagging for manual review.
# This is NOT a label correction rule.
CROSS_FIELD_SUSPECT_VALUES = {
    "noise": {
        "not distorted",
        "very smooth",
    },
    "distortion": {
        "very smooth",
        "not noticeable",
    },
    "unnatural_pause": {
        "not distorted",
        "not noticeable",
    },
}

# Broad range candidate detector. Audit only.
TIME_RANGE_RE = re.compile(
    r"(?<!\d)"
    r"(\d+(?:\.\d+)?)"
    r"\s*(?:s|sec|secs|second|seconds)?"
    r"\s*(?:-|–|—|~|to)\s*"
    r"(\d+(?:\.\d+)?)"
    r"\s*(?:s|sec|secs|second|seconds)?"
    r"(?!\d)",
    flags=re.IGNORECASE,
)

SINGLE_TIME_RE = re.compile(
    r"(?<!\d)"
    r"\d+(?:\.\d+)?"
    r"\s*(?:s|sec|secs|second|seconds)"
    r"(?!\w)",
    flags=re.IGNORECASE,
)

ID_PATTERN = re.compile(
    r"^(?P<prefix>[^-]+)-(?P<suffix>[^-]+)\.wav$",
    flags=re.IGNORECASE,
)

SUFFIX_NUMERIC_RE = re.compile(r"^\d{7}$")
SUFFIX_BVCC_RE = re.compile(r"^utt[0-9a-f]{7}$", flags=re.IGNORECASE)

SOURCE_ORDER = [
    "bvcc",
    "nisqa_sim",
    "gigaspeech",
    "nisqa_live",
    "recent_tts",
]

# This mirrors train_cached_full.py exactly as of the uploaded repository.
EXPECTED_SOURCE_COUNTS = {
    "bvcc": 4546,
    "nisqa_sim": 2883,
    "gigaspeech": 2000,
    "nisqa_live": 529,
    "recent_tts": 600,
}


# ============================================================================
# Utilities
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit QualiSpeech metadata and repository split/group conventions."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"QualiSpeech CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Markdown report path (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--frequency-dir",
        type=Path,
        default=ROOT / "reports" / "annotation_frequency",
        help="Directory for full annotation value-frequency CSVs.",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=None,
        help="Optional directory containing WAV files.",
    )
    parser.add_argument(
        "--wav-zip",
        type=Path,
        default=None,
        help="Optional ZIP containing WAV files.",
    )
    parser.add_argument(
        "--audio-hash",
        action="store_true",
        help="When audio is supplied, compute SHA-256 of WAV byte contents.",
    )
    parser.add_argument(
        "--top-groups",
        type=int,
        default=20,
        help="Number of largest/smallest groups shown.",
    )
    parser.add_argument(
        "--top-duplicates",
        type=int,
        default=20,
        help="Number of duplicate metadata groups shown.",
    )
    return parser.parse_args()


def clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def strip_wav_suffix(value: str) -> str:
    return re.sub(r"\.wav$", "", value, flags=re.IGNORECASE)


def get_group(sample_id: str) -> str:
    """Current repository heuristic: prefix before the first '-'."""
    if "-" not in sample_id:
        return ""
    return sample_id.split("-", 1)[0].strip()


def get_source(group: str) -> str:
    """Mirror source mapping in train_cached_full.py."""
    group = str(group).strip().lower()

    if group == "sysnists":
        return "nisqa_sim"
    if group == "sysnistl":
        return "nisqa_live"
    if group == "sys0giga":
        return "gigaspeech"
    if (
        group.startswith("sys")
        and group.endswith("tts")
        and group[3:-3].isdigit()
    ):
        return "recent_tts"
    return "bvcc"


def safe_pct(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den) * 100.0


def md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def fmt_number(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NaN"
        return f"{float(value):.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)

    if view.empty:
        return "_No rows._"

    headers = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(md_escape(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in view.iterrows():
        cells = [md_escape(fmt_number(v)) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def add_status(statuses: list[dict], level: str, item: str, detail: str) -> None:
    statuses.append({"level": level, "item": item, "detail": detail})


# ============================================================================
# Split implementations
# ============================================================================


def group_disjoint_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror check_split.py / analyze_group_targets.py / eval_by_group.py."""
    work = df.copy()
    groups = work["group"].drop_duplicates().tolist()

    rng = np.random.RandomState(SEED)
    rng.shuffle(groups)

    group_sizes = work.groupby("group").size().to_dict()
    target_val_size = round(len(work) * VAL_RATIO)

    selected_groups: list[str] = []
    val_size = 0
    for group in groups:
        if val_size >= target_val_size:
            break
        selected_groups.append(group)
        val_size += int(group_sizes[group])

    val_groups = set(selected_groups)
    train_df = work.loc[~work["group"].isin(val_groups)].copy()
    val_df = work.loc[work["group"].isin(val_groups)].copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    return train_df, val_df


def source_balanced_iid_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror train_cached_full.py make_source_balanced_split()."""
    rng = np.random.RandomState(SEED)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []

    for source in SOURCE_ORDER:
        source_df = df[df["source"] == source].copy()
        if source_df.empty:
            raise RuntimeError(f"Source '{source}' has zero samples.")

        source_train_parts: list[pd.DataFrame] = []
        source_val_parts: list[pd.DataFrame] = []

        for target_value in sorted(source_df["target"].unique()):
            class_df = source_df[source_df["target"] == target_value].copy()
            indices = np.arange(len(class_df))
            rng.shuffle(indices)

            n_val = int(round(len(class_df) * VAL_RATIO))
            if len(class_df) >= 2:
                n_val = max(1, min(n_val, len(class_df) - 1))
            else:
                n_val = 0

            source_val_parts.append(class_df.iloc[indices[:n_val]])
            source_train_parts.append(class_df.iloc[indices[n_val:]])

        source_train = pd.concat(source_train_parts, ignore_index=True)
        source_val = pd.concat(source_val_parts, ignore_index=True)

        source_train = source_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        source_val = source_val.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

        train_parts.append(source_train)
        val_parts.append(source_val)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"

    return train_df, val_df


def summarize_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    train_groups = set(train_df["group"])
    val_groups = set(val_df["group"])
    overlap = sorted(train_groups & val_groups)

    source_table = (
        pd.DataFrame(
            {
                "train": train_df["source"].value_counts(),
                "val": val_df["source"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    source_table["total"] = source_table["train"] + source_table["val"]
    source_table["val_ratio_pct"] = (
        source_table["val"] / source_table["total"] * 100.0
    )
    source_table = source_table.reindex(SOURCE_ORDER).fillna(0)

    target_rows: list[dict] = []
    for split_name, part in [("FULL", pd.concat([train_df, val_df], ignore_index=True)),
                             ("TRAIN", train_df),
                             ("VALIDATION", val_df)]:
        counts = part["target"].value_counts().sort_index()
        for target, count in counts.items():
            target_rows.append(
                {
                    "split": split_name,
                    "target": target,
                    "count": int(count),
                    "pct": safe_pct(count, len(part)),
                }
            )

    # Group split ratio audit.
    combined = pd.concat(
        [train_df.assign(_split="train"), val_df.assign(_split="val")],
        ignore_index=True,
    )
    group_split = (
        combined.groupby(["group", "source", "_split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "train" not in group_split.columns:
        group_split["train"] = 0
    if "val" not in group_split.columns:
        group_split["val"] = 0
    group_split["total"] = group_split["train"] + group_split["val"]
    group_split["val_ratio_pct"] = group_split["val"] / group_split["total"] * 100.0

    return {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "group_overlap": overlap,
        "source_table": source_table.reset_index(names="source"),
        "target_distribution": pd.DataFrame(target_rows),
        "group_split": group_split.sort_values("total", ascending=False),
    }


# ============================================================================
# Audits
# ============================================================================


def audit_schema(df: pd.DataFrame) -> dict:
    columns = [str(c) for c in df.columns]
    missing_required = sorted(REQUIRED_COLUMNS - set(columns))

    table = pd.DataFrame(
        {
            "column": columns,
            "dtype": [str(df[c].dtype) for c in columns],
            "missing_count": [int(df[c].isna().sum()) for c in columns],
            "missing_pct": [safe_pct(df[c].isna().sum(), len(df)) for c in columns],
            "unique_count": [int(df[c].nunique(dropna=False)) for c in columns],
        }
    )
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "missing_required": missing_required,
        "table": table,
    }


def audit_ids(df: pd.DataFrame) -> dict:
    ids = df[ID_COLUMN].map(clean_id)
    duplicate_mask = ids.duplicated(keep=False)
    blank_mask = ids.eq("")

    parsed = ids.map(lambda x: ID_PATTERN.match(x) if x else None)
    malformed_mask = parsed.map(lambda m: m is None)

    prefix = ids.map(lambda x: m.group("prefix") if (m := ID_PATTERN.match(x)) else "")
    suffix = ids.map(lambda x: m.group("suffix") if (m := ID_PATTERN.match(x)) else "")

    suffix_pattern = suffix.map(
        lambda x: (
            "7-digit numeric" if SUFFIX_NUMERIC_RE.fullmatch(x)
            else "utt+7-hex" if SUFFIX_BVCC_RE.fullmatch(x)
            else "other"
        )
    )

    duplicates = (
        ids[duplicate_mask]
        .value_counts()
        .rename_axis("id")
        .reset_index(name="count")
        .sort_values(["count", "id"], ascending=[False, True])
    )

    prefix_table = prefix.value_counts().rename_axis("group").reset_index(name="samples")

    return {
        "total": len(ids),
        "unique": int(ids.nunique(dropna=False)),
        "duplicate_rows": int(duplicate_mask.sum()),
        "blank_rows": int(blank_mask.sum()),
        "malformed_rows": int(malformed_mask.sum()),
        "no_hyphen_rows": int(ids.str.count("-").eq(0).sum()),
        "prefix_table": prefix_table,
        "suffix_pattern_table": suffix_pattern.value_counts()
        .rename_axis("suffix_pattern")
        .reset_index(name="count"),
        "duplicates": duplicates,
    }


def make_metadata_key_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    excluded = {ID_COLUMN, "sample_id", "group", "source", "target", "split"}
    metadata_cols = [c for c in df.columns if c not in excluded]

    # Convert NaN/NA to a stable sentinel and strings to avoid dtype-only differences.
    work = df[metadata_cols].copy()
    for col in metadata_cols:
        work[col] = work[col].map(lambda x: "<NA>" if pd.isna(x) else str(x))
    return work, metadata_cols


def audit_metadata_duplicates(df: pd.DataFrame) -> dict:
    work, metadata_cols = make_metadata_key_frame(df)
    duplicated_mask = work.duplicated(keep=False)

    key = work.astype(str).agg("\x1f".join, axis=1)
    key_counts = key.value_counts()
    duplicate_keys = set(key_counts[key_counts > 1].index)

    details = df.loc[duplicated_mask, [ID_COLUMN, "group", "split"]].copy()
    details["_key"] = key.loc[duplicated_mask].to_numpy()

    rows: list[dict] = []
    for k in sorted(duplicate_keys, key=str):
        sub = details[details["_key"] == k].sort_values(ID_COLUMN)
        rows.append(
            {
                "duplicate_rows": len(sub),
                "groups": sub["group"].nunique(),
                "splits": sub["split"].nunique(),
                "group_ids": ", ".join(sorted(sub["group"].astype(str).unique())[:10]),
                "sample_ids": ", ".join(sorted(sub[ID_COLUMN].astype(str))[:5]),
            }
        )

    duplicate_groups = (
        pd.DataFrame(rows).sort_values(
            ["duplicate_rows", "groups", "splits", "group_ids", "sample_ids"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        if rows
        else pd.DataFrame(columns=["duplicate_rows", "groups", "splits", "group_ids", "sample_ids"])
    )

    return {
        "metadata_columns": metadata_cols,
        "duplicate_rows": int(duplicated_mask.sum()),
        "duplicate_groups": int(len(duplicate_groups)),
        "cross_split_groups": int((duplicate_groups["splits"] > 1).sum()) if not duplicate_groups.empty else 0,
        "cross_split_rows": int(duplicate_groups.loc[duplicate_groups["splits"] > 1, "duplicate_rows"].sum()) if not duplicate_groups.empty else 0,
        "cross_group_groups": int((duplicate_groups["groups"] > 1).sum()) if not duplicate_groups.empty else 0,
        "cross_group_rows": int(duplicate_groups.loc[duplicate_groups["groups"] > 1, "duplicate_rows"].sum()) if not duplicate_groups.empty else 0,
        "table": duplicate_groups,
    }


def audit_source_mapping(df: pd.DataFrame) -> dict:
    source_counts = (
        df["source"].value_counts()
        .reindex(SOURCE_ORDER, fill_value=0)
        .rename_axis("source")
        .reset_index(name="actual")
    )
    source_counts["expected_in_repo"] = source_counts["source"].map(EXPECTED_SOURCE_COUNTS)
    source_counts["matches_repo_expectation"] = (
        source_counts["actual"] == source_counts["expected_in_repo"]
    )

    group_source = (
        df.groupby(["group", "source"])
        .size()
        .rename("samples")
        .reset_index()
    )

    source_group_counts = (
        df.groupby("source")["group"].nunique()
        .reindex(SOURCE_ORDER, fill_value=0)
        .rename("groups")
        .reset_index()
    )

    return {
        "counts": source_counts,
        "group_source": group_source,
        "source_group_counts": source_group_counts,
    }


def audit_group_structure(df: pd.DataFrame, top_n: int) -> dict:
    sizes = (
        df.groupby(["group", "source"])
        .size()
        .rename("samples")
        .reset_index()
        .sort_values(["samples", "group"], ascending=[False, True])
    )
    stats = sizes["samples"].describe()

    bins = pd.cut(
        sizes["samples"],
        bins=[-np.inf, 10, 20, 50, 100, 500, np.inf],
        labels=["<=10", "11-20", "21-50", "51-100", "101-500", ">500"],
        include_lowest=True,
    )
    bin_table = bins.value_counts().sort_index().rename_axis("size_bin").reset_index(name="group_count")

    return {
        "n_groups": int(len(sizes)),
        "sizes": sizes,
        "stats": stats,
        "bin_table": bin_table,
        "largest": sizes.head(top_n),
        "smallest": sizes.sort_values(["samples", "group"]).head(top_n),
    }


def audit_quality(df: pd.DataFrame) -> dict:
    numeric = pd.to_numeric(df[QUALITY_COLUMN], errors="coerce")
    invalid_numeric = int(numeric.isna().sum())
    out_of_range = int(((numeric < 1) | (numeric > 5)).fillna(False).sum())

    dist = (
        numeric.value_counts(dropna=False)
        .sort_index()
        .rename_axis("target")
        .reset_index(name="count")
    )
    dist["pct"] = dist["count"].map(lambda x: safe_pct(x, len(df)))

    return {
        "invalid_numeric": invalid_numeric,
        "out_of_range": out_of_range,
        "min": None if numeric.dropna().empty else float(numeric.min()),
        "max": None if numeric.dropna().empty else float(numeric.max()),
        "mean": None if numeric.dropna().empty else float(numeric.mean()),
        "std": None if numeric.dropna().empty else float(numeric.std()),
        "distribution": dist,
    }


def normalize_annotation(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def classify_fault_annotation(fault_type: str, value: object) -> str:
    text = normalize_annotation(value)
    if not text:
        return "missing_or_empty"

    if text in EXPLICIT_ABSENT_PATTERNS[fault_type]:
        return "explicit_absent"

    if TIME_RANGE_RE.search(text):
        return "time_range_candidate"

    if SINGLE_TIME_RE.search(text):
        return "single_time_candidate"

    # Possible absence-like / field-misaligned values. Audit only.
    no_fault_words = {
        "noise": {"no", "none", "not noticeable"},
        "distortion": {"no", "none", "not distorted"},
        "unnatural_pause": {"no", "none", "very smooth"},
    }
    if any(token in text for token in no_fault_words[fault_type]):
        return "possible_absent_needs_review"

    return "other_description"



# ---------------------------------------------------------------------------
# Strict initial Unnatural Pause categorization for pipeline handoff.
# ---------------------------------------------------------------------------

STRICT_UNNATURAL_TOTALS = {
    "no_fault": 5907,
    "regex_parseable": 4550,
    "single_ambiguous": 56,
    "abnormal": 45,
}

STRICT_RANGE_RE = re.compile(
    r"(?<!\d)"
    r"(\d+(?:[\.,]\d+)?)"
    r"\s*(?:s|sec|secs|second|seconds)?"
    r"\s*(?:-|–|—|~|to)\s*"
    r"(\d+(?:[\.,]\d+)?)"
    r"\s*(?:s|sec|secs|second|seconds)?"
    r"(?!\d)",
    flags=re.IGNORECASE,
)

STRICT_SINGLE_FULL_RE = re.compile(
    r"^\s*(\d+(?:[\.,]\d+)?)"
    r"\s*(?:s|sec|secs|second|seconds)"
    r"\s*[\.,]?\s*$",
    flags=re.IGNORECASE,
)


def is_strict_no_fault_pause(value: object) -> bool:
    """Match the team-specified strict no-fault bucket conservatively.

    The supplied pipeline specification expects 5907 no-fault rows.  In the
    current CSV this corresponds to smooth-like descriptions and
    'not noticeable' variants. This is audit categorization only; it is not
    the final fault-status resolver.
    """
    text = normalize_annotation(value)
    return bool(text) and ("smooth" in text or "not noticeable" in text)


def strict_unnatural_pause_category(value: object) -> str:
    """Categorize Unnatural pause for the *initial strict* pipeline audit.

    Policy is intentionally conservative:
      1. smooth/not-noticeable -> no_fault
      2. explicit range expression -> regex_parseable, except conjunction-linked
         multi-ranges containing ' and ' are held as residual for parser-policy
         discussion rather than silently accepting them in this strict baseline
      3. one standalone timestamp -> single_ambiguous
      4. everything else -> abnormal

    This policy reproduces the counts stated in the team's current guide
    (5907 / 4550 / 56 / 45) on the current train.csv. The guide also gives an
    example of parsing multiple intervals joined by 'and'; therefore the strict
    bucket is deliberately provisional and should be replaced/aligned when Đạt
    publishes the deterministic parser contract.
    """
    text = normalize_annotation(value)
    if not text:
        return "abnormal"
    if is_strict_no_fault_pause(text):
        return "no_fault"
    if STRICT_RANGE_RE.search(text) and " and " not in text:
        return "regex_parseable"
    if STRICT_SINGLE_FULL_RE.match(text):
        return "single_ambiguous"
    return "abnormal"


def audit_strict_unnatural_pause(df: pd.DataFrame) -> dict:
    categories = df[FAULT_COLUMNS["unnatural_pause"]].map(strict_unnatural_pause_category)
    counts = categories.value_counts().reindex(
        ["no_fault", "regex_parseable", "single_ambiguous", "abnormal"],
        fill_value=0,
    )
    summary = pd.DataFrame(
        {
            "category": counts.index,
            "count": counts.values,
        }
    )
    summary["pct"] = summary["count"].map(lambda x: safe_pct(x, len(df)))
    summary["guide_expected"] = summary["category"].map(STRICT_UNNATURAL_TOTALS)
    summary["delta"] = summary["count"] - summary["guide_expected"]

    tmp = df[[ID_COLUMN, FAULT_COLUMNS["unnatural_pause"], "split", "group", "source"]].copy()
    tmp["strict_category"] = categories.values

    return {
        "summary": summary,
        "categories": categories,
        "examples": (
            tmp.sort_values(["strict_category", ID_COLUMN])
            .groupby("strict_category", group_keys=False)
            .head(10)
        ),
        "matches_guide": bool((summary["delta"] == 0).all()),
    }


def write_annotation_frequency_tables(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for fault_type, column in FAULT_COLUMNS.items():
        freq = (
            df[column]
            .fillna("")
            .astype(str)
            .map(lambda x: x.strip())
            .value_counts(dropna=False)
            .rename_axis("raw_value")
            .reset_index(name="count")
        )
        freq["pct"] = freq["count"].map(lambda x: safe_pct(x, len(df)))
        path = output_dir / f"{fault_type}_value_frequency.csv"
        freq.to_csv(path, index=False)
        outputs[fault_type] = path
    return outputs


def audit_fault_annotations(df: pd.DataFrame, examples_per_status: int = 5) -> dict:
    rows = []
    examples: dict[str, pd.DataFrame] = {}

    for fault_type, column in FAULT_COLUMNS.items():
        status = df[column].map(lambda x: classify_fault_annotation(fault_type, x))
        counts = status.value_counts()
        table = pd.DataFrame(
            {
                "fault_type": fault_type,
                "audit_status": counts.index,
                "count": counts.values,
            }
        )
        table["pct"] = table["count"].map(lambda x: safe_pct(x, len(df)))
        rows.append(table)

        tmp = df[[ID_COLUMN, column, "split", "group", "source"]].copy()
        tmp["_audit_status"] = status.values
        ex = (
            tmp.sort_values(["_audit_status", ID_COLUMN])
            .groupby("_audit_status", group_keys=False)
            .head(examples_per_status)
        )
        examples[fault_type] = ex

    # Strong, field-specific anomaly candidates.
    mismatch_rows = []
    for fault_type, column in FAULT_COLUMNS.items():
        normalized = df[column].map(normalize_annotation)
        mask = normalized.isin(CROSS_FIELD_SUSPECT_VALUES[fault_type])
        for _, row in df.loc[mask, [ID_COLUMN, column, "split", "group", "source"]].iterrows():
            mismatch_rows.append(
                {
                    "fault_type": fault_type,
                    "id": row[ID_COLUMN],
                    "value": row[column],
                    "split": row["split"],
                    "group": row["group"],
                    "source": row["source"],
                }
            )
    mismatch_table = pd.DataFrame(mismatch_rows)

    return {
        "summary": pd.concat(rows, ignore_index=True),
        "examples": examples,
        "cross_field_suspects": mismatch_table,
    }


def audit_group_target_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "group", "source"])["target"]
        .agg(n="size", mean="mean", std="std", min="min", max="max")
        .reset_index()
    )


def audit_cross_split_metadata(duplicates: dict) -> dict:
    return {
        "group_count": duplicates["cross_split_groups"],
        "row_count": duplicates["cross_split_rows"],
    }


# ============================================================================
# Optional audio audit
# ============================================================================


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_wav_bytes(data: bytes) -> dict:
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        nframes = wav.getnframes()
        duration = nframes / rate if rate else 0.0
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width": sample_width,
        "frames": nframes,
        "duration": duration,
    }


def audit_audio_inventory(
    df: pd.DataFrame,
    wav_dir: Path | None,
    wav_zip: Path | None,
    compute_hash: bool,
) -> dict | None:
    if wav_dir is not None and wav_zip is not None:
        raise ValueError("Provide only one of --wav-dir or --wav-zip.")
    if wav_dir is None and wav_zip is None:
        return None

    csv_ids = set(df[ID_COLUMN].map(clean_id))
    audio_ids: set[str] = set()
    duplicate_basenames: list[str] = []
    invalid_files: list[str] = []
    audio_rows: list[dict] = []

    def consume(name: str, data: bytes, source_name: str) -> None:
        basename = Path(name).name
        if basename in audio_ids:
            duplicate_basenames.append(basename)
        audio_ids.add(basename)
        try:
            info = inspect_wav_bytes(data)
            info["id"] = basename
            info["source_path"] = source_name
            info["sha256"] = sha256_bytes(data) if compute_hash else ""
            audio_rows.append(info)
        except (wave.Error, OSError, EOFError):
            invalid_files.append(source_name)

    if wav_dir is not None:
        if not wav_dir.exists():
            raise FileNotFoundError(f"WAV directory not found: {wav_dir}")
        for path in sorted(wav_dir.rglob("*.wav")):
            consume(path.name, path.read_bytes(), str(path))
    else:
        if not wav_zip.exists():
            raise FileNotFoundError(f"WAV ZIP not found: {wav_zip}")
        with zipfile.ZipFile(wav_zip, "r") as zf:
            for member in zf.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".wav"):
                    continue
                try:
                    consume(member.filename, zf.read(member), member.filename)
                except (KeyError, RuntimeError, zipfile.BadZipFile):
                    invalid_files.append(member.filename)

    audio_table = pd.DataFrame(audio_rows)
    missing_audio = sorted(csv_ids - audio_ids)
    orphan_audio = sorted(audio_ids - csv_ids)

    duration_stats = {}
    if not audio_table.empty:
        duration_stats = {
            "count": int(len(audio_table)),
            "min": float(audio_table["duration"].min()),
            "mean": float(audio_table["duration"].mean()),
            "median": float(audio_table["duration"].median()),
            "max": float(audio_table["duration"].max()),
        }

    duplicate_hash_table = pd.DataFrame()
    if compute_hash and not audio_table.empty:
        dup_hash = audio_table[audio_table.duplicated("sha256", keep=False)].copy()
        if not dup_hash.empty:
            duplicate_hash_table = (
                dup_hash.groupby("sha256")["id"]
                .agg(list)
                .rename("ids")
                .reset_index()
            )
            duplicate_hash_table["count"] = duplicate_hash_table["ids"].map(len)
            duplicate_hash_table = duplicate_hash_table.sort_values("count", ascending=False)

    return {
        "audio_count": len(audio_ids),
        "csv_count": len(csv_ids),
        "missing_audio": missing_audio,
        "orphan_audio": orphan_audio,
        "duplicate_basenames": duplicate_basenames,
        "invalid_files": invalid_files,
        "audio_table": audio_table,
        "duration_stats": duration_stats,
        "duplicate_hash_table": duplicate_hash_table,
    }


# ============================================================================
# Findings / report
# ============================================================================


def build_findings(
    schema: dict,
    ids: dict,
    metadata_dups: dict,
    source_audit: dict,
    group_audit: dict,
    group_split: dict,
    training_split: dict,
    quality: dict,
    fault_audit: dict,
    strict_pause_audit: dict,
    audio_audit: dict | None,
) -> list[dict]:
    statuses: list[dict] = []

    # Schema / IDs.
    if schema["missing_required"]:
        add_status(statuses, "HIGH-PRIORITY", "required_columns", str(schema["missing_required"]))
    else:
        add_status(statuses, "PASS", "required_columns", "All required audit columns are present.")

    if ids["duplicate_rows"] == 0 and ids["blank_rows"] == 0:
        add_status(statuses, "PASS", "id_uniqueness", "No duplicate or blank ID rows.")
    else:
        add_status(statuses, "HIGH-PRIORITY", "id_uniqueness", "Duplicate/blank IDs require investigation.")

    if ids["malformed_rows"] == 0:
        add_status(statuses, "PASS", "id_format", "All IDs match <prefix>-<suffix>.wav.")
    else:
        add_status(statuses, "WARN", "id_format", f"{ids['malformed_rows']} IDs do not match the expected pattern.")

    # Metadata duplicates.
    if metadata_dups["duplicate_rows"] == 0:
        add_status(statuses, "PASS", "metadata_duplicates", "No exact metadata duplicates across original metadata columns.")
    else:
        add_status(
            statuses,
            "WARN",
            "metadata_duplicates",
            f"{metadata_dups['duplicate_rows']} rows in {metadata_dups['duplicate_groups']} exact-metadata duplicate groups; not proof of audio duplication.",
        )

    if metadata_dups["cross_split_groups"] > 0:
        add_status(
            statuses,
            "WARN",
            "cross_split_metadata_repetition",
            f"{metadata_dups['cross_split_groups']} metadata keys ({metadata_dups['cross_split_rows']} rows) appear in both splits under the audited split definition; inspect as metadata/template repetition, not automatic leakage.",
        )

    # Source mapping.
    if bool(source_audit["counts"]["matches_repo_expectation"].all()):
        add_status(statuses, "PASS", "source_composition", "Observed source counts match the repository's hard-coded expected composition.")
    else:
        add_status(statuses, "WARN", "source_composition", "Observed source counts differ from the repository's expected composition.")

    # Group-disjoint split.
    if len(group_split["group_overlap"]) == 0:
        add_status(statuses, "PASS", "group_disjoint_split", "No inferred group overlaps between train and validation in the group-disjoint implementation.")
    else:
        add_status(statuses, "HIGH-PRIORITY", "group_disjoint_split", f"{len(group_split['group_overlap'])} inferred groups overlap between train/validation.")

    # Actual training split consistency.
    add_status(
        statuses,
        "HIGH-PRIORITY" if (
            group_split["train_rows"] != training_split["train_rows"]
            or group_split["val_rows"] != training_split["val_rows"]
            or len(training_split["group_overlap"]) != 0
        ) else "PASS",
        "repository_split_consistency",
        (
            "check_split.py/analyze_group_targets.py/eval_by_group.py use a group-disjoint split, "
            "while train_cached_full.py uses a source-balanced IID sample-level split; these are not the same protocol."
            if (
                group_split["train_rows"] != training_split["train_rows"]
                or group_split["val_rows"] != training_split["val_rows"]
                or len(training_split["group_overlap"]) != 0
            )
            else "Split implementations are consistent."
        ),
    )

    # Quality.
    if quality["invalid_numeric"] == 0 and quality["out_of_range"] == 0:
        add_status(statuses, "PASS", "quality_target", "All quality targets are numeric and inside [1, 5].")
    else:
        add_status(statuses, "HIGH-PRIORITY", "quality_target", "Invalid or out-of-range quality targets require investigation.")

    if len(fault_audit["cross_field_suspects"]) > 0:
        add_status(
            statuses,
            "WARN",
            "fault_field_anomalies",
            f"{len(fault_audit['cross_field_suspects'])} strong cross-field phrase candidates need manual review; script does not rewrite labels.",
        )
    else:
        add_status(statuses, "PASS", "fault_field_anomalies", "No strong cross-field phrase candidates detected by the audit rule.")

    if strict_pause_audit["matches_guide"]:
        add_status(
            statuses,
            "PASS",
            "unnatural_pause_strict_categories",
            "Strict initial categorization reproduces the team-specified 5907 / 4550 / 56 / 45 counts. This is a provisional audit policy, not the final parser contract.",
        )
    else:
        add_status(
            statuses,
            "WARN",
            "unnatural_pause_strict_categories",
            "Strict initial categorization does not reproduce the team-specified reference counts; reconcile the parser policy before using these categories downstream.",
        )

    # Audio.
    if audio_audit is None:
        add_status(statuses, "NOT-AUDITED", "audio_duplicates", "Audio archive/directory not supplied; waveform-level duplicates cannot be assessed.")
    elif audio_audit["missing_audio"]:
        add_status(statuses, "WARN", "csv_audio_alignment", f"{len(audio_audit['missing_audio'])} CSV IDs have no matching audio basename.")
    else:
        add_status(statuses, "PASS", "csv_audio_alignment", "Every CSV ID has a matching audio basename in the supplied audio inventory.")

    if audio_audit is not None and len(audio_audit["duplicate_basenames"]) > 0:
        add_status(statuses, "WARN", "audio_duplicate_basenames", "Duplicate WAV basenames were found in the supplied audio source.")

    return statuses


def build_report(
    df: pd.DataFrame,
    schema: dict,
    ids: dict,
    metadata_dups: dict,
    training_metadata_dups: dict,
    source_audit: dict,
    group_audit: dict,
    group_split: dict,
    training_split: dict,
    quality: dict,
    fault_audit: dict,
    strict_pause_audit: dict,
    audio_audit: dict | None,
    findings: list[dict],
    top_n: int,
    top_duplicates: int,
) -> str:
    lines: list[str] = []
    lines.append("# QualiSpeech Dataset Audit")
    lines.append("")
    lines.append(
        "> Generated by `scripts/dataset_audit.py`. This report audits metadata and repository split/group conventions. It does not by itself establish semantic source identity or audio-level duplicates."
    )
    lines.append("")

    # 1 overview
    lines.append("## 1. Dataset overview")
    lines.append("")
    lines.append(f"- Rows: **{schema['row_count']:,}**")
    lines.append(f"- Columns: **{schema['column_count']}**")
    lines.append(f"- Current group-disjoint split: **{group_split['train_rows']:,} train / {group_split['val_rows']:,} validation**")
    lines.append(f"- Actual training split mirrored from `train_cached_full.py`: **{training_split['train_rows']:,} train / {training_split['val_rows']:,} validation**")
    lines.append(f"- Inferred groups: **{group_audit['n_groups']}**")
    lines.append(f"- Quality target: `{QUALITY_COLUMN}`")
    lines.append("")
    lines.append("### Columns")
    lines.append("")
    lines.append(dataframe_to_markdown(schema["table"]))
    lines.append("")
    lines.append("**Required audit columns:** " + ("all present." if not schema["missing_required"] else ", ".join(schema["missing_required"])))
    lines.append("")

    # 2 ID/dup
    lines.append("## 2. ID and duplicate audit")
    lines.append("")
    lines.append(f"- Total ID rows: **{ids['total']:,}**")
    lines.append(f"- Unique IDs: **{ids['unique']:,}**")
    lines.append(f"- Duplicate ID rows: **{ids['duplicate_rows']:,}**")
    lines.append(f"- Blank ID rows: **{ids['blank_rows']:,}**")
    lines.append(f"- Malformed ID rows: **{ids['malformed_rows']:,}**")
    lines.append("")
    lines.append("### ID suffix patterns")
    lines.append("")
    lines.append(dataframe_to_markdown(ids["suffix_pattern_table"]))
    lines.append("")
    if ids["duplicates"].empty:
        lines.append("No duplicate ID values were found.")
    else:
        lines.append("### Duplicate ID values")
        lines.append("")
        lines.append(dataframe_to_markdown(ids["duplicates"], max_rows=top_duplicates))
        lines.append("")

    lines.append("### Exact metadata-duplicate candidates")
    lines.append("")
    lines.append(
        f"Rows identical across all original metadata columns except `{ID_COLUMN}`: **{metadata_dups['duplicate_rows']:,}** across **{metadata_dups['duplicate_groups']:,}** metadata keys."
    )
    lines.append(f"- Cross-group duplicate keys: **{metadata_dups['cross_group_groups']:,}** ({metadata_dups['cross_group_rows']:,} rows)")
    lines.append(f"- Cross-split duplicate keys under the group-disjoint split: **{metadata_dups['cross_split_groups']:,}** ({metadata_dups['cross_split_rows']:,} rows)")
    lines.append(f"- Cross-split duplicate keys under the actual training split: **{training_metadata_dups['cross_split_groups']:,}** ({training_metadata_dups['cross_split_rows']:,} rows)")
    lines.append("")
    lines.append("> Exact metadata repetition is not proof of duplicate audio or source leakage; it can arise from repeated annotation/template text across different recordings.")
    lines.append("> The two cross-split counts differ because the repository's two split protocols assign samples to train/validation differently.")
    lines.append("")
    if not metadata_dups["table"].empty:
        lines.append(dataframe_to_markdown(metadata_dups["table"], max_rows=top_duplicates))
        lines.append("")

    # 3 source/group
    lines.append("## 3. Source / group audit")
    lines.append("")
    lines.append("Current repository group rule:")
    lines.append("")
    lines.append("```python")
    lines.append("group = str(sample_id).split(\"-\", 1)[0]")
    lines.append("```")
    lines.append("")
    lines.append("> The prefix-before-hyphen value is a repository heuristic. It should not be treated as a final semantic `source_group_id` without dataset-semantic validation.")
    lines.append("")
    lines.append("### Repository source mapping")
    lines.append("")
    lines.append(dataframe_to_markdown(source_audit["counts"]))
    lines.append("")
    lines.append(dataframe_to_markdown(source_audit["source_group_counts"]))
    lines.append("")
    lines.append("> The observed source counts match the expectations hard-coded in `train_cached_full.py`. This supports consistency with the repository's current mapping, but is not an independent proof of dataset semantics.")
    lines.append("")

    lines.append("### Group-size summary")
    lines.append("")
    gs = group_audit["stats"]
    lines.append(dataframe_to_markdown(pd.DataFrame([{
        "count": gs.get("count", np.nan),
        "mean": gs.get("mean", np.nan),
        "std": gs.get("std", np.nan),
        "min": gs.get("min", np.nan),
        "25%": gs.get("25%", np.nan),
        "50%": gs.get("50%", np.nan),
        "75%": gs.get("75%", np.nan),
        "max": gs.get("max", np.nan),
    }])))
    lines.append("")
    lines.append(dataframe_to_markdown(group_audit["bin_table"]))
    lines.append("")
    lines.append("### Largest groups")
    lines.append("")
    lines.append(dataframe_to_markdown(group_audit["largest"]))
    lines.append("")
    lines.append("### Smallest groups")
    lines.append("")
    lines.append(dataframe_to_markdown(group_audit["smallest"]))
    lines.append("")

    # 4 split comparison
    lines.append("## 4. Repository split consistency")
    lines.append("")
    lines.append("> The repository currently contains two different split protocols. This section is intentionally explicit so the team can choose one canonical protocol before model selection/threshold calibration.")
    lines.append("")
    lines.append("### A. Group-disjoint split")
    lines.append("")
    lines.append("Used by `check_split.py`, `analyze_group_targets.py`, and `eval_by_group.py`.")
    lines.append("")
    lines.append(f"- Train: **{group_split['train_rows']:,}** samples / **{group_split['train_groups']}** groups")
    lines.append(f"- Validation: **{group_split['val_rows']:,}** samples / **{group_split['val_groups']}** groups")
    lines.append(f"- Group overlap: **{len(group_split['group_overlap'])}**")
    lines.append("")
    lines.append(dataframe_to_markdown(group_split["source_table"]))
    lines.append("")
    lines.append(dataframe_to_markdown(group_split["target_distribution"]))
    lines.append("")

    lines.append("### B. Source-balanced IID training split")
    lines.append("")
    lines.append("Used by `train_cached_full.py` / `make_source_balanced_split()`.")
    lines.append("")
    lines.append(f"- Train: **{training_split['train_rows']:,}** samples / **{training_split['train_groups']}** groups")
    lines.append(f"- Validation: **{training_split['val_rows']:,}** samples / **{training_split['val_groups']}** groups")
    lines.append(f"- Group overlap: **{len(training_split['group_overlap'])}**")
    lines.append("")
    lines.append(dataframe_to_markdown(training_split["source_table"]))
    lines.append("")
    lines.append(dataframe_to_markdown(training_split["target_distribution"]))
    lines.append("")

    lines.append("### Consistency finding")
    lines.append("")
    lines.append(
        f"The group-disjoint audit reproduces **{group_split['train_rows']:,}/{group_split['val_rows']:,}** train/validation samples, while the training script produces **{training_split['train_rows']:,}/{training_split['val_rows']:,}**. "
        f"Under the training script, **{len(training_split['group_overlap'])} of {group_audit['n_groups']} groups** appear in both train and validation."
    )
    lines.append("")
    lines.append("> **Action required from the team:** decide and document the canonical split protocol before comparing model results. Do not silently mix metrics from the two split definitions.")
    lines.append("")

    # Group target extremes in actual training split.
    group_target_stats = audit_group_target_stats(
        pd.concat(
            [
                group_disjoint_split(df)[0],
                group_disjoint_split(df)[1],
            ],
            ignore_index=True,
        )
    )
    lines.append("### Group target statistics under group-disjoint split")
    lines.append("")
    lines.append("Highest group means:")
    lines.append("")
    lines.append(dataframe_to_markdown(group_target_stats.sort_values("mean", ascending=False).head(15)))
    lines.append("")
    lines.append("Lowest group means:")
    lines.append("")
    lines.append(dataframe_to_markdown(group_target_stats.sort_values("mean", ascending=True).head(15)))
    lines.append("")

    # 5 quality
    lines.append("## 5. Quality-target audit")
    lines.append("")
    lines.append(f"- Numeric parsing failures: **{quality['invalid_numeric']}**")
    lines.append(f"- Out-of-range scores outside [1, 5]: **{quality['out_of_range']}**")
    lines.append(f"- Min: **{fmt_number(quality['min'])}**")
    lines.append(f"- Max: **{fmt_number(quality['max'])}**")
    lines.append(f"- Mean: **{fmt_number(quality['mean'])}**")
    lines.append(f"- Std: **{fmt_number(quality['std'])}**")
    lines.append("")
    lines.append(dataframe_to_markdown(quality["distribution"]))
    lines.append("")

    # 6 faults
    lines.append("## 6. Fault annotation audit")
    lines.append("")
    lines.append("> Categories are deliberately conservative. `time_range_candidate` means a time-like range was detected in text; it is not a confirmed localized interval. Final `fault_status` resolution belongs to the parser/resolver workflow.")
    lines.append("")
    lines.append(dataframe_to_markdown(fault_audit["summary"]))
    lines.append("")

    if not fault_audit["cross_field_suspects"].empty:
        lines.append("### Cross-field phrase candidates for manual review")
        lines.append("")
        lines.append(dataframe_to_markdown(fault_audit["cross_field_suspects"], max_rows=50))
        lines.append("")

    for fault_type in ["noise", "distortion", "unnatural_pause"]:
        lines.append(f"### Examples: {fault_type}")
        lines.append("")
        lines.append(dataframe_to_markdown(fault_audit["examples"][fault_type]))
        lines.append("")

    # 7 strict Unnatural pause pipeline handoff
    lines.append("## 7. Unnatural pause — strict initial pipeline categorization")
    lines.append("")
    lines.append(
        "> This section operationalizes the current team guidance for the initial strict audit. It is deliberately separate from Đạt's final deterministic parser and from Hiếu's human QA."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(strict_pause_audit["summary"]))
    lines.append("")
    lines.append(
        "> The current strict policy treats conjunction-linked multi-range strings containing ` and ` as residual (`abnormal`) for audit purposes. The pipeline guide also gives multi-range-with-`and` as a parseable example, so this exact policy must be reconciled with Đạt before parser labels are frozen."
    )
    lines.append("")
    lines.append("### Pipeline boundary")
    lines.append("")
    lines.extend([
        "- **Kiên:** audits raw values/frequencies and maintains audit statistics.",
        "- **Đạt:** owns deterministic text-to-interval parsing and normalization.",
        "- **Hiếu:** owns human QA for ambiguous/residual cases and DSP candidates.",
        "- **Kiên:** after QA, records the approved final label/version; this is downstream work and not part of raw audit.",
    ])
    lines.append("")
    lines.append("### Audit interpretation")
    lines.append("")
    lines.extend([
        "- `no_fault` means the raw text is in the strict no-fault bucket only; it is not a blanket semantic correction rule.",
        "- `regex_parseable` means a deterministic range candidate is syntactically detectable; it still needs duration-aware validation before final localization.",
        "- `single_ambiguous` is intentionally not converted to an interval.",
        "- `abnormal` is residual data for parser review/QA; it is not automatically a positive or negative label.",
    ])
    lines.append("")

    lines.append("### Full raw-value frequency tables")
    lines.append("")
    lines.append("CSV exports are generated under `reports/annotation_frequency/` for all three fault-description columns.")
    lines.append("")
    for fault_type, column in FAULT_COLUMNS.items():
        lines.append(f"#### Top raw values: {fault_type}")
        lines.append("")
        freq = (
            df[column]
            .fillna("")
            .astype(str)
            .map(lambda x: x.strip())
            .value_counts(dropna=False)
            .rename_axis("raw_value")
            .reset_index(name="count")
        )
        freq["pct"] = freq["count"].map(lambda x: safe_pct(x, len(df)))
        lines.append(dataframe_to_markdown(freq.head(30)))
        lines.append("")

    # 8 audio
    lines.append("## 8. Audio inventory audit")
    lines.append("")
    if audio_audit is None:
        lines.append("Not run: no `--wav-dir` or `--wav-zip` was supplied.")
        lines.append("")
        lines.append("> Audio-level duplicate detection remains unresolved until the corresponding audio archive/directory is available.")
    else:
        lines.append(f"- Audio basenames: **{audio_audit['audio_count']:,}**")
        lines.append(f"- CSV IDs: **{audio_audit['csv_count']:,}**")
        lines.append(f"- CSV IDs without matching audio: **{len(audio_audit['missing_audio']):,}**")
        lines.append(f"- Orphan audio files: **{len(audio_audit['orphan_audio']):,}**")
        lines.append(f"- Duplicate basenames: **{len(audio_audit['duplicate_basenames']):,}**")
        lines.append(f"- Invalid/unreadable WAV files: **{len(audio_audit['invalid_files']):,}**")
        lines.append("")
        if audio_audit["duration_stats"]:
            lines.append("### Duration summary")
            lines.append("")
            lines.append(dataframe_to_markdown(pd.DataFrame([audio_audit["duration_stats"]])))
            lines.append("")
        if not audio_audit["duplicate_hash_table"].empty:
            lines.append("### SHA-256 duplicate audio candidates")
            lines.append("")
            lines.append(dataframe_to_markdown(audio_audit["duplicate_hash_table"], max_rows=50))
            lines.append("")
        lines.append("> Audio SHA-256 duplicates indicate identical WAV file bytes. They are strong duplicate candidates but still do not by themselves prove the same semantic source utterance if files were independently produced with identical bytes.")
    lines.append("")

    # 9 findings
    lines.append("## 9. Findings and action items")
    lines.append("")
    for level in ["HIGH-PRIORITY", "WARN", "PASS", "NOT-AUDITED"]:
        selected = [x for x in findings if x["level"] == level]
        if not selected:
            continue
        lines.append(f"### {level}")
        lines.append("")
        for item in selected:
            lines.append(f"- **{item['item']}** — {item['detail']}")
        lines.append("")

    lines.append("## 10. Recommended next actions")
    lines.append("")
    lines.extend([
        "1. Resolve the repository split inconsistency: choose one canonical validation protocol and update/document all dependent scripts.",
        "2. Manually inspect the largest inferred groups (`sysnists`, `sys0giga`, `sysnistl`, and the largest BVCC groups) and confirm what the prefix represents.",
        "3. Review the cross-group/cross-split metadata repetition candidates; distinguish repeated templates from genuine repeated source utterances.",
        "4. Keep the current group rule as a documented repository heuristic until the team confirms the semantic `source_group_id` definition.",
        "5. When the audio archive is available, rerun the audio inventory; use file/audio hashes for duplicate candidates.",
        "6. Reconcile the strict Unnatural pause categories with Đạt's deterministic parser contract, especially multi-range expressions joined by `and`.",
        "7. Define the handoff schema from parser -> Hiếu QA -> Kiên final-label management, including status, intervals, confidence and review metadata.",
        "8. Keep audit categories separate from final fault labels; do not silently convert `time_range_candidate`, `single_ambiguous`, or `abnormal` into training labels.",
    ])
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    args = parse_args()

    csv_path = args.csv.resolve()
    report_path = args.report.resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    if args.wav_dir is not None and args.wav_zip is not None:
        print("ERROR: provide only one of --wav-dir or --wav-zip", file=sys.stderr)
        return 1

    print("=" * 78)
    print("QualiSpeech Dataset Audit")
    print("=" * 78)
    print(f"CSV: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    schema = audit_schema(df_raw)

    if schema["missing_required"]:
        print(
            "ERROR: missing required audit columns: "
            + ", ".join(schema["missing_required"]),
            file=sys.stderr,
        )
        return 2

    df = df_raw.copy()
    df[ID_COLUMN] = df[ID_COLUMN].map(clean_id)
    df["sample_id"] = df[ID_COLUMN].map(strip_wav_suffix)
    df["group"] = df["sample_id"].map(get_group)
    df["source"] = df["group"].map(get_source)
    df["target"] = pd.to_numeric(df[QUALITY_COLUMN], errors="coerce")

    ids = audit_ids(df)

    # Audit both split protocols.
    gd_train, gd_val = group_disjoint_split(df)
    tr_train, tr_val = source_balanced_iid_split(df)
    group_split = summarize_split(gd_train, gd_val)
    training_split = summarize_split(tr_train, tr_val)

    # Keep the group-disjoint split as the primary audit split while preserving
    # the separate training-split audit above. This `split` column is derived
    # only for reporting; original metadata columns are never overwritten.
    group_val_ids = set(gd_val[ID_COLUMN])
    df["split"] = np.where(df[ID_COLUMN].isin(group_val_ids), "val", "train")

    metadata_dups = audit_metadata_duplicates(df)
    training_metadata_dups = audit_metadata_duplicates(
        pd.concat([tr_train, tr_val], ignore_index=True)
    )

    source_audit = audit_source_mapping(df)
    group_audit = audit_group_structure(df, args.top_groups)
    quality = audit_quality(df)
    fault_audit = audit_fault_annotations(df)
    strict_pause_audit = audit_strict_unnatural_pause(df)
    frequency_paths = write_annotation_frequency_tables(df, args.frequency_dir.resolve())
    audio_audit = audit_audio_inventory(
        df,
        args.wav_dir.resolve() if args.wav_dir else None,
        args.wav_zip.resolve() if args.wav_zip else None,
        args.audio_hash,
    )

    print("\nSTRICT Unnatural pause categories:")
    print(strict_pause_audit["summary"].to_string(index=False))
    print(f"\nRaw frequency tables written to: {args.frequency_dir.resolve()}")
    for fault_type, path in frequency_paths.items():
        print(f"  - {fault_type}: {path.resolve()}")

    findings = build_findings(
        schema=schema,
        ids=ids,
        metadata_dups=metadata_dups,
        source_audit=source_audit,
        group_audit=group_audit,
        group_split=group_split,
        training_split=training_split,
        quality=quality,
        fault_audit=fault_audit,
        strict_pause_audit=strict_pause_audit,
        audio_audit=audio_audit,
    )

    report = build_report(
        df=df,
        schema=schema,
        ids=ids,
        metadata_dups=metadata_dups,
        training_metadata_dups=training_metadata_dups,
        source_audit=source_audit,
        group_audit=group_audit,
        group_split=group_split,
        training_split=training_split,
        quality=quality,
        fault_audit=fault_audit,
        strict_pause_audit=strict_pause_audit,
        audio_audit=audio_audit,
        findings=findings,
        top_n=args.top_groups,
        top_duplicates=args.top_duplicates,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print()
    print(f"Rows: {len(df):,}")
    print(f"Unique IDs: {ids['unique']:,}")
    print(f"Duplicate ID rows: {ids['duplicate_rows']:,}")
    print(f"Groups: {group_audit['n_groups']}")
    print(
        f"Group-disjoint split: {group_split['train_rows']:,} train / "
        f"{group_split['val_rows']:,} val; overlap={len(group_split['group_overlap'])}"
    )
    print(
        f"Training split: {training_split['train_rows']:,} train / "
        f"{training_split['val_rows']:,} val; group_overlap={len(training_split['group_overlap'])}"
    )
    print(f"Metadata duplicate rows: {metadata_dups['duplicate_rows']:,}")
    print(f"Cross-split metadata duplicate keys (group-disjoint): {metadata_dups['cross_split_groups']:,}")
    print(f"Cross-split metadata duplicate keys (training split): {training_metadata_dups['cross_split_groups']:,}")
    print(f"Quality invalid/out-of-range: {quality['invalid_numeric']}/{quality['out_of_range']}")
    print(f"Cross-field fault suspects: {len(fault_audit['cross_field_suspects'])}")
    print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
