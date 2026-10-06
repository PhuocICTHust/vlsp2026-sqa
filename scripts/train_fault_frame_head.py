from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqa.evaluator import FAULT_TYPES, compute_f1_macro, compute_frame_iou_macro
from sqa.frame_grid import EVALUATOR_GRID, MODEL_GRID, intervals_to_model_frame_targets
from sqa.frame_head import FaultFrameHead
from sqa.postprocess import BASELINE_POSTPROCESS, predict_fault_type, serialize_intervals

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"
CACHE_DIR = ROOT / "data" / "qualispeech" / "wavlm_cache"
REVIEWED_JSON_PATH = ROOT / "label_studio" / "reviewed_labels_for_kien.json"
LABEL_V2_PATH = ROOT / "data" / "qualispeech" / "label_v2" / "parsed_labels.jsonl"
OUT_DIR = ROOT / "outputs" / "fault_head"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = OUT_DIR / "best_fault_frame_head.pt"

SEED = 42
VAL_RATIO = 0.20
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NO_FAULT_PATTERNS = {
    "noise": {"not noticeable", "not noticeable.", "none", "no background noise", "not noticeable "},
    "distortion": {"not distorted", "not distorted.", "none", "no distortion", "not distorted "},
    "unnatural_pause": {"very smooth", "very smooth.", "none", "no unnatural pause", "no pause", "very smooth "},
}

TIME_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*(?:-|to|~|–|—)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?",
    re.IGNORECASE,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_group(sample_id: str) -> str:
    sample_id = str(sample_id).strip()
    return sample_id.split("-", 1)[0]


def get_source(group: str) -> str:
    group = str(group).strip().lower()
    if group == "sysnists":
        return "nisqa_sim"
    if group == "sysnistl":
        return "nisqa_live"
    if group == "sys0giga":
        return "gigaspeech"
    if group.startswith("sys") and group.endswith("tts"):
        middle = group[3:-3]
        if middle.isdigit():
            return "recent_tts"
    return "bvcc"


def parse_text_intervals(text: str, fault_type: str) -> list[list[float]]:
    if not isinstance(text, str):
        return []
    cleaned = text.strip().lower()
    if cleaned in NO_FAULT_PATTERNS.get(fault_type, set()):
        return []

    matches = TIME_RANGE_RE.findall(cleaned)
    intervals = []
    for m in matches:
        s, e = float(m[0]), float(m[1])
        if s < e:
            intervals.append([round(s, 2), round(e, 2)])
    return intervals


def build_annotation_lookup(
    df: pd.DataFrame,
    reviewed_path: Path | None = None,
    label_v2_path: Path | None = None,
) -> dict[str, dict]:
    lookup = {}

    # If Labels v2 (parsed_labels.jsonl) is available, use it as the primary gold-standard!
    if label_v2_path is not None and label_v2_path.exists():
        print(f"Loading official Labels v2 from: {label_v2_path}")
        with open(label_v2_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                fid = str(rec["file_id"]).strip()
                ft = str(rec["fault"]).strip()
                st = str(rec["status"]).strip()
                if fid not in lookup:
                    lookup[fid] = {k: [] for k in FAULT_TYPES}
                # USABLE_STATUSES: no_fault, parsed, normalized, audited, rule_confirmed
                if st in ("no_fault", "parsed", "normalized", "audited", "rule_confirmed"):
                    ivs = []
                    for s, e in rec.get("intervals", []):
                        if s is not None and e is not None and float(s) < float(e):
                            ivs.append([round(float(s), 2), round(float(e), 2)])
                    lookup[fid][ft] = ivs
        print(f"Successfully loaded {len(lookup)} files with Labels v2.")
        return lookup

    col_map = {
        "noise": "Noise Description",
        "distortion": "Distortion description",
        "unnatural_pause": "Unnatural pause",
    }

    for _, row in df.iterrows():
        fid = str(row["id"]).strip()
        fault_dict = {}
        for ft in FAULT_TYPES:
            col_name = col_map[ft]
            val = row[col_name] if col_name in row else ""
            fault_dict[ft] = parse_text_intervals(val, ft)
        lookup[fid] = fault_dict

    # Override with human-reviewed Label Studio annotations if available
    if reviewed_path is not None and reviewed_path.exists():
        try:
            reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
            for item in reviewed:
                fid = item.get("file_id")
                if fid and fid in lookup and "faults" in item:
                    for ft in FAULT_TYPES:
                        if ft in item["faults"]:
                            lookup[fid][ft] = item["faults"][ft]
            print(f"Loaded {len(reviewed)} human-verified labels from {reviewed_path.name}")
        except Exception as e:
            print(f"Warning: could not load reviewed labels: {e}")

    return lookup


def make_source_balanced_split(df: pd.DataFrame):
    rng = np.random.RandomState(SEED)
    train_parts = []
    val_parts = []

    source_order = ["bvcc", "nisqa_sim", "gigaspeech", "nisqa_live", "recent_tts"]
    for source in source_order:
        source_df = df[df["source"] == source].copy()
        source_train_parts = []
        source_val_parts = []

        for target_value in sorted(source_df["target"].unique()):
            class_df = source_df[source_df["target"] == target_value].copy()
            indices = np.arange(len(class_df))
            rng.shuffle(indices)

            n_val = int(round(len(class_df) * VAL_RATIO))
            if len(class_df) >= 2:
                n_val = max(1, min(n_val, len(class_df) - 1))
            else:
                n_val = 0

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            source_val_parts.append(class_df.iloc[val_indices])
            source_train_parts.append(class_df.iloc[train_indices])

        train_parts.append(pd.concat(source_train_parts, ignore_index=True))
        val_parts.append(pd.concat(source_val_parts, ignore_index=True))

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return train_df, val_df


class CachedFaultDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_dir: Path, annotation_lookup: dict[str, dict]):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.annotation_lookup = annotation_lookup

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_id = str(row["id"]).strip()
        stem = Path(file_id).stem
        pt_path = self.cache_dir / f"{stem}.pt"

        if not pt_path.exists():
            raise FileNotFoundError(f"Cached embedding missing: {pt_path}")

        cached = torch.load(pt_path, map_location="cpu")
        hidden = cached["hidden"]  # [T, 1024]
        valid_mask = cached["valid_mask"]  # [T]
        n_frames = len(hidden)

        fault_dict = self.annotation_lookup.get(file_id, {ft: [] for ft in FAULT_TYPES})
        target_frames = np.zeros((n_frames, len(FAULT_TYPES)), dtype=np.float32)

        for i, ft in enumerate(FAULT_TYPES):
            intervals = fault_dict.get(ft, [])
            target_frames[:, i] = intervals_to_model_frame_targets(intervals, n_frames, MODEL_GRID)

        return {
            "hidden": hidden,
            "valid_mask": valid_mask,
            "target_frames": torch.from_numpy(target_frames),
            "file_id": file_id,
            "gt_faults": fault_dict,
        }


def fault_collate_fn(batch):
    max_len = max(item["hidden"].shape[0] for item in batch)
    hidden_dim = batch[0]["hidden"].shape[1]
    n_faults = batch[0]["target_frames"].shape[1]
    b = len(batch)

    padded_hidden = torch.zeros(b, max_len, hidden_dim, dtype=torch.float32)
    padded_valid_mask = torch.zeros(b, max_len, dtype=torch.bool)
    padded_targets = torch.zeros(b, max_len, n_faults, dtype=torch.float32)

    file_ids = []
    gt_faults = []

    for i, item in enumerate(batch):
        t = item["hidden"].shape[0]
        padded_hidden[i, :t] = item["hidden"]
        padded_valid_mask[i, :t] = item["valid_mask"][:t]
        padded_targets[i, :t] = item["target_frames"][:t]
        file_ids.append(item["file_id"])
        gt_faults.append(item["gt_faults"])

    return {
        "hidden": padded_hidden,
        "valid_mask": padded_valid_mask,
        "targets": padded_targets,
        "file_ids": file_ids,
        "gt_faults": gt_faults,
    }


def compute_focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    mask = valid_mask.unsqueeze(-1).expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = alpha_t * (1.0 - p_t).pow(gamma) * bce
    return (focal * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate_fault_head(
    head: FaultFrameHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    all_preds_entries = []
    all_gts_entries = []
    all_durations = []

    for batch in loader:
        hidden = batch["hidden"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        probs = head.predict_proba(hidden, valid_mask).cpu().numpy()  # [B, T, 3]

        for i, fid in enumerate(batch["file_ids"]):
            mask_i = batch["valid_mask"][i].cpu().numpy()
            valid_len = int(mask_i.sum())
            duration = valid_len * MODEL_GRID.frame_hop_sec + MODEL_GRID.offset_sec
            all_durations.append(duration)

            pred_faults = {}
            for ft_idx, ft in enumerate(FAULT_TYPES):
                intervals, _, _, _ = predict_fault_type(
                    probs[i, :valid_len, ft_idx],
                    MODEL_GRID,
                    duration,
                    config=BASELINE_POSTPROCESS,
                )
                pred_faults[ft] = serialize_intervals(intervals, round_digits=2, duration=duration)

            all_preds_entries.append({"file_id": fid, "faults": pred_faults})
            all_gts_entries.append({"file_id": fid, "faults": batch["gt_faults"][i]})

    f1_macro, f1_dict = compute_f1_macro(all_preds_entries, all_gts_entries)
    iou_macro, iou_dict = compute_frame_iou_macro(all_preds_entries, all_gts_entries, all_durations, EVALUATOR_GRID)
    composite_fault_score = 0.5 * f1_macro + 0.5 * iou_macro

    return {
        "f1_macro": f1_macro,
        "iou_macro": iou_macro,
        "composite_fault_score": composite_fault_score,
        "f1_noise": f1_dict.get("noise", 0.0),
        "f1_distortion": f1_dict.get("distortion", 0.0),
        "f1_unnatural_pause": f1_dict.get("unnatural_pause", 0.0),
        "iou_noise": iou_dict.get("noise", 0.0),
        "iou_distortion": iou_dict.get("distortion", 0.0),
        "iou_unnatural_pause": iou_dict.get("unnatural_pause", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train FaultFrameHead on cached WavLM features.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=5)
    args = parser.parse_args()

    set_seed(SEED)

    print("=" * 70)
    print("TRAINING FAULT FRAME HEAD ON WAVLM CACHED FEATURES")
    print("=" * 70)
    print(f"Device:      {DEVICE}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Epochs:      {args.epochs}")
    print(f"LR:          {args.lr}")
    print(f"Cache dir:   {CACHE_DIR}")

    df = pd.read_csv(CSV_PATH)
    df["sample_id"] = df["id"].astype(str).str.strip().str.replace(r"\.wav$", "", regex=True)
    df["target"] = pd.to_numeric(df["Overall quality"], errors="raise").astype(float)
    df["group"] = df["sample_id"].map(get_group)
    df["source"] = df["group"].map(get_source)

    annotation_lookup = build_annotation_lookup(df, REVIEWED_JSON_PATH, label_v2_path=LABEL_V2_PATH)
    train_df, val_df = make_source_balanced_split(df)

    train_dataset = CachedFaultDataset(train_df, CACHE_DIR, annotation_lookup)
    val_dataset = CachedFaultDataset(val_df, CACHE_DIR, annotation_lookup)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=fault_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=fault_collate_fn,
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    fault_head = FaultFrameHead(
        input_dim=1024,
        n_faults=len(FAULT_TYPES),
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        dropout=0.1,
    ).to(DEVICE)

    trainable_params = sum(p.numel() for p in fault_head.parameters() if p.requires_grad)
    print(f"FaultFrameHead trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(fault_head.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_score = -float("inf")
    best_epoch = -1
    history = []

    print("\nINITIAL EVALUATION (EPOCH 0)")
    init_metrics = evaluate_fault_head(fault_head, val_loader, DEVICE)
    print(
        f"Init | F1_macro={init_metrics['f1_macro']:.4f} | "
        f"IoU_macro={init_metrics['iou_macro']:.4f} | "
        f"Score={init_metrics['composite_fault_score']:.4f}"
    )

    print("\n" + "=" * 70)
    print("STARTING TRAINING LOOP")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        fault_head.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            hidden = batch["hidden"].to(DEVICE)
            valid_mask = batch["valid_mask"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = fault_head(hidden, valid_mask)
            loss = compute_focal_bce_loss(logits, targets, valid_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fault_head.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_loss = running_loss / max(num_batches, 1)

        val_metrics = evaluate_fault_head(fault_head, val_loader, DEVICE)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"VAL F1_macro={val_metrics['f1_macro']:.4f} | "
            f"IoU_macro={val_metrics['iou_macro']:.4f} | "
            f"Fault_Score={val_metrics['composite_fault_score']:.4f}"
        )

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)

        if val_metrics["composite_fault_score"] > best_score:
            best_score = val_metrics["composite_fault_score"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "fault_head_state_dict": fault_head.state_dict(),
                    "fault_head_kwargs": {
                        "input_dim": 1024,
                        "n_faults": len(FAULT_TYPES),
                        "hidden_dim": args.hidden_dim,
                        "n_blocks": args.n_blocks,
                        "kernel_size": args.kernel_size,
                    },
                    "metrics": val_metrics,
                },
                BEST_CHECKPOINT,
            )
            print(f"  -> NEW BEST CHECKPOINT (epoch={epoch}, Fault Score={best_score:.4f})")

        pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Best Epoch:       {best_epoch}")
    print(f"Best Fault Score: {best_score:.4f}")
    print(f"Saved Checkpoint: {BEST_CHECKPOINT}")


if __name__ == "__main__":
    main()
