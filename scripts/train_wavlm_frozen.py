"""
Train QualityHead on pre-extracted Frozen WavLM-Large cached representations.
Supports explicit train, val, and test splits (QualiSpeech train.csv, val.csv, test.csv).
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.sqa.model import QualityHead, UtmosNisqaQualityHead, quality_loss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedQualiSpeechDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_dir: Path):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir

        self.df["sample_id"] = (
            self.df["id"].astype(str).str.strip().str.replace(r"\.wav$", "", regex=True)
        )
        self.df["target"] = pd.to_numeric(self.df["Overall quality"], errors="raise").astype(float)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        target = float(row["target"])

        pt_path = self.cache_dir / f"{sample_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing cache item: {pt_path}")

        item = torch.load(pt_path, map_location="cpu")
        hidden = item["hidden"].float()  # [T, 1024]
        valid_mask = item["valid_mask"].bool()  # [T]

        return {
            "hidden": hidden,
            "valid_mask": valid_mask,
            "target": torch.tensor(target, dtype=torch.float32),
            "file_id": sample_id,
        }


def cached_collate_fn(batch: list[dict]) -> dict:
    hidden_list = [x["hidden"] for x in batch]
    mask_list = [x["valid_mask"] for x in batch]
    targets = torch.stack([x["target"] for x in batch])
    file_ids = [x["file_id"] for x in batch]

    padded_hidden = pad_sequence(hidden_list, batch_first=True, padding_value=0.0)
    padded_mask = pad_sequence(mask_list, batch_first=True, padding_value=False)

    return {
        "hidden": padded_hidden,
        "valid_mask": padded_mask,
        "targets": targets,
        "file_ids": file_ids,
    }


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    pcc = float(np.corrcoef(preds, targets)[0, 1])

    vx, vy = np.var(preds), np.var(targets)
    mx, my = np.mean(preds), np.mean(targets)
    cov = np.mean((preds - mx) * (targets - my))
    ccc = float(2.0 * cov / (vx + vy + (mx - my) ** 2 + 1e-8))

    return {"mse": mse, "mae": mae, "ccc": ccc, "pcc": pcc}


@torch.no_grad()
def evaluate(model: QualityHead, loader: DataLoader, device: torch.device) -> dict[str, any]:
    model.eval()
    all_preds = []
    all_targets = []
    all_fids = []

    for batch in loader:
        hidden = batch["hidden"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        targets = batch["targets"].cpu().numpy()

        preds = model(hidden, valid_mask).squeeze(-1).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(targets)
        all_fids.extend(batch["file_ids"])

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)

    metrics = compute_metrics(preds_arr, targets_arr)
    metrics["predictions"] = preds_arr
    metrics["targets"] = targets_arr
    metrics["file_ids"] = all_fids
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train QualityHead on Frozen WavLM cached representations.")
    parser.add_argument("--train-csv", type=str, default="data/qualispeech/train.csv", help="Train CSV path")
    parser.add_argument("--val-csv", type=str, default="data/qualispeech/val.csv", help="Val CSV path")
    parser.add_argument("--test-csv", type=str, default="data/qualispeech/test.csv", help="Test CSV path")
    parser.add_argument("--cache-dir", type=str, default="data/qualispeech/wavlm_cache", help="Cache directory")
    parser.add_argument("--out-dir", type=str, default="outputs/cached_full", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="QualityHead dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--w-mse", type=float, default=0.4, help="MSE loss weight")
    parser.add_argument("--w-ccc", type=float, default=0.3, help="CCC loss weight")
    parser.add_argument("--w-pcc", type=float, default=0.3, help="PCC loss weight")
    parser.add_argument("--quality-head-type", type=str, default="standard", choices=["standard", "utmos_nisqa"], help="Quality head architecture")
    parser.add_argument("--bilstm-hidden", type=int, default=128, help="BiLSTM hidden units for utmos_nisqa")
    parser.add_argument("--w-rank", type=float, default=0.0, help="Pairwise ranking loss weight")
    parser.add_argument("--rank-margin", type=float, default=0.1, help="Ranking loss margin")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv_path = ROOT / args.train_csv
    val_csv_path = ROOT / args.val_csv
    test_csv_path = ROOT / args.test_csv if args.test_csv else None
    cache_dir = ROOT / args.cache_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WAVLM-LARGE (FROZEN) — FULL DATA TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device:      {device}")
    if device.type == "cuda":
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"Cache dir:   {cache_dir}")
    print(f"Output dir:  {out_dir}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Epochs:      {args.epochs}")
    print(f"LR:          {args.lr}")
    print(f"Loss:        {args.w_mse}*MSE + {args.w_ccc}*(1-CCC) + {args.w_pcc}*(1-PCC)")
    print()

    # Load CSV splits
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path) if test_csv_path and test_csv_path.exists() else None

    train_dataset = CachedQualiSpeechDataset(train_df, cache_dir)
    val_dataset = CachedQualiSpeechDataset(val_df, cache_dir)
    test_dataset = CachedQualiSpeechDataset(test_df, cache_dir) if test_df is not None else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=cached_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=cached_collate_fn,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=cached_collate_fn,
        )
        if test_dataset is not None
        else None
    )

    print("SPLIT SUMMARY:")
    print(f"  Train samples: {len(train_dataset)} | Batches: {len(train_loader)}")
    print(f"  Val samples:   {len(val_dataset)}   | Batches: {len(val_loader)}")
    if test_dataset:
        print(f"  Test samples:  {len(test_dataset)}  | Batches: {len(test_loader)}")
    print()

    # Initialize QualityHead
    hidden_size = 1024
    if args.quality_head_type == "utmos_nisqa":
        model = UtmosNisqaQualityHead(
            hidden_size=hidden_size,
            bilstm_hidden=args.bilstm_hidden,
            dropout=args.dropout,
            use_fault_features=False,
        ).to(device)
    else:
        model = QualityHead(hidden_size=hidden_size, dropout=args.dropout).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ({args.quality_head_type}) trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Initial evaluation
    print()
    print("INITIAL VALIDATION (EPOCH 0)")
    init_metrics = evaluate(model, val_loader, device)
    print(
        f"Init | MSE={init_metrics['mse']:.4f} | MAE={init_metrics['mae']:.4f} | "
        f"CCC={init_metrics['ccc']:.4f} | PCC={init_metrics['pcc']:.4f}"
    )

    best_val_pcc = -float("inf")
    best_epoch = -1
    best_ckpt_path = out_dir / "best_quality_head.pt"
    history = []

    print()
    print("=" * 70)
    print("TRAINING LOOP")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            hidden = batch["hidden"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(hidden, valid_mask).squeeze(-1)

            loss = quality_loss(
                preds,
                targets,
                w_mse=args.w_mse,
                w_ccc=args.w_ccc,
                w_pcc=args.w_pcc,
                w_rank=args.w_rank,
                rank_margin=args.rank_margin,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_train_loss = running_loss / max(1, n_batches)

        # Evaluate on validation
        val_metrics = evaluate(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | train_loss={epoch_train_loss:.4f} | "
            f"VAL MSE={val_metrics['mse']:.4f} | MAE={val_metrics['mae']:.4f} | "
            f"CCC={val_metrics['ccc']:.4f} | PCC={val_metrics['pcc']:.4f} | lr={current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_ccc": val_metrics["ccc"],
            "val_pcc": val_metrics["pcc"],
            "lr": current_lr,
        })

        if val_metrics["pcc"] > best_val_pcc:
            best_val_pcc = val_metrics["pcc"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "hidden_size": hidden_size,
                    "quality_head_type": args.quality_head_type,
                    "quality_head_kwargs": {
                        "bilstm_hidden": args.bilstm_hidden,
                        "dropout": args.dropout,
                        "use_fault_features": False,
                    } if args.quality_head_type == "utmos_nisqa" else {},
                    "val_mse": val_metrics["mse"],
                    "val_mae": val_metrics["mae"],
                    "val_ccc": val_metrics["ccc"],
                    "val_pcc": val_metrics["pcc"],
                    "best_epoch": best_epoch,
                },
                best_ckpt_path,
            )
            print(f"  -> NEW BEST CHECKPOINT (epoch={epoch}, VAL PCC={best_val_pcc:.4f})")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    print()
    print("=" * 70)
    print("RESTORE BEST CHECKPOINT")
    print("=" * 70)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Restored best checkpoint from Epoch {ckpt['epoch']} (VAL PCC: {ckpt['val_pcc']:.4f})")

    # Final Validation evaluation
    final_val = evaluate(model, val_loader, device)
    print()
    print("=" * 70)
    print("FINAL RESULT — BEST VALIDATION CHECKPOINT")
    print("=" * 70)
    print(f"VAL MSE: {final_val['mse']:.6f}")
    print(f"VAL MAE: {final_val['mae']:.6f}")
    print(f"VAL CCC: {final_val['ccc']:.6f}")
    print(f"VAL PCC: {final_val['pcc']:.6f}")

    val_pred_df = pd.DataFrame({
        "id": [f"{fid}.wav" for fid in final_val["file_ids"]],
        "ground_truth": final_val["targets"],
        "prediction": final_val["predictions"],
    })
    val_pred_df.to_csv(out_dir / "val_predictions.csv", index=False)
    print(f"Saved val predictions to: {out_dir / 'val_predictions.csv'}")

    # Final Blind Test evaluation
    if test_loader is not None:
        final_test = evaluate(model, test_loader, device)
        print()
        print("=" * 70)
        print("FINAL BLIND TEST EVALUATION (QualiSpeech test.csv)")
        print("=" * 70)
        print(f"TEST MSE: {final_test['mse']:.6f}")
        print(f"TEST MAE: {final_test['mae']:.6f}")
        print(f"TEST CCC: {final_test['ccc']:.6f}")
        print(f"TEST PCC: {final_test['pcc']:.6f}")

        # Save test predictions
        test_pred_df = pd.DataFrame({
            "id": [f"{fid}.wav" for fid in final_test["file_ids"]],
            "ground_truth": final_test["targets"],
            "prediction": final_test["predictions"],
        })
        test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
        print(f"Saved test predictions to: {out_dir / 'test_predictions.csv'}")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
