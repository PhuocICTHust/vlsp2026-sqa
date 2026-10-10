"""
Extract and cache frozen HuBERT-Large (facebook/hubert-large-ll60k) representations
for all audio clips in QualiSpeech.
Output format matches wavlm_cache:
  sample_id.pt -> {'hidden': torch.HalfTensor [T, 1024], 'valid_mask': torch.BoolTensor [T], 'file_id': str}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transformers
try:
    import transformers.utils.import_utils as _iu
    _iu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from transformers import HubertModel
from src.sqa.dataset import QualiSpeechDataset, qualispeech_collate
from src.sqa.model import normalize_waveform


def make_frame_valid_mask(
    backbone: nn.Module,
    attention_mask: torch.Tensor,
    max_frames: int,
) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1)
    if hasattr(backbone, "_get_feat_extract_output_lengths"):
        frame_lengths = backbone._get_feat_extract_output_lengths(lengths)
    elif hasattr(backbone, "base_model") and hasattr(backbone.base_model, "_get_feat_extract_output_lengths"):
        frame_lengths = backbone.base_model._get_feat_extract_output_lengths(lengths)
    else:
        frame_lengths = (lengths // 320).clamp(min=1, max=max_frames)
    frame_lengths = frame_lengths.to(attention_mask.device)
    frame_idx = torch.arange(max_frames, device=attention_mask.device).unsqueeze(0)
    return frame_idx < frame_lengths.unsqueeze(1)


def main():
    parser = argparse.ArgumentParser(description="Cache HuBERT-Large representations")
    parser.add_argument("--model-name", type=str, default="facebook/hubert-large-ll60k")
    parser.add_argument("--csv-path", type=str, default="data/qualispeech/train_full.csv")
    parser.add_argument("--wav-dir", type=str, default="data/qualispeech/wav")
    parser.add_argument("--cache-dir", type=str, default="data/qualispeech/hubert_large_cache")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalize-wav", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("EXTRACT & CACHE HUBERT-LARGE REPRESENTATIONS")
    print("=" * 70)
    print("Device:        ", device)
    if device.type == "cuda":
        print("GPU:           ", torch.cuda.get_device_name(0))
    print("Model:         ", args.model_name)
    print("CSV:           ", args.csv_path)
    print("WAV Dir:       ", args.wav_dir)
    print("Cache Dir:     ", args.cache_dir)
    print("Batch size:    ", args.batch_size)
    print("Normalize wav: ", args.normalize_wav)

    cache_dir = ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ROOT / args.csv_path
    wav_dir = ROOT / args.wav_dir

    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df):,}")

    pending_df = df.copy()
    pending_df["sample_id"] = pending_df["id"].astype(str).str.strip().str.replace(r"\.wav$", "", regex=True)
    existing = set(p.stem for p in cache_dir.glob("*.pt"))
    print(f"Already cached: {len(existing):,} files")

    pending_df = pending_df[~pending_df["sample_id"].isin(existing)].reset_index(drop=True)
    print(f"Remaining to cache: {len(pending_df):,} files")

    if len(pending_df) == 0:
        print("All samples are already cached! Done.")
        return

    print("\nLoading HuBERT backbone...")
    backbone = HubertModel.from_pretrained(args.model_name).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    pending_csv = cache_dir / "_pending_cache.csv"
    pending_df.to_csv(pending_csv, index=False)
    dataset = QualiSpeechDataset(csv_path=pending_csv, wav_dir=wav_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=qualispeech_collate,
    )

    print(f"\nStarting feature extraction for {len(dataset):,} samples...")
    saved_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Caching HuBERT-Large")):
            wav = batch["wav"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            fids = batch["file_id"]

            if args.normalize_wav:
                wav = normalize_waveform(wav, attention_mask)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                hidden = backbone(input_values=wav, attention_mask=attention_mask).last_hidden_state
                valid_mask = make_frame_valid_mask(backbone, attention_mask, hidden.shape[1])

            for i, fid in enumerate(fids):
                clean_fid = Path(fid).stem
                sample_valid = valid_mask[i]
                length = sample_valid.sum().item()
                if length == 0:
                    length = hidden.shape[1]

                sample_hidden = hidden[i, :length].cpu().half()
                sample_mask = sample_valid[:length].cpu().bool()

                out_path = cache_dir / f"{clean_fid}.pt"
                torch.save({
                    "hidden": sample_hidden,
                    "valid_mask": sample_mask,
                    "file_id": clean_fid,
                }, out_path)
                saved_count += 1

            del wav, attention_mask, hidden, valid_mask
            if batch_idx % 20 == 0:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if pending_csv.exists():
        pending_csv.unlink()

    print(f"\nFinished! Successfully cached {saved_count:,} new files in: {cache_dir}")


if __name__ == "__main__":
    main()
