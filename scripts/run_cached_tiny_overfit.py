from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import copy
import io
import random
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset, DataLoader

from src.sqa.model import SQAModel, quality_loss


# ============================================================================
# CONFIG
# ============================================================================

CSV_PATH = ROOT / "data" / "qualispeech" / "train.csv"

ZIP_PATH = ROOT / "data" / "qualispeech" / "wav_part1.zip"

CACHE_DIR = ROOT / "data" / "qualispeech" / "wavlm_cache"

CHECKPOINT_PATH = ROOT / "checkpoints" / "cached_tiny_best.pt"

MODEL_NAME = "microsoft/wavlm-large"

SEED = 42

SAMPLES_PER_CLASS = {
    1: 3,
    2: 3,
    3: 4,
    4: 3,
    5: 3,
}

BATCH_SIZE = 8

STEPS = 300

LR = 1e-3

EXPECTED_HIDDEN_SIZE = 1024

MAX_HIDDEN_DIFF = 5e-3

MAX_PRED_DIFF = 1e-4


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# SEED
# ============================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# TINY INDICES
# ============================================================================

def build_tiny_indices(df):

    selected = []

    for quality, count in SAMPLES_PER_CLASS.items():

        candidates = df.index[
            df["Overall quality"] == quality
        ].tolist()

        if len(candidates) < count:
            raise RuntimeError(
                f"Not enough samples for quality={quality}"
            )

        selected.extend(
            candidates[:count]
        )

    if len(selected) != 16:
        raise RuntimeError(
            f"Expected 16 samples, got {len(selected)}"
        )

    return selected


# ============================================================================
# CCC
# ============================================================================

def ccc_value(pred, gt):

    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    pred_mean = pred.mean()
    gt_mean = gt.mean()

    pred_var = pred.var(unbiased=False)
    gt_var = gt.var(unbiased=False)

    covariance = (
        (pred - pred_mean)
        * (gt - gt_mean)
    ).mean()

    denominator = (
        pred_var
        + gt_var
        + (pred_mean - gt_mean).pow(2)
    )

    return (
        2.0 * covariance
    ) / (
        denominator + 1e-8
    )


# ============================================================================
# CACHE
# ============================================================================

def cache_path_for(file_id):

    file_id = str(file_id)

    if file_id.lower().endswith(".wav"):
        stem = file_id[:-4]
    else:
        stem = Path(file_id).stem

    return CACHE_DIR / f"{stem}.pt"


def load_cache(file_id):

    path = cache_path_for(file_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache:\n{path}"
        )

    item = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "hidden",
        "valid_mask",
        "target",
        "file_id",
    }

    missing = required - set(item.keys())

    if missing:
        raise RuntimeError(
            f"{path}: missing keys {missing}"
        )

    hidden = item["hidden"].float()
    valid_mask = item["valid_mask"].bool()

    if hidden.ndim != 2:
        raise RuntimeError(
            f"{file_id}: hidden shape "
            f"{hidden.shape}"
        )

    if hidden.shape[1] != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(
            f"{file_id}: expected hidden size "
            f"{EXPECTED_HIDDEN_SIZE}, "
            f"got {hidden.shape[1]}"
        )

    if valid_mask.ndim != 1:
        raise RuntimeError(
            f"{file_id}: valid_mask shape "
            f"{valid_mask.shape}"
        )

    if hidden.shape[0] != valid_mask.shape[0]:
        raise RuntimeError(
            f"{file_id}: hidden/mask mismatch"
        )

    if not torch.isfinite(hidden).all():
        raise RuntimeError(
            f"{file_id}: hidden contains NaN/Inf"
        )

    if int(valid_mask.sum()) <= 0:
        raise RuntimeError(
            f"{file_id}: no valid frames"
        )

    cached_file_id = str(item["file_id"])

    if cached_file_id != str(file_id):
        raise RuntimeError(
            f"file_id mismatch:\n"
            f"CSV:   {file_id}\n"
            f"Cache: {cached_file_id}"
        )

    target = float(
        item["target"]
    )

    return {
        "hidden": hidden,
        "valid_mask": valid_mask,
        "target": target,
        "file_id": cached_file_id,
    }


# ============================================================================
# CACHED DATASET
# ============================================================================

class CachedTinyDataset(Dataset):

    def __init__(self, csv_path, indices):

        self.df = pd.read_csv(csv_path)

        self.samples = []

        for csv_index in indices:

            row = self.df.iloc[csv_index]

            file_id = str(row["id"])

            csv_target = float(
                row["Overall quality"]
            )

            cache = load_cache(file_id)

            if abs(
                cache["target"]
                - csv_target
            ) > 1e-6:

                raise RuntimeError(
                    f"Target mismatch: {file_id}"
                )

            self.samples.append({
                "hidden": cache["hidden"],
                "valid_mask": cache["valid_mask"],
                "target": csv_target,
                "file_id": file_id,
                "csv_index": csv_index,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


# ============================================================================
# COLLATE
# ============================================================================

def cached_collate(batch):

    max_t = max(
        x["hidden"].shape[0]
        for x in batch
    )

    hidden_size = (
        batch[0]["hidden"].shape[1]
    )

    hidden = torch.zeros(
        len(batch),
        max_t,
        hidden_size,
        dtype=torch.float32,
    )

    valid_mask = torch.zeros(
        len(batch),
        max_t,
        dtype=torch.bool,
    )

    target = torch.tensor(
        [
            x["target"]
            for x in batch
        ],
        dtype=torch.float32,
    )

    for i, item in enumerate(batch):

        t = item["hidden"].shape[0]

        hidden[i, :t] = item["hidden"]

        valid_mask[i, :t] = (
            item["valid_mask"]
        )

    return {
        "hidden": hidden,
        "valid_mask": valid_mask,
        "target": target,
        "file_id": [
            x["file_id"]
            for x in batch
        ],
    }


# ============================================================================
# LOAD WAV DIRECTLY FROM ZIP
# ============================================================================

class ZipAudioLoader:

    def __init__(self, zip_path):

        self.zip_path = zip_path

        self.zf = zipfile.ZipFile(
            zip_path,
            "r",
        )

        self.name_map = {}

        for name in self.zf.namelist():

            if name.lower().endswith(".wav"):

                basename = Path(name).name

                self.name_map[
                    basename
                ] = name

    def get(self, file_id):

        file_id = Path(
            str(file_id)
        ).name

        if file_id not in self.name_map:

            raise FileNotFoundError(
                f"{file_id} not found in ZIP.\n"
                f"Available WAV files: "
                f"{len(self.name_map)}"
            )

        zip_name = self.name_map[
            file_id
        ]

        raw = self.zf.read(
            zip_name
        )

        waveform, sample_rate = (
            torchaudio.load(
                io.BytesIO(raw)
            )
        )

        return waveform, sample_rate

    def close(self):

        self.zf.close()


# ============================================================================
# WAV -> MODEL INPUT
# ============================================================================

def prepare_wav(
    waveform,
    sample_rate,
):

    # WavLM expects mono waveform.
    if waveform.ndim == 2:

        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

    waveform = waveform.squeeze(0)

    # QualiSpeech audio is expected to be
    # compatible with WavLM's 16 kHz input.
    if sample_rate != 16000:

        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            16000,
        )

    waveform = waveform.float()

    return waveform


# ============================================================================
# CACHE vs ORIGINAL
# ============================================================================

@torch.no_grad()
def compare_cached_vs_original(
    model,
    cached_dataset,
    zip_audio,
    device,
):

    print()
    print("=" * 72)
    print(
        "CACHE vs ORIGINAL WAV SANITY CHECK"
    )
    print("=" * 72)

    model.eval()

    max_hidden_diff = 0.0

    max_pred_diff = 0.0

    for i in range(
        len(cached_dataset)
    ):

        item = cached_dataset[i]

        file_id = item[
            "file_id"
        ]

        # --------------------------------------------------------------------
        # Cached representation
        # --------------------------------------------------------------------

        h_cached = item[
            "hidden"
        ].to(device)

        valid_cached = item[
            "valid_mask"
        ].to(device)

        valid_count = int(
            valid_cached.sum()
        )

        h_cached_valid = (
            h_cached[:valid_count]
        )

        mask_cached = torch.ones(
            1,
            valid_count,
            dtype=torch.bool,
            device=device,
        )

        pred_cached = (
            model.quality_head(
                h_cached_valid.unsqueeze(0),
                mask_cached,
            )
            .squeeze(0)
        )

        # --------------------------------------------------------------------
        # Original WAV
        # --------------------------------------------------------------------

        waveform, sample_rate = (
            zip_audio.get(file_id)
        )

        waveform = prepare_wav(
            waveform,
            sample_rate,
        )

        waveform = waveform.to(
            device
        )

        wav = waveform.unsqueeze(0)

        # IMPORTANT:
        #
        # We intentionally do not provide an attention mask here.
        #
        # This makes the comparison:
        #
        #     raw WAV -> WavLM
        #
        # against the cached WavLM output for the exact unpadded audio.
        #
        h_original = model.backbone(
            wav,
            attention_mask=None,
        )

        h_original = (
            h_original.squeeze(0)
        )

        if h_original.shape[0] != valid_count:

            raise RuntimeError(
                f"{file_id}: frame count mismatch:\n"
                f"cached={valid_count}\n"
                f"original={h_original.shape[0]}"
            )

        pred_original = (
            model.quality_head(
                h_original.unsqueeze(0),
                torch.ones(
                    1,
                    valid_count,
                    dtype=torch.bool,
                    device=device,
                ),
            )
            .squeeze(0)
        )

        # --------------------------------------------------------------------
        # Differences
        # --------------------------------------------------------------------

        hidden_diff = (
            h_cached_valid
            - h_original
        ).abs().max().item()

        pred_diff = (
            pred_cached
            - pred_original
        ).abs().item()

        max_hidden_diff = max(
            max_hidden_diff,
            hidden_diff,
        )

        max_pred_diff = max(
            max_pred_diff,
            pred_diff,
        )

        print(
            f"[{i + 1:2d}/16] "
            f"{file_id:<40} "
            f"hidden_max_diff="
            f"{hidden_diff:.8f} "
            f"pred_diff="
            f"{pred_diff:.8f}"
        )

    print()

    print(
        f"Maximum hidden difference:     "
        f"{max_hidden_diff:.10f}"
    )

    print(
        f"Maximum prediction difference: "
        f"{max_pred_diff:.10f}"
    )

    print()

    print(
        f"Hidden threshold:     "
        f"{MAX_HIDDEN_DIFF:.10f}"
    )

    print(
        f"Prediction threshold: "
        f"{MAX_PRED_DIFF:.10f}"
    )

    if max_hidden_diff > MAX_HIDDEN_DIFF:

        raise RuntimeError(
            "Cached WavLM representations "
            "differ too much."
        )

    print(
        "Hidden representation numerical sanity: PASS"
    )

    if max_pred_diff > MAX_PRED_DIFF:

        raise RuntimeError(
            "Cached and original predictions "
            "differ too much."
        )

    print(
        "Quality prediction equivalence: PASS"
    )

    print()
    print(
        ">>> CACHE vs ORIGINAL: PASS"
    )


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_cached_dataset(
    model,
    dataset,
    device,
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=cached_collate,
        num_workers=0,
    )

    model.eval()

    preds = []
    targets = []

    for batch in loader:

        hidden = batch[
            "hidden"
        ].to(device)

        valid_mask = batch[
            "valid_mask"
        ].to(device)

        target = batch[
            "target"
        ].to(device)

        pred = model.quality_head(
            hidden,
            valid_mask,
        )

        preds.append(
            pred
        )

        targets.append(
            target
        )

    pred = torch.cat(
        preds
    )

    target = torch.cat(
        targets
    )

    mse = F.mse_loss(
        pred,
        target,
    ).item()

    mae = F.l1_loss(
        pred,
        target,
    ).item()

    ccc = ccc_value(
        pred,
        target,
    ).item()

    loss = quality_loss(
        pred,
        target,
    ).item()

    return {
        "pred": pred.cpu(),
        "gt": target.cpu(),
        "mse": mse,
        "mae": mae,
        "ccc": ccc,
        "loss": loss,
    }


# ============================================================================
# TRAINABLE PARAMETER CHECK
# ============================================================================

def check_trainable_parameters(
    model,
):

    print()
    print(
        "Trainable parameters:"
    )

    total = 0

    for name, param in model.named_parameters():

        if param.requires_grad:

            print(
                f" {name}: "
                f"{param.numel()}"
            )

            if not name.startswith(
                "quality_head."
            ):

                raise RuntimeError(
                    f"Unexpected trainable parameter: "
                    f"{name}"
                )

            total += param.numel()

    print(
        f"Total {total}"
    )

    print(
        "Trainable module check: "
        "QualityHead ONLY"
    )


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_best_checkpoint(
    model,
    optimizer,
    step,
    metrics,
):

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "step": step,
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "ccc": metrics["ccc"],
            "loss": metrics["loss"],
            "quality_head_state_dict":
                copy.deepcopy(
                    model.quality_head.state_dict()
                ),
            "optimizer_state_dict":
                copy.deepcopy(
                    optimizer.state_dict()
                ),
            "config": {
                "model_name": MODEL_NAME,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "steps": STEPS,
                "lr": LR,
                "source": "wavlm_cache",
            },
        },
        CHECKPOINT_PATH,
    )


def restore_best_checkpoint(
    model,
):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.quality_head.load_state_dict(
        checkpoint[
            "quality_head_state_dict"
        ]
    )

    return checkpoint


# ============================================================================
# METRICS
# ============================================================================

def print_metrics(
    title,
    metrics,
):

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"MSE:  {metrics['mse']:.8f}"
    )

    print(
        f"MAE:  {metrics['mae']:.8f}"
    )

    print(
        f"CCC:  {metrics['ccc']:.8f}"
    )

    print(
        f"Loss: {metrics['loss']:.8f}"
    )


def print_predictions(
    metrics,
):

    pred = metrics["pred"]
    gt = metrics["gt"]

    print()
    print("=" * 72)
    print(
        "FINAL FULL-16 PREDICTIONS"
    )
    print("=" * 72)

    print(
        f"{'Index':>5} "
        f"{'GT':>8} "
        f"{'Prediction':>14} "
        f"{'Error':>14}"
    )

    print("-" * 50)

    for i in range(
        len(pred)
    ):

        g = float(gt[i])
        p = float(pred[i])

        print(
            f"{i + 1:5d} "
            f"{g:8.4f} "
            f"{p:14.6f} "
            f"{p - g:14.6f}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    set_seed(SEED)

    print("=" * 72)
    print(
        "QualiSpeech CACHED FEATURE TINY OVERFIT"
    )
    print("=" * 72)

    print(
        f"Device:        {DEVICE}"
    )

    print(
        f"CSV:           {CSV_PATH}"
    )

    print(
        f"ZIP:           {ZIP_PATH}"
    )

    print(
        f"Cache:         {CACHE_DIR}"
    )

    print(
        f"Best checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            CSV_PATH
        )

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            ZIP_PATH
        )

    if not CACHE_DIR.exists():
        raise FileNotFoundError(
            CACHE_DIR
        )

    # ------------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH
    )

    print()
    print(
        f"CSV rows:            {len(df)}"
    )

    # ------------------------------------------------------------------------
    # ZIP
    # ------------------------------------------------------------------------

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:

        wav_names = [
            x
            for x in zf.namelist()
            if x.lower().endswith(".wav")
        ]

    print(
        f"Rows with ZIP audio: "
        f"{len(wav_names)}"
    )

    # ------------------------------------------------------------------------
    # Tiny
    # ------------------------------------------------------------------------

    tiny_indices = build_tiny_indices(
        df
    )

    tiny_df = df.iloc[
        tiny_indices
    ]

    print()
    print(
        "Tiny subset:"
    )

    print(
        tiny_df[
            [
                "id",
                "Overall quality",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Tiny target distribution:"
    )

    print(
        tiny_df[
            "Overall quality"
        ].value_counts(
            sort=False
        ).to_string()
    )

    # ------------------------------------------------------------------------
    # Cached dataset
    # ------------------------------------------------------------------------

    print()
    print(
        "Loading 16 cached representations..."
    )

    cached_dataset = (
        CachedTinyDataset(
            CSV_PATH,
            tiny_indices,
        )
    )

    print(
        f"Cached tiny dataset: "
        f"{len(cached_dataset)} samples"
    )

    # ------------------------------------------------------------------------
    # ZIP loader
    # ------------------------------------------------------------------------

    zip_audio = ZipAudioLoader(
        ZIP_PATH
    )

    # Verify all 16 are actually available.
    print()
    print(
        "Checking tiny WAV files in ZIP..."
    )

    for i in range(
        len(cached_dataset)
    ):

        file_id = cached_dataset[i][
            "file_id"
        ]

        zip_audio.get(
            file_id
        )

    print(
        "Tiny ZIP audio check: PASS"
    )

    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    print()
    print(
        "Loading SQAModel..."
    )

    model = SQAModel(
        model_name=MODEL_NAME,
        freeze_backbone=True,
    ).to(
        DEVICE
    )

    # Freeze backbone.
    for name, param in model.named_parameters():

        if name.startswith(
            "backbone."
        ):

            param.requires_grad = False

    # Freeze fault head.
    for name, param in model.named_parameters():

        if name.startswith(
            "fault_head."
        ):

            param.requires_grad = False

    # QualityHead trainable.
    for param in model.quality_head.parameters():

        param.requires_grad = True

    check_trainable_parameters(
        model
    )

    # ------------------------------------------------------------------------
    # SANITY CHECK
    # ------------------------------------------------------------------------

    compare_cached_vs_original(
        model,
        cached_dataset,
        zip_audio,
        DEVICE,
    )

    # ------------------------------------------------------------------------
    # INITIAL EVAL
    # ------------------------------------------------------------------------

    initial_metrics = (
        evaluate_cached_dataset(
            model,
            cached_dataset,
            DEVICE,
        )
    )

    print_metrics(
        "INITIAL CACHED EVALUATION",
        initial_metrics,
    )

    # ------------------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.quality_head.parameters(),
        lr=LR,
    )

    # ------------------------------------------------------------------------
    # TRAIN LOADER
    # ------------------------------------------------------------------------

    loader = DataLoader(
        cached_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=cached_collate,
        num_workers=0,
    )

    iterator = iter(
        loader
    )

    best_mse = float("inf")

    best_step = -1

    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "CACHED QUALITY HEAD TRAINING"
    )
    print("=" * 72)

    print(
        f"Steps:     {STEPS}"
    )

    print(
        f"Batch:     {BATCH_SIZE}"
    )

    print(
        f"LR:        {LR}"
    )

    print(
        "WavLM:     FROZEN / NOT EXECUTED"
    )

    print(
        "Features:  CACHED"
    )

    for step in range(
        1,
        STEPS + 1,
    ):

        try:
            batch = next(
                iterator
            )

        except StopIteration:

            iterator = iter(
                loader
            )

            batch = next(
                iterator
            )

        hidden = batch[
            "hidden"
        ].to(DEVICE)

        valid_mask = batch[
            "valid_mask"
        ].to(DEVICE)

        target = batch[
            "target"
        ].to(DEVICE)

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model.quality_head(
            hidden,
            valid_mask,
        )

        loss = quality_loss(
            pred,
            target,
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        loss.backward()

        # Only QualityHead may have gradients.
        for name, param in model.named_parameters():

            if param.grad is not None:

                if not name.startswith(
                    "quality_head."
                ):

                    raise RuntimeError(
                        f"Unexpected gradient: "
                        f"{name}"
                    )

        optimizer.step()

        # Evaluate ALL 16.
        metrics = (
            evaluate_cached_dataset(
                model,
                cached_dataset,
                DEVICE,
            )
        )

        if metrics["mse"] < best_mse:

            best_mse = metrics["mse"]

            best_step = step

            save_best_checkpoint(
                model,
                optimizer,
                step,
                metrics,
            )

        if (
            step == 1
            or step % 10 == 0
            or step == STEPS
        ):

            print(
                f"Step {step:3d}/{STEPS} | "
                f"loss={loss.item():.8f} | "
                f"MSE={metrics['mse']:.8f} | "
                f"MAE={metrics['mae']:.8f} | "
                f"CCC={metrics['ccc']:.8f}"
            )

    # ------------------------------------------------------------------------
    # RESTORE BEST
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 72)

    print(
        f"Best step: {best_step}"
    )

    print(
        f"Best MSE:  {best_mse:.8f}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print()
    print(
        "Restoring BEST checkpoint..."
    )

    checkpoint = (
        restore_best_checkpoint(
            model
        )
    )

    print(
        f"Restored step: "
        f"{checkpoint['step']}"
    )

    # ------------------------------------------------------------------------
    # FINAL FULL 16
    # ------------------------------------------------------------------------

    final_metrics = (
        evaluate_cached_dataset(
            model,
            cached_dataset,
            DEVICE,
        )
    )

    print_metrics(
        "FINAL BEST CHECKPOINT — FULL 16 SAMPLES",
        final_metrics,
    )

    print_predictions(
        final_metrics
    )

    # ------------------------------------------------------------------------
    # ACCEPTANCE
    # ------------------------------------------------------------------------

    mse = final_metrics["mse"]

    ccc = final_metrics["ccc"]

    finite = bool(
        torch.isfinite(
            final_metrics["pred"]
        ).all()
    )

    pass_mse = mse < 0.10

    pass_ccc = ccc > 0.95

    print()
    print("=" * 72)
    print(
        "CACHED TINY OVERFIT ACCEPTANCE"
    )
    print("=" * 72)

    print(
        f"MSE < 0.10:   "
        f"{mse:.8f} -> "
        f"{'PASS' if pass_mse else 'FAIL'}"
    )

    print(
        f"CCC > 0.95:   "
        f"{ccc:.8f} -> "
        f"{'PASS' if pass_ccc else 'FAIL'}"
    )

    print(
        f"Finite preds: "
        f"{finite} -> "
        f"{'PASS' if finite else 'FAIL'}"
    )

    if (
        pass_mse
        and pass_ccc
        and finite
    ):

        print()
        print(
            ">>> CACHED TINY OVERFIT: PASS"
        )

    else:

        print()
        print(
            ">>> CACHED TINY OVERFIT: FAIL"
        )

        raise RuntimeError(
            "Tiny overfit did not satisfy "
            "MSE < 0.10 and CCC > 0.95."
        )

    zip_audio.close()


if __name__ == "__main__":
    main()