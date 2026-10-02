from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.optim import Adam


# =====================================================================
# PROJECT PATH
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqa.model import QualityHead, quality_loss


# =====================================================================
# CONFIG
# =====================================================================

CSV_PATH = (
    ROOT
    / "data"
    / "qualispeech"
    / "train.csv"
)

CACHE_DIR = (
    ROOT
    / "data"
    / "qualispeech"
    / "hubert_cache"
)

CHECKPOINT_DIR = ROOT / "checkpoints"

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "hubert_cached_tiny_best.pt"
)

SEED = 42

EXPECTED_HIDDEN_SIZE = 768

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

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# UTILITIES
# =====================================================================

def normalize_sample_id(value: str) -> str:

    value = str(value).strip()

    if value.lower().endswith(".wav"):
        value = value[:-4]

    return value


def cache_path_for_sample(
    sample_id: str,
) -> Path:

    sample_id = normalize_sample_id(
        sample_id
    )

    return (
        CACHE_DIR
        / f"{sample_id}.pt"
    )


def ccc_value(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:

    pred = (
        pred.detach()
        .float()
        .view(-1)
    )

    target = (
        target.detach()
        .float()
        .view(-1)
    )

    if pred.numel() < 2:
        return 0.0

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_var = pred.var(
        unbiased=False
    )

    target_var = target.var(
        unbiased=False
    )

    covariance = (
        (pred - pred_mean)
        * (target - target_mean)
    ).mean()

    ccc = (
        2.0 * covariance
        / (
            pred_var
            + target_var
            + (
                pred_mean
                - target_mean
            ).pow(2)
            + 1e-8
        )
    )

    return float(
        ccc.item()
    )


def extract_quality_prediction(
    output,
):

    if torch.is_tensor(output):
        return output

    if isinstance(
        output,
        dict,
    ):

        for key in (
            "quality",
            "quality_score",
            "score",
        ):

            if key in output:
                return output[key]

        raise KeyError(
            "Output dict does not contain "
            "a recognized quality key: "
            f"{list(output.keys())}"
        )

    if isinstance(
        output,
        (tuple, list),
    ):

        if not output:
            raise RuntimeError(
                "Model returned an "
                "empty tuple/list."
            )

        return output[0]

    raise TypeError(
        "Unsupported output type: "
        f"{type(output)}"
    )


# =====================================================================
# CACHED HuBERT DATASET
# =====================================================================

class CachedHuBERTDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        df: pd.DataFrame,
    ):

        self.df = (
            df
            .reset_index(drop=True)
            .copy()
        )

        if len(self.df) != 16:

            raise RuntimeError(
                "Expected 16 samples, "
                f"got {len(self.df)}"
            )

        self.items = []

        for _, row in self.df.iterrows():

            sample_id = normalize_sample_id(
                row["id"]
            )

            target = float(
                row["Overall quality"]
            )

            cache_path = (
                cache_path_for_sample(
                    sample_id
                )
            )

            if not cache_path.exists():

                raise FileNotFoundError(
                    "Missing HuBERT cache:\n"
                    f"{cache_path}"
                )

            item = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=False,
            )

            if "hidden" not in item:

                raise KeyError(
                    f"{cache_path} does not "
                    "contain 'hidden'"
                )

            if "valid_mask" not in item:

                raise KeyError(
                    f"{cache_path} does not "
                    "contain 'valid_mask'"
                )

            hidden = item["hidden"]

            valid_mask = item[
                "valid_mask"
            ]

            if hidden.ndim != 2:

                raise RuntimeError(
                    f"{sample_id}: expected "
                    f"hidden [T,H], "
                    f"got {hidden.shape}"
                )

            if (
                hidden.shape[1]
                != EXPECTED_HIDDEN_SIZE
            ):

                raise RuntimeError(
                    f"{sample_id}: expected "
                    f"hidden size "
                    f"{EXPECTED_HIDDEN_SIZE}, "
                    f"got {hidden.shape[1]}"
                )

            if valid_mask.ndim != 1:

                raise RuntimeError(
                    f"{sample_id}: expected "
                    f"valid_mask [T], "
                    f"got {valid_mask.shape}"
                )

            if (
                valid_mask.shape[0]
                != hidden.shape[0]
            ):

                raise RuntimeError(
                    f"{sample_id}: hidden and "
                    "valid_mask length "
                    "mismatch: "
                    f"{hidden.shape[0]} vs "
                    f"{valid_mask.shape[0]}"
                )

            self.items.append(
                {
                    "hidden": hidden.float(),
                    "valid_mask": (
                        valid_mask.bool()
                    ),
                    "target": torch.tensor(
                        target,
                        dtype=torch.float32,
                    ),
                    "file_id": sample_id,
                }
            )

    def __len__(self):

        return len(self.items)

    def __getitem__(
        self,
        index,
    ):

        return self.items[index]


# =====================================================================
# COLLATE
# =====================================================================

def cached_hubert_collate(
    batch,
):

    max_frames = max(
        item["hidden"].shape[0]
        for item in batch
    )

    batch_size = len(batch)

    hidden = torch.zeros(
        batch_size,
        max_frames,
        EXPECTED_HIDDEN_SIZE,
        dtype=torch.float32,
    )

    valid_mask = torch.zeros(
        batch_size,
        max_frames,
        dtype=torch.bool,
    )

    targets = []
    file_ids = []

    for i, item in enumerate(
        batch
    ):

        frames = item[
            "hidden"
        ].shape[0]

        hidden[
            i,
            :frames
        ] = item["hidden"]

        valid_mask[
            i,
            :frames
        ] = item["valid_mask"]

        targets.append(
            item["target"]
        )

        file_ids.append(
            item["file_id"]
        )

    targets = torch.stack(
        targets
    )

    return {
        "hidden": hidden,
        "valid_mask": valid_mask,
        "target": targets,
        "file_id": file_ids,
    }


# =====================================================================
# FULL DATASET EVALUATION
# =====================================================================

def evaluate_full_dataset(
    model,
    dataset,
):

    model.eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=cached_hubert_collate,
    )

    all_predictions = []
    all_targets = []
    all_ids = []

    with torch.no_grad():

        for batch in loader:

            hidden = batch[
                "hidden"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            valid_mask = batch[
                "valid_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            target = batch[
                "target"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            output = model(
                hidden,
                valid_mask=valid_mask,
            )

            pred = (
                extract_quality_prediction(
                    output
                )
            )

            pred = pred.view(-1)
            target = target.view(-1)

            all_predictions.append(
                pred.cpu()
            )

            all_targets.append(
                target.cpu()
            )

            all_ids.extend(
                batch["file_id"]
            )

    predictions = torch.cat(
        all_predictions
    )

    targets = torch.cat(
        all_targets
    )

    if (
        len(predictions)
        != len(dataset)
    ):

        raise RuntimeError(
            f"Expected {len(dataset)} "
            "predictions, "
            f"got {len(predictions)}"
        )

    mse = torch.mean(
        (
            predictions
            - targets
        ) ** 2
    ).item()

    mae = torch.mean(
        torch.abs(
            predictions
            - targets
        )
    ).item()

    ccc = ccc_value(
        predictions,
        targets,
    )

    loss = quality_loss(
        predictions,
        targets,
    ).item()

    return (
        mse,
        mae,
        ccc,
        loss,
        predictions,
        targets,
        all_ids,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )

    print("=" * 72)
    print(
        "QualiSpeech HuBERT "
        "CACHED REAL TINY OVERFIT"
    )
    print("=" * 72)

    print(
        f"Device:       {DEVICE}"
    )

    print(
        f"CSV:          {CSV_PATH}"
    )

    print(
        f"Cache:        {CACHE_DIR}"
    )

    print(
        f"Checkpoint:   "
        f"{BEST_CHECKPOINT_PATH}"
    )

    print(
        f"Hidden size:  "
        f"{EXPECTED_HIDDEN_SIZE}"
    )

    print()

    # =================================================================
    # 1. LOAD CSV
    # =================================================================

    df = pd.read_csv(
        CSV_PATH
    )

    df["sample_id"] = (
        df["id"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.wav$",
            "",
            regex=True,
        )
    )

    df["target"] = pd.to_numeric(
        df["Overall quality"],
        errors="raise",
    )

    # =================================================================
    # 2. SELECT ONLY SAMPLES WHOSE HuBERT CACHE EXISTS
    #
    # This is the important fix.
    #
    # cache_hubert.py already created exactly 16 cached samples.
    # We must select from those cached samples instead of blindly
    # taking the first rows from the CSV.
    # =================================================================

    selected_parts = []

    print(
        "Searching for cached samples "
        "by target class..."
    )

    for quality, count in (
        SAMPLES_PER_CLASS.items()
    ):

        candidates = df[
            df["Overall quality"]
            == quality
        ].copy()

        candidates[
            "cache_exists"
        ] = candidates[
            "sample_id"
        ].map(
            lambda sample_id:
                cache_path_for_sample(
                    sample_id
                ).exists()
        )

        candidates = candidates[
            candidates["cache_exists"]
        ].copy()

        if len(candidates) < count:

            raise RuntimeError(
                f"Not enough cached samples "
                f"for quality={quality}: "
                f"need={count}, "
                f"found={len(candidates)}"
            )

        selected_parts.append(
            candidates.iloc[:count]
        )

    tiny_df = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    tiny_df = tiny_df.drop(
        columns=["cache_exists"]
    )

    if len(tiny_df) != 16:

        raise RuntimeError(
            "Expected 16 tiny samples, "
            f"got {len(tiny_df)}"
        )

    # =================================================================
    # 3. PRINT EXACT SELECTED SUBSET
    # =================================================================

    print()
    print(
        "Selected cached HuBERT tiny subset:"
    )

    for _, row in tiny_df.iterrows():

        sample_id = row[
            "sample_id"
        ]

        target = float(
            row["Overall quality"]
        )

        cache_path = (
            cache_path_for_sample(
                sample_id
            )
        )

        print(
            f"{sample_id:35s} "
            f"target={target:.1f} "
            f"cache=OK"
        )

    print()

    print(
        "Tiny target distribution:"
    )

    print(
        tiny_df[
            "Overall quality"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    # =================================================================
    # 4. FINAL CACHE VERIFICATION
    # =================================================================

    print(
        "Verifying all selected caches..."
    )

    for sample_id in tiny_df[
        "sample_id"
    ]:

        cache_path = (
            cache_path_for_sample(
                sample_id
            )
        )

        if not cache_path.exists():

            raise FileNotFoundError(
                "Missing cache:\n"
                f"{cache_path}"
            )

    print(
        f"All {len(tiny_df)} "
        "HuBERT caches exist."
    )

    print()

    # =================================================================
    # 5. DATASET
    # =================================================================

    dataset = (
        CachedHuBERTDataset(
            tiny_df
        )
    )

    if len(dataset) != 16:

        raise RuntimeError(
            f"Expected dataset size 16, "
            f"got {len(dataset)}"
        )

    # =================================================================
    # 6. DATALOADER
    # =================================================================

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=cached_hubert_collate,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    # =================================================================
    # 7. HuBERT QUALITY HEAD
    # =================================================================

    print(
        "Creating HuBERT QualityHead..."
    )

    model = QualityHead(
        hidden_size=EXPECTED_HIDDEN_SIZE
    ).to(
        DEVICE
    )

    trainable = [
        (
            name,
            parameter,
        )
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    print()
    print(
        "Trainable parameters:"
    )

    for name, parameter in trainable:

        print(
            f"  {name}: "
            f"{parameter.numel():,} "
            "parameters"
        )

    total_trainable = sum(
        parameter.numel()
        for _, parameter
        in trainable
    )

    print(
        f"Total trainable parameters: "
        f"{total_trainable:,}"
    )

    print()

    if total_trainable == 0:

        raise RuntimeError(
            "No trainable parameters found."
        )

    optimizer = Adam(
        [
            parameter
            for _, parameter
            in trainable
        ],
        lr=LR,
    )

    # =================================================================
    # 8. INITIAL FORWARD
    # =================================================================

    model.eval()

    with torch.no_grad():

        first_batch = next(
            iter(loader)
        )

        hidden = first_batch[
            "hidden"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        valid_mask = first_batch[
            "valid_mask"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        target = first_batch[
            "target"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        pred = model(
            hidden,
            valid_mask=valid_mask,
        )

        pred = pred.view(-1)
        target = target.view(-1)

        initial_mse = torch.mean(
            (
                pred
                - target
            ) ** 2
        ).item()

        initial_ccc = ccc_value(
            pred,
            target,
        )

        print(
            f"Initial batch: "
            f"B={hidden.shape[0]}, "
            f"T={hidden.shape[1]}, "
            f"H={hidden.shape[2]}"
        )

        print(
            f"Initial MSE="
            f"{initial_mse:.6f}, "
            f"CCC="
            f"{initial_ccc:.6f}"
        )

        print(
            "Initial predictions:",
            [
                round(
                    float(x),
                    4,
                )
                for x in pred.cpu()
            ],
        )

        print()

    # =================================================================
    # 9. TRAINING
    # =================================================================

    best_full_mse = float(
        "inf"
    )

    best_full_mae = float(
        "inf"
    )

    best_full_ccc = float(
        "-inf"
    )

    best_full_loss = float(
        "inf"
    )

    best_step = 0

    best_state = None

    for step in range(
        1,
        STEPS + 1,
    ):

        model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_ccc = 0.0
        batches = 0

        for batch in loader:

            hidden = batch[
                "hidden"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            valid_mask = batch[
                "valid_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            target = batch[
                "target"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                hidden,
                valid_mask=valid_mask,
            )

            pred = pred.view(-1)
            target = target.view(-1)

            loss = quality_loss(
                pred,
                target,
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Non-finite loss at "
                    f"step {step}: "
                    f"{loss.item()}"
                )

            loss.backward()

            # ---------------------------------------------------------
            # Gradient check
            # ---------------------------------------------------------

            bad_gradients = []

            for (
                name,
                parameter,
            ) in trainable:

                if parameter.grad is None:

                    bad_gradients.append(
                        f"{name}: None"
                    )

                elif not torch.isfinite(
                    parameter.grad
                ).all():

                    bad_gradients.append(
                        f"{name}: non-finite"
                    )

            if bad_gradients:

                raise RuntimeError(
                    "Invalid gradients:\n"
                    + "\n".join(
                        bad_gradients
                    )
                )

            optimizer.step()

            batch_mse = torch.mean(
                (
                    pred.detach()
                    - target.detach()
                ) ** 2
            ).item()

            batch_ccc = ccc_value(
                pred,
                target,
            )

            total_loss += float(
                loss.item()
            )

            total_mse += batch_mse

            total_ccc += batch_ccc

            batches += 1

        avg_loss = (
            total_loss
            / batches
        )

        avg_mse = (
            total_mse
            / batches
        )

        avg_ccc = (
            total_ccc
            / batches
        )

        # =============================================================
        # FULL 16-SAMPLE EVALUATION
        # =============================================================

        (
            full_mse,
            full_mae,
            full_ccc,
            full_loss,
            _,
            _,
            _,
        ) = evaluate_full_dataset(
            model,
            dataset,
        )

        # =============================================================
        # BEST CHECKPOINT
        # =============================================================

        if (
            full_mse
            < best_full_mse
        ):

            best_full_mse = (
                full_mse
            )

            best_full_mae = (
                full_mae
            )

            best_full_ccc = (
                full_ccc
            )

            best_full_loss = (
                full_loss
            )

            best_step = step

            best_state = {
                name:
                    parameter
                    .detach()
                    .cpu()
                    .clone()
                for name, parameter
                in trainable
            }

            CHECKPOINT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "backbone":
                        "facebook/"
                        "hubert-base-ls960",

                    "hidden_size":
                        EXPECTED_HIDDEN_SIZE,

                    "step":
                        best_step,

                    "mse":
                        best_full_mse,

                    "mae":
                        best_full_mae,

                    "ccc":
                        best_full_ccc,

                    "loss":
                        best_full_loss,

                    "quality_head_state_dict":
                        {
                            name:
                                tensor.clone()
                            for name, tensor
                            in best_state.items()
                        },

                    "seed":
                        SEED,

                    "samples_per_class":
                        SAMPLES_PER_CLASS,

                    "batch_size":
                        BATCH_SIZE,

                    "steps":
                        STEPS,

                    "lr":
                        LR,
                },
                BEST_CHECKPOINT_PATH,
            )

            print(
                f"  -> NEW BEST FULL-16 "
                f"CHECKPOINT: "
                f"step={best_step}, "
                f"MSE={best_full_mse:.6f}, "
                f"CCC={best_full_ccc:.6f}, "
                f"Loss={best_full_loss:.6f}"
            )

            print(
                f"     Saved checkpoint: "
                f"{BEST_CHECKPOINT_PATH}"
            )

        # =============================================================
        # PROGRESS
        # =============================================================

        if (
            step == 1
            or step % 25 == 0
            or step == STEPS
        ):

            print(
                f"step {step:3d} | "
                f"train_loss="
                f"{avg_loss:.6f} | "
                f"train_MSE="
                f"{avg_mse:.6f} | "
                f"train_CCC="
                f"{avg_ccc:.6f} | "
                f"full16_MSE="
                f"{full_mse:.6f} | "
                f"full16_CCC="
                f"{full_ccc:.6f}"
            )

    # =================================================================
    # 10. RESTORE BEST CHECKPOINT
    # =================================================================

    if best_state is None:

        raise RuntimeError(
            "No best checkpoint "
            "was saved."
        )

    with torch.no_grad():

        for (
            name,
            parameter,
        ) in trainable:

            parameter.copy_(
                best_state[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )

    print()

    print(
        f"Restored BEST FULL-16 "
        f"checkpoint: step={best_step}"
    )

    print(
        f"Checkpoint: "
        f"{BEST_CHECKPOINT_PATH}"
    )

    print(
        f"Best FULL-16 MSE: "
        f"{best_full_mse:.6f}"
    )

    print(
        f"Best FULL-16 CCC: "
        f"{best_full_ccc:.6f}"
    )

    # =================================================================
    # 11. FINAL EVALUATION
    # =================================================================

    (
        final_mse,
        final_mae,
        final_ccc,
        final_loss,
        predictions,
        targets,
        all_ids,
    ) = evaluate_full_dataset(
        model,
        dataset,
    )

    # =================================================================
    # 12. FINAL RESULT
    # =================================================================

    print()

    print("=" * 72)

    print(
        "FINAL RESULT — "
        "HuBERT BEST FULL-16 CHECKPOINT"
    )

    print("=" * 72)

    print(
        "Backbone:   "
        "facebook/hubert-base-ls960"
    )

    print(
        f"Hidden:     "
        f"{EXPECTED_HIDDEN_SIZE}"
    )

    print(
        f"Best step:  "
        f"{best_step}"
    )

    print(
        f"Loss:       "
        f"{final_loss:.6f}"
    )

    print(
        f"MSE:        "
        f"{final_mse:.6f}"
    )

    print(
        f"MAE:        "
        f"{final_mae:.6f}"
    )

    print(
        f"CCC:        "
        f"{final_ccc:.6f}"
    )

    print()

    print(
        f"{'file_id':40s} "
        f"{'target':>8s} "
        f"{'pred':>8s} "
        f"{'abs_err':>8s}"
    )

    print(
        "-" * 72
    )

    for (
        file_id,
        target,
        pred,
    ) in zip(
        all_ids,
        targets.tolist(),
        predictions.tolist(),
    ):

        print(
            f"{file_id:40s} "
            f"{target:8.3f} "
            f"{pred:8.3f} "
            f"{abs(pred - target):8.3f}"
        )

    # =================================================================
    # 13. ACCEPTANCE CRITERIA
    # =================================================================

    passed = (
        final_mse < 0.10
        and final_ccc > 0.95
        and torch.isfinite(
            predictions
        ).all()
    )

    print()

    if passed:

        print(
            ">>> CACHED HuBERT "
            "TINY OVERFIT: PASS"
        )

        print(
            "Full 16-sample evaluation "
            "satisfies:"
        )

        print(
            f"  MSE < 0.10  : "
            f"{final_mse:.6f}"
        )

        print(
            f"  CCC > 0.95  : "
            f"{final_ccc:.6f}"
        )

        print()

        print(
            "HuBERT cached-feature -> "
            "QualityHead -> loss -> "
            "backward pipeline is working."
        )

    else:

        print(
            ">>> CACHED HuBERT "
            "TINY OVERFIT: FAIL"
        )

        print(
            "Do NOT start full "
            "HuBERT training yet."
        )

        print(
            "Inspect cache -> "
            "valid_mask -> "
            "AttentionPool -> "
            "QualityHead -> "
            "loss -> gradients."
        )

    print()

    if not passed:
        raise SystemExit(1)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()