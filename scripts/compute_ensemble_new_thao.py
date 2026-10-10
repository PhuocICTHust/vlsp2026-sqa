"""
Benchmark and optimize ensembles incorporating Thảo's new model:
vdmhth123/vlsp2026-sqa-wav2vec2-xls-r-300m-utmos-trainval
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
LORA_CSV = ROOT / "outputs" / "lora_server_test_preds.csv"
FROZEN_CSV = ROOT / "outputs" / "cached_trainval_utmos_tuned" / "test_predictions.csv"
THAO_OLD_CSV = ROOT / "outputs" / "thao_xlsr_test_preds.csv"
THAO_NEW_CSV = ROOT / "outputs" / "thao_xlsr_utmos_test_preds.csv"
HUBERT_CSV = ROOT / "outputs" / "hubert_server_test_preds.csv"
FAULT_CFG = ROOT / "outputs" / "fault_head_trainval" / "calibrated_postprocess_config.json"


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    pcc = float(np.corrcoef(preds, targets)[0, 1])
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    vx, vy = np.var(preds), np.var(targets)
    mx, my = np.mean(preds), np.mean(targets)
    cov = np.mean((preds - mx) * (targets - my))
    ccc = float(2.0 * cov / (vx + vy + (mx - my) ** 2 + 1e-8))
    return {"pcc": pcc, "ccc": ccc, "mse": mse, "mae": mae}


def optimize_ensemble(pred_matrix: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    n_models = pred_matrix.shape[1]

    def obj(w):
        w = w / np.sum(w)
        comb = pred_matrix @ w
        return -float(np.corrcoef(comb, targets)[0, 1])

    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(obj, init_w, bounds=bounds, constraints=constraints, method="SLSQP")
    opt_w = res.x / np.sum(res.x)
    best_pcc = -res.fun
    return opt_w, best_pcc


def main():
    print("=" * 70)
    print("VLSP 2026 SQA — EVALUATION WITH NEW THẢO XLS-R UTMOS-NISQA")
    print("=" * 70)

    d_lora = pd.read_csv(LORA_CSV)
    d_frozen = pd.read_csv(FROZEN_CSV)
    d_thao_old = pd.read_csv(THAO_OLD_CSV)
    d_thao_new = pd.read_csv(THAO_NEW_CSV)
    d_hubert = pd.read_csv(HUBERT_CSV)

    df = pd.merge(d_lora[["id", "target", "predicted"]].rename(columns={"predicted": "p_lora"}),
                  d_frozen[["id", "prediction"]].rename(columns={"prediction": "p_frozen"}), on="id")
    df = pd.merge(df, d_thao_old[["id", "prediction"]].rename(columns={"prediction": "p_thao_old"}), on="id")
    df = pd.merge(df, d_thao_new[["id", "prediction"]].rename(columns={"prediction": "p_thao_new"}), on="id")
    df = pd.merge(df, d_hubert[["id", "predicted"]].rename(columns={"predicted": "p_hubert"}), on="id")

    targets = df["target"].values
    p_lora = df["p_lora"].values
    p_frozen = df["p_frozen"].values
    p_thao_old = df["p_thao_old"].values
    p_thao_new = df["p_thao_new"].values
    p_hubert = df["p_hubert"].values

    print("\n--- INDIVIDUAL MODEL PERFORMANCE ON DEV (1,852 SAMPLES) ---")
    models = {
        "1. WavLM-Large LoRA UtmosNisqa": p_lora,
        "2. Frozen WavLM UtmosNisqa": p_frozen,
        "3. Thảo XLS-R (Old AttentionPool)": p_thao_old,
        "4. Thảo XLS-R (New UtmosNisqa)": p_thao_new,
        "5. HuBERT-Large LoRA UtmosNisqa": p_hubert,
    }

    for name, p in models.items():
        m = compute_metrics(p, targets)
        print(f"{name:38s} | PCC: {m['pcc']:.4f} | CCC: {m['ccc']:.4f} | MSE: {m['mse']:.4f} | MAE: {m['mae']:.4f}")

    with open(FAULT_CFG, "r", encoding="utf-8") as f:
        fault_data = json.load(f)
    f1_macro = fault_data["calibrated_metrics"]["f1_macro"]
    iou_macro = fault_data["calibrated_metrics"]["iou_macro"]

    mat_old = np.column_stack([p_lora, p_frozen, p_thao_old])
    w_old, pcc_old = optimize_ensemble(mat_old, targets)
    p_comb_old = mat_old @ w_old
    m_comb_old = compute_metrics(p_comb_old, targets)
    score_old = 0.40 * m_comb_old["pcc"] + 0.30 * f1_macro + 0.30 * iou_macro

    print("\n" + "=" * 70)
    print("1. PREVIOUS SOTA TRI-ENSEMBLE (with Old Thảo Model):")
    print(f"   Weights: {w_old[0]*100:.1f}% WavLM LoRA + {w_old[1]*100:.1f}% Frozen + {w_old[2]*100:.1f}% Thảo Old")
    print(f"   PCC: {m_comb_old['pcc']:.4f} | CCC: {m_comb_old['ccc']:.4f} | MSE: {m_comb_old['mse']:.4f}")
    print(f"   FINAL COMPETITION SCORE: {score_old:.4f} ({score_old*100:.2f}%)")

    mat_new = np.column_stack([p_lora, p_frozen, p_thao_new])
    w_new, pcc_new = optimize_ensemble(mat_new, targets)
    p_comb_new = mat_new @ w_new
    m_comb_new = compute_metrics(p_comb_new, targets)
    score_new = 0.40 * m_comb_new["pcc"] + 0.30 * f1_macro + 0.30 * iou_macro

    print("\n" + "=" * 70)
    print("2. UPGRADED TRI-ENSEMBLE (with NEW Thảo XLS-R UtmosNisqa):")
    print(f"   Weights: {w_new[0]*100:.1f}% WavLM LoRA + {w_new[1]*100:.1f}% Frozen + {w_new[2]*100:.1f}% Thảo New")
    print(f"   PCC: {m_comb_new['pcc']:.4f} | CCC: {m_comb_new['ccc']:.4f} | MSE: {m_comb_new['mse']:.4f}")
    print(f"   FINAL COMPETITION SCORE: {score_new:.4f} ({score_new*100:.2f}%)")

    mat_quad = np.column_stack([p_lora, p_frozen, p_thao_new, p_hubert])
    w_quad, pcc_quad = optimize_ensemble(mat_quad, targets)
    p_comb_quad = mat_quad @ w_quad
    m_comb_quad = compute_metrics(p_comb_quad, targets)
    score_quad = 0.40 * m_comb_quad["pcc"] + 0.30 * f1_macro + 0.30 * iou_macro

    print("\n" + "=" * 70)
    print("3. QUAD-ENSEMBLE (WavLM LoRA + Frozen + New Thảo + HuBERT):")
    print(f"   Weights: {w_quad[0]*100:.1f}% LoRA + {w_quad[1]*100:.1f}% Frozen + {w_quad[2]*100:.1f}% Thảo + {w_quad[3]*100:.1f}% HuBERT")
    print(f"   PCC: {m_comb_quad['pcc']:.4f} | CCC: {m_comb_quad['ccc']:.4f} | MSE: {m_comb_quad['mse']:.4f}")
    print(f"   FINAL COMPETITION SCORE: {score_quad:.4f} ({score_quad*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
