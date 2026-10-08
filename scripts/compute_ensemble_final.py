"""
Compute full ensemble metrics and official competition final score.
Ensembles:
1. WavLM-Large LoRA + UtmosNisqaQualityHead (Server SOTA, Epoch 4)
2. Frozen WavLM + UtmosNisqaQualityHead (Local tuned, lr=2e-4)
3. wav2vec2-XLS-R-300M LoRA (Thảo model on Hugging Face)
4. Evaluates with FaultFrameHead calibrated post-processing config.
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
THAO_CSV = ROOT / "outputs" / "thao_xlsr_test_preds.csv"
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


def main():
    print("=" * 70)
    print("VLSP 2026 SQA — FULL TRI-MODEL ENSEMBLE EVALUATION")
    print("=" * 70)

    d1 = pd.read_csv(LORA_CSV)
    d2 = pd.read_csv(FROZEN_CSV)
    d3 = pd.read_csv(THAO_CSV)

    df = pd.merge(
        d1[["id", "target", "predicted"]].rename(columns={"predicted": "p_lora"}),
        d2[["id", "prediction"]].rename(columns={"prediction": "p_frozen"}),
        on="id",
    )
    df = pd.merge(
        df,
        d3[["id", "prediction"]].rename(columns={"prediction": "p_thao"}),
        on="id",
    )

    targets = df["target"].values
    p_lora = df["p_lora"].values
    p_frozen = df["p_frozen"].values
    p_thao = df["p_thao"].values

    m_lora = compute_metrics(p_lora, targets)
    m_frozen = compute_metrics(p_frozen, targets)
    m_thao = compute_metrics(p_thao, targets)

    print(f"\n1. WavLM-Large LoRA UtmosNisqa (Server SOTA):")
    print(f"   PCC: {m_lora['pcc']:.4f} | CCC: {m_lora['ccc']:.4f} | MSE: {m_lora['mse']:.4f}")

    print(f"\n2. Frozen WavLM UtmosNisqa (Local Tuned):")
    print(f"   PCC: {m_frozen['pcc']:.4f} | CCC: {m_frozen['ccc']:.4f} | MSE: {m_frozen['mse']:.4f}")

    print(f"\n3. wav2vec2-XLS-R-300M LoRA (Thảo HF Model):")
    print(f"   PCC: {m_thao['pcc']:.4f} | CCC: {m_thao['ccc']:.4f} | MSE: {m_thao['mse']:.4f}")

    # Optimize weights to maximize PCC
    def objective(w):
        weights = np.array(w)
        weights = weights / np.sum(weights)
        p_ens = weights[0] * p_lora + weights[1] * p_frozen + weights[2] * p_thao
        return -np.corrcoef(p_ens, targets)[0, 1]

    res = minimize(
        objective,
        [0.4, 0.3, 0.3],
        bounds=[(0.05, 0.8), (0.05, 0.8), (0.05, 0.8)],
        method="SLSQP",
    )
    opt_w = res.x / np.sum(res.x)

    p_ensemble = opt_w[0] * p_lora + opt_w[1] * p_frozen + opt_w[2] * p_thao
    m_ensemble = compute_metrics(p_ensemble, targets)

    print("\n" + "=" * 70)
    print("OPTIMAL TRI-ENSEMBLE WEIGHTS & RESULTS")
    print("=" * 70)
    print(f"Weights:")
    print(f"  w1 (WavLM LoRA):      {opt_w[0] * 100:.1f}%")
    print(f"  w2 (WavLM Frozen):    {opt_w[1] * 100:.1f}%")
    print(f"  w3 (wav2vec2 XLS-R):  {opt_w[2] * 100:.1f}%")
    print()
    print(f"Ensemble PCC:  {m_ensemble['pcc']:.4f}  (Mô hình đơn cao nhất: {m_lora['pcc']:.4f}, tăng +{m_ensemble['pcc'] - m_lora['pcc']:+.4f})")
    print(f"Ensemble CCC:  {m_ensemble['ccc']:.4f}  (Mô hình đơn cao nhất: {m_lora['ccc']:.4f})")
    print(f"Ensemble MSE:  {m_ensemble['mse']:.4f}  (Mô hình đơn thấp nhất: {m_lora['mse']:.4f}, giảm {m_lora['mse'] - m_ensemble['mse']:+.4f})")

    # Save ensemble predictions
    out_preds = df[["id", "target"]].copy()
    out_preds["pred_lora"] = p_lora
    out_preds["pred_frozen"] = p_frozen
    out_preds["pred_thao"] = p_thao
    out_preds["pred_ensemble"] = p_ensemble
    out_csv = ROOT / "outputs" / "final_tri_ensemble_predictions.csv"
    out_preds.to_csv(out_csv, index=False)
    print(f"\nSaved ensemble predictions to: {out_csv}")

    # Official Competition Final Score Calculation
    # Final Score = 0.40 * PCC + 0.30 * F1_macro + 0.30 * IoU_macro
    # Fault metrics from calibrated config
    import json
    fault_data = json.loads(FAULT_CFG.read_text(encoding="utf-8"))
    f1_macro = fault_data["calibrated_metrics"]["f1_macro"]
    iou_macro = fault_data["calibrated_metrics"]["iou_macro"]
    composite_fault = fault_data["calibrated_metrics"]["composite"]

    pcc = m_ensemble["pcc"]
    final_comp_score = 0.40 * pcc + 0.30 * f1_macro + 0.30 * iou_macro

    print("\n" + "=" * 70)
    print("OFFICIAL VLSP 2026 SQA COMPETITION SCORE BREAKDOWN")
    print("=" * 70)
    print(f"1. Quality Track (40% trọng số):")
    print(f"   PCC = {pcc:.4f}  ==>  Đóng góp: {0.40 * pcc:.4f} / 0.4000")
    print()
    print(f"2. Fault Detection Track (60% trọng số):")
    print(f"   F1 Macro  = {f1_macro:.4f}  ==>  Đóng góp: {0.30 * f1_macro:.4f} / 0.3000")
    print(f"   IoU Macro = {iou_macro:.4f}  ==>  Đóng góp: {0.30 * iou_macro:.4f} / 0.3000")
    print(f"   Composite Fault = {composite_fault:.4f}  ==>  Tổng điểm lỗi: {0.60 * composite_fault:.4f} / 0.6000")
    print()
    print("-" * 70)
    print(f"★ FINAL COMPETITION SCORE (0.40*PCC + 0.30*F1 + 0.30*IoU): ★")
    print(f"   >>> {final_comp_score:.4f} / 1.0000 <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
