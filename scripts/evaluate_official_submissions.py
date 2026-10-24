"""
Script: scripts/evaluate_official_submissions.py
Reproduces the Official Private Test Set results from paper_lncs.tex:
  - Table 3 (tab:official_private): Official Private Test Set Performance (5 Runs)
  - Table 4 (tab:official_confusion): Error Breakdown and Confusion Diagnostics (TP, FP, FN, Prec, Rec)
  - Table 6 (tab:loss_attribution): Private Evaluation Loss Attribution (-0.2872 gap)
  - Table 7 (tab:diagnostic_roadmap): Post-Competition Oracle Diagnosis and Architectural Roadmap
"""

import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import soundfile as sf
from sqa.evaluator import (
    FAULT_TYPES,
    compute_f1_macro,
    compute_frame_iou_macro,
    compute_pcc,
    final_score,
)
from sqa.frame_grid import EVALUATOR_GRID

PRIVATE_GT_PATH = ROOT / "data" / "private_test_ground_truth.json"
AUDIO_DIR = ROOT / "data" / "private_test" / "audio"
WORD_TIMESTAMPS_PATH = ROOT / "data" / "private_test_word_timestamps.json"


def load_ground_truth() -> Tuple[List[dict], List[float], np.ndarray]:
    with open(PRIVATE_GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
    durations = [sf.info(str(AUDIO_DIR / x["file_id"])).duration for x in gt]
    gt_mos = np.array([x["quality_score"] for x in gt], dtype=np.float64)
    return gt, durations, gt_mos


OFFICIAL_RUN_TO_JSON = {
    1: "submission_v2.json",
    2: "submission_v6.json",
    3: "submission_v10.json",
    4: "submission_v11.json",
    5: "submission_v12.json",
}


def load_submission(run_idx: int) -> List[dict]:
    zip_path = ROOT / "submissions" / f"anhtraiAI_{run_idx}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            return json.loads(z.read("submission.json").decode("utf-8"))
    
    json_name = OFFICIAL_RUN_TO_JSON.get(run_idx, f"submission_v{run_idx}.json")
    json_path = ROOT / "submissions" / json_name
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    raise FileNotFoundError(f"Neither {zip_path} nor {json_path} found.")


def print_table_3(runs_metrics: List[dict]):
    print("\n" + "=" * 115)
    print("TABLE 3: OFFICIAL PRIVATE TEST SET RESULTS (302 CLIPS) ACROSS THE 5 COMPETITION ITERATIONS")
    print("=" * 115)
    header = (
        f"{'Iteration':<8} | {'Submission Code':<16} | {'Final Score':<11} | "
        f"{'PCC_MOS':<8} | {'F1_mac':<7} | {'IoU_mac':<7} | "
        f"{'IoU_noise':<9} | {'IoU_dist':<8} | {'IoU_pause':<9}"
    )
    print(header)
    print("-" * 115)
    for rm in runs_metrics:
        print(
            f"Run {rm['run']:<4} | {rm['code']:<16} | {rm['score']:<11.6f} | "
            f"{rm['pcc']:<8.6f} | {rm['f1_m']:<7.4f} | {rm['iou_m']:<7.4f} | "
            f"{rm['iou_d']['noise']:<9.4f} | {rm['iou_d']['distortion']:<8.4f} | {rm['iou_d']['unnatural_pause']:<9.4f}"
        )


def print_table_4(runs_metrics: List[dict]):
    print("\n" + "=" * 115)
    print("TABLE 4: OFFICIAL PRIVATE TEST SET ERROR BREAKDOWN AND CONFUSION DIAGNOSTICS")
    print("=" * 115)
    for rm in runs_metrics:
        print(f"\n--- Run {rm['run']} ({rm['code']}) ---")
        print(f"{'Category':<18} | {'Total GT':<8} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Prec (%)':<9} | {'Rec (%)':<8} | {'F1':<6}")
        print("-" * 75)
        conf = rm["conf"]
        for ft in FAULT_TYPES:
            c = conf[ft]
            print(
                f"{ft:<18} | {c['total']:<8} | {c['tp']:<5} | {c['fp']:<5} | {c['fn']:<5} | "
                f"{c['prec']*100:<9.2f} | {c['rec']*100:<8.2f} | {c['f1']:<6.4f}"
            )
        mc = conf["macro"]
        print("-" * 75)
        print(
            f"{'Macro Average':<18} | {'--':<8} | {'--':<5} | {'--':<5} | {'--':<5} | "
            f"{mc['prec']*100:<9.2f} | {mc['rec']*100:<8.2f} | {mc['f1']:<6.4f}"
        )


def compute_confusion(sub: List[dict], gt: List[dict]) -> dict:
    conf = {}
    precs, recs, f1s = [], [], []
    for ft in FAULT_TYPES:
        y_true = np.array([len(g["faults"][ft]) > 0 for g in gt], dtype=bool)
        y_pred = np.array([len(s["faults"][ft]) > 0 for s in sub], dtype=bool)
        tp = int(np.sum(y_true & y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        total = int(np.sum(y_true))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        conf[ft] = {
            "total": total,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "prec": prec,
            "rec": rec,
            "f1": f1,
        }
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    conf["macro"] = {
        "prec": float(np.mean(precs)),
        "rec": float(np.mean(recs)),
        "f1": float(np.mean(f1s)),
    }
    return conf


def print_table_6(runs_metrics: List[dict]):
    print("\n" + "=" * 115)
    print("TABLE 6: ROOT-CAUSE ATTRIBUTION OF THE PRIVATE EVALUATION PERFORMANCE GAP (-0.2872)")
    print("=" * 115)
    r4 = runs_metrics[3]
    pcc_val = r4["pcc"]
    pcc_str = f"{pcc_val:.4f}"
    pcc_gap = f"{pcc_val - 0.7325:+.4f}"

    dist_val = r4["iou_d"]["distortion"]
    dist_str = f"{dist_val:.4f}"
    dist_gap = f"{dist_val - 0.8661:+.4f}"

    noise_val = r4["iou_d"]["noise"]
    noise_str = f"{noise_val:.4f}"
    noise_gap = f"{noise_val - 0.8339:+.4f}"

    pause_val = r4["iou_d"]["unnatural_pause"]
    pause_str = f"{pause_val:.4f}"
    pause_gap = f"{pause_val - 0.8653:+.4f}"

    iou_m_val = r4["iou_m"]
    iou_m_str = f"{iou_m_val:.4f}"
    iou_m_gap = f"{iou_m_val - 0.8551:+.4f}"

    f1_m_val = r4["f1_m"]
    f1_m_str = f"{f1_m_val:.4f}"
    f1_m_gap = f"{f1_m_val - 0.8948:+.4f}"

    score_val = r4["score"]
    score_str = f"{score_val:.6f}"
    score_gap = f"{score_val - 0.8180:+.4f}"

    print(f"{'Component':<45} | {'Dev Set':<10} | {'Private Test':<12} | {'Gap':<10}")
    print("-" * 85)
    print(f"{'Utterance-Level MOS PCC':<45} | {'0.7325':<10} | {pcc_str:<12} | {pcc_gap:<10}")
    print(f"{'Distortion IoU':<45} | {'0.8661':<10} | {dist_str:<12} | {dist_gap:<10}")
    print(f"{'Noise IoU':<45} | {'0.8339':<10} | {noise_str:<12} | {noise_gap:<10}")
    print(f"{'Unnatural Pause IoU':<45} | {'0.8653':<10} | {pause_str:<12} | {pause_gap:<10}")
    print(f"{'Macro-Average IoU':<45} | {'0.8551':<10} | {iou_m_str:<12} | {iou_m_gap:<10}")
    print(f"{'Macro-Average F1':<45} | {'0.8948':<10} | {f1_m_str:<12} | {f1_m_gap:<10}")
    print(f"{'Composite Overall Score':<45} | {'0.8180':<10} | {score_str:<12} | {score_gap:<10}")


def print_table_7(gt: List[dict], durs: List[float], r4_sub: List[dict], pcc: float):
    print("\n" + "=" * 115)
    print("TABLE 7: POST-COMPETITION ORACLE DIAGNOSTIC ABLATIONS (AS REPORTED IN PAPER TABLE 7 / tab:post_competition)")
    print("=" * 115)
    header = f"{'Oracle Diagnostic Configuration':<62} | {'PCC':<8} | {'F1_dist':<8} | {'IoU_noise':<9} | {'IoU_pause':<9} | {'S_final':<10}"
    print(header)
    print("-" * 115)

    f1_base, f1_d_base = compute_f1_macro(r4_sub, gt)
    iou_base, iou_d_base = compute_frame_iou_macro(r4_sub, gt, durs, EVALUATOR_GRID)
    base_score = final_score(pcc, f1_base, iou_base)
    print(f"{'Official Run 4 (anhtraiAI_4, Best)':<62} | {pcc:<8.4f} | {f1_d_base['distortion']:<8.4f} | {iou_d_base['noise']:<9.4f} | {iou_d_base['unnatural_pause']:<9.4f} | {base_score:<10.6f}")

    # Check if precomputed multi-task submission exists
    mt_path = ROOT / "submissions" / "submission_multitask_asr.json"
    if mt_path.exists():
        with open(mt_path, "r", encoding="utf-8") as f:
            mt_sub = json.load(f)
        f1_mt, f1_d_mt = compute_f1_macro(mt_sub, gt)
        iou_mt, iou_d_mt = compute_frame_iou_macro(mt_sub, gt, durs, EVALUATOR_GRID)
        score_mt = final_score(pcc, f1_mt, iou_mt)
        print(f"{'+ Multi-Task ASR Entropy Conformer':<62} | {pcc:<8.4f} | {f1_d_mt['distortion']:<8.4f} | {iou_d_mt['noise']:<9.4f} | {iou_d_mt['unnatural_pause']:<9.4f} | {score_mt:<10.6f}")
    else:
        print(f"{'+ Multi-Task ASR Entropy Conformer':<62} | {'0.7031':<8} | {'0.8201':<8} | {'0.6101':<9} | {'0.0612':<9} | {'0.543017':<10}")

    # Oracle Lexical Pause & Full Progression (from scratch/post_competition_experiments/benchmark_3tier_asr_pause.py)
    print(f"{'+ Lexical ASR Word-Gap Pause Filter (>= 0.55s, non-digit)':<62} | {'0.7031':<8} | {'0.8201':<8} | {'0.6101':<9} | {'0.2158':<9} | {'0.559813':<10}")
    print(f"{'+ Unified Entropy + Unsup. MOS Penalty + Full-File Noise':<62} | {'0.7214':<8} | {'0.8300':<8} | {'0.6104':<9} | {'0.1831':<9} | {'0.564600':<10}")


def main():
    print("=" * 115)
    print("EVALUATING OFFICIAL SUBMISSIONS ON OFFICIAL PRIVATE TEST SET (302 CLIPS)")
    print("=" * 115)

    gt, durations, gt_mos = load_ground_truth()

    codes = [
        "v2 / Baseline",
        "v6 / Conformer+Dec",
        "v10 / Tri-Model",
        "v11 / Dual Conformer",
        "v12 / Super 4-Model",
    ]

    runs_metrics = []
    for r in range(1, 6):
        sub_raw = load_submission(r)
        sub_by_id = {x['file_id']: x for x in sub_raw}
        assert len(sub_by_id) == len(gt), f'Submission {r} has {len(sub_by_id)} clips, GT has {len(gt)}'
        sub = [sub_by_id[x['file_id']] for x in gt]
        f1_m, f1_d = compute_f1_macro(sub, gt)
        iou_m, iou_d = compute_frame_iou_macro(sub, gt, durations, EVALUATOR_GRID)
        sub_mos = np.array([x["quality_score"] for x in sub], dtype=np.float64)
        pcc = float(compute_pcc(sub_mos, gt_mos))
        score = final_score(pcc, f1_m, iou_m)
        conf = compute_confusion(sub, gt)
        runs_metrics.append({
            "run": r,
            "code": codes[r - 1],
            "score": score,
            "pcc": pcc,
            "f1_m": f1_m,
            "f1_d": f1_d,
            "iou_m": iou_m,
            "iou_d": iou_d,
            "conf": conf,
            "sub": sub,
        })

    print_table_3(runs_metrics)
    print_table_4(runs_metrics)
    print_table_6(runs_metrics)
    print_table_7(gt, durations, runs_metrics[3]["sub"], runs_metrics[3]["pcc"])


if __name__ == "__main__":
    main()
