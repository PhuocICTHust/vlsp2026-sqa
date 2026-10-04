"""
Chạy toàn bộ Milestone A (mục 3.3 kế hoạch): audio -> resample -> WavLM ->
quality head + (stub) fault head -> post-processing -> JSON -> validator ->
evaluator -> lưu artifact. In ra bảng PASS/FAIL theo đúng exit criteria.

Chạy (sau khi đã `python scripts/make_acceptance_set.py`):
    python scripts/run_milestone_a.py --data data/acceptance_set --out runs/milestone_a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqa.evaluator import compute_pcc
from sqa.frame_grid import MODEL_GRID
from sqa.model import SQAModel
from sqa.pipeline import run_on_folder
from sqa.validator import validate_submission
from sqa.postprocess import BASELINE_POSTPROCESS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/acceptance_set")
    parser.add_argument("--out", type=str, default="runs/milestone_a")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checklist = {}

    # ---- 1. Audio loading + feature extraction + heads, hết bộ nghiệm thu ----
    print(f"[1/5] Load model (WavLM-large) — lần đầu sẽ tải checkpoint, có thể mất vài phút...")
    model = SQAModel()
    print(f"[2/5] Chạy pipeline trên {args.data} ...")
    try:
        entries, duration_lookup = run_on_folder(args.data, model)
        checklist["audio_loading_and_inference"] = f"PASS ({len(entries)} file)"
    except Exception as exc:  # noqa: BLE001 - muốn thấy lỗi cụ thể ngay, không nuốt exception
        checklist["audio_loading_and_inference"] = f"FAIL: {exc!r}"
        _print_report(checklist)
        raise

    # ---- 2. Lưu JSON thô (chưa validate) để debug nếu cần ----
    raw_path = out_dir / "predictions_raw.json"
    raw_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    cfg_path= out_dir / "postprocess_config.json"
    cfg_path.write_text(json.dumps(BASELINE_POSTPROCESS.to_dict(), indent=2))
    checklist["postprocess_is_baseline"] = "PASS" if BASELINE_POSTPROCESS.is_baseline else "FAIL"

    print(f"[3/5] Đã lưu {raw_path}")

    # ---- 3. Validator: chạy strict=False để thấy HẾT lỗi 1 lần, không chỉ lỗi đầu ----
    print("[4/5] Chạy validator...")
    errors = validate_submission(entries, duration_lookup, strict=False)
    if errors:
        checklist["validator"] = f"FAIL ({len(errors)} lỗi)"
        for e in errors[:20]:
            print("   ", e)
    else:
        checklist["validator"] = "PASS"

    # ---- 4. Case validator riêng: prediction rỗng hoàn toàn + score ngoài range ----
    from sqa.validator import validate_submission_entry

    empty_entry = {
        "file_id": next(iter(duration_lookup)),
        "quality_score": 3.0,
        "faults": {"noise": [], "distortion": [], "unnatural_pause": []},
    }
    checklist["validator_accepts_empty_prediction"] = (
        "PASS" if validate_submission_entry(empty_entry, duration_lookup) else "FAIL"
    )

    bad_entry = dict(empty_entry)
    bad_entry["quality_score"] = 7.0  # cố tình ngoài [1,5]
    errs = []
    ok = validate_submission_entry(bad_entry, duration_lookup, collect_errors=errs)
    checklist["validator_rejects_out_of_range_score"] = "PASS" if not ok and errs else "FAIL"

    # ---- 5. Evaluator chạy được (dùng chính prediction làm "GT giả" chỉ để
    #          xác nhận evaluator không crash — KHÔNG phải đánh giá chất lượng
    #          model, vì fault head hiện tại là StubFaultHead chưa train) ----
    print("[5/5] Chạy thử evaluator (self-comparison, chỉ kiểm tra evaluator không crash)...")
    try:
        pcc = compute_pcc([e["quality_score"] for e in entries], [e["quality_score"] for e in entries])
        checklist["evaluator_runs"] = f"PASS (self-PCC={pcc:.3f}, kỳ vọng ~1.0)"
    except Exception as exc:  # noqa: BLE001
        checklist["evaluator_runs"] = f"FAIL: {exc!r}"

    checklist["saved_artifacts"] = f"PASS ({raw_path})"

    _print_report(checklist)


def _print_report(checklist: dict) -> None:
    print("\n=== MILESTONE A — báo cáo nghiệm thu (mục 3.3 kế hoạch) ===")
    for k, v in checklist.items():
        print(f"  [{'OK' if str(v).startswith('PASS') else 'XX'}] {k}: {v}")
    if all(str(v).startswith("PASS") for v in checklist.values()):
        print("\n>>> MILESTONE A: PASS — có thể báo cáo với team và bắt đầu Milestone B.")
    else:
        print("\n>>> MILESTONE A: CHƯA PASS — xem các dòng FAIL ở trên.")


if __name__ == "__main__":
    main()
