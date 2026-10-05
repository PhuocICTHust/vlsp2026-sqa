#!/usr/bin/env python3
"""Parse exported annotations from Label Studio into Kiên's DB update format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.sqa.postprocess import normalize_intervals
except ImportError:
    def normalize_intervals(intervals, duration, merge_gap=0.0, min_duration=0.0, round_digits=None):
        iv = [[float(s), float(e)] for s, e in intervals if float(s) < float(e)]
        iv = [[max(0.0, s), min(duration, e)] for s, e in iv if max(0.0, s) < min(duration, e)]
        iv.sort(key=lambda x: x[0])
        merged = []
        for s, e in iv:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged


def parse_label_studio_export(export_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output_records = []

    for task in export_data:
        data = task.get("data", {})
        file_id = data.get("file_id")
        duration = float(data.get("duration", 0.0))

        annotations = task.get("annotations", [])
        if not annotations:
            continue

        latest_annot = annotations[-1]
        results = latest_annot.get("result", [])

        fault_intervals: Dict[str, List[List[float]]] = {
            "noise": [],
            "distortion": [],
            "unnatural_pause": [],
        }

        fault_status: Dict[str, str] = {
            "noise": "unknown",
            "distortion": "unknown",
            "unnatural_pause": "unknown",
        }

        dsp_decision = "not_applicable"
        audit_notes = ""

        for res in results:
            from_name = res.get("from_name")
            val = res.get("value", {})

            if from_name == "fault_labels":
                labels = val.get("labels", [])
                start = val.get("start", 0.0)
                end = val.get("end", 0.0)
                for lbl in labels:
                    if lbl in fault_intervals:
                        fault_intervals[lbl].append([start, end])

            elif from_name == "status_noise":
                choices = val.get("choices", [])
                if choices:
                    fault_status["noise"] = choices[0]

            elif from_name == "status_distortion":
                choices = val.get("choices", [])
                if choices:
                    fault_status["distortion"] = choices[0]

            elif from_name == "status_unnatural_pause":
                choices = val.get("choices", [])
                if choices:
                    fault_status["unnatural_pause"] = choices[0]

            elif from_name == "dsp_decision":
                choices = val.get("choices", [])
                if choices:
                    dsp_decision = choices[0]

            elif from_name == "audit_notes":
                text = val.get("text", [])
                if text:
                    audit_notes = "\n".join(text)

        normalized_faults = {}
        for ftype, ivs in fault_intervals.items():
            norm_ivs = normalize_intervals(ivs, duration=duration, merge_gap=0.0)
            normalized_faults[ftype] = norm_ivs
            if norm_ivs and fault_status[ftype] == "unknown":
                fault_status[ftype] = "localized"

        output_records.append(
            {
                "file_id": file_id,
                "duration": duration,
                "fault_status": fault_status,
                "faults": normalized_faults,
                "dsp_decision": dsp_decision,
                "audit_notes": audit_notes,
                "resolved_by": "manual_correction:Hieu",
                "annotation_id": latest_annot.get("id"),
                "updated_at": latest_annot.get("updated_at"),
            }
        )

    return output_records


def main():
    parser = argparse.ArgumentParser(description="Parse Label Studio exported annotations")
    parser.add_argument("--input", "-i", required=True, help="Path to exported JSON from Label Studio")
    parser.add_argument("--output", "-o", default="reviewed_labels.json", help="Path to output cleaned JSON for Kiên")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        export_data = json.load(f)

    cleaned = parse_label_studio_export(export_data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(cleaned)} reviewed tasks. Output saved to {args.output}")


if __name__ == "__main__":
    main()
