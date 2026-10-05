#!/usr/bin/env python3
"""Convert candidate / audit samples into Label Studio task import JSON.

Pre-populates Label Studio predictions so that candidate intervals appear
directly on the audio waveform for fast review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def create_label_studio_task(
    file_id: str,
    audio_url: str,
    duration: float,
    description: str = "",
    label_source: str = "dsp",
    candidate_status: str = "pending",
    initial_intervals: Dict[str, List[List[float]]] | None = None,
) -> Dict[str, Any]:
    """Create a single Label Studio task with pre-annotations (predictions)."""
    task_data = {
        "audio": audio_url,
        "file_id": file_id,
        "duration": duration,
        "description": description or "(Không có mô tả)",
        "label_source": label_source,
        "candidate_status": candidate_status,
    }

    results = []
    if initial_intervals:
        for fault_type, intervals in initial_intervals.items():
            if fault_type not in {"noise", "distortion", "unnatural_pause"}:
                continue
            for s, e in intervals:
                results.append(
                    {
                        "from_name": "fault_labels",
                        "to_name": "audio",
                        "type": "labels",
                        "value": {
                            "start": float(s),
                            "end": float(e),
                            "labels": [fault_type],
                            "channel": 0,
                        },
                    }
                )

    task: Dict[str, Any] = {"data": task_data}
    if results:
        task["predictions"] = [
            {
                "model_version": f"pre_annotation_{label_source}",
                "result": results,
            }
        ]

    return task


def main():
    parser = argparse.ArgumentParser(description="Generate Label Studio import tasks")
    parser.add_argument("--input", "-i", type=str, help="Input JSON file with candidates")
    parser.add_argument("--output", "-o", type=str, default="tasks_to_import.json", help="Output JSON path")
    args = parser.parse_args()

    sample_tasks = [
        create_label_studio_task(
            file_id="sample_001.wav",
            audio_url="/data/local-files/?d=sample_001.wav",
            duration=3.5,
            description="Continuous background hiss from 0.5s to 2.1s",
            label_source="dsp",
            candidate_status="pending",
            initial_intervals={"noise": [[0.5, 2.1]]},
        )
    ]

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample_tasks, f, indent=2, ensure_ascii=False)

    print(f"Sample import file created at: {out_path}")


if __name__ == "__main__":
    main()
