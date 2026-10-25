"""Checkpoint / artifact manifest for reproducing the five official submissions.

The repository does not ship model weights (``*.pt`` is git-ignored).  This helper records the
exact files the builders need together with SHA-256 digests, so that anybody who downloads the
released weights can verify them, and can verify a rebuilt ``submission.json`` against the
official one.

    python scripts/checkpoint_manifest.py --write     # (maintainers) regenerate checkpoints_manifest.json
    python scripts/checkpoint_manifest.py --verify    # check local files against the manifest
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checkpoints_manifest.json"

# group -> list of paths (relative to repo root)
ARTIFACTS = {
    "runs_2_to_5_fault_heads": [
        "outputs/fault_head_v5/best_conformer_fault_head.pt",   # Run 2 (Conformer 10 ep)
        "outputs/fault_head_v5/best_decoupled_fault_head.pt",   # Runs 2, 3, 5
        "outputs/fault_head_v5/best_conformer_25ep.pt",         # Runs 3, 4, 5 (Conformer Peak)
        "outputs/fault_head_v5/best_conformer_pro.pt",          # Runs 3, 5
        "outputs/fault_head_v5/best_dual_backbone_conformer.pt",  # Runs 4, 5 (2048-d Dual)
    ],
    "shared_mos_predictions": [
        "submissions/submission_v5.json",  # Tri-Ensemble MOS for all runs (identical to submission_v2.json)
    ],
    "run_1_extra": [
        "checkpoints/wavlm_utmos_nisqa_lora_trainval_best.pt",
        "outputs/cached_trainval_utmos_tuned/best_quality_head.pt",
        "outputs/fault_head_trainval/best_fault_frame_head.pt",
        "outputs/fault_head_trainval/calibrated_postprocess_config.json",
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def official_submission_digests() -> dict:
    out = {}
    for i in range(1, 6):
        zp = ROOT / "submissions" / f"anhtraiAI_{i}.zip"
        if zp.exists():
            with zipfile.ZipFile(zp) as z:
                out[f"anhtraiAI_{i}"] = hashlib.sha256(z.read("submission.json")).hexdigest()
    return out


def write_manifest() -> None:
    manifest = {"artifacts": {}, "official_submission_json_sha256": official_submission_digests()}
    for group, paths in ARTIFACTS.items():
        manifest["artifacts"][group] = []
        for rel in paths:
            p = ROOT / rel
            if not p.exists():
                print(f"[skip] missing: {rel}")
                continue
            manifest["artifacts"][group].append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST}")


def verify_manifest(group_filter: str | None = None) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    groups_to_check = {group_filter: manifest["artifacts"][group_filter]} if group_filter else manifest["artifacts"]
    for group, items in groups_to_check.items():
        for it in items:
            p = ROOT / it["path"]
            if not p.exists():
                print(f"MISSING  [{group}] {it['path']}")
                bad += 1
            elif sha256(p) != it["sha256"]:
                print(f"MISMATCH [{group}] {it['path']}")
                bad += 1
            else:
                print(f"OK       [{group}] {it['path']}")
    if group_filter is None:
        for name, digest in manifest.get("official_submission_json_sha256", {}).items():
            zp = ROOT / "submissions" / f"{name}.zip"
            if zp.exists():
                with zipfile.ZipFile(zp) as z:
                    ok = hashlib.sha256(z.read("submission.json")).hexdigest() == digest
                print(f"{'OK      ' if ok else 'MISMATCH'} [official] {name}.zip -> submission.json")
                bad += 0 if ok else 1
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write manifest from local files")
    ap.add_argument("--verify", action="store_true", help="verify local files against manifest")
    ap.add_argument("--group", type=str, default=None, help="limit verification to a specific artifact group")
    args = ap.parse_args()

    if args.write:
        write_manifest()
        return
    if args.verify:
        sys.exit(verify_manifest(args.group))
    ap.print_help()


if __name__ == "__main__":
    main()
