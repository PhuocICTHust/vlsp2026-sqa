"""
Package final submission.json into <TeamName>_<Attempt>.zip for VLSP 2026.
"""
import argparse
import json
import zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Package submission into zip file.")
    parser.add_argument("--team-name", type=str, required=True, help="Tên đội thi (e.g. ICTHust)")
    parser.add_argument("--attempt", type=int, default=1, help="Lần nộp thứ mấy (mặc định là 1)")
    parser.add_argument("--submission-json", type=str, default="submissions/submission.json", help="Path to submission.json")
    parser.add_argument("--out-dir", type=str, default="submissions", help="Folder to save zip file")
    args = parser.parse_args()

    json_path = Path(args.submission_json)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Validated JSON: {len(data)} entries found in {json_path.name}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{args.team_name}_{args.attempt}.zip"
    zip_path = out_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")

    print(f"Successfully packaged: {zip_path.resolve()} ({zip_path.stat().st_size / 1024:.2f} KB)")
    print(f"Contains: {[info.filename for info in zipfile.ZipFile(zip_path).filelist]}")

if __name__ == "__main__":
    main()
