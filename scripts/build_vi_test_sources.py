#!/usr/bin/env python3
"""
Build the 125 CLEAN source utterances for the internal Vietnamese test set `vi_test_v1`
(fast-track plan, step 1): ~100 human + ~25 TTS, 16 kHz mono PCM16, 2.5-10 s, clean,
from speakers / voices that are NOT in the 1k train set.

Speaker independence is enforced automatically from the train metadata
(data/vietnamese_clean metadata.csv: columns source, speaker, orig_path):
  * VIVOS  : train uses VIVOSDEV05-12 and VIVOSSPK13-39 -> only the other speakers are used
  * Common Voice : train uses 63 client_ids (speaker = "cv_" + client_id[:12]) -> excluded
  * TTS    : train uses edge-tts HoaiMy + NamMinh -> edge-tts is NOT used; test TTS comes from systems
             not seen in train: FPT.AI (6 voices, north/central/south; needs an API key), VieNeu-TTS presets
             (pip install vieneu, offline CPU), gTTS (Google, online, 1 voice), Piper vi_VN-vais1000 and
             vi_VN-25hours_single (offline), Meta MMS-TTS vie (offline, needs torch + transformers)

Sources (give the ones you have; at least one human source is required):
  --vivos-dir  root of VIVOS (contains train/ and test/ with waves/<SPK>/*.wav and prompts.txt)
  --cv-dir     Common Voice vi folder (contains validated.tsv and clips/)
  --fleurs     download Google FLEURS vi_vn from Hugging Face (new speakers, mixed accents)

Usage:
    pip install numpy scipy soundfile gTTS piper-tts    # MMS uses torch + transformers if installed
    python scripts/build_vi_test_sources.py --train-metadata data/vietnamese_audio_clean/metadata.csv \
        --vivos-dir /content/vivos --cv-dir /content/cv_vi/cv-corpus-27.0-2026-09-11/vi \
        --out-dir data/vietnamese_clean_v3/test_sources

Two-step run (VIVOS not downloaded yet): run CV + TTS now with --n-human <what CV can give>,
then when VIVOS is ready rerun with --vivos-dir ... --n-human 100 --resume: the existing files are kept,
numbering continues, and only the missing human slots are filled (no duplicate clips / sentences).

Outputs in --out-dir: vi_test_src_h_XXX.wav, tts_vi_test_src_t_XXX.wav,
vi_test_v1_sources.csv, selection_report.md
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment_engine import SR, db, frame_rms, load_audio, write_wav_pcm16  # noqa: E402

FLEURS_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/vi_vn"
PIPER_VOICES = ["vi_VN-vais1000-medium", "vi_VN-25hours_single-low"]   # NOT vi_VN-vivos (VIVOS speakers)


def load_train_exclusion(path: str | None) -> dict:
    ex = {"speakers": set(), "paths": set(), "texts": set(), "tts_voices": set()}
    if not path:
        print("[warn] no --train-metadata: speaker independence is NOT checked")
        return ex
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        spk = r.get("speaker", "")
        ex["speakers"].add(spk)
        ex["paths"].add(Path(r.get("orig_path", "")).name)
        ex["texts"].add(norm_text(r.get("transcript", "")))
        if r.get("source", "").startswith("tts"):
            ex["tts_voices"].add(spk.split("_")[-1])          # e.g. vi-VN-HoaiMyNeural
    print(f"[train] {len(ex['speakers'])} speakers, {len(ex['paths'])} files excluded; "
          f"TTS voices in train: {sorted(ex['tts_voices'])}")
    return ex


def norm_text(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def vivos_candidates(root: Path) -> list[dict]:
    rows = []
    for split in ("train", "test"):
        d = root / split
        if not d.exists():
            continue
        prompts = {}
        pf = d / "prompts.txt"
        if pf.exists():
            for line in pf.read_text(encoding="utf-8").splitlines():
                k, _, txt = line.partition(" ")
                prompts[k] = txt
        genders = {}
        gf = d / "genders.txt"
        if gf.exists():
            for line in gf.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    genders[parts[0]] = {"m": "male", "f": "female"}.get(parts[1].lower(), "unknown")
        for w in sorted((d / "waves").rglob("*.wav")):
            spk = w.parent.name
            rows.append({"path": w, "source": "vivos", "speaker": f"vivos_{spk}", "gender": genders.get(spk, "unknown"),
                         "region": "south", "transcript": prompts.get(w.stem, ""), "origin_id": w.stem})
    print(f"[vivos] {len(rows)} files found")
    return rows


def cv_candidates(root: Path, min_votes: int = 2) -> list[dict]:
    tsv = root / "validated.tsv"
    if not tsv.exists():
        print(f"[warn] {tsv} not found")
        return []
    rows = []
    # CV sentences contain raw quote characters: read the TSV without quote handling
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    for r in csv.DictReader(open(tsv, encoding="utf-8", newline=""), delimiter="\t", quoting=csv.QUOTE_NONE):
        try:
            up, down = int(r.get("up_votes") or 0), int(r.get("down_votes") or 0)
        except ValueError:
            continue
        if up < min_votes or down > 0:                                   # only well-validated clips
            continue
        g = (r.get("gender") or "").lower()
        g = "male" if g.startswith("male") else "female" if g.startswith("female") else "unknown"
        acc = (r.get("accents") or r.get("accent") or "").lower()
        region = "north" if ("bắc" in acc or "ha noi" in acc or "hà nội" in acc or "north" in acc) else \
                 "south" if ("nam" in acc or "sai gon" in acc or "south" in acc) else \
                 "central" if ("trung" in acc or "central" in acc or "huế" in acc) else "unknown"
        rows.append({"path": root / "clips" / r["path"], "source": "common_voice",
                     "speaker": "cv_" + r["client_id"][:12], "gender": g, "region": region,
                     "transcript": r.get("sentence", ""), "origin_id": Path(r["path"]).stem})
    print(f"[cv] {len(rows)} validated clips (up>={min_votes}, down=0)")
    return rows


def http_get(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def fleurs_candidates(cache: Path) -> list[dict]:
    rows = []
    for split in ("test", "dev"):
        try:
            tsv = http_get(f"{FLEURS_BASE}/{split}.tsv", cache / f"{split}.tsv")
            tar = http_get(f"{FLEURS_BASE}/audio/{split}.tar.gz", cache / f"{split}.tar.gz")
        except Exception as ex:
            print(f"[warn] FLEURS {split}: {ex}")
            continue
        out = cache / split
        if not out.exists():
            with tarfile.open(tar) as t:
                t.extractall(out)
        index = {p.name: p for p in out.rglob("*.wav")}
        for line in tsv.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 7 and parts[1] in index:
                rows.append({"path": index[parts[1]], "source": "fleurs", "speaker": f"fleurs_{split}_{Path(parts[1]).stem}",
                             "gender": parts[6].lower() if parts[6].lower() in ("male", "female") else "unknown",
                             "region": "unknown", "transcript": parts[2], "origin_id": f"fleurs_sent_{parts[0]}"})
    print(f"[fleurs] {len(rows)} files")
    return rows


def trim_silence(x: np.ndarray, pad_s: float = 0.15) -> np.ndarray:
    r = db(frame_rms(x, 0.01))
    act = np.flatnonzero(r > np.percentile(r, 95) - 35)
    if len(act) == 0:
        return x
    return x[max(0, act[0] * 160 - int(pad_s * SR)): min(len(x), (act[-1] + 1) * 160 + int(pad_s * SR))]


def check_clean(x: np.ndarray, min_snr: float, dmin: float, dmax: float) -> tuple[bool, str, dict]:
    dur = len(x) / SR
    info = {"duration": round(dur, 3)}
    if not (dmin <= dur <= dmax):
        return False, "duration", info
    if not np.all(np.isfinite(x)) or np.max(np.abs(x)) < 1e-4:
        return False, "nan_or_silent", info
    if np.mean(np.abs(x) >= 0.999) > 1e-4:
        return False, "clipped", info
    r = db(frame_rms(x, 0.02))
    snr = float(np.percentile(r, 95) - np.percentile(r, 10))
    info["est_snr_db"] = round(snr, 1)
    if snr < min_snr:
        return False, "noisy", info
    edge, peak = int(0.03 * SR), float(np.percentile(r, 95))
    if db(np.sqrt(np.mean(x[:edge] ** 2))) > peak - 15 or db(np.sqrt(np.mean(x[-edge:] ** 2))) > peak - 15:
        return False, "cut_at_boundary", info
    return True, "ok", info


def normalise(x: np.ndarray, target_rms: float = 0.07) -> np.ndarray:
    r = frame_rms(x, 0.02)
    act = r[db(r) > np.percentile(db(r), 95) - 35]
    y = x * (target_rms / (np.sqrt(np.mean(act ** 2)) + 1e-9))
    pk = float(np.max(np.abs(y)))
    return (y * (0.95 / pk) if pk > 0.95 else y).astype(np.float32)


DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])        # non-personalised mapping, dnsmos_local.py
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
_DNSMOS: dict = {}


def dnsmos_load(model_path: Path) -> None:
    import onnxruntime as ort
    if not model_path.exists():
        http_get(DNSMOS_URL, model_path)
    _DNSMOS["s"] = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    _DNSMOS["in"] = _DNSMOS["s"].get_inputs()[0].name


def dnsmos_score(x: np.ndarray) -> tuple[float, float, float]:
    """DNSMOS P.835 (SIG, BAK, OVRL) at 16 kHz: 9.01 s windows, 1 s hop, averaged (as in dnsmos_local.py)."""
    L = int(9.01 * SR)
    a = np.asarray(x, dtype=np.float32)
    while len(a) < L:
        a = np.concatenate([a, a])
    out = []
    for i in range(int(np.floor(len(a) / SR - 9.01)) + 1):
        seg = a[i * SR: i * SR + L]
        if len(seg) < L:
            continue
        sig, bak, ovr = _DNSMOS["s"].run(None, {_DNSMOS["in"]: seg[None, :]})[0][0]
        out.append((_P_SIG(sig), _P_BAK(bak), _P_OVR(ovr)))
    m = np.mean(out, axis=0)
    return round(float(m[0]), 2), round(float(m[1]), 2), round(float(m[2]), 2)


def decode_file(p: Path) -> np.ndarray:
    try:
        return load_audio(p)[0]
    except Exception:
        wav = p.with_suffix(".dec.wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(p), "-ar", str(SR), "-ac", "1", str(wav)], check=True)
        return load_audio(wav)[0]


def tts_gtts(text: str, tmp: Path) -> np.ndarray:
    from gtts import gTTS
    f = tmp / "g.mp3"
    gTTS(text=text, lang="vi").save(str(f))
    return decode_file(f)


_VIENEU: dict = {}


def vieneu_voices(max_voices: int) -> list[str]:
    """VieNeu-TTS (pip install vieneu; Apache-2.0; CPU via ONNX). Returns preset voice ids."""
    from vieneu import Vieneu
    if "tts" not in _VIENEU:
        _VIENEU["tts"] = Vieneu()
    pres = list(_VIENEU["tts"].list_preset_voices())
    _VIENEU["labels"] = {vid: lab for lab, vid in pres}
    print(f"[vieneu] {len(pres)} preset voices")
    # labels look like "Hải Đăng — Nam · Bắc · Phong cách tự nhiên": balance region x gender, skip joke voices
    groups = defaultdict(list)
    for lab, vid in pres:
        if "bựa" in lab.lower():
            continue
        groups[vieneu_meta(vid)].append(vid)
    keys = sorted(groups, key=lambda k: -len(groups[k]))
    chosen = []
    while len(chosen) < max_voices and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(chosen) < max_voices:
                chosen.append(groups[k].pop(0))
    print(f"[vieneu] using {len(chosen)}: {[(_VIENEU['labels'][v].split('·')[0].strip(' ⭐'), *vieneu_meta(v)) for v in chosen]}")
    return chosen


def vieneu_meta(vid: str) -> tuple[str, str]:
    """(gender, region) from a VieNeu label 'Name — Nam/Nữ · Bắc/Trung/Nam · style'."""
    parts = [p.strip().lower() for p in re.split(r"[—·]", _VIENEU.get("labels", {}).get(vid, ""))]
    g = {"nam": "male", "nữ": "female"}.get(parts[1], "unknown") if len(parts) > 2 else "unknown"
    r = {"bắc": "north", "trung": "central", "nam": "south"}.get(parts[2], "unknown") if len(parts) > 2 else "unknown"
    return g, r


def tts_vieneu(text: str, voice: str, tmp: Path) -> np.ndarray:
    f = tmp / "vieneu.wav"
    tts = _VIENEU["tts"]
    tts.save(tts.infer(text, voice=voice), str(f))
    return decode_file(f)                   # 48/24 kHz -> 16 kHz mono


_PIPER: dict = {}


def tts_piper(text: str, voice: str, tmp: Path, voices_dir: Path) -> np.ndarray:
    """Piper (offline neural TTS). Voice files are downloaded once from rhasspy/piper-voices."""
    import wave
    from piper import PiperVoice
    model = voices_dir / f"{voice}.onnx"
    if not model.exists():
        lang, rest = voice.split("-", 1)
        name, quality = rest.rsplit("-", 1)
        base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/{lang}/{name}/{quality}/{voice}"
        http_get(base + ".onnx", model)
        http_get(base + ".onnx.json", voices_dir / f"{voice}.onnx.json")
    if voice not in _PIPER:
        _PIPER[voice] = PiperVoice.load(str(model))
    v = _PIPER[voice]
    f = tmp / "p.wav"
    with wave.open(str(f), "wb") as w:
        if hasattr(v, "synthesize_wav"):          # piper-tts >= 1.3
            v.synthesize_wav(text, w)
        else:                                     # piper-tts 1.2
            v.synthesize(text, w)
    return decode_file(f)


FPT_VOICES = {"banmai": ("female", "north"), "leminh": ("male", "north"), "myan": ("female", "central"),
              "giahuy": ("male", "central"), "linhsan": ("female", "south"), "minhquang": ("male", "south")}


FPT_MIN_INTERVAL = 4.0     # seconds between FPT requests (raise with --fpt-interval if you still get 429)
_FPT_LAST = [0.0]


def tts_fpt(text: str, voice: str, tmp: Path, api_key: str) -> np.ndarray:
    """FPT.AI TTS v5. The API answers with an 'async' URL; the mp3 appears there after a few seconds."""
    import json
    import time
    resp = None
    for attempt in range(4):                  # free keys are rate-limited: throttle + back off on 429
        wait = FPT_MIN_INTERVAL - (time.time() - _FPT_LAST[0])
        if wait > 0:
            time.sleep(wait)
        _FPT_LAST[0] = time.time()
        req = urllib.request.Request("https://api.fpt.ai/hmi/tts/v5", data=text.encode("utf-8"), method="POST",
                                     headers={"api-key": api_key, "voice": voice, "speed": "0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 3:
                raise
            ra = e.headers.get("Retry-After") if e.headers else None
            pause = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 10.0 * 2 ** attempt
            print(f"[fpt] 429 rate limit ({voice}), waiting {pause:.0f} s")
            time.sleep(min(pause, 120.0))
    url = resp.get("async")
    if not url or resp.get("error"):
        raise RuntimeError(f"FPT error: {resp}")
    f = tmp / "fpt.mp3"
    for _ in range(40):                       # poll up to ~40 s
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            if len(data) > 1000:
                f.write_bytes(data)
                return decode_file(f)
        except urllib.error.HTTPError:
            continue
    raise TimeoutError("FPT audio not ready after 40 s")


_MMS: dict = {}


def tts_mms(text: str) -> np.ndarray:
    """Meta MMS-TTS Vietnamese (facebook/mms-tts-vie, VITS; CC-BY-NC 4.0, research use)."""
    import torch
    from scipy import signal as sps
    from transformers import AutoTokenizer, VitsModel
    if not _MMS:
        _MMS["tok"] = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
        _MMS["model"] = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    with torch.no_grad():
        y = _MMS["model"](**_MMS["tok"](text.lower(), return_tensors="pt")).waveform[0].numpy()
    sr = _MMS["model"].config.sampling_rate
    if sr != SR:
        y = sps.resample_poly(y, SR, sr)
    return y.astype(np.float32)


def main() -> None:
    global FPT_MIN_INTERVAL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-metadata", required=True, help="metadata.csv of the 1k train clean set")
    ap.add_argument("--vivos-dir")
    ap.add_argument("--cv-dir")
    ap.add_argument("--fleurs", action="store_true", help="also download FLEURS vi_vn (Hugging Face)")
    ap.add_argument("--cache-dir", default="data/_cache_sources")
    ap.add_argument("--out-dir", default="data/vietnamese_clean_v3/test_sources")
    ap.add_argument("--n-human", type=int, default=100)
    ap.add_argument("--n-tts", type=int, default=25)
    ap.add_argument("--max-per-speaker", type=int, default=5)
    ap.add_argument("--tts-engines", nargs="*", default=["fpt", "vieneu", "piper", "gtts"],
                    help="fpt (API key, 6 voices 3 regions), vieneu (offline CPU, preset voices), gtts (online, 1 voice), "
                         "piper (offline, 2 voices), "
                         "mms (offline, transformers); edge is refused")
    ap.add_argument("--fpt-key", default=os.environ.get("FPT_TTS_API_KEY"), help="FPT.AI API key (or env FPT_TTS_API_KEY)")
    ap.add_argument("--fpt-voices", nargs="*", default=list(FPT_VOICES))
    ap.add_argument("--vieneu-max-voices", type=int, default=12, help="max VieNeu preset voices used (balanced region x gender)")
    ap.add_argument("--fpt-interval", type=float, default=FPT_MIN_INTERVAL, help="seconds between FPT requests")
    ap.add_argument("--min-snr", type=float, default=30.0)
    ap.add_argument("--min-dur", type=float, default=2.5)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dnsmos", action="store_true",
                    help="also filter human clips with DNSMOS P.835 (onnxruntime; model auto-downloaded from GitHub)")
    ap.add_argument("--dnsmos-model", default=None, help="path to sig_bak_ovr.onnx if the download is blocked")
    ap.add_argument("--min-sig", type=float, default=3.5)
    ap.add_argument("--min-bak", type=float, default=4.0)
    ap.add_argument("--min-ovrl", type=float, default=3.2)
    ap.add_argument("--min-votes", type=int, default=2, help="Common Voice: minimum up_votes")
    ap.add_argument("--review-dir", default=None,
                    help="listening review: write NEW human candidates to this folder (+ review.csv) instead of the "
                         "test set; the test set is not touched. Delete the bad wavs, then run --merge-review")
    ap.add_argument("--review-n", type=int, default=0, help="number of candidates to stage (default 2x the missing slots)")
    ap.add_argument("--merge-review", default=None,
                    help="move the wavs still present in this review folder into the test set (fills missing human slots)")
    ap.add_argument("--redo-human", action="store_true",
                    help="with --resume: keep the TTS files but delete and re-select all human files")
    ap.add_argument("--redo-tts", action="store_true",
                    help="with --resume: keep the human files but delete and regenerate all TTS files")
    ap.add_argument("--resume", action="store_true",
                    help="keep files already listed in out-dir/vi_test_v1_sources.csv and only fill the missing "
                         "human / TTS slots (e.g. run CV + TTS first, add VIVOS later)")
    args = ap.parse_args()
    FPT_MIN_INTERVAL = args.fpt_interval

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex = load_train_exclusion(args.train_metadata)

    csv_path = out / "vi_test_v1_sources.csv"
    prev = []
    if args.resume and csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            prev = [r for r in csv.DictReader(f) if (out / r["file"]).exists()]
        for r in prev:
            r["duration"], r["est_snr_db"] = float(r["duration"]), float(r["est_snr_db"])
            for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"):
                r[k] = float(r[k]) if r.get(k) else ""
        print(f"[resume] keeping {len(prev)} existing files from {csv_path}")
    elif csv_path.exists():
        print(f"[warn] {csv_path} exists and will be overwritten (use --resume to keep it)")
    prev_h = [r for r in prev if r["source"] == "human"]
    prev_t = [r for r in prev if r["source"] == "tts"]
    if args.redo_human and prev_h:
        for r in prev_h:
            (out / r["file"]).unlink(missing_ok=True)
        print(f"[resume] --redo-human: removed {len(prev_h)} old human files")
        prev = prev_t
        prev_h = []
    if args.redo_tts and prev_t:
        for r in prev_t:
            (out / r["file"]).unlink(missing_ok=True)
        print(f"[resume] --redo-tts: removed {len(prev_t)} old TTS files")
        prev = prev_h
        prev_t = []
    need_h = max(0, args.n_human - len(prev_h))
    need_t = max(0, args.n_tts - len(prev_t))
    staging = bool(args.review_dir)
    merging = bool(args.merge_review)
    if staging and merging:
        sys.exit("[error] use --review-dir and --merge-review in two separate runs")
    if staging:
        hum_dir = Path(args.review_dir)
        hum_dir.mkdir(parents=True, exist_ok=True)
        for old in hum_dir.glob("cand_*.wav"):
            old.unlink()
        n_loop = args.review_n or 2 * need_h
        print(f"[review] staging {n_loop} candidates in {hum_dir} (test set unchanged)")
    else:
        hum_dir, n_loop = out, (0 if merging else need_h)
    prev_ids = {r["origin_id"] for r in prev_h}
    prev_txt = {norm_text(r["transcript"]) for r in prev if r["transcript"]}

    cands = []
    if merging:
        pass
    elif args.vivos_dir:
        cands += vivos_candidates(Path(args.vivos_dir))
    if args.cv_dir and not merging:
        cands += cv_candidates(Path(args.cv_dir), args.min_votes)
    if args.fleurs and not merging:
        cands += fleurs_candidates(Path(args.cache_dir) / "fleurs")
    if not cands and n_loop > 0:
        sys.exit("[error] no human source: give --vivos-dir and/or --cv-dir and/or --fleurs")

    before = len(cands)
    cands = [c for c in cands if c["speaker"] not in ex["speakers"] and c["path"].name not in ex["paths"]
             and norm_text(c["transcript"]) not in ex["texts"]
             and c["origin_id"] not in prev_ids and norm_text(c["transcript"]) not in prev_txt]
    print(f"[human] {before} -> {len(cands)} candidates after removing train speakers / files / sentences")
    spk_free = sorted({c["speaker"] for c in cands})
    print(f"[human] {len(spk_free)} new speakers available, by source: "
          f"{dict(Counter(c['source'] for c in {c['speaker']: c for c in cands}.values()))}")

    # round-robin over sources and speakers -> many speakers, balanced sources, <= max-per-speaker
    by_src = defaultdict(lambda: defaultdict(list))
    for c in cands:
        by_src[c["source"]][c["speaker"]].append(c)
    for s in by_src.values():
        for lst in s.values():
            rng.shuffle(lst)
    order = []
    srcs = sorted(by_src)
    spk_lists = {s: sorted(by_src[s]) for s in srcs}
    for s in srcs:
        rng.shuffle(spk_lists[s])
    rnd = 0
    while any(by_src[s][k] for s in srcs for k in spk_lists[s]) and rnd < 50:
        for s in srcs:
            for k in spk_lists[s]:
                if by_src[s][k]:
                    order.append(by_src[s][k].pop())
        rnd += 1

    if args.dnsmos and n_loop > 0:
        dnsmos_load(Path(args.dnsmos_model) if args.dnsmos_model else Path(args.cache_dir) / "dnsmos" / "sig_bak_ovr.onnx")
        print(f"[dnsmos] filter on: SIG>={args.min_sig} BAK>={args.min_bak} OVRL>={args.min_ovrl}")
    picked, reasons = [], Counter()
    per_spk = Counter(r["speaker_id"] for r in prev_h)
    gcount = Counter(r["gender"] for r in prev_h)
    used_txt = set(prev_txt)
    half = args.n_human // 2
    for c in order:
        if len(picked) >= n_loop:
            break
        if per_spk[c["speaker"]] >= args.max_per_speaker or (c["transcript"] and norm_text(c["transcript"]) in used_txt):
            continue
        if c["gender"] in ("male", "female") and gcount[c["gender"]] >= half + 5 and len(prev_h) + len(picked) < args.n_human - 5 \
                and not staging:
            continue
        try:
            x = trim_silence(load_audio(c["path"])[0])
        except Exception:
            reasons["unreadable"] += 1
            continue
        ok, why, info = check_clean(x, args.min_snr, args.min_dur, args.max_dur)
        if not ok:
            reasons[why] += 1
            continue
        if args.dnsmos:
            sig, bak, ovr = dnsmos_score(normalise(x))
            info.update(dnsmos_sig=sig, dnsmos_bak=bak, dnsmos_ovrl=ovr)
            if sig < args.min_sig or bak < args.min_bak or ovr < args.min_ovrl:
                reasons["dnsmos_low"] += 1
                continue
        fn = (f"cand_{len(picked) + 1:03d}.wav" if staging else f"vi_test_src_h_{len(prev_h) + len(picked) + 1:03d}.wav")
        write_wav_pcm16(hum_dir / fn, normalise(x))
        picked.append({"file": fn, "speaker_id": c["speaker"], "gender": c["gender"], "region": c["region"],
                       "source": "human", "origin": c["source"], "origin_id": c["origin_id"],
                       "transcript": c["transcript"], **info})
        per_spk[c["speaker"]] += 1
        gcount[c["gender"]] += 1
        used_txt.add(norm_text(c["transcript"]))
    print(f"[human] picked {len(picked)} new (+{len(prev_h)} kept) / {args.n_human} from {len(per_spk)} speakers; "
          f"rejected {dict(reasons)}")

    rkeys = ["file", "speaker_id", "gender", "region", "source", "origin", "origin_id", "transcript", "duration",
             "est_snr_db", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"]
    if staging:
        with open(hum_dir / "review.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rkeys, extrasaction="ignore")
            w.writeheader()
            w.writerows(picked)
        print(f"[review] {len(picked)} candidates written to {hum_dir} (review.csv lists them).\n"
              f"[review] listen, DELETE the bad cand_*.wav files, then run the same command with "
              f"--merge-review {hum_dir} instead of --review-dir (needs {need_h} good files).")
        return
    if merging:
        rd = Path(args.merge_review)
        with open(rd / "review.csv", encoding="utf-8") as f:
            staged = list(csv.DictReader(f))
        kept = [r for r in staged if (rd / r["file"]).exists()]
        kept = [r for r in kept if r["speaker_id"] not in ex["speakers"] and r["origin_id"] not in prev_ids
                and norm_text(r["transcript"]) not in prev_txt]
        print(f"[merge] {len(kept)}/{len(staged)} reviewed files still present; filling {need_h} missing human slots")
        if len(kept) < need_h:
            print(f"[merge] WARNING: only {len(kept)} good files for {need_h} slots - stage more with --review-dir")
        import shutil
        for r in kept[:need_h]:
            fn = f"vi_test_src_h_{len(prev_h) + len(picked) + 1:03d}.wav"
            shutil.copy2(rd / r["file"], out / fn)
            for k in ("duration", "est_snr_db", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"):
                r[k] = float(r[k]) if r.get(k) else ""
            picked.append({**r, "file": fn, "review_file": r["file"]})
            per_spk[r["speaker_id"]] += 1
        reasons["removed_in_listening_review"] = len(staged) - len([r for r in staged if (rd / r["file"]).exists()])
        print(f"[merge] added {len(picked)} files to {out}")

    # TTS: texts = unused candidate sentences (not in train, not in human part); engines NOT in train
    tts_rows, fails = [], Counter()
    if need_t and args.tts_engines:
        texts = [c["transcript"] for c in order if c["transcript"] and norm_text(c["transcript"]) not in used_txt
                 and 8 <= len(c["transcript"].split()) <= 30]
        texts = [t.lower().capitalize() if t.isupper() else t for t in texts]   # VIVOS prompts are ALL CAPS
        texts = list(dict.fromkeys(texts))
        rng.shuffle(texts)
        jobs = []
        for e in args.tts_engines:
            if e == "piper":
                jobs += [("piper", v) for v in PIPER_VOICES]
            elif e == "gtts":
                jobs.append(("gtts", "gtts-vi"))
            elif e == "mms":
                jobs.append(("mms", "mms-tts-vie"))
            elif e == "vieneu":
                try:
                    jobs += [("vieneu", v) for v in vieneu_voices(args.vieneu_max_voices)]
                except Exception as ex:
                    print(f"[skip] vieneu: {type(ex).__name__}: {ex} (pip install vieneu)")
            elif e == "fpt":
                if args.fpt_key:
                    jobs += [("fpt", v) for v in args.fpt_voices]
                else:
                    print("[skip] fpt: no API key (--fpt-key or env FPT_TTS_API_KEY)")
            elif e == "edge":
                print("[skip] edge-tts voices are already in the train set")
        used_voice = Counter(r["origin_id"] for r in prev_t)
        jobs.sort(key=lambda j: used_voice[j[1]])      # top-up runs start with the least-used voices
        tmp = Path(tempfile.mkdtemp())
        dead, i, ti = set(), 0, 0
        while len(tts_rows) < need_t and ti < len(texts) and len(dead) < len(jobs):
            eng, voice = jobs[i % len(jobs)]
            i += 1
            if (eng, voice) in dead:
                continue
            text = texts[ti]
            ti += 1
            try:
                x = (tts_gtts(text, tmp) if eng == "gtts" else tts_mms(text) if eng == "mms"
                     else tts_fpt(text, voice, tmp, args.fpt_key) if eng == "fpt"
                     else tts_vieneu(text, voice, tmp) if eng == "vieneu"
                     else tts_piper(text, voice, tmp, Path(args.cache_dir) / "piper"))
            except Exception as e:
                fails[f"{voice}:{type(e).__name__}"] += 1
                if eng == "fpt" and getattr(e, "code", None) == 429:   # still 429 after back-off: quota of the key
                    dead.update(j for j in jobs if j[0] == "fpt")
                    print(f"[tts] FPT still 429 after back-off -> key quota/rate exhausted, disabling all FPT voices")
                    continue
                if fails[f"{voice}:{type(e).__name__}"] >= 3:
                    dead.add((eng, voice))
                    print(f"[tts] disabling {voice}: {e}")
                continue
            x = trim_silence(np.pad(np.asarray(x, dtype=np.float32), int(0.2 * SR)))   # some engines start/stop abruptly
            ok, why, info = check_clean(x, 25.0, args.min_dur, args.max_dur)
            if not ok:
                fails[why] += 1
                continue
            fn = f"tts_vi_test_src_t_{len(prev_t) + len(tts_rows) + 1:03d}.wav"
            write_wav_pcm16(out / fn, normalise(x))
            g_, r_ = FPT_VOICES.get(voice, ("unknown", "unknown")) if eng == "fpt" else ("unknown", "unknown")
            if eng == "vieneu":
                g_, r_ = vieneu_meta(voice)
            tts_rows.append({"file": fn, "speaker_id": f"tts_{eng}_{voice}", "gender": g_, "region": r_,
                             "source": "tts", "origin": eng, "origin_id": voice, "transcript": text, **info})
        print(f"[tts] generated {len(tts_rows)} new (+{len(prev_t)} kept) / {args.n_tts}; issues {dict(fails)}")

    picked = prev_h + picked
    tts_rows = prev_t + tts_rows
    rows = picked + tts_rows
    keys = ["file", "speaker_id", "gender", "region", "source", "origin", "origin_id", "transcript", "duration", "est_snr_db",
            "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    overlap = {r["speaker_id"] for r in rows} & ex["speakers"]
    d = [r["duration"] for r in rows] or [0]
    rep = ["# vi_test_v1 — clean sources", "",
           f"- total **{len(rows)}** (human {len(picked)}, TTS {len(tts_rows)}), target {args.n_human + args.n_tts}",
           f"- human by source: {dict(Counter(r['origin'] for r in picked))}; speakers {len(per_spk)}; "
           f"gender {dict(Counter(r['gender'] for r in picked))}; region {dict(Counter(r['region'] for r in picked))}",
           f"- TTS by voice: {dict(Counter(r['speaker_id'] for r in tts_rows))}",
           f"- duration s: min {min(d):.2f} / median {np.median(d):.2f} / max {max(d):.2f}",
           f"- rejected (human): {dict(reasons)}",
           (f"- DNSMOS (human, median): SIG {np.median([r['dnsmos_sig'] for r in picked if r.get('dnsmos_sig')]):.2f} / "
            f"BAK {np.median([r['dnsmos_bak'] for r in picked if r.get('dnsmos_bak')]):.2f} / "
            f"OVRL {np.median([r['dnsmos_ovrl'] for r in picked if r.get('dnsmos_ovrl')]):.2f}"
            if any(r.get("dnsmos_ovrl") for r in picked) else "- DNSMOS: not used"),
           f"- **speaker overlap with train: {len(overlap)}** {sorted(overlap) if overlap else '(none)'}",
           "", "Checks: 16 kHz mono PCM16, duration in range, no clipping, est. SNR >= threshold, not cut at boundary,",
           "silence trimmed to <= 0.3 s, active RMS 0.07; train speakers, files, sentences and TTS voices excluded."]
    (out / "selection_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[2:10]))
    if overlap:
        sys.exit("[error] speaker overlap with train - do not use")


if __name__ == "__main__":
    main()
