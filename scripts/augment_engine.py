#!/usr/bin/env python3
"""
VLSP 2026 SQA - Vietnamese DSP augmentation engine  (owner: Thao, Speech Lead)   v1.8

Two sub-commands (CPU only, no GPU needed):

  audit     Automatic part of the gatekeeping checklist for the clean-speech dir (Dat) and the
            noise bank (Hoang). Writes reports/gatekeeping_{clean,noise}.md + .csv. Listening is
            still manual: the report lists the flagged files to listen to first.

  generate  Builds the degraded set:
              <out>/wav/vi_aug_XXXXXX.wav   16 kHz mono PCM16
              <out>/labels.jsonl            fault intervals (file_id, faults) - quality_score is Hieu's part
              <out>/params.jsonl            every random draw (reproduction, Hieu's MOS formula, paper)
              <out>/metadata.csv            flat table for Hieu / Kien (source_group_id for folds)
              <out>/generation_report.md    statistics for the report / paper

Processing chain per output file (clean input normalised to active-speech RMS 0.07):
    unnatural_pause : 1-2 stops (0.4-0.8 s) INSERTED at syllable gaps; a pure break that
                      only contains the file's own very low floor (never noise, never zeros)
    [reverb]        : optional (--reverb), labelled distortion
    noise           : 1-2 short bursts (0.3-1.5 s, <= 50 % of the file), each starting at a syllable
                      gap; per file 75 % OVERLAY (mixed on top of the following speech) / 25 % INSERT
                      (noise-only burst put between the two syllables, file gets longer); SNR 0-15 dB between active levels;
                      fan x3, rain/traffic/motorbike x0.4; loudspeaker 10-20 dB + distance fx;
                      never inside a pause; labels = audible frames only.
                      (--p-noise-insert sets the insert share; 0 = overlay only)
    distortion      : narrowband G.711-like / clipping / low-bitrate codec artefacts / packet loss,
                      1-2 SHORT segments (0.4-1.5 s, <= 50 % of the file; no whole-file distortion),
                      one kind per file, always loudness-matched (bit reduction available but off:
                      its error is white hiss and sounds like noise)
    output          : random gain +-6 dB (peak-safe), PCM16

Every interval in labels.jsonl is on the OUTPUT time axis, rounded outward to 10 ms.

Usage
-----
python scripts/augment_engine.py audit \
    --clean-dir data/vietnamese_audio_clean --noise-dir data/noise_bank/vietnamese_noise_bank

python scripts/augment_engine.py generate --reject-list reports/reject.txt --seed 2026 \
    --clean-dir data/vietnamese_audio_clean --noise-dir data/noise_bank/vietnamese_noise_bank \
    --out-dir data/vietnamese_aug

Quick listening preview:  add  --limit 20 --out-dir data/vi_aug_preview

Dependencies: numpy, scipy. soundfile is used if installed (needed for .flac);
otherwise WAV files are read/written with scipy.io.wavfile.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy import signal

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None
from scipy.io import wavfile

SR = 16000
HOP = 0.02                      # evaluator grid (20 ms), see SQA_ARCHITECTURE_SPEC M4
FAULTS = ("noise", "distortion", "unnatural_pause")
ENGINE_VERSION = "vi_aug_engine_v2.7"
AUDIO_EXT = (".wav", ".flac")


# =============================================================================
# I/O
# =============================================================================
def load_audio(path: Path, target_sr: int = SR) -> tuple[np.ndarray, dict]:
    """float32 mono in [-1, 1] at target_sr, plus header info for the audit."""
    if sf is not None:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        subtype = sf.info(str(path)).subtype
    else:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(f"{path.name}: install soundfile to read {path.suffix}")
        sr, x = wavfile.read(str(path))
        subtype = str(x.dtype)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
        elif x.dtype == np.uint8:
            x = (x.astype(np.float32) - 128.0) / 128.0
        else:
            x = x.astype(np.float32)
        if x.ndim == 1:
            x = x[:, None]
    info = {"orig_sr": int(sr), "channels": int(x.shape[1]), "subtype": subtype,
            "orig_samples": int(x.shape[0])}
    x = x.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        x = signal.resample_poly(x, target_sr // g, int(sr) // g).astype(np.float32)
    return x, info


def write_wav_pcm16(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, -1.0, 32767 / 32768)
    if sf is not None:
        sf.write(str(path), x, sr, subtype="PCM_16")
    else:
        wavfile.write(str(path), sr, np.round(x * 32768).astype(np.int16))


def list_audio(d: Path) -> list[Path]:
    if not d.is_dir():
        raise SystemExit(f"[error] folder not found: {d.resolve()}\n"
                         f"        pass the real path with --clean-dir / --noise-dir")
    files = sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"[error] no .wav/.flac files under {d.resolve()}")
    return files


NOISE_NAME = re.compile(r"noise_(.+?)(?:_\d+)+$")


def noise_category(stem: str) -> str | None:
    """noise_fan_01 -> fan ; noise_fan_395645_01 -> fan ; noise_rain_tin_23087_01 -> rain_tin
    (any trailing numeric groups such as a source/recording id and the clip index are dropped)."""
    m = NOISE_NAME.match(stem)
    return m.group(1) if m else None


# =============================================================================
# Basic signal measurements
# =============================================================================
def frame_rms(x: np.ndarray, win: float = 0.02, sr: int = SR) -> np.ndarray:
    n = int(round(win * sr))
    m = len(x) // n
    if m == 0:
        return np.array([np.sqrt(np.mean(x ** 2) + 1e-12)])
    fr = x[: m * n].reshape(m, n)
    return np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)


def db(v: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(v, 1e-10))


def active_mask(x: np.ndarray, sr: int = SR, rel_db: float = -35.0) -> np.ndarray:
    """Energy VAD on 20 ms frames: active if within rel_db of the 95th percentile frame."""
    r = db(frame_rms(x, 0.02, sr))
    return r > (np.percentile(r, 95) + rel_db)


def active_speech_power(x: np.ndarray, sr: int = SR) -> float:
    n = int(0.02 * sr)
    m = active_mask(x, sr)
    fr = x[: len(m) * n].reshape(len(m), n)
    sel = fr[m] if m.any() else fr
    return float(np.mean(sel ** 2) + 1e-12)


def noise_floor_rms(x: np.ndarray, sr: int = SR) -> float:
    return float(np.percentile(frame_rms(x, 0.02, sr), 10))


def fade_window(n: int, ramp: int) -> np.ndarray:
    """1 inside, raised-cosine ramps of `ramp` samples at both ends."""
    w = np.ones(n, dtype=np.float32)
    ramp = min(ramp, n // 2)
    if ramp > 0:
        r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp, dtype=np.float32))
        w[:ramp] = r
        w[n - ramp:] = r[::-1]
    return w


def splice(base: np.ndarray, effected: np.ndarray, s: int, e: int, ramp: int) -> np.ndarray:
    """Crossfade `effected` into `base` over [s, e) so there is no click at the edges."""
    out = base.copy()
    w = fade_window(e - s, ramp)
    out[s:e] = (1 - w) * base[s:e] + w * effected[s:e]
    return out


# =============================================================================
# AUDIT (gatekeeping, automatic part)
# =============================================================================
def audit_file(path: Path, kind: str) -> dict:
    row = {"file": path.name, "kind": kind, "flags": []}
    try:
        x, info = load_audio(path, SR)
    except Exception as ex:
        row["flags"].append(f"UNREADABLE:{ex}")
        return row
    row.update(info)
    dur = len(x) / SR
    row["duration"] = round(dur, 3)
    row["rms"] = float(np.sqrt(np.mean(x ** 2)))
    row["peak"] = float(np.max(np.abs(x))) if len(x) else 0.0
    row["clip_frac"] = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0
    row["sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()
    fl = row["flags"]
    if not np.all(np.isfinite(x)):
        fl.append("NAN_INF")
    if info["orig_sr"] != SR:
        fl.append(f"SR_{info['orig_sr']}")
    if info["channels"] != 1:
        fl.append(f"CH_{info['channels']}")
    if row["clip_frac"] > 1e-4:
        fl.append("CLIPPED")
    # spectral check: almost no energy above 4 kHz => narrowband / low-bitrate source
    if len(x) >= 2048:
        f, p = signal.welch(x, SR, nperseg=1024)
        tot = p.sum() + 1e-20
        row["hf_ratio_4k"] = float(p[f >= 4000].sum() / tot)
        row["hf_ratio_7k"] = float(p[f >= 7000].sum() / tot)
        if row["hf_ratio_4k"] < 1e-4:
            fl.append("BANDLIMITED_<4k")
        elif row["hf_ratio_7k"] < 1e-6:
            fl.append("NO_ENERGY_>7k(check mp3/codec)")
    if kind == "clean":
        r = db(frame_rms(x))
        floor_db, peak_db = float(np.percentile(r, 10)), float(np.percentile(r, 95))
        row["floor_dbfs"], row["speech_dbfs"] = round(floor_db, 1), round(peak_db, 1)
        row["est_snr_db"] = round(peak_db - floor_db, 1)
        if not (2.5 <= dur <= 10.0):
            fl.append("DURATION_OUT_OF_2.5-10s")
        if not (0.02 <= row["rms"] <= 0.2):
            fl.append("RMS_OUT_OF_RANGE")
        if row["est_snr_db"] < 30:
            fl.append("NOT_CLEAN(est_snr<30dB)")
        # word cut at the boundaries: speech energy in the first/last 30 ms
        edge = int(0.03 * SR)
        if len(x) > 4 * edge:
            e0 = db(np.sqrt(np.mean(x[:edge] ** 2)))
            e1 = db(np.sqrt(np.mean(x[-edge:] ** 2)))
            if e0 > peak_db - 15 or e1 > peak_db - 15:
                fl.append("CUT_AT_BOUNDARY")
    else:  # noise
        if not (10.0 <= dur <= 15.5):
            fl.append("DURATION_OUT_OF_10-15s")
        r = db(frame_rms(x))
        row["level_range_db"] = round(float(np.percentile(r, 95) - np.percentile(r, 5)), 1)
        if row["rms"] < 1e-3:
            fl.append("NEAR_SILENT")
        row["active_minus_mean_db"] = round(float(10 * np.log10(active_speech_power(x) / (np.mean(x ** 2) + 1e-12))), 1)
        if row["active_minus_mean_db"] > 6:
            fl.append("BURSTY(active level >6 dB above mean; fine, SNR uses active level)")
        if row["level_range_db"] > 30:
            fl.append("VERY_DYNAMIC(listen: foreground speech?)")
        cat = noise_category(path.stem)
        row["category"] = cat or "UNKNOWN"
        if cat is None:
            fl.append("BAD_NAME(noise_<cat>_NN.wav)")
        elif any(row["category"].startswith(c) for c in ("street_vendor", "loudspeaker")) or "talk" in row["category"]:
            fl.append("SPEECH_LIKE(listen: no close intelligible voice; loudspeaker is mixed at 10-20 dB + distance fx)")
    return row


def cmd_audit(args) -> None:
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    for kind, d in (("clean", args.clean_dir), ("noise", args.noise_dir)):
        if d is None:
            continue
        files = list_audio(Path(d))
        rows = [audit_file(p, kind) for p in files]
        dup = Counter(r.get("sha1") for r in rows if r.get("sha1"))
        for r in rows:
            if r.get("sha1") and dup[r["sha1"]] > 1:
                r["flags"].append("DUPLICATE")
        keys = sorted({k for r in rows for k in r})
        with open(out / f"gatekeeping_{kind}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({**r, "flags": ";".join(r["flags"])})
        flagged = [r for r in rows if r["flags"]]
        fc = Counter(fl.split("(")[0] for r in rows for fl in r["flags"])
        lines = [f"# Gatekeeping report - {kind} ({d})", "",
                 f"Engine: {ENGINE_VERSION}", "",
                 f"- files: **{len(rows)}**, flagged: **{len(flagged)}**"]
        durs = [r["duration"] for r in rows if "duration" in r]
        if durs:
            lines.append(f"- duration s: min {min(durs):.2f} / median {np.median(durs):.2f} / max {max(durs):.2f}"
                         f" / total {sum(durs)/60:.1f} min")
        if kind == "noise":
            cats = Counter(r.get("category", "?") for r in rows)
            lines.append("- categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
        lines += ["", "## Flag counts", ""] + [f"- `{k}`: {v}" for k, v in fc.most_common()]
        lines += ["", "## Flagged files (listen to these first)", "", "| file | flags |", "|---|---|"]
        lines += [f"| {r['file']} | {'; '.join(r['flags'])} |" for r in flagged]
        lines += ["", "## Manual listening checklist (guide 4.1 / 4.2)", ""]
        if kind == "clean":
            lines += ["- [ ] Random 30-50 files: silent background, no fan hum, no strong reverb",
                      "- [ ] No cut first/last syllable", "- [ ] Consistent loudness, no overdrive",
                      "- [ ] Region split approx. 40% North / 40% South / 20% Central"]
        else:
            lines += ["- [ ] Categories present: fan, rain, traffic/motorbike, cafe babble, loudspeaker (street_vendor removed 09/10)",
                      "- [ ] No close, intelligible single talker in the noise",
                      "- [ ] No low-bitrate metallic mp3 artefacts"]
        (out / f"gatekeeping_{kind}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[audit] {kind}: {len(rows)} files, {len(flagged)} flagged -> {out}/gatekeeping_{kind}.md")


# =============================================================================
# Degradations
# =============================================================================
@dataclass
class Event:
    fault: str
    kind: str
    start: float
    end: float
    params: dict = field(default_factory=dict)


def find_pause_points(x: np.ndarray, rng: np.random.Generator, k: int, min_gap_s: float = 0.4,
                      dip_db: float = 12.0, exclude: list[tuple[float, float]] | None = None) -> list[int]:
    """Pick up to k insertion points at the DEEPEST energy dips inside the speech region
    (gaps between syllables/words; Vietnamese is monosyllabic). A dip must be at least `dip_db`
    below the speech level, otherwise it is probably inside a syllable and is not used:
    fewer pauses is better than cutting a syllable in half."""
    hop = int(0.01 * SR)
    r = db(frame_rms(x, 0.01))
    act = r > (np.percentile(r, 95) - 35)
    idx = np.flatnonzero(act)
    if len(idx) < 50:
        return []
    lo, hi = idx[0] + 30, idx[-1] - 30            # >= 0.3 s away from speech onset/offset
    if hi <= lo:
        return []
    sm = np.convolve(r, np.ones(3) / 3, mode="same")
    speech_db = float(np.percentile(r[act], 90))
    cand = [i for i in range(lo + 2, hi - 2)
            if sm[i] <= sm[i - 2:i + 3].min() and sm[i] <= speech_db - dip_db]
    if exclude:   # keep away (0.1 s) from spans already used by another fault, e.g. inserted pauses
        cand = [i for i in cand if all(not (a - 0.1 <= i * 0.01 <= b + 0.1) for a, b in exclude)]
    if not cand:
        return []
    cand.sort(key=lambda i: sm[i])                      # deepest boundaries first
    pool = cand[: max(2 * k + 2, 4)]
    rng.shuffle(pool)
    cand = pool + [c for c in cand if c not in pool]
    chosen: list[int] = []
    for c in cand:
        if all(abs(c - o) * 0.01 >= min_gap_s for o in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    return sorted(c * hop for c in chosen)


def comfort_floor(n: int, level_rms: float, rng: np.random.Generator) -> np.ndarray:
    """Low-level coloured noise matching the file's own floor, so inserted gaps are not
    digital zeros (that would be a trivial, unrealistic cue)."""
    w = rng.standard_normal(n + 64).astype(np.float32)
    b, a = signal.butter(2, 3000 / (SR / 2), "low")
    w = signal.lfilter(b, a, w)[64:].astype(np.float32)
    w *= level_rms / (np.sqrt(np.mean(w ** 2)) + 1e-12)
    return w


def room_tone(x: np.ndarray, n: int, rng: np.random.Generator, max_rel_db: float = -45.0) -> tuple[np.ndarray, dict]:
    """Fill for an inserted pause: the recording's OWN background (room tone), cut from its quietest
    10 ms frames (>= 40 dB below the speech level) and stitched with 2.5 ms crossfades.
    Its level is capped at speech level + max_rel_db (default -45 dB), so a pause is always heard
    as silence. If the file has no real silence (tightly trimmed clip), fall back to a very low
    hiss at speech - 60 dB. (v1.8 used the 10th-percentile frame level, which on tightly trimmed
    clips is a quiet consonant, ~-30 dB re speech, and was heard as a noise burst.)"""
    fr = int(0.01 * SR)
    r = db(frame_rms(x, 0.01))
    act = r > (np.percentile(r, 95) - 35)
    speech_db = float(np.percentile(r[act], 90)) if act.any() else float(np.max(r))
    cap = 10 ** ((speech_db + max_rel_db) / 20)
    quiet = np.flatnonzero(r <= speech_db - 40)
    if len(quiet) >= 5:
        frames = [x[i * fr:(i + 1) * fr].astype(np.float32) for i in quiet]
        xf = fr // 4
        w = np.linspace(0, 1, xf, dtype=np.float32)
        out = frames[0].copy()
        while len(out) < n + xf:
            f = frames[int(rng.integers(len(frames)))]
            out[-xf:] = out[-xf:] * (1 - w) + f[:xf] * w
            out = np.concatenate([out, f[xf:]])
        y, src = out[:n], "room_tone"
    else:
        y = rng.standard_normal(n + 64).astype(np.float32)
        b, a = signal.butter(2, 3000 / (SR / 2), "low")
        y = signal.lfilter(b, a, y)[64:].astype(np.float32)
        y *= 10 ** ((speech_db - 60) / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-12)
        src = "synthetic_-60dB"
    lvl = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    if lvl > cap:
        y = y * (cap / lvl)
        lvl = cap
    return y.astype(np.float32), {"fill": src, "fill_rel_db": float(20 * np.log10(lvl) - speech_db)}


def apply_pauses(x: np.ndarray, rng: np.random.Generator, cfg) -> tuple[np.ndarray, list[Event]]:
    n_ev = int(rng.integers(cfg.pause_count_min, cfg.pause_count_max + 1))
    floor = max(noise_floor_rms(x), 10 ** (-70 / 20))
    ramp = int(0.005 * SR)
    events: list[Event] = []
    if rng.random() < cfg.p_dropout:
        # mid-speech dropout: a stretch of active speech is replaced by the floor (duration kept)
        pts = find_pause_points(x, rng, 1)
        if pts:
            d = float(rng.uniform(0.25, 0.6))
            s = pts[0] - int(d * SR / 2)
            s = max(0, s)
            e = min(len(x), s + int(d * SR))
            y = splice(x, room_tone(x, len(x), rng)[0], s, e, ramp)
            return y, [Event("unnatural_pause", "dropout", s / SR, e / SR, {"len_s": (e - s) / SR})]
    pts = find_pause_points(x, rng, n_ev)
    if not pts:
        return x, []
    out, last, offset = [], 0, 0
    for p in pts:
        d = float(rng.uniform(cfg.pause_min, cfg.pause_max))
        n = int(round(d * SR))
        out.append(x[last:p])
        gap, fill = room_tone(x, n, rng)
        out.append(gap)
        start = (p + offset) / SR
        events.append(Event("unnatural_pause", "insert_silence", start, start + n / SR, {"len_s": n / SR, **fill}))
        offset += n
        last = p
    out.append(x[last:])
    y = np.concatenate(out).astype(np.float32)
    for ev in events:  # de-click edges
        s, e = int(ev.start * SR), int(ev.end * SR)
        for c in (s, e):
            a, b = max(0, c - ramp), min(len(y), c + ramp)
            y[a:b] *= np.minimum(1.0, np.abs(np.arange(a, b) - c) / ramp + 0.0).astype(np.float32)
    return y, events


def synth_rir(rng: np.random.Generator, rt60: float) -> np.ndarray:
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    h[0] = 1.0
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def apply_reverb(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[Event]]:
    rt60 = float(rng.uniform(0.3, 0.9))
    y = signal.fftconvolve(x, synth_rir(rng, rt60))[: len(x)].astype(np.float32)
    y *= np.sqrt(np.mean(x ** 2) / (np.mean(y ** 2) + 1e-12))
    return y, [Event("distortion", "reverb", 0.0, len(y) / SR, {"rt60": rt60})]


class NoiseBank:
    def __init__(self, d: Path, exclude: set[str], weights: dict | None = None):
        self.by_cat: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for p in list_audio(d):
            if p.name in exclude:
                continue
            x, _ = load_audio(p)
            cat = noise_category(p.stem) or "other"
            if np.sqrt(np.mean(x ** 2)) > 1e-4:
                self.by_cat[cat].append((p.name, x))
        if not self.by_cat:
            raise SystemExit(f"noise bank {d} is empty")
        self.cats = sorted(self.by_cat)
        w = np.array([float(_cat_lookup(weights or {}, c, 1.0)) for c in self.cats])
        self.cat_p = w / w.sum()

    def pick_category(self, rng: np.random.Generator) -> str:
        return self.cats[rng.choice(len(self.cats), p=self.cat_p)]   # weighted over categories

    def draw(self, n: int, rng: np.random.Generator, cat: str | None = None) -> tuple[np.ndarray, str, str, float]:
        if cat is None:
            cat = self.pick_category(rng)
        name, z = self.by_cat[cat][rng.integers(len(self.by_cat[cat]))]
        if len(z) < n:                                           # loop with crossfade
            xf = int(0.1 * SR)
            reps = [z]
            while sum(len(r) for r in reps) - xf * (len(reps) - 1) < n + xf:
                reps.append(z)
            out = reps[0].copy()
            w = np.linspace(0, 1, xf, dtype=np.float32)
            for r in reps[1:]:
                out[-xf:] = out[-xf:] * (1 - w) + r[:xf] * w
                out = np.concatenate([out, r[xf:]])
            z = out
        off = int(rng.integers(0, len(z) - n + 1))
        return z[off:off + n].astype(np.float32), cat, name, off / SR


def apply_noise(x: np.ndarray, rng: np.random.Generator, bank: NoiseBank, cfg,
                prior: list[Event]) -> tuple[np.ndarray, list[Event]]:
    """Short noise bursts (default 1-2 x 0.3-1.5 s) anchored at gaps BETWEEN syllables.
    Per file, one of two modes:
      overlay : the burst starts in a syllable gap and runs on top of the following speech
                (a fan/motorbike/rain cutting in while someone talks); duration unchanged.
      insert  : the two syllables are pushed apart and a noise-only burst is put between them
                (like a pause, but filled with noise); the file gets longer and every earlier
                event after the insertion point is shifted (prior events are updated in place).
    One category and one SNR per file (one environment)."""
    dur = len(x) / SR
    mode = "insert" if rng.random() < cfg.p_noise_insert else "overlay"
    n = int(rng.integers(cfg.noise_seg_count_min, cfg.noise_seg_count_max + 1))
    busy = [(e.start, e.end) for e in prior]
    pts = find_pause_points(x, rng, n, min_gap_s=0.6, exclude=busy)
    if not pts:
        return x, []
    cat = bank.pick_category(rng)
    # Speech-like noise (loudspeakers): at 0-5 dB a single intelligible voice sounds like a second
    # talker, so such categories get their own higher SNR range and a "distance" treatment.
    lo, hi = _cat_lookup(cfg.category_snr, cat, (cfg.snr_min, cfg.snr_max))
    snr = float(rng.uniform(lo, hi))
    is_distant = bool(_cat_lookup({c: True for c in cfg.distant_categories}, cat, False))
    p_speech = active_speech_power(x)
    floor = max(noise_floor_rms(x), 10 ** (-70 / 20))
    fade = int(0.02 * SR)
    budget = cfg.noise_max_cover * dur
    # burst lengths: overlay bursts may not run into the next anchor or past the end of the file
    plan = []
    for k_, p0 in enumerate(pts):
        L = float(rng.uniform(cfg.noise_seg_min, cfg.noise_seg_max))
        if mode == "overlay":
            limit = (pts[k_ + 1] / SR - 0.35) if k_ + 1 < len(pts) else dur   # keep bursts separate in the labels
            # an unnatural_pause is a pure break: an overlay burst must stop before the next pause
            nxt = [a for a, _b in busy if a > p0 / SR]
            if nxt:
                limit = min(limit, min(nxt) - 0.1)
            L = min(L, limit - p0 / SR)
        if L < min(cfg.noise_seg_min, 0.3) or sum(q[1] for q in plan) + L > budget:
            continue
        plan.append((p0, L))
    if not plan:
        return x, []

    def chunk(nsamp):
        z, _, name, off = bank.draw(nsamp, rng, cat=cat)
        distant = None
        if is_distant:
            z, distant = make_distant(z, rng)
        # SNR between ACTIVE levels of speech and noise (P.56-style), not mean power
        p_act = active_speech_power(z)
        g = math.sqrt(p_speech / (p_act * 10 ** (snr / 10)))
        info = {"noise_file": name, "noise_offset_s": off, "distant": distant,
                "snr_mean_power_db": snr + 10 * math.log10(p_act / (np.mean(z ** 2) + 1e-12))}
        return (z * g * fade_window(nsamp, fade)).astype(np.float32), info

    seg_info = []
    if mode == "overlay":
        y = x.copy()
        zz = np.zeros(len(x), np.float32)
        for p0, L in plan:
            s0, e0 = p0, min(len(x), p0 + int(L * SR))
            z, info = chunk(e0 - s0)
            y[s0:e0] += z
            zz[s0:e0] += z
            seg_info.append({"span": [s0 / SR, e0 / SR], **info})
    else:
        out, zparts, last, shift = [], [], 0, 0
        for p0, L in sorted(plan):
            nL = int(round(L * SR))
            z, info = chunk(nL)
            gap = room_tone(x, nL, rng)[0] + z
            out += [x[last:p0], gap]
            zparts += [np.zeros(p0 - last, np.float32), z]
            start = (p0 + shift) / SR
            for ev in prior:      # shift earlier-applied events (pauses) that lie after this point
                if ev.start * SR >= p0 + shift - 1:
                    ev.start += nL / SR
                    ev.end += nL / SR
            seg_info.append({"span": [start, start + nL / SR], "inserted_len_s": nL / SR, **info})
            shift += nL
            last = p0
        out.append(x[last:])
        zparts.append(np.zeros(len(x) - last, np.float32))
        y = np.concatenate(out).astype(np.float32)
        zz = np.concatenate(zparts).astype(np.float32)
    # label only where the added noise is audible (fade tails / near-silent stretches are not labelled)
    thr = p_speech * 10 ** (-cfg.noise_audible_snr / 10)
    nfr = int(HOP * SR)
    m = len(y) // nfr + (1 if len(y) % nfr else 0)
    zp = np.pad(zz, (0, m * nfr - len(y)))
    loud = np.mean(zp.reshape(m, nfr) ** 2, axis=1) >= thr
    loud = _close_open(loud, close_frames=10, min_frames=5)          # fill gaps < 0.2 s, drop < 0.1 s
    common = {"snr_db": snr, "category": cat, "snr_range": [lo, hi], "mode": mode,
              "n_segments": len(seg_info), "segments": seg_info, "spans": [si["span"] for si in seg_info],
              "inserts": [[si["span"][0], si["inserted_len_s"]] for si in seg_info if "inserted_len_s" in si],
              "noise_file": seg_info[0]["noise_file"], "distant": seg_info[0]["distant"]}
    evs = [Event("noise", mode, a * HOP, min(b * HOP, len(y) / SR), dict(common)) for a, b in _runs(loud)]
    if not evs and mode == "overlay":   # entirely inaudible overlay -> keep clean signal, no label
        return x, []
    return y, evs


def _cat_lookup(table: dict, cat: str, default):
    """Exact category match first, then prefix match (street_vendor matches street_vendor_hn)."""
    if cat in table:
        return table[cat]
    for k, v in table.items():
        if cat.startswith(k):
            return v
    return default


def make_distant(z: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Distance cues for a close-miked voice: air/HF loss (low-pass) + diffuse room/street tail."""
    fc = float(rng.uniform(2500, 4500))
    b, a = signal.butter(2, fc / (SR / 2), "low")
    y = signal.lfilter(b, a, z)
    rt60 = float(rng.uniform(0.4, 0.9))
    wet = signal.fftconvolve(y, synth_rir(rng, rt60))[: len(z)]
    drr_db = float(rng.uniform(-6, 0))                       # direct-to-reverberant ratio
    g = 10 ** (-drr_db / 20) * np.sqrt(np.mean(y ** 2) / (np.mean(wet ** 2) + 1e-12))
    y = y + g * wet
    return y.astype(np.float32), {"lowpass_hz": fc, "rt60": rt60, "drr_db": drr_db}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _close_open(mask: np.ndarray, close_frames: int, min_frames: int) -> np.ndarray:
    m = mask.copy()
    r = _runs(m)
    for (a0, b0), (a1, b1) in zip(r, r[1:]):
        if a1 - b0 < close_frames:
            m[b0:a1] = True
    for a, b in _runs(m):
        if b - a < min_frames:
            m[a:b] = False
    return m


def fx_narrowband(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """G.711-like: band-limit to telephone band, 8 kHz, 8-bit mu-law, back to 16 kHz."""
    y = signal.resample_poly(x, 1, 2)
    b, a = signal.butter(4, [300 / 4000, 3400 / 4000], "band")
    y = signal.lfilter(b, a, y)
    mu = 255.0
    pk = np.max(np.abs(y)) + 1e-9
    c = np.sign(y / pk) * np.log1p(mu * np.abs(y / pk)) / np.log1p(mu)
    c = np.round((c + 1) * 127.5) / 127.5 - 1                      # 8-bit
    y = np.sign(c) * (np.power(1 + mu, np.abs(c)) - 1) / mu * pk
    y = signal.resample_poly(y, 2, 1)[: len(x)]
    return y.astype(np.float32), {"codec": "g711_mulaw_8k"}


def fx_clipping(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Hard clip at a threshold below the peak, then restore loudness.
    The guide's `gain 1.5-3 then clip at 0.95` does not clip at all on files normalised to
    RMS 0.05-0.1 (peaks ~0.3-0.6), so the threshold is set relative to the signal instead:
    a random fraction (2-15 %) of speech samples gets clipped."""
    frac = float(rng.uniform(0.02, 0.15))
    act = np.abs(x[np.abs(x) > 1e-3])
    thr = float(np.quantile(act, 1 - frac)) if len(act) else 0.5
    y = np.clip(x, -thr, thr)
    y *= np.sqrt(np.mean(x ** 2) / (np.mean(y ** 2) + 1e-12))
    return y.astype(np.float32), {"clip_frac_target": frac, "threshold_rel_peak": thr / (np.max(np.abs(x)) + 1e-9)}


def fx_quantize(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    bits = int(rng.integers(4, 7))
    q = 2.0 ** (bits - 1)
    pk = np.max(np.abs(x)) + 1e-9
    y = np.round(x / pk * q) / q * pk
    return y.astype(np.float32), {"bits": bits}


def fx_packet_loss(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """VoIP-style 20 ms frame erasures concealed by repeating the previous frame (robotic/buzzy)."""
    n = int(0.02 * SR)
    y = x.copy()
    rate = float(rng.uniform(0.1, 0.3))
    lost = 0
    for i in range(1, len(x) // n):
        if rng.random() < rate:
            y[i * n:(i + 1) * n] = y[(i - 1) * n:i * n] * 0.9
            lost += 1
    return y, {"loss_rate": rate, "frames_lost": lost}


def fx_codec_lowbitrate(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Low-bitrate perceptual-codec artefacts (MP3/AAC/Opus at very low bitrate): per 32 ms frame only
    the strongest spectral bins survive, the rest is dropped -> metallic / 'underwater' / warbling
    'musical noise' and lost high band, all tied to the speech itself (silent when speech is silent).
    Replaces `quantize` as default: bit reduction adds a white, signal-independent hiss that sounds
    like (and gets confused with) background noise."""
    keep = float(rng.uniform(0.06, 0.15))                  # fraction of bins kept per frame
    cutoff = float(rng.uniform(3500, 5500))                # codec low-pass
    nper = 512
    f, t, Z = signal.stft(x, SR, nperseg=nper, noverlap=nper * 3 // 4)
    Z[f > cutoff, :] = 0
    mag = np.abs(Z)
    k = max(1, int(keep * np.count_nonzero(f <= cutoff)))
    thr = -np.sort(-mag, axis=0)[k - 1:k, :]               # k-th largest per frame
    Z = np.where(mag >= thr, Z, 0)
    _, y = signal.istft(Z, SR, nperseg=nper, noverlap=nper * 3 // 4)
    y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
    return y.astype(np.float32), {"bins_kept": keep, "lowpass_hz": cutoff}


DIST_FX = {"narrowband": fx_narrowband, "clipping": fx_clipping, "codec_lowbitrate": fx_codec_lowbitrate,
           "quantize": fx_quantize, "packet_loss": fx_packet_loss}


def apply_distortion(x: np.ndarray, rng: np.random.Generator, cfg) -> tuple[np.ndarray, list[Event]]:
    kinds = list(cfg.dist_kinds)
    w = np.array([cfg.dist_weights.get(k, 1.0) for k in kinds], dtype=float)
    kind = kinds[rng.choice(len(kinds), p=w / w.sum())]
    dur = len(x) / SR
    y_full, params = DIST_FX[kind](x, rng)
    # keep the active level unchanged (narrowband alone drops it by ~5 dB), otherwise a segment
    # splice creates a loudness step that is an extra, unlabelled cue
    lvl = math.sqrt(active_speech_power(x) / active_speech_power(y_full))
    y_full = (y_full * lvl).astype(np.float32)
    params["level_match_db"] = 20 * math.log10(lvl)
    if rng.random() < cfg.p_dist_full:          # off by default: whole-file distortion is not noticeable
        return y_full, [Event("distortion", kind, 0.0, dur, {**params, "scope": "full"})]
    # SHORT segments: the contrast clean -> distorted -> clean is what makes distortion audible
    n = int(rng.integers(cfg.dist_seg_count_min, cfg.dist_seg_count_max + 1))
    cap = min(cfg.dist_seg_max, 0.35 * dur)
    lo = min(cfg.dist_seg_min, cap)
    segs: list[tuple[float, float]] = []
    for _ in range(n):
        L = float(rng.uniform(lo, cap))
        if sum(b - a for a, b in segs) + L > cfg.dist_max_cover * dur:
            break
        for _try in range(30):
            a = float(rng.uniform(0, dur - L))
            if all(a + L + 0.3 <= s0 or a >= s1 + 0.3 for s0, s1 in segs):
                segs.append((a, a + L))
                break
    if not segs:                                # very short file: one minimal segment
        a0 = float(rng.uniform(0, dur - lo))
        segs = [(a0, a0 + lo)]
    y = x
    evs = []
    for a, b in sorted(segs):
        s0, e0 = int(a * SR), min(len(x), int(b * SR))
        y = splice(y, y_full, s0, e0, int(0.01 * SR))
        evs.append(Event("distortion", kind, s0 / SR, e0 / SR, {**params, "scope": "segment", "n_segments": len(segs)}))
    return y, evs


# =============================================================================
# Interval utilities (same semantics as sqa/timegrid.normalize_intervals)
# =============================================================================
def normalize_intervals(iv: list[list[float]], dur: float) -> list[list[float]]:
    iv = sorted([max(0.0, s), min(dur, e)] for s, e in iv if e > s)
    out: list[list[float]] = []
    for s, e in iv:
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def serialize(iv: list[list[float]], dur: float) -> list[list[float]]:
    """Round to 10 ms outward (start floor, end ceil). An interval that reaches the end of the
    file keeps the exact duration (6 decimals, floored) so the last 20 ms evaluator frame is
    never dropped; nothing ever exceeds `dur`."""
    dur6 = math.floor(dur * 1e6) / 1e6
    out = []
    for s, e in iv:
        s2 = round(math.floor(s * 100 + 1e-9) / 100, 2)
        e2 = round(math.ceil(e * 100 - 1e-9) / 100, 2)
        if e2 > dur6 or e >= dur - 1e-6:
            e2 = dur6
        if e2 > s2:
            out.append([s2, e2])
    return normalize_intervals(out, dur6)


# =============================================================================
# GENERATE
# =============================================================================
RECIPES = {  # name -> faults applied
    "clean": (),
    "noise": ("noise",),
    "distortion": ("distortion",),
    "pause": ("unnatural_pause",),
    "noise+distortion": ("noise", "distortion"),
    "noise+pause": ("noise", "unnatural_pause"),
    "distortion+pause": ("distortion", "unnatural_pause"),
    "all": ("noise", "distortion", "unnatural_pause"),
}
DEFAULT_RECIPE_P = {"clean": 0.12, "noise": 0.20, "distortion": 0.18, "pause": 0.15,
                    "noise+distortion": 0.10, "noise+pause": 0.10, "distortion+pause": 0.08, "all": 0.07}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_reject_list(path: str | None) -> set[str]:
    if not path:
        return set()
    return {l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}


def cmd_generate(args) -> None:
    clean_dir, out = Path(args.clean_dir), Path(args.out_dir)
    reject = load_reject_list(args.reject_list)
    clean = [p for p in list_audio(clean_dir) if p.name not in reject]
    if args.limit:
        clean = clean[: args.limit]
    args.category_weights = json.loads(args.category_weights or "{}")
    bank = NoiseBank(Path(args.noise_dir), reject, args.category_weights)
    print(f"[gen] clean={len(clean)} (rejected {len(reject)}), noise categories (files, P(draw)): "
          f"{ {c: (len(bank.by_cat[c]), round(float(q), 2)) for c, q in zip(bank.cats, bank.cat_p)} }")

    rp = dict(DEFAULT_RECIPE_P)
    if args.recipe_probs:
        rp.update(json.loads(args.recipe_probs))
    names = list(rp)
    probs = np.array([rp[k] for k in names], dtype=float)
    probs /= probs.sum()
    args.dist_weights = json.loads(args.dist_weights) if args.dist_weights else {}
    args.category_snr = {k: tuple(v) for k, v in json.loads(args.category_snr or "{}").items()}

    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    f_lab = open(out / "labels.jsonl", "w", encoding="utf-8")
    f_par = open(out / "params.jsonl", "w", encoding="utf-8")
    meta_rows = []
    k = 0
    for ci, cp in enumerate(clean):
        try:
            x0, info = load_audio(cp)
        except Exception as ex:
            print(f"[gen] skip {cp.name}: {ex}", file=sys.stderr)
            continue
        if len(x0) < int(1.0 * SR) or not np.all(np.isfinite(x0)):
            print(f"[gen] skip {cp.name}: too short or NaN", file=sys.stderr)
            continue
        # normalise the clean input to a fixed active-speech level so SNRs are comparable
        x0 = x0 * (args.target_rms / math.sqrt(active_speech_power(x0)))
        has_clean_copy = False
        for v in range(args.variants_per_clean):
            seed = int(hashlib.sha1(f"{args.seed}|{cp.name}|{v}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            recipe = names[rng.choice(len(names), p=probs)]
            if recipe == "clean" and has_clean_copy:
                # at most ONE no-fault copy per clean utterance: with several variants, a second clean
                # copy would be a near-duplicate (only the gain differs) -> draw a faulty recipe instead
                nf = np.array([0.0 if k_ == "clean" else rp[k_] for k_ in names])
                recipe = names[rng.choice(len(names), p=nf / nf.sum())]
            has_clean_copy |= recipe == "clean"
            faults = RECIPES[recipe]
            x = x0.copy()
            events: list[Event] = []
            if "unnatural_pause" in faults:
                x, ev = apply_pauses(x, rng, args)
                events += ev
            use_reverb = "distortion" in faults and args.reverb and rng.random() < args.p_reverb
            if use_reverb:                      # room stage, replaces the channel distortion
                x, ev = apply_reverb(x, rng)
                events += ev
            if "noise" in faults:
                x, ev = apply_noise(x, rng, bank, args, events)   # may shift earlier events (insert mode)
                events += ev
            if "distortion" in faults and not use_reverb:
                x, ev = apply_distortion(x, rng, args)
                events += ev
            # random presentation level so absolute loudness is not a shortcut for "clean";
            # reduced if it would push the peak over 0.99 (never re-clips)
            gain_db = float(rng.uniform(-args.level_jitter_db, args.level_jitter_db))
            gain = 10 ** (gain_db / 20)
            pk = float(np.max(np.abs(x)))
            peak_gain = 1.0
            if pk * gain > 0.99:
                peak_gain = 0.99 / (pk * gain)
            out_gain = gain * peak_gain
            x = x * out_gain
            speech_lvl = 10 * math.log10(active_speech_power(x))
            rms_dbfs = 20 * math.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12)
            peak_dbfs = 20 * math.log10(float(np.max(np.abs(x))) + 1e-12)
            dur = len(x) / SR
            fid = f"vi_aug_{k:06d}.wav"
            write_wav_pcm16(wav_dir / fid, x)
            k += 1
            fl = {f: serialize(normalize_intervals([[e.start, e.end] for e in events if e.fault == f], dur), dur)
                  for f in FAULTS}
            # quality_score (MOS) is NOT produced here: it is Hieu's part, computed from params.jsonl
            f_lab.write(json.dumps({"file_id": fid, "faults": fl},
                                   ensure_ascii=False, default=_json_default) + "\n")
            f_par.write(json.dumps({
                "file_id": fid, "engine": ENGINE_VERSION, "seed": seed, "recipe": recipe,
                "source_clean": cp.name, "source_group_id": cp.stem, "variant": v,
                "duration": dur, "clean_duration": len(x0) / SR, "peak_safety_gain": peak_gain,
                "output_gain": out_gain, "output_gain_db": 20 * math.log10(out_gain),
                "active_level_dbfs": speech_lvl, "rms_dbfs": rms_dbfs, "peak_dbfs": peak_dbfs,
                "events": [asdict(e) for e in events],
            }, ensure_ascii=False, default=_json_default) + "\n")
            nz = [e for e in events if e.fault == "noise"]
            ds = [e for e in events if e.fault == "distortion"]
            ps = [e for e in events if e.fault == "unnatural_pause"]
            meta_rows.append({
                "file_id": fid, "source_clean": cp.name, "source_group_id": cp.stem,
                "source_type": "vi_aug", "split": "train", "duration": round(dur, 3),
                "n_frames": math.ceil(dur / HOP), "recipe": recipe,
                "has_noise": int(bool(fl["noise"])), "has_distortion": int(bool(fl["distortion"])),
                "has_unnatural_pause": int(bool(fl["unnatural_pause"])),
                "active_level_dbfs": round(speech_lvl, 2), "rms_dbfs": round(rms_dbfs, 2),
                "peak_dbfs": round(peak_dbfs, 2), "peak_limited": int(peak_gain < 1.0), "output_gain_db": round(20 * math.log10(out_gain), 2),
                "noise_snr_db": round(nz[0].params["snr_db"], 2) if nz else "",
                "noise_category": nz[0].params["category"] if nz else "",
                "noise_mode": nz[0].kind if nz else "",
                "noise_n_segments": nz[0].params["n_segments"] if nz else "",
                "noise_cover_s": round(sum(e.end - e.start for e in nz), 3),
                "distortion_kind": "|".join(e.kind for e in ds),
                "n_pauses": len(ps), "pause_total_s": round(sum(e.end - e.start for e in ps), 3),
                "seed": seed,
            })
        if (ci + 1) % 100 == 0:
            print(f"[gen] {ci + 1}/{len(clean)} clean files -> {k} outputs")
    f_lab.close()
    f_par.close()
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0]))
        w.writeheader()
        w.writerows(meta_rows)
    write_report(out, meta_rows, args)
    print(f"[gen] done: {k} files -> {out}")


def write_report(out: Path, rows: list[dict], args) -> None:
    n = len(rows)
    lines = [f"# Vietnamese augmentation - generation report", "",
             f"- engine: `{ENGINE_VERSION}`, seed: `{args.seed}`, variants/clean: {args.variants_per_clean}",
             f"- outputs: **{n}**, unique clean sources: {len({r['source_group_id'] for r in rows})}",
             f"- duration s: min {min(r['duration'] for r in rows):.2f} / "
             f"median {np.median([r['duration'] for r in rows]):.2f} / max {max(r['duration'] for r in rows):.2f}",
             f"- active speech level dBFS: p5 {np.percentile([r['active_level_dbfs'] for r in rows], 5):.1f}"
             f" / median {np.median([r['active_level_dbfs'] for r in rows]):.1f}"
             f" / p95 {np.percentile([r['active_level_dbfs'] for r in rows], 95):.1f};"
             f" peak dBFS max {max(r['peak_dbfs'] for r in rows):.2f};"
             f" files whose level jitter was reduced for peak safety: "
             f"{sum(r['peak_limited'] for r in rows)}",
             "", "## Fault prevalence", ""]
    for f in FAULTS:
        c = sum(r[f"has_{f}"] for r in rows)
        lines.append(f"- {f}: {c} ({100 * c / n:.1f} %)")
    lines.append(f"- no fault: {sum(1 for r in rows if r['recipe'] == 'clean')}")
    lines += ["", "## Recipes", ""] + [f"- {k}: {v}" for k, v in Counter(r["recipe"] for r in rows).most_common()]
    lines += ["", "## Noise categories", ""] + [f"- {k}: {v}" for k, v in
                                                 Counter(r["noise_category"] for r in rows if r["noise_category"]).most_common()]
    lines += ["", "## Distortion kinds", ""] + [f"- {k}: {v}" for k, v in
                                                 Counter(r["distortion_kind"] for r in rows if r["distortion_kind"]).most_common()]
    nzr = [r for r in rows if r["noise_category"]]
    if nzr:
        snr = np.array([r["noise_snr_db"] for r in nzr], float)
        cov = np.array([r["noise_cover_s"] / r["duration"] for r in nzr])
        lines += ["", "## Noise bursts", "",
                  "- mode: " + ", ".join(f"{k} {v}" for k, v in Counter(r["noise_mode"] for r in nzr).most_common()),
                  "- bursts per file: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r["noise_n_segments"] for r in nzr).items())),
                  f"- labelled coverage of the file: mean {cov.mean():.0%}, max {cov.max():.0%}",
                  f"- SNR dB: min {snr.min():.1f} / median {np.median(snr):.1f} / max {snr.max():.1f}",
                  "- SNR histogram: " + ", ".join(f"{a}-{a + 3} dB: {c}" for a, c in
                                                   zip(range(0, 21, 3), np.histogram(snr, bins=range(0, 24, 3))[0]))]
    pr = [r for r in rows if r["n_pauses"]]
    if pr:
        lines += ["", "## Pauses", "",
                  "- stops per file: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r["n_pauses"] for r in pr).items())),
                  f"- total stop time per file s: mean {np.mean([r['pause_total_s'] for r in pr]):.2f}, "
                  f"max {max(r['pause_total_s'] for r in pr):.2f}"]
    lines += ["", "## Config", "", "```json",
              json.dumps({k: v for k, v in vars(args).items() if k != "func"}, indent=1, default=str), "```"]
    (out / "generation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="automatic gatekeeping checks")
    a.add_argument("--clean-dir", default="data/vietnamese_audio_clean")
    a.add_argument("--noise-dir", default="data/noise_bank/vietnamese_noise_bank")
    a.add_argument("--report-dir", default="reports")
    a.set_defaults(func=cmd_audit)

    g = sub.add_parser("generate", help="build the degraded set")
    g.add_argument("--clean-dir", default="data/vietnamese_audio_clean")
    g.add_argument("--noise-dir", default="data/noise_bank/vietnamese_noise_bank")
    g.add_argument("--out-dir", default="data/vietnamese_aug")
    g.add_argument("--reject-list", default=None, help="txt with file names rejected at gatekeeping")
    g.add_argument("--variants-per-clean", type=int, default=1)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--seed", type=int, default=2026)
    g.add_argument("--target-rms", type=float, default=0.07, help="active-speech RMS after normalisation")
    g.add_argument("--level-jitter-db", type=float, default=6.0,
                   help="random output gain U(-x, +x) dB (0 = every file at the same speech level)")
    g.add_argument("--noise-audible-snr", type=float, default=30.0,
                   help="noise frames quieter than speech level minus this (dB) are not labelled")
    g.add_argument("--recipe-probs", default=None, help='JSON override, e.g. \'{"clean":0.2}\'')
    g.add_argument("--snr-min", type=float, default=0.0)
    g.add_argument("--snr-max", type=float, default=15.0)
    g.add_argument("--category-snr", default='{"street_vendor": [10, 20], "loudspeaker": [10, 20]}',
                   help="JSON {category: [snr_min, snr_max]} overriding --snr-min/--snr-max for speech-like noise")
    g.add_argument("--distant-categories", nargs="*", default=["street_vendor", "loudspeaker"],
                   help="categories that get low-pass + reverb so they sound far away (empty to disable)")
    g.add_argument("--p-noise-insert", type=float, default=0.25,
                   help="P(noise burst is INSERTED into a syllable gap, pushing speech apart); "
                        "otherwise it starts in a gap and overlays the following speech")
    g.add_argument("--noise-seg-count-min", type=int, default=1, help="noise bursts per noisy file")
    g.add_argument("--noise-seg-count-max", type=int, default=2)
    g.add_argument("--noise-seg-min", type=float, default=0.3, help="burst length (s)")
    g.add_argument("--noise-seg-max", type=float, default=1.5)
    g.add_argument("--noise-max-cover", type=float, default=0.5, help="max fraction of the file covered by noise")
    g.add_argument("--dist-kinds", nargs="+", default=["narrowband", "clipping", "codec_lowbitrate", "packet_loss"],
                   choices=list(DIST_FX))
    g.add_argument("--dist-weights",
                   default='{"narrowband":0.3,"clipping":0.3,"codec_lowbitrate":0.25,"packet_loss":0.15}')
    g.add_argument("--p-dist-full", type=float, default=0.0,
                   help="P(distortion covers the whole file); default 0 = only short segments")
    g.add_argument("--dist-seg-count-min", type=int, default=1, help="distorted segments per file")
    g.add_argument("--dist-seg-count-max", type=int, default=2)
    g.add_argument("--dist-seg-min", type=float, default=0.4, help="segment length (s)")
    g.add_argument("--dist-seg-max", type=float, default=1.5)
    g.add_argument("--dist-max-cover", type=float, default=0.5, help="max fraction of the file distorted")
    g.add_argument("--pause-min", type=float, default=0.4, help="length of each inserted stop (s)")
    g.add_argument("--pause-max", type=float, default=0.8)
    g.add_argument("--pause-count-min", type=int, default=1, help="stops per file (inserted between syllables)")
    g.add_argument("--pause-count-max", type=int, default=2)
    g.add_argument("--p-dropout", type=float, default=0.0,
                   help="P(pause event mutes a stretch of speech instead); 0 = never mute syllables")
    g.add_argument("--category-weights",
                   default='{"fan": 3, "rain": 0.4, "traffic": 0.4, "motorbike": 0.4, "horn": 0.4, "street": 0.4}',
                   help="JSON {category_prefix: weight} for how often each noise category is drawn (default 1)")
    g.add_argument("--reverb", action="store_true", help="enable synthetic-RIR reverb, labelled distortion")
    g.add_argument("--p-reverb", type=float, default=0.15)
    g.set_defaults(func=cmd_generate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()