#!/usr/bin/env python3
"""
VLSP 2026 SQA - Vietnamese DSP augmentation engine  (owner: Thao, Speech Lead)

Two sub-commands:

  audit     Automatic part of the gatekeeping checklist (guide section 4.1 / 4.2)
            for the clean-speech dir (Dat) and the noise bank (Hoang).
            Writes reports/gatekeeping_<name>.md + .csv. Listening review is
            still manual: the report lists which files to listen to first.

  generate  Builds the degraded set (guide section 4.3):
              data/vietnamese_aug/wav/*.wav           16 kHz mono PCM16
              data/vietnamese_aug/labels.jsonl        submission schema (file_id, quality_score, faults)
              data/vietnamese_aug/params.jsonl        every random draw, for reproduction and the paper
              data/vietnamese_aug/metadata.csv        flat table for Hieu / Kien
              data/vietnamese_aug/generation_report.md

Processing chain per output file (order matters, it mirrors a real call path):
    clean speech -> [unnatural_pause]  (changes the time axis, so it goes first)
                 -> [reverb]           (room; optional, off by default)
                 -> [noise]            (background at the microphone)
                 -> [distortion]       (codec / clipping / quantisation in the channel)
                 -> peak safety + PCM16 write

Every interval written to labels.jsonl is the exact sample span the effect
was applied to (including fade ramps), in seconds on the OUTPUT time axis.

Usage
-----
python scripts/augment_engine.py audit \
    --clean-dir data/vietnamese_audio_clean --noise-dir data/vietnamese_noise_bank

python scripts/augment_engine.py generate \
    --clean-dir data/vietnamese_audio_clean --noise-dir data/vietnamese_noise_bank \
    --out-dir data/vietnamese_aug --variants-per-clean 1 --seed 2026

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
ENGINE_VERSION = "vi_aug_engine_v1.0"
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
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT)


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
        if row["level_range_db"] > 30:
            fl.append("VERY_DYNAMIC(listen: foreground speech?)")
        m = re.match(r"noise_(.+?)_\d+$", path.stem)
        row["category"] = m.group(1) if m else "UNKNOWN"
        if not m:
            fl.append("BAD_NAME(noise_<cat>_NN.wav)")
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
            lines += ["- [ ] Every category present (traffic/horn, rain, fan, cafe babble, vendor)",
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


def find_pause_points(x: np.ndarray, rng: np.random.Generator, k: int, min_gap_s: float = 0.6) -> list[int]:
    """Pick k insertion points at energy dips inside the speech region.
    Vietnamese is monosyllabic, so local energy minima approximate syllable/word boundaries."""
    hop = int(0.01 * SR)
    r = db(frame_rms(x, 0.01))
    act = r > (np.percentile(r, 95) - 35)
    idx = np.flatnonzero(act)
    if len(idx) < 50:
        return []
    lo, hi = idx[0] + 30, idx[-1] - 30            # >= 0.3 s away from speech onset/offset
    if hi <= lo:
        return []
    sm = np.convolve(r, np.ones(5) / 5, mode="same")
    cand = [i for i in range(lo + 2, hi - 2)
            if sm[i] <= sm[i - 2:i + 3].min() and sm[i] < np.percentile(sm[lo:hi], 40)]
    if not cand:
        cand = list(range(lo, hi))
    rng.shuffle(cand)
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


def apply_pauses(x: np.ndarray, rng: np.random.Generator, cfg) -> tuple[np.ndarray, list[Event]]:
    n_ev = int(rng.choice([1, 2], p=[0.75, 0.25]))
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
            y = splice(x, comfort_floor(len(x), floor, rng), s, e, ramp)
            return y, [Event("unnatural_pause", "dropout", s / SR, e / SR, {"len_s": (e - s) / SR})]
    pts = find_pause_points(x, rng, n_ev)
    if not pts:
        return x, []
    out, last, offset = [], 0, 0
    for p in pts:
        d = float(rng.uniform(cfg.pause_min, cfg.pause_max))
        n = int(round(d * SR))
        out.append(x[last:p])
        gap = comfort_floor(n, floor, rng)
        # short fades on the speech either side of the gap to avoid clicks
        out.append(gap)
        start = (p + offset) / SR
        events.append(Event("unnatural_pause", "insert_silence", start, start + n / SR, {"len_s": n / SR}))
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
    def __init__(self, d: Path, exclude: set[str]):
        self.by_cat: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for p in list_audio(d):
            if p.name in exclude:
                continue
            x, _ = load_audio(p)
            m = re.match(r"noise_(.+?)_\d+$", p.stem)
            cat = m.group(1) if m else "other"
            if np.sqrt(np.mean(x ** 2)) > 1e-4:
                self.by_cat[cat].append((p.name, x))
        if not self.by_cat:
            raise SystemExit(f"noise bank {d} is empty")
        self.cats = sorted(self.by_cat)

    def draw(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, str, str, float]:
        cat = self.cats[rng.integers(len(self.cats))]          # uniform over categories
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


def apply_noise(x: np.ndarray, rng: np.random.Generator, bank: NoiseBank, cfg) -> tuple[np.ndarray, list[Event]]:
    dur = len(x) / SR
    snr = float(rng.uniform(cfg.snr_min, cfg.snr_max))
    full = rng.random() < cfg.p_noise_full
    if full or dur < 1.5:
        s, e = 0, len(x)
    else:
        seg = float(rng.uniform(0.8, min(0.7 * dur, max(1.0, dur - 0.4))))
        s = int(rng.uniform(0, dur - seg) * SR)
        e = min(len(x), s + int(seg * SR))
    z, cat, name, off = bank.draw(e - s, rng)
    p_speech = active_speech_power(x)
    p_noise = float(np.mean(z ** 2) + 1e-12)
    g = math.sqrt(p_speech / (p_noise * 10 ** (snr / 10)))
    z = z * g * (fade_window(e - s, int(0.05 * SR)) if not full else 1.0)
    y = x.copy()
    y[s:e] += z
    return y, [Event("noise", "full" if (s == 0 and e == len(x)) else "segment", s / SR, e / SR,
                     {"snr_db": snr, "category": cat, "noise_file": name, "noise_offset_s": off})]


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


DIST_FX = {"narrowband": fx_narrowband, "clipping": fx_clipping,
           "quantize": fx_quantize, "packet_loss": fx_packet_loss}


def apply_distortion(x: np.ndarray, rng: np.random.Generator, cfg) -> tuple[np.ndarray, list[Event]]:
    kinds = list(cfg.dist_kinds)
    w = np.array([cfg.dist_weights.get(k, 1.0) for k in kinds], dtype=float)
    kind = kinds[rng.choice(len(kinds), p=w / w.sum())]
    dur = len(x) / SR
    y_full, params = DIST_FX[kind](x, rng)
    full = rng.random() < cfg.p_dist_full or dur < 1.5
    if full:
        return y_full, [Event("distortion", kind, 0.0, dur, {**params, "scope": "full"})]
    seg = float(rng.uniform(0.6, min(0.6 * dur, max(0.8, dur - 0.4))))
    s = int(rng.uniform(0, dur - seg) * SR)
    e = min(len(x), s + int(seg * SR))
    y = splice(x, y_full, s, e, int(0.01 * SR))
    return y, [Event("distortion", kind, s / SR, e / SR, {**params, "scope": "segment"})]


# =============================================================================
# Pseudo-MOS (guide section 5, v0 - Hieu owns the final formula)
# =============================================================================
def pseudo_mos(events: list[Event], dur: float, rng: np.random.Generator, cov_w: float) -> tuple[float, dict]:
    base = float(rng.uniform(4.5, 4.9))
    pen = {}
    for ev in events:
        cov = min(1.0, (ev.end - ev.start) / max(dur, 1e-6))
        scale = (1 - cov_w) + cov_w * cov          # cov_w=0 -> exactly the guide formula
        if ev.fault == "noise":
            lo, hi = (1.5, 2.2) if ev.params["snr_db"] <= 5 else (0.6, 1.2)
            p = rng.uniform(lo, hi) * scale
        elif ev.fault == "distortion":
            p = rng.uniform(1.2, 2.0) * (scale if ev.kind != "reverb" else 1.0)
        else:
            p = rng.uniform(0.8, 1.5) if (ev.end - ev.start) > 0.5 else rng.uniform(0.4, 0.8)
        pen[f"{ev.fault}:{ev.kind}"] = pen.get(f"{ev.fault}:{ev.kind}", 0.0) + float(p)
    mos = float(min(5.0, max(1.0, base - sum(pen.values()))))
    return mos, {"base": base, "penalties": pen}


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
    bank = NoiseBank(Path(args.noise_dir), reject)
    print(f"[gen] clean={len(clean)} (rejected {len(reject)}), noise categories="
          f"{ {c: len(v) for c, v in bank.by_cat.items()} }")

    rp = dict(DEFAULT_RECIPE_P)
    if args.recipe_probs:
        rp.update(json.loads(args.recipe_probs))
    names = list(rp)
    probs = np.array([rp[k] for k in names], dtype=float)
    probs /= probs.sum()
    args.dist_weights = json.loads(args.dist_weights) if args.dist_weights else {}

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
        for v in range(args.variants_per_clean):
            seed = int(hashlib.sha1(f"{args.seed}|{cp.name}|{v}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            recipe = names[rng.choice(len(names), p=probs)]
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
                x, ev = apply_noise(x, rng, bank, args)
                events += ev
            if "distortion" in faults and not use_reverb:
                x, ev = apply_distortion(x, rng, args)
                events += ev
            pk = float(np.max(np.abs(x)))
            peak_gain = 1.0
            if pk > 0.99:                       # safety only; clipping fx already applied above
                peak_gain = 0.99 / pk
                x = x * peak_gain
            dur = len(x) / SR
            fid = f"vi_aug_{k:06d}.wav"
            write_wav_pcm16(wav_dir / fid, x)
            k += 1
            fl = {f: serialize(normalize_intervals([[e.start, e.end] for e in events if e.fault == f], dur), dur)
                  for f in FAULTS}
            mos, mos_detail = pseudo_mos(events, dur, rng, args.mos_coverage_weight)
            f_lab.write(json.dumps({"file_id": fid, "quality_score": round(mos, 3), "faults": fl},
                                   ensure_ascii=False, default=_json_default) + "\n")
            f_par.write(json.dumps({
                "file_id": fid, "engine": ENGINE_VERSION, "seed": seed, "recipe": recipe,
                "source_clean": cp.name, "source_group_id": cp.stem, "variant": v,
                "duration": dur, "clean_duration": len(x0) / SR, "peak_safety_gain": peak_gain,
                "events": [asdict(e) for e in events], "mos": mos_detail,
            }, ensure_ascii=False, default=_json_default) + "\n")
            nz = [e for e in events if e.fault == "noise"]
            ds = [e for e in events if e.fault == "distortion"]
            ps = [e for e in events if e.fault == "unnatural_pause"]
            meta_rows.append({
                "file_id": fid, "source_clean": cp.name, "source_group_id": cp.stem,
                "source_type": "vi_aug", "split": "train", "duration": round(dur, 3),
                "n_frames": math.ceil(dur / HOP), "recipe": recipe, "quality_score": round(mos, 3),
                "has_noise": int(bool(fl["noise"])), "has_distortion": int(bool(fl["distortion"])),
                "has_unnatural_pause": int(bool(fl["unnatural_pause"])),
                "noise_snr_db": round(nz[0].params["snr_db"], 2) if nz else "",
                "noise_category": nz[0].params["category"] if nz else "",
                "noise_scope": nz[0].kind if nz else "",
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
    q = np.array([r["quality_score"] for r in rows])
    lines = [f"# Vietnamese augmentation - generation report", "",
             f"- engine: `{ENGINE_VERSION}`, seed: `{args.seed}`, variants/clean: {args.variants_per_clean}",
             f"- outputs: **{n}**, unique clean sources: {len({r['source_group_id'] for r in rows})}",
             f"- duration s: min {min(r['duration'] for r in rows):.2f} / "
             f"median {np.median([r['duration'] for r in rows]):.2f} / max {max(r['duration'] for r in rows):.2f}",
             f"- pseudo-MOS: mean {q.mean():.2f}, std {q.std():.2f}, min {q.min():.2f}, max {q.max():.2f}",
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
    hist = np.histogram(q, bins=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.0001])[0]
    lines += ["", "## Pseudo-MOS histogram", "", "| bin | count |", "|---|---|"]
    edges = ["1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0"]
    lines += [f"| {e} | {h} |" for e, h in zip(edges, hist)]
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
    a.add_argument("--noise-dir", default="data/vietnamese_noise_bank")
    a.add_argument("--report-dir", default="reports")
    a.set_defaults(func=cmd_audit)

    g = sub.add_parser("generate", help="build the degraded set")
    g.add_argument("--clean-dir", default="data/vietnamese_audio_clean")
    g.add_argument("--noise-dir", default="data/vietnamese_noise_bank")
    g.add_argument("--out-dir", default="data/vietnamese_aug")
    g.add_argument("--reject-list", default=None, help="txt with file names rejected at gatekeeping")
    g.add_argument("--variants-per-clean", type=int, default=1)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--seed", type=int, default=2026)
    g.add_argument("--target-rms", type=float, default=0.07, help="active-speech RMS after normalisation")
    g.add_argument("--recipe-probs", default=None, help='JSON override, e.g. \'{"clean":0.2}\'')
    g.add_argument("--snr-min", type=float, default=0.0)
    g.add_argument("--snr-max", type=float, default=15.0)
    g.add_argument("--p-noise-full", type=float, default=0.5, help="P(noise covers the whole file)")
    g.add_argument("--dist-kinds", nargs="+", default=["narrowband", "clipping", "quantize", "packet_loss"],
                   choices=list(DIST_FX))
    g.add_argument("--dist-weights", default='{"narrowband":0.35,"clipping":0.35,"quantize":0.15,"packet_loss":0.15}')
    g.add_argument("--p-dist-full", type=float, default=0.5, help="P(distortion covers the whole file)")
    g.add_argument("--pause-min", type=float, default=0.3)
    g.add_argument("--pause-max", type=float, default=1.2)
    g.add_argument("--p-dropout", type=float, default=0.25, help="P(pause event is a mid-speech dropout)")
    g.add_argument("--reverb", action="store_true", help="enable synthetic-RIR reverb, labelled distortion")
    g.add_argument("--p-reverb", type=float, default=0.15)
    g.add_argument("--mos-coverage-weight", type=float, default=0.5,
                   help="0 = guide formula exactly; >0 scales noise/distortion penalty by time coverage")
    g.set_defaults(func=cmd_generate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()