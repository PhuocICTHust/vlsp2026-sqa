"""Self-test for augment_engine.py on synthetic data.

Builds fake clean speech (harmonic syllables with Vietnamese-like 4-6 syl/s rhythm) and a fake
noise bank, runs audit + generate, then checks that every label is exactly where the signal changed:
  * outside the labelled intervals the output must equal the (level-normalised) clean input,
  * removing inserted-pause spans must give back the clean input,
  * schema / bounds / sort / overlap, 16 kHz mono PCM16, no NaN.
Run:  python tests/test_augment_engine.py
"""
import json, math, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy import signal

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "scripts" / "augment_engine.py"
sys.path.insert(0, str(ENGINE.parent))
import augment_engine as E  # noqa: E402

EXPECTED_ENGINE = "vi_aug_engine_v3.1"
if E.ENGINE_VERSION != EXPECTED_ENGINE:
    sys.exit(f"Version mismatch: scripts/augment_engine.py is {E.ENGINE_VERSION}, this test expects "
             f"{EXPECTED_ENGINE}. Download both files again from the same version.")

SR = 16000
rng = np.random.default_rng(0)


def fake_speech(dur):
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    pos = 0.25
    while pos < dur - 0.35:
        L = rng.uniform(0.12, 0.25)
        f0 = rng.uniform(100, 220)
        m = (t >= pos) & (t < pos + L)
        env = np.sin(np.pi * (t[m] - pos) / L)
        y[m] = env * sum(np.sin(2 * np.pi * f0 * h * t[m]) / h for h in range(1, 8))
        y[m] += env * 0.05 * rng.standard_normal(m.sum())   # broadband (fricative-like) part
        pos += L + rng.uniform(0.03, 0.12)
    y = y / np.max(np.abs(y)) * 0.5 + rng.standard_normal(len(y)) * 1e-4
    return y.astype(np.float32)


def w16(p, x, sr=SR):
    p.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x)
    wavfile.write(str(p), sr, np.round(np.clip(x, -1, 0.99997) * 32767).astype(np.int16))


def main():
    for stem, cat in [("noise_fan_01", "fan"), ("noise_fan_395645_01", "fan"),
                      ("noise_rain_tin_23087_01", "rain_tin"), ("noise_loudspeaker_350722_06", "loudspeaker"),
                      ("noise_cafe_babble_01", "cafe_babble"), ("fan_01", None)]:
        assert E.noise_category(stem) == cat, (stem, E.noise_category(stem))
    root = Path(tempfile.mkdtemp())
    clean, noise, out = root / "clean", root / "noise", root / "aug"
    for i in range(40):
        w16(clean / f"vi_clean_{i:03d}.wav", fake_speech(rng.uniform(2.5, 9.5)))
    # deliberately bad files for the audit
    w16(clean / "bad_stereo_44k.wav", np.stack([signal.resample_poly(fake_speech(3), 441, 160)] * 2, 1), 44100)
    w16(clean / "bad_clipped.wav", np.clip(fake_speech(4) * 5, -1, 1))
    w16(clean / "bad_short.wav", fake_speech(1.2))
    for cat in ["traffic", "fan", "rain", "cafe_babble"]:
        for j in range(3):
            z = rng.standard_normal(int(12 * SR))
            if cat == "fan":
                z = np.convolve(z, np.ones(40) / 40, "same") + 0.3 * np.sin(2 * np.pi * 120 * np.arange(len(z)) / SR)
            w16(noise / f"noise_{cat}_{j + 1:02d}.wav", z / np.max(np.abs(z)) * 0.3)
    # bursty noise: quiet street + two 0.8 s horns  (mean-power SNR would make the horns too loud)
    for j in range(2):
        z = rng.standard_normal(int(12 * SR)) * 0.002
        tt = np.arange(int(0.8 * SR)) / SR
        for c in (3.0 + j, 8.0 - j):
            z[int(c * SR):int(c * SR) + len(tt)] += 0.4 * np.sign(np.sin(2 * np.pi * 420 * tt)) * np.hanning(len(tt)) ** 0.2
        w16(noise / f"noise_horn_{j + 1:02d}.wav", z)
    # close, loud street-vendor call (harmonic "voice"), should be pushed to SNR >= 10 dB and made distant
    for j in range(2):
        v = fake_speech(12.0) * 0.6
        w16(noise / f"noise_street_vendor_{j + 1:02d}.wav", v + rng.standard_normal(len(v)) * 0.002)

    py = sys.executable
    subprocess.run([py, str(ENGINE), "audit", "--clean-dir", str(clean), "--noise-dir", str(noise),
                    "--report-dir", str(root / "reports")], check=True)
    rep = (root / "reports" / "gatekeeping_clean.md").read_text()
    for name, flag in [("bad_stereo_44k", "CH_2"), ("bad_clipped", "CLIPPED"), ("bad_short", "DURATION")]:
        line = [l for l in rep.splitlines() if name in l]
        assert line and flag in line[0], f"audit missed {flag} on {name}"
    (root / "reject.txt").write_text("bad_stereo_44k.wav\nbad_clipped.wav\nbad_short.wav\n")

    subprocess.run([py, str(ENGINE), "generate", "--clean-dir", str(clean), "--noise-dir", str(noise),
                    "--out-dir", str(out), "--reject-list", str(root / "reject.txt"),
                    "--variants-per-clean", "5", "--seed", "7", "--reverb",
                    # round-1 regression run (round-2 mix is tested in test_round2.py)
                    "--recipe-set", "round1", "--no-exact-ratios"], check=True)

    labels = [json.loads(l) for l in open(out / "labels.jsonl")]
    params = {p["file_id"]: p for p in map(json.loads, open(out / "params.jsonl"))}
    assert len(labels) == 200, len(labels)
    worst, levels, level_steps = {}, [], []
    for lab in labels:
        fid = lab["file_id"]
        sr, y = wavfile.read(out / "wav" / fid)
        assert sr == SR and y.ndim == 1 and y.dtype == np.int16
        y = y.astype(np.float32) / 32768
        assert np.all(np.isfinite(y))
        dur = len(y) / SR
        assert set(lab) == {"file_id", "faults"}, "labels.jsonl must not contain quality_score (Hieu's part)"
        assert set(lab["faults"]) == set(E.FAULTS)
        for f, iv in lab["faults"].items():
            prev = -1
            for s, e in iv:
                assert 0 <= s < e <= dur + 1e-9, (fid, f, s, e, dur)
                assert s >= prev
                prev = e
        p = params[fid]
        assert np.max(np.abs(y)) <= 0.991
        levels.append(p["active_level_dbfs"])
        y = y / p["output_gain"]                      # undo the random presentation level
        x0, _ = E.load_audio(clean / p["source_clean"])
        x0 = x0 * (0.07 / math.sqrt(E.active_speech_power(x0)))
        # undo inserted pauses (they are the only events that change the time axis)
        # every stretch that was INSERTED (pauses and insert-mode noise) as (start_s, len_s), final time axis
        cuts = [(e["start"], e["params"]["len_s"]) for e in p["events"] if e["kind"] == "insert_silence"]
        nev = [e for e in p["events"] if e["fault"] == "noise"]
        if nev:
            cuts += [tuple(c) for c in nev[0]["params"]["inserts"]]
        keep = np.ones(len(y), bool)
        for c0s, ls in cuts:
            keep[int(round(c0s * SR)):int(round(c0s * SR)) + int(round(ls * SR))] = False
        y_back = y[keep]
        assert abs(len(y_back) - len(x0)) <= 2, (fid, len(y_back), len(x0))
        L = min(len(y_back), len(x0))
        diff = np.abs(y_back[:L] - x0[:L])
        # mask of samples that may differ: labelled spans (mapped back) + 6 ms de-click ramps around pauses
        allowed = np.zeros(len(y), bool)
        for f, iv in lab["faults"].items():
            for s, e in iv:
                allowed[int(s * SR):int(math.ceil(e * SR))] = True
        for c0s, ls in cuts:
            c0 = int(round(c0s * SR)); c1 = c0 + int(round(ls * SR))
            for c in (c0, c1):
                allowed[max(0, c - 100):c + 100] = True
        allowed_back = allowed[keep][:L]
        # inaudible (unlabelled) noise is allowed inside the mixing span, but must stay below the
        # audibility threshold: speech level - 30 dB  => RMS < 0.07 * 10^(-30/20) per 20 ms frame
        span = np.zeros(len(y), bool)
        for e in p["events"]:
            if e["fault"] == "noise":
                for a0, b0 in e["params"]["spans"]:
                    span[int(a0 * SR):int(math.ceil(b0 * SR))] = True
        span_back = span[keep][:L]
        strict = ~allowed_back & ~span_back
        err = float(diff[strict].max()) if strict.any() else 0.0
        worst[p["recipe"]] = max(worst.get(p["recipe"], 0.0), err)
        assert err < 2e-4 / p["output_gain"], (fid, p["recipe"], err)  # PCM16 quantisation only
        loose = ~allowed_back & span_back
        if loose.sum() > 320:
            d2 = np.where(loose, diff, 0.0)[: (L // 320) * 320].reshape(-1, 320)
            cnt = loose[: (L // 320) * 320].reshape(-1, 320).sum(1)
            fr = np.sqrt((d2 ** 2).sum(1)[cnt == 320] / 320)
            if len(fr):
                assert fr.max() < 0.07 * 10 ** (-30 / 20) * 1.1, (fid, "unlabelled noise audible", fr.max())
        # distortion segment must not change the loudness (no level-step cue at the splice)
        dseg = [e for e in p["events"] if e["fault"] == "distortion" and e["params"].get("scope") == "segment"]
        if dseg and all(e["fault"] == "distortion" for e in p["events"]):
            a, b = int(dseg[0]["start"] * SR), int(dseg[0]["end"] * SR)
            lin = lambda v: 10 * np.log10(E.active_speech_power(v))
            step = (lin(y[a:b]) - lin(np.concatenate([y[:a], y[b:]]))) - \
                   (lin(x0[a:b]) - lin(np.concatenate([x0[:a], x0[b:]])))
            level_steps.append(abs(step))
            assert abs(step) < 1.5, (fid, dseg[0]["kind"], step)
        # every labelled span must actually contain a change (no empty labels)
        for f, iv in lab["faults"].items():
            for s, e in iv:
                if f == "unnatural_pause":
                    seg = y[int(s * SR) + 240:int(e * SR) - 240]  # skip 15 ms rounding edges
                    assert len(seg) == 0 or any(ev["fault"] != "unnatural_pause" for ev in p["events"]) or \
                        np.sqrt(np.mean(seg ** 2)) < 0.01, (fid, "pause not quiet")
    lv = np.array(levels)
    print(f"active level dBFS p5/median/p95: {np.percentile(lv,5):.1f}/{np.median(lv):.1f}/{np.percentile(lv,95):.1f}")
    assert np.percentile(lv, 95) - np.percentile(lv, 5) > 6, "level jitter missing"
    print(f"distortion-segment level step |dB|: max {max(level_steps):.2f} over {len(level_steps)} files")
    # noise = short bursts only: no full-file noise, each burst 0.5-2 s, <= 50 % of the file
    nfiles = [p for p in params.values() if any(e["fault"] == "noise" for e in p["events"])]
    for p in nfiles:
        ev = [e for e in p["events"] if e["fault"] == "noise"]
        assert all(e["kind"] in ("overlay", "insert") for e in ev), p["file_id"]
        sp = ev[0]["params"]["spans"]
        assert 1 <= len(sp) <= 2 and all(0.3 - 1e-3 <= b - a <= 1.5 + 1e-3 for a, b in sp), sp
        assert sum(b - a for a, b in sp) <= 0.5 * p["duration"] + 1e-3
        assert sum(e["end"] - e["start"] for e in ev) <= 0.5 * p["duration"] + 0.05
    modes = Counter(next(e for e in p["events"] if e["fault"] == "noise")["kind"] for p in nfiles)
    print("noise modes:", dict(modes))
    share = modes["insert"] / (modes["insert"] + modes["overlay"])   # target 25 %, small sample -> wide band
    assert 0.10 <= share <= 0.45 and modes["overlay"] > modes["insert"], f"expected ~75/25 overlay/insert, got {share:.0%}"
    # overlay bursts must START in a syllable gap (clean energy there >= 10 dB below speech level)
    starts_ok = starts_n = 0
    for p in nfiles:
        ev = next(e for e in p["events"] if e["fault"] == "noise")
        if ev["kind"] != "overlay" or any(e["fault"] == "unnatural_pause" for e in p["events"]):
            continue
        x0, _ = E.load_audio(clean / p["source_clean"])
        r = 20 * np.log10(E.frame_rms(x0, 0.01))
        ref = np.percentile(r[r > np.percentile(r, 95) - 35], 90)
        for a, _b in ev["params"]["spans"]:
            starts_n += 1
            starts_ok += r[int(a / 0.01)] <= ref - 10
    # unnatural_pause = pure break: no noise burst may overlap any inserted pause
    n_both = 0
    for p in params.values():
        ps = [(e["start"], e["end"]) for e in p["events"] if e["fault"] == "unnatural_pause"]
        nz = [e for e in p["events"] if e["fault"] == "noise"]
        if ps and nz:
            n_both += 1
            for a0, b0 in nz[0]["params"]["spans"]:
                assert all(b0 <= c or a0 >= d for c, d in ps), (p["file_id"], (a0, b0), ps)
    print(f"files with pause + noise: {n_both}, no noise inside any pause")
    print(f"overlay bursts starting in a syllable gap: {starts_ok}/{starts_n}")
    assert starts_ok == starts_n
    covers = [sum(e["end"] - e["start"] for e in p["events"] if e["fault"] == "noise") / p["duration"] for p in nfiles]
    print(f"noise files: {len(nfiles)}, labelled noise coverage mean {np.mean(covers):.0%}, max {max(covers):.0%}, "
          f"bursts/file {dict(sorted(Counter(len(next(e for e in p['events'] if e['fault']=='noise')['params']['spans']) for p in nfiles).items()))}")
    horn = [e for p in params.values() for e in p["events"] if e["fault"] == "noise" and "horn" in e["params"]["noise_file"]]
    print(f"horn noise events: {len(horn)}, mean labelled length {np.mean([e['end']-e['start'] for e in horn]):.2f}s"
          f" vs mean mixing span {np.mean([sum(b-a for a, b in e['params']['spans']) for e in horn]):.2f}s")
    sv = [e for p in params.values() for e in p["events"] if e["fault"] == "noise" and e["params"]["category"] == "street_vendor"]
    other = [e for p in params.values() for e in p["events"] if e["fault"] == "noise" and e["params"]["category"] != "street_vendor"]
    assert sv and all(10 <= e["params"]["snr_db"] <= 20 and e["params"]["distant"] for e in sv)
    assert all(0 <= e["params"]["snr_db"] <= 15 and e["params"]["distant"] is None for e in other)
    print(f"street_vendor events: {len(sv)}, SNR range {min(e['params']['snr_db'] for e in sv):.1f}-"
          f"{max(e['params']['snr_db'] for e in sv):.1f} dB, all distant-processed")
    # distortion = short segments only (reverb, optional, stays whole-file)
    dfiles = [p for p in params.values() if any(e["fault"] == "distortion" and e["kind"] != "reverb" for e in p["events"])]
    for p in dfiles:
        de = [e for e in p["events"] if e["fault"] == "distortion"]
        assert all(e["params"]["scope"] == "segment" for e in de), p["file_id"]
        assert 1 <= len(de) <= 3 and all(e["end"] - e["start"] <= 1.5 + 1e-3 for e in de), de
        assert sum(e["end"] - e["start"] for e in de) <= 0.5 * p["duration"] + 0.02
    dcov = [sum(e["end"] - e["start"] for e in p["events"] if e["fault"] == "distortion") / p["duration"] for p in dfiles]
    print(f"distortion files: {len(dfiles)}, segments/file {dict(Counter(sum(e['fault']=='distortion' for e in p['events']) for p in dfiles))}, "
          f"coverage mean {np.mean(dcov):.0%}, max {max(dcov):.0%}")
    cats = Counter(e["params"]["category"] for p in params.values() for e in p["events"] if e["fault"] == "noise")
    print("noise events per category:", dict(cats))
    assert cats["fan"] > 3 * max(cats["rain"], cats["traffic"]), "category weights not applied (fan x3, rain = traffic x0.4)"
    assert cats["rain"] < 2 * cats["traffic"] + 5 and cats["traffic"] < 2 * cats["rain"] + 5, "rain should be as rare as traffic"
    pz = [e for p in params.values() for e in p["events"] if e["fault"] == "unnatural_pause"]
    assert all(e["kind"] == "insert_silence" and 0.4 - 1e-6 <= e["params"]["len_s"] <= 0.8 + 1e-6 for e in pz)
    assert all(e["params"]["fill_rel_db"] <= -45 + 1e-6 for e in pz), "pause fill too loud"
    print(f"pause fill level re speech: max {max(e['params']['fill_rel_db'] for e in pz):.1f} dB, "
          f"sources {dict(Counter(e['params']['fill'] for e in pz))}")
    per_file = Counter(p["file_id"] for p in params.values() for e in p["events"] if e["fault"] == "unnatural_pause")
    assert max(per_file.values()) <= 2, "at most 2 stops per file"
    print(f"pause stops: {len(pz)} over {len(per_file)} files, per file {min(per_file.values())}-{max(per_file.values())}")
    cc = Counter(p["source_clean"] for p in params.values() if p["recipe"] == "clean")
    assert all(v == 1 for v in cc.values()), "more than one clean copy of the same utterance"
    print(f"clean copies: {sum(cc.values())} (max 1 per source over 5 variants)")
    print("max |out-clean| outside labels per recipe:", {k: f"{v:.1e}" for k, v in worst.items()})
    print((out / "generation_report.md").read_text()[:1500])
    print("ALL TESTS PASSED  ->", out)


if __name__ == "__main__":
    main()