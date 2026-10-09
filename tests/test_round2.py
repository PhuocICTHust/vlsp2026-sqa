"""Self-test for the round-2 (v3.0) mix of augment_engine.py on synthetic data.

Checks: exact recipe quotas (25/40/15/15/5), bgm + long noise cover >= 60 % of the file and are labelled
only where audible, distortion segments sit on speech, crackle leaves silence untouched, outside every
label the output equals the clean input (bgm / long noise: below the audibility threshold), test-set
naming (--id-prefix / --split).   Run:  python tests/test_round2.py [real_noise_bank_dir]
"""
import json, math, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "scripts" / "augment_engine.py"
sys.path.insert(0, str(ENGINE.parent))
import augment_engine as E  # noqa: E402
sys.path.insert(0, str(HERE))
from test_augment_engine import fake_speech, w16, SR  # noqa: E402

assert E.ENGINE_VERSION == "vi_aug_engine_v3.1", E.ENGINE_VERSION
rng = np.random.default_rng(1)


def fake_bank(d: Path):
    t = np.arange(int(15 * SR)) / SR
    piano = sum(np.sin(2 * np.pi * f * t) * np.exp(-3 * ((t * 2) % 1)) for f in (262, 330, 392)) * 0.001
    beat = np.sign(np.sin(2 * np.pi * 2 * t)) * np.exp(-20 * ((t * 2) % 1)) * 0.5 + 0.1 * np.sin(2 * np.pi * 110 * t)
    w16(d / "bgm_piano_01.wav", piano)                      # very quiet like the real file (-62 dBFS)
    w16(d / "bgm_electronic_beat_01.wav", beat)
    w16(d / "noise_ceiling_fan_01.wav", np.convolve(rng.standard_normal(len(t)), np.ones(30) / 30, "same") * 0.3)
    w16(d / "noise_cicada_01.wav", np.sin(2 * np.pi * 4500 * t) * (rng.random(len(t)) < 0.6) * 0.003)
    w16(d / "noise_electrical_hum_01.wav", 0.4 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t))
    k = np.zeros(len(t)); k[(rng.random(len(t)) < 0.0008)] = 1
    w16(d / "noise_keyboard_01.wav", np.convolve(k, np.hanning(80), "same") * 0.5)


def main():
    root = Path(tempfile.mkdtemp())
    clean, out, outt = root / "clean", root / "train", root / "test"
    for i in range(40):
        w16(clean / f"vi_clean_{i:03d}.wav", fake_speech(rng.uniform(2.5, 9.5)))
    if len(sys.argv) > 1:
        bank = Path(sys.argv[1])
    else:
        bank = root / "bank"
        fake_bank(bank)
    py = sys.executable
    subprocess.run([py, str(ENGINE), "generate", "--clean-dir", str(clean), "--noise-dir", str(bank),
                    "--out-dir", str(out), "--seed", "11"], check=True)
    labels = {l["file_id"]: l for l in map(json.loads, open(out / "labels.jsonl"))}
    params = {p["file_id"]: p for p in map(json.loads, open(out / "params.jsonl"))}
    rc = Counter(p["recipe"] for p in params.values())
    assert rc == Counter({"distortion": 16, "clean": 10, "bgm": 6, "noise_long": 6, "pause": 2}), rc
    print("recipes (exact quotas on 40 files):", dict(rc))
    thr_rms = 0.07 * 10 ** (-30 / 20) * 1.1
    covm = []
    for fid, p in params.items():
        lab = labels[fid]
        sr, y = wavfile.read(out / "wav" / fid)
        y = y.astype(np.float32) / 32768 / p["output_gain"]
        x0, _ = E.load_audio(clean / p["source_clean"])
        x0 = x0 * (0.07 / math.sqrt(E.active_speech_power(x0)))
        r = p["recipe"]
        if r == "clean":
            assert all(not v for v in lab["faults"].values()) and np.abs(y - x0).max() < 3e-4 / p["output_gain"]
            continue
        if r == "pause":
            assert lab["faults"]["unnatural_pause"] and not lab["faults"]["noise"] and not lab["faults"]["distortion"]
            continue
        assert len(y) == len(x0)
        allowed = np.zeros(len(y), bool)
        for iv in lab["faults"].values():
            for s, e in iv:
                allowed[int(s * SR):int(math.ceil(e * SR))] = True
        d = np.where(allowed, 0.0, np.abs(y - x0))
        if r == "distortion":
            assert lab["faults"]["distortion"] and not lab["faults"]["noise"], fid
            assert d.max() < 3e-4 / p["output_gain"], (fid, "distortion leaked outside its label", d.max())
            act = E.active_mask(x0)
            ev = [e for e in p["events"] if e["fault"] == "distortion"]
            assert 1 <= len(ev) <= 3 and all(0.2 - 1e-6 <= e["end"] - e["start"] <= 1.5 + 1e-6 for e in ev), ev
            for e in ev:
                a, b = int(e["start"] / 0.02), int(math.ceil(e["end"] / 0.02))
                assert act[a:b].mean() >= 0.6, (fid, e["kind"], "distortion not on speech", act[a:b].mean())
        else:                                  # bgm / noise_long: unlabelled part must be inaudible
            assert lab["faults"]["noise"] and not lab["faults"]["distortion"], fid
            ev = [e for e in p["events"] if e["fault"] == "noise"]
            assert ev[0]["kind"] == ("bgm" if r == "bgm" else "long")
            pr = ev[0]["params"]
            if pr["coverage"] == "whole":
                assert pr["cover"] == 1.0
            elif pr["coverage"] == "long":
                assert 0.6 - 1e-6 <= pr["cover"] <= 0.9 + 1e-6
            else:
                assert 1 <= pr["n_segments"] <= 3 and pr["cover"] <= 0.6 + 1e-3
                lo_, hi_ = (0.4, 0.6) if pr["category"].startswith("keyboard") else (0.5, 4.0)
                assert all(lo_ - 1e-3 <= b - a <= hi_ + 1e-3 for a, b in pr["spans"]) or pr["n_segments"] == 1, pr["spans"]
            if pr["category"].startswith("keyboard"):
                assert pr["coverage"] == "partial" and pr["n_segments"] <= 2, pr
            covm.append(pr["coverage"])
            n = len(d) // 320 * 320
            fr = np.sqrt((d[:n].reshape(-1, 320) ** 2).mean(1))
            assert fr.max() < thr_rms, (fid, "audible noise outside the label", fr.max())
            if r == "bgm":
                assert ev[0]["params"]["category"].startswith("bgm_")
            else:
                assert not ev[0]["params"]["category"].startswith("bgm_")
    dk = Counter(e["kind"] for p in params.values() for e in p["events"] if e["fault"] == "distortion")
    print("distortion kinds:", dict(dk))
    cov = [sum(e["end"] - e["start"] for e in p["events"] if e["fault"] == "noise") / p["duration"]
           for p in params.values() if p["recipe"] in ("bgm", "noise_long")]
    print(f"bgm / long noise: labelled coverage mean {np.mean(cov):.0%}, min {min(cov):.0%}; modes {dict(Counter(covm))}")
    # crackle on its own: silence untouched, speech changed
    x = fake_speech(4.0)
    yc, _ = E.fx_crackle(x, np.random.default_rng(3))
    quiet = np.sqrt(np.convolve(x ** 2, np.ones(480) / 480, "same")) < 1e-3     # 30 ms windows of silence
    assert np.abs(yc - x)[quiet].max() < 5e-3 and np.abs(yc - x)[~quiet].max() > 0.05
    # test-set naming
    subprocess.run([py, str(ENGINE), "generate", "--clean-dir", str(clean), "--noise-dir", str(bank),
                    "--out-dir", str(outt), "--seed", "2026", "--limit", "12", "--id-prefix", "vi_test_v1",
                    "--split", "test"], check=True)
    meta = (outt / "metadata.csv").read_text().splitlines()
    assert all(l.startswith("vi_test_v1_") for l in meta[1:]) and ",test," in meta[1]
    print("ROUND-2 TESTS PASSED ->", out)


if __name__ == "__main__":
    main()