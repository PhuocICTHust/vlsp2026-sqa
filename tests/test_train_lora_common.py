"""End-to-end check of scripts/train_lora_common.py on a tiny random wav2vec2 and a few fake wavs.

Covers: official split + test evaluation, output files, overwrite guard, batch guard for models
without attention mask, --resume (extension and exact continuation), utmos_nisqa head + ranking loss
with train+val training and test as dev (--select-on-test), leakage guard.
Runs on CPU in a few minutes: python -m pytest tests/test_train_lora_common.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_lora_common.py"


def write_wav(path: Path, seconds: float, rng: np.random.Generator) -> None:
    samples = (rng.standard_normal(int(16000 * seconds)) * 3000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samples.tobytes())


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lora_e2e")
    torch.manual_seed(0)
    cfg = transformers.Wav2Vec2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
        conv_dim=(16, 16), conv_kernel=(10, 3), conv_stride=(5, 2),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        feat_extract_norm="group", layerdrop=0.0, mask_time_prob=0.0,
    )
    model_dir = tmp / "tiny_wav2vec2"
    transformers.Wav2Vec2Model(cfg).save_pretrained(model_dir)

    rng = np.random.default_rng(0)
    wav_dir = tmp / "wav"
    wav_dir.mkdir()
    for split, n in [("train", 12), ("val", 5), ("test", 5)]:
        rows = []
        for i in range(n):
            fid = f"sys01tts-{split}{i:03d}.wav"
            write_wav(wav_dir / fid, rng.uniform(0.3, 0.8), rng)
            rows.append(f"{fid},{rng.integers(1, 6)}")
        (tmp / f"{split}.csv").write_text("id,Overall quality\n" + "\n".join(rows) + "\n")
    return tmp, model_dir, wav_dir


def run(setup, out, *extra):
    tmp, model_dir, wav_dir = setup
    cmd = [sys.executable, str(SCRIPT), "--backbone", "wav2vec2", "--model-name", str(model_dir),
           "--train-csv", str(tmp / "train.csv"), "--val-csv", str(tmp / "val.csv"),
           "--test-csv", str(tmp / "test.csv"), "--wav-dir", str(wav_dir),
           "--batch-size", "1", "--grad-accum", "4", "--num-workers", "0", "--patience", "0",
           "--lora-r", "4", "--lora-alpha", "8", "--output-dir", str(out), *extra]
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}")
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)


def test_train_then_extend_with_resume(setup):
    out = setup[0] / "run_a"
    r = run(setup, out, "--epochs", "2")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for name in ["best_lora.pt", "last.pt", "train.log", "run_config.json", "training_history.csv",
                 "val_predictions.csv", "test_predictions.csv", "final_metrics.json"]:
        assert (out / name).exists(), name
    first = json.loads((out / "final_metrics.json").read_text())
    assert first["epochs_run"] == 2 and "test" in first and first["resume"] is None

    # same output dir without --resume -> refused
    r = run(setup, out, "--epochs", "2")
    assert r.returncode != 0 and "already has best_lora.pt" in (r.stdout + r.stderr)

    # train 1 more epoch after the run finished
    r = run(setup, out, "--epochs", "3", "--resume")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads((out / "final_metrics.json").read_text())
    assert res["epochs_run"] == 3
    assert res["resume"]["mode"].startswith("extension")
    assert res["resume"]["from_epoch"] == 2
    archived = list(out.glob("previous_*/final_metrics.json"))
    assert len(archived) == 1
    assert len(list(out.glob("run_config_resume_*.json"))) == 1
    assert "RESUME" in (out / "train.log").read_text()

    # nothing left to do
    r = run(setup, out, "--epochs", "3", "--resume")
    assert r.returncode != 0 and "already has 3 epochs" in (r.stdout + r.stderr)


def test_exact_continuation_after_interruption(setup):
    out = setup[0] / "run_b"
    r = run(setup, out, "--epochs", "2")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    # pretend the run had been launched with --epochs 3 and was cut after epoch 2
    ckpt = torch.load(out / "last.pt", weights_only=False)
    ckpt["epochs_total"] = 3
    torch.save(ckpt, out / "last.pt")
    r = run(setup, out, "--epochs", "3", "--resume")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads((out / "final_metrics.json").read_text())
    assert res["resume"]["mode"].startswith("exact")
    assert res["epochs_run"] == 3


def test_batch_guard_without_attention_mask(setup):
    r = run(setup, setup[0] / "run_c", "--epochs", "1", "--batch-size", "2")
    assert r.returncode != 0 and "attention mask" in (r.stdout + r.stderr)


def test_utmos_head_rank_loss_train_plus_val(setup):
    """utmos_nisqa head + ranking loss, trained on train+val with test as dev (recipe B)."""
    tmp = setup[0]
    out = tmp / "run_utmos_b"
    recipe_b = ["--quality-head-type", "utmos_nisqa", "--bilstm-hidden", "8", "--w-rank", "0.1",
                "--train-csv", str(tmp / "train.csv"), str(tmp / "val.csv"), "--val-csv", str(tmp / "test.csv")]

    # epoch selection on test must be asked for explicitly
    r = run(setup, out, "--epochs", "1", *recipe_b)
    assert r.returncode != 0 and "--select-on-test" in (r.stdout + r.stderr)

    r = run(setup, out, "--epochs", "1", *recipe_b, "--select-on-test")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads((out / "final_metrics.json").read_text())
    assert res["quality_head_type"] == "utmos_nisqa" and res["selection_on_test"] is True
    ckpt = torch.load(out / "best_lora.pt", weights_only=False)
    assert ckpt["quality_head_kwargs"]["bilstm_hidden"] == 8 and ckpt["loss"]["w_rank"] == 0.1
    assert any(k.startswith("bilstm.") for k in ckpt["quality_head_state_dict"])

    # resuming with another head type is refused
    r = run(setup, out, "--epochs", "2", "--resume", "--select-on-test",
            "--train-csv", str(tmp / "train.csv"), str(tmp / "val.csv"), "--val-csv", str(tmp / "test.csv"))
    assert r.returncode != 0 and "quality_head_type" in (r.stdout + r.stderr)


def test_leakage_guard(setup):
    tmp = setup[0]
    r = run(setup, tmp / "run_leak", "--epochs", "1", "--train-csv", str(tmp / "train.csv"), str(tmp / "test.csv"))
    assert r.returncode != 0 and "also in --train-csv" in (r.stdout + r.stderr)