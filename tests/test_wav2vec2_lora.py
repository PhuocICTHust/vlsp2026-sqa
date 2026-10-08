"""Offline sanity checks for wav2vec2 + PEFT LoRA (tiny random config, no download).

Run: pytest tests/test_wav2vec2_lora.py -q
"""
from __future__ import annotations

import pytest
import torch

peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import Wav2Vec2Config, Wav2Vec2Model

from sqa.model import QualityHead
from sqa.peft_compat import disable_incompatible_torchao

disable_incompatible_torchao()

NUM_LAYERS = 2


def tiny_wav2vec2(feat_extract_norm: str = "group") -> Wav2Vec2Model:
    torch.manual_seed(0)
    cfg = Wav2Vec2Config(
        hidden_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        conv_dim=(16, 16),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=4,
        feat_extract_norm=feat_extract_norm,
        do_stable_layer_norm=(feat_extract_norm == "layer"),
        layerdrop=0.0,
        mask_time_prob=0.0,
    )
    return Wav2Vec2Model(cfg)


def wrap(model):
    cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        # No task_type: the generic PeftModel passes input_values through unchanged.
        # (task_type="FEATURE_EXTRACTION" renames the first argument to input_ids, which wav2vec2/HuBERT reject.)
    )
    return get_peft_model(model, cfg)


@pytest.mark.parametrize("norm", ["group", "layer"])
def test_lora_targets_and_gradients(norm):
    backbone = wrap(tiny_wav2vec2(norm))
    backbone.base_model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_names = [n for n, p in backbone.named_parameters() if p.requires_grad]
    # q_proj + v_proj, A + B, per layer
    assert len(lora_names) == NUM_LAYERS * 2 * 2
    assert all("lora_" in n for n in lora_names)

    head = QualityHead(hidden_size=32)
    backbone.train()
    wav = torch.randn(1, 4000)
    mask = torch.ones_like(wav, dtype=torch.long)
    hidden = backbone(wav, attention_mask=mask).last_hidden_state
    pred = head(hidden)
    loss = (pred - 3.0).pow(2).mean()
    loss.backward()

    # lora_B starts at 0 -> its grad must be non-zero; lora_A grad is 0 at step 0
    b_grads = [p.grad for n, p in backbone.named_parameters() if "lora_B" in n]
    assert all(g is not None and g.abs().sum() > 0 for g in b_grads)
    frozen = [p for n, p in backbone.named_parameters() if not p.requires_grad]
    assert all(p.grad is None for p in frozen)


def test_lora_is_identity_at_init_and_roundtrip():
    base = tiny_wav2vec2()
    base.eval()
    wav = torch.randn(1, 4000)
    with torch.no_grad():
        ref = base(wav).last_hidden_state.clone()

    backbone = wrap(base)
    backbone.eval()
    with torch.no_grad():
        out = backbone(wav).last_hidden_state
    assert torch.allclose(ref, out, atol=1e-6)  # B = 0 -> no change

    with torch.no_grad():
        for n, p in backbone.named_parameters():
            if "lora_B" in n:
                p.normal_()
        changed = backbone(wav).last_hidden_state
    assert not torch.allclose(ref, changed)

    state = {k: v.clone() for k, v in get_peft_model_state_dict(backbone).items()}
    fresh = wrap(tiny_wav2vec2())
    fresh.eval()
    set_peft_model_state_dict(fresh, state)
    with torch.no_grad():
        restored = fresh(wav).last_hidden_state
    assert torch.allclose(changed, restored, atol=1e-6)