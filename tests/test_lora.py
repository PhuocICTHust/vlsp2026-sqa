from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from sqa.lora import (
    ManualLoRA,
    extract_lora_state_dict,
    load_lora_state_dict,
)


def test_manual_lora_init():
    lora = ManualLoRA(in_features=64, out_features=128, r=4, alpha=8.0)
    assert lora.scaling == 2.0
    assert lora.lora_A.shape == (4, 64)
    assert lora.lora_B.shape == (128, 4)

    # Initial delta_weight must be zero because B is zero-initialized
    delta = lora.delta_weight()
    assert delta.shape == (128, 64)
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_manual_lora_delta_computation():
    lora = ManualLoRA(in_features=2, out_features=2, r=1, alpha=2.0)
    with torch.no_grad():
        lora.lora_A.copy_(torch.tensor([[1.0, 2.0]]))
        lora.lora_B.copy_(torch.tensor([[3.0], [4.0]]))

    # scaling = alpha / r = 2.0 / 1 = 2.0
    # B @ A = [[3*1, 3*2], [4*1, 4*2]] = [[3, 6], [4, 8]]
    # delta = 2.0 * [[3, 6], [4, 8]] = [[6, 12], [8, 16]]
    expected = torch.tensor([[6.0, 12.0], [8.0, 16.0]])
    assert torch.allclose(lora.delta_weight(), expected)


def test_manual_lora_invalid_rank():
    with pytest.raises(ValueError, match="LoRA rank must be > 0"):
        ManualLoRA(in_features=64, out_features=64, r=0)


def test_extract_and_load_lora_state_dict():
    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.manual_lora_q = ManualLoRA(16, 16, r=2, alpha=4.0)
            self.manual_lora_v = ManualLoRA(16, 16, r=2, alpha=4.0)

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = DummyAttention()

    m1 = DummyBackbone()
    # Change weights
    with torch.no_grad():
        m1.attn.manual_lora_q.lora_B.fill_(1.5)
        m1.attn.manual_lora_v.lora_B.fill_(2.5)

    extracted = extract_lora_state_dict(m1)
    assert "attn.manual_lora_q.lora_B" in extracted
    assert "attn.manual_lora_v.lora_B" in extracted

    m2 = DummyBackbone()
    assert not torch.allclose(m2.attn.manual_lora_q.lora_B, m1.attn.manual_lora_q.lora_B)

    load_lora_state_dict(m2, extracted)
    assert torch.allclose(m2.attn.manual_lora_q.lora_B, m1.attn.manual_lora_q.lora_B)
    assert torch.allclose(m2.attn.manual_lora_v.lora_B, m1.attn.manual_lora_v.lora_B)
