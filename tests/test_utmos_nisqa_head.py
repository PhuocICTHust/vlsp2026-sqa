from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from sqa.model import UtmosNisqaQualityHead, QualityHead, SQAModel


def test_utmos_nisqa_head_init():
    head = UtmosNisqaQualityHead(
        hidden_size=64,
        bilstm_hidden=32,
        mlp_hidden=64,
        dropout=0.1,
        use_fault_features=True,
        n_fault_types=3,
    )
    assert head.hidden_size == 64
    assert head.bilstm_hidden == 32
    assert head.bilstm.input_size == 64
    assert head.bilstm.hidden_size == 32
    assert head.attn_proj.in_features == 64  # 32 * 2 (bidirectional)
    # pooled dim = 64 * 2 (mean + std) + 6 (mean + max of 3 faults) = 134
    assert head.mlp[0].in_features == 134


def test_utmos_nisqa_head_forward_basic():
    head = UtmosNisqaQualityHead(
        hidden_size=64,
        bilstm_hidden=32,
        mlp_hidden=64,
        use_fault_features=False,
    )
    B, T, D = 4, 25, 64
    h = torch.randn(B, T, D)
    out = head(h)

    assert out.shape == (B,)
    assert torch.all(out >= 1.0) and torch.all(out <= 5.0)


def test_utmos_nisqa_head_forward_with_mask():
    head = UtmosNisqaQualityHead(
        hidden_size=64,
        bilstm_hidden=32,
        mlp_hidden=64,
        use_fault_features=False,
    )
    B, T, D = 3, 30, 64
    h = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 15:] = False
    mask[1, 20:] = False

    out = head(h, valid_mask=mask)
    assert out.shape == (B,)
    assert torch.all(out >= 1.0) and torch.all(out <= 5.0)


def test_utmos_nisqa_head_forward_with_fault_features():
    head = UtmosNisqaQualityHead(
        hidden_size=64,
        bilstm_hidden=32,
        mlp_hidden=64,
        use_fault_features=True,
        n_fault_types=3,
    )
    B, T, D = 2, 20, 64
    h = torch.randn(B, T, D)
    fault_probs = torch.sigmoid(torch.randn(B, T, 3))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 15:] = False

    # With fault_probs provided
    out_with_faults = head(h, valid_mask=mask, fault_probs=fault_probs)
    assert out_with_faults.shape == (B,)
    assert torch.all(out_with_faults >= 1.0) and torch.all(out_with_faults <= 5.0)

    # Without fault_probs (fallback branch)
    out_fallback = head(h, valid_mask=mask, fault_probs=None)
    assert out_fallback.shape == (B,)
    assert torch.all(out_fallback >= 1.0) and torch.all(out_fallback <= 5.0)


def test_utmos_nisqa_head_backward():
    head = UtmosNisqaQualityHead(
        hidden_size=32,
        bilstm_hidden=16,
        mlp_hidden=32,
        use_fault_features=True,
    )
    h = torch.randn(2, 10, 32, requires_grad=True)
    fault_probs = torch.sigmoid(torch.randn(2, 10, 3))
    out = head(h, fault_probs=fault_probs)
    loss = out.sum()
    loss.backward()

    assert h.grad is not None
    assert head.bilstm.weight_ih_l0.grad is not None
    assert head.attn_proj.weight.grad is not None
    assert head.mlp[0].weight.grad is not None


def test_sqa_model_integration():
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 32
        def forward(self, wav, attention_mask=None):
            B = wav.shape[0]
            return torch.randn(B, 15, self.hidden_size)
        def frame_valid_mask(self, attention_mask):
            return attention_mask.bool()

    dummy_backbone = DummyBackbone()
    model = SQAModel(
        backbone=dummy_backbone,
        quality_head_type="utmos_nisqa",
        quality_head_kwargs={"bilstm_hidden": 16, "use_fault_features": True},
        fault_head_kwargs={"input_dim": 32, "n_faults": 3, "temporal_type": "bilstm", "temporal_dim": 16},
    )

    wav = torch.randn(2, 16000)
    attn_mask = torch.ones(2, 15, dtype=torch.bool)
    quality, fault_probs, h = model(wav, attention_mask=attn_mask)

    assert quality.shape == (2,)
    assert fault_probs.shape == (2, 15, 3)
    assert torch.all(quality >= 1.0) and torch.all(quality <= 5.0)
    assert torch.all(fault_probs >= 0.0) and torch.all(fault_probs <= 1.0)


def test_utmos_nisqa_head_padding_invariant():
    """A clip scores the same alone and inside a zero-padded batch (packed BiLSTM)."""
    torch.manual_seed(0)
    head = UtmosNisqaQualityHead(hidden_size=16, bilstm_hidden=8, mlp_hidden=16, use_fault_features=False).eval()
    short, long_ = torch.randn(1, 12, 16), torch.randn(1, 20, 16)
    batch = torch.zeros(2, 20, 16)
    batch[0, :12], batch[1] = short[0], long_[0]
    batch[0, 12:] = 7.0                      # garbage in the padding must not matter
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[0, :12] = True
    mask[1] = True
    with torch.no_grad():
        together = head(batch, valid_mask=mask)
        alone = torch.stack([head(short)[0], head(long_)[0]])
    assert torch.allclose(together, alone, atol=1e-5), (together, alone)