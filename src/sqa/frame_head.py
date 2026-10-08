"""Fault frame head: h [B,T,H] -> conv blocks -> logits [B,T,C]."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

FAULT_TYPES = ("noise", "distortion", "unnatural_pause")
PAD_LOGIT = -30.0


def _check_valid_mask(valid_mask: torch.Tensor | None, b: int, t: int, device) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(b, t, dtype=torch.bool, device=device)
    if valid_mask.shape != (b, t):
        raise ValueError(f"valid_mask must have shape [B, T]=({b}, {t}), received {tuple(valid_mask.shape)}")
    if valid_mask.dtype != torch.bool:
        if not torch.all((valid_mask == 0) | (valid_mask == 1)):
            raise ValueError("valid_mask must be binary (bool or 0/1), not continuous weights")
        valid_mask = valid_mask.bool()
    return valid_mask.to(device)


class LayerWeightedSum(nn.Module):
    """[B, L, T, H] -> [B, T, H]."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 4 or h.shape[1] != self.num_layers:
            raise ValueError(f"need [B, L={self.num_layers}, T, H], received {tuple(h.shape)}")
        w = torch.softmax(self.logits, dim=0).to(h.dtype)
        return torch.einsum("l,bltd->btd", w, h)

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits.detach(), dim=0)


class _ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        y = self.norm(x) * mask_f  # padding = 0 trước conv
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        y = self.drop(F.gelu(y))
        return (x + y) * mask_f


class FaultFrameHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_faults: int = len(FAULT_TYPES),
        hidden_dim: int = 256,
        n_blocks: int = 2,
        kernel_size: int = 5,
        dilations: Sequence[int] | None = None,
        dropout: float = 0.1,
        num_input_layers: int = 1,
        prior_prob: float = 0.05,
        use_bi_gru: bool = False,
        gru_hidden_dim: int = 128,
        temporal_type: str = "none",
        temporal_dim: int = 128,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, received {kernel_size}")
        if not 0.0 < prior_prob < 1.0:
            raise ValueError(f"prior_prob must in range (0, 1), received {prior_prob}")
        dilations = tuple(dilations) if dilations is not None else (1,) * n_blocks
        if len(dilations) != n_blocks:
            raise ValueError(f"need {n_blocks} dilation, received {dilations}")

        self.input_dim = input_dim
        self.n_faults = n_faults
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.layer_sum = LayerWeightedSum(num_input_layers) if num_input_layers > 1 else None

        self.in_norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(_ConvBlock(hidden_dim, kernel_size, d, dropout) for d in dilations)

        # Support temporal_type: "bilstm" | "bigru" | "none"
        if temporal_type == "none" and use_bi_gru:
            temporal_type = "bigru"
            temporal_dim = gru_hidden_dim

        self.temporal_type = temporal_type
        self.temporal_dim = temporal_dim

        if temporal_type == "bilstm":
            self.temporal = nn.LSTM(
                hidden_dim,
                temporal_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.temporal_proj = nn.Linear(temporal_dim * 2, hidden_dim)
            self.temporal_norm = nn.LayerNorm(hidden_dim)
        elif temporal_type == "bigru":
            self.temporal = nn.GRU(
                hidden_dim,
                temporal_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.temporal_proj = nn.Linear(temporal_dim * 2, hidden_dim)
            self.temporal_norm = nn.LayerNorm(hidden_dim)
        else:
            self.temporal = None

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, n_faults)

        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.constant_(self.out_proj.bias, math.log(prior_prob / (1.0 - prior_prob)))

    @property
    def receptive_field_frames(self) -> int:
        return 1 + sum(d * (self.kernel_size - 1) for d in self.dilations)

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits; frame padding = PAD_LOGIT."""
        if self.layer_sum is not None:
            h = self.layer_sum(h)
        elif h.dim() != 3:
            raise ValueError(f"need [B, T, H] (num_input_layers=1), received {tuple(h.shape)}")
        if h.shape[-1] != self.input_dim:
            raise ValueError(f"hidden size {h.shape[-1]} != input_dim {self.input_dim}")

        b, t, _ = h.shape
        mask = _check_valid_mask(valid_mask, b, t, h.device)
        mask_f = mask.unsqueeze(-1).to(h.dtype)

        x = self.in_proj(self.in_norm(h)) * mask_f
        for block in self.blocks:
            x = block(x, mask_f)

        if self.temporal is not None:
            rnn_out, _ = self.temporal(x)
            x = self.temporal_norm((x + self.temporal_proj(rnn_out)) * mask_f) * mask_f

        logits = self.out_proj(self.out_norm(x))
        return logits.masked_fill(~mask.unsqueeze(-1), PAD_LOGIT)

    @torch.no_grad()
    def predict_proba(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(h, valid_mask))


def topk_presence(
    frame_probs: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    k_ratio: float = 0.05,
    min_k: int = 3,
) -> torch.Tensor:
    """Bản torch của postprocess.presence_from_frames: [B,T,C] -> [B,C]."""
    if frame_probs.dim() != 3:
        raise ValueError(f"need [B, T, C], received {tuple(frame_probs.shape)}")
    b, t, _ = frame_probs.shape
    mask = _check_valid_mask(valid_mask, b, t, frame_probs.device)

    lengths = mask.sum(dim=1).to(torch.float64)
    k = torch.minimum(lengths, torch.clamp(torch.round(lengths * k_ratio), min=min_k))
    sorted_probs, _ = torch.sort(frame_probs.masked_fill(~mask.unsqueeze(-1), -1.0), dim=1, descending=True)
    rank = torch.arange(t, device=frame_probs.device).unsqueeze(0)
    take = (rank < k.to(rank.device).unsqueeze(1)).unsqueeze(-1).to(frame_probs.dtype)
    denom = k.clamp(min=1).to(frame_probs.dtype).unsqueeze(-1)
    return (sorted_probs * take).sum(dim=1) / denom