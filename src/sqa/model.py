from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


class WavLMBackbone(nn.Module):

    def __init__(self, model_name: str = "microsoft/wavlm-large", freeze: bool = True):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze = freeze

    def forward(self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.backbone(wav, attention_mask=attention_mask)
        return out.last_hidden_state

    @torch.no_grad()
    def output_length(self, num_samples: int) -> int:
        return int(self.backbone._get_feat_extract_output_lengths(torch.tensor([num_samples])).item())
    @torch.no_grad()
    def frame_valid_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must have shape [B, N], got {attention_mask.shape}"
            )

        lengths = attention_mask.long().sum(dim=1)

        frame_lengths = self.backbone._get_feat_extract_output_lengths(
            lengths
        ).to(attention_mask.device)

        max_frames = int(
            self.backbone._get_feat_extract_output_lengths(
                torch.tensor(
                    [attention_mask.shape[1]],
                    device=attention_mask.device,
                )
            ).item()
        )

        frame_idx = torch.arange(
            max_frames,
            device=attention_mask.device,
        ).unsqueeze(0)

        return frame_idx < frame_lengths.unsqueeze(1)

class AttentionPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.proj(h).squeeze(-1)  # [B,T]
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B,T,1]
        return (weights * h).sum(dim=1)  # [B,H]


class QualityHead(nn.Module):

    def __init__(self, hidden_size: int, mlp_hidden: int = 256):
        super().__init__()
        self.pool = AttentionPool(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.pool(h, valid_mask)  # [B,H]
        raw = self.mlp(pooled).squeeze(-1)  # [B]
        return torch.sigmoid(raw) * 4.0 + 1.0  # scale [1,5]


def concordance_correlation_coefficient(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    if pred.numel() != gt.numel():
        raise ValueError(
            f"pred and gt must have the same number of elements, "
            f"got {pred.numel()} and {gt.numel()}"
        )

    if pred.numel() < 2:
        return pred.new_zeros(())

    pred_mean = pred.mean()
    gt_mean = gt.mean()

    pred_var = pred.var(unbiased=False)
    gt_var = gt.var(unbiased=False)

    covariance = (
        (pred - pred_mean) * (gt - gt_mean)
    ).mean()

    denominator = (
        pred_var
        + gt_var
        + (pred_mean - gt_mean).pow(2)
    )

    return (2.0 * covariance) / (denominator + 1e-8)


def quality_loss(pred: torch.Tensor, gt: torch.Tensor, w_mse: float = 0.5, w_ccc: float = 0.5) -> torch.Tensor:
    mse = F.mse_loss(pred, gt)
    ccc = concordance_correlation_coefficient(pred, gt)
    return w_mse * mse + w_ccc * (1 - ccc)


class StubFaultHead(nn.Module):

    def __init__(self, hidden_size: int, n_fault_types: int = 3):
        super().__init__()
        self.proj = nn.Linear(hidden_size, n_fault_types)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.proj(h)  # [B,T,3] — random init, CHƯA train
        return torch.sigmoid(logits)


class SQAModel(nn.Module):
    def __init__(self, model_name: str = "microsoft/wavlm-large", freeze_backbone: bool = True):
        super().__init__()
        self.backbone = WavLMBackbone(model_name, freeze=freeze_backbone)
        self.quality_head = QualityHead(self.backbone.hidden_size)
        self.fault_head = StubFaultHead(self.backbone.hidden_size)  # TODO: thay bằng head thật của Thảo

    def forward(
            self,
            wav: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
    ):
        h = self.backbone(wav, attention_mask)  # [B,T,H]

        if attention_mask is not None:
            valid_mask = self.backbone.frame_valid_mask(attention_mask)
        else:
            valid_mask = None

        quality = self.quality_head(h, valid_mask)  # [B]
        fault_probs = self.fault_head(h)  # [B,T,3]

        return quality, fault_probs, h
