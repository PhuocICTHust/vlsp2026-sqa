from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel

from .frame_head import FaultFrameHead
class WavLMBackbone(nn.Module):

    def __init__(self, model_name: str = "microsoft/wavlm-large", freeze: bool = True):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(model_name, use_safetensors=True)
        self.hidden_size = self.backbone.config.hidden_size
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze = freeze
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self
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

    def __init__(self, hidden_size: int, mlp_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.pool = AttentionPool(hidden_size)
        layers = [nn.Linear(hidden_size, mlp_hidden)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(mlp_hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.pool(h, valid_mask)  # [B,H]
        raw = self.mlp(pooled).squeeze(-1)  # [B]
        return torch.sigmoid(raw) * 4.0 + 1.0  # scale [1,5]

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        adapted = dict(state_dict)
        if "mlp.3.weight" in adapted and "mlp.2.weight" not in adapted and len(self.mlp) == 3:
            adapted["mlp.2.weight"] = adapted.pop("mlp.3.weight")
            adapted["mlp.2.bias"] = adapted.pop("mlp.3.bias")
        elif "mlp.2.weight" in adapted and "mlp.3.weight" not in adapted and len(self.mlp) == 4:
            adapted["mlp.3.weight"] = adapted.pop("mlp.2.weight")
            adapted["mlp.3.bias"] = adapted.pop("mlp.2.bias")
        return super().load_state_dict(adapted, strict=strict)


class UtmosNisqaQualityHead(nn.Module):
    """
    Advanced Speech Quality Assessment Head inspired by UTMOS and NISQA:
    - UTMOS: BiLSTM temporal contextualization across speech frames.
    - UTMOS / X-Vector: Statistics Pooling (Attention-weighted Mean + Standard Deviation).
    - NISQA: Multi-dimensional fault awareness (integrating frame fault probabilities).
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 1,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        use_fault_features: bool = True,
        n_fault_types: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bilstm_hidden = bilstm_hidden
        self.use_fault_features = use_fault_features
        self.n_fault_types = n_fault_types

        # 1. UTMOS-style Temporal BiLSTM Context
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = bilstm_hidden * 2  # 256

        # 2. Attention projection for frame weighting
        self.attn_proj = nn.Linear(lstm_out_dim, 1)

        # 3. Statistics pooling (Mean + Std -> 2 * lstm_out_dim)
        pooled_dim = lstm_out_dim * 2  # 512
        if use_fault_features:
            # Fault statistics: [mean, max] per fault type -> 2 * n_fault_types (6)
            pooled_dim += 2 * n_fault_types

        # 4. Final Quality Regression MLP
        layers = [
            nn.Linear(pooled_dim, mlp_hidden),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        ])
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        fault_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        h: [B, T, hidden_size]
        valid_mask: [B, T] bool
        fault_probs: [B, T, n_fault_types] optional (probabilities from FaultFrameHead)
        """
        # Step 1: BiLSTM temporal contextualization
        lstm_out, _ = self.bilstm(h)  # [B, T, lstm_out_dim]

        # Step 2: Attention weighting
        scores = self.attn_proj(lstm_out).squeeze(-1)  # [B, T]
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]

        # Step 3: Mean & Std statistics pooling
        mean = (weights * lstm_out).sum(dim=1)  # [B, lstm_out_dim]
        diff_sq = (lstm_out - mean.unsqueeze(1)).pow(2)
        var = (weights * diff_sq).sum(dim=1)  # [B, lstm_out_dim]
        std = torch.sqrt(var.clamp_min(1e-7))
        pooled = torch.cat([mean, std], dim=-1)  # [B, lstm_out_dim * 2]

        # Step 4: NISQA fault statistics injection
        if self.use_fault_features:
            if fault_probs is not None:
                if valid_mask is not None:
                    mask_f = valid_mask.unsqueeze(-1).float()
                    f_mean = (fault_probs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
                    f_masked = fault_probs.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
                    f_max = f_masked.max(dim=1)[0].clamp_min(0.0)
                else:
                    f_mean = fault_probs.mean(dim=1)
                    f_max = fault_probs.max(dim=1)[0]
                fault_feats = torch.cat([f_mean, f_max], dim=-1)  # [B, 2 * n_fault_types]
            else:
                fault_feats = h.new_zeros(h.shape[0], 2 * self.n_fault_types)
            pooled = torch.cat([pooled, fault_feats], dim=-1)

        # Step 5: Bounded MOS regression [1.0, 5.0]
        raw = self.mlp(pooled).squeeze(-1)  # [B]
        return torch.sigmoid(raw) * 4.0 + 1.0


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


def pearson_correlation(
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

    pred_c = pred - pred.mean()
    gt_c = gt - gt.mean()
    denom = torch.sqrt(
        (pred_c.square().sum() * gt_c.square().sum()).clamp_min(1e-12)
    )
    return pred_c.mul(gt_c).sum() / denom


def normalize_waveform(wav: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Per-utterance zero-mean unit-variance normalization on the valid audio portion.
    wav: [B, T] or [T]
    attention_mask: [B, T] (1 for valid audio, 0 for padding)
    """
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
        squeezed = True
    else:
        squeezed = False

    if attention_mask is None:
        mean = wav.mean(dim=-1, keepdim=True)
        var = wav.var(dim=-1, keepdim=True, unbiased=False)
        norm_wav = (wav - mean) / torch.sqrt(var + 1e-7)
    else:
        mask = attention_mask.float()
        lengths = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (wav * mask).sum(dim=-1, keepdim=True) / lengths
        var = (((wav - mean) * mask) ** 2).sum(dim=-1, keepdim=True) / lengths
        norm_wav = ((wav - mean) / torch.sqrt(var + 1e-7)) * mask

    return norm_wav.squeeze(0) if squeezed else norm_wav


def pairwise_ranking_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    margin: float = 0.1,
    min_diff: float = 0.0,
) -> torch.Tensor:
    """
    Pairwise margin ranking loss to align predicted ranking with ground truth.
    Penalizes pairs (i, j) where gt[i] > gt[j] + min_diff but pred[i] - pred[j] < margin.
    """
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    if pred.numel() < 2:
        return pred.new_zeros(())

    diff_gt = gt.unsqueeze(1) - gt.unsqueeze(0)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)

    valid_mask = diff_gt > min_diff
    if not valid_mask.any():
        return pred.new_zeros(())

    pair_losses = F.relu(margin - diff_pred[valid_mask])
    return pair_losses.mean()


def quality_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    w_mse: float = 0.4,
    w_ccc: float = 0.3,
    w_pcc: float = 0.3,
    w_rank: float = 0.0,
    rank_margin: float = 0.1,
) -> torch.Tensor:
    mse = F.mse_loss(pred, gt)
    loss = w_mse * mse
    if w_ccc > 0.0:
        ccc = concordance_correlation_coefficient(pred, gt)
        loss = loss + w_ccc * (1.0 - ccc)
    if w_pcc > 0.0:
        pcc = pearson_correlation(pred, gt)
        loss = loss + w_pcc * (1.0 - pcc)
    if w_rank > 0.0:
        rank = pairwise_ranking_loss(pred, gt, margin=rank_margin)
        loss = loss + w_rank * rank
    return loss



# class StubFaultHead(nn.Module):

#     def __init__(self, hidden_size: int, n_fault_types: int = 3):
#         super().__init__()
#         self.proj = nn.Linear(hidden_size, n_fault_types)

#     def forward(self, h: torch.Tensor) -> torch.Tensor:
#         logits = self.proj(h)  # [B,T,3] — random init, CHƯA train
#         return torch.sigmoid(logits)


class SQAModel(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze_backbone: bool = True,
        fault_head_kwargs: dict | None = None,
        quality_head_kwargs: dict | None = None,
        quality_head_type: str = "standard",
        backbone: nn.Module | None = None,
        normalize_wav: bool = False,
    ):
        super().__init__()
        self.normalize_wav = normalize_wav
        self.quality_head_type = quality_head_type
        self.backbone = backbone if backbone is not None else WavLMBackbone(model_name, freeze=freeze_backbone)

        q_kwargs = dict(quality_head_kwargs) if quality_head_kwargs else {}
        if quality_head_type == "utmos_nisqa":
            self.quality_head = UtmosNisqaQualityHead(self.backbone.hidden_size, **q_kwargs)
        else:
            self.quality_head = QualityHead(self.backbone.hidden_size, **q_kwargs)

        f_kwargs = dict(fault_head_kwargs) if fault_head_kwargs else {}
        input_dim = f_kwargs.pop("input_dim", self.backbone.hidden_size)
        self.fault_head = FaultFrameHead(input_dim, **f_kwargs)

    def forward(
            self,
            wav: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            return_fault_logits: bool = False,
    ):
        if self.normalize_wav:
            wav = normalize_waveform(wav, attention_mask)
        h = self.backbone(wav, attention_mask)  # [B,T,H]

        if attention_mask is not None:
            valid_mask = self.backbone.frame_valid_mask(attention_mask)
            if valid_mask.shape != h.shape[:2]:
                raise ValueError(f"frame_valid_mask {tuple(valid_mask.shape)} lệch hidden {tuple(h.shape[:2])}")
        else:
            valid_mask = None

        fault_logits = self.fault_head(h, valid_mask)  # [B,T,F]
        fault_probs = torch.sigmoid(fault_logits)

        if isinstance(self.quality_head, UtmosNisqaQualityHead):
            quality = self.quality_head(h, valid_mask, fault_probs=fault_probs)
        else:
            quality = self.quality_head(h, valid_mask)

        fault_out = fault_logits if return_fault_logits else fault_probs
        return quality, fault_out, h