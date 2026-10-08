from __future__ import annotations

import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualLoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer:
    W' = W + (alpha / r) * (B @ A)
    A: [r, in_features]
    B: [out_features, r]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0.")

        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(r)

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)


def manual_lora_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.Tensor,
    gated_position_bias: torch.FloatTensor,
):
    """
    Forward method replacement for WavLMAttention to inject LoRA weights
    into q_proj and v_proj directly in F.multi_head_attention_forward.
    """
    query = key = value = hidden_states.transpose(0, 1)
    key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None

    bias_k = None
    bias_v = None
    add_zero_attn = False

    q_weight = self.q_proj.weight + self.manual_lora_q.delta_weight()
    k_weight = self.k_proj.weight
    v_weight = self.v_proj.weight + self.manual_lora_v.delta_weight()

    attn_output, attn_weights = F.multi_head_attention_forward(
        query,
        key,
        value,
        self.embed_dim,
        self.num_heads,
        torch.empty([0], device=hidden_states.device, dtype=hidden_states.dtype),
        torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
        bias_k,
        bias_v,
        add_zero_attn,
        self.dropout,
        self.out_proj.weight,
        self.out_proj.bias,
        self.training,
        key_padding_mask,
        need_weights=False,
        attn_mask=gated_position_bias,
        use_separate_proj_weight=True,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
    )

    attn_output = attn_output.transpose(0, 1)
    if attn_weights is not None:
        attn_weights = attn_weights[:, None].broadcast_to(
            attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
        )
    return attn_output, attn_weights


def attach_manual_lora(backbone, r: int = 8, alpha: float = 16.0) -> list[str]:
    """
    Attaches manual LoRA modules (manual_lora_q and manual_lora_v) to all
    encoder attention layers of a WavLM model.
    """
    layers = backbone.encoder.layers
    target_names = []
    for layer_idx, layer in enumerate(layers):
        attention = layer.attention
        q_proj = attention.q_proj
        v_proj = attention.v_proj

        q_proj.weight.requires_grad = False
        if q_proj.bias is not None:
            q_proj.bias.requires_grad = False

        v_proj.weight.requires_grad = False
        if v_proj.bias is not None:
            v_proj.bias.requires_grad = False

        q_lora = ManualLoRA(
            in_features=q_proj.in_features,
            out_features=q_proj.out_features,
            r=r,
            alpha=alpha,
        ).to(
            device=q_proj.weight.device,
            dtype=q_proj.weight.dtype,
        )

        v_lora = ManualLoRA(
            in_features=v_proj.in_features,
            out_features=v_proj.out_features,
            r=r,
            alpha=alpha,
        ).to(
            device=v_proj.weight.device,
            dtype=v_proj.weight.dtype,
        )

        attention.add_module("manual_lora_q", q_lora)
        attention.add_module("manual_lora_v", v_lora)

        target_names.extend(
            [
                f"encoder.layers.{layer_idx}.attention.q_proj",
                f"encoder.layers.{layer_idx}.attention.v_proj",
            ]
        )
    return target_names


def patch_wavlm_attention(backbone) -> int:
    """
    Patches torch_multi_head_self_attention on all encoder layers to use
    manual_lora_attention_forward.
    """
    patched = 0
    for layer in backbone.encoder.layers:
        attention = layer.attention
        if not hasattr(attention, "manual_lora_q"):
            raise RuntimeError(
                "Manual LoRA was not attached before attention patch."
            )
        attention.torch_multi_head_self_attention = types.MethodType(
            manual_lora_attention_forward,
            attention,
        )
        patched += 1
    return patched


def extract_lora_state_dict(backbone) -> dict[str, torch.Tensor]:
    """Extract only the manual LoRA parameters (~3-4 MB instead of 1.27 GB)."""
    state_dict = {}
    for name, param in backbone.named_parameters():
        if "manual_lora_q" in name or "manual_lora_v" in name:
            state_dict[name] = param.data.clone().cpu()
    return state_dict


def load_lora_state_dict(backbone, lora_dict: dict[str, torch.Tensor]) -> None:
    """Load manual LoRA parameters back into the backbone."""
    backbone_params = dict(backbone.named_parameters())
    for name, tensor in lora_dict.items():
        if name in backbone_params:
            backbone_params[name].data.copy_(tensor.to(backbone_params[name].device))
        else:
            raise KeyError(f"Parameter {name} not found in backbone.")


def apply_lora_to_backbone(
    backbone,
    lora_dict: dict[str, torch.Tensor] | None = None,
    r: int = 8,
    alpha: float = 16.0,
) -> None:
    """
    High-level helper to attach, patch, and optionally load LoRA weights.
    """
    if lora_dict is not None:
        for k, v in lora_dict.items():
            if "lora_A" in k:
                r = v.shape[0]
                alpha = float(r * 2.0)
                break
    attach_manual_lora(backbone, r=r, alpha=alpha)
    patch_wavlm_attention(backbone)
    if lora_dict is not None:
        load_lora_state_dict(backbone, lora_dict)
