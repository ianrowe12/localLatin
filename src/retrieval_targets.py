"""Differentiable forward wrappers for Captum retrieval-level token attribution.

Two scalar target functions for explaining ABTT via the actual retrieval metric:
1. BaselineCosSimTarget: cos(mean_pool(hidden[L]), partner_emb) — raw similarity
2. ABTTCosSimTarget: cos(ABTT(mean_pool(hidden[L])), partner_abtt_emb) — post-ABTT similarity

The partner embedding is a frozen constant (registered buffer); only the query's
tokens receive gradients. This way, Captum IG tells us which tokens in the query
drive (or hurt) its similarity to the true partner.

Usage with Captum:
    target = BaselineCosSimTarget(model, layer_idx=6, partner_emb=partner_tensor)
    lig = LayerIntegratedGradients(target, emb_layer)
    attr = lig.attribute(input_ids, additional_forward_args=(attention_mask,))
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCosSimTarget(nn.Module):
    """Scalar target: cosine similarity of mean-pooled hidden state to a frozen partner."""

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        partner_emb: torch.Tensor,
        token_keep_lookup: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        # Frozen partner embedding — no gradient flows through it
        self.register_buffer("partner_emb", partner_emb.float())  # (hidden_dim,)
        if token_keep_lookup is not None:
            self.register_buffer("token_keep_lookup", token_keep_lookup.float())
        else:
            self.token_keep_lookup = None

    def _pooled_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[self.layer_idx].float()  # (batch, seq, dim)
        mask = attention_mask.float()
        if self.token_keep_lookup is not None:
            mask = mask * self.token_keep_lookup[input_ids]
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self._pooled_hidden(input_ids, attention_mask)
        # Cosine similarity to frozen partner
        return F.cosine_similarity(pooled, self.partner_emb.unsqueeze(0), dim=-1)  # (batch,)


class ABTTCosSimTarget(nn.Module):
    """Scalar target: cosine similarity of ABTT-cleaned hidden state to ABTT-cleaned partner."""

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        partner_abtt_emb: torch.Tensor,
        pcs: torch.Tensor,
        mean_vec: torch.Tensor,
        token_keep_lookup: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.register_buffer("partner_abtt_emb", partner_abtt_emb.float())  # (hidden_dim,)
        self.register_buffer("pcs", pcs.float())              # (D, hidden_dim)
        self.register_buffer("mean_vec", mean_vec.float())      # (hidden_dim,)
        if token_keep_lookup is not None:
            self.register_buffer("token_keep_lookup", token_keep_lookup.float())
        else:
            self.token_keep_lookup = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[self.layer_idx].float()
        mask = attention_mask.float()
        if self.token_keep_lookup is not None:
            mask = mask * self.token_keep_lookup[input_ids]
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        # ABTT cleaning
        centered = pooled - self.mean_vec
        proj = centered @ self.pcs.T @ self.pcs
        cleaned = centered - proj  # (batch, dim)
        # Cosine similarity to frozen ABTT-cleaned partner
        return F.cosine_similarity(cleaned, self.partner_abtt_emb.unsqueeze(0), dim=-1)
