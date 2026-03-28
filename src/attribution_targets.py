"""Differentiable forward wrappers for Captum token-level attribution.

Two scalar target functions for explaining ABTT:
1. PC1DotTarget: dot(mean_pool(hidden[L]), pc1) — which tokens load onto the dominant PC
2. ABTTNormTarget: ||ABTT(mean_pool(hidden[L]))||₂ — which tokens matter after ABTT cleaning

Usage with Captum:
    from captum.attr import LayerIntegratedGradients
    target = PC1DotTarget(model, layer_idx=6, pc1=pc1_tensor, mean_vec=mean_tensor)
    lig = LayerIntegratedGradients(target, emb_layer)
    attr = lig.attribute(input_ids, additional_forward_args=(attention_mask,))
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PC1DotTarget(nn.Module):
    """Scalar target: dot product of mean-pooled hidden state with PC1."""

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        pc1: torch.Tensor,
        mean_vec: torch.Tensor,
        token_keep_lookup: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.register_buffer("pc1", pc1.float())           # (hidden_dim,)
        self.register_buffer("mean_vec", mean_vec.float())  # (hidden_dim,)
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
        hidden = outputs.hidden_states[self.layer_idx].float()  # (batch, seq, dim)
        mask = attention_mask.float()
        if self.token_keep_lookup is not None:
            mask = mask * self.token_keep_lookup[input_ids]
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        centered = pooled - self.mean_vec
        return (centered * self.pc1).sum(dim=-1)  # (batch,)


class ABTTNormTarget(nn.Module):
    """Scalar target: L2 norm of ABTT-cleaned mean-pooled hidden state."""

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        pcs: torch.Tensor,
        mean_vec: torch.Tensor,
        token_keep_lookup: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.register_buffer("pcs", pcs.float())            # (D, hidden_dim)
        self.register_buffer("mean_vec", mean_vec.float())   # (hidden_dim,)
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
        centered = pooled - self.mean_vec             # (batch, dim)
        proj = centered @ self.pcs.T @ self.pcs       # (batch, dim)
        cleaned = centered - proj
        return cleaned.norm(dim=-1)                   # (batch,)


def get_embedding_layer(model: nn.Module, model_type: str) -> nn.Module:
    """Return the word embedding layer for a given model type.

    Args:
        model: The HuggingFace model (or its encoder for T5).
        model_type: One of "t5", "bert", "decoder", "decoder_wrapped".
    """
    if model_type == "t5":
        # model may be the full Seq2Seq model or already the encoder
        if hasattr(model, "get_encoder"):
            return model.get_encoder().embed_tokens
        # Already the encoder (e.g. passed from attribution script)
        return model.embed_tokens
    if model_type == "bert":
        return model.embeddings.word_embeddings
    if model_type == "decoder":
        return model.embed_tokens
    if model_type == "decoder_wrapped":
        return model.model.embed_tokens
    raise ValueError(f"Unknown model_type: {model_type}")


def get_model_for_forward(model: nn.Module, model_type: str) -> nn.Module:
    """Return the module to use in forward pass (encoder for T5, model itself otherwise)."""
    if model_type == "t5":
        return model.get_encoder() if hasattr(model, "get_encoder") else model
    return model
