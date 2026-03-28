from __future__ import annotations

from typing import Literal

import numpy as np
import torch

TokenFilter = Literal["all", "tokenizer_empty", "no_empty", "content_only"]
TOKEN_FILTER_CHOICES: tuple[TokenFilter, ...] = (
    "all",
    "tokenizer_empty",
    "no_empty",
    "content_only",
)

LATIN_PUNCTUATION = set(".,;:!?()[]{}\"'-/\\@#$%^&*+=<>~`")


def classify_token(token_str: str) -> str:
    """Classify a decoded token string into a coarse content bucket."""
    s = token_str.strip()
    if not s:
        return "empty"
    clean = s.lstrip("▁##Ġ")
    if not clean:
        return "empty"
    if all(c in LATIN_PUNCTUATION for c in clean):
        return "punctuation"
    if clean.isdigit():
        return "number"
    if len(clean) <= 2:
        return "short_subword"
    return "content"


def classify_token_for_filtering(token_str: str) -> str:
    """Collapse non-lexical categories into the filter-oriented taxonomy."""
    category = classify_token(token_str)
    if category in {"punctuation", "number"}:
        return "empty"
    return category


def token_filter_suffix(token_filter: str) -> str:
    if token_filter == "all":
        return ""
    if token_filter == "tokenizer_empty":
        return "_tokempty"
    if token_filter == "no_empty":
        return "_noempty"
    if token_filter == "content_only":
        return "_contentonly"
    raise ValueError(f"Unknown token_filter: {token_filter}")


def token_filter_subdir(repr_name: str, pooling: str, token_filter: str) -> str:
    return f"{repr_name}_{pooling}{token_filter_suffix(token_filter)}"


def keep_token_for_filter(category: str, token_filter: str) -> float:
    if token_filter == "all":
        return 1.0
    if token_filter == "tokenizer_empty":
        return 0.0 if category == "empty" else 1.0
    if token_filter == "no_empty":
        return 0.0 if category in {"empty", "punctuation", "number"} else 1.0
    if token_filter == "content_only":
        return 1.0 if category == "content" else 0.0
    raise ValueError(f"Unknown token_filter: {token_filter}")


def build_token_keep_lookup(tokenizer, token_filter: str) -> np.ndarray:
    """Build a 0/1 lookup vector over tokenizer IDs for a filter policy."""
    vocab_size = len(tokenizer)
    lookup = np.ones(vocab_size, dtype=np.float32)
    if token_filter == "all":
        return lookup

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        category = classify_token(token_str)
        lookup[token_id] = keep_token_for_filter(category, token_filter)
    return lookup


def numpy_token_keep_mask(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    token_keep_lookup: np.ndarray | None,
) -> np.ndarray:
    mask = attention_mask.astype(np.float32)
    if token_keep_lookup is None:
        return mask
    keep = token_keep_lookup[input_ids].astype(np.float32)
    return mask * keep


def torch_token_keep_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_keep_lookup: np.ndarray | None,
) -> torch.Tensor:
    mask = attention_mask.float()
    if token_keep_lookup is None:
        return mask
    keep_lookup = torch.as_tensor(
        token_keep_lookup, device=input_ids.device, dtype=torch.float32
    )
    keep = keep_lookup[input_ids]
    return mask * keep
