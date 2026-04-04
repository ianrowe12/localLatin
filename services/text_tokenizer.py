"""Lightweight Latin-aware tokenizer for display purposes.

Splits on whitespace and punctuation boundaries. Classifies each token
using the same logic as src/token_filtering.classify_token() but without
importing torch.
"""

from __future__ import annotations

import re

from web.models import TokenInfo

_LATIN_PUNCTUATION = set(".,;:!?()[]{}\"'-/\\@#$%^&*+=<>~`")

# Split on whitespace, keeping punctuation as separate tokens
_TOKEN_RE = re.compile(r"(\s+|[.,;:!?()\[\]{}\"\'\-/\\·]+)")


def classify_token(token_str: str) -> str:
    """Classify a token string into a coarse content bucket.

    Mirrors src/token_filtering.classify_token() without torch dependency.
    """
    s = token_str.strip()
    if not s:
        return "empty"
    # Strip common subword prefixes
    clean = s.lstrip("▁##Ġ")
    if not clean:
        return "empty"
    if all(c in _LATIN_PUNCTUATION for c in clean):
        return "punctuation"
    if clean.isdigit():
        return "number"
    if len(clean) <= 2:
        return "short_subword"
    return "content"


def latin_tokenize(text: str) -> list[TokenInfo]:
    """Tokenize Latin text into display tokens with classification."""
    parts = _TOKEN_RE.split(text)
    tokens: list[TokenInfo] = []
    idx = 0
    for part in parts:
        if not part:
            continue
        # Skip pure whitespace
        if part.isspace():
            continue
        # Split punctuation clusters into individual tokens
        stripped = part.strip()
        if not stripped:
            continue
        category = classify_token(stripped)
        tokens.append(TokenInfo(text=stripped, index=idx, category=category))
        idx += 1
    return tokens
