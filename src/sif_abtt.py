from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


def _chunked(seq: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def _count_token_ids(
    tokenizer,
    texts: Sequence[str],
    batch_size: int = 128,
    max_length: int = 512,
) -> Tuple[Dict[int, int], int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    counts: Dict[int, int] = {}
    total = 0
    for batch in _chunked(texts, batch_size):
        enc = tokenizer(
            list(batch),
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )
        input_ids = enc["input_ids"]
        for seq in input_ids:
            for tok in seq:
                if tok in special_ids:
                    continue
                counts[tok] = counts.get(tok, 0) + 1
                total += 1
    return counts, total


def token_probabilities(
    tokenizer,
    texts: Sequence[str],
    batch_size: int = 128,
    max_length: int = 512,
) -> Dict[int, float]:
    """Estimate unigram token probabilities p(w) from text.

    Uses tokenizer IDs directly to avoid string-level normalization issues.
    """
    counts, total = _count_token_ids(
        tokenizer=tokenizer, texts=texts, batch_size=batch_size, max_length=max_length
    )
    if total == 0:
        return {}
    return {tok: count / total for tok, count in counts.items()}


def sif_weights_from_ids(
    input_ids: np.ndarray,
    token_probs: Dict[int, float],
    a: float = 1e-3,
    special_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Compute SIF weights for each token id.

    weight(w) = a / (a + p(w))  [Arora et al., Sec 3]
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be 2D array [batch, seq_len].")
    weights = np.ones_like(input_ids, dtype=np.float32)
    if token_probs:
        for tok, prob in token_probs.items():
            weights[input_ids == tok] = a / (a + float(prob))
    if special_ids:
        for tok in special_ids:
            weights[input_ids == tok] = 0.0
    return weights


def weighted_mean_pool(
    token_embeddings: np.ndarray,
    attention_mask: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must be [batch, seq_len, dim].")
    mask = attention_mask.astype(np.float32)
    weights = weights.astype(np.float32)
    scaled = token_embeddings * (mask * weights)[..., None]
    denom = (mask * weights).sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1.0)
    return scaled.sum(axis=1) / denom


def compute_pcs(
    X: np.ndarray, num_components: int, center: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute top principal components with NumPy SVD.

    If center=True, subtract mean before computing PCs.
    """
    if num_components <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32), None
    mean_vec = X.mean(axis=0) if center else None
    X_use = X - mean_vec if mean_vec is not None else X
    _, _, vt = np.linalg.svd(X_use, full_matrices=False)
    pcs = vt[:num_components].astype(np.float32)
    return pcs, mean_vec.astype(np.float32) if mean_vec is not None else None


def remove_top_components(
    X: np.ndarray, num_components: int, center: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Remove top PCs from embeddings.

    Uses ABTT-style removal when center=True: subtract mean, then remove
    projections on top-D components. [Mu & Viswanath, Eq 3]
    """
    if num_components <= 0:
        return X.copy(), None, np.zeros((0, X.shape[1]), dtype=np.float32)
    pcs, mean_vec = compute_pcs(X, num_components, center=center)
    X_use = X - mean_vec if mean_vec is not None else X
    proj = X_use @ pcs.T @ pcs
    cleaned = X_use - proj
    return cleaned, mean_vec, pcs


@dataclass
class UnigramProbEstimator:
    token_probs: Dict[int, float]
    total_tokens: int

    @classmethod
    def from_texts(
        cls,
        tokenizer,
        texts: Sequence[str],
        batch_size: int = 128,
        max_length: int = 512,
    ) -> "UnigramProbEstimator":
        counts, total = _count_token_ids(
            tokenizer=tokenizer, texts=texts, batch_size=batch_size, max_length=max_length
        )
        probs = {tok: count / total for tok, count in counts.items()} if total else {}
        return cls(token_probs=probs, total_tokens=total)

    def save(self, path: Union[str, Path]) -> None:
        """Save token probabilities to JSON."""
        payload = {
            "token_probs": {str(k): float(v) for k, v in self.token_probs.items()},
            "total_tokens": self.total_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "UnigramProbEstimator":
        """Load token probabilities from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        token_probs = {int(k): float(v) for k, v in payload["token_probs"].items()}
        return cls(token_probs=token_probs, total_tokens=int(payload["total_tokens"]))


@dataclass
class EmbeddingCleaner:
    num_components: int = 10
    center: bool = True
    a: float = 1e-3
    token_probs: Optional[Dict[int, float]] = None
    mean_vec: Optional[np.ndarray] = None
    pcs: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "EmbeddingCleaner":
        cleaned, mean_vec, pcs = remove_top_components(
            X, self.num_components, center=self.center
        )
        self.mean_vec = mean_vec
        self.pcs = pcs
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pcs is None:
            raise ValueError("EmbeddingCleaner.transform called before fit().")
        X_use = X - self.mean_vec if self.mean_vec is not None else X
        proj = X_use @ self.pcs.T @ self.pcs
        return X_use - proj

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save mean_vec and pcs to .npz file."""
        if self.pcs is None:
            raise ValueError("EmbeddingCleaner.save called before fit().")
        data = {"pcs": self.pcs, "num_components": np.array(self.num_components)}
        if self.mean_vec is not None:
            data["mean_vec"] = self.mean_vec
        np.savez(path, **data)

    @classmethod
    def load(cls, path: Union[str, Path], center: bool = True) -> "EmbeddingCleaner":
        """Load fitted EmbeddingCleaner from .npz file."""
        loaded = np.load(path)
        pcs = loaded["pcs"]
        num_components = int(loaded["num_components"])
        mean_vec = loaded["mean_vec"] if "mean_vec" in loaded else None
        cleaner = cls(num_components=num_components, center=center)
        cleaner.pcs = pcs
        cleaner.mean_vec = mean_vec
        return cleaner
