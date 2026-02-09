"""Shared pair-based evaluation utilities for both canon and MUSTS experiments."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def cosine_similarities_for_pairs(
    embeddings_norm: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity for indexed pairs.

    Parameters
    ----------
    embeddings_norm : [N, dim] L2-normalized embeddings
    idx_a, idx_b : integer index arrays (same length)

    Returns
    -------
    1D array of cosine similarities
    """
    return np.sum(embeddings_norm[idx_a] * embeddings_norm[idx_b], axis=1)


def safe_spearman(sims: np.ndarray, labels: np.ndarray) -> float:
    """Spearman correlation, NaN on failure."""
    if len(sims) < 3:
        return float("nan")
    try:
        result = spearmanr(sims, labels)
        corr = result.statistic if hasattr(result, "statistic") else result[0]
        return float(corr)
    except Exception:
        return float("nan")


def safe_auc_roc(sims: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC for binary labels, NaN on failure."""
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, sims))
    except Exception:
        return float("nan")


def retrieval_accuracy_at_k(
    sim: np.ndarray,
    folder_ids: np.ndarray,
    query_mask: np.ndarray,
    k: int,
) -> float:
    """Retrieval acc@k within a similarity matrix.

    For each query (query_mask=True), check whether any of the top-k
    most-similar non-self files share the same folder_id.
    """
    folder_ids = np.asarray(folder_ids)
    query_mask = np.asarray(query_mask, dtype=bool)
    n = sim.shape[0]
    correct = 0
    total = int(query_mask.sum())
    for i in range(n):
        if not query_mask[i]:
            continue
        scores = sim[i].copy()
        scores[i] = -np.inf
        topk_idx = np.argpartition(-scores, k)[:k]
        if np.any(folder_ids[topk_idx] == folder_ids[i]):
            correct += 1
    return correct / total if total > 0 else 0.0
