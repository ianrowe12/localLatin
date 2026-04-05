"""Direct Logit Attribution (DLA) for token-pair matrices.

DLA is a single-pass geometric decomposition that measures how much each
token's hidden-state vector aligns with each token in the other sequence.
Unlike IG (which needs ~50 gradient forward passes), DLA operates directly
on hidden representations.

Reference: Elhage et al., "A Mathematical Framework for Transformer Circuits"
"""
from __future__ import annotations

import numpy as np


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix between a (N,D) and b (M,D)."""
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a_n @ b_n.T


def dla_pair_matrix(
    q_hidden: np.ndarray,
    c_hidden: np.ndarray,
) -> np.ndarray:
    """DLA pair matrix: cosine * normalized geometric mean of token norms.

    Parameters
    ----------
    q_hidden : (N_q, D) query token hidden states (typically ABTT-cleaned)
    c_hidden : (N_c, D) candidate token hidden states (typically ABTT-cleaned)

    Returns
    -------
    (N_q, N_c) matrix where entry [i,j] represents how much query token i
    aligns with candidate token j, weighted by their representation magnitudes.
    Tokens with larger norms contribute more.
    """
    cos = cosine_matrix(q_hidden, c_hidden)
    q_norms = np.linalg.norm(q_hidden, axis=1)
    c_norms = np.linalg.norm(c_hidden, axis=1)
    weight = np.sqrt(q_norms[:, None] * c_norms[None, :])
    w_min, w_max = weight.min(), weight.max()
    if w_max - w_min > 1e-12:
        weight = (weight - w_min) / (w_max - w_min)
    else:
        weight = np.ones_like(weight)
    return cos * weight
