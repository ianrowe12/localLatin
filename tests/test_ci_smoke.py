"""CI smoke tests for the data-free parts of ``src/``.

These exercise pure-Python / NumPy helpers that do not need ``data/``,
``runs/``, model weights, or torch, so they run on a clean GitHub-hosted
checkout. Model-dependent modules (``extract_*_cli``, ``token_filtering``,
``retrieval_*``, ``attribution_targets``) import torch/transformers and are
deliberately out of scope here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from canon_retrieval import (  # noqa: E402
    accuracy_at_k,
    l2_normalize,
    last_token_pool,
    similarity_matrix,
)
from cli_utils import extract_layer_numbers, parse_layers  # noqa: E402
from sif_abtt import EmbeddingCleaner, weighted_mean_pool  # noqa: E402


# --- src/cli_utils.py -------------------------------------------------------


def test_parse_layers_expands_ranges_and_sorts_unique() -> None:
    assert parse_layers("0-3") == [0, 1, 2, 3]
    assert parse_layers("0,6,12") == [0, 6, 12]
    assert parse_layers("12,0-2,1") == [0, 1, 2, 12]
    assert parse_layers(" 4 , 4 ") == [4]


def test_parse_layers_empty_input_returns_empty_list() -> None:
    assert parse_layers("") == []
    assert parse_layers(" , ") == []


def test_parse_layers_rejects_descending_range() -> None:
    with pytest.raises(ValueError, match="Invalid layer range"):
        parse_layers("5-2")


def test_extract_layer_numbers_pulls_layer_ids_from_paths() -> None:
    paths = [
        "runs/x/ff1_layer3_embeddings.npy",
        "runs/x/ff1_layer12_embeddings_sif.npy",
        "runs/x/ff1_layer3_embeddings_norm.npy",
        "runs/x/not_a_match.npy",
    ]
    assert extract_layer_numbers(paths, r"layer(\d+)") == [3, 12]


# --- src/canon_retrieval.py -------------------------------------------------


def test_last_token_pool_picks_the_final_unpadded_position() -> None:
    hidden = np.array(
        [
            [[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]],
            [[5.0, 5.0], [7.0, 7.0], [8.0, 8.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
    np.testing.assert_allclose(last_token_pool(hidden, mask), [[3.0, 3.0], [8.0, 8.0]])


def test_l2_normalize_gives_unit_rows_and_cosine_similarity() -> None:
    emb = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    normed = l2_normalize(emb)
    np.testing.assert_allclose(np.linalg.norm(normed, axis=1), [1.0, 1.0], atol=1e-6)

    sim = similarity_matrix(normed)
    assert sim.shape == (2, 2)
    np.testing.assert_allclose(np.diag(sim), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(sim, sim.T, atol=1e-6)


def test_accuracy_at_k_scores_only_winnable_queries() -> None:
    # a0/a1 are near-duplicates; b0 is the only member of its folder.
    emb = l2_normalize(
        np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]], dtype=np.float32)
    )
    sim = similarity_matrix(emb)
    folder_ids = ["a", "a", "b"]

    winnable = [True, True, False]
    assert accuracy_at_k(sim, folder_ids, winnable, k=1) == 1.0

    # Including the unwinnable singleton drags accuracy down to 2/3.
    all_queries = [True, True, True]
    assert accuracy_at_k(sim, folder_ids, all_queries, k=1) == pytest.approx(2 / 3)


# --- src/sif_abtt.py --------------------------------------------------------


def test_weighted_mean_pool_normalises_by_masked_weight_mass() -> None:
    # Weights deliberately do not sum to 1 and the third position is padding:
    # dropping the mask or the /denom normalisation both change the result.
    tokens = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    weights = np.array([[0.5, 1.5, 2.0]], dtype=np.float32)

    # (0.5 * [1, 2] + 1.5 * [3, 4]) / (0.5 + 1.5) = [5, 7] / 2
    np.testing.assert_allclose(
        weighted_mean_pool(tokens, mask, weights), [[2.5, 3.5]], rtol=1e-6
    )


def test_weighted_mean_pool_returns_zeros_when_everything_is_masked() -> None:
    # Denominator is clamped to >= 1.0, so an all-padding row must not divide by 0.
    tokens = np.array([[[7.0, 9.0]]], dtype=np.float32)
    mask = np.array([[0]], dtype=np.int64)
    weights = np.array([[0.75]], dtype=np.float32)
    np.testing.assert_allclose(weighted_mean_pool(tokens, mask, weights), [[0.0, 0.0]])


def test_embedding_cleaner_removes_the_dominant_direction_and_keeps_the_rest() -> None:
    # Axis 0 carries ~20x the standard deviation of every other axis, so the top
    # principal direction is axis 0. Removing any *other* component (a random
    # direction, or the least significant PC) leaves axis-0 variance intact and
    # destroys variance elsewhere, so both assertions below are direction-specific.
    rng = np.random.default_rng(0)
    scales = np.array([20.0, 1.0, 0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    offset = np.array([3.0, -2.0, 1.0, 0.0, 0.5, -0.5], dtype=np.float32)

    train = (rng.normal(size=(512, 6)) * scales + offset).astype(np.float32)
    cleaner = EmbeddingCleaner(num_components=1).fit(train)
    assert cleaner.pcs is not None and cleaner.pcs.shape == (1, 6)
    # The fitted component really is the dominant axis, not some other direction.
    assert abs(float(cleaner.pcs[0, 0])) > 0.99

    held_out = (rng.normal(size=(256, 6)) * scales + offset).astype(np.float32)
    cleaned = cleaner.transform(held_out)

    var_before = held_out.var(axis=0)
    var_after = cleaned.var(axis=0)

    # Dominant direction collapses ...
    assert var_after[0] / var_before[0] < 1e-3
    # ... while the remaining directions survive essentially untouched.
    np.testing.assert_allclose(var_after[1:], var_before[1:], rtol=0.05)

    # No residual projection onto the removed component, and centering happened.
    assert np.abs(cleaned @ cleaner.pcs.T).max() < 1e-2
    assert np.abs(cleaned.mean(axis=0)).max() < 0.5


def test_embedding_cleaner_requires_fit_before_transform() -> None:
    X = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError):
        EmbeddingCleaner(num_components=1).transform(X)
