"""Scoring functions behind the lexical baselines (issue #122).

The evaluation half of ``scripts/resubmit/lexical_baselines.py`` is deliberately
not retested here: it hands its score matrices to
``run_resubmit_evaluate.evaluate_from_similarity`` and
``run_taskb_mseed.evaluate_query_vs_directory``, which are the same functions the
embedding pipeline uses. What is new, and therefore what these tests cover, is
the scoring: tokenisation, the vectorised BM25, symmetrisation, and the
train-fitted rescaling that keeps the threshold sweep well posed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

from lexical_baselines import (  # noqa: E402
    apply_minmax,
    bm25_score_matrix,
    char_tfidf_score_matrix,
    fit_bm25,
    fit_minmax,
    levenshtein_score_matrix,
    normalise_text,
    symmetrise,
    tokenise,
)

# A tiny stand-in corpus: two near-duplicate canons, one unrelated text, and one
# that repeats a rare term so term saturation actually bites.
FIXTURE = [
    "Si quis episcopus, aut presbyter, contra canones egerit; deponatur.",
    "SI QUIS episcopus aut presbyter contra canones egerit deponatur!",
    "De ieiunio quadragesimae, et de oratione dominica.",
    "canones canones canones canones canones egerit.",
]


# --- normalisation ---------------------------------------------------------


def test_normalisation_lowercases_and_drops_punctuation() -> None:
    assert normalise_text("Si QUIS, episcopus;") == "si quis episcopus"


def test_normalisation_collapses_whitespace_runs() -> None:
    assert normalise_text("de\n\n ieiunio \t quadragesimae") == "de ieiunio quadragesimae"


def test_normalisation_survives_none_and_empty() -> None:
    assert normalise_text(None) == ""
    assert normalise_text("   ...   ") == ""


def test_the_two_near_duplicate_canons_tokenise_identically() -> None:
    # Case and punctuation are the only difference between fixture 0 and 1, so
    # the tokeniser must erase it. This is what makes them a positive pair.
    assert tokenise(FIXTURE[0]) == tokenise(FIXTURE[1])


def test_digits_are_kept_as_tokens() -> None:
    # Canon numbers carry signal; dropping digits would silently merge canons.
    assert tokenise("canon 12 et canon 13") == ["canon", "12", "et", "canon", "13"]


# --- BM25 ------------------------------------------------------------------


def naive_bm25(query_tokens, doc_tokens, index) -> float:
    """Textbook BM25, one query/document pair at a time, no vectorisation."""
    doc_len = len(doc_tokens)
    score = 0.0
    for term in query_tokens:  # with multiplicity, as rank_bm25 does
        col = index.vocab.get(term)
        if col is None:
            continue
        freq = doc_tokens.count(term)
        score += (
            index.idf[col]
            * freq
            * (index.k1 + 1.0)
            / (freq + index.k1 * (1.0 - index.b + index.b * doc_len / index.avgdl))
        )
    return score


def test_vectorised_bm25_matches_the_textbook_loop() -> None:
    tokens = [tokenise(t) for t in FIXTURE]
    index = fit_bm25(tokens[:2])  # train = first two documents
    scores = bm25_score_matrix(tokens, index)

    expected = np.array(
        [[naive_bm25(q, d, index) for d in tokens] for q in tokens]
    )
    assert np.allclose(scores, expected)


def test_bm25_hyperparameters_are_k1_1p5_b_0p75() -> None:
    index = fit_bm25([tokenise(t) for t in FIXTURE])
    assert (index.k1, index.b) == (1.5, 0.75)


def test_bm25_idf_and_length_stats_come_from_train_only() -> None:
    tokens = [tokenise(t) for t in FIXTURE]
    train_only = fit_bm25(tokens[:2])
    everything = fit_bm25(tokens)

    # "ieiunio" occurs only in the held-out documents, so a train-fitted index
    # must never have heard of it. That is the leak-free protocol in one line.
    assert "ieiunio" not in train_only.vocab
    assert "ieiunio" in everything.vocab
    assert train_only.n_train_docs == 2
    assert train_only.avgdl == pytest.approx(
        (len(tokens[0]) + len(tokens[1])) / 2
    )


def test_out_of_vocabulary_terms_contribute_nothing() -> None:
    tokens = [tokenise(t) for t in FIXTURE]
    index = fit_bm25(tokens[:2])
    scores = bm25_score_matrix(tokens, index)

    # Fixture 2 shares no train-vocabulary term with fixture 0.
    assert set(tokens[2]).isdisjoint(index.vocab)
    assert scores[2, 0] == 0.0
    assert scores[0, 2] == 0.0


def test_bm25_ranks_the_near_duplicate_above_the_unrelated_text() -> None:
    tokens = [tokenise(t) for t in FIXTURE]
    index = fit_bm25(tokens)
    scores = symmetrise(bm25_score_matrix(tokens, index))
    off_diagonal = scores[0].copy()
    off_diagonal[0] = -np.inf
    assert int(np.argmax(off_diagonal)) == 1


def test_bm25_saturates_repeated_document_terms() -> None:
    # Term saturation is what separates BM25 from raw term frequency: a document
    # containing "canones" five times must score better than one containing it
    # once, but nowhere near five times better. b=0 removes length normalisation
    # so the two documents differ in frequency alone.
    corpus = [["canones"], ["canones"] * 5, ["egerit"]]
    index = fit_bm25(corpus, b=0.0)
    scores = bm25_score_matrix(corpus, index)

    once, five_times = scores[0, 0], scores[0, 1]
    assert once < five_times < 5 * once


def test_fit_bm25_rejects_out_of_range_hyperparameters() -> None:
    tokens = [tokenise(t) for t in FIXTURE]
    with pytest.raises(ValueError):
        fit_bm25(tokens, k1=0.0)
    with pytest.raises(ValueError):
        fit_bm25(tokens, b=1.5)


# --- symmetrisation --------------------------------------------------------


def test_symmetrise_averages_both_directions() -> None:
    raw = np.array([[1.0, 4.0], [2.0, 3.0]])
    out = symmetrise(raw)
    assert np.allclose(out, [[1.0, 3.0], [3.0, 3.0]])
    assert np.allclose(out, out.T)


def test_symmetrise_rejects_non_square_input() -> None:
    with pytest.raises(ValueError):
        symmetrise(np.zeros((2, 3)))


# --- train-fitted rescaling ------------------------------------------------


def test_minmax_is_fitted_off_the_diagonal_of_the_train_block() -> None:
    scores = np.array(
        [
            [100.0, 2.0, 9.0],
            [2.0, 100.0, 9.0],
            [9.0, 9.0, 100.0],
        ]
    )
    train_idx = np.array([0, 1])
    lo, hi = fit_minmax(scores, train_idx)
    # Only the single off-diagonal train pair counts. The self-scores of 100 are
    # not retrieval scores and must not set the scale, and neither may the
    # test-row score of 9, which the split has not shown us yet.
    assert (lo, hi) == (2.0, 2.0)


def test_minmax_maps_the_train_range_onto_the_unit_interval() -> None:
    rng = np.random.default_rng(0)
    raw = symmetrise(rng.normal(size=(6, 6)))
    train_idx = np.array([0, 1, 2, 3])
    lo, hi = fit_minmax(raw, train_idx)
    rescaled = apply_minmax(raw, lo, hi)

    block = rescaled[np.ix_(train_idx, train_idx)]
    triu = block[np.triu_indices(len(train_idx), k=1)]
    assert triu.min() == pytest.approx(0.0)
    assert triu.max() == pytest.approx(1.0)


def test_minmax_preserves_ranking() -> None:
    rng = np.random.default_rng(1)
    raw = symmetrise(rng.normal(size=(8, 8)))
    train_idx = np.arange(5)
    lo, hi = fit_minmax(raw, train_idx)
    rescaled = apply_minmax(raw, lo, hi)
    # AUROC and every accuracy in the evaluator are rank-based, so the rescaling
    # must not touch the ordering of any row.
    assert np.array_equal(np.argsort(raw, axis=1), np.argsort(rescaled, axis=1))


def test_minmax_refuses_a_degenerate_range() -> None:
    with pytest.raises(ValueError):
        apply_minmax(np.zeros((3, 3)), 1.0, 1.0)


def test_fit_minmax_needs_at_least_two_train_documents() -> None:
    with pytest.raises(ValueError):
        fit_minmax(np.zeros((3, 3)), np.array([0]))


# --- Levenshtein -----------------------------------------------------------


def test_levenshtein_is_symmetric_bounded_and_unit_on_the_diagonal() -> None:
    pytest.importorskip("rapidfuzz")
    scores = levenshtein_score_matrix(FIXTURE, workers=1)
    assert scores.shape == (4, 4)
    assert np.allclose(scores, scores.T)
    assert np.allclose(np.diag(scores), 1.0)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_levenshtein_scores_the_near_duplicate_highest() -> None:
    pytest.importorskip("rapidfuzz")
    scores = levenshtein_score_matrix(FIXTURE, workers=1)
    off_diagonal = scores[0].copy()
    off_diagonal[0] = -np.inf
    assert int(np.argmax(off_diagonal)) == 1
    # And it does so only because normalisation erased case and punctuation.
    assert scores[0, 1] == pytest.approx(1.0)


# --- character n-gram TF-IDF -----------------------------------------------


def test_char_tfidf_is_a_symmetric_cosine_fitted_on_train() -> None:
    pytest.importorskip("sklearn")
    train_idx = np.array([0, 1])
    scores = char_tfidf_score_matrix(FIXTURE, train_idx)
    assert scores.shape == (4, 4)
    assert np.allclose(scores, scores.T)
    assert np.allclose(np.diag(scores)[:2], 1.0)
    assert scores.min() >= -1e-9 and scores.max() <= 1.0 + 1e-9


def test_char_tfidf_ranks_the_near_duplicate_highest() -> None:
    pytest.importorskip("sklearn")
    scores = char_tfidf_score_matrix(FIXTURE, np.arange(len(FIXTURE)))
    off_diagonal = scores[0].copy()
    off_diagonal[0] = -np.inf
    assert int(np.argmax(off_diagonal)) == 1
