"""Stratification and pair-level metrics for scripts/resubmit/lexical_vs_embedding.py.

Everything here runs on a tiny hand-built fixture whose right answers can be
checked by eye, so the numbers the real run produces are backed by something
other than "it ran without crashing". No embeddings, no corpus, no sklearn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

from lexical_vs_embedding import (  # noqa: E402
    ALL_STRATUM,
    HARD_STRATUM,
    TERCILE_STRATA,
    assign_terciles,
    best_partner_similarity,
    directed_rank_table,
    enumerate_pairs,
    frac_above_tau,
    negative_confusion_rate,
    partner_ranks,
    rank_metrics,
    stratum_bounds,
    stratum_masks,
    tercile_edges,
)


# --------------------------------------------------------------------------
# Fixture: six test files in four directories.
#
#   index : 0    1    2    3    4    5
#   dir   : A    A    A    B    B    C
#
# Directory A has three members (so a query there has two partners, which is
# what separates the residual rank from the literal one), B has two, C is a
# singleton with no partner at all.
# --------------------------------------------------------------------------

FOLDER_IDS = np.array(["A", "A", "A", "B", "B", "C"])


@pytest.fixture
def overlap() -> np.ndarray:
    """Symmetric surface-overlap matrix with a distinct value per pair."""
    o = np.zeros((6, 6))
    values = {
        (0, 1): 0.90, (0, 2): 0.50, (1, 2): 0.10,   # positives in A
        (3, 4): 0.30,                                # positive in B
        (0, 3): 0.80, (0, 4): 0.05, (0, 5): 0.02,    # negatives
        (1, 3): 0.04, (1, 4): 0.03, (1, 5): 0.01,
        (2, 3): 0.07, (2, 4): 0.06, (2, 5): 0.08,
        (3, 5): 0.09, (4, 5): 0.11,
    }
    for (a, b), v in values.items():
        o[a, b] = o[b, a] = v
    np.fill_diagonal(o, 1.0)
    return o


# --------------------------------------------------------------------------
# Pair enumeration
# --------------------------------------------------------------------------


def test_enumerate_positive_pairs_are_same_directory(overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    assert list(zip(pos.i.tolist(), pos.j.tolist())) == [(0, 1), (0, 2), (1, 2), (3, 4)]
    assert pos.overlap.tolist() == [0.90, 0.50, 0.10, 0.30]


def test_enumerate_negative_pairs_exclude_same_directory_and_singleton_is_kept(overlap):
    neg = enumerate_pairs(FOLDER_IDS, overlap, positive=False)
    assert len(neg) == 15 - 4  # 6 choose 2, minus the four positives
    pairs = set(zip(neg.i.tolist(), neg.j.tolist()))
    assert (0, 1) not in pairs
    assert (0, 5) in pairs  # the singleton still forms negative pairs


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------


def test_tercile_edges_are_the_thirds_of_the_distribution():
    lo, hi = tercile_edges(np.arange(1.0, 10.0))  # 1..9
    assert lo == pytest.approx(3.6667, abs=1e-3)
    assert hi == pytest.approx(6.3333, abs=1e-3)


def test_tercile_edges_reject_too_few_pairs():
    with pytest.raises(ValueError):
        tercile_edges(np.array([0.1, 0.2]))


def test_assign_terciles_uses_half_open_bins():
    labels = assign_terciles(np.array([0.0, 0.2, 0.3, 0.5, 0.7, 1.0]), (0.3, 0.7))
    assert labels.tolist() == [
        "overlap_low",   # 0.0
        "overlap_low",   # 0.2
        "overlap_mid",   # 0.3 sits on the lower edge and goes up
        "overlap_mid",   # 0.5
        "overlap_high",  # 0.7 sits on the upper edge and goes up
        "overlap_high",  # 1.0
    ]


def test_assign_terciles_rejects_inverted_edges():
    with pytest.raises(ValueError):
        assign_terciles(np.array([0.5]), (0.8, 0.2))


def test_stratum_masks_partition_the_terciles_and_overlap_the_hard_slice(overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    edges = (0.25, 0.60)
    masks = stratum_masks(pos, edges, hard_threshold=0.40)

    tercile_stack = np.vstack([masks[s] for s in TERCILE_STRATA])
    assert tercile_stack.sum(axis=0).tolist() == [1, 1, 1, 1]  # a partition
    assert masks[ALL_STRATUM].all()

    # overlaps 0.90 / 0.50 / 0.10 / 0.30
    assert masks[TERCILE_STRATA[0]].tolist() == [False, False, True, False]
    assert masks[TERCILE_STRATA[1]].tolist() == [False, True, False, True]
    assert masks[TERCILE_STRATA[2]].tolist() == [True, False, False, False]
    # The hard slice is not a tercile: it takes 0.10 and 0.30, straddling low and mid.
    assert masks[HARD_STRATUM].tolist() == [False, False, True, True]


def test_stratum_bounds_report_the_actual_ranges(overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    bounds = stratum_bounds(pos, (0.25, 0.60), hard_threshold=0.40)
    assert bounds[TERCILE_STRATA[0]] == (0.10, 0.25)
    assert bounds[TERCILE_STRATA[1]] == (0.25, 0.60)
    assert bounds[TERCILE_STRATA[2]] == (0.60, 0.90)
    assert bounds[HARD_STRATUM] == (0.10, 0.40)
    assert bounds[ALL_STRATUM] == (0.10, 0.90)


# --------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------


@pytest.fixture
def sim() -> np.ndarray:
    """A similarity matrix with a known ranking for query 0.

    From query 0 the descending order is 1 (0.9), 2 (0.8), 3 (0.7), 4 (0.6),
    5 (0.5). Files 1 and 2 are both partners of 0, so the residual rank of
    partner 2 drops file 1 out of the pool and becomes 1, while its literal rank
    stays 2.
    """
    s = np.array(
        [
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            [0.9, 1.0, 0.2, 0.3, 0.1, 0.0],
            [0.8, 0.2, 1.0, 0.4, 0.1, 0.0],
            [0.7, 0.3, 0.4, 1.0, 0.15, 0.6],
            [0.6, 0.1, 0.1, 0.15, 1.0, 0.2],
            [0.5, 0.0, 0.0, 0.6, 0.2, 1.0],
        ]
    )
    return s


def test_partner_ranks_top_partner_is_rank_one_both_ways(sim):
    assert partner_ranks(sim, query=0, partner=1, folder_ids=FOLDER_IDS) == (1, 1)


def test_partner_ranks_residual_drops_the_other_partner(sim):
    residual, literal = partner_ranks(sim, query=0, partner=2, folder_ids=FOLDER_IDS)
    assert (residual, literal) == (1, 2)


def test_partner_ranks_counts_only_strictly_better_distractors(sim):
    # From query 3 the order is 0 (0.7), 5 (0.6), 2 (0.4), 1 (0.3), 4 (0.15):
    # partner 4 is last, and B has no other member, so both ranks agree.
    assert partner_ranks(sim, query=3, partner=4, folder_ids=FOLDER_IDS) == (5, 5)


def test_partner_ranks_ignore_the_self_similarity(sim):
    # sim[0, 0] = 1.0 is the largest entry in the row and must not push ranks down.
    assert partner_ranks(sim, query=0, partner=1, folder_ids=FOLDER_IDS)[1] == 1


def test_directed_rank_table_has_two_rows_per_pair(sim, overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    ranks = directed_rank_table(sim, pos, FOLDER_IDS)
    assert len(ranks) == 2 * len(pos)
    assert sorted(ranks["pair_index"].unique()) == list(range(len(pos)))
    # Pair 0 is (0, 1): rank 1 from either end (0's best is 1, and 1's best is 0).
    first = ranks[ranks["pair_index"] == 0]
    assert first["rank_residual"].tolist() == [1, 1]


def test_rank_metrics_recall_and_mrr_over_a_masked_stratum(sim, overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    ranks = directed_rank_table(sim, pos, FOLDER_IDS)

    only_pair_3_4 = np.array([False, False, False, True])
    got = rank_metrics(ranks, only_pair_3_4)
    # 3 -> 4 is rank 5; 4 -> 3 sees 0 (0.6) and 5 (0.2) above 3 (0.15), so rank 3.
    assert got["n_directed"] == 2
    assert got["recall_at_1"] == pytest.approx(0.0)
    assert got["mrr"] == pytest.approx((1 / 5 + 1 / 3) / 2)


def test_rank_metrics_residual_and_literal_recall_differ(sim, overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    ranks = directed_rank_table(sim, pos, FOLDER_IDS)
    only_pair_0_2 = np.array([False, True, False, False])
    got = rank_metrics(ranks, only_pair_0_2)
    # 0 -> 2 is residual 1 / literal 2; 2 -> 0 is rank 1 either way.
    assert got["recall_at_1"] == pytest.approx(1.0)
    assert got["recall_at_1_all"] == pytest.approx(0.5)


def test_rank_metrics_on_an_empty_stratum_are_nan_not_zero(sim, overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    ranks = directed_rank_table(sim, pos, FOLDER_IDS)
    got = rank_metrics(ranks, np.zeros(len(pos), dtype=bool))
    assert got["n_directed"] == 0
    assert np.isnan(got["recall_at_1"])
    assert np.isnan(got["mrr"])


# --------------------------------------------------------------------------
# Threshold coverage
# --------------------------------------------------------------------------


def test_frac_above_tau_counts_pairs_at_or_above_the_cut(sim, overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    mask = np.ones(len(pos), dtype=bool)
    # Positive pair scores under `sim`: (0,1)=0.9, (0,2)=0.8, (1,2)=0.2, (3,4)=0.15.
    assert frac_above_tau(sim, pos, mask, tau=0.8) == pytest.approx(0.5)
    assert frac_above_tau(sim, pos, mask, tau=0.0) == pytest.approx(1.0)
    assert frac_above_tau(sim, pos, mask, tau=0.95) == pytest.approx(0.0)


def test_frac_above_tau_on_an_empty_stratum_is_nan(sim, overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    assert np.isnan(frac_above_tau(sim, pos, np.zeros(len(pos), dtype=bool), tau=0.5))


# --------------------------------------------------------------------------
# Reverse slice: high-overlap negatives
# --------------------------------------------------------------------------


def test_best_partner_similarity_ignores_self_and_marks_singletons(sim):
    best = best_partner_similarity(sim, FOLDER_IDS)
    assert best[0] == pytest.approx(0.9)   # max(sim[0,1], sim[0,2])
    assert best[3] == pytest.approx(0.15)  # only partner is 4
    assert np.isneginf(best[5])            # directory C is a singleton


def test_negative_confusion_rate_flags_impostors_beating_every_partner(sim, overlap):
    neg = enumerate_pairs(FOLDER_IDS, overlap, positive=False)
    # The single highest-overlap negative pair is (0, 3) at 0.80.
    mask = (neg.i == 0) & (neg.j == 3)
    assert mask.sum() == 1

    rate, n = negative_confusion_rate(sim, neg, mask, FOLDER_IDS)
    # 0 -> 3: sim 0.7 vs best partner 0.9, not confused.
    # 3 -> 0: sim 0.7 vs best partner 0.15, confused.
    assert n == 2
    assert rate == pytest.approx(0.5)


def test_negative_confusion_rate_skips_endpoints_without_a_partner(sim, overlap):
    neg = enumerate_pairs(FOLDER_IDS, overlap, positive=False)
    mask = (neg.i == 0) & (neg.j == 5)  # file 5 is the singleton
    rate, n = negative_confusion_rate(sim, neg, mask, FOLDER_IDS)
    assert n == 1  # only the 0 -> 5 direction is scored
    # 0 -> 5: sim 0.5 vs best partner 0.9, not confused.
    assert rate == pytest.approx(0.0)


def test_negative_confusion_rate_is_nan_when_no_direction_is_usable(sim):
    singletons = np.array(["A", "B", "C", "D", "E", "F"])
    neg = enumerate_pairs(singletons, np.zeros((6, 6)), positive=False)
    rate, n = negative_confusion_rate(sim, neg, np.ones(len(neg), dtype=bool), singletons)
    assert n == 0
    assert np.isnan(rate)


# --------------------------------------------------------------------------
# End-to-end row assembly
# --------------------------------------------------------------------------


def test_collect_rows_emits_one_row_per_stratum_with_the_reverse_slice(sim, overlap):
    from lexical_vs_embedding import COLUMN_ORDER, NEG_STRATUM, Method, collect_rows

    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    neg = enumerate_pairs(FOLDER_IDS, overlap, positive=False)
    edges = (0.25, 0.60)
    masks = stratum_masks(pos, edges, hard_threshold=0.40)
    bounds = stratum_bounds(pos, edges, hard_threshold=0.40)
    neg_mask = neg.overlap >= 0.10

    method = Method("fixture", "lexical", "fixture", None, None, 0.5, sim)
    rows = collect_rows(method, pos, masks, neg, neg_mask, FOLDER_IDS, bounds, 0.10)

    frame = pd.DataFrame(rows)[COLUMN_ORDER]
    assert frame["stratum"].tolist() == [
        *TERCILE_STRATA, HARD_STRATUM, ALL_STRATUM, NEG_STRATUM,
    ]
    assert frame["n_pairs"].tolist()[:5] == [1, 2, 1, 2, 4]
    # Only the reverse row carries a confusion rate, and only it lacks recall.
    neg_row = frame[frame["stratum"] == NEG_STRATUM].iloc[0]
    assert pd.notna(neg_row["neg_confusion_rate"])
    assert pd.isna(neg_row["recall_at_1"])
    assert frame[frame["stratum"] != NEG_STRATUM]["neg_confusion_rate"].isna().all()
