"""Stratification and pair-level metrics for scripts/resubmit/lexical_vs_embedding.py.

Everything here runs on a tiny hand-built fixture whose right answers can be
checked by eye, so the numbers the real run produces are backed by something
other than "it ran without crashing". No corpus and no cached embeddings; the
one test of the leak-free protocol builds its own 40x8 matrix and reaches
scikit-learn through the evaluator, which CI installs.
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
    mcnemar_exact_p,
    negative_confusion_flags,
    negative_confusion_rate,
    operating_point,
    paired_discordance,
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


# --------------------------------------------------------------------------
# Paired significance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "b, c, expected",
    [
        (0, 0, 1.0),          # nothing discordant: no evidence of a difference
        (1, 1, 1.0),          # one each way
        (10, 0, 2 / 2 ** 10), # every discordant observation on one side
        (3, 1, 0.625),        # 2 * (C(4,0) + C(4,1)) / 2**4
        (1, 3, 0.625),        # and it is symmetric
    ],
)
def test_mcnemar_exact_p_matches_the_hand_computed_binomial(b, c, expected):
    assert mcnemar_exact_p(b, c) == pytest.approx(expected)


def test_mcnemar_exact_p_is_never_above_one():
    # b == c doubles a tail that is already more than half the mass.
    assert all(mcnemar_exact_p(k, k) <= 1.0 for k in range(1, 12))


def test_mcnemar_exact_p_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact_p(-1, 3)


def test_paired_discordance_counts_each_side_separately():
    reference = np.array([True, True, False, False, True])
    other = np.array([True, False, True, False, False])
    assert paired_discordance(reference, other) == (2, 1)


def test_paired_discordance_rejects_unaligned_vectors():
    with pytest.raises(ValueError):
        paired_discordance(np.array([True, False]), np.array([True]))


def test_operating_point_reports_tpr_fpr_and_precision(sim, overlap):
    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    neg = enumerate_pairs(FOLDER_IDS, overlap, positive=False)
    op = operating_point(sim, pos, neg, tau=0.50)
    n_pos_hit = int((sim[pos.i, pos.j] >= 0.50).sum())
    n_neg_hit = int((sim[neg.i, neg.j] >= 0.50).sum())
    assert op["op_n_tp"] == n_pos_hit
    assert op["op_n_fp"] == n_neg_hit
    assert op["op_tpr"] == pytest.approx(n_pos_hit / len(pos))
    assert op["op_fpr"] == pytest.approx(n_neg_hit / len(neg))
    assert op["op_precision"] == pytest.approx(n_pos_hit / (n_pos_hit + n_neg_hit))


def test_negative_confusion_flags_line_up_with_the_rate(sim, overlap):
    neg = enumerate_pairs(FOLDER_IDS, overlap, positive=False)
    mask = neg.overlap >= 0.05
    flags = negative_confusion_flags(sim, neg, mask, FOLDER_IDS)
    rate, n = negative_confusion_rate(sim, neg, mask, FOLDER_IDS)
    assert flags.size == n
    assert flags.mean() == pytest.approx(rate)


def test_negative_confusion_flags_are_aligned_across_methods(sim, overlap):
    """A different score matrix must not change *which* observations exist.

    That is what makes the reverse-slice McNemar pairing legitimate: usability
    depends on the directory structure alone.
    """
    neg = enumerate_pairs(FOLDER_IDS, overlap, positive=False)
    mask = neg.overlap >= 0.05
    other = np.zeros_like(sim)
    np.fill_diagonal(other, 1.0)
    assert (
        negative_confusion_flags(sim, neg, mask, FOLDER_IDS).size
        == negative_confusion_flags(other, neg, mask, FOLDER_IDS).size
    )


def test_collect_rows_carries_mcnemar_columns_against_a_reference(sim, overlap):
    from lexical_vs_embedding import (
        COLUMN_ORDER,
        NEG_STRATUM,
        Method,
        collect_rows,
        method_outcomes,
    )

    pos = enumerate_pairs(FOLDER_IDS, overlap, positive=True)
    neg = enumerate_pairs(FOLDER_IDS, overlap, positive=False)
    edges = (0.25, 0.60)
    masks = stratum_masks(pos, edges, hard_threshold=0.40)
    bounds = stratum_bounds(pos, edges, hard_threshold=0.40)
    neg_mask = neg.overlap >= 0.10

    reference = Method("ref", "lexical", "ref", None, None, 0.5, sim)
    # Blunt the second method on exactly one positive pair, (0, 1), the only
    # member of the high tercile. The reference puts that partner at rank 1 from
    # both endpoints; the copy drops it below every distractor, so it loses two
    # directed observations and wins none back.
    worse_sim = sim.copy()
    worse_sim[0, 1] = worse_sim[1, 0] = 0.001
    other = Method("other", "abtt", "other", 1, 10, 0.5, worse_sim)

    ref_out = method_outcomes(reference, pos, neg, neg_mask, FOLDER_IDS)
    rows = pd.DataFrame(
        collect_rows(other, pos, masks, neg, neg_mask, FOLDER_IDS, bounds, 0.10,
                     reference=ref_out)
    )[COLUMN_ORDER]

    high = rows[rows["stratum"] == TERCILE_STRATA[2]].iloc[0]
    assert (high["mcnemar_b"], high["mcnemar_c"]) == (2, 0)
    assert high["mcnemar_p"] == pytest.approx(0.5)
    # The routing outcome moves too: that pair drops below tau for the copy only.
    assert (high["routing_b"], high["routing_c"]) == (1, 0)
    assert high["routing_p"] == pytest.approx(1.0)
    # Nothing changed in the low tercile, so both sides agree there.
    low = rows[rows["stratum"] == TERCILE_STRATA[0]].iloc[0]
    assert (low["mcnemar_b"], low["mcnemar_c"]) == (0, 0)
    assert low["mcnemar_p"] == pytest.approx(1.0)

    # The reference method itself is compared against nothing.
    self_rows = pd.DataFrame(
        collect_rows(reference, pos, masks, neg, neg_mask, FOLDER_IDS, bounds, 0.10)
    )[COLUMN_ORDER]
    assert self_rows["mcnemar_p"].isna().all()
    assert self_rows["routing_p"].isna().all()
    # Operating-point columns are method-level, so they repeat on every row.
    assert self_rows["op_tpr"].nunique() == 1
    assert pd.notna(self_rows[self_rows["stratum"] == NEG_STRATUM].iloc[0]["op_tpr"])


# --------------------------------------------------------------------------
# Configuration selection and the embedding cache
# --------------------------------------------------------------------------


def test_paper_config_takes_the_highest_train_aucroc_row():
    from lexical_vs_embedding import paper_config

    results = pd.DataFrame(
        {
            "model": ["m", "m", "m", "other"],
            "method": ["abtt_optimal", "abtt_optimal", "baseline", "abtt_optimal"],
            "layer": [3, 7, 7, 1],
            "train_aucroc": [0.80, 0.91, 0.99, 0.99],
        }
    )
    assert int(paper_config(results, "m", "abtt_optimal")["layer"]) == 7


def test_paper_config_ignores_rows_without_a_train_aucroc():
    from lexical_vs_embedding import paper_config

    results = pd.DataFrame(
        {
            "model": ["m", "m"],
            "method": ["baseline", "baseline"],
            "layer": [3, 7],
            "train_aucroc": [0.80, np.nan],
        }
    )
    assert int(paper_config(results, "m", "baseline")["layer"]) == 3


def test_paper_config_refuses_to_guess_when_the_model_is_absent():
    from lexical_vs_embedding import paper_config

    results = pd.DataFrame(
        {"model": ["m"], "method": ["baseline"], "layer": [1], "train_aucroc": [0.9]}
    )
    with pytest.raises(SystemExit):
        paper_config(results, "missing", "baseline")


@pytest.mark.parametrize(
    "pooling, tail",
    [
        ("mean", "hidden_mean_tokempty/hidden_layer4_embeddings.npy"),
        ("sif", "hidden_sif_tokempty/hidden_layer4_embeddings_sif.npy"),
        ("lasttok", "hidden_lasttok_tokempty/hidden_layer4_embeddings_lasttok.npy"),
    ],
)
def test_embedding_path_follows_the_repr_and_pooling_columns(pooling, tail):
    from lexical_vs_embedding import embedding_path

    path = embedding_path(Path("/bases"), "org/Model", "hidden", pooling, 4)
    assert path == Path("/bases/phase9_bases/org_Model") / tail


def test_embedding_path_rejects_an_unknown_pooling():
    from lexical_vs_embedding import embedding_path

    with pytest.raises(SystemExit):
        embedding_path(Path("/bases"), "org/Model", "hidden", "weighted", 4)


# --------------------------------------------------------------------------
# Row alignment.
#
# The script used to carry its own filename-alignment helper. It now calls
# src/embedding_alignment.AlignmentResolver, the one implementation every other
# consumer of these caches uses, so the tests below pin two things: that the
# shared resolver resolves exactly the rows the retired helper did, and that
# this script keeps its own stricter contract on top of it, namely that a cache
# the resolver cannot verify is fatal rather than read positionally.
# --------------------------------------------------------------------------


def _retired_row_index(cached_names, split_filenames) -> np.ndarray:
    """The helper this script used to own, kept as the reference to match.

    Verbatim in behaviour: look up each split filename's position in the cached
    row order and return those positions, so ``cache[index]`` sits in split
    order. Deleted from the script in #140; retained here only so the shared
    resolver has something to be proved identical to.
    """
    position = {name: k for k, name in enumerate(cached_names)}
    return np.array([position[f] for f in split_filenames], dtype=int)


def _write_cache(tmp_path: Path, cached_names, matrix, manifest=True) -> Path:
    """Build a bases root: row_order.csv beside phase9_bases, one cached layer.

    Mirrors the production layout, where the manifest sits at the
    ``phase9_bases`` root rather than inside each run directory, so the
    resolver has to find it by walking the cache's ancestors.
    """
    from lexical_vs_embedding import embedding_path

    path = embedding_path(tmp_path, "org/Model", "hidden", "mean", 2)
    path.parent.mkdir(parents=True)
    np.save(path, np.asarray(matrix, dtype=float))
    if manifest:
        pd.DataFrame(
            {
                "file_id": range(len(cached_names)),
                "path": [f"canon_labelled/d/{n}" for n in cached_names],
            }
        ).to_csv(tmp_path / "phase9_bases" / "row_order.csv", index=False)
    return path


def _split_meta(filenames) -> pd.DataFrame:
    return pd.DataFrame({"filename": list(filenames)})


def _load(tmp_path, split_filenames):
    from embedding_alignment import AlignmentResolver
    from lexical_vs_embedding import load_layer_embeddings

    resolver = AlignmentResolver(_split_meta(split_filenames))
    return load_layer_embeddings(tmp_path, "org/Model", 2, "hidden", "mean", resolver)


@pytest.mark.parametrize(
    "cached, split",
    [
        # identity
        (["a.txt", "b.txt", "c.txt"], ["a.txt", "b.txt", "c.txt"]),
        # full reversal
        (["a.txt", "b.txt", "c.txt"], ["c.txt", "b.txt", "a.txt"]),
        # a rotation, which is the shape of a directory relabelling
        (["a.txt", "b.txt", "c.txt"], ["c.txt", "a.txt", "b.txt"]),
        # a contiguous block moving inside a longer run, as #131's did
        (
            [f"f{k:02d}.txt" for k in range(12)],
            [f"f{k:02d}.txt" for k in range(6)]
            + ["f09.txt", "f10.txt", "f11.txt", "f06.txt", "f07.txt", "f08.txt"],
        ),
    ],
)
def test_shared_resolver_resolves_the_same_rows_as_the_retired_helper(
    tmp_path, cached, split
):
    """One implementation repo-wide, and it moves the rows the old one moved."""
    cache = np.arange(len(cached), dtype=float).reshape(-1, 1) * 10.0
    _write_cache(tmp_path, cached, cache)

    expected = cache[_retired_row_index(cached, split)]
    assert _load(tmp_path, split).tolist() == expected.tolist()


def test_load_layer_embeddings_reorders_the_cached_rows(tmp_path):
    _write_cache(tmp_path, ["a.txt", "b.txt", "c.txt"],
                 [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    loaded = _load(tmp_path, ["c.txt", "a.txt", "b.txt"])
    assert loaded[:, 0].tolist() == [3.0, 1.0, 2.0]


def test_load_layer_embeddings_is_the_identity_when_nothing_moved(tmp_path):
    _write_cache(tmp_path, ["a.txt", "b.txt", "c.txt"],
                 [[1.0], [2.0], [3.0]])

    assert _load(tmp_path, ["a.txt", "b.txt", "c.txt"]).ravel().tolist() == [
        1.0, 2.0, 3.0
    ]


def test_load_layer_embeddings_rejects_a_split_file_the_cache_never_saw(tmp_path):
    _write_cache(tmp_path, ["a.txt", "b.txt"], [[1.0], [2.0]])

    with pytest.raises(SystemExit):
        _load(tmp_path, ["a.txt", "z.txt"])


def test_load_layer_embeddings_rejects_a_row_order_with_repeated_filenames(tmp_path):
    _write_cache(tmp_path, ["a.txt", "a.txt"], [[1.0], [2.0]])

    with pytest.raises(SystemExit):
        _load(tmp_path, ["a.txt", "b.txt"])


def test_load_layer_embeddings_rejects_a_cache_that_is_the_wrong_length(tmp_path):
    _write_cache(tmp_path, ["a.txt", "b.txt", "c.txt"], [[1.0], [2.0], [3.0]])

    with pytest.raises(SystemExit):
        _load(tmp_path, ["a.txt", "b.txt"])


def test_load_layer_embeddings_refuses_a_cache_with_no_manifest(tmp_path):
    """This script is stricter than the shared resolver, on purpose.

    ``AlignmentResolver`` falls back to a positional read with a warning when it
    finds no manifest, so caches predating the manifest still run. A silent
    positional read is the exact bug that made #140's re-run necessary, so here
    it has to stop the run instead.
    """
    _write_cache(tmp_path, ["a.txt"], [[1.0], [2.0], [3.0]], manifest=False)

    with pytest.raises(SystemExit):
        _load(tmp_path, ["a.txt", "b.txt", "c.txt"])


def test_load_layer_embeddings_reports_a_missing_cache_file(tmp_path):
    from embedding_alignment import AlignmentResolver
    from lexical_vs_embedding import load_layer_embeddings

    resolver = AlignmentResolver(_split_meta(["a.txt"]))
    with pytest.raises(SystemExit):
        load_layer_embeddings(tmp_path, "org/Model", 2, "hidden", "mean", resolver)


def test_build_embedding_method_fits_the_cleaner_on_train_rows_only():
    """The leak-free invariant: test rows must not reach the PC fit.

    Rewriting the test block while leaving the train block alone changes
    nothing the cleaner or ``tau`` are allowed to see, so the learned ``tau``
    must be identical. If the cleaner were fitted on all rows, the removed
    components would move and the train similarities with them.
    """
    from lexical_vs_embedding import _ensure_import_paths, build_embedding_method

    _ensure_import_paths()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(40, 8))
    train_mask = np.arange(40) < 24
    test_mask = ~train_mask
    train_folder_ids = np.repeat(np.arange(12), 2)

    def run(matrix):
        return build_embedding_method(
            "k", "abtt", "M", matrix, train_mask, test_mask,
            train_folder_ids, layer=5, D=3,
        )

    shifted = emb.copy()
    shifted[test_mask] = rng.normal(loc=50.0, scale=20.0, size=(16, 8))

    assert run(emb).tau == run(shifted).tau
