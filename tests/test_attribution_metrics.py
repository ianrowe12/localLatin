"""Unit tests for the attribution-quality metrics added in issue #124.

Every case here runs on a tiny synthetic pair whose "true" per-token
contribution is known by construction, so each metric has an ordering we can
assert rather than a number we have to eyeball. Three attributions are used
throughout:

  * ``aligned``  -- equal to the true contribution (a perfect explanation)
  * ``reversed`` -- anti-correlated with it (a perfectly wrong explanation)
  * ``flat``     -- constant (no information at all)

A faithful metric must order these aligned > flat > reversed, and must give
``flat`` the same score as its own shuffles.

The suite also covers the hidden-state backend in
``scripts/ig/run_attribution_metrics.py``, because that is where the masked
cosine actually comes from in the CPU run: its vectorised prefix curves are
checked against a naive per-mask loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

# scipy is imported unguarded on purpose. It is pinned in
# .github/ci-requirements.txt, so a missing scipy is a CI failure rather than a
# silent skip of this whole file: these cases are the only automated check on
# the metric definitions the paper's attribution table is built from.
from attribution_metrics import (  # noqa: E402
    FULL_COS_FLOOR,
    METRIC_REGISTRY,
    PairContext,
    aopc,
    comprehensiveness,
    deletion_auc,
    insertion_auc,
    kendall_tau_loo,
    loo_correlation,
    randomization_check,
    rank_order,
    sufficiency,
    top_k_mask,
)

N = 24


# ---------------------------------------------------------------------------
# Fixtures: a synthetic pair whose true token contributions are known
# ---------------------------------------------------------------------------
def _true_contrib() -> np.ndarray:
    """Distinct positive contributions, scattered across token positions.

    Deliberately NOT monotone in position: if the largest contribution sat at
    index 0 then the stable tie-break in ``rank_order`` would make a constant
    attribution accidentally perfect, and the ``flat`` control would stop being
    a control.
    """
    values = np.linspace(1.0, 0.1, N)
    return values[np.random.default_rng(11).permutation(N)]


def _linear_context(*, with_curves: bool, full_scale: float = 1.0) -> PairContext:
    """A pair whose score is the sum of the kept tokens' contributions.

    Linear in the mask, so every metric has an analytic answer and the fast and
    fallback curve paths must agree exactly.
    """
    contrib = _true_contrib() * full_scale
    full = float(contrib.sum())

    def eval_masked_cos(mask01: np.ndarray) -> float:
        return float((contrib * np.asarray(mask01, dtype=np.float64)).sum())

    single = np.array(
        [full - contrib[i] for i in range(N)],
        dtype=np.float64,
    )

    prefix_curves = None
    if with_curves:
        def prefix_curves(order: np.ndarray):  # noqa: F811
            csum = np.concatenate([[0.0], np.cumsum(contrib[order])])
            return csum, full - csum

    return PairContext(
        n_q=N,
        variant="baseline",
        full_cos=full,
        single_ablation_cos=single,
        eval_masked_cos=eval_masked_cos,
        prefix_curves=prefix_curves,
    )


@pytest.fixture
def ctx() -> PairContext:
    return _linear_context(with_curves=True)


@pytest.fixture
def aligned() -> np.ndarray:
    return _true_contrib()


@pytest.fixture
def reversed_scores() -> np.ndarray:
    # Anti-correlated with the true contribution in magnitude, which is what
    # every metric here ranks on (all of them take |a|).
    return 1.0 / (1e-9 + _true_contrib())


@pytest.fixture
def flat() -> np.ndarray:
    return np.full(N, 0.5)


# ---------------------------------------------------------------------------
# rank_order / top_k_mask contract
# ---------------------------------------------------------------------------
def test_rank_order_agrees_with_top_k_mask(aligned: np.ndarray) -> None:
    order = rank_order(aligned)
    for k in (1, 5, N - 1, N):
        expected = np.zeros(N, dtype=bool)
        expected[order[:k]] = True
        assert np.array_equal(top_k_mask(aligned, k), expected)


def test_rank_order_breaks_ties_by_position(flat: np.ndarray) -> None:
    assert np.array_equal(rank_order(flat), np.arange(N))


def test_rank_order_ranks_by_magnitude_not_sign() -> None:
    scores = np.array([-9.0, 1.0, 5.0])
    assert np.array_equal(rank_order(scores), np.array([0, 2, 1]))


# ---------------------------------------------------------------------------
# curves(): endpoint convention and fast/fallback equivalence
# ---------------------------------------------------------------------------
def test_curves_pin_the_empty_query_endpoints(ctx: PairContext, aligned: np.ndarray) -> None:
    keep, drop = ctx.curves(rank_order(aligned))
    assert keep.shape == (N + 1,) and drop.shape == (N + 1,)
    assert keep[0] == 0.0
    assert drop[N] == 0.0
    assert keep[N] == pytest.approx(ctx.full_cos)
    assert drop[0] == pytest.approx(ctx.full_cos)


def test_fast_and_fallback_curves_agree(aligned: np.ndarray) -> None:
    fast = _linear_context(with_curves=True)
    slow = _linear_context(with_curves=False)
    order = rank_order(aligned)
    kf, df = fast.curves(order)
    ks, ds = slow.curves(order)
    assert np.allclose(kf, ks)
    assert np.allclose(df, ds)


def test_curves_rejects_a_bad_order(ctx: PairContext) -> None:
    with pytest.raises(ValueError):
        ctx.curves(np.arange(N - 1))


# ---------------------------------------------------------------------------
# AOPC
# ---------------------------------------------------------------------------
def test_aopc_orders_aligned_above_flat_above_reversed(
    ctx: PairContext, aligned: np.ndarray, flat: np.ndarray, reversed_scores: np.ndarray
) -> None:
    a = aopc(ctx, aligned)
    f = aopc(ctx, flat)
    r = aopc(ctx, reversed_scores)
    for key in ("aopc_suff_ratio", "aopc_comp_ratio"):
        assert a[key] > f[key] > r[key], (key, a[key], f[key], r[key])


def test_aopc_matches_the_compute_aopc_flag(ctx: PairContext, aligned: np.ndarray) -> None:
    """The `--aopc` CLI flag and the registered `aopc` metric must not diverge."""
    metric = aopc(ctx, aligned)
    suff = sufficiency(ctx, aligned, compute_aopc=True)
    comp = comprehensiveness(ctx, aligned, compute_aopc=True)
    assert metric["aopc_suff_raw"] == pytest.approx(suff["suff_aopc_raw"])
    assert metric["aopc_suff_ratio"] == pytest.approx(suff["suff_aopc_ratio"])
    assert metric["aopc_comp_raw"] == pytest.approx(comp["comp_aopc_drop"])
    assert metric["aopc_comp_ratio"] == pytest.approx(comp["comp_aopc_ratio"])


def test_aopc_has_the_closed_form_value_on_a_two_token_pair() -> None:
    """n=2, contributions [3, 1]: keep curve is [0, 3, 4], drop curve is [4, 1, 0].

    aopc_suff_raw = mean(3, 4) = 3.5; aopc_comp_raw = mean(4-1, 4-0) = 3.5.
    """
    contrib = np.array([3.0, 1.0])

    def eval_masked_cos(mask01: np.ndarray) -> float:
        return float((contrib * np.asarray(mask01, dtype=np.float64)).sum())

    small = PairContext(
        n_q=2, variant="baseline", full_cos=4.0,
        single_ablation_cos=np.array([1.0, 3.0]),
        eval_masked_cos=eval_masked_cos,
    )
    out = aopc(small, contrib)
    assert out["aopc_suff_raw"] == pytest.approx(3.5)
    assert out["aopc_comp_raw"] == pytest.approx(3.5)
    assert out["aopc_suff_ratio"] == pytest.approx(3.5 / 4.0)


def test_aopc_ratio_is_nan_below_the_cosine_floor(aligned: np.ndarray) -> None:
    tiny = _linear_context(with_curves=True, full_scale=1e-4)
    assert abs(tiny.full_cos) < FULL_COS_FLOOR
    out = aopc(tiny, aligned)
    assert np.isnan(out["aopc_suff_ratio"])
    assert np.isnan(out["aopc_comp_ratio"])
    # The raw columns stay finite; only the ratio is undefined.
    assert np.isfinite(out["aopc_suff_raw"])


# ---------------------------------------------------------------------------
# Deletion / insertion AUC
# ---------------------------------------------------------------------------
def test_insertion_auc_beats_random_for_an_aligned_attribution(
    ctx: PairContext, aligned: np.ndarray
) -> None:
    out = insertion_auc(ctx, aligned)
    assert out["ins_auc"] > out["ins_auc_random"]
    assert out["ins_auc_gap"] > 0


def test_deletion_auc_undercuts_random_for_an_aligned_attribution(
    ctx: PairContext, aligned: np.ndarray
) -> None:
    out = deletion_auc(ctx, aligned)
    assert out["del_auc"] < out["del_auc_random"]
    assert out["del_auc_gap"] > 0


def test_reversed_attribution_loses_to_random_on_both_aucs(
    ctx: PairContext, reversed_scores: np.ndarray
) -> None:
    assert insertion_auc(ctx, reversed_scores)["ins_auc_gap"] < 0
    assert deletion_auc(ctx, reversed_scores)["del_auc_gap"] < 0


def test_a_flat_attribution_scores_near_the_random_reference(
    ctx: PairContext, flat: np.ndarray, aligned: np.ndarray
) -> None:
    # A constant score vector carries no information: its order is whatever the
    # positional tie-break happens to produce, so its gap must be small next to
    # the gap a real ranking earns. (An exact zero is not available here -- one
    # arbitrary ordering is a single draw, not the mean over draws.)
    for metric in (insertion_auc, deletion_auc):
        key = "ins_auc_gap" if metric is insertion_auc else "del_auc_gap"
        assert abs(metric(ctx, flat)[key]) < 0.5 * metric(ctx, aligned)[key]


def test_deletion_and_insertion_share_the_random_reference(
    ctx: PairContext, aligned: np.ndarray
) -> None:
    """Same seed, same orderings: the two references must be computed on one set.

    On a linear score the deletion and insertion curves sum to the constant
    ``full``, so their normalised AUCs must sum to 1 for any single ordering,
    and therefore also for the average over a shared set of orderings.
    """
    ins = insertion_auc(ctx, aligned)
    dele = deletion_auc(ctx, aligned)
    assert ins["ins_auc"] + dele["del_auc"] == pytest.approx(1.0)
    assert ins["ins_auc_random"] + dele["del_auc_random"] == pytest.approx(1.0)


def test_aucs_are_nan_below_the_cosine_floor(aligned: np.ndarray) -> None:
    tiny = _linear_context(with_curves=True, full_scale=1e-4)
    assert np.isnan(insertion_auc(tiny, aligned)["ins_auc"])
    assert np.isnan(deletion_auc(tiny, aligned)["del_auc"])


# ---------------------------------------------------------------------------
# Kendall tau vs LOO
# ---------------------------------------------------------------------------
def test_kendall_tau_is_one_for_a_perfectly_aligned_attribution(
    ctx: PairContext, aligned: np.ndarray
) -> None:
    assert kendall_tau_loo(ctx, aligned)["loo_tau"] == pytest.approx(1.0)


def test_kendall_tau_is_minus_one_for_a_reversed_attribution(
    ctx: PairContext, reversed_scores: np.ndarray
) -> None:
    assert kendall_tau_loo(ctx, reversed_scores)["loo_tau"] == pytest.approx(-1.0)


def test_kendall_tau_agrees_in_sign_with_spearman(
    ctx: PairContext, aligned: np.ndarray, reversed_scores: np.ndarray
) -> None:
    for scores in (aligned, reversed_scores):
        tau = kendall_tau_loo(ctx, scores)["loo_tau"]
        rho = loo_correlation(ctx, scores)["loo_rho"]
        assert np.sign(tau) == np.sign(rho)
        # Kendall tau is bounded by Spearman rho in magnitude for a monotone
        # relation, which is the regime these fixtures construct.
        assert abs(tau) <= abs(rho) + 1e-9


def test_kendall_tau_is_nan_for_a_constant_attribution(
    ctx: PairContext, flat: np.ndarray
) -> None:
    out = kendall_tau_loo(ctx, flat)
    assert np.isnan(out["loo_tau"])
    assert out["loo_tau_n_used"] == N


def test_kendall_tau_is_nan_on_a_too_short_query() -> None:
    short = PairContext(
        n_q=2, variant="baseline", full_cos=1.0,
        single_ablation_cos=np.array([0.5, 0.5]),
        eval_masked_cos=lambda m: 0.0,
    )
    assert np.isnan(kendall_tau_loo(short, np.array([1.0, 2.0]))["loo_tau"])


# ---------------------------------------------------------------------------
# Randomization sanity check
# ---------------------------------------------------------------------------
GAP_KEYS = (
    "rand_loo_rho_gap",
    "rand_loo_tau_gap",
    "rand_aopc_suff_ratio_gap",
    "rand_aopc_comp_ratio_gap",
    "rand_ins_auc_gap_gap",
    "rand_del_auc_gap_gap",
)


def test_randomization_gap_is_positive_for_an_aligned_attribution(
    ctx: PairContext, aligned: np.ndarray
) -> None:
    out = randomization_check(ctx, aligned)
    for key in GAP_KEYS:
        assert out[key] > 0, (key, out[key])


def test_randomization_gap_is_negative_for_a_reversed_attribution(
    ctx: PairContext, reversed_scores: np.ndarray
) -> None:
    out = randomization_check(ctx, reversed_scores)
    for key in GAP_KEYS:
        assert out[key] < 0, (key, out[key])


def test_randomization_gap_is_exactly_zero_for_a_constant_attribution(
    ctx: PairContext, flat: np.ndarray
) -> None:
    """Permuting a constant vector is a no-op, so real and shuffled must tie.

    This is the check that the gap really is 'real minus shuffled' on the same
    quantity and not two differently-computed numbers.
    """
    out = randomization_check(ctx, flat)
    assert out["rand_aopc_suff_ratio_gap"] == pytest.approx(0.0)
    assert out["rand_aopc_comp_ratio_gap"] == pytest.approx(0.0)
    assert out["rand_ins_auc_gap_gap"] == pytest.approx(0.0)
    assert out["rand_del_auc_gap_gap"] == pytest.approx(0.0)
    # rho/tau are NaN on a constant vector, so their gaps are NaN, not 0.
    assert np.isnan(out["rand_loo_rho_gap"])


def test_randomization_covers_the_chance_corrected_aucs(
    ctx: PairContext, aligned: np.ndarray
) -> None:
    """The shuffled control must have a number for every main-table candidate.

    Issue #135 review: criterion 5 of the selection memo was applied to
    AOPC-Suff but waived for InsAUC gap, which has no shuffled-attribution
    number unless the control computes one. This pins that it does.
    """
    out = randomization_check(ctx, aligned)
    for key in ("rand_ins_auc_gap", "rand_ins_auc_gap_gap",
                "rand_del_auc_gap", "rand_del_auc_gap_gap"):
        assert key in out
        assert np.isfinite(out[key]), (key, out[key])


def test_randomization_auc_gap_level_matches_the_metric_itself(
    ctx: PairContext, aligned: np.ndarray
) -> None:
    """``rand_<k>`` is the shuffled level, so real minus it must be ``rand_<k>_gap``.

    Pins that the control's real-side AUC gap is the same quantity
    ``insertion_auc`` / ``deletion_auc`` report, i.e. that the shared
    random-order reference is drawn identically in both places.
    """
    out = randomization_check(ctx, aligned)
    real_ins = insertion_auc(ctx, aligned)["ins_auc_gap"]
    real_del = deletion_auc(ctx, aligned)["del_auc_gap"]
    assert real_ins - out["rand_ins_auc_gap"] == pytest.approx(out["rand_ins_auc_gap_gap"])
    assert real_del - out["rand_del_auc_gap"] == pytest.approx(out["rand_del_auc_gap_gap"])


def test_auc_gap_shuffle_equals_its_aopc_twin(
    ctx: PairContext, aligned: np.ndarray, reversed_scores: np.ndarray
) -> None:
    """AOPC-Suff and InsAUC are the same statistic, so they share one verdict.

    The random-order reference is a property of the pair, identical for the real
    and every shuffled attribution, and the mean-over-k versus trapezoid offset
    is a constant; both cancel in the gap. So the shuffled-attribution gap of
    ``ins_auc_gap`` must equal that of ``aopc_suff_ratio`` exactly, and likewise
    for deletion and AOPC-Comp. This is why the selection memo cannot pass one
    and fail the other.
    """
    for scores in (aligned, reversed_scores):
        out = randomization_check(ctx, scores)
        assert out["rand_ins_auc_gap_gap"] == pytest.approx(
            out["rand_aopc_suff_ratio_gap"], abs=1e-12)
        assert out["rand_del_auc_gap_gap"] == pytest.approx(
            out["rand_aopc_comp_ratio_gap"], abs=1e-12)


def test_randomization_reports_the_shuffled_level_too(
    ctx: PairContext, aligned: np.ndarray
) -> None:
    out = randomization_check(ctx, aligned)
    # The shuffled AOPC should sit near the random-order reference, well below
    # the aligned attribution's own value.
    assert out["rand_aopc_suff_ratio"] < aopc(ctx, aligned)["aopc_suff_ratio"]


def test_randomization_is_deterministic(ctx: PairContext, aligned: np.ndarray) -> None:
    assert randomization_check(ctx, aligned) == randomization_check(ctx, aligned)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------
def test_every_new_metric_is_registered() -> None:
    expected = {
        "sufficiency", "comprehensiveness", "compactness", "loo_correlation",
        "kendall_tau_loo", "aopc", "deletion_auc", "insertion_auc",
        "randomization_check",
    }
    assert expected <= set(METRIC_REGISTRY)


def test_registry_functions_all_accept_ctx_and_scores(
    ctx: PairContext, aligned: np.ndarray
) -> None:
    for name, fn in METRIC_REGISTRY.items():
        out = fn(ctx, aligned)
        assert isinstance(out, dict) and out, name


# ---------------------------------------------------------------------------
# Hidden-state backend (the evaluator the CPU run is built on)
# ---------------------------------------------------------------------------
@pytest.fixture
def hidden_pair():
    rng = np.random.default_rng(7)
    n, d, n_pc = 9, 6, 2
    q = rng.normal(size=(n, d))
    c = rng.normal(size=(4, d))
    pcs = np.linalg.qr(rng.normal(size=(d, n_pc)))[0].T  # (n_pc, d) orthonormal
    mean_vec = rng.normal(size=d)
    return q, c, pcs, mean_vec


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("variant", ["baseline", "abtt"])
def test_hidden_evaluator_reproduces_a_hand_computed_cosine(hidden_pair, variant) -> None:
    from run_attribution_metrics import HiddenPairEvaluator

    q, c, pcs, mean_vec = hidden_pair

    def clean(v):
        if variant != "abtt":
            return v
        centered = v - mean_vec
        return centered - (centered @ pcs.T) @ pcs

    expected = _cos(clean(q.mean(axis=0)), clean(c.mean(axis=0)))
    ev = HiddenPairEvaluator(q, c, pcs, mean_vec, variant)
    assert ev.full_cos == pytest.approx(expected)


@pytest.mark.parametrize("variant", ["baseline", "abtt"])
def test_hidden_prefix_curves_match_a_naive_mask_loop(hidden_pair, variant) -> None:
    from run_attribution_metrics import HiddenPairEvaluator

    q, c, pcs, mean_vec = hidden_pair
    n = q.shape[0]
    ev = HiddenPairEvaluator(q, c, pcs, mean_vec, variant)
    order = np.array([3, 0, 7, 1, 8, 2, 6, 4, 5])

    keep, drop = ev.prefix_curves(order)
    for k in range(1, n):
        mask = np.zeros(n, dtype=np.int64)
        mask[order[:k]] = 1
        assert keep[k] == pytest.approx(ev.eval_masked_cos(mask))
        assert drop[k] == pytest.approx(ev.eval_masked_cos(1 - mask))
    assert keep[0] == 0.0 and drop[n] == 0.0
    assert keep[n] == pytest.approx(ev.full_cos)
    assert drop[0] == pytest.approx(ev.full_cos)


@pytest.mark.parametrize("variant", ["baseline", "abtt"])
def test_hidden_single_ablation_matches_the_mask_loop(hidden_pair, variant) -> None:
    from run_attribution_metrics import HiddenPairEvaluator

    q, c, pcs, mean_vec = hidden_pair
    n = q.shape[0]
    ev = HiddenPairEvaluator(q, c, pcs, mean_vec, variant)
    loo = ev.single_ablation_cos()
    for i in range(n):
        mask = np.ones(n, dtype=np.int64)
        mask[i] = 0
        assert loo[i] == pytest.approx(ev.eval_masked_cos(mask))


def test_hidden_abtt_variant_removes_the_principal_components(hidden_pair) -> None:
    """The ABTT path must actually project the top PCs out of the pooled vector."""
    from run_attribution_metrics import HiddenPairEvaluator

    q, c, pcs, mean_vec = hidden_pair
    ev = HiddenPairEvaluator(q, c, pcs, mean_vec, "abtt")
    cleaned = ev._clean(q.mean(axis=0))
    assert np.allclose(cleaned @ pcs.T, 0.0, atol=1e-10)


def test_hidden_context_carries_the_fast_curve_path(hidden_pair) -> None:
    from run_attribution_metrics import HiddenPairEvaluator

    q, c, pcs, mean_vec = hidden_pair
    ctx_hidden = HiddenPairEvaluator(q, c, pcs, mean_vec, "baseline").context()
    assert ctx_hidden.has_fast_curves
    assert ctx_hidden.n_q == q.shape[0]
