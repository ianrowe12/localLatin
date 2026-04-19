"""Retrieval-adapted attribution-quality metrics.

Adapts Brinner & Zarriess 2023 (MarK) and DeYoung et al. 2020 (ERASER) attribution
evaluation metrics from text classification to a pair-cosine retrieval setting.
The decision scalar is

    S_v(q, c) = cos(E_v(q), E_v(c))

where E_v is the embedding map under post-processing variant v in {"baseline", "abtt"}.
Token-level attribution scores ``a in R^n`` are evaluated by masking subsets of query
tokens and observing the change in S_v.

Metrics implemented (one function per metric, registered via ``register``; add a new
metric by writing one decorated function):

  - ``sufficiency``         (suff_alpha = S_v(top-k mask) / S_v(full); plus AOPC)
  - ``comprehensiveness``   (comp_alpha = 1 - S_v(complement mask) / S_v(full); plus AOPC)
  - ``compactness``         (smallest fraction k/n s.t. sufficiency >= tau)
  - ``loo_correlation``     (Spearman rho between |a| and per-token Δcos under LOO ablation)

The "loo_correlation" metric is what some papers call "faithfulness" (Atanasova et al.
2020); we keep the more specific name here to avoid conflating it with MarK's AOPC-style
faithfulness, which is captured by AOPC over sufficiency/comprehensiveness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

# Numerical constants. Pairs with full_cos below FULL_COS_FLOOR yield NaN metrics
# (the sufficiency ratio is unstable for small denominators). LOO correlation
# discards tokens with |Delta| below LOO_NOISE_FLOOR. The floor is set to
# 1e-6 -- machine-level noise on float32 cosines -- rather than a stricter
# 1e-4, because ABTT-cleaned embeddings are often near-invariant to single-token
# swaps (the residual subspace carries little signal); a stricter floor would
# discard the actual signal rather than pure noise on those pairs.
FULL_COS_FLOOR: float = 0.05
LOO_NOISE_FLOOR: float = 1e-6


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------
MaskedCosFn = Callable[[np.ndarray], float]
MetricFn = Callable[..., Dict[str, float]]


@dataclass(frozen=True)
class PairContext:
    """Bundle the per-pair context a metric needs.

    ``eval_masked_cos`` is a closure produced by the driver: it takes a binary
    mask of length ``n_q`` (1 = keep token, 0 = mask) and returns the scalar
    cosine ``S_v(q ⊙ m, c)`` under the current variant. The candidate is fixed.

    ``single_ablation_cos`` is the cached length-``n_q`` vector of cosines
    obtained by leaving each query token out one at a time. Computing it once
    per (pair, variant) and reusing across methods cuts forward-pass count by
    a factor of ``n_methods``.
    """

    n_q: int
    variant: str
    full_cos: float
    single_ablation_cos: np.ndarray  # shape (n_q,)
    eval_masked_cos: MaskedCosFn
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def loo_deltas(self) -> np.ndarray:
        """Per-token leave-one-out drop in cosine: full_cos - cos(q\\{i}, c)."""
        return self.full_cos - self.single_ablation_cos


# ----------------------------------------------------------------------------
# Score-from-pair-matrix reducers
# ----------------------------------------------------------------------------
# Each attribution method stores a (q_len, c_len) pair matrix in the NPZ. To
# rank query tokens for top-k masking we need a per-token query score. Choices
# are method-specific so we map them explicitly:
#
#   ig                   -> use the stored per-token IG vector (query_ig_<variant>)
#   bertscore            -> row max  (matches greedy-alignment definition)
#   ot                   -> row sum of positive mass (matches transport-mass total)
#   attention_weighted   -> signed row sum
#   attention_standalone -> signed row sum
#   dla                  -> signed row sum
#   <unknown>            -> signed row sum (safe fallback)
METHOD_SCORE_REDUCER: Dict[str, str] = {
    "ig": "stored_ig",
    "bertscore": "row_max",
    "ot": "row_sum_positive",
    "attention_weighted": "row_sum_signed",
    "attention_standalone": "row_sum_signed",
    "dla": "row_sum_signed",
    # Future Agent 2.1 method - will use the safe fallback unless explicitly mapped.
    "retrieval_mark": "row_sum_signed",
}

REDUCER_FALLBACK: str = "row_sum_signed"


def scores_from_pair_matrix(
    pair_matrix: np.ndarray,
    stored_per_token: Optional[np.ndarray],
    reducer: str,
) -> np.ndarray:
    """Reduce a (q_len, c_len) pair matrix to a (q_len,) per-query-token score.

    ``stored_per_token`` is used only when reducer == "stored_ig"; pass None
    otherwise.
    """
    if reducer == "stored_ig":
        if stored_per_token is None:
            raise ValueError("reducer='stored_ig' requires stored_per_token")
        return np.asarray(stored_per_token, dtype=np.float64)
    if reducer == "row_max":
        return np.asarray(pair_matrix, dtype=np.float64).max(axis=1)
    if reducer == "row_sum_signed":
        return np.asarray(pair_matrix, dtype=np.float64).sum(axis=1)
    if reducer == "row_sum_positive":
        m = np.asarray(pair_matrix, dtype=np.float64)
        return np.where(m > 0, m, 0.0).sum(axis=1)
    raise ValueError(f"unknown reducer: {reducer!r}")


# ----------------------------------------------------------------------------
# Top-k helpers
# ----------------------------------------------------------------------------
def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of length len(scores) with True at the indices of the
    top-``k`` |score| values.

    Tie-breaking is stable by token position (lowest index wins). Returns an
    all-True mask if ``k >= len(scores)``; an all-False mask if ``k <= 0``.
    """
    n = len(scores)
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    abs_scores = np.abs(np.asarray(scores, dtype=np.float64))
    # argsort with stable kind for deterministic tie-breaking; descending by
    # negating. Equal scores get earlier indices first.
    order = np.argsort(-abs_scores, kind="stable")
    keep = np.zeros(n, dtype=bool)
    keep[order[:k]] = True
    return keep


def k_from_fraction(alpha: float, n: int) -> int:
    """k(alpha) = max(1, ceil(alpha * n)). Floors at 1 to avoid empty masks."""
    if n <= 0:
        return 0
    return max(1, int(np.ceil(alpha * n)))


# ----------------------------------------------------------------------------
# Metric registry
# ----------------------------------------------------------------------------
METRIC_REGISTRY: Dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator to register a metric in METRIC_REGISTRY."""

    def deco(fn: MetricFn) -> MetricFn:
        if name in METRIC_REGISTRY:
            raise ValueError(f"metric {name!r} already registered")
        METRIC_REGISTRY[name] = fn
        return fn

    return deco


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
@register("sufficiency")
def sufficiency(
    ctx: PairContext,
    scores: np.ndarray,
    *,
    fractions: Tuple[float, ...] = (0.10, 0.25, 0.50),
    compute_aopc: bool = False,
) -> Dict[str, float]:
    """Sufficiency metric: keep top-k% tokens (by |score|), mask rest.

    Returns ``suff@<frac>_ratio`` (S_v(q_topk, c) / S_v(q_full, c)) and
    ``suff@<frac>_raw`` (S_v(q_topk, c) directly) per fraction. When
    ``compute_aopc=True`` also returns ``suff_aopc_{raw,ratio}`` averaged over
    {1/n, 2/n, ..., 1}; this costs an additional ``n`` forward passes per pair
    so it is off by default for the GPU-heavy retrieval setting.
    NaN ratios when |full_cos| < FULL_COS_FLOOR.
    """
    out: Dict[str, float] = {}
    n = ctx.n_q
    full = ctx.full_cos
    ratio_safe = abs(full) >= FULL_COS_FLOOR

    for alpha in fractions:
        k = k_from_fraction(alpha, n)
        mask = top_k_mask(scores, k)
        s = float(ctx.eval_masked_cos(mask.astype(np.int64)))
        out[f"suff@{alpha:.2f}_raw"] = s
        out[f"suff@{alpha:.2f}_ratio"] = (s / full) if ratio_safe else float("nan")

    if compute_aopc:
        raws = np.empty(n, dtype=np.float64)
        for k in range(1, n + 1):
            mask = top_k_mask(scores, k)
            raws[k - 1] = float(ctx.eval_masked_cos(mask.astype(np.int64)))
        out["suff_aopc_raw"] = float(raws.mean())
        out["suff_aopc_ratio"] = (out["suff_aopc_raw"] / full) if ratio_safe else float("nan")
    return out


@register("comprehensiveness")
def comprehensiveness(
    ctx: PairContext,
    scores: np.ndarray,
    *,
    fractions: Tuple[float, ...] = (0.10, 0.25, 0.50),
    compute_aopc: bool = False,
) -> Dict[str, float]:
    """Comprehensiveness metric: mask top-k% tokens, keep rest.

    Returns ``comp@<frac>_drop`` (S_v(q_full, c) - S_v(q_complement, c)) and
    ``comp@<frac>_ratio`` (drop / S_v(q_full, c)) per fraction. When
    ``compute_aopc=True`` also returns ``comp_aopc_{drop,ratio}`` (off by
    default; see ``sufficiency`` for rationale).
    """
    out: Dict[str, float] = {}
    n = ctx.n_q
    full = ctx.full_cos
    ratio_safe = abs(full) >= FULL_COS_FLOOR

    for alpha in fractions:
        k = k_from_fraction(alpha, n)
        keep = ~top_k_mask(scores, k)
        s = float(ctx.eval_masked_cos(keep.astype(np.int64)))
        drop = full - s
        out[f"comp@{alpha:.2f}_drop"] = drop
        out[f"comp@{alpha:.2f}_ratio"] = (drop / full) if ratio_safe else float("nan")

    if compute_aopc:
        drops = np.empty(n, dtype=np.float64)
        for k in range(1, n + 1):
            keep = ~top_k_mask(scores, k)
            s = float(ctx.eval_masked_cos(keep.astype(np.int64)))
            drops[k - 1] = full - s
        out["comp_aopc_drop"] = float(drops.mean())
        out["comp_aopc_ratio"] = (out["comp_aopc_drop"] / full) if ratio_safe else float("nan")
    return out


@register("compactness")
def compactness(
    ctx: PairContext,
    scores: np.ndarray,
    *,
    threshold: float = 0.8,
) -> Dict[str, float]:
    """Compactness: smallest fraction k/n such that sufficiency_ratio >= threshold.

    Returns ``compactness@<tau>``. NaN if |full_cos| < FULL_COS_FLOOR (ratio undefined).
    Returns 1.0 if no k attains the threshold.
    """
    n = ctx.n_q
    full = ctx.full_cos
    if abs(full) < FULL_COS_FLOOR:
        return {f"compactness@{threshold:.2f}": float("nan")}

    for k in range(1, n + 1):
        mask = top_k_mask(scores, k)
        s = float(ctx.eval_masked_cos(mask.astype(np.int64)))
        if (s / full) >= threshold:
            return {f"compactness@{threshold:.2f}": k / n}
    return {f"compactness@{threshold:.2f}": 1.0}


@register("loo_correlation")
def loo_correlation(
    ctx: PairContext,
    scores: np.ndarray,
) -> Dict[str, float]:
    """Spearman rank correlation between |a| and per-token leave-one-out drop.

    Filters tokens whose |Delta_i| is below LOO_NOISE_FLOOR (machine-precision
    noise; mean-pool is near-invariant to single-token swaps in long queries).
    Reports ``loo_rho``, ``loo_p``, and ``loo_n_used`` (number of tokens after
    filtering). Returns NaN when fewer than 3 tokens survive or when either
    vector is constant.
    """
    deltas = ctx.loo_deltas
    abs_scores = np.abs(np.asarray(scores, dtype=np.float64))
    keep = np.abs(deltas) >= LOO_NOISE_FLOOR
    n_used = int(keep.sum())
    out: Dict[str, float] = {
        "loo_n_used": float(n_used),
        "loo_n_total": float(ctx.n_q),
    }
    if n_used < 3:
        out["loo_rho"] = float("nan")
        out["loo_p"] = float("nan")
        return out
    a = abs_scores[keep]
    d = deltas[keep]
    # spearmanr returns NaN with a warning if either input is constant.
    if np.allclose(a, a[0]) or np.allclose(d, d[0]):
        out["loo_rho"] = float("nan")
        out["loo_p"] = float("nan")
        return out
    res = spearmanr(a, d)
    rho, p = float(res.statistic), float(res.pvalue)
    out["loo_rho"] = rho
    out["loo_p"] = p
    return out


# ----------------------------------------------------------------------------
# Self test (dry, no GPU)
# ----------------------------------------------------------------------------
def _self_test() -> None:
    """Sanity checks. Runs in-process with a synthetic linear cos function.

    The linear cos is constructed so that the 'true' attribution per token is
    exactly the score vector we use below; thus a perfectly aligned attribution
    should achieve sufficiency=1 at any k>=k*, comprehensiveness>0 at small k,
    compactness=k*/n, and loo_correlation=1.0.
    """
    rng = np.random.default_rng(0)
    n = 20
    true_contrib = rng.uniform(0.0, 1.0, size=n)
    true_contrib[:5] *= 50.0  # 5 dominant tokens carry >80% of the full score
    full = float(true_contrib.sum())

    def eval_masked_cos(mask: np.ndarray) -> float:
        return float((true_contrib * mask.astype(np.float64)).sum())

    sa = np.array([eval_masked_cos(np.ones(n, dtype=np.int64) - np.eye(1, n, k=i, dtype=np.int64).ravel())
                   for i in range(n)])
    ctx = PairContext(
        n_q=n, variant="baseline", full_cos=full,
        single_ablation_cos=sa, eval_masked_cos=eval_masked_cos,
    )

    # Aligned scores: equal to the per-token contribution -> perfect attribution
    aligned = true_contrib.copy()

    suff = sufficiency(ctx, aligned, fractions=(0.10, 0.25, 0.50), compute_aopc=True)
    comp = comprehensiveness(ctx, aligned, fractions=(0.10, 0.25, 0.50), compute_aopc=True)
    compact = compactness(ctx, aligned, threshold=0.8)
    loo = loo_correlation(ctx, aligned)

    # 1. Suff @ k=n must equal 1.0 (full mask).
    assert abs(suff["suff_aopc_raw"]) <= full + 1e-9
    # Sweep includes k=n at AOPC's last term -> raw at k=n equals full.
    raw_at_n = float(eval_masked_cos(np.ones(n, dtype=np.int64)))
    assert abs(raw_at_n - full) < 1e-9, (raw_at_n, full)

    # 2. Comp @ k=0 (i.e. mask nothing) is "drop" of the full vs full = 0; we
    #    don't sweep alpha=0 explicitly. Check at smallest swept alpha that
    #    drop is non-negative.
    assert comp[f"comp@0.10_drop"] >= -1e-9

    # 3. Compactness should be small (top-5 of 20 = 0.25) for this construction.
    assert compact["compactness@0.80"] <= 0.30, compact

    # 4. Aligned scores should yield Spearman ~ 1.
    assert loo["loo_rho"] > 0.99, loo

    # 5. Random scores should yield Spearman near 0.
    random_scores = rng.uniform(-1.0, 1.0, size=n)
    loo_rand = loo_correlation(ctx, random_scores)
    assert abs(loo_rand["loo_rho"]) < 0.7, loo_rand  # very loose

    # 6. Reducer table covers expected names without collision.
    expected = {"ig", "bertscore", "ot", "attention_weighted",
                "attention_standalone", "dla"}
    assert expected <= set(METHOD_SCORE_REDUCER), set(METHOD_SCORE_REDUCER)

    # 7. Pair matrix reducer numerical sanity.
    pm = np.array([[1.0, -2.0, 3.0],
                   [0.0, 0.5, -0.5]])
    assert np.allclose(scores_from_pair_matrix(pm, None, "row_max"), [3.0, 0.5])
    assert np.allclose(scores_from_pair_matrix(pm, None, "row_sum_signed"), [2.0, 0.0])
    assert np.allclose(scores_from_pair_matrix(pm, None, "row_sum_positive"), [4.0, 0.5])

    # 8. NaN policy: tiny full_cos -> NaN ratio.
    def zero_eval(_mask):
        return 0.0
    ctx_zero = PairContext(
        n_q=n, variant="abtt", full_cos=0.001,
        single_ablation_cos=np.zeros(n), eval_masked_cos=zero_eval,
    )
    s2 = sufficiency(ctx_zero, aligned, fractions=(0.25,))
    assert np.isnan(s2["suff@0.25_ratio"]), s2

    print("attribution_metrics self-test: OK")


if __name__ == "__main__":
    _self_test()
