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
  - ``aopc``                (area under the k=1..n sufficiency and comprehensiveness curves)
  - ``deletion_auc``        (AUC of S_v as tokens are deleted most-important-first, vs random)
  - ``insertion_auc``       (AUC of S_v as tokens are inserted most-important-first, vs random)
  - ``kendall_tau_loo``     (Kendall tau-b between |a| and the same LOO deltas rho_LOO uses)
  - ``randomization_check`` (Adebayo-style: shuffle ``a``, recompute, report real - shuffled)

The "loo_correlation" metric is what some papers call "faithfulness" (Atanasova et al.
2020); we keep the more specific name here to avoid conflating it with MarK's AOPC-style
faithfulness, which is captured by AOPC over sufficiency/comprehensiveness.

Cost model. ``aopc``, ``deletion_auc``, ``insertion_auc`` and ``randomization_check``
all need the full k=1..n masking curve rather than a handful of thresholds. A driver
that recomputes S_v with a model forward pass per mask pays O(n) forwards per curve; a
driver that masks a *frozen* pooled representation can hand ``PairContext`` a
``prefix_curves`` closure that returns both curves for a given token ordering in one
vectorised pass. Every curve-based metric here goes through ``PairContext.curves`` and
therefore gets the cheap path automatically when the driver supplies one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr

# Numerical constants. Pairs with full_cos below FULL_COS_FLOOR yield NaN metrics
# (the sufficiency ratio is unstable for small denominators). LOO correlation
# discards tokens with |Delta| below LOO_NOISE_FLOOR. The floor is set to
# 1e-6 -- machine-level noise on float32 cosines -- rather than a stricter
# 1e-4, because ABTT-cleaned embeddings are often near-invariant to single-token
# swaps (the residual subspace carries little signal); a stricter floor would
# discard the actual signal rather than pure noise on those pairs.
FULL_COS_FLOOR: float = 0.05
LOO_NOISE_FLOOR: float = 1e-6
DEFAULT_COMPACTNESS_THRESHOLDS: Tuple[float, ...] = (0.70, 0.80, 0.90, 0.95)

# Number of reference orderings averaged over. The random-order reference for
# deletion/insertion AUC and the shuffled-attribution reference for the
# randomization check are both Monte-Carlo estimates; 5 draws keeps the driver
# cheap while making the reference far less noisy than a single draw.
DEFAULT_RANDOM_ORDER_DRAWS: int = 5
DEFAULT_SHUFFLE_DRAWS: int = 5
# Fixed seeds so that every method within one (pair, variant) is scored against
# the *same* random orderings. That makes the method-to-method comparison paired
# and lets a driver cache the reference curves.
RANDOM_ORDER_SEED: int = 20260906
SHUFFLE_SEED: int = 20260907


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------
MaskedCosFn = Callable[[np.ndarray], float]
# Given a token ordering (most-important-first), return (keep_curve, drop_curve),
# each of length n+1 and indexed by k: keep_curve[k] = S_v(only order[:k] kept),
# drop_curve[k] = S_v(everything except order[:k] kept). By convention
# keep_curve[0] = drop_curve[n] = 0.0 (the empty query) and
# keep_curve[n] = drop_curve[0] = full_cos.
PrefixCurveFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
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

    ``prefix_curves`` is the optional fast path for curve-based metrics (AOPC,
    deletion/insertion AUC, the randomization check). A driver that masks a frozen
    pooled representation can compute both prefix curves for an ordering in one
    vectorised pass; when it is None the curves are assembled by calling
    ``eval_masked_cos`` n times per curve, which is correct but O(n) model
    forwards.
    """

    n_q: int
    variant: str
    full_cos: float
    single_ablation_cos: np.ndarray  # shape (n_q,)
    eval_masked_cos: MaskedCosFn
    metadata: Dict[str, object] = field(default_factory=dict)
    prefix_curves: Optional[PrefixCurveFn] = None
    # Memo keyed by token ordering. ``curves`` is a pure function of ``order``
    # for a fixed context, and the metrics ask for the same orderings more than
    # once: insertion_auc and deletion_auc both request the attribution order
    # and the same seeded random orders, and the randomization check requests
    # them again. On the ``model`` backend one curve costs 2(n-1) encoder
    # forwards, so the memo is the difference between a spot check that runs in
    # minutes and one that does not run at all. Mutating the dict is allowed on
    # a frozen dataclass; it is excluded from equality and repr.
    _curve_cache: Dict[bytes, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, compare=False, repr=False)

    @property
    def loo_deltas(self) -> np.ndarray:
        """Per-token leave-one-out drop in cosine: full_cos - cos(q\\{i}, c)."""
        return self.full_cos - self.single_ablation_cos

    @property
    def has_fast_curves(self) -> bool:
        """True when the driver supplied a vectorised prefix-curve closure."""
        return self.prefix_curves is not None

    def curves(self, order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(keep_curve, drop_curve)`` for ``order``, both length n_q + 1.

        ``order`` is a permutation of ``range(n_q)``, most important first.
        ``keep_curve[k]`` is S_v with only ``order[:k]`` kept, ``drop_curve[k]``
        is S_v with ``order[:k]`` removed. Index 0 and index n are pinned to the
        analytic endpoints so the empty-query convention is identical on both
        the fast and the fallback path.
        """
        order = np.asarray(order, dtype=np.int64)
        n = self.n_q
        if order.shape != (n,):
            raise ValueError(f"order must have shape ({n},), got {order.shape}")
        cache_key = order.tobytes()
        cached = self._curve_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.prefix_curves is not None:
            keep, drop = self.prefix_curves(order)
            # Copy rather than view: the endpoints are pinned below, and the
            # result is cached, so we must own the buffer instead of writing
            # into whatever the driver handed back.
            keep = np.array(keep, dtype=np.float64, copy=True)
            drop = np.array(drop, dtype=np.float64, copy=True)
            if keep.shape != (n + 1,) or drop.shape != (n + 1,):
                raise ValueError("prefix_curves must return two length-(n+1) arrays")
        else:
            keep = np.empty(n + 1, dtype=np.float64)
            drop = np.empty(n + 1, dtype=np.float64)
            for k in range(1, n):
                mask = np.zeros(n, dtype=np.int64)
                mask[order[:k]] = 1
                keep[k] = float(self.eval_masked_cos(mask))
                drop[k] = float(self.eval_masked_cos(1 - mask))
        keep[0] = 0.0
        keep[n] = float(self.full_cos)
        drop[0] = float(self.full_cos)
        drop[n] = 0.0
        keep.setflags(write=False)
        drop.setflags(write=False)
        self._curve_cache[cache_key] = (keep, drop)
        return keep, drop


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
def rank_order(scores: np.ndarray) -> np.ndarray:
    """Token indices sorted by descending ``|score|``, ties broken by position.

    This is the single definition of "attribution order" used by every
    curve-based metric, and it agrees with :func:`top_k_mask` by construction:
    ``top_k_mask(scores, k)`` is exactly the mask of ``rank_order(scores)[:k]``.
    """
    abs_scores = np.abs(np.asarray(scores, dtype=np.float64))
    return np.argsort(-abs_scores, kind="stable")


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
        keep_curve, _ = ctx.curves(rank_order(scores))
        out["suff_aopc_raw"] = float(keep_curve[1:].mean())
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
        _, drop_curve = ctx.curves(rank_order(scores))
        out["comp_aopc_drop"] = float((full - drop_curve[1:]).mean())
        out["comp_aopc_ratio"] = (out["comp_aopc_drop"] / full) if ratio_safe else float("nan")
    return out


@register("compactness")
def compactness(
    ctx: PairContext,
    scores: np.ndarray,
    *,
    threshold: Optional[float] = None,
    thresholds: Optional[Tuple[float, ...]] = None,
) -> Dict[str, float]:
    """Minimum recovery fraction: smallest k/n with sufficiency_ratio >= tau.

    Returns ``compactness@<tau>`` for each requested tau. This is a
    sparsity/recovery-at-threshold metric, not MaRC's contiguity regularizer;
    the key name is kept for compatibility with existing artifacts.

    NaN if |full_cos| < FULL_COS_FLOOR (ratio undefined). Returns 1.0 if no
    k attains a threshold. When multiple thresholds are requested, all
    thresholds are computed in one scan over k to avoid repeated model
    forward passes in the GPU-backed driver.
    """
    if threshold is not None and thresholds is not None:
        raise ValueError("pass either threshold or thresholds, not both")
    if thresholds is None:
        thresholds = (threshold,) if threshold is not None else DEFAULT_COMPACTNESS_THRESHOLDS
    thresholds = tuple(float(t) for t in thresholds)
    out: Dict[str, float] = {f"compactness@{t:.2f}": float("nan") for t in thresholds}

    n = ctx.n_q
    full = ctx.full_cos
    if abs(full) < FULL_COS_FLOOR:
        return out

    remaining = set(thresholds)

    # With a vectorised prefix-curve closure the whole k=1..n curve costs the
    # same as one point, so take it in one shot; otherwise keep the incremental
    # loop, which can stop early and save model forward passes.
    keep_curve = ctx.curves(rank_order(scores))[0] if ctx.has_fast_curves else None

    for k in range(1, n + 1):
        if keep_curve is not None:
            s = float(keep_curve[k])
        else:
            mask = top_k_mask(scores, k)
            s = float(ctx.eval_masked_cos(mask.astype(np.int64)))
        ratio = s / full
        for t in tuple(remaining):
            if ratio >= t:
                out[f"compactness@{t:.2f}"] = k / n
                remaining.remove(t)
        if not remaining:
            return out

    for t in remaining:
        out[f"compactness@{t:.2f}"] = 1.0
    return out


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


@register("kendall_tau_loo")
def kendall_tau_loo(
    ctx: PairContext,
    scores: np.ndarray,
) -> Dict[str, float]:
    """Kendall tau-b between ``|a|`` and the per-token leave-one-out drop.

    This is pure post-processing of the same ``ctx.loo_deltas`` vector that
    ``loo_correlation`` consumes, under the same LOO_NOISE_FLOOR filter, so it
    costs nothing extra and is directly comparable to ``loo_rho``. Tau-b is the
    tie-corrected variant, which matters because attribution methods with
    built-in sparsity produce many exactly-zero scores. Reports ``loo_tau``,
    ``loo_tau_p`` and ``loo_tau_n_used``; NaN under the same degenerate
    conditions as ``loo_rho``.
    """
    deltas = ctx.loo_deltas
    abs_scores = np.abs(np.asarray(scores, dtype=np.float64))
    keep = np.abs(deltas) >= LOO_NOISE_FLOOR
    n_used = int(keep.sum())
    out: Dict[str, float] = {"loo_tau_n_used": float(n_used)}
    if n_used < 3:
        out["loo_tau"] = float("nan")
        out["loo_tau_p"] = float("nan")
        return out
    a = abs_scores[keep]
    d = deltas[keep]
    if np.allclose(a, a[0]) or np.allclose(d, d[0]):
        out["loo_tau"] = float("nan")
        out["loo_tau_p"] = float("nan")
        return out
    res = kendalltau(a, d, variant="b")
    out["loo_tau"] = float(res.statistic)
    out["loo_tau_p"] = float(res.pvalue)
    return out


def _aopc_from_curves(keep_curve: np.ndarray, drop_curve: np.ndarray,
                      full: float) -> Tuple[float, float]:
    """Raw sufficiency and comprehensiveness AOPC from one pair of curves.

    Both are means over k = 1..n, matching DeYoung et al. 2020's "average over
    bins" formulation with every k its own bin.
    """
    suff_raw = float(keep_curve[1:].mean())
    comp_raw = float((full - drop_curve[1:]).mean())
    return suff_raw, comp_raw


@register("aopc")
def aopc(
    ctx: PairContext,
    scores: np.ndarray,
) -> Dict[str, float]:
    """Area over the perturbation curve for sufficiency and comprehensiveness.

    Sweeps the whole k = 1..n grid instead of the three fixed fractions that
    ``sufficiency`` and ``comprehensiveness`` report, so the number no longer
    depends on a threshold choice. Both are reported higher-is-better, matching
    the ratio convention used elsewhere in this module:

      ``aopc_suff_ratio`` = mean_k S_v(top-k kept) / S_v(full)
      ``aopc_comp_ratio`` = mean_k [S_v(full) - S_v(top-k removed)] / S_v(full)

    ``aopc_suff_ratio`` is therefore the complement of DeYoung's sufficiency
    (they report the drop, where lower is better); the raw columns
    ``aopc_suff_raw`` / ``aopc_comp_raw`` carry the un-normalised cosines.
    These are numerically identical to the ``suff_aopc_*`` / ``comp_aopc_*``
    keys emitted by ``sufficiency(..., compute_aopc=True)`` and
    ``comprehensiveness(..., compute_aopc=True)``; the duplication is deliberate
    so the flag stays backward compatible while AOPC also exists as a
    first-class registry entry.
    """
    full = ctx.full_cos
    ratio_safe = abs(full) >= FULL_COS_FLOOR
    keep_curve, drop_curve = ctx.curves(rank_order(scores))
    suff_raw, comp_raw = _aopc_from_curves(keep_curve, drop_curve, full)
    return {
        "aopc_suff_raw": suff_raw,
        "aopc_suff_ratio": (suff_raw / full) if ratio_safe else float("nan"),
        "aopc_comp_raw": comp_raw,
        "aopc_comp_ratio": (comp_raw / full) if ratio_safe else float("nan"),
    }


def _curve_auc(values: np.ndarray) -> float:
    """Trapezoidal AUC of ``values`` (length n+1) over x = k/n in [0, 1]."""
    n = len(values) - 1
    if n < 1:
        return float("nan")
    x = np.arange(n + 1, dtype=np.float64) / n
    return float(np.trapezoid(values, x)) if hasattr(np, "trapezoid") else float(np.trapz(values, x))


def _random_orders(n: int, draws: int, seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        yield rng.permutation(n)


@register("deletion_auc")
def deletion_auc(
    ctx: PairContext,
    scores: np.ndarray,
    *,
    random_draws: int = DEFAULT_RANDOM_ORDER_DRAWS,
    seed: int = RANDOM_ORDER_SEED,
) -> Dict[str, float]:
    """Petsiuk-style deletion AUC: remove tokens most-important-first.

    The curve is ``S_v(remaining) / S_v(full)`` against the fraction of tokens
    already removed, from 1.0 at k=0 down to 0.0 at k=n. A faithful attribution
    destroys the score early, so **lower is better**.

    The number is only interpretable against a reference, because the curve's
    height is dominated by how redundant the query is rather than by the
    attribution: we therefore also score ``random_draws`` uniformly random
    deletion orders and report the gap. ``del_auc_gap = random - attribution``
    is higher-is-better and is the part of the metric that actually measures the
    ranking. The random orders are drawn from a fixed seed, so every method
    scored on the same pair is compared against the same reference.
    """
    full = ctx.full_cos
    if abs(full) < FULL_COS_FLOOR:
        return {"del_auc": float("nan"), "del_auc_random": float("nan"),
                "del_auc_gap": float("nan")}
    _, drop_curve = ctx.curves(rank_order(scores))
    attr = _curve_auc(drop_curve / full)
    rand = [
        _curve_auc(ctx.curves(order)[1] / full)
        for order in _random_orders(ctx.n_q, random_draws, seed)
    ]
    rand_mean = float(np.mean(rand)) if rand else float("nan")
    return {
        "del_auc": attr,
        "del_auc_random": rand_mean,
        "del_auc_gap": rand_mean - attr,
    }


@register("insertion_auc")
def insertion_auc(
    ctx: PairContext,
    scores: np.ndarray,
    *,
    random_draws: int = DEFAULT_RANDOM_ORDER_DRAWS,
    seed: int = RANDOM_ORDER_SEED,
) -> Dict[str, float]:
    """Petsiuk-style insertion AUC: build the query up most-important-first.

    The curve is ``S_v(kept) / S_v(full)`` against the fraction of tokens
    inserted, from 0.0 at k=0 up to 1.0 at k=n. A faithful attribution recovers
    the score early, so **higher is better**. As with ``deletion_auc`` the
    random-order reference and the gap ``ins_auc_gap = attribution - random``
    are what isolate the ranking from the query's intrinsic redundancy. The same
    seed is used as for deletion, so the two share their reference orderings.
    """
    full = ctx.full_cos
    if abs(full) < FULL_COS_FLOOR:
        return {"ins_auc": float("nan"), "ins_auc_random": float("nan"),
                "ins_auc_gap": float("nan")}
    keep_curve, _ = ctx.curves(rank_order(scores))
    attr = _curve_auc(keep_curve / full)
    rand = [
        _curve_auc(ctx.curves(order)[0] / full)
        for order in _random_orders(ctx.n_q, random_draws, seed)
    ]
    rand_mean = float(np.mean(rand)) if rand else float("nan")
    return {
        "ins_auc": attr,
        "ins_auc_random": rand_mean,
        "ins_auc_gap": attr - rand_mean,
    }


# Keys the shuffled-attribution control evaluates. Every curve metric that can
# enter the main table has to appear here, otherwise criterion 5 of the
# selection memo is applied to some candidates and waived for others. In
# particular ``ins_auc_gap`` and ``del_auc_gap`` are listed explicitly rather
# than left to be inferred from their AOPC twins.
RANDOMIZATION_CHECK_KEYS: Tuple[str, ...] = (
    "loo_rho",
    "loo_tau",
    "aopc_suff_ratio",
    "aopc_comp_ratio",
    "ins_auc_gap",
    "del_auc_gap",
)


def _curve_stats(ctx: "PairContext", scores: np.ndarray,
                 full: float, ratio_safe: bool) -> Tuple[float, float, float, float]:
    """One pass over the curves for ``scores``: both AOPC ratios and both AUCs.

    Returns ``(aopc_suff_ratio, aopc_comp_ratio, ins_auc, del_auc)``, each NaN
    when the full cosine is below ``FULL_COS_FLOOR``. Identical by construction
    to what :func:`aopc`, :func:`insertion_auc` and :func:`deletion_auc` return
    for the same ``scores``; sharing one ``ctx.curves`` call is what keeps the
    shuffled-attribution control affordable over six keys instead of four.
    """
    if not ratio_safe:
        nan = float("nan")
        return nan, nan, nan, nan
    keep_curve, drop_curve = ctx.curves(rank_order(scores))
    suff_raw, comp_raw = _aopc_from_curves(keep_curve, drop_curve, full)
    return (
        suff_raw / full,
        comp_raw / full,
        _curve_auc(keep_curve / full),
        _curve_auc(drop_curve / full),
    )


@register("randomization_check")
def randomization_check(
    ctx: PairContext,
    scores: np.ndarray,
    *,
    shuffle_draws: int = DEFAULT_SHUFFLE_DRAWS,
    seed: int = SHUFFLE_SEED,
    random_draws: int = DEFAULT_RANDOM_ORDER_DRAWS,
    random_seed: int = RANDOM_ORDER_SEED,
) -> Dict[str, float]:
    """Adebayo-style sanity check: permute ``a``, recompute, report the gap.

    Randomly permuting the attribution vector destroys the token-to-score
    assignment while preserving the score distribution exactly, so anything the
    shuffled version still scores is an artefact of the metric, the query, or
    the score distribution rather than evidence that the attribution located
    anything. A metric on which real and shuffled attributions score the same is
    not measuring attribution quality on this data.

    Reports, for each key in :data:`RANDOMIZATION_CHECK_KEYS`, the mean over
    ``shuffle_draws`` permutations (``rand_<key>``) and the signed gap
    ``rand_<key>_gap = real - shuffled``. Every underlying key is
    higher-is-better, so a positive gap is the passing direction.

    The two chance-corrected AUC keys are included so the control covers the
    metrics that are actually candidates for the paper's main table. Their
    random-*order* reference (``ins_auc_random`` / ``del_auc_random``) is a
    property of the pair, not of the attribution, so it is identical for the
    real and every shuffled attribution and is computed once here; it therefore
    cancels in the gap, which makes ``rand_ins_auc_gap_gap`` algebraically equal
    to ``rand_aopc_suff_ratio_gap`` and ``rand_del_auc_gap_gap`` equal to
    ``rand_aopc_comp_ratio_gap`` (the mean-over-k versus trapezoid offset
    ``1/(2n)`` is a constant that also cancels). Reporting them explicitly
    rather than leaving them to be derived is the point: the selection memo
    applies its shuffled-attribution criterion to a column only if that column
    has a number behind it.
    """
    full = ctx.full_cos
    ratio_safe = abs(full) >= FULL_COS_FLOOR
    scores = np.asarray(scores, dtype=np.float64)

    # Random-order reference, shared by the real and every shuffled attribution.
    if ratio_safe and random_draws > 0:
        ins_refs: List[float] = []
        del_refs: List[float] = []
        for order in _random_orders(ctx.n_q, random_draws, random_seed):
            keep_curve, drop_curve = ctx.curves(order)
            ins_refs.append(_curve_auc(keep_curve / full))
            del_refs.append(_curve_auc(drop_curve / full))
        ins_ref = float(np.mean(ins_refs))
        del_ref = float(np.mean(del_refs))
    else:
        ins_ref = float("nan")
        del_ref = float("nan")

    def _all_keys(vec: np.ndarray) -> Dict[str, float]:
        suff, comp, ins_auc_v, del_auc_v = _curve_stats(ctx, vec, full, ratio_safe)
        return {
            "loo_rho": loo_correlation(ctx, vec)["loo_rho"],
            "loo_tau": kendall_tau_loo(ctx, vec)["loo_tau"],
            "aopc_suff_ratio": suff,
            "aopc_comp_ratio": comp,
            "ins_auc_gap": ins_auc_v - ins_ref,
            "del_auc_gap": del_ref - del_auc_v,
        }

    real = _all_keys(scores)

    rng = np.random.default_rng(seed)
    collected: Dict[str, List[float]] = {k: [] for k in RANDOMIZATION_CHECK_KEYS}
    for _ in range(shuffle_draws):
        shuffled_vals = _all_keys(rng.permutation(scores))
        for key in RANDOMIZATION_CHECK_KEYS:
            collected[key].append(shuffled_vals[key])

    out: Dict[str, float] = {}
    for key in RANDOMIZATION_CHECK_KEYS:
        arr = np.asarray(collected[key], dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        shuffled_mean = float(finite.mean()) if finite.size else float("nan")
        out[f"rand_{key}"] = shuffled_mean
        out[f"rand_{key}_gap"] = float(real[key]) - shuffled_mean
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
    compact = compactness(ctx, aligned, thresholds=DEFAULT_COMPACTNESS_THRESHOLDS)
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
    assert set(compact) == {f"compactness@{t:.2f}" for t in DEFAULT_COMPACTNESS_THRESHOLDS}
    assert compactness(ctx, aligned, threshold=0.8) == {
        "compactness@0.80": compact["compactness@0.80"]
    }

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

    # 9. The --aopc flag and the registered `aopc` metric must agree exactly.
    a_metric = aopc(ctx, aligned)
    assert np.isclose(a_metric["aopc_suff_raw"], suff["suff_aopc_raw"]), (a_metric, suff)
    assert np.isclose(a_metric["aopc_comp_raw"], comp["comp_aopc_drop"]), (a_metric, comp)

    # 10. Insertion beats random, deletion undercuts random, for aligned scores.
    ins = insertion_auc(ctx, aligned)
    dele = deletion_auc(ctx, aligned)
    assert ins["ins_auc_gap"] > 0, ins
    assert dele["del_auc_gap"] > 0, dele

    # 11. Kendall tau agrees in sign with Spearman on the aligned attribution.
    tau = kendall_tau_loo(ctx, aligned)
    assert tau["loo_tau"] > 0.99, tau

    # 12. Randomization: the aligned attribution must beat its own shuffles.
    rand = randomization_check(ctx, aligned, shuffle_draws=5)
    assert rand["rand_loo_rho_gap"] > 0, rand
    assert rand["rand_aopc_suff_ratio_gap"] > 0, rand

    print("attribution_metrics self-test: OK")


if __name__ == "__main__":
    _self_test()
