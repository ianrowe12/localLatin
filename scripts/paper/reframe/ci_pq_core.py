"""Vectorized evaluation, directory bootstrap and routing checks (issue #233).

Pure numpy; no file I/O. ``ci_pq.py`` feeds it similarity matrices built by the
paper's own pipeline and uses it for three things the paper's evaluator
(``scripts/resubmit/run_resubmit_evaluate.py``) cannot do quickly:

1. Per-unit outcomes. Task A AUROC is a function of the test pair scores; Task B
   assignment accuracy and DirAcc@1 are means of per-file 0/1 outcomes. The
   evaluator only returns the aggregates. Here the same quantities come back per
   pair and per file, so a bootstrap can reweight them. ``ci_pq.py`` checks
   that the aggregates equal the evaluator's to the last count before it
   resamples anything.
2. Directory-level bootstrap weights. A replicate draws the 514 test
   directories with replacement. A directory drawn ``c`` times contributes each
   of its files ``c`` times, each of its within-directory (positive) pairs ``c``
   times, and each pair it forms with a directory drawn ``c'`` times ``c * c'``
   times. Pairs between two copies of the same directory are not formed: a
   file is never paired with its own copy. All scores are computed once on the
   full test set, with the threshold, principal components and SIF
   probabilities fit on the fixed training split, so only the test evaluation
   is resampled.
3. Threshold searches that reproduce ``sweep_thresholds`` on the paper's grid
   exactly and extend it to a fine grid and to the exact best-F1 cut.

Tie and precision conventions follow the evaluator as run for the paper
(NumPy 1.26): grid thresholds are compared with float32 scores in float32, and
the per-file max-cosine comparisons with tau are made in float64.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

PAPER_GRID = np.linspace(0, 1, 200)
FINE_GRID = np.linspace(0, 1, 10001)


# --------------------------------------------------------------------------- #
# Threshold search (best F1 over same- vs different-directory pairs)
# --------------------------------------------------------------------------- #


def pair_arrays(sim: np.ndarray, folder_ids: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Upper-triangle scores and same-directory labels, in ``upper_triangle`` order."""
    idx = np.triu_indices(sim.shape[0], k=1)
    fids = np.asarray(folder_ids)
    return sim[idx], fids[idx[0]] == fids[idx[1]]


def _f1_from_counts(tp: np.ndarray, count_ge: np.ndarray, n_pos: int) -> np.ndarray:
    """F1 per threshold with the operation order of ``sweep_thresholds``."""
    tp = tp.astype(np.float64)
    fp = count_ge.astype(np.float64) - tp
    fn = float(n_pos) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    return f1


def fit_tau(
    sims: np.ndarray,
    labels: np.ndarray,
    grid: str = "paper",
) -> float:
    """Best-F1 threshold over train pairs.

    ``grid``: ``"paper"`` is ``linspace(0, 1, 200)`` and returns exactly what
    ``run_resubmit_evaluate.learn_tau_from_similarity`` returns; ``"fine"`` is
    ``linspace(0, 1, 10001)`` (step 1e-4); ``"exact"`` searches every distinct
    train score, which is the limit of any finer or quantile grid. Ties in F1
    go to the lowest threshold, as ``DataFrame.idxmax`` does.
    """
    labels = np.asarray(labels, dtype=bool)
    s_all = np.sort(sims)
    s_pos = np.sort(sims[labels])
    n, n_pos = len(s_all), len(s_pos)
    if grid in ("paper", "fine"):
        thresholds = PAPER_GRID if grid == "paper" else FINE_GRID
        # sweep_thresholds compares a score array with a Python float, which
        # NumPy evaluates in the array's dtype.
        t_cmp = thresholds.astype(sims.dtype)
    elif grid == "exact":
        thresholds = np.unique(sims).astype(np.float64)
        t_cmp = thresholds.astype(sims.dtype)
    else:
        raise ValueError(f"unknown grid {grid!r}")
    count_ge = n - np.searchsorted(s_all, t_cmp, side="left")
    tp = n_pos - np.searchsorted(s_pos, t_cmp, side="left")
    f1 = _f1_from_counts(tp, count_ge, n_pos)
    return float(thresholds[int(np.argmax(f1))])


# --------------------------------------------------------------------------- #
# Task B per-file outcomes
# --------------------------------------------------------------------------- #


@dataclass
class TaskBUnits:
    """Per-test-file quantities behind the Task B aggregates."""

    max_cos: np.ndarray        # float64, best score to any other test file
    top_dir_correct: np.ndarray  # bool, best directory is the file's own
    existing: np.ndarray       # bool, file has a same-directory partner in test
    assign_correct: np.ndarray  # bool, existing-vs-new decision right at tau
    dir1_correct: np.ndarray   # bool, DirAcc@1 hit at tau


def directory_scores(sim: np.ndarray, folder_ids: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Max score of each file to each directory, excluding the file itself.

    Returns ``(dir_sim, dir_labels)``: ``dir_sim`` is ``n x n_dirs`` float64 with
    ``-inf`` where a directory's only member is the file itself, and the
    directories are sorted lexicographically (Python ``str`` order).
    """
    fids = np.asarray([str(f) for f in folder_ids], dtype=object)
    labels, inv = np.unique(fids.astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    s = sim.astype(np.float64, copy=True)
    np.fill_diagonal(s, -np.inf)
    s = s[:, order]
    starts = np.flatnonzero(np.r_[True, np.diff(inv[order]) != 0])
    dir_sim = np.maximum.reduceat(s, starts, axis=1)
    return dir_sim, labels


def taskb_units(
    sim: np.ndarray,
    folder_ids: Sequence,
    has_partner: Sequence[bool],
    tau: float,
) -> TaskBUnits:
    """Per-file Task B outcomes at threshold ``tau``.

    ``assign_correct`` is ``compute_assignment_acc``: an existing file is right
    when its max score is >= tau, a new file when it is < tau.
    ``dir1_correct`` is ``directory_assignment_accuracy_at_k(k=1)``: directories
    and a ``__NEW__`` entry scoring tau are ranked by score, ties broken by
    label, and the top entry must be the file's directory (existing) or
    ``__NEW__`` (new).
    """
    has_partner = np.asarray(has_partner, dtype=bool)
    dir_sim, labels = directory_scores(sim, folder_ids)
    fids = np.asarray([str(f) for f in folder_ids])
    own = np.searchsorted(labels, fids)
    best = dir_sim.max(axis=1)
    # argmax takes the first maximum, i.e. the lexicographically smallest label.
    best_idx = np.argmax(dir_sim, axis=1)
    top_is_own = best_idx == own
    new_label = "__NEW__"
    tau = float(tau)
    new_wins = np.array(
        [
            (b < tau) or (b == tau and new_label < labels[j])
            for b, j in zip(best, best_idx)
        ],
        dtype=bool,
    )
    dir1 = np.where(has_partner, top_is_own & ~new_wins, new_wins)
    assign = np.where(has_partner, best >= tau, best < tau)
    return TaskBUnits(
        max_cos=best,
        top_dir_correct=top_is_own,
        existing=has_partner,
        assign_correct=assign,
        dir1_correct=dir1,
    )


def at_threshold(units: TaskBUnits, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(assign_correct, dir1_correct)`` for another threshold, same scores.

    Exact for DirAcc@1 except at a score tied with tau, where the label
    tie-break of :func:`taskb_units` would be needed; callers that need exact
    agreement use :func:`taskb_units` itself.
    """
    ge = units.max_cos >= tau
    assign = np.where(units.existing, ge, ~ge)
    dir1 = np.where(units.existing, units.top_dir_correct & ge, ~ge)
    return assign, dir1


def train_dir_acc_at_1(sim: np.ndarray, folder_ids: Sequence, tau: float) -> float:
    """``directory_assignment_accuracy_at_k(k=1)`` on a train matrix."""
    fids = np.asarray([str(f) for f in folder_ids])
    _, counts = np.unique(fids, return_counts=True)
    multi = set(np.unique(fids)[counts > 1])
    has_partner = np.array([f in multi for f in fids], dtype=bool)
    return float(taskb_units(sim, fids, has_partner, tau).dir1_correct.mean())


# --------------------------------------------------------------------------- #
# Weighted AUROC (Mann-Whitney with half credit for ties)
# --------------------------------------------------------------------------- #


@dataclass
class RankIndex:
    """Scores sorted once, so each bootstrap replicate is a cumulative sum."""

    order: np.ndarray   # argsort of the scores
    pos_lo: np.ndarray  # for each positive: first sorted index of its score
    pos_hi: np.ndarray  # for each positive: one past the last sorted index
    pos_idx: np.ndarray  # original indices of the positives
    is_pos_sorted: np.ndarray


def rank_index(scores: np.ndarray, is_pos: np.ndarray) -> RankIndex:
    scores = np.asarray(scores)
    is_pos = np.asarray(is_pos, dtype=bool)
    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    pos_idx = np.flatnonzero(is_pos)
    lo = np.searchsorted(s_sorted, scores[pos_idx], side="left")
    hi = np.searchsorted(s_sorted, scores[pos_idx], side="right")
    return RankIndex(order, lo, hi, pos_idx, is_pos[order])


def weighted_auroc(ri: RankIndex, w: np.ndarray) -> np.ndarray:
    """AUROC under unit weights ``w`` (shape ``(n_units,)`` or ``(b, n_units)``).

    Equals ``sklearn.metrics.roc_auc_score(labels, scores, sample_weight=w)``.
    """
    w = np.atleast_2d(np.asarray(w, dtype=np.float64))
    w_sorted = w[:, ri.order]
    w_neg = np.where(ri.is_pos_sorted[None, :], 0.0, w_sorted)
    c = np.concatenate([np.zeros((w.shape[0], 1)), np.cumsum(w_neg, axis=1)], axis=1)
    below = c[:, ri.pos_lo]
    tied = c[:, ri.pos_hi] - below
    w_pos = w[:, ri.pos_idx]
    num = (w_pos * (below + 0.5 * tied)).sum(axis=1)
    den = w_pos.sum(axis=1) * c[:, -1]
    return num / den


# --------------------------------------------------------------------------- #
# Directory bootstrap
# --------------------------------------------------------------------------- #


def directory_codes(folder_ids: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """``(labels, code)``: sorted unique directories and each file's index."""
    labels, code = np.unique(np.asarray([str(f) for f in folder_ids]), return_inverse=True)
    return labels, code


def bootstrap_counts(n_dirs: int, B: int, seed: int) -> np.ndarray:
    """``B x n_dirs`` draw counts: ``n_dirs`` directories drawn with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_dirs, np.full(n_dirs, 1.0 / n_dirs), size=B).astype(np.float64)


@dataclass
class PairDirs:
    """Directory codes of the two files of every upper-triangle test pair."""

    a: np.ndarray
    b: np.ndarray
    is_pos: np.ndarray

    @classmethod
    def from_codes(cls, code: np.ndarray) -> "PairDirs":
        i, j = np.triu_indices(len(code), k=1)
        return cls(code[i], code[j], code[i] == code[j])

    def weights(self, counts: np.ndarray) -> np.ndarray:
        """Pair weights for each replicate (rows of ``counts``)."""
        counts = np.atleast_2d(counts)
        wa = counts[:, self.a]
        wb = counts[:, self.b]
        return np.where(self.is_pos[None, :], wa, wa * wb)


def percentile_ci(values: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Routing checks
# --------------------------------------------------------------------------- #


def oracle_accuracy(max_cos: np.ndarray, credit_if_ge: np.ndarray, is_new: np.ndarray,
                    w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Best accuracy over every threshold, and the threshold that attains it.

    A file earns credit ``credit_if_ge`` when its score is >= t and ``is_new``
    when it is < t (assignment: credit_if_ge = existing; DirAcc@1: credit_if_ge
    = existing and own directory on top). Cuts are placed at every distinct
    score and above the maximum. ``w`` may be ``(b, n)`` replicate weights.
    """
    n = len(max_cos)
    w = np.ones((1, n)) if w is None else np.atleast_2d(np.asarray(w, dtype=np.float64))
    uniq = np.unique(max_cos)
    cuts = np.r_[uniq, np.inf]
    # group index of each file among the distinct scores
    g = np.searchsorted(uniq, max_cos)
    ge_credit = np.zeros((w.shape[0], len(uniq)))
    lt_credit = np.zeros((w.shape[0], len(uniq)))
    for r in range(w.shape[0]):
        ge_credit[r] = np.bincount(g, weights=w[r] * credit_if_ge, minlength=len(uniq))
        lt_credit[r] = np.bincount(g, weights=w[r] * is_new, minlength=len(uniq))
    # threshold at cuts[k]: files in groups >= k are ">= t"
    ge_tail = np.concatenate([np.cumsum(ge_credit[:, ::-1], axis=1)[:, ::-1],
                              np.zeros((w.shape[0], 1))], axis=1)
    lt_head = np.concatenate([np.zeros((w.shape[0], 1)), np.cumsum(lt_credit, axis=1)], axis=1)
    acc = (ge_tail + lt_head) / w.sum(axis=1, keepdims=True)
    k = np.argmax(acc, axis=1)
    return acc[np.arange(acc.shape[0]), k], cuts[k]


def k_occurrence(sim: np.ndarray, k: int = 10) -> np.ndarray:
    """N_k: how often each point is among the k nearest neighbours of the others.

    Neighbours by cosine, excluding the point itself (Radovanovic et al., 2010).
    """
    s = sim.astype(np.float64, copy=True)
    np.fill_diagonal(s, -np.inf)
    nn = np.argpartition(-s, k, axis=1)[:, :k]
    return np.bincount(nn.ravel(), minlength=sim.shape[0])


def skewness(x: np.ndarray) -> float:
    """Standardized third moment, E[(x - mu)^3] / sigma^3 (population moments)."""
    x = np.asarray(x, dtype=np.float64)
    d = x - x.mean()
    sd = np.sqrt((d ** 2).mean())
    return float((d ** 3).mean() / sd ** 3) if sd > 0 else float("nan")


def hubness(sim: np.ndarray, k: int = 10) -> Dict[str, float]:
    nk = k_occurrence(sim, k)
    return {
        "hub_skew": skewness(nk),
        "hub_max": float(nk.max()),
        "hub_antihub_share": float((nk == 0).mean()),
    }
