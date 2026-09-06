"""Where do embeddings beat surface overlap? (issue #132)

Issue #122 / PR #130 found that a character 3--5-gram TF-IDF cosine matches or
beats every embedding configuration on Task A and Task B. That is a corpus-level
summary, and it does not say *where* the two families disagree. This script
splits the test-set positive pairs by how much surface overlap they actually
have and asks each method to retrieve the partner inside each slice.

Design
------
The stratifying variable is the **raw** character 3--5-gram TF-IDF cosine of a
pair, with the vectoriser fitted on train files only (the same scorer as
``lexical_baselines.tfidf_char35``, before its train min-max rescaling: the
rescaling is monotone, so it moves no boundary, but the raw cosine is the
readable quantity). Positive pairs are same-directory test pairs; they are cut
into overlap terciles, plus a ``hard`` slice of positives that fall *below* the
TF-IDF Task-A decision threshold, i.e. the pairs the lexical scorer itself would
call unrelated. The hard slice deliberately overlaps the low tercile.

Three method families are scored on exactly the same pairs:

``tfidf_char35``
    The lexical scorer, on the train-rescaled scale so its ``tau`` is the one
    reported in PR #130.
``baseline_<model>``
    Mean-pooled hidden states with no correction, at the layer the paper picks
    for the Baseline column (highest train AUROC).
``abtt_<model>``
    The same mean-pooled hidden states at the paper's ABTT layer and ``D``, with
    the :class:`sif_abtt.EmbeddingCleaner` fitted on train embeddings and applied
    to all. This is ABTT only, no SIF weighting, matching the ABTT column of the
    headline tables.

Layer, ``D``, the representation and the pooling all come from
``phase_resubmit_results.csv`` rather than being re-tuned here, so the
configurations are the paper's. The check that they really are is that each
``tau`` re-learned here reproduces the one in that CSV; a disagreement stops the
run rather than warning, because it means the results CSV and the split have
come apart and nothing computed afterwards would be the paper's configuration.

Cached embedding matrices are keyed by row index, and issue #131's key
corrections renumbered 17 ``file_id``s without touching a byte of text, so the
caches are re-ordered here by **filename** (``phase9_bases/row_order.csv``)
rather than read positionally. That makes the run immune to that permutation and
to any future one.

Metrics, per method and per stratum
-----------------------------------
``recall_at_1`` / ``mrr``
    A positive pair contributes two *directed* observations, one per endpoint
    used as the query. The partner's rank is taken over the test files, with the
    query's **other** partners removed from the candidate pool: a query whose
    directory holds four test files would otherwise be unable to put more than
    one of its three partners at rank 1, which would confound directory size
    with the overlap strata (large directories skew high-overlap). The literal
    "top-1 among all test files" variant is reported alongside as
    ``recall_at_1_all``. Ranks are optimistic under ties
    (``1 + #{strictly better}``); ties are vanishingly rare at float64.
``frac_above_tau``
    Fraction of the (undirected) positive pairs scored at or above the method's
    own train-learned ``tau``, the Task-B existing-versus-new cut. This is the
    routing question rather than the ranking question. Two methods' ``tau``
    values are only comparable if they are the same operating point, so
    ``op_tpr`` / ``op_fpr`` / ``op_precision`` report where each ``tau`` sits on
    the whole test block.
``mcnemar_*`` / ``routing_*``
    Exact two-sided McNemar test of the lexical scorer against this method,
    paired observation by observation inside the stratum: ``mcnemar_*`` on the
    ranking outcome (partner at residual rank 1), ``routing_*`` on the routing
    outcome (pair at or above ``tau``). ``b`` counts observations only TF-IDF
    gets, ``c`` observations only this method gets. On the reverse slice the
    ``mcnemar_*`` columns compare confusions instead, so a large ``b`` is the
    lexical scorer's failure, not its win.

Reverse slice
-------------
``neg_top_decile`` takes the different-directory test pairs in the top decile of
TF-IDF overlap: near-duplicate wording across genuinely different fragments.
For each such pair and each endpoint that has a test partner, the method is
confused if it scores the high-overlap impostor above every true partner of that
query. Because the slice is defined by TF-IDF overlap alone, all methods are
scored on an identical observation set.

Outputs
-------
- ``runs/active/resubmit/results/lexical_vs_embedding.csv`` (tidy, one row per
  method x stratum)
- ``overleaf_drafts/figures/fig_lexical_vs_embedding.{pdf,png}``
- printed summary for ``docs/research/lexical_vs_embedding.md``

Run it in one command from the repo root, CPU only, a few minutes:

    python scripts/resubmit/lexical_vs_embedding.py

Re-run it with the same command if PR #130's TF-IDF implementation changes, or
whenever the split changes. Issue #131 already did that once: it relabels two
directories, which adds a positive test pair and renumbers part of the embedding
cache.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Display names and slugs for the six models the paper reports. Kept in sync
# with build_taskA_per_model_table.MODEL_DISPLAY.
MODEL_DISPLAY = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "google/mt5-base": "mT5-base",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
}

TERCILE_STRATA = ("overlap_low", "overlap_mid", "overlap_high")
HARD_STRATUM = "hard_below_tfidf_tau"
ALL_STRATUM = "all_positives"
NEG_STRATUM = "neg_top_decile"

LEXICAL_KEY = "tfidf_char35"


def _ensure_import_paths() -> None:
    """Put ``src/`` and ``scripts/resubmit/`` on ``sys.path``."""
    for path in (REPO_ROOT / "src", Path(__file__).resolve().parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


# --------------------------------------------------------------------------
# Pair enumeration and stratification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairSet:
    """Undirected index pairs ``(i, j)`` with ``i < j`` and their overlap."""

    i: np.ndarray
    j: np.ndarray
    overlap: np.ndarray

    def __len__(self) -> int:
        return int(self.i.size)


def enumerate_pairs(folder_ids: Sequence, overlap: np.ndarray, positive: bool) -> PairSet:
    """All ``i < j`` pairs that are (or are not) in the same directory."""
    folder_ids = np.asarray(folder_ids)
    n = folder_ids.size
    iu, ju = np.triu_indices(n, k=1)
    same = folder_ids[iu] == folder_ids[ju]
    keep = same if positive else ~same
    return PairSet(i=iu[keep], j=ju[keep], overlap=np.asarray(overlap)[iu[keep], ju[keep]])


def tercile_edges(values: np.ndarray) -> Tuple[float, float]:
    """The 33.3rd and 66.7th percentiles used to cut the overlap terciles."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        raise ValueError("Need at least three pairs to form terciles.")
    return (
        float(np.percentile(values, 100.0 / 3.0)),
        float(np.percentile(values, 200.0 / 3.0)),
    )


def assign_terciles(values: np.ndarray, edges: Tuple[float, float]) -> np.ndarray:
    """Label each value ``overlap_low`` / ``overlap_mid`` / ``overlap_high``.

    Boundaries are half-open upwards (``value < lo`` is low), so the lowest
    tercile can never swallow a value that equals the upper edge.
    """
    lo, hi = edges
    if hi < lo:
        raise ValueError(f"Tercile edges out of order: {edges}.")
    values = np.asarray(values, dtype=np.float64)
    labels = np.full(values.shape, TERCILE_STRATA[2], dtype=object)
    labels[values < hi] = TERCILE_STRATA[1]
    labels[values < lo] = TERCILE_STRATA[0]
    return labels


def stratum_masks(
    pairs: PairSet,
    edges: Tuple[float, float],
    hard_threshold: float,
) -> Dict[str, np.ndarray]:
    """Boolean masks over ``pairs`` for every positive-pair stratum.

    ``hard_below_tfidf_tau`` is not a fourth tercile: it is the set of positives
    the lexical scorer would itself route as new, and it overlaps the low
    tercile by construction.
    """
    labels = assign_terciles(pairs.overlap, edges)
    masks = {name: labels == name for name in TERCILE_STRATA}
    masks[HARD_STRATUM] = pairs.overlap < hard_threshold
    masks[ALL_STRATUM] = np.ones(len(pairs), dtype=bool)
    return masks


def stratum_bounds(
    pairs: PairSet,
    edges: Tuple[float, float],
    hard_threshold: float,
) -> Dict[str, Tuple[float, float]]:
    """Reported ``[lo, hi)`` overlap range of each stratum, for the CSV."""
    lowest = float(np.min(pairs.overlap))
    highest = float(np.max(pairs.overlap))
    return {
        TERCILE_STRATA[0]: (lowest, edges[0]),
        TERCILE_STRATA[1]: edges,
        TERCILE_STRATA[2]: (edges[1], highest),
        HARD_STRATUM: (lowest, hard_threshold),
        ALL_STRATUM: (lowest, highest),
    }


# --------------------------------------------------------------------------
# Pair-level ranking metrics
# --------------------------------------------------------------------------


def partner_ranks(
    sim: np.ndarray,
    query: int,
    partner: int,
    folder_ids: np.ndarray,
) -> Tuple[int, int]:
    """Rank of ``partner`` in ``sim[query]``, residual and literal.

    Returns ``(rank_residual, rank_all)``. Both are ``1 + #{strictly better}``.
    The residual rank drops the query's other same-directory files from the
    candidate pool; the literal rank keeps every test file but the query itself.
    """
    scores = sim[query]
    target = scores[partner]
    better = scores > target
    better[query] = False  # a file is never its own distractor
    rank_all = 1 + int(better.sum())

    other_partners = (folder_ids == folder_ids[query]).copy()
    other_partners[query] = False
    other_partners[partner] = False
    rank_residual = 1 + int((better & ~other_partners).sum())
    return rank_residual, rank_all


def directed_rank_table(
    sim: np.ndarray,
    pairs: PairSet,
    folder_ids: np.ndarray,
) -> pd.DataFrame:
    """One row per directed observation: ``pair_index``, both ranks.

    Each undirected pair yields two rows, using each endpoint as the query.
    """
    folder_ids = np.asarray(folder_ids)
    rows = []
    for p in range(len(pairs)):
        a, b = int(pairs.i[p]), int(pairs.j[p])
        for query, partner in ((a, b), (b, a)):
            residual, literal = partner_ranks(sim, query, partner, folder_ids)
            rows.append(
                {
                    "pair_index": p,
                    "query": query,
                    "partner": partner,
                    "rank_residual": residual,
                    "rank_all": literal,
                }
            )
    return pd.DataFrame(rows, columns=["pair_index", "query", "partner", "rank_residual", "rank_all"])


def rank_metrics(ranks: pd.DataFrame, mask: np.ndarray) -> Dict[str, float]:
    """recall@1 and MRR over the directed observations of the masked pairs."""
    keep = np.asarray(mask)[ranks["pair_index"].to_numpy()]
    sub = ranks.loc[keep]
    if len(sub) == 0:
        return {"n_directed": 0, "recall_at_1": float("nan"),
                "recall_at_1_all": float("nan"), "mrr": float("nan")}
    return {
        "n_directed": int(len(sub)),
        "recall_at_1": float((sub["rank_residual"].to_numpy() == 1).mean()),
        "recall_at_1_all": float((sub["rank_all"].to_numpy() == 1).mean()),
        "mrr": float((1.0 / sub["rank_residual"].to_numpy()).mean()),
    }


def frac_above_tau(sim: np.ndarray, pairs: PairSet, mask: np.ndarray, tau: float) -> float:
    """Fraction of the masked positive pairs scoring at or above ``tau``."""
    mask = np.asarray(mask)
    if not mask.any():
        return float("nan")
    scores = sim[pairs.i[mask], pairs.j[mask]]
    return float((scores >= tau).mean())


def best_partner_similarity(sim: np.ndarray, folder_ids: np.ndarray) -> np.ndarray:
    """Per file, the highest similarity to any other file in its directory.

    ``-inf`` for a file with no test partner, which excludes it from the reverse
    slice: there is nothing for an impostor to outrank.
    """
    folder_ids = np.asarray(folder_ids)
    same_dir = folder_ids[:, None] == folder_ids[None, :]
    np.fill_diagonal(same_dir, False)
    masked = np.where(same_dir, sim, -np.inf)
    return masked.max(axis=1)


def negative_confusion_flags(
    sim: np.ndarray,
    neg_pairs: PairSet,
    mask: np.ndarray,
    folder_ids: np.ndarray,
) -> np.ndarray:
    """Per directed observation, whether the impostor outranks every true partner.

    One observation per endpoint of a masked negative pair, kept only when that
    endpoint actually has a test partner to be confused with. Which endpoints
    are usable depends on ``folder_ids`` alone, so the returned vector is
    aligned across methods and can be paired.
    """
    mask = np.asarray(mask)
    best = best_partner_similarity(sim, folder_ids)
    i, j = neg_pairs.i[mask], neg_pairs.j[mask]

    flags: List[np.ndarray] = []
    for query, impostor in ((i, j), (j, i)):
        usable = np.isfinite(best[query])
        flags.append(sim[query[usable], impostor[usable]] > best[query[usable]])
    return np.concatenate(flags) if flags else np.zeros(0, dtype=bool)


def negative_confusion_rate(
    sim: np.ndarray,
    neg_pairs: PairSet,
    mask: np.ndarray,
    folder_ids: np.ndarray,
) -> Tuple[float, int]:
    """Rate and observation count of :func:`negative_confusion_flags`."""
    flags = negative_confusion_flags(sim, neg_pairs, mask, folder_ids)
    if flags.size == 0:
        return float("nan"), 0
    return float(flags.mean()), int(flags.size)


# --------------------------------------------------------------------------
# Paired significance and matched operating points
# --------------------------------------------------------------------------


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts ``b`` and ``c``.

    The binomial test of ``b ~ Binom(b + c, 1/2)``, doubled and capped at 1.
    Exact rather than chi-square because several strata here have well under
    the 25 discordant observations the asymptotic form wants. With no
    discordant observations at all the two methods agree everywhere and the
    p-value is 1.0: no evidence of a difference.
    """
    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative, got ({b}, {c}).")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 1 << n)))


def paired_discordance(reference: np.ndarray, other: np.ndarray) -> Tuple[int, int]:
    """``(reference-only successes, other-only successes)`` over aligned outcomes."""
    reference = np.asarray(reference, dtype=bool)
    other = np.asarray(other, dtype=bool)
    if reference.shape != other.shape:
        raise ValueError(
            f"Outcome vectors are not aligned: {reference.shape} vs {other.shape}."
        )
    return int((reference & ~other).sum()), int((~reference & other).sum())


def operating_point(
    sim: np.ndarray, pos_pairs: PairSet, neg_pairs: PairSet, tau: float
) -> Dict[str, float]:
    """Where a method's own train-learned ``tau`` sits on the whole test block.

    Two methods can only be compared on ``frac_above_tau`` inside a stratum if
    their thresholds are the same operating point overall; these are the
    numbers that establish that, or fail to.
    """
    pos = sim[pos_pairs.i, pos_pairs.j]
    neg = sim[neg_pairs.i, neg_pairs.j]
    n_tp = int((pos >= tau).sum())
    n_fp = int((neg >= tau).sum())
    return {
        "op_tpr": float(n_tp / len(pos)) if len(pos) else float("nan"),
        "op_fpr": float(n_fp / len(neg)) if len(neg) else float("nan"),
        "op_precision": float(n_tp / (n_tp + n_fp)) if (n_tp + n_fp) else float("nan"),
        "op_n_tp": n_tp,
        "op_n_fp": n_fp,
    }


# --------------------------------------------------------------------------
# Method construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Method:
    """One scorer evaluated on the test block."""

    key: str
    family: str  # "lexical" | "baseline" | "abtt"
    model_display: str
    layer: Optional[int]
    D: Optional[int]
    tau: float
    test_sim: np.ndarray


def paper_config(results: pd.DataFrame, model: str, method: str) -> pd.Series:
    """The paper's chosen row for one model and method: highest train AUROC.

    The headline tables read "the layer chosen by training-set AUROC", so the
    selection is reproduced from the same CSV rather than re-tuned here.
    """
    sub = results[
        (results["model"] == model)
        & (results["method"] == method)
        & results["train_aucroc"].notna()
    ]
    if sub.empty:
        raise SystemExit(f"No {method!r} rows for {model!r} in the results CSV.")
    return sub.loc[sub["train_aucroc"].idxmax()]


# Cached embedding matrices are written as ``{repr}_layer{N}_embeddings{suffix}``
# inside ``{repr}_{pooling}_tokempty`` (see CLAUDE.md, "Embedding file naming").
POOLING_SUFFIX = {"mean": "", "sif": "_sif", "lasttok": "_lasttok"}


def embedding_path(bases_root: Path, model: str, repr_name: str, pooling: str, layer: int) -> Path:
    """Cache path for one (model, repr, pooling, layer), from the results-CSV columns.

    The results CSV carries ``repr`` and ``pooling`` per row, so a future CSV
    whose train-AUROC winner is an ``ff1`` or ``lasttok`` row resolves to those
    vectors instead of silently scoring mean-pooled hidden states.
    """
    if pooling not in POOLING_SUFFIX:
        raise SystemExit(
            f"Unknown pooling {pooling!r} for {model!r}; expected one of "
            f"{sorted(POOLING_SUFFIX)}."
        )
    slug = model.replace("/", "_")
    subdir = f"{repr_name}_{pooling}_tokempty"
    fname = f"{repr_name}_layer{layer}_embeddings{POOLING_SUFFIX[pooling]}.npy"
    return bases_root / "phase9_bases" / slug / subdir / fname


def embedding_row_index(row_order_csv: Path, filenames: Sequence[str]) -> np.ndarray:
    """Positions in the cached matrices that line up with the split CSV's rows.

    The caches are keyed by row index, and issue #131's key corrections
    renumbered 17 ``file_id``s without changing a single byte of text, so a
    positional read would hand seventeen files each other's vectors. Filenames
    are unique across the corpus (that is what the corrected split's
    carry-over is matched on), so aligning on them is immune to any future
    renumbering as well.
    """
    if not row_order_csv.exists():
        raise SystemExit(
            f"Missing embedding row order: {row_order_csv}. It records which file "
            "each cached row belongs to and is what keeps the vectors aligned "
            "with the split CSV; pass --row_order_csv if it lives elsewhere."
        )
    order = pd.read_csv(row_order_csv)
    if "path" not in order.columns:
        raise SystemExit(f"{row_order_csv} has no 'path' column.")
    cached = [Path(str(rel)).name for rel in order["path"].values]
    if len(set(cached)) != len(cached):
        raise SystemExit(f"{row_order_csv} repeats a filename; cannot align by name.")
    position = {name: k for k, name in enumerate(cached)}
    missing = [f for f in filenames if f not in position]
    if missing:
        raise SystemExit(
            f"{len(missing)} split files are absent from {row_order_csv} "
            f"(first: {missing[0]}). The caches predate the current corpus; "
            "re-extract before re-running."
        )
    return np.array([position[f] for f in filenames], dtype=int)


def load_layer_embeddings(
    bases_root: Path,
    model: str,
    layer: int,
    repr_name: str,
    pooling: str,
    row_index: np.ndarray,
) -> np.ndarray:
    """Load one cached layer and reorder its rows into split-CSV order."""
    path = embedding_path(bases_root, model, repr_name, pooling, layer)
    if not path.exists():
        raise SystemExit(f"Missing embeddings: {path}")
    emb = np.load(path)
    if emb.shape[0] <= int(row_index.max()):
        raise SystemExit(
            f"{path} has {emb.shape[0]} rows; the row order needs at least "
            f"{int(row_index.max()) + 1}."
        )
    return emb[row_index].astype(np.float64)


def build_embedding_method(
    key: str,
    family: str,
    model_display: str,
    emb_all: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    train_folder_ids: np.ndarray,
    layer: int,
    D: Optional[int],
) -> Method:
    """Post-process, L2-normalise, and learn ``tau`` on train, as the paper does.

    The cleaner is fitted on the train rows only and then applied to both
    blocks, which is the leak-free protocol CLAUDE.md requires.
    """
    from canon_retrieval import l2_normalize, similarity_matrix
    from run_resubmit_evaluate import learn_tau_from_similarity
    from sif_abtt import EmbeddingCleaner

    train_emb = emb_all[train_mask]
    test_emb = emb_all[test_mask]
    if D is not None:
        cleaner = EmbeddingCleaner(num_components=int(D), center=True)
        cleaner.fit(train_emb)
        train_emb = cleaner.transform(train_emb)
        test_emb = cleaner.transform(test_emb)

    train_sim = similarity_matrix(l2_normalize(train_emb))
    test_sim = similarity_matrix(l2_normalize(test_emb))
    tau = learn_tau_from_similarity(train_sim, train_folder_ids)
    return Method(key, family, model_display, int(layer), None if D is None else int(D), tau, test_sim)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def load_texts(split_meta: pd.DataFrame, data_root: Path) -> List[str]:
    texts = []
    for rel in split_meta["path"].values:
        path = Path(rel)
        if not path.is_absolute():
            path = data_root / path
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return texts


@dataclass(frozen=True)
class Outcomes:
    """One method's per-observation successes, in an order shared by all methods.

    ``rank_ok`` follows :func:`directed_rank_table`'s row order, ``routing_ok``
    follows the positive ``PairSet``, and ``neg_confused`` follows
    :func:`negative_confusion_flags`. None of those orders depends on the
    method, which is what makes the McNemar pairing legitimate.
    """

    ranks: pd.DataFrame
    rank_ok: np.ndarray
    routing_ok: np.ndarray
    neg_confused: np.ndarray


def method_outcomes(
    method: Method,
    pos_pairs: PairSet,
    neg_pairs: PairSet,
    neg_mask: np.ndarray,
    test_folder_ids: np.ndarray,
) -> Outcomes:
    ranks = directed_rank_table(method.test_sim, pos_pairs, test_folder_ids)
    return Outcomes(
        ranks=ranks,
        rank_ok=ranks["rank_residual"].to_numpy() == 1,
        routing_ok=method.test_sim[pos_pairs.i, pos_pairs.j] >= method.tau,
        neg_confused=negative_confusion_flags(
            method.test_sim, neg_pairs, neg_mask, test_folder_ids
        ),
    )


def _paired_columns(
    reference: Optional[np.ndarray], other: np.ndarray, prefix: str
) -> Dict[str, object]:
    """McNemar counts and p-value for one comparison, or NA when self-compared."""
    if reference is None or other.size == 0:
        return {f"{prefix}_b": pd.NA, f"{prefix}_c": pd.NA, f"{prefix}_p": pd.NA}
    b, c = paired_discordance(reference, other)
    return {f"{prefix}_b": b, f"{prefix}_c": c, f"{prefix}_p": mcnemar_exact_p(b, c)}


def collect_rows(
    method: Method,
    pos_pairs: PairSet,
    pos_masks: Dict[str, np.ndarray],
    neg_pairs: PairSet,
    neg_mask: np.ndarray,
    test_folder_ids: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    neg_cut: float,
    reference: Optional[Outcomes] = None,
) -> List[Dict]:
    """Every output row for one method: the positive strata, then the reverse slice.

    ``reference`` is the lexical method's outcomes. When it is given, each row
    also carries the exact McNemar comparison of that method against this one,
    on the ranking outcome (``mcnemar_*``) and on the routing outcome
    (``routing_*``), paired observation by observation inside the stratum. The
    reference method itself is compared against nothing and gets NA.
    """
    own = method_outcomes(method, pos_pairs, neg_pairs, neg_mask, test_folder_ids)
    base = {
        "method": method.key,
        "family": method.family,
        "model": method.model_display,
        "layer": method.layer if method.layer is not None else pd.NA,
        "D": method.D if method.D is not None else pd.NA,
        "tau": method.tau,
    }
    base.update(operating_point(method.test_sim, pos_pairs, neg_pairs, method.tau))

    rows: List[Dict] = []
    for stratum in (*TERCILE_STRATA, HARD_STRATUM, ALL_STRATUM):
        mask = np.asarray(pos_masks[stratum])
        directed = mask[own.ranks["pair_index"].to_numpy()]
        row = dict(base)
        row.update(
            {
                "stratum": stratum,
                "overlap_lo": bounds[stratum][0],
                "overlap_hi": bounds[stratum][1],
                "n_pairs": int(mask.sum()),
                "frac_above_tau": frac_above_tau(method.test_sim, pos_pairs, mask, method.tau),
                "neg_confusion_rate": pd.NA,
            }
        )
        row.update(rank_metrics(own.ranks, mask))
        row.update(
            _paired_columns(
                None if reference is None else reference.rank_ok[directed],
                own.rank_ok[directed],
                "mcnemar",
            )
        )
        row.update(
            _paired_columns(
                None if reference is None else reference.routing_ok[mask],
                own.routing_ok[mask],
                "routing",
            )
        )
        rows.append(row)

    neg_row = dict(base)
    neg_row.update(
        {
            "stratum": NEG_STRATUM,
            "overlap_lo": neg_cut,
            "overlap_hi": float(neg_pairs.overlap.max()),
            "n_pairs": int(np.asarray(neg_mask).sum()),
            "n_directed": int(own.neg_confused.size),
            "recall_at_1": pd.NA,
            "recall_at_1_all": pd.NA,
            "mrr": pd.NA,
            "frac_above_tau": pd.NA,
            "neg_confusion_rate": (
                float(own.neg_confused.mean()) if own.neg_confused.size else float("nan")
            ),
            "routing_b": pd.NA,
            "routing_c": pd.NA,
            "routing_p": pd.NA,
        }
    )
    # On the reverse slice the McNemar columns compare *confusions*, so b counts
    # the impostors only the reference falls for.
    neg_row.update(
        _paired_columns(
            None if reference is None else reference.neg_confused,
            own.neg_confused,
            "mcnemar",
        )
    )
    rows.append(neg_row)
    return rows


COLUMN_ORDER = [
    "method", "family", "model", "layer", "D", "tau", "stratum",
    "overlap_lo", "overlap_hi", "n_pairs", "n_directed",
    "recall_at_1", "recall_at_1_all", "mrr", "frac_above_tau",
    "neg_confusion_rate",
    "mcnemar_b", "mcnemar_c", "mcnemar_p",
    "routing_b", "routing_c", "routing_p",
    "op_tpr", "op_fpr", "op_precision", "op_n_tp", "op_n_fp",
]


def render_figure(results: pd.DataFrame, out_pdf: Path) -> None:
    """recall@1 against overlap tercile: baselines left, ABTT right, TF-IDF on both."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(TERCILE_STRATA))
    # Bounds are identical across methods, so any one row per stratum carries them.
    bounds = results.drop_duplicates("stratum").set_index("stratum")
    n_pairs = [int(bounds.loc[st, "n_pairs"]) for st in TERCILE_STRATA]
    tick_labels = [
        f"low\n(<{bounds.loc[TERCILE_STRATA[0], 'overlap_hi']:.2f})\nn={n_pairs[0]}",
        f"mid\n({bounds.loc[TERCILE_STRATA[1], 'overlap_lo']:.2f}"
        f"-{bounds.loc[TERCILE_STRATA[1], 'overlap_hi']:.2f})\nn={n_pairs[1]}",
        f"high\n(>{bounds.loc[TERCILE_STRATA[2], 'overlap_lo']:.2f})\nn={n_pairs[2]}",
    ]
    colors = plt.get_cmap("tab10").colors

    def series(df: pd.DataFrame) -> np.ndarray:
        by_stratum = df.set_index("stratum")["recall_at_1"]
        return np.array([by_stratum.get(s, np.nan) for s in TERCILE_STRATA], dtype=float)

    def stderr(values: np.ndarray) -> np.ndarray:
        """Binomial SE on the *pair* count, not the directed count.

        Each pair contributes two directed observations that share a text, so
        the directed count would understate the error. This is the conservative
        reading, and it is what makes the tercile orderings here readable as
        the noise they mostly are.
        """
        n = np.asarray(n_pairs, dtype=float)
        return np.sqrt(np.clip(values * (1.0 - values), 0.0, None) / n)

    lexical = series(results[results["method"] == LEXICAL_KEY])
    models = [m for m in MODEL_DISPLAY.values() if m in set(results["model"])]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, family, title in zip(
        axes, ("baseline", "abtt"),
        ("Mean-pool baseline embeddings", "ABTT embeddings (paper's layer, $D$)"),
    ):
        ax.errorbar(x, lexical, yerr=stderr(lexical), color="black", marker="s",
                    linewidth=2.4, capsize=3, zorder=5,
                    label="char 3-5-gram TF-IDF")
        for c, model in enumerate(models):
            sub = results[(results["family"] == family) & (results["model"] == model)]
            if sub.empty:
                continue
            values = series(sub)
            ax.errorbar(x, values, yerr=stderr(values), marker="o", markersize=4,
                        linewidth=1.3, capsize=2, elinewidth=0.8,
                        color=colors[c % len(colors)], label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("char 3-5-gram TF-IDF cosine of the positive pair")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("partner recall@1")
    axes[1].legend(fontsize=7.5, loc="lower right", ncol=2, framealpha=0.9)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--split_csv",
        default=str(REPO_ROOT / "runs/active/resubmit/data/phase_resubmit_split.csv"),
    )
    p.add_argument("--data_root", default=str(REPO_ROOT),
                   help="Root the 'path' column of the split CSV is relative to.")
    p.add_argument(
        "--results_csv",
        default=str(REPO_ROOT / "runs/active/resubmit/results/phase_resubmit_results.csv"),
        help="Source of the paper's per-model best layer and D.",
    )
    p.add_argument("--bases_root", default=str(REPO_ROOT / "runs/active/resubmit_bases"),
                   help="Root holding phase9_bases/<slug>/<subdir>/.")
    p.add_argument(
        "--row_order_csv",
        default=str(
            REPO_ROOT / "runs/active/resubmit_bases/phase9_bases/row_order.csv"
        ),
        help="file_id -> path listing of the cached embedding rows, used to "
             "align the caches to the split CSV by filename.",
    )
    p.add_argument(
        "--allow-tau-drift", "--allow_tau_drift", dest="allow_tau_drift",
        action="store_true",
        help="Continue when a re-learned tau disagrees with the results CSV. "
             "Off by default: agreement is the check that these really are the "
             "paper's configurations, so a drift means the results CSV and the "
             "split have come apart and the numbers must not be published.",
    )
    p.add_argument("--tau_tolerance", type=float, default=5e-3,
                   help="How far a re-learned tau may sit from the results CSV.")
    p.add_argument(
        "--out_csv",
        default=str(REPO_ROOT / "runs/active/resubmit/results/lexical_vs_embedding.csv"),
    )
    p.add_argument(
        "--out_fig",
        default=str(REPO_ROOT / "overleaf_drafts/figures/fig_lexical_vs_embedding.pdf"),
    )
    p.add_argument("--neg_decile", type=float, default=0.9,
                   help="Quantile cut for the high-overlap negative slice.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _ensure_import_paths()
    from lexical_baselines import apply_minmax, char_tfidf_score_matrix, fit_minmax
    from run_resubmit_evaluate import learn_tau_from_similarity

    split_meta = pd.read_csv(args.split_csv).reset_index(drop=True)
    train_mask = split_meta["split"].values == "train"
    test_mask = split_meta["split"].values == "test"
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    train_folder_ids = split_meta["folder_id"].values[train_idx]
    test_folder_ids = split_meta["folder_id"].values[test_idx]
    print(f"{len(split_meta)} files: {train_idx.size} train / {test_idx.size} test")

    # ── the stratifying variable and the lexical method ───────────────────
    texts = load_texts(split_meta, Path(args.data_root))
    raw_tfidf = char_tfidf_score_matrix(texts, train_idx)
    lo, hi = fit_minmax(raw_tfidf, train_idx)
    scaled_tfidf = apply_minmax(raw_tfidf, lo, hi)
    lex_tau = learn_tau_from_similarity(
        scaled_tfidf[np.ix_(train_idx, train_idx)], train_folder_ids
    )
    # tau lives on the rescaled scale; map it back so it can cut raw cosines.
    lex_tau_raw = lo + lex_tau * (hi - lo)
    print(f"char TF-IDF tau={lex_tau:.4f} (raw cosine {lex_tau_raw:.4f})")

    overlap_test = raw_tfidf[np.ix_(test_idx, test_idx)]
    pos_pairs = enumerate_pairs(test_folder_ids, overlap_test, positive=True)
    neg_pairs = enumerate_pairs(test_folder_ids, overlap_test, positive=False)
    edges = tercile_edges(pos_pairs.overlap)
    pos_masks = stratum_masks(pos_pairs, edges, lex_tau_raw)
    pos_bounds = stratum_bounds(pos_pairs, edges, lex_tau_raw)
    neg_cut = float(np.quantile(neg_pairs.overlap, args.neg_decile))
    neg_mask = neg_pairs.overlap >= neg_cut
    print(
        f"{len(pos_pairs)} positive test pairs, tercile edges "
        f"{edges[0]:.4f}/{edges[1]:.4f}, hard slice n={int(pos_masks[HARD_STRATUM].sum())}; "
        f"{len(neg_pairs)} negatives, top-decile cut {neg_cut:.4f} "
        f"(n={int(neg_mask.sum())})"
    )

    methods = [
        Method(LEXICAL_KEY, "lexical", "char TF-IDF 3-5", None, None, lex_tau,
               scaled_tfidf[np.ix_(test_idx, test_idx)])
    ]

    # ── the paper's embedding configurations ──────────────────────────────
    results = pd.read_csv(args.results_csv)
    bases_root = Path(args.bases_root)
    row_index = embedding_row_index(
        Path(args.row_order_csv), list(split_meta["filename"].astype(str))
    )
    n_moved = int((row_index != np.arange(row_index.size)).sum())
    print(
        f"Embedding rows aligned to the split by filename via {args.row_order_csv} "
        f"({n_moved} of {row_index.size} rows sit at a different cached index)"
    )

    drifted: List[str] = []
    for model, display in MODEL_DISPLAY.items():
        base_cfg = paper_config(results, model, "baseline")
        abtt_cfg = paper_config(results, model, "abtt_optimal")
        for family, cfg, D in (
            ("baseline", base_cfg, None),
            ("abtt", abtt_cfg, int(abtt_cfg["D"])),
        ):
            layer = int(cfg["layer"])
            emb = load_layer_embeddings(
                bases_root, model, layer, str(cfg["repr"]), str(cfg["pooling"]), row_index
            )
            if emb.shape[0] != len(split_meta):
                raise SystemExit(
                    f"{display} layer {layer}: {emb.shape[0]} rows for "
                    f"{len(split_meta)} split rows."
                )
            method = build_embedding_method(
                f"{family}_{display}", family, display, emb,
                train_mask, test_mask, train_folder_ids, layer, D,
            )
            drift = abs(method.tau - float(cfg["tau"]))
            flag = "" if drift < args.tau_tolerance else "  <-- tau drift vs results CSV"
            if flag:
                drifted.append(
                    f"{display} {family}: tau={method.tau:.4f} vs csv "
                    f"{float(cfg['tau']):.4f} (drift {drift:.4f})"
                )
            print(
                f"{display:>11s} {family:<8s} {cfg['repr']}/{cfg['pooling']} "
                f"layer={layer:<3d} D={D} "
                f"tau={method.tau:.4f} (csv {float(cfg['tau']):.4f}){flag}"
            )
            methods.append(method)

    if drifted and not args.allow_tau_drift:
        raise SystemExit(
            "Re-learned tau disagrees with the results CSV on "
            f"{len(drifted)} of {len(methods) - 1} configurations:\n  "
            + "\n  ".join(drifted)
            + "\nThat agreement is what certifies these as the paper's "
            "configurations, so the run stops rather than writing numbers that "
            "look like the paper's and are not. Regenerate "
            f"{args.results_csv} against {args.split_csv}, or pass "
            "--allow-tau-drift if you know why they differ."
        )

    reference = method_outcomes(methods[0], pos_pairs, neg_pairs, neg_mask, test_folder_ids)
    rows: List[Dict] = []
    for k, method in enumerate(methods):
        rows.extend(
            collect_rows(method, pos_pairs, pos_masks, neg_pairs, neg_mask,
                         test_folder_ids, pos_bounds, neg_cut,
                         reference=None if k == 0 else reference)
        )

    out = pd.DataFrame(rows)[COLUMN_ORDER]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out)} rows to {out_csv}")

    render_figure(out, Path(args.out_fig))
    print(f"Saved figure to {args.out_fig}")

    pivot = out[out["stratum"].isin(TERCILE_STRATA)].pivot_table(
        index=["family", "model"], columns="stratum", values="recall_at_1"
    )
    print("\nrecall@1 by overlap tercile\n")
    print(pivot[list(TERCILE_STRATA)].round(3).to_string())
    hard = out[out["stratum"] == HARD_STRATUM].set_index(["family", "model"])
    print("\nhard slice (positives below the TF-IDF tau)\n")
    print(hard[["n_pairs", "recall_at_1", "mrr", "frac_above_tau"]].round(3).to_string())
    neg = out[out["stratum"] == NEG_STRATUM].set_index(["family", "model"])
    print("\nhigh-overlap negative confusion rate (mcnemar_* vs TF-IDF)\n")
    print(neg[["n_pairs", "n_directed", "neg_confusion_rate",
               "mcnemar_b", "mcnemar_c", "mcnemar_p"]].round(4).to_string())

    print("\nmatched operating points (each method at its own train-learned tau)\n")
    ops = out[out["stratum"] == ALL_STRATUM].set_index("method")
    print(ops[["tau", "op_tpr", "op_fpr", "op_precision", "op_n_tp",
               "op_n_fp"]].round(5).to_string())

    print("\npaired McNemar vs char TF-IDF (recall@1 | frac_above_tau)\n")
    stats = out[out["stratum"].isin((*TERCILE_STRATA, HARD_STRATUM, ALL_STRATUM))]
    stats = stats[stats["method"] != LEXICAL_KEY]
    print(
        stats.set_index(["method", "stratum"])[
            ["recall_at_1", "mcnemar_b", "mcnemar_c", "mcnemar_p",
             "frac_above_tau", "routing_b", "routing_c", "routing_p"]
        ].round(4).to_string()
    )


if __name__ == "__main__":
    main()
