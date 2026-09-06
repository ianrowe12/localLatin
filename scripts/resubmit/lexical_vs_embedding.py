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

Layer, ``D`` and the method family all come from ``phase_resubmit_results.csv``
rather than being re-tuned here, so the configurations are the paper's.

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
    routing question rather than the ranking question.

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
after issue #112, which relabels two directories and so changes the positive
pair set.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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


def negative_confusion_rate(
    sim: np.ndarray,
    neg_pairs: PairSet,
    mask: np.ndarray,
    folder_ids: np.ndarray,
) -> Tuple[float, int]:
    """How often a high-overlap impostor outranks every true partner.

    One directed observation per endpoint of a masked negative pair, kept only
    when that endpoint actually has a test partner to be confused with.
    """
    mask = np.asarray(mask)
    best = best_partner_similarity(sim, folder_ids)
    i, j = neg_pairs.i[mask], neg_pairs.j[mask]

    hits = 0
    total = 0
    for query, impostor in ((i, j), (j, i)):
        usable = np.isfinite(best[query])
        total += int(usable.sum())
        hits += int((sim[query[usable], impostor[usable]] > best[query[usable]]).sum())
    if total == 0:
        return float("nan"), 0
    return hits / total, total


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


def load_layer_embeddings(
    bases_root: Path, model: str, layer: int, subdir: str
) -> np.ndarray:
    slug = model.replace("/", "_")
    path = bases_root / "phase9_bases" / slug / subdir / f"hidden_layer{layer}_embeddings.npy"
    if not path.exists():
        raise SystemExit(f"Missing embeddings: {path}")
    return np.load(path).astype(np.float64)


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
    """Post-process, L2-normalise, and learn ``tau`` on train, as the paper does."""
    _ensure_import_paths()
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


def collect_rows(
    method: Method,
    pos_pairs: PairSet,
    pos_masks: Dict[str, np.ndarray],
    neg_pairs: PairSet,
    neg_mask: np.ndarray,
    test_folder_ids: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    neg_cut: float,
) -> List[Dict]:
    """Every output row for one method: the positive strata, then the reverse slice."""
    ranks = directed_rank_table(method.test_sim, pos_pairs, test_folder_ids)
    base = {
        "method": method.key,
        "family": method.family,
        "model": method.model_display,
        "layer": method.layer if method.layer is not None else pd.NA,
        "D": method.D if method.D is not None else pd.NA,
        "tau": method.tau,
    }

    rows: List[Dict] = []
    for stratum in (*TERCILE_STRATA, HARD_STRATUM, ALL_STRATUM):
        mask = pos_masks[stratum]
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
        row.update(rank_metrics(ranks, mask))
        rows.append(row)

    rate, n_directed = negative_confusion_rate(
        method.test_sim, neg_pairs, neg_mask, test_folder_ids
    )
    neg_row = dict(base)
    neg_row.update(
        {
            "stratum": NEG_STRATUM,
            "overlap_lo": neg_cut,
            "overlap_hi": float(neg_pairs.overlap.max()),
            "n_pairs": int(neg_mask.sum()),
            "n_directed": n_directed,
            "recall_at_1": pd.NA,
            "recall_at_1_all": pd.NA,
            "mrr": pd.NA,
            "frac_above_tau": pd.NA,
            "neg_confusion_rate": rate,
        }
    )
    rows.append(neg_row)
    return rows


COLUMN_ORDER = [
    "method", "family", "model", "layer", "D", "tau", "stratum",
    "overlap_lo", "overlap_hi", "n_pairs", "n_directed",
    "recall_at_1", "recall_at_1_all", "mrr", "frac_above_tau",
    "neg_confusion_rate",
]


def render_figure(results: pd.DataFrame, out_pdf: Path) -> None:
    """recall@1 against overlap tercile: baselines left, ABTT right, TF-IDF on both."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(TERCILE_STRATA))
    # Bounds are identical across methods, so any one row per stratum carries them.
    bounds = results.drop_duplicates("stratum").set_index("stratum")
    tick_labels = [
        f"low\n(<{bounds.loc[TERCILE_STRATA[0], 'overlap_hi']:.2f})",
        f"mid\n({bounds.loc[TERCILE_STRATA[1], 'overlap_lo']:.2f}"
        f"-{bounds.loc[TERCILE_STRATA[1], 'overlap_hi']:.2f})",
        f"high\n(>{bounds.loc[TERCILE_STRATA[2], 'overlap_lo']:.2f})",
    ]
    colors = plt.get_cmap("tab10").colors

    def series(df: pd.DataFrame) -> np.ndarray:
        by_stratum = df.set_index("stratum")["recall_at_1"]
        return np.array([by_stratum.get(s, np.nan) for s in TERCILE_STRATA], dtype=float)

    lexical = series(results[results["method"] == LEXICAL_KEY])
    models = [m for m in MODEL_DISPLAY.values() if m in set(results["model"])]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, family, title in zip(
        axes, ("baseline", "abtt"),
        ("Mean-pool baseline embeddings", "ABTT embeddings (paper's layer, $D$)"),
    ):
        ax.plot(x, lexical, color="black", marker="s", linewidth=2.4,
                zorder=5, label="char 3-5-gram TF-IDF")
        for c, model in enumerate(models):
            sub = results[(results["family"] == family) & (results["model"] == model)]
            if sub.empty:
                continue
            ax.plot(x, series(sub), marker="o", markersize=4, linewidth=1.3,
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
    p.add_argument("--mean_subdir", default="hidden_mean_tokempty")
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
    for model, display in MODEL_DISPLAY.items():
        base_cfg = paper_config(results, model, "baseline")
        abtt_cfg = paper_config(results, model, "abtt_optimal")
        for family, cfg, D in (
            ("baseline", base_cfg, None),
            ("abtt", abtt_cfg, int(abtt_cfg["D"])),
        ):
            layer = int(cfg["layer"])
            emb = load_layer_embeddings(bases_root, model, layer, args.mean_subdir)
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
            flag = "" if drift < 5e-3 else "  <-- tau drift vs results CSV"
            print(
                f"{display:>11s} {family:<8s} layer={layer:<3d} D={D} "
                f"tau={method.tau:.4f} (csv {float(cfg['tau']):.4f}){flag}"
            )
            methods.append(method)

    rows: List[Dict] = []
    for method in methods:
        rows.extend(
            collect_rows(method, pos_pairs, pos_masks, neg_pairs, neg_mask,
                         test_folder_ids, pos_bounds, neg_cut)
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
    print("\nhigh-overlap negative confusion rate\n")
    print(neg[["n_pairs", "n_directed", "neg_confusion_rate"]].round(3).to_string())


if __name__ == "__main__":
    main()
