"""Lexical retrieval baselines for Task A and Task B (issue #122).

Three surface-form scorers stand in for the embedding pipeline, so the paper can
say how much of the retrieval signal is plain word or character overlap:

``bm25_word``
    Okapi BM25 over whitespace-tokenised, lowercased, punctuation-stripped text,
    with ``k1=1.5`` and ``b=0.75``. Document frequencies, the vocabulary and the
    average document length all come from the *train* half of the split only, so
    the scorer obeys the same leak-free protocol as SIF token probabilities.
    BM25 is asymmetric (it scores a query against a document), so the pairwise
    matrix is symmetrised as ``(score(a,b) + score(b,a)) / 2``.
``tfidf_char35``
    Cosine over character 3--5-gram TF-IDF vectors. The vectoriser is fitted on
    train texts only; test texts are transformed with the train vocabulary and
    train IDF. Symmetric by construction.
``levenshtein``
    Normalised Levenshtein similarity (``rapidfuzz``) over the same normalised
    strings. Symmetric by construction.

Every scorer produces a full ``n x n`` score matrix over the 1,705 labelled
files, which is then min-max rescaled with the *train* upper triangle's min and
max so the learned threshold sweep over ``[0, 1]`` is well posed. The rescaling
is a monotone affine map, so it changes only ``tau``, ``same_avg``, ``diff_avg``
and ``gap``; AUROC and every accuracy are invariant to it. Test scores are not
clipped, so a test pair may land slightly outside ``[0, 1]``.

Metrics are not reimplemented here. Task A (AUROC, score gap) and single-split
Task B (assignment accuracy with a train-learned ``tau``, directory accuracy at
1 and 3) come from ``run_resubmit_evaluate.evaluate_from_similarity``; the
5-seed Task B protocol reuses ``run_taskb_mseed.evaluate_query_vs_directory``
and ``canon_split_v2.canon_taskb_query_reference_split``.

Run it in one command from the repo root:

    python scripts/resubmit/lexical_baselines.py

Re-run it after issue #112 lands, since that changes two directory labels and
therefore the positive-pair sets these numbers are computed over.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Only numpy/pandas are imported at module scope. scikit-learn (via the
# evaluator), scipy and rapidfuzz are imported inside the functions that need
# them, so the scoring helpers below stay importable in a bare test environment.


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

# Unicode general categories treated as separators: all punctuation (P*) and all
# symbols (S*). Digits and letters survive. Marks (M*) survive too, so a
# combining accent stays attached to its base letter.
_SEPARATOR_CATEGORIES = ("P", "S")


def normalise_text(text: str) -> str:
    """Lowercase, replace punctuation and symbols with spaces, collapse runs.

    This is the single normalisation shared by all three baselines: BM25 splits
    the result on whitespace, and the character n-gram and Levenshtein scorers
    consume the string itself.

    Deliberately *not* done: u/v and i/j folding, and expansion of scribal
    abbreviations. Both are editorial decisions on early-medieval Latin that
    would make the baseline a small normalisation system in its own right, and
    the point of the baseline is to be the obvious thing a reader would try.
    """
    if text is None:
        return ""
    normalised = unicodedata.normalize("NFC", str(text)).casefold()
    chars = [
        " " if unicodedata.category(ch)[0] in _SEPARATOR_CATEGORIES else ch
        for ch in normalised
    ]
    return " ".join("".join(chars).split())


def tokenise(text: str) -> List[str]:
    """Whitespace tokens of :func:`normalise_text`."""
    return normalise_text(text).split()


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BM25Index:
    """Corpus statistics for BM25, fitted on the train half only."""

    vocab: Dict[str, int]
    idf: np.ndarray
    avgdl: float
    k1: float
    b: float
    n_train_docs: int


def fit_bm25(
    train_token_lists: Sequence[Sequence[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    """Fit vocabulary, IDF and average document length on train documents.

    IDF is the Lucene form ``ln(1 + (N - df + 0.5) / (df + 0.5))``, which is
    strictly positive for every term and so needs none of the negative-IDF
    floors that the textbook Robertson/Sparck-Jones form requires.
    """
    if k1 <= 0:
        raise ValueError(f"k1 must be positive, got {k1}.")
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"b must lie in [0, 1], got {b}.")

    vocab: Dict[str, int] = {}
    doc_freq: List[int] = []
    total_len = 0
    for tokens in train_token_lists:
        total_len += len(tokens)
        for term in set(tokens):
            idx = vocab.get(term)
            if idx is None:
                vocab[term] = len(doc_freq)
                doc_freq.append(1)
            else:
                doc_freq[idx] += 1

    n_docs = len(train_token_lists)
    df = np.asarray(doc_freq, dtype=np.float64)
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    avgdl = (total_len / n_docs) if n_docs else 0.0
    return BM25Index(
        vocab=vocab,
        idf=idf,
        avgdl=float(avgdl),
        k1=float(k1),
        b=float(b),
        n_train_docs=int(n_docs),
    )


def _count_matrix(token_lists: Sequence[Sequence[str]], vocab: Dict[str, int]):
    """Sparse (n_docs, |vocab|) term-count matrix; out-of-vocabulary terms drop."""
    from scipy import sparse

    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    for tokens in token_lists:
        counts: Dict[int, int] = {}
        for term in tokens:
            col = vocab.get(term)
            if col is not None:
                counts[col] = counts.get(col, 0) + 1
        indices.extend(counts.keys())
        data.extend(float(v) for v in counts.values())
        indptr.append(len(indices))

    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(token_lists), max(len(vocab), 1)),
    )


def bm25_score_matrix(
    token_lists: Sequence[Sequence[str]],
    index: BM25Index,
) -> np.ndarray:
    """Raw, *asymmetric* BM25 scores: ``out[a, b]`` scores document ``b`` for query ``a``.

    Query terms count with multiplicity, matching ``rank_bm25.BM25Okapi``, which
    sums over the query token list rather than over distinct query terms.
    Document length in the saturation term is the full token count, including
    out-of-vocabulary tokens: dropping them would make a document look shorter
    than it is and inflate its scores.
    """
    counts = _count_matrix(token_lists, index.vocab)
    doc_lens = np.asarray([len(tokens) for tokens in token_lists], dtype=np.float64)

    weights = counts.copy()
    if weights.nnz:
        # Per-nonzero row index, so the length normaliser can be applied elementwise.
        row_of_nnz = np.repeat(
            np.arange(weights.shape[0]), np.diff(weights.indptr)
        )
        avgdl = index.avgdl if index.avgdl > 0 else 1.0
        denom_norm = index.k1 * (
            1.0 - index.b + index.b * doc_lens[row_of_nnz] / avgdl
        )
        tf = weights.data
        weights.data = (
            index.idf[weights.indices] * tf * (index.k1 + 1.0) / (tf + denom_norm)
        )

    return np.asarray((counts @ weights.T).todense(), dtype=np.float64)


def symmetrise(scores: np.ndarray) -> np.ndarray:
    """``(S + S.T) / 2``, the symmetric form used for pairwise retrieval."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"symmetrise expects a square matrix, got {scores.shape}.")
    return (scores + scores.T) / 2.0


# --------------------------------------------------------------------------
# Character n-gram TF-IDF and Levenshtein
# --------------------------------------------------------------------------


def char_tfidf_score_matrix(
    texts: Sequence[str],
    train_idx: np.ndarray,
    ngram_range: Tuple[int, int] = (3, 5),
) -> np.ndarray:
    """Cosine over character n-gram TF-IDF, vocabulary and IDF fitted on train."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    normalised = [normalise_text(t) for t in texts]
    vectoriser = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        lowercase=False,  # normalise_text already casefolded
        norm="l2",
    )
    vectoriser.fit([normalised[i] for i in np.asarray(train_idx)])
    matrix = vectoriser.transform(normalised)
    return np.asarray((matrix @ matrix.T).todense(), dtype=np.float64)


def levenshtein_score_matrix(texts: Sequence[str], workers: int = -1) -> np.ndarray:
    """Normalised Levenshtein similarity in ``[0, 1]`` over normalised strings.

    Has no fitted parameters, so there is nothing to leak: the train split plays
    no part in it at all.
    """
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    normalised = [normalise_text(t) for t in texts]
    scores = process.cdist(
        normalised,
        normalised,
        scorer=Levenshtein.normalized_similarity,
        workers=workers,
    )
    return np.asarray(scores, dtype=np.float64)


# --------------------------------------------------------------------------
# Train-fitted min-max rescaling
# --------------------------------------------------------------------------


def fit_minmax(scores: np.ndarray, train_idx: np.ndarray) -> Tuple[float, float]:
    """Min and max of the off-diagonal train block, for the rescaling below.

    The diagonal is excluded because a document's score against itself is not a
    retrieval score: for BM25 it is unboundedly larger than any cross-document
    score and would flatten the whole matrix towards zero.
    """
    train_idx = np.asarray(train_idx)
    if train_idx.size < 2:
        raise ValueError("fit_minmax needs at least two train documents.")
    block = np.asarray(scores)[np.ix_(train_idx, train_idx)]
    triu = block[np.triu_indices(block.shape[0], k=1)]
    return float(np.min(triu)), float(np.max(triu))


def apply_minmax(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Affine map sending the train range onto ``[0, 1]``; nothing is clipped."""
    span = hi - lo
    if span <= 0:
        raise ValueError(
            f"Degenerate train score range (min={lo}, max={hi}); cannot rescale."
        )
    return (np.asarray(scores, dtype=np.float64) - lo) / span


# --------------------------------------------------------------------------
# Evaluation (metrics come from the embedding evaluators, unchanged)
# --------------------------------------------------------------------------

def _ensure_import_paths() -> None:
    """Put ``src/`` and ``scripts/resubmit/`` on ``sys.path``.

    Called from every function that reaches for an evaluator rather than at
    module scope, so importing the scoring helpers costs nothing.
    """
    for path in (REPO_ROOT / "src", Path(__file__).resolve().parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


BASELINE_LABELS = {
    "bm25_word": "BM25 (word)",
    "tfidf_char35": "TF-IDF char 3-5",
    "levenshtein": "Levenshtein",
}

# CSV labels stay plain ASCII; the table wants a proper en dash in the range.
LATEX_LABELS = {"TF-IDF char 3-5": "TF-IDF char 3--5"}


def build_score_matrix(
    baseline: str,
    texts: Sequence[str],
    train_idx: np.ndarray,
    workers: int = -1,
) -> np.ndarray:
    """Dispatch to one scorer and return the train-rescaled ``n x n`` matrix."""
    if baseline == "bm25_word":
        token_lists = [tokenise(t) for t in texts]
        index = fit_bm25([token_lists[i] for i in train_idx])
        raw = symmetrise(bm25_score_matrix(token_lists, index))
    elif baseline == "tfidf_char35":
        raw = symmetrise(char_tfidf_score_matrix(texts, train_idx))
    elif baseline == "levenshtein":
        raw = symmetrise(levenshtein_score_matrix(texts, workers=workers))
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    lo, hi = fit_minmax(raw, train_idx)
    return apply_minmax(raw, lo, hi)


def evaluate_single_split(
    sim_all: np.ndarray,
    split_meta: pd.DataFrame,
    label: str,
) -> Dict:
    """Task A + single-split Task B, straight through the embedding evaluator."""
    _ensure_import_paths()
    from run_resubmit_evaluate import evaluate_from_similarity

    train_idx = np.flatnonzero(split_meta["split"].values == "train")
    test_idx = np.flatnonzero(split_meta["split"].values == "test")

    metrics = evaluate_from_similarity(
        train_sim=sim_all[np.ix_(train_idx, train_idx)],
        test_sim=sim_all[np.ix_(test_idx, test_idx)],
        train_folder_ids=split_meta["folder_id"].values[train_idx],
        test_folder_ids=split_meta["folder_id"].values[test_idx],
        test_has_partner=split_meta["has_test_partner"].values[test_idx].astype(bool),
    )

    row = {
        "model": label,
        "repr": "lexical",
        "pooling": "none",
        "layer": pd.NA,
        "method": "lexical",
        "D": pd.NA,
        "sif_a": pd.NA,
    }
    row.update(metrics)
    return row


def evaluate_mseed(
    sim_all: np.ndarray,
    split_meta: pd.DataFrame,
    label: str,
    base_seed: int,
    n_seeds: int,
    top_k: int,
) -> List[Dict]:
    """Task B under the M-seed query/reference protocol, one row per seed.

    The train/test split is fixed, so ``tau`` is learned once; only the
    query/reference partition of the test half is redrawn per seed, exactly as
    ``run_taskb_mseed.py`` does for embeddings.
    """
    _ensure_import_paths()
    from canon_split_v2 import canon_taskb_query_reference_split
    from run_resubmit_evaluate import learn_tau_from_similarity
    from run_taskb_mseed import evaluate_query_vs_directory

    train_idx = np.flatnonzero(split_meta["split"].values == "train")
    tau = learn_tau_from_similarity(
        sim_all[np.ix_(train_idx, train_idx)],
        split_meta["folder_id"].values[train_idx],
    )

    rows: List[Dict] = []
    for seed in range(base_seed, base_seed + n_seeds):
        meta = canon_taskb_query_reference_split(split_meta, random_seed=seed)
        query_idx = np.flatnonzero(meta["taskb_role"].values == "query")
        ref_idx = np.flatnonzero(meta["taskb_role"].values == "reference")

        qvd = evaluate_query_vs_directory(
            None,
            None,
            meta["folder_id"].values[query_idx],
            meta["folder_id"].values[ref_idx],
            meta["has_reference_dir"].values[query_idx].astype(bool),
            tau,
            top_k=top_k,
            rect_sim=sim_all[np.ix_(query_idx, ref_idx)],
        )
        row = {
            "model": label,
            "repr": "lexical",
            "pooling": "none",
            "layer": pd.NA,
            "method": "lexical",
            "D": pd.NA,
            "tau": tau,
            "seed": seed,
        }
        row.update(qvd)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# LaTeX table
# --------------------------------------------------------------------------


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def render_latex_table(results: pd.DataFrame, mseed: Optional[pd.DataFrame]) -> str:
    """Render the lexical rows in the headline tables' metric columns."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Task A}} & \multicolumn{2}{c}{\textbf{Task B}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Baseline} & AUROC & Gap & Assign. & Dir@1 \\",
        r"\midrule",
    ]
    for _, row in results.iterrows():
        label = LATEX_LABELS.get(row["model"], row["model"])
        lines.append(
            f"{label} & {_fmt(row['aucroc'])} & {_fmt(row['gap'])} & "
            f"{_fmt(100 * row['overall_assignment_acc'], 1)} & "
            f"{_fmt(100 * row['dir_acc_at_1'], 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    best = results.loc[results["dir_acc_at_1"].idxmax()]
    caption = (
        "Lexical baselines on the same 50/50 split, the same held-out test set and "
        "the same evaluation code as the embedding models, so the numbers are "
        "directly comparable to the headline Task A and Task B tables. Task A is "
        "measured over the test $n \\times n$ score matrix: AUROC treats "
        "same-directory test pairs as positives, and Gap is the mean same-directory "
        "minus the mean different-directory score. Task B is reported in percent: "
        "Assign.\\ is the existing-versus-new decision against a threshold $\\tau$ "
        "fitted on train, and Dir@1 additionally requires the correct directory. "
        f"{LATEX_LABELS.get(best['model'], best['model'])} is the strongest of the "
        f"three at {_fmt(100 * best['dir_acc_at_1'], 1)} percent Dir@1, which puts "
        "plain surface-form overlap within reach of the best embedding "
        "configurations on this corpus of hand-copied witnesses. BM25 separates "
        "pairs well but routes poorly: a raw BM25 score grows with query length, so "
        "each file's maximum score tracks document length as much as relatedness, "
        "and the train-fitted $\\tau$ transfers badly. BM25 uses $k_1=1.5$, "
        "$b=0.75$ with document frequencies, vocabulary and average document length "
        "taken from train files only; the character TF-IDF vectoriser is likewise "
        "fitted on train only, and normalised Levenshtein has nothing to fit. Every "
        "score matrix is min-max rescaled using the train block, a monotone map "
        "that leaves AUROC and every accuracy unchanged."
    )
    if mseed is not None and len(mseed):
        by_model = mseed.set_index("model")
        parts = []
        for name in results["model"]:  # table order, not groupby order
            if name not in by_model.index:
                continue
            row = by_model.loc[name]
            parts.append(
                f"{LATEX_LABELS.get(name, name)} "
                f"{_fmt(100 * row['dir_acc_at_1_mean'], 1)} "
                f"$\\pm$ {_fmt(100 * row['dir_acc_at_1_std'], 1)}"
            )
        n_seeds = int(mseed["n_seeds"].max())
        caption += (
            f" Under the {n_seeds}-seed query-versus-reference Task B protocol, in "
            "which only the query/reference partition of the test half is redrawn, "
            "directory accuracy at rank 1 is " + "; ".join(parts) + "."
        )

    lines += [
        f"\\caption{{{caption}}}",
        r"\label{tab:lexical_baselines}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lexical (BM25 / char TF-IDF / Levenshtein) retrieval baselines."
    )
    parser.add_argument(
        "--split_csv",
        default=str(REPO_ROOT / "runs/active/resubmit/data/phase_resubmit_split.csv"),
        help="Split CSV with split / folder_id / has_test_partner columns.",
    )
    parser.add_argument(
        "--data_root",
        default=str(REPO_ROOT),
        help="Root the 'path' column of the split CSV is relative to.",
    )
    parser.add_argument(
        "--out_csv",
        default=str(REPO_ROOT / "runs/active/resubmit/results/lexical_baselines.csv"),
        help="Output results CSV (single-split rows).",
    )
    parser.add_argument(
        "--mseed_csv",
        default=str(
            REPO_ROOT / "runs/active/resubmit/results/lexical_baselines_mseed.csv"
        ),
        help="Output per-seed Task B CSV.",
    )
    parser.add_argument(
        "--out_tex",
        default=str(REPO_ROOT / "overleaf_drafts/tables/lexical_baselines.tex"),
        help="Output LaTeX table.",
    )
    parser.add_argument(
        "--baselines",
        default="bm25_word,tfidf_char35,levenshtein",
        help="Comma-separated subset of bm25_word,tfidf_char35,levenshtein.",
    )
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--M", type=int, default=5, help="Seeds for the Task B protocol.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--workers", type=int, default=-1,
        help="Threads for the Levenshtein matrix (-1 = all cores).",
    )
    return parser.parse_args(argv)


def load_texts(split_meta: pd.DataFrame, data_root: Path) -> List[str]:
    texts = []
    for rel in split_meta["path"].values:
        path = Path(rel)
        if not path.is_absolute():
            path = data_root / path
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return texts


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _ensure_import_paths()

    from run_taskb_mseed import aggregate_results

    # reset_index because canon_taskb_query_reference_split writes into arrays by
    # label, so a non-positional index would silently scramble the roles.
    split_meta = pd.read_csv(args.split_csv).reset_index(drop=True)
    texts = load_texts(split_meta, Path(args.data_root))
    train_idx = np.flatnonzero(split_meta["split"].values == "train")
    print(
        f"Loaded {len(texts)} files "
        f"({len(train_idx)} train / {len(texts) - len(train_idx)} test) "
        f"from {args.split_csv}"
    )

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    rows: List[Dict] = []
    mseed_rows: List[Dict] = []

    for baseline in baselines:
        label = BASELINE_LABELS[baseline]
        print(f"\n=== {label} ===")
        sim_all = build_score_matrix(baseline, texts, train_idx, workers=args.workers)

        row = evaluate_single_split(sim_all, split_meta, label)
        rows.append(row)
        print(
            f"  AUROC={row['aucroc']:.4f} gap={row['gap']:.4f} "
            f"assign={row['overall_assignment_acc']:.4f} "
            f"dir@1={row['dir_acc_at_1']:.4f} tau={row['tau']:.4f}"
        )

        if args.M > 0:
            seed_rows = evaluate_mseed(
                sim_all, split_meta, label, args.base_seed, args.M, args.top_k
            )
            mseed_rows.extend(seed_rows)
            dir1 = np.array([r["dir_acc_at_1"] for r in seed_rows])
            print(
                f"  {args.M}-seed dir@1={dir1.mean():.4f} +/- {dir1.std(ddof=1):.4f}"
            )

    results = pd.DataFrame(rows)
    agg = None
    if mseed_rows:
        mseed_df = pd.DataFrame(mseed_rows)
        Path(args.mseed_csv).parent.mkdir(parents=True, exist_ok=True)
        mseed_df.to_csv(args.mseed_csv, index=False)
        agg = aggregate_results(mseed_df, args.top_k)
        # Fold the headline M-seed numbers onto the single-split rows so the CSV
        # is self-contained for the table builder. The two metrics come out equal
        # here, and that is a property of the query-versus-reference protocol
        # rather than a bug: a query is routed correctly exactly when its own
        # directory (or the __NEW__ pseudo-directory) ranks first, which is also
        # the definition of directory accuracy at rank 1. They diverge only in
        # the file-level single-split protocol above.
        for metric in ("dir_acc_at_1", "overall_assignment_acc"):
            for stat in ("mean", "std"):
                col = f"{metric}_{stat}"
                mapping = dict(zip(agg["model"], agg[col]))
                results[f"mseed_{col}"] = results["model"].map(mapping)
        results["mseed_n_seeds"] = args.M
        print(f"\nSaved {len(mseed_df)} per-seed rows to {args.mseed_csv}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_csv, index=False)
    print(f"Saved {len(results)} rows to {args.out_csv}")

    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text(render_latex_table(results, agg), encoding="utf-8")
    print(f"Saved LaTeX table to {args.out_tex}")


if __name__ == "__main__":
    main()
