"""Score the paper's models on dataset 2, the reviewed unlabelled queries (issue #289).

Dataset 2 is the reviewer pilot's verdicts on the 2,238 unlabelled queries,
exported as a pseudonymised snapshot (``runs/active/dataset2/snapshot_*``, never
committed). A query whose reviewers confirmed one CCL key is ``agreed``; one
where they named different keys is ``conflict``; ``no_key`` means "none of the
top 10 served" with no key typed, which says the match was not in the served
list and nothing about whether one exists.

Subcommands
-----------

``select_qwen8b``
    Qwen3-Embedding-8B has no rows in ``phase_resubmit_results.csv``; its only
    paper-side record is the April last-token run (``results/lasttok/``, ABTT
    methods only, on the pre-correction labels). This applies the paper's
    routing rule to it on benchmark v1 *train* only: for every layer 1..36 and
    method, the training DirAcc@1 computed with the paper's own functions
    (``run_resubmit_evaluate.py``), D for the ``*_optimal`` methods tuned the
    same way. Writes ``runs/active/dataset2/qwen8b_train_selection.csv``.

``score``
    For each model and method, at the layer (and D) selected on benchmark v1
    train: rank all 840 labelled directories for every unlabelled query,
    directory score = max cosine over its files (the retrieval script's
    aggregation and degenerate-file guard), save the full ranking, then score
    the reviewed queries. Refuses to score unless mT5-base ``sif_abtt`` at its
    deployed layer reproduces the served top-10.

``manifest``
    sha256, shape and row-order check of every file in the given run dirs.

Layer selection (never on dataset 2)
------------------------------------
The paper's routing layers are the argmax of ``train_dir_acc_at_1`` per
(model, method) in ``runs/active/resubmit/results/phase_resubmit_results.csv``,
ties to the lowest layer; ``build_headline_tables.py`` prints them in
``overleaf_drafts/tables/selected_layers.tex`` ("Routing layer"). ``score``
re-derives them from the CSV and stops if they disagree with
``PAPER_ROUTING_LAYERS`` below, which is copied from that table. Methods the
table does not show (``sif_abtt_fixed``, ``whitening``) use the same rule on the
same CSV.

Dataset 2 has no "new text" queries with ground truth, so ranking is over the
840 directories only, without the paper's ``__NEW__`` pseudo-directory; the
threshold tau plays no part.

Whitening
---------
Whitening is the paper's own (``run_resubmit_evaluate.py``, ``evaluate_single``):
``PCA(whiten=True)`` fit on the train rows, then ``transform``, in the cache's
dtype (float32). It is undefined when the train covariance is rank deficient:
centred, n_train rows span at most n_train - 1 directions, so with
n_train <= d at least one direction has zero variance, and dividing by its
square root lets that one direction swamp every vector outside the fit. That is
Qwen3-0.6B (1024-d), KaLM-mini (896-d) and Qwen3-8B (4096-d) against 847 train
files. Those cells are written as ``not_applicable_rank_deficient`` with the
reason and no number, in every output. The test is n_train <= d alone: an
eigenvalue-ratio threshold cannot separate the cases in float32, where LaBSE
layer 12's healthy fit has a smaller ratio (6e-14: a near-null direction that
every vector shares, so whitening it does no harm) than Qwen3-0.6B layer 5's
degenerate one (2e-12). How the paper reports whitening for d >= n is issue #291, not here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "src"))
sys.path.append(str(REPO / "scripts" / "resubmit"))

from canon_retrieval import l2_normalize, similarity_matrix, zero_norm_mask  # noqa: E402
from embedding_alignment import AlignmentResolver, load_row_order  # noqa: E402
from run_resubmit_evaluate import (  # noqa: E402
    directory_assignment_accuracy_at_k,
    find_optimal_D_phase11,
    has_partner_flags,
    learn_tau_from_similarity,
)
from run_resubmit_unlabelled_retrieval import (  # noqa: E402
    find_optimal_D as find_optimal_D_deployed,
    read_blank_source_mask,
)
from sif_abtt import EmbeddingCleaner  # noqa: E402

QWEN8B = "Qwen/Qwen3-Embedding-8B"
MT5 = "google/mt5-base"

# (model id, display). The paper's six for the SRW draft plus mT5-base, which is
# not one of them but produced 94% of the rankings reviewers saw.
MODELS: List[Tuple[str, str]] = [
    ("bowphs/LaTa", "LaTa"),
    ("bowphs/PhilTa", "PhilTa"),
    ("sentence-transformers/LaBSE", "LaBSE"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B"),
    ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", "KaLM-mini"),
    (QWEN8B, "Qwen3-8B"),
    (MT5, "mT5-base"),
]
DISPLAY = dict(MODELS)

# method -> (pooling, post-processing). ``abtt_optimal`` is the paper's "ABTT"
# column; ``lasttok_abtt_optimal`` is scored only where a last-token unlabelled
# cache exists (Qwen3-8B).
METHODS: Dict[str, Tuple[str, str]] = {
    "baseline": ("mean", "none"),
    "sif_only": ("sif", "none"),
    "abtt_optimal": ("mean", "abtt_optimal"),
    "sif_abtt_fixed": ("sif", "abtt_fixed"),
    "sif_abtt_optimal": ("sif", "abtt_optimal"),
    "whitening": ("mean", "whitening"),
    "lasttok_abtt_optimal": ("lasttok", "abtt_optimal"),
}
RESULTS_METHOD = {m: ("abtt_optimal" if m == "lasttok_abtt_optimal" else m) for m in METHODS}
FIXED_D = 10
D_VALUES = [1, 2, 3, 5, 7, 10]
TRAIN_METRIC = "train_dir_acc_at_1"

# overleaf_drafts/tables/selected_layers.tex, "Routing layer" columns
# (Base, SIF, ABTT, SIF+ABTT). A check, not a source: the layers are re-derived.
PAPER_ROUTING_LAYERS = {
    "bowphs/LaTa": (1, 1, 8, 12),
    "bowphs/PhilTa": (1, 1, 1, 1),
    MT5: (12, 1, 1, 2),
    "sentence-transformers/LaBSE": (11, 12, 11, 11),
    "Qwen/Qwen3-Embedding-0.6B": (28, 27, 5, 5),
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": (23, 22, 3, 1),
}
PAPER_TABLE_METHODS = ("baseline", "sif_only", "abtt_optimal", "sif_abtt_optimal")

SUFFIX = {"mean": "", "sif": "_sif", "lasttok": "_lasttok"}
KS = (1, 5, 10)
SERVED_K = 10
NONE_MARKER = "(none_of_top_k)"
UNRESOLVED_MARKER = "(unresolved)"
LIST_SEPARATOR = " | "
DECISION_OUTCOMES = ("matched_rank", "none_of_top_k")
SCORED = "scored"
NOT_APPLICABLE = "not_applicable_rank_deficient"

DEFAULTS = {
    "split_csv": "runs/active/resubmit/data/phase_resubmit_split.csv",
    "unlabelled_meta": "runs/active/resubmit/data/unlabelled_meta.csv",
    "labelled_bases": "runs/active/resubmit_bases/phase9_bases",
    "unlabelled_bases": "runs/active/resubmit/unlabelled/bases",
    "results_csv": "runs/active/resubmit/results/phase_resubmit_results.csv",
    "qwen8b_selection": "runs/active/dataset2/qwen8b_train_selection.csv",
    "april_lasttok_dir": "runs/active/resubmit/results/lasttok",
    "served_dir": "runs/active/resubmit/unlabelled",
    "deployed_layers_json": "scripts/resubmit/deployed_unlabelled_layers.json",
    "out_dir": "runs/active/dataset2",
}


def slug(model: str) -> str:
    return model.replace("/", "_")


def emb_path(bases: Path, model: str, pooling: str, layer: int) -> Path:
    return (
        Path(bases) / slug(model) / f"hidden_{pooling}_tokempty"
        / f"hidden_layer{layer}_embeddings{SUFFIX[pooling]}.npy"
    )


# --- CCL key matching (web/services/ccl_keys.py, reimplemented: no imports
# cross the webapp boundary) --------------------------------------------------

_WS = re.compile(r"\s+")


def match_form(key) -> str:
    """Trimmed, inner whitespace collapsed, casefolded: how the webapp compares keys."""
    if key is None or (isinstance(key, float) and math.isnan(key)):
        return ""
    return _WS.sub(" ", str(key).strip()).casefold()


def key_to_dir_index(dir_names: Sequence[str]) -> Dict[str, int]:
    """match_form -> directory column. Ties go to the first name in sorted order."""
    out: Dict[str, int] = {}
    for name in sorted(dir_names):
        out.setdefault(match_form(name), list(dir_names).index(name))
    return out


# --- post-processing ---------------------------------------------------------


class RankDeficientWhitening(ValueError):
    """Whitening was asked for on a train set that cannot define it."""


def whitening_rank_deficiency(n_train: int, dim: int) -> Optional[str]:
    """The reason whitening is undefined for an (n_train, dim) fit, or None."""
    if n_train <= dim:
        return (f"train covariance rank deficient: n_train={n_train} <= d={dim}, "
                f"so centred train rows span at most {n_train - 1} of {dim} directions")
    return None


def paper_whitening(train: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """``run_resubmit_evaluate.evaluate_single``'s whitening: fit on train, then
    transform, at the input dtype. Raises on a rank-deficient fit."""
    reason = whitening_rank_deficiency(*train.shape)
    if reason:
        raise RankDeficientWhitening(reason)
    pca = PCA(whiten=True)
    pca.fit(train)
    return tuple(pca.transform(a) for a in arrays)


def post_process(
    method_post: str, lab: np.ndarray, unlab: np.ndarray, train_mask: np.ndarray,
    train_folder_ids: np.ndarray, d_override: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """Fit on the labelled train rows only, apply to the pool and the queries."""
    train = lab[train_mask]
    if method_post == "none":
        return lab, unlab, None
    if method_post in ("abtt_fixed", "abtt_optimal"):
        if method_post == "abtt_fixed":
            D = FIXED_D
        elif d_override is not None:
            D = d_override
        else:
            D, _ = find_optimal_D_phase11(train, train_folder_ids, D_VALUES)
        cleaner = EmbeddingCleaner(num_components=D, center=True).fit(train)
        return cleaner.transform(lab), cleaner.transform(unlab), D
    if method_post == "whitening":
        lab_w, unlab_w = paper_whitening(train, lab, unlab)
        return lab_w, unlab_w, None
    raise ValueError(method_post)


def directory_scores(
    lab_norm: np.ndarray, unlab_norm: np.ndarray, folder_ids: np.ndarray,
    dir_names: Sequence[str], exclude_lab: np.ndarray,
) -> np.ndarray:
    """(n_query, n_dir) max cosine to any usable file of each directory.

    A directory whose files are all excluded scores -inf, so it ranks last,
    where the retrieval script drops it from the candidate set.
    """
    cross = unlab_norm @ lab_norm.T
    out = np.full((cross.shape[0], len(dir_names)), -np.inf, dtype=np.float64)
    usable = ~np.asarray(exclude_lab, dtype=bool)
    for j, name in enumerate(dir_names):
        cols = np.flatnonzero((folder_ids == name) & usable)
        if len(cols):
            out[:, j] = cross[:, cols].max(axis=1)
    return out


def rank_order(scores: np.ndarray) -> np.ndarray:
    """Column order per row, best first; ties keep directory order (stable), as served."""
    return np.argsort(-scores, axis=1, kind="stable")


# --- select_qwen8b -------------------------------------------------------------


def _train_score(train: np.ndarray, folder_ids: np.ndarray) -> float:
    """The paper's train DirAcc@1 (``evaluate_from_similarity``), train rows only."""
    sim = similarity_matrix(l2_normalize(train))
    tau = learn_tau_from_similarity(sim, folder_ids)
    return directory_assignment_accuracy_at_k(
        sim, folder_ids, has_partner_flags(folder_ids), tau, k=1
    )


def _select_layer_job(job) -> List[Dict]:
    bases, split_csv, pooling, layer = job
    split = pd.read_csv(split_csv)
    path = emb_path(Path(bases), QWEN8B, pooling, layer)
    if not path.exists():
        return []
    native = AlignmentResolver(split).load(path)
    emb = native.astype(np.float64)
    train_mask = split["split"].values == "train"
    fids = split.loc[train_mask, "folder_id"].values
    train = emb[train_mask]
    rows = []

    def row(method, D, score, status=SCORED, reason=""):
        rows.append({"model": QWEN8B, "repr": "hidden", "pooling": pooling, "layer": layer,
                     "method": method, "D": D, TRAIN_METRIC: score,
                     "status": status, "reason": reason})

    if pooling == "mean":
        row("baseline", None, _train_score(train, fids))
        try:
            (white,) = paper_whitening(native[train_mask], native[train_mask])
            row("whitening", None, _train_score(white, fids))
        except RankDeficientWhitening as e:
            row("whitening", None, float("nan"), NOT_APPLICABLE, str(e))
    if pooling == "sif":
        row("sif_only", None, _train_score(train, fids))
    prefix = "sif_" if pooling == "sif" else ""
    fixed = EmbeddingCleaner(num_components=FIXED_D, center=True).fit_transform(train)
    row(f"{prefix}abtt_fixed", FIXED_D, _train_score(fixed, fids))
    D, score = find_optimal_D_phase11(train, fids, D_VALUES)
    row(f"{prefix}abtt_optimal", D, score)
    print(f"  {pooling} layer {layer}: done", flush=True)
    return rows


def cmd_select_qwen8b(args) -> None:
    jobs = [(args.labelled_bases, args.split_csv, pooling, layer)
            for pooling in ("mean", "sif", "lasttok") for layer in range(1, 37)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for chunk in pool.map(_select_layer_job, jobs) for r in chunk]
    df = pd.DataFrame(rows).sort_values(["pooling", "method", "layer"], kind="stable")
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")

    # How the April last-token run (pre-correction labels) would have chosen.
    april = []
    for name, pooling in (("phase_resubmit_results_mean_abttonly.csv", "mean"),
                          ("phase_resubmit_results_lasttok.csv", "lasttok")):
        p = Path(args.april_lasttok_dir) / name
        if p.exists():
            a = pd.read_csv(p)
            a = a[(a["model"] == QWEN8B) & (a["method"] == "abtt_optimal")].sort_values("layer")
            if len(a):
                best = a.loc[a[TRAIN_METRIC].idxmax()]
                april.append((pooling, int(best["layer"]), int(best["D"])))
    for pooling, layer, D in april:
        ours = df[(df["pooling"] == pooling) & (df["method"] == "abtt_optimal")].sort_values("layer")
        b = ours.loc[ours[TRAIN_METRIC].idxmax()]
        print(f"  {pooling} abtt_optimal: April CSV layer {layer} D {D}; "
              f"benchmark v1 layer {int(b['layer'])} D {int(b['D'])}")


# --- layer selection -------------------------------------------------------------


def select_layers(results_csv: Path, qwen8b_csv: Path) -> pd.DataFrame:
    """One row per (model, method): layer, D and the train score it was chosen on."""
    paper = pd.read_csv(results_csv)
    q8 = pd.read_csv(qwen8b_csv) if Path(qwen8b_csv).exists() else pd.DataFrame()
    rows = []
    for model, _ in MODELS:
        src_df, src = (q8, str(qwen8b_csv)) if model == QWEN8B else (paper, str(results_csv))
        for method, (pooling, _) in METHODS.items():
            if src_df.empty:
                continue
            sub = src_df[(src_df["model"] == model) & (src_df["repr"] == "hidden")
                         & (src_df["method"] == RESULTS_METHOD[method])
                         & (src_df["pooling"] == pooling)]
            if sub.empty:
                continue
            sub = sub.sort_values("layer", kind="stable")
            if sub[TRAIN_METRIC].isna().all():
                # No layer has a train score (whitening with n_train <= d at
                # every layer), so there is nothing to select.
                reasons = sub["reason"].dropna() if "reason" in sub else []
                rows.append({
                    "model": model, "display": DISPLAY[model], "method": method,
                    "pooling": pooling, "layer": None, "D": None, TRAIN_METRIC: None,
                    "source": src, "status": NOT_APPLICABLE,
                    "reason": "every layer: " + (reasons.iloc[0] if len(reasons)
                                                 else "no train score"),
                })
                continue
            best = sub.loc[sub[TRAIN_METRIC].idxmax()]
            D = best["D"]
            rows.append({
                "model": model, "display": DISPLAY[model], "method": method,
                "pooling": pooling, "layer": int(best["layer"]),
                "D": (int(D) if METHODS[method][1].startswith("abtt") and not pd.isna(D) else None),
                TRAIN_METRIC: float(best[TRAIN_METRIC]), "source": src,
                "status": SCORED, "reason": "",
            })
    sel = pd.DataFrame(rows)
    sel["layer"] = sel["layer"].astype("Int64")
    sel["D"] = sel["D"].astype("Int64")

    bad = []
    for model, layers in PAPER_ROUTING_LAYERS.items():
        for method, want in zip(PAPER_TABLE_METHODS, layers):
            got = sel[(sel["model"] == model) & (sel["method"] == method)]
            if got.empty or int(got["layer"].iloc[0]) != want:
                bad.append((model, method, want, None if got.empty else int(got["layer"].iloc[0])))
    if bad:
        raise SystemExit(f"train selection disagrees with selected_layers.tex: {bad}")
    return sel


# --- snapshot ---------------------------------------------------------------------


def load_snapshot(snapshot: Path, query_names: Sequence[str]) -> pd.DataFrame:
    """One row per reviewed query: label status, keys, and the model(s) reviewers saw."""
    reviews = pd.read_csv(snapshot / "reviews.csv")
    keys = pd.read_csv(snapshot / "ccl_key_assignments.csv")
    pos = {name: i for i, name in enumerate(query_names)}

    dec = reviews[reviews["outcome"].isin(DECISION_OUTCOMES)]
    seen = dec.groupby("xml_id")["model_slug"].agg(lambda s: sorted(set(s)))
    q = keys[["xml_id", "ccl_key", "status", "conflicting_keys"]].copy()
    q["seen_models"] = q["xml_id"].map(seen)
    q["seen_group"] = [
        (m[0] if len(m) == 1 else "mixed") if isinstance(m, list) else "none"
        for m in q["seen_models"]
    ]
    q["filename"] = q["xml_id"].astype(str) + ".txt"
    q["query_row"] = q["filename"].map(pos)
    if q["query_row"].isna().any():
        raise SystemExit(f"{int(q['query_row'].isna().sum())} xml_id(s) not among the queries")
    q["query_row"] = q["query_row"].astype(int)
    return q


def label_dirs(q: pd.DataFrame, key_index: Dict[str, int]) -> Tuple[List[List[int]], List[int]]:
    """Directory columns each query's label names, and how many named keys did not map."""
    cols, unmapped = [], []
    for _, r in q.iterrows():
        if r["status"] == "agreed":
            names = [r["ccl_key"]]
        elif r["status"] == "conflict":
            names = [k for k in str(r["conflicting_keys"]).split(LIST_SEPARATOR)
                     if k not in (NONE_MARKER, UNRESOLVED_MARKER) and k.strip()]
        else:
            names = []
        mapped = [key_index[match_form(k)] for k in names if match_form(k) in key_index]
        cols.append(sorted(set(mapped)))
        unmapped.append(len(names) - len([k for k in names if match_form(k) in key_index]))
    return cols, unmapped


def load_served(served_dir: Path, model_slug: str, query_names: Sequence[str]) -> Optional[List[List[str]]]:
    path = Path(served_dir) / f"unlabelled_predictions_sif_abtt_{model_slug}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).set_index("filename")
    cols = [f"rank{k}_dir" for k in range(1, SERVED_K + 1)]
    out = []
    for name in query_names:
        vals = df.loc[name, cols].tolist() if name in df.index else []
        out.append([v for v in vals if isinstance(v, str)])
    return out


# --- metrics -------------------------------------------------------------------


def metric_block(ranks: np.ndarray) -> Dict[str, float]:
    r = ranks.astype(float)
    out = {f"acc@{k}": float(np.mean(r <= k)) for k in KS}
    out["mrr"] = float(np.mean(1.0 / r))
    return out


def bootstrap(ranks: np.ndarray, idx: np.ndarray) -> Dict[str, Tuple[float, float]]:
    r = ranks.astype(float)[idx]  # (B, n)
    out = {}
    for k in KS:
        v = (r <= k).mean(axis=1)
        out[f"acc@{k}"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    v = (1.0 / r).mean(axis=1)
    out["mrr"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


# --- score -------------------------------------------------------------------------


def score_one(
    model: str, method: str, layer: int, D: Optional[int], ctx: Dict,
) -> Optional[Dict]:
    """Directory scores for one cell, or ``{"status": NOT_APPLICABLE, ...}``."""
    pooling, post = METHODS[method]
    lab_p = emb_path(ctx["labelled_bases"], model, pooling, layer)
    unlab_p = emb_path(ctx["unlabelled_bases"], model, pooling, layer)
    if not lab_p.exists() or not unlab_p.exists():
        print(f"  {model} {method} L{layer}: missing {lab_p if not lab_p.exists() else unlab_p}")
        return None
    lab_native = ctx["lab_resolver"].load(lab_p)
    unlab_native = ctx["unlab_resolver"].load(unlab_p)
    lab = lab_native.astype(np.float64)
    unlab = unlab_native.astype(np.float64)
    exclude_lab = ctx["lab_blank"] | zero_norm_mask(lab)
    exclude_q = ctx["unlab_blank"] | zero_norm_mask(unlab)
    if post == "whitening":
        # The paper whitens the cache as loaded (float32), not a float64 copy.
        lab, unlab = lab_native, unlab_native
    try:
        lab, unlab, D_used = post_process(post, lab, unlab, ctx["train_mask"],
                                          ctx["train_fids"], d_override=D)
    except RankDeficientWhitening as e:
        return {"status": NOT_APPLICABLE, "reason": str(e), "D": None,
                "excluded_queries": exclude_q, "excluded_refs": exclude_lab}
    scores = directory_scores(l2_normalize(lab), l2_normalize(unlab), ctx["folder_ids"],
                              ctx["dir_names"], exclude_lab)
    scores[exclude_q] = np.nan
    return {"status": SCORED, "reason": "", "scores": scores, "D": D_used,
            "excluded_queries": exclude_q, "excluded_refs": exclude_lab}


def save_ranking(path: Path, res: Dict, model: str, method: str, layer: int, ctx: Dict) -> None:
    scores = res["scores"]
    order = rank_order(np.nan_to_num(scores, nan=-np.inf)).astype(np.int16)
    np.savez_compressed(
        path,
        scores=scores.astype(np.float32),
        order=order,
        dir_names=np.array(ctx["dir_names"], dtype=object),
        query_filenames=np.array(ctx["query_names"], dtype=object),
        excluded_queries=res["excluded_queries"],
        excluded_refs=res["excluded_refs"],
        model=model, method=method, layer=layer,
        D=-1 if res["D"] is None else res["D"],
    )


def ranks_of(order: np.ndarray, label_cols: List[List[int]], rows: np.ndarray) -> np.ndarray:
    """1-based rank of the best-ranked labelled directory, per query (inf if none)."""
    out = np.full(len(rows), np.inf)
    for i, (r, cols) in enumerate(zip(rows, label_cols)):
        if cols:
            pos = np.empty(order.shape[1], dtype=np.int64)
            pos[order[r]] = np.arange(order.shape[1])
            out[i] = 1 + min(pos[c] for c in cols)
    return out


def sanity_served(ctx: Dict, args) -> Dict:
    """mT5-base sif_abtt at its deployed layer must reproduce the served top-10."""
    deployed = json.loads(Path(args.deployed_layers_json).read_text())["layers"]["sif_abtt"][MT5]
    # The served CSVs tune D by assignment accuracy (the retrieval script's
    # find_optimal_D), not by the paper's DirAcc@1, so reproduce that here.
    lab = ctx["lab_resolver"].load(emb_path(ctx["labelled_bases"], MT5, "sif", deployed))
    D = find_optimal_D_deployed(
        lab.astype(np.float64)[ctx["train_mask"]], ctx["train_fids"], D_VALUES
    )
    res = score_one(MT5, "sif_abtt_optimal", deployed, D, ctx)
    order = rank_order(np.nan_to_num(res["scores"], nan=-np.inf))
    served = load_served(args.served_dir, slug(MT5), ctx["query_names"])
    dn = ctx["dir_names"]
    top1 = exact = n = 0
    for i, s in enumerate(served):
        if not s or res["excluded_queries"][i]:
            continue
        n += 1
        mine = [dn[j] for j in order[i, :SERVED_K]]
        top1 += mine[0] == s[0]
        exact += mine[:len(s)] == s
    return {"model": MT5, "variant": "sif_abtt", "deployed_layer": int(deployed), "D": int(D),
            "queries_compared": n, "top1_agree": top1, "top10_exact_agree": exact,
            "top1_agree_frac": top1 / n if n else float("nan"),
            "top10_exact_agree_frac": exact / n if n else float("nan")}


def cmd_score(args) -> None:
    out = Path(args.out_dir)
    (out / "rankings").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(args.split_csv)
    qmeta = pd.read_csv(args.unlabelled_meta)
    query_names = qmeta["filename"].astype(str).tolist()
    dir_names = list(dict.fromkeys(split["folder_id"].astype(str)))  # retrieval's order
    ctx = {
        "split": split,
        "labelled_bases": Path(args.labelled_bases),
        "unlabelled_bases": Path(args.unlabelled_bases),
        "lab_resolver": AlignmentResolver(split),
        "unlab_resolver": AlignmentResolver(qmeta),
        "query_names": query_names,
        "dir_names": dir_names,
        "folder_ids": split["folder_id"].astype(str).values,
        "train_mask": split["split"].values == "train",
        "train_fids": split.loc[split["split"] == "train", "folder_id"].values,
        "lab_blank": read_blank_source_mask(split["path"].tolist(), Path("data")),
        "unlab_blank": read_blank_source_mask(qmeta["path"].tolist(), Path("data")),
    }

    sanity = {"served_reproduction": sanity_served(ctx, args)}
    sr = sanity["served_reproduction"]
    print(f"Served reproduction: {sr}")

    reviews = pd.read_csv(Path(args.snapshot) / "reviews.csv")
    matched = reviews[reviews["outcome"] == "matched_rank"]
    sanity["matched_rank"] = {
        "rows": int(len(matched)),
        "rank1_rows": int((matched["correct_rank"] == 1).sum()),
        "rows_mt5": int((matched["model_slug"] == slug(MT5)).sum()),
        "rank1_rows_mt5": int(((matched["correct_rank"] == 1)
                               & (matched["model_slug"] == slug(MT5))).sum()),
    }
    print(f"matched_rank: {sanity['matched_rank']}")
    (out / "results" / "sanity.json").write_text(json.dumps(sanity, indent=2) + "\n")
    if sr["top10_exact_agree_frac"] < args.min_served_agreement and not args.force:
        raise SystemExit(
            f"mT5 sif_abtt reproduces only {sr['top10_exact_agree_frac']:.4f} of served "
            f"top-10 lists (< {args.min_served_agreement}); stopping before scoring."
        )

    sel = select_layers(Path(args.results_csv), Path(args.qwen8b_selection))
    sel.to_csv(out / "results" / "selected_layers.csv", index=False)
    print(sel.to_string(index=False))

    q = load_snapshot(Path(args.snapshot), query_names)
    key_index = key_to_dir_index(dir_names)
    q["label_cols"], q["unmapped_keys"] = label_dirs(q, key_index)
    served_mt5 = load_served(args.served_dir, slug(MT5), query_names)
    served_by_slug = {g: load_served(args.served_dir, g, query_names)
                      for g in q["seen_group"].unique() if g not in ("mixed", "none")}

    label_sets = {
        "agreed": q["status"] == "agreed",
        "agreed_plus_conflict": q["status"].isin(["agreed", "conflict"]),
        "conflict": q["status"] == "conflict",
    }
    rng = np.random.default_rng(args.seed)
    boot_idx: Dict[Tuple[str, str], np.ndarray] = {}

    per_query = q[["xml_id", "status", "seen_group", "unmapped_keys"]].copy()
    acc_rows, nokey_rows = [], []
    for _, s in sel.iterrows():
        model, method = s["model"], s["method"]
        layer = None if pd.isna(s["layer"]) else int(s["layer"])
        D = None if pd.isna(s["D"]) else int(s["D"])
        col = f"{DISPLAY[model]}|{method}"
        ranking_path = out / "rankings" / f"{slug(model)}__{method}.npz"
        print(f"Scoring {DISPLAY[model]} {method} L{layer} D{D}", flush=True)
        # D is re-tuned on train rather than read from the CSV, and the two are
        # compared: a mismatch would mean the cache is not what the paper scored.
        if layer is None:
            res = {"status": s["status"], "reason": s["reason"], "D": None}
        else:
            res = score_one(model, method, layer, None, ctx)
        if res is None:
            continue
        sel.loc[s.name, ["status", "reason"]] = [res["status"], res["reason"]]
        if res["status"] == NOT_APPLICABLE:
            print(f"  {NOT_APPLICABLE}: {res['reason']}")
            ranking_path.unlink(missing_ok=True)  # never leave a stale ranking behind
            per_query[f"rank:{col}"] = NOT_APPLICABLE
            acc_rows += not_applicable_rows(q, label_sets, DISPLAY[model], method, layer,
                                            res["reason"])
            nokey_rows.append({"model": DISPLAY[model], "method": method, "layer": layer,
                               "status": NOT_APPLICABLE, "reason": res["reason"]})
            continue
        sel.loc[s.name, "D_retuned"] = res["D"]
        if METHODS[method][1] == "abtt_optimal" and D is not None and res["D"] != D:
            print(f"  WARNING: re-tuned D {res['D']} != recorded D {D}")
        save_ranking(ranking_path, res, model, method, layer, ctx)
        order = rank_order(np.nan_to_num(res["scores"], nan=-np.inf))
        excl = res["excluded_queries"]
        rows = q["query_row"].values
        ranks = ranks_of(order, q["label_cols"].tolist(), rows)
        ranks[excl[rows]] = np.nan
        per_query[f"rank:{col}"] = ranks

        for set_name, mask in label_sets.items():
            groups = {"all": mask.values}
            for g in sorted(q.loc[mask, "seen_group"].unique()):
                groups[g] = (mask & (q["seen_group"] == g)).values
            for g, gmask in groups.items():
                r = ranks[gmask]
                r = r[~np.isnan(r)]
                n = len(r)
                if n == 0:
                    continue
                key = (set_name, g)
                if key not in boot_idx or boot_idx[key].shape[1] != n:
                    boot_idx[key] = rng.integers(0, n, size=(args.n_boot, n))
                point = metric_block(r)
                ci = bootstrap(r, boot_idx[key])
                for m, v in point.items():
                    acc_rows.append({
                        "label_set": set_name, "seen_group": g, "n": n,
                        "n_excluded": int(gmask.sum() - n),
                        "model": DISPLAY[model], "method": method, "layer": layer,
                        "D": res["D"], "metric": m, "value": v,
                        "ci_lo": ci[m][0], "ci_hi": ci[m][1],
                        "status": SCORED, "reason": "",
                    })

        nk = (q["status"] == "no_key").values
        top1 = order[:, 0]
        in_mt5, in_seen, n_seen, n = 0, 0, 0, 0
        top1_names = []
        for i in np.flatnonzero(nk):
            r = int(q["query_row"].iloc[i])
            if excl[r]:
                top1_names.append(None)
                continue
            name = dir_names[top1[r]]
            top1_names.append(name)
            n += 1
            in_mt5 += name in (served_mt5[r] if served_mt5 else [])
            g = q["seen_group"].iloc[i]
            if g in served_by_slug and served_by_slug[g]:
                n_seen += 1
                in_seen += name in served_by_slug[g][r]
        nokey_rows.append({
            "model": DISPLAY[model], "method": method, "layer": layer, "n": n,
            "top1_in_mt5_served_top10": in_mt5,
            "frac_in_mt5_served_top10": in_mt5 / n if n else float("nan"),
            "n_single_seen_model": n_seen,
            "top1_in_seen_served_top10": in_seen,
            "frac_in_seen_served_top10": in_seen / n_seen if n_seen else float("nan"),
            "status": SCORED, "reason": "",
        })

    sel.to_csv(out / "results" / "selected_layers.csv", index=False)
    acc = pd.DataFrame(acc_rows)
    for c in ("layer", "D", "n_excluded"):
        acc[c] = acc[c].astype("Int64")
    acc.to_csv(out / "results" / "accuracy_long.csv", index=False)
    nokey = pd.DataFrame(nokey_rows)
    nokey["layer"] = nokey["layer"].astype("Int64")
    nokey.to_csv(out / "results" / "nokey_top1_vs_served.csv", index=False)
    per_query.to_csv(out / "results" / "per_query_ranks.csv", index=False)
    headline_table(acc).to_csv(out / "results" / "headline.csv", index=False)

    counts = {
        "queries_with_snapshot_row": int(len(q)),
        "status": q["status"].value_counts().to_dict(),
        "seen_group": q["seen_group"].value_counts().to_dict(),
        "keyed_queries_with_unmapped_key": int(((q["unmapped_keys"] > 0)
                                                & (q["status"] != "no_key")).sum()),
        "keyed_queries_with_no_mapped_dir": int(((q["label_cols"].str.len() == 0)
                                                 & (q["status"] != "no_key")).sum()),
    }
    (out / "results" / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    write_summary(out / "results" / "summary.md", acc, nokey, sel, sanity, counts)
    print(ctx["lab_resolver"].summary())
    print(ctx["unlab_resolver"].summary())
    print(f"Done -> {out / 'results'}")


def not_applicable_rows(q: pd.DataFrame, label_sets: Dict[str, pd.Series], display: str,
                        method: str, layer: Optional[int], reason: str) -> List[Dict]:
    """accuracy_long rows for a cell with no number: same keys, value left empty."""
    rows = []
    for set_name, mask in label_sets.items():
        groups = {"all": mask.values}
        for g in sorted(q.loc[mask, "seen_group"].unique()):
            groups[g] = (mask & (q["seen_group"] == g)).values
        for g, gmask in groups.items():
            if not gmask.any():
                continue
            for m in [f"acc@{k}" for k in KS] + ["mrr"]:
                rows.append({
                    "label_set": set_name, "seen_group": g, "n": int(gmask.sum()),
                    "n_excluded": None, "model": display, "method": method,
                    "layer": layer, "D": None, "metric": m, "value": None,
                    "ci_lo": None, "ci_hi": None, "status": NOT_APPLICABLE,
                    "reason": reason,
                })
    return rows


def headline_table(acc: pd.DataFrame) -> pd.DataFrame:
    """One row per (label set, model, method) over all queries; metrics as columns."""
    keys = ["label_set", "model", "method", "layer", "D", "n", "status", "reason"]
    sub = acc[acc["seen_group"] == "all"]
    wide = sub.pivot(index=["label_set", "model", "method"], columns="metric", values="value")
    meta = sub.drop_duplicates(["label_set", "model", "method"])[keys]
    return meta.merge(wide.reset_index(), on=["label_set", "model", "method"], how="left")


def write_summary(path: Path, acc, nokey, sel, sanity, counts) -> None:
    def fmt(v, lo, hi):
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

    lines = ["# Dataset 2 scoring (issue #289)", "",
             "Layers and D selected on benchmark v1 train (see `selected_layers.csv`).",
             "Ranking over the 840 labelled directories; 95% percentile bootstrap CIs.", "",
             f"Counts: `{json.dumps(counts)}`", "",
             f"Sanity: `{json.dumps(sanity)}`", "",
             "Whitening is the paper's (`PCA(whiten=True)` fit on the train rows, float32).",
             f"A cell marked `{NOT_APPLICABLE}` has no number: with n_train <= d the train",
             "covariance has a zero-variance direction, and whitening by it collapses every",
             "query onto a few directories. Paper-side handling is issue #291.", ""]
    for set_name in ("agreed", "agreed_plus_conflict", "conflict"):
        sub = acc[(acc["label_set"] == set_name) & (acc["seen_group"] == "all")]
        if sub.empty:
            continue
        n = int(sub["n"].iloc[0])
        lines += [f"## {set_name} (n={n})", "",
                  "| model | method | layer | D | acc@1 | acc@5 | acc@10 | MRR |",
                  "|---|---|---|---|---|---|---|---|"]
        for (model, method), g in sub.groupby(["model", "method"], sort=False):
            g = g.set_index("metric")
            D = g["D"].iloc[0]
            layer = g["layer"].iloc[0]
            layer = "" if pd.isna(layer) else int(layer)
            if g["status"].iloc[0] == NOT_APPLICABLE:
                lines.append(f"| {model} | {method} | {layer} | | {NOT_APPLICABLE}: "
                             f"{g['reason'].iloc[0]} | | | |")
                continue
            cells = [fmt(g.loc[m, "value"], g.loc[m, "ci_lo"], g.loc[m, "ci_hi"])
                     for m in ("acc@1", "acc@5", "acc@10", "mrr")]
            lines.append(f"| {model} | {method} | {layer} | "
                         f"{'' if pd.isna(D) else int(D)} | " + " | ".join(cells) + " |")
        lines.append("")
    nokey = nokey.astype(object).where(nokey.notna(), "")  # no "nan" in place of a number
    try:
        table = nokey.to_markdown(index=False)
    except ImportError:  # tabulate is optional
        table = "```\n" + nokey.to_string(index=False) + "\n```"
    lines += ["## no_key: top-1 inside the served top-10", "", table, ""]
    path.write_text("\n".join(lines) + "\n")


# --- manifest -----------------------------------------------------------------------


def cmd_manifest(args) -> None:
    entries = []
    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        order = load_row_order(run_dir / "meta.csv")
        cfg = json.loads((run_dir / "config.json").read_text())
        files = []
        for p in sorted(run_dir.iterdir()):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            rec = {"file": p.name, "bytes": p.stat().st_size, "sha256": h.hexdigest()}
            if p.suffix == ".npy":
                arr = np.load(p, mmap_mode="r")
                rec["shape"] = list(arr.shape)
                rec["dtype"] = str(arr.dtype)
                if arr.shape[0] != len(order):
                    raise SystemExit(f"{p}: {arr.shape[0]} rows, meta.csv has {len(order)}")
            files.append(rec)
        expected = None
        if args.expected_meta:
            for em in args.expected_meta:
                names = load_row_order(em)
                if set(names) == set(order):
                    expected = {"meta": str(em), "same_set": True,
                                "same_order": names == order}
        entries.append({"run_dir": str(run_dir), "rows": len(order), "config": cfg,
                        "alignment": expected, "files": files})
        print(f"{run_dir}: {len(files)} files, {len(order)} rows, alignment {expected}")
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"issue": 289, "entries": entries}, indent=2) + "\n")
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select_qwen8b")
    s.add_argument("--labelled_bases", default=DEFAULTS["labelled_bases"])
    s.add_argument("--split_csv", default=DEFAULTS["split_csv"])
    s.add_argument("--april_lasttok_dir", default=DEFAULTS["april_lasttok_dir"])
    s.add_argument("--out_csv", default=DEFAULTS["qwen8b_selection"])
    s.add_argument("--workers", type=int, default=8)

    c = sub.add_parser("score")
    for k in ("split_csv", "unlabelled_meta", "labelled_bases", "unlabelled_bases",
              "results_csv", "qwen8b_selection", "served_dir", "deployed_layers_json",
              "out_dir"):
        c.add_argument(f"--{k}", default=DEFAULTS[k])
    c.add_argument("--snapshot", required=True)
    c.add_argument("--n_boot", type=int, default=2000)
    c.add_argument("--seed", type=int, default=289)
    c.add_argument("--min_served_agreement", type=float, default=0.98)
    c.add_argument("--force", action="store_true",
                   help="Score even if the served-top-10 reproduction check fails.")

    m = sub.add_parser("manifest")
    m.add_argument("--run_dirs", nargs="+", required=True)
    m.add_argument("--expected_meta", nargs="*", default=[])
    m.add_argument("--out_json", required=True)

    args = p.parse_args()
    {"select_qwen8b": cmd_select_qwen8b, "score": cmd_score, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()
