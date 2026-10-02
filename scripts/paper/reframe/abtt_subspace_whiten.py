#!/usr/bin/env python3
"""Reframe experiments E3, H1 and WHITEN (issue #232) on cached pooled vectors.

Three questions about the post-hoc corrections of Section 3/5, all CPU work on the
mean-pooled hidden states the paper already scores (``hidden_mean_tokempty``):

  e3      Subspace split. At every layer, Task A AUROC of the pooled vectors projected
          onto PC1 alone, onto PCs 2..D, onto the removed subspace (PCs 1..D) and onto
          the retained subspace (the ABTT output), D = the train-selected ABTT D of that
          layer, plus two dimension-matched controls inside the retained subspace: the
          next D components (PCs D+1..2D) and random D-dim projections (5 seeds). Pre-trained panel (100 model-layers) and the three fine-tuned encoders
          (LaTa, Qwen3-0.6B, KaLM-mini; 64 model-layers).
  h1      ABTT D ablation. D in {0,1,2,3,5,7,10,15,20,30,50} at all 100 pre-trained
          model-layers, plus the uncorrected vectors ("raw"); D=0 is centering alone.
          Test AUROC, train AUROC, train DirAcc@1 and the Task B test metrics per D.
  whiten  PCA whitening to k in {64,128,256} components, plus full rank (the paper's
          excluded "whitening" row), fit on train, at all 100 pre-trained model-layers.
  pc1     H1 robustness at the 36 T5 layers: train- vs test-fitted PC1 (|cos|), AUROC
          after removing 1-3 test-fitted PCs (an oracle no reported cell uses), and the
          share of remaining test variance on the next direction after train D=1.
  render  Tables, figure and a facts file from the three CSVs above, plus the
          reproduction check against runs/active/resubmit/results/phase_resubmit_results.csv.

Protocol (identical to the paper's evaluator, scripts/resubmit/run_resubmit_evaluate.py):
  * rows are aligned to the split by filename (AlignmentResolver), never by position;
  * the train mean, the principal components, the whitening transform and D are fit
    on the 847 TRAIN passages only, then applied to train and test;
  * Task A AUROC is sklearn roc_auc_score over the upper triangle of the test cosine
    matrix (858 passages), same- vs different-directory pairs; every metric block goes
    through run_resubmit_evaluate.evaluate_from_similarity, the single definition of
    the reported metrics;
  * train-selected D = first argmax of train DirAcc@1 over {1,2,3,5,7,10} (the grid of
    find_optimal_D_phase11); train-selected layer = first argmax of train AUROC.

Two implementation notes that change numbers if forgotten:
  * EmbeddingCleaner(num_components=0) returns the input UNcentered (sif_abtt.py
    remove_top_components short-circuits), so D=0 here centers explicitly.
  * sklearn PCA(n_components=k) with svd_solver="auto" picks the randomized solver for
    k < 0.8 * min(n, d), which is seed-dependent; whitening here pins svd_solver="full".
    PCA(whiten=True) with no n_components keeps min(n, d) = 847 components although the
    centered train matrix has rank <= 846, so the last one is divided by ~0.

Outputs (small CSVs, force-added; embeddings are never written):
  runs/active/reframe/h1/h1_d_ablation.csv
  runs/active/reframe/e3/e3_subspace_split.csv
  runs/active/reframe/whiten/whiten_reduced.csv
  runs/active/reframe/h1/h1_pc1_robustness.csv
  runs/active/reframe/facts_e3_h1_whiten.md            (render)
  overleaf_drafts/tables/e3_subspace_split.tex          (tab:e3_subspace_split)
  overleaf_drafts/tables/d_ablation.tex                 (tab:d_ablation)
  overleaf_drafts/tables/whiten_reduced.tex             (tab:whiten_reduced)
  overleaf_drafts/figures/fig_d_ablation.pdf            (fig:d_ablation)

Run from the repo root (the embedding caches are gitignored; point --bases_root and
--ft_bases_root at a checkout that has them):
  python scripts/paper/reframe/abtt_subspace_whiten.py h1 --workers 16 \
      --bases_root /u/irowerojas/localLatin/runs/active/resubmit_bases
  python scripts/paper/reframe/abtt_subspace_whiten.py render
Python 3.10, numpy / pandas / scikit-learn / matplotlib.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "resubmit"))

from canon_retrieval import (  # noqa: E402
    l2_normalize,
    similarity_matrix,
    upper_triangle,
    upper_triangle_labels,
)
from embedding_alignment import AlignmentResolver  # noqa: E402
from pair_evaluation import safe_auc_roc  # noqa: E402
from sif_abtt import EmbeddingCleaner  # noqa: E402
import run_resubmit_evaluate as paper_eval  # noqa: E402

SPLIT_CSV = Path("runs/active/resubmit/data/phase_resubmit_split.csv")
RES_CSV = Path("runs/active/resubmit/results/phase_resubmit_results.csv")
BASES_ROOT = Path("runs/active/resubmit_bases")
FT_BASES_ROOT = Path("runs/active/resubmit_finetune_bases")
OUT_ROOT = Path("runs/active/reframe")
TAB_DIR = Path("overleaf_drafts/tables")
FIG_DIR = Path("overleaf_drafts/figures")
SUBDIR = "hidden_mean_tokempty"
HEADER = "% generated table"

H1_CSV = OUT_ROOT / "h1" / "h1_d_ablation.csv"
E3_CSV = OUT_ROOT / "e3" / "e3_subspace_split.csv"
WH_CSV = OUT_ROOT / "whiten" / "whiten_reduced.csv"
PC1_CSV = OUT_ROOT / "h1" / "h1_pc1_robustness.csv"
FACTS_MD = OUT_ROOT / "facts_e3_h1_whiten.md"

H1_D = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
SEL_GRID = [1, 2, 3, 5, 7, 10]  # the paper's abtt_optimal grid
WH_K = [64, 128, 256]
COLLAPSE_AUROC = 0.70  # paper Sec. 4: collapsed = baseline test AUROC below 0.70
PC1_FLAG = 0.60  # paper Sec. 4: top-PC share >= 0.6 flags all collapsed layers
RANK1_BAR = 0.80  # H1: D=1 must recover >= 80% of the D=10 gain

MODELS = [  # id, display, is_t5, colour (Okabe-Ito), marker; same as geometry_vs_retrieval.py
    ("bowphs/LaTa", "LaTa", True, "#0072B2", "o"),
    ("bowphs/PhilTa", "PhilTa", True, "#E69F00", "s"),
    ("google/mt5-base", "mT5-base", True, "#009E73", "^"),
    ("sentence-transformers/LaBSE", "LaBSE", False, "#CC79A7", "D"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B", False, "#D55E00", "v"),
    ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", "KaLM-mini", False,
     "#56B4E9", "P"),
]
FT_MODELS = [  # fine-tuned id (as in finetune_*_layer_results.csv), display, pre-trained id
    ("bowphs/LaTa-ft", "LaTa (fine-tuned)", "bowphs/LaTa"),
    ("Qwen/Qwen3-Embedding-0.6B-ft", "Qwen3-0.6B (fine-tuned)", "Qwen/Qwen3-Embedding-0.6B"),
    ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5-ft", "KaLM-mini (fine-tuned)",
     "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"),
]
DISP = {m[0]: m[1] for m in MODELS} | {m[0]: m[1] for m in FT_MODELS}
ORDER = [m[1] for m in MODELS]
T5 = [m[1] for m in MODELS if m[2]]


def slug(model_id: str) -> str:
    return model_id.replace("/", "_")


# --------------------------------------------------------------------------- #
# Pure transforms (unit-tested on synthetic data)
# --------------------------------------------------------------------------- #

def center(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """D=0: subtract the train mean only."""
    mu = train.mean(axis=0)
    return train - mu, test - mu


def abtt(train: np.ndarray, test: np.ndarray, D: int) -> Tuple[np.ndarray, np.ndarray]:
    """ABTT with D removed components, fit on train; D=0 is centering (see module doc)."""
    if D == 0:
        return center(train, test)
    cleaner = EmbeddingCleaner(num_components=D, center=True).fit(train)
    return cleaner.transform(train), cleaner.transform(test)


def pc_scores(train: np.ndarray, test: np.ndarray, D: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates of the centered vectors on the top-D train PCs (same PCs as ABTT).

    Returns (train_scores [n_tr, D], test_scores [n_te, D], train eigenvalue shares of
    all components). The PCs are EmbeddingCleaner's, so the removed subspace here is
    exactly the one ABTT subtracts: centered = scores @ pcs + retained.
    """
    cleaner = EmbeddingCleaner(num_components=D, center=True).fit(train)
    mu, pcs = cleaner.mean_vec, cleaner.pcs
    tr_c, te_c = train - mu, test - mu
    s = np.linalg.svd(tr_c, compute_uv=False)
    ev = s ** 2
    return tr_c @ pcs.T, te_c @ pcs.T, ev / ev.sum()


def whiten(train: np.ndarray, test: np.ndarray, k: Optional[int]
           ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """PCA whitening fit on train. k=None is the paper's full-rank row (PCA(whiten=True)).

    svd_solver="full" keeps reduced k deterministic; for k=None it is what "auto" picks
    anyway, so the full-rank row reproduces the published "whitening" cells.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=k, whiten=True, svd_solver="full").fit(train)
    ev = pca.explained_variance_
    info = {"n_components": int(len(ev)), "eig_first": float(ev[0]), "eig_last": float(ev[-1]),
            "cond": float(ev[0] / ev[-1]) if ev[-1] > 0 else float("inf")}
    return pca.transform(train), pca.transform(test), info


RAND_SEEDS = (0, 1, 2, 3, 4)


def random_basis(d: int, D: int, seed: int) -> np.ndarray:
    """A random orthonormal d x D basis (QR of a seeded Gaussian matrix)."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, D)))
    return q


def pair_auroc(emb: np.ndarray, labels_ut: np.ndarray) -> float:
    """Task A AUROC exactly as evaluate_from_similarity: cosine, upper triangle."""
    return safe_auc_roc(upper_triangle(similarity_matrix(l2_normalize(emb))), labels_ut)


def absdiff_auroc(scores_1d: np.ndarray, labels_ut: np.ndarray) -> float:
    """AUROC of -|s_i - s_j| for a 1-D score (cosine in 1-D is only a sign match)."""
    s = scores_1d.reshape(-1)
    iu = np.triu_indices(len(s), k=1)
    return safe_auc_roc(-np.abs(s[iu[0]] - s[iu[1]]), labels_ut)


def rank1_fractions(raw: float, d0: float, d1: float, d10: float) -> Tuple[float, float]:
    """Share of the D=10 AUROC gain over the raw vectors recovered by D=1 and by D=0."""
    gain = d10 - raw
    if not np.isfinite(gain) or abs(gain) < 1e-12:
        return float("nan"), float("nan")
    return (d1 - raw) / gain, (d0 - raw) / gain


def select_D(rows: pd.DataFrame, grid: Sequence[int] = SEL_GRID) -> int:
    """First argmax of train DirAcc@1 over the grid, as find_optimal_D_phase11."""
    best, best_score = grid[0], -1.0
    for D in grid:
        score = float(rows.loc[rows["D"] == D, "train_dir_acc_at_1"].iloc[0])
        if score > best_score:
            best, best_score = D, score
    return best


# --------------------------------------------------------------------------- #
# Worker plumbing
# --------------------------------------------------------------------------- #

_CTX: Dict[str, object] = {}


def _init(split_csv: str) -> None:
    split = pd.read_csv(split_csv)
    tr = split["split"].values == "train"
    te = split["split"].values == "test"
    _CTX.update(
        split=split, resolver=AlignmentResolver(split), tr=tr, te=te,
        tr_ids=split.loc[tr, "folder_id"].values, te_ids=split.loc[te, "folder_id"].values,
        te_partner=split.loc[te, "has_test_partner"].values.astype(bool),
        lab_te=upper_triangle_labels(split.loc[te, "folder_id"].values),
        lab_tr=upper_triangle_labels(split.loc[tr, "folder_id"].values),
    )


def _load(bases_root: str, model_slug: str, layer: int) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(bases_root) / "phase9_bases" / model_slug / SUBDIR / f"hidden_layer{layer}_embeddings.npy"
    emb = _CTX["resolver"].load(path)
    return emb[_CTX["tr"]], emb[_CTX["te"]]


def _metrics(train: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    """The paper's full metric block (Task A + Task B, tau learned on train)."""
    return paper_eval.evaluate_from_similarity(
        train_sim=similarity_matrix(l2_normalize(train)),
        test_sim=similarity_matrix(l2_normalize(test)),
        train_folder_ids=_CTX["tr_ids"], test_folder_ids=_CTX["te_ids"],
        test_has_partner=_CTX["te_partner"])


KEEP = ["aucroc", "train_aucroc", "train_dir_acc_at_1", "dir_acc_at_1",
        "overall_assignment_acc", "tau", "gap"]


def discover(bases_root: Path, model_id: str) -> List[int]:
    return paper_eval.discover_layers(bases_root, slug(model_id), "hidden", "mean",
                                      subdir_override=SUBDIR)


def task_h1(args) -> List[Dict]:
    bases_root, model_id, layer = args
    t0 = time.time()
    tr, te = _load(bases_root, slug(model_id), layer)
    rows = []
    m = _metrics(tr, te)
    rows.append({"model": model_id, "layer": layer, "variant": "raw", "D": -1,
                 **{k: m[k] for k in KEEP}})
    for D in H1_D:
        a, b = abtt(tr, te, D)
        m = _metrics(a, b)
        rows.append({"model": model_id, "layer": layer, "variant": "center" if D == 0 else "abtt",
                     "D": D, **{k: m[k] for k in KEEP}})
    print(f"  h1 {DISP.get(model_id, model_id)} L{layer}: raw {rows[0]['aucroc']:.3f} "
          f"D1 {rows[2]['aucroc']:.3f} D10 {rows[7]['aucroc']:.3f} ({time.time() - t0:.1f}s)",
          flush=True)
    return rows


def task_e3(args) -> List[Dict]:
    bases_root, model_id, layer = args
    t0 = time.time()
    tr, te = _load(bases_root, slug(model_id), layer)
    lab_te, lab_tr = _CTX["lab_te"], _CTX["lab_tr"]

    # train-selected D on the paper's grid, via the paper's own selector
    D_sel, _ = paper_eval.find_optimal_D_phase11(tr, _CTX["tr_ids"], SEL_GRID)
    out = []
    for D, tag in ((D_sel, "selected"), (10, "fixed10")):
        if tag == "fixed10" and D_sel == 10:
            continue
        s_tr, s_te, shares = pc_scores(tr, te, D)
        ret_tr, ret_te = abtt(tr, te, D)
        c_tr, c_te = center(tr, te)
        row = {"model": model_id, "layer": layer, "D_rule": tag, "D": D, "D_selected": D_sel,
               "removed_var_share_train": float(shares[:D].sum()),
               "pc1_share_train": float(shares[0])}
        variants = {
            "raw": (tr, te),
            "centered": (c_tr, c_te),
            "pc1": (s_tr[:, :1], s_te[:, :1]),
            "pc2_D": (s_tr[:, 1:], s_te[:, 1:]) if D >= 2 else None,
            "removed": (s_tr, s_te),
            "retained": (ret_tr, ret_te),
        }
        for name, pair in variants.items():
            if pair is None:
                row[f"auc_{name}"] = row[f"train_auc_{name}"] = float("nan")
                continue
            row[f"auc_{name}"] = pair_auroc(pair[1], lab_te)
            row[f"train_auc_{name}"] = pair_auroc(pair[0], lab_tr)
        row["auc_pc1_absdiff"] = absdiff_auroc(s_te[:, 0], lab_te)
        row["train_auc_pc1_absdiff"] = absdiff_auroc(s_tr[:, 0], lab_tr)
        # Dimension-matched controls: the next D components (PCs D+1..2D), and random
        # D-dimensional projections of the retained vectors (mean over RAND_SEEDS seeds).
        n_tr, n_te, _ = pc_scores(tr, te, 2 * D)
        row["auc_next_D"] = pair_auroc(n_te[:, D:], lab_te)
        row["train_auc_next_D"] = pair_auroc(n_tr[:, D:], lab_tr)
        row["auc_rand_retained"] = float(np.mean(
            [pair_auroc(ret_te @ random_basis(tr.shape[1], D, s), lab_te) for s in RAND_SEEDS]))
        out.append(row)
    print(f"  e3 {DISP.get(model_id, model_id)} L{layer}: D={D_sel} raw {out[0]['auc_raw']:.3f} "
          f"pc1 {out[0]['auc_pc1']:.3f} removed {out[0]['auc_removed']:.3f} "
          f"retained {out[0]['auc_retained']:.3f} ({time.time() - t0:.1f}s)", flush=True)
    return out


def _pcs(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt, s ** 2


def task_pc1(args) -> List[Dict]:
    """Is the D=1 failure an artefact of estimating PC1 on train? (H1 robustness)

    Compares the train-fitted PC1 with one fitted on the test passages themselves
    (an oracle that no reported cell uses), and measures how much of the remaining
    test variance the next direction holds after the train-fitted D=1 removal.
    """
    bases_root, model_id, layer = args
    tr, te = _load(bases_root, slug(model_id), layer)
    lab_te = _CTX["lab_te"]
    mu, vt, _ = _pcs(tr)
    mut, vtt, _ = _pcs(te)
    z, zt = te - mu, te - mut
    row = {"model": model_id, "layer": layer,
           "abs_cos_pc1_train_test": float(abs(vt[0] @ vtt[0]))}
    for D in (1, 2, 3):
        row[f"auc_trainfit_D{D}"] = pair_auroc(z - (z @ vt[:D].T) @ vt[:D], lab_te)
        row[f"auc_testfit_D{D}"] = pair_auroc(zt - (zt @ vtt[:D].T) @ vtt[:D], lab_te)
    r = z - (z @ vt[:1].T) @ vt[:1]
    s = np.linalg.svd(r - r.mean(axis=0), compute_uv=False) ** 2
    row["test_top_share_after_train_D1"] = float(s[0] / s.sum())
    print(f"  pc1 {DISP.get(model_id, model_id)} L{layer}: |cos| {row['abs_cos_pc1_train_test']:.4f} "
          f"train D1 {row['auc_trainfit_D1']:.3f} test-fit D1 {row['auc_testfit_D1']:.3f} "
          f"next share {row['test_top_share_after_train_D1']:.3f}", flush=True)
    return [row]


def task_whiten(args) -> List[Dict]:
    bases_root, model_id, layer = args
    t0 = time.time()
    tr, te = _load(bases_root, slug(model_id), layer)
    rows = []
    for k in WH_K + [None]:
        a, b, info = whiten(tr, te, k)
        m = _metrics(a, b)
        rows.append({"model": model_id, "layer": layer, "k": "full" if k is None else str(k),
                     **info, **{c: m[c] for c in KEEP}})
    print(f"  whiten {DISP.get(model_id, model_id)} L{layer}: "
          + " ".join(f"k={r['k']} {r['aucroc']:.3f}" for r in rows)
          + f" ({time.time() - t0:.1f}s)", flush=True)
    return rows


def run_pool(fn, tasks, split_csv: Path, workers: int) -> List[Dict]:
    rows: List[Dict] = []
    if workers <= 1:
        _init(str(split_csv))
        for t in tasks:
            rows.extend(fn(t))
        return rows
    with Pool(workers, initializer=_init, initargs=(str(split_csv),)) as pool:
        for part in pool.imap(fn, tasks, chunksize=1):
            rows.extend(part)
    return rows


def build_tasks(bases_root: Path, model_ids: Sequence[str], layers: Optional[List[int]]):
    tasks = []
    for mid in model_ids:
        found = discover(bases_root, mid)
        if not found:
            raise SystemExit(f"no layers for {mid} under {bases_root}/phase9_bases/{slug(mid)}/{SUBDIR}")
        for layer in found:
            if layers is None or layer in layers:
                tasks.append((str(bases_root), mid, layer))
    return tasks


def pick_models(names: str, pool) -> List[str]:
    if not names:
        return [m[0] for m in pool]
    wanted = {n.strip() for n in names.split(",") if n.strip()}
    return [m[0] for m in pool if m[0] in wanted or m[1] in wanted]


def cmd_compute(args) -> None:
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    t0 = time.time()
    if args.cmd == "h1":
        tasks = build_tasks(args.bases_root, pick_models(args.models, MODELS), layers)
        rows = run_pool(task_h1, tasks, args.split_csv, args.workers)
        out = args.out or H1_CSV
    elif args.cmd == "whiten":
        tasks = build_tasks(args.bases_root, pick_models(args.models, MODELS), layers)
        rows = run_pool(task_whiten, tasks, args.split_csv, args.workers)
        out = args.out or WH_CSV
    elif args.cmd == "pc1":
        t5 = [m[0] for m in MODELS if m[2]]
        tasks = build_tasks(args.bases_root, [m for m in pick_models(args.models, MODELS) if m in t5],
                            layers)
        rows = run_pool(task_pc1, tasks, args.split_csv, args.workers)
        out = args.out or PC1_CSV
    else:
        tasks = build_tasks(args.bases_root, pick_models(args.models, MODELS), layers)
        tasks += build_tasks(args.ft_bases_root, pick_models(args.models, FT_MODELS), layers)
        rows = run_pool(task_e3, tasks, args.split_csv, args.workers)
        out = args.out or E3_CSV
    df = pd.DataFrame(rows)
    order = {m[0]: i for i, m in enumerate(MODELS + [(f[0],) for f in FT_MODELS])}
    df = df.sort_values(by=["model", "layer"], key=lambda s: s.map(order) if s.name == "model" else s,
                        kind="stable").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.10g")
    print(f"wrote {out} ({len(df)} rows, {len(tasks)} model-layers) in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Render: tables, figure, facts, reproduction check
# --------------------------------------------------------------------------- #

def f3(x) -> str:
    return "--" if not np.isfinite(x) else f"{x:.3f}"


def first_argmax_layer(sub: pd.DataFrame, col: str) -> int:
    sub = sub.sort_values("layer")
    return int(sub.loc[sub[col].idxmax(), "layer"])


def load_results(res_csv: Path) -> pd.DataFrame:
    r = pd.read_csv(res_csv)
    return r[r["repr"] == "hidden"] if "repr" in r.columns else r


def reproduction(h1: pd.DataFrame, wh: pd.DataFrame, e3: pd.DataFrame, res: pd.DataFrame
                 ) -> pd.DataFrame:
    """Published cell vs recomputed cell, for every model-layer we recompute."""
    pub = res.set_index(["model", "layer", "method"])
    rows = []
    for (mid, layer), g in h1.groupby(["model", "layer"], sort=False):
        D_sel = select_D(g)
        pairs = [("baseline", g[g.variant == "raw"].iloc[0]),
                 ("abtt_fixed", g[(g.variant == "abtt") & (g.D == 10)].iloc[0]),
                 ("abtt_optimal", g[(g.variant == "abtt") & (g.D == D_sel)].iloc[0])]
        w = wh[(wh.model == mid) & (wh.layer == layer) & (wh.k == "full")]
        if len(w):
            pairs.append(("whitening", w.iloc[0]))
        for method, mine in pairs:
            key = (mid, layer, method)
            if key not in pub.index:
                continue
            p = pub.loc[key]
            rows.append({"model": mid, "layer": layer, "method": method,
                         "published_auroc": float(p["aucroc"]), "ours_auroc": float(mine["aucroc"]),
                         "published_D": int(p["D"]),
                         "ours_D": {"abtt_optimal": D_sel, "abtt_fixed": 10}.get(method, -1),
                         "published_train_dir1": float(p["train_dir_acc_at_1"]),
                         "ours_train_dir1": float(mine["train_dir_acc_at_1"])})
    e3s = e3[(e3.D_rule == "selected") & e3.model.isin([m[0] for m in MODELS])]
    for _, x in e3s.iterrows():
        key = (x.model, x.layer, "abtt_optimal")
        if key in pub.index:
            rows.append({"model": x.model, "layer": x.layer, "method": "e3_retained_vs_abtt_optimal",
                         "published_auroc": float(pub.loc[key, "aucroc"]),
                         "ours_auroc": float(x.auc_retained), "published_D": int(pub.loc[key, "D"]),
                         "ours_D": int(x.D), "published_train_dir1": float("nan"),
                         "ours_train_dir1": float("nan")})
    return pd.DataFrame(rows)


def ft_reproduction(e3: pd.DataFrame, ft_csvs: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for path in ft_csvs:
        if not path.exists():
            continue
        f = pd.read_csv(path).set_index(["model", "layer", "method"])
        for _, x in e3[e3.D_rule == "selected"].iterrows():
            for method, col in (("baseline", "auc_raw"), ("abtt_optimal", "auc_retained")):
                key = (x.model, x.layer, method)
                if key in f.index:
                    rows.append({"model": x.model, "layer": x.layer, "method": method,
                                 "published_auroc": float(f.loc[key, "aucroc"]),
                                 "ours_auroc": float(x[col]), "published_D": int(f.loc[key, "D"]),
                                 "ours_D": int(x.D) if method != "baseline" else -1})
    return pd.DataFrame(rows)


def h1_wide(h1: pd.DataFrame) -> pd.DataFrame:
    """One row per model-layer: auc_raw, auc_D0..auc_D50, dir_D*, plus derived columns."""
    rows = []
    for (mid, layer), g in h1.groupby(["model", "layer"], sort=False):
        r = {"model": mid, "m": DISP[mid], "layer": int(layer)}
        raw = g[g.variant == "raw"].iloc[0]
        r["auc_raw"], r["dir_raw"] = raw.aucroc, raw.train_dir_acc_at_1
        for D in H1_D:
            x = g[g.D == D].iloc[0]
            r[f"auc_D{D}"], r[f"dir_D{D}"] = x.aucroc, x.train_dir_acc_at_1
            r[f"trauc_D{D}"] = x.train_aucroc
        r["D_sel"] = select_D(g)
        r["D_sel_wide"] = select_D(g, H1_D[1:])
        r["frac_D1"], r["frac_D0"] = rank1_fractions(r["auc_raw"], r["auc_D0"], r["auc_D1"],
                                                     r["auc_D10"])
        rows.append(r)
    w = pd.DataFrame(rows)
    w["is_t5"] = w["m"].isin(T5)
    w["collapsed"] = w["auc_raw"] < COLLAPSE_AUROC
    return w


def attach_pc1(w: pd.DataFrame, geom_csv: Path) -> pd.DataFrame:
    if not geom_csv.exists():
        w["pc1_tr"] = np.nan
        return w
    g = pd.read_csv(geom_csv)
    g = g[(g.split == "train") & (g.view == "raw")][["model", "layer", "pc1_variance_ratio"]]
    w = w.merge(g.rename(columns={"pc1_variance_ratio": "pc1_tr"}), on=["model", "layer"],
                how="left", validate="1:1")
    return w


def write_d_table(w: pd.DataFrame, path: Path) -> None:
    Ds = H1_D
    cols = "l" + "r" + "r" * (1 + len(Ds))
    head = (r"\textbf{Layers} & $n$ & raw & " + " & ".join(f"{D}" for D in Ds) + r" \\")
    lines = [HEADER, r"\begin{table*}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{3.1pt}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             rf"& & & \multicolumn{{{len(Ds)}}}{{c}}{{\textbf{{Removed components $D$}}}} \\",
             rf"\cmidrule(lr){{4-{3 + len(Ds)}}}", head, r"\midrule",
             rf"\multicolumn{{{3 + len(Ds)}}}{{l}}{{\emph{{Lowest-AUROC baseline layer of each model}}}} \\"]
    for m in ORDER:
        s = w[w.m == m]
        x = s.loc[s.auc_raw.idxmin()]
        lines.append(f"{m} ({int(x.layer)}) & 1 & {f3(x.auc_raw)} & "
                     + " & ".join(f3(x[f'auc_D{D}']) for D in Ds) + r" \\")
    lines += [r"\midrule",
              rf"\multicolumn{{{3 + len(Ds)}}}{{l}}{{\emph{{Median over model-layers}}}} \\"]
    groups = [("Collapsed T5", w[w.collapsed])]
    groups += [(f"\\quad {m}", w[w.collapsed & (w.m == m)]) for m in T5]
    groups += [("Other T5", w[w.is_t5 & ~w.collapsed]),
              ("Embedding-trained", w[~w.is_t5]),
              ("All", w)]
    for name, s in groups:
        lines.append(f"{name} & {len(s)} & {f3(s.auc_raw.median())} & "
                     + " & ".join(f3(s[f'auc_D{D}'].median()) for D in Ds) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Test ranking AUROC against the number $D$ of top principal components "
              r"that ABTT removes, fit on training embeddings only. Raw: mean-pooled vectors with no "
              r"correction; $D=0$: centering on the training mean alone. The top block reads each "
              r"model at its lowest-AUROC baseline layer (layer in parentheses); the bottom block "
              r"gives medians over the 26 collapsed T5 layers (baseline AUROC below 0.70), also "
              r"split by model, the "
              r"other 10 T5 layers, the 64 layers of the embedding-trained models, and all 100 "
              r"model-layers. The paper's selection grid for $D$ stops at 10.}",
              r"\label{tab:d_ablation}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def _at_d10(e3: pd.DataFrame, mid: str, layer: int) -> Dict:
    """The E3 row of (model, layer) with D=10: the selected row if D_sel=10, else fixed10."""
    s = e3[(e3.model == mid) & (e3.layer == layer) & (e3.D == 10)]
    return s.iloc[0].to_dict()


def e3_rows(e3: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Rows of the E3 table.

    Block 1: each pre-trained model at its train-selected Task A ABTT layer (argmax train
    AUROC of the retained view, which is the abtt_optimal output) and train-selected D.
    Block 2: the fine-tuning contrast. Each fine-tuned encoder at its own train-selected
    ABTT layer, next to the pre-trained model at the SAME layer, both at D=10, so the
    only difference between the two rows is the fine-tuning.
    """
    sel = e3[e3.D_rule == "selected"]
    block1 = []
    for mid, name, *_ in MODELS:
        s = sel[sel.model == mid]
        layer = first_argmax_layer(s.assign(t=s.train_auc_retained), "t")
        block1.append({"name": name, **s[s.layer == layer].iloc[0].to_dict()})
    block2 = []
    for mid, name, pre in FT_MODELS:
        s = sel[sel.model == mid]
        if s.empty:
            continue
        layer = first_argmax_layer(s.assign(t=s.train_auc_retained), "t")
        block2.append({"name": f"{DISP[pre]}, pretrained", **_at_d10(e3, pre, layer)})
        block2.append({"name": f"{DISP[pre]}, fine-tuned", **_at_d10(e3, mid, layer)})
    return block1, block2


def write_e3_table(block1: List[Dict], block2: List[Dict], coll: pd.DataFrame, path: Path) -> None:
    cols = [("auc_raw", "Raw"), ("auc_centered", "Cent."), ("auc_pc1", "PC1"),
            ("auc_pc2_D", "PCs 2--$D$"), ("auc_removed", "PCs 1--$D$"),
            ("auc_next_D", "Next $D$"), ("auc_rand_retained", r"Rand.\ $D$"),
            ("auc_retained", "All")]
    lines = [HEADER, r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
             r"& & & & \multicolumn{2}{c}{\textbf{Full vector}} & "
             r"\multicolumn{3}{c}{\textbf{Removed subspace}} & "
             r"\multicolumn{3}{c}{\textbf{Retained subspace}} \\",
             r"\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
             r"\textbf{Model} & $\ell$ & $D$ & Var. & " + " & ".join(c[1] for c in cols) + r" \\",
             r"\midrule"]

    def row(x, name=None):
        return (f"{name or x['name']} & {int(x['layer'])} & {int(x['D'])} & "
                f"{x['removed_var_share_train']:.2f} & "
                + " & ".join(f3(x[c]) for c, _ in cols) + r" \\")

    ncol = 4 + len(cols)
    lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{Pretrained, train-selected layer and $D$}}}} \\")
    lines += [row(x) for x in block1]
    lines.append(f"Collapsed T5, median ({len(coll)}) & -- & -- & "
                 f"{coll.removed_var_share_train.median():.2f} & "
                 + " & ".join(f3(coll[c].median()) for c, _ in cols) + r" \\")
    if block2:
        lines += [r"\midrule",
                  rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{Fine-tuning contrast, same layer, $D=10$}}}} \\"]
        lines += [row(x) for x in block2]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Test ranking AUROC of cosine within parts of the pooled vector. ABTT "
              r"removes the top $D$ principal components of the centered training embeddings; Var.\ "
              r"is their share of the centered training variance. Raw: uncorrected vector; Cent.: "
              r"centered on the training mean; PC1: projection onto the first component alone, "
              r"where cosine reduces to a sign match; PCs 2--$D$ and PCs 1--$D$: projections onto "
              r"the rest of the removed subspace and onto all of it. Retained subspace, a "
              r"dimension-matched control: Next $D$, PCs $D{+}1$ to $2D$; Rand.\ $D$, "
              r"random $D$-dimensional projections of the retained vectors (mean over five seeds); "
              r"All, the ABTT output. "
              r"Top block: each pretrained model at its train-selected ABTT layer $\ell$ with $D$ "
              r"chosen on training DirAcc@1, and the median over the 26 collapsed T5 layers "
              r"(baseline AUROC below 0.70), each at its own layer and $D$. Bottom block: each "
              r"contrastively fine-tuned encoder at its train-selected ABTT layer, next to its "
              r"pretrained model at the same layer, both with $D=10$. All components are fit on "
              r"training embeddings only.}",
              r"\label{tab:e3_subspace_split}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def whiten_rows(wh: pd.DataFrame, res: pd.DataFrame) -> List[Dict]:
    """Per model: each setting at its own train-selected layer, for Task A and Task B.

    Task A layer = first argmax of training AUROC; Task B layer = first argmax of
    training DirAcc@1 (the paper's rule for each task). Task B numbers are the
    evaluator's single-split test DirAcc@1 and assignment accuracy, not the five-seed
    Task B protocol of the headline table.
    """
    out = []
    for mid, name, *_ in MODELS:
        r = res[res.model == mid]
        row = {"name": name, "model": mid}
        for meth, tag in (("baseline", "base"), ("abtt_optimal", "abtt")):
            s = r[r.method == meth]
            layer = first_argmax_layer(s, "train_aucroc")
            row[f"{tag}_layer"], row[tag] = layer, float(s[s.layer == layer].aucroc.iloc[0])
            lb = first_argmax_layer(s, "train_dir_acc_at_1")
            x = s[s.layer == lb].iloc[0]
            row[f"{tag}_B_layer"] = lb
            row[f"{tag}_dir1"] = float(x.dir_acc_at_1)
            row[f"{tag}_assign"] = float(x.overall_assignment_acc)
        for k in [str(k) for k in WH_K] + ["full"]:
            s = wh[(wh.model == mid) & (wh.k == k)]
            layer = first_argmax_layer(s, "train_aucroc")
            x = s[s.layer == layer].iloc[0]
            row[f"w{k}_layer"], row[f"w{k}"] = layer, float(x.aucroc)
            row[f"w{k}_cond"] = float(x["cond"])
            # also at the ABTT-selected layer, to separate method from layer choice
            y = s[s.layer == row["abtt_layer"]].iloc[0]
            row[f"w{k}_at_abtt_layer"] = float(y.aucroc)
            lb = first_argmax_layer(s, "train_dir_acc_at_1")
            z = s[s.layer == lb].iloc[0]
            row[f"w{k}_B_layer"] = lb
            row[f"w{k}_dir1"] = float(z.dir_acc_at_1)
            row[f"w{k}_assign"] = float(z.overall_assignment_acc)
            row[f"w{k}_tau"] = float(z.tau)
        out.append(row)
    return out


def pct(x) -> str:
    return "--" if not np.isfinite(x) else f"{100 * x:.1f}"


def write_whiten_table(rows: List[Dict], path: Path) -> None:
    ks = [str(k) for k in WH_K] + ["full"]
    kh = " & ".join(ks)
    lines = [HEADER, r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{lrrrrrrrrrrrrrrrr}", r"\toprule",
             r"& \multicolumn{6}{c}{\textbf{Ranking AUROC}} & "
             r"\multicolumn{5}{c}{\textbf{Routing DirAcc@1}} & "
             r"\multicolumn{5}{c}{\textbf{Routing assignment}} \\",
             r"\cmidrule(lr){2-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}",
             r"& & & \multicolumn{4}{c}{whitening, $r$} & & \multicolumn{4}{c}{whitening, $r$} & "
             r"& \multicolumn{4}{c}{whitening, $r$} \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){9-12}\cmidrule(lr){14-17}",
             r"\textbf{Model} & Base & ABTT & " + kh + " & ABTT & " + kh + " & ABTT & " + kh + r" \\",
             r"\midrule"]
    for i, x in enumerate(rows):
        if i == 3:
            lines.append(r"\midrule")
        lines.append(f"{x['name']} & {f3(x['base'])} & {f3(x['abtt'])} & "
                     + " & ".join(f3(x[f'w{k}']) for k in ks) + " & "
                     + f"{pct(x['abtt_dir1'])} & " + " & ".join(pct(x[f'w{k}_dir1']) for k in ks)
                     + " & " + f"{pct(x['abtt_assign'])} & "
                     + " & ".join(pct(x[f'w{k}_assign']) for k in ks) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{PCA whitening to $r$ components against the baseline and ABTT. Whitening "
              r"keeps the top $r$ principal components of the centered training embeddings and "
              r"rescales each to unit variance. Full keeps every component: 768 for the T5 encoders "
              r"and LaBSE, and 847, the number of training passages, for Qwen3-0.6B and KaLM-mini, "
              r"whose last component then has numerically zero variance. Ranking cells are read at "
              r"each setting's train-selected layer (highest training AUROC), routing cells at the "
              r"layer with the highest training DirAcc@1. Routing values are single-split test "
              r"DirAcc@1 and assignment accuracy in percent, with the threshold learned on training "
              r"pairs. Every "
              r"transform is fit on training embeddings only. Whitening ranks on par with ABTT, but "
              r"routes below it.}",
              r"\label{tab:whiten_reduced}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "grid.linewidth": 0.4, "grid.color": "#dddddd", "font.family": "DejaVu Sans"})


def fig_d_ablation(w: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style()
    xs = ["raw"] + [str(D) for D in H1_D]
    xpos = np.arange(len(xs))
    fig, axes = plt.subplots(2, 1, figsize=(3.0, 3.55), sharex=True)
    handles = []
    for mid, name, t5, c, mk in MODELS:
        s = w[w.m == name]
        x = s.loc[s.auc_raw.idxmin()]
        for ax, pre in ((axes[0], "auc"), (axes[1], "dir")):
            ys = [x[f"{pre}_raw"]] + [x[f"{pre}_D{D}"] for D in H1_D]
            ax.plot(xpos, ys, color=c, lw=1.1, linestyle="-" if t5 else "--", marker=mk,
                    markersize=3.4, markerfacecolor=c if t5 else "white", markeredgecolor=c,
                    markeredgewidth=0.8, zorder=3)
        handles.append(Line2D([0], [0], color=c, lw=1.1, linestyle="-" if t5 else "--", marker=mk,
                              markersize=3.4, markerfacecolor=c if t5 else "white",
                              markeredgecolor=c, label=f"{name} ({int(x.layer)})"))
    axes[0].set_ylabel("Test AUROC")
    axes[1].set_ylabel("Train DirAcc@1")
    axes[0].set_ylim(0.45, 1.0)
    axes[0].axhline(0.5, color="#888888", lw=0.6, zorder=1)
    for ax in axes:
        ax.grid(True, zorder=0)
        ax.axvline(xpos[H1_D.index(10) + 1], color="#888888", lw=0.6, ls=":", zorder=1)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(xs)
    axes[1].set_xlabel("Removed components $D$")
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.54, 1.0), handletextpad=0.4, columnspacing=0.8,
               handlelength=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.83), h_pad=0.4)
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def fmt_frac(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100 * x:.0f}%"


def facts(h1: pd.DataFrame, w: pd.DataFrame, e3: pd.DataFrame, e3r: List[Dict],
          whr: List[Dict], wh: pd.DataFrame, rep: pd.DataFrame, ftrep: pd.DataFrame,
          path: Path, res: Optional[pd.DataFrame] = None,
          pc1: Optional[pd.DataFrame] = None) -> None:
    L: List[str] = []
    a = L.append
    a("# E3 / H1 / WHITEN numbers (generated)")
    a("")
    a("Generated by `scripts/paper/reframe/abtt_subspace_whiten.py render`. Test AUROC unless "
      "marked train. 3 dp.")
    a("")
    a("## Reproduction check against phase_resubmit_results.csv")
    for meth, g in rep.groupby("method", sort=False):
        d = (g.ours_auroc - g.published_auroc).abs()
        dmatch = (g.ours_D == g.published_D).mean() if meth.startswith("abtt") or meth.startswith("e3") else float("nan")
        a(f"- {meth}: {len(g)} cells, max |diff| {d.max():.2e}, cells equal to 3 dp "
          f"{int((g.ours_auroc.round(3) == g.published_auroc.round(3)).sum())}/{len(g)}"
          + (f", D matches {dmatch * 100:.0f}%" if np.isfinite(dmatch) else ""))
    if len(ftrep):
        d = (ftrep.ours_auroc - ftrep.published_auroc).abs()
        a(f"- fine-tuned (finetune_*_layer_results.csv, baseline + abtt_optimal): {len(ftrep)} "
          f"cells, max |diff| {d.max():.2e}, D matches "
          f"{(ftrep[ftrep.method == 'abtt_optimal'].ours_D == ftrep[ftrep.method == 'abtt_optimal'].published_D).mean() * 100:.0f}%")
    a("- sample cells (model, layer, method: published / ours):")
    for mid, layer, meth in [("bowphs/LaTa", 7, "baseline"), ("bowphs/LaTa", 7, "abtt_optimal"),
                             ("bowphs/LaTa", 12, "baseline"), ("bowphs/LaTa", 12, "abtt_optimal"),
                             ("bowphs/PhilTa", 9, "abtt_optimal"), ("google/mt5-base", 6, "baseline"),
                             ("sentence-transformers/LaBSE", 11, "abtt_optimal"),
                             ("Qwen/Qwen3-Embedding-0.6B", 26, "baseline"),
                             ("bowphs/LaTa", 7, "whitening")]:
        g = rep[(rep.model == mid) & (rep.layer == layer) & (rep.method == meth)]
        if len(g):
            x = g.iloc[0]
            a(f"  - {DISP[mid]} L{layer} {meth}: {x.published_auroc:.3f} / {x.ours_auroc:.3f}")
    a("")
    a("## H1: D ablation")
    a(f"- model-layers: {len(w)} (" + ", ".join(f"{m} {int((w.m == m).sum())}" for m in ORDER) + ")")
    a(f"- collapsed (baseline AUROC < {COLLAPSE_AUROC}): {int(w.collapsed.sum())} layers, all T5: "
      f"{bool(w[w.collapsed].is_t5.all())}")
    if w.pc1_tr.notna().any():
        flag = w.pc1_tr >= PC1_FLAG
        a(f"- top-PC share >= {PC1_FLAG} flags {int(flag.sum())} layers; flagged-not-collapsed: "
          + ", ".join(f"{x.m} {int(x.layer)}" for _, x in w[flag & ~w.collapsed].iterrows()))
    for label, s in [("collapsed T5 (AUROC<0.70)", w[w.collapsed]),
                     ("flagged by top-PC share>=0.6", w[w.pc1_tr >= PC1_FLAG] if w.pc1_tr.notna().any() else w.iloc[:0])]:
        if s.empty:
            continue
        a(f"### Rank-1 test over {label}: n={len(s)}")
        a(f"- D=1 share of D=10 gain: median {fmt_frac(s.frac_D1.median())}, min "
          f"{fmt_frac(s.frac_D1.min())}, max {fmt_frac(s.frac_D1.max())}; layers with >= 80%: "
          f"{int((s.frac_D1 >= RANK1_BAR).sum())}/{len(s)}")
        a(f"- D=0 (centering) share: median {fmt_frac(s.frac_D0.median())}, min "
          f"{fmt_frac(s.frac_D0.min())}, max {fmt_frac(s.frac_D0.max())}")
        a(f"- AUROC medians: raw {f3(s.auc_raw.median())}, D0 {f3(s.auc_D0.median())}, D1 "
          f"{f3(s.auc_D1.median())}, D2 {f3(s.auc_D2.median())}, D3 {f3(s.auc_D3.median())}, D10 "
          f"{f3(s.auc_D10.median())}, D50 {f3(s.auc_D50.median())}")
        gain = s.auc_D10 - s.auc_raw
        for D in (2, 3, 5):
            f = (s[f"auc_D{D}"] - s.auc_raw) / gain
            a(f"- D={D} share of D=10 gain: median {fmt_frac(f.median())}, min {fmt_frac(f.min())}; "
              f"layers with >= 80%: {int((f >= RANK1_BAR).sum())}/{len(s)}")
        for m in T5:
            t = s[s.m == m]
            if t.empty:
                continue
            a(f"  - {m} ({len(t)} layers {t.layer.min()}-{t.layer.max()}): D=1 share median "
              f"{fmt_frac(t.frac_D1.median())} (range {fmt_frac(t.frac_D1.min())} to "
              f"{fmt_frac(t.frac_D1.max())}); D=0 share median {fmt_frac(t.frac_D0.median())}; "
              f"AUROC raw {f3(t.auc_raw.min())}-{f3(t.auc_raw.max())}, D1 {f3(t.auc_D1.min())}-"
              f"{f3(t.auc_D1.max())}, D10 {f3(t.auc_D10.min())}-{f3(t.auc_D10.max())}")
    a("")
    a("### Per collapsed layer (model layer: raw / D0 / D1 / D2 / D3 / D5 / D10 / D20 / D50; frac D1, frac D0)")
    for _, x in w[w.collapsed].iterrows():
        a(f"- {x.m} {int(x.layer)}: " + " / ".join(f3(x[c]) for c in
          ["auc_raw", "auc_D0", "auc_D1", "auc_D2", "auc_D3", "auc_D5", "auc_D10", "auc_D20", "auc_D50"])
          + f"; {fmt_frac(x.frac_D1)}, {fmt_frac(x.frac_D0)}")
    a("")
    a("### Where the AUROC curve peaks over D (argmax test AUROC over D>=1, for description only)")
    for m in ORDER:
        s = w[w.m == m]
        pk = s[[f"auc_D{D}" for D in H1_D[1:]]].to_numpy().argmax(axis=1)
        a(f"- {m}: " + ", ".join(f"L{int(x.layer)}:{H1_D[1:][p]}" for (_, x), p in zip(s.iterrows(), pk)))
    a("")
    a("### Train DirAcc@1 selection")
    a(f"- over the paper grid {SEL_GRID}: " + ", ".join(
        f"D={k}: {v}" for k, v in w.D_sel.value_counts().sort_index().items()))
    a(f"- over the extended grid {H1_D[1:]}: " + ", ".join(
        f"D={k}: {v}" for k, v in w.D_sel_wide.value_counts().sort_index().items()))
    for label, s in [("collapsed T5", w[w.collapsed]), ("other T5", w[w.is_t5 & ~w.collapsed]),
                     ("embedding-trained", w[~w.is_t5])]:
        a(f"  - {label}: extended-grid selection " + ", ".join(
            f"D={k}: {v}" for k, v in s.D_sel_wide.value_counts().sort_index().items()))
    a(f"- test AUROC at extended-grid D minus at paper-grid D: median "
      f"{np.median([x[f'auc_D{int(x.D_sel_wide)}'] - x[f'auc_D{int(x.D_sel)}'] for _, x in w.iterrows()]):+.4f}, "
      f"range {min(x[f'auc_D{int(x.D_sel_wide)}'] - x[f'auc_D{int(x.D_sel)}'] for _, x in w.iterrows()):+.4f} to "
      f"{max(x[f'auc_D{int(x.D_sel_wide)}'] - x[f'auc_D{int(x.D_sel)}'] for _, x in w.iterrows()):+.4f}")
    a("")
    a("### Non-T5 and healthy T5: D=1 and centering")
    for label, s in [("other T5", w[w.is_t5 & ~w.collapsed]), ("embedding-trained", w[~w.is_t5])]:
        a(f"- {label} (n={len(s)}): AUROC median raw {f3(s.auc_raw.median())}, D0 "
          f"{f3(s.auc_D0.median())}, D1 {f3(s.auc_D1.median())}, D10 {f3(s.auc_D10.median())}, "
          f"D50 {f3(s.auc_D50.median())}; D=10 minus raw range "
          f"{(s.auc_D10 - s.auc_raw).min():+.3f} to {(s.auc_D10 - s.auc_raw).max():+.3f}; D=50 minus "
          f"D=10 range {(s.auc_D50 - s.auc_D10).min():+.3f} to {(s.auc_D50 - s.auc_D10).max():+.3f}")
    a("")
    a("## E3: subspace split")
    a("Columns: layer, D (train-selected), removed variance share (train), AUROC raw / centered / PC1 "
      "(cosine=sign) / PC1 (-|diff|) / PCs 2..D / PCs 1..D (removed) / retained (ABTT).")
    for x in e3r:
        a(f"- {x['name']} L{int(x['layer'])}, D={int(x['D'])}: var {x['removed_var_share_train']:.3f}; "
          f"{f3(x['auc_raw'])} / {f3(x['auc_centered'])} / {f3(x['auc_pc1'])} / "
          f"{f3(x['auc_pc1_absdiff'])} / {f3(x['auc_pc2_D'])} / {f3(x['auc_removed'])} / "
          f"{f3(x['auc_retained'])}; next D (PCs D+1..2D) {f3(x['auc_next_D'])}, random D of "
          f"retained {f3(x['auc_rand_retained'])}")
    sel = e3[e3.D_rule == "selected"].copy()
    sel["m"] = sel.model.map(DISP)
    base = w.set_index(["model", "layer"])
    sel["collapsed"] = [base.loc[(r.model, r.layer), "collapsed"] if (r.model, r.layer) in base.index
                        else False for r in sel.itertuples()]
    a("")
    a("### Removed vs retained over all layers (pre-trained and fine-tuned)")
    for m in ORDER + [f[1] for f in FT_MODELS]:
        s = sel[sel.m == m]
        if s.empty:
            continue
        diff = s.auc_removed - s.auc_retained
        a(f"- {m} ({len(s)} layers): removed {f3(s.auc_removed.min())}-{f3(s.auc_removed.max())} "
          f"(median {f3(s.auc_removed.median())}), retained {f3(s.auc_retained.min())}-"
          f"{f3(s.auc_retained.max())}, removed minus retained median {diff.median():+.3f} "
          f"(max {diff.max():+.3f}); PCs 2..D median {f3(s.auc_pc2_D.median())}; PC1 median "
          f"{f3(s.auc_pc1.median())} (-|diff| {f3(s.auc_pc1_absdiff.median())})")
    c = sel[sel.collapsed]
    a(f"- collapsed T5 layers (n={len(c)}): removed median {f3(c.auc_removed.median())} (range "
      f"{f3(c.auc_removed.min())}-{f3(c.auc_removed.max())}), PC1 median {f3(c.auc_pc1.median())} "
      f"(-|diff| {f3(c.auc_pc1_absdiff.median())}), PCs 2..D median {f3(c.auc_pc2_D.median())} "
      f"(range {f3(c.auc_pc2_D.min())}-{f3(c.auc_pc2_D.max())}), retained median "
      f"{f3(c.auc_retained.median())}")
    a(f"- collapsed T5 layers: train PC1 variance share median {c.pc1_share_train.median():.3f} "
      f"(min {c.pc1_share_train.min():.3f}); PCs 2..D share median "
      f"{(c.removed_var_share_train - c.pc1_share_train).median():.3f}; selected D "
      + ", ".join(f"D={k}: {v}" for k, v in c.D.value_counts().sort_index().items()))
    pre = sel[sel.model.isin([m[0] for m in MODELS])]
    gap = pre.auc_retained - pre.auc_removed
    t5 = pre[pre.m.isin(T5)]
    a(f"- pre-trained, all 100 layers: retained minus removed min {gap.min():+.3f} "
      f"({pre.loc[gap.idxmin(), 'm']} L{int(pre.loc[gap.idxmin(), 'layer'])}); T5 only min "
      f"{(t5.auc_retained - t5.auc_removed).min():+.3f}; removed AUROC max in T5 {f3(t5.auc_removed.max())}")
    f10 = e3[e3.D_rule == "fixed10"]
    if len(f10):
        a(f"- {len(f10)} layers where the selected D is not 10 also carry a D=10 row in the CSV.")
    a("")
    a("## WHITEN")
    a("Each cell at its own train-selected layer (argmax train AUROC); [layer]. 'at ABTT layer' reads "
      "whitening at the ABTT-selected layer.")
    for x in whr:
        a(f"- {x['name']}: base {f3(x['base'])} [{x['base_layer']}], ABTT {f3(x['abtt'])} "
          f"[{x['abtt_layer']}], " + ", ".join(
              f"k={k} {f3(x[f'w{k}'])} [{x[f'w{k}_layer']}] (at ABTT layer {f3(x[f'w{k}_at_abtt_layer'])}, "
              f"cond {x[f'w{k}_cond']:.3g})" for k in [str(k) for k in WH_K] + ["full"]))
    a("")
    a("### Whitening over all 100 layers (test AUROC range; median)")
    for k in [str(k) for k in WH_K] + ["full"]:
        s = wh[wh.k == k]
        a(f"- k={k}: T5 {f3(s[s.model.map(DISP).isin(T5)].aucroc.min())}-"
          f"{f3(s[s.model.map(DISP).isin(T5)].aucroc.max())}; non-T5 "
          f"{f3(s[~s.model.map(DISP).isin(T5)].aucroc.min())}-"
          f"{f3(s[~s.model.map(DISP).isin(T5)].aucroc.max())}; median {f3(s.aucroc.median())}; "
          f"n_components {sorted(s.n_components.unique().tolist())}; smallest eigenvalue range "
          f"{s.eig_last.min():.2e}-{s.eig_last.max():.2e}")
    a("")
    a("### Whitening minus ABTT (train-selected D) at the same layer, all 100 layers")
    abtt_sel = {(x.model, x.layer): x[f"auc_D{int(x.D_sel)}"] for _, x in w.iterrows()}
    coll = set(zip(w[w.collapsed].model, w[w.collapsed].layer))
    for k in [str(k) for k in WH_K] + ["full"]:
        s = wh[wh.k == k]
        d = np.array([x.aucroc - abtt_sel[(x.model, x.layer)] for x in s.itertuples()])
        cs = s[[(x.model, x.layer) in coll for x in s.itertuples()]]
        a(f"- k={k}: median {np.median(d):+.4f}, range {d.min():+.3f} to {d.max():+.3f}, whitening "
          f"higher at {int((d > 0).sum())}/{len(d)}; collapsed T5 layers whitening AUROC "
          f"{f3(cs.aucroc.min())}-{f3(cs.aucroc.max())}")
    a("")
    L.extend(extra_facts(w, e3, wh, whr, res, pc1))
    path.write_text("\n".join(L) + "\n")


def extra_facts(w: pd.DataFrame, e3: pd.DataFrame, wh: pd.DataFrame, whr: List[Dict],
                res: Optional[pd.DataFrame], pc1: Optional[pd.DataFrame]) -> List[str]:
    """Numbers added in the review round: per-model H1, PC1 robustness, E3 controls, routing."""
    L: List[str] = []
    a = L.append
    a("## Review-round additions")
    a("### H1 per model at collapsed layers (median share of the D=10 gain; AUROC ranges)")
    c = w[w.collapsed]
    for m in T5:
        t = c[c.m == m]
        g = t.auc_D10 - t.auc_raw
        a(f"- {m} (n={len(t)}): " + ", ".join(
            f"D={D} {fmt_frac(((t[f'auc_D{D}'] - t.auc_raw) / g).median())} "
            f"(min {fmt_frac(((t[f'auc_D{D}'] - t.auc_raw) / g).min())})" for D in (1, 2, 3, 5, 7))
          + f"; AUROC D5 {f3(t.auc_D5.min())}-{f3(t.auc_D5.max())}, D7 {f3(t.auc_D7.min())}-"
          f"{f3(t.auc_D7.max())}, D10 {f3(t.auc_D10.min())}-{f3(t.auc_D10.max())}")
    if pc1 is not None:
        keys = set(zip(c.model, c.layer))
        pcc = pc1[[k in keys for k in zip(pc1.model, pc1.layer)]].copy()
        pcc["m"] = pcc.model.map(DISP)
        a(f"### PC1 robustness at the {len(pcc)} collapsed layers (h1_pc1_robustness.csv)")
        a(f"- |cos(train PC1, test PC1)|: {pcc.abs_cos_pc1_train_test.min():.4f}-"
          f"{pcc.abs_cos_pc1_train_test.max():.4f}")
        a(f"- D=1 AUROC, train-fit {f3(pcc.auc_trainfit_D1.min())}-{f3(pcc.auc_trainfit_D1.max())}; "
          f"test-fit (oracle) {f3(pcc.auc_testfit_D1.min())}-{f3(pcc.auc_testfit_D1.max())}; test-fit "
          f"minus train-fit {(pcc.auc_testfit_D1 - pcc.auc_trainfit_D1).min():+.3f} to "
          f"{(pcc.auc_testfit_D1 - pcc.auc_trainfit_D1).max():+.3f}")
        for m in T5:
            t = pcc[pcc.m == m]
            a(f"- {m}: after train D=1, the next direction holds {t.test_top_share_after_train_D1.min():.3f}-"
              f"{t.test_top_share_after_train_D1.max():.3f} of the remaining test variance")
        for m, layer in (("LaTa", 6), ("PhilTa", 10), ("mT5-base", 8)):
            t = pcc[(pcc.m == m) & (pcc.layer == layer)]
            if len(t):
                x = t.iloc[0]
                a(f"  - {m} L{layer}: |cos| {x.abs_cos_pc1_train_test:.4f}; D1 train-fit "
                  f"{f3(x.auc_trainfit_D1)} / test-fit {f3(x.auc_testfit_D1)}; D2 {f3(x.auc_trainfit_D2)} / "
                  f"{f3(x.auc_testfit_D2)}; D3 {f3(x.auc_trainfit_D3)} / {f3(x.auc_testfit_D3)}; next share "
                  f"{x.test_top_share_after_train_D1:.3f}")
    a("### E3 dimension-matched controls (selected D)")
    sel = e3[(e3.D_rule == "selected") & e3.model.isin([m[0] for m in MODELS])].copy()
    d = sel.auc_removed - sel.auc_next_D
    a(f"- pre-trained, all 100 layers: removed minus next-D median {d.median():+.3f}, removed below "
      f"next-D at {int((d < 0).sum())}/100")
    keys = set(zip(c.model, c.layer))
    nc = sel[[k not in keys for k in zip(sel.model, sel.layer)]]
    dn = nc.auc_removed - nc.auc_next_D
    a(f"- pre-trained, 74 non-collapsed layers: removed below next-D at {int((dn < 0).sum())}/{len(nc)}")
    if res is not None:
        a("### WHITEN on routing (single-split evaluator numbers, not the five-seed Task B)")
        for x in whr:
            a(f"- {x['name']}: Task B layer ABTT {x['abtt_B_layer']} DirAcc@1 {pct(x['abtt_dir1'])} "
              f"assign {pct(x['abtt_assign'])}; " + "; ".join(
                  f"k={k} [{x[f'w{k}_B_layer']}] {pct(x[f'w{k}_dir1'])} / {pct(x[f'w{k}_assign'])} "
                  f"(tau {x[f'w{k}_tau']:.3f})" for k in [str(k) for k in WH_K] + ["full"]))
        for k in [str(k) for k in WH_K]:
            # differences of the printed (1 dp) table cells, so prose matches the table
            dd = [float(pct(x[f"w{k}_dir1"])) - float(pct(x["abtt_dir1"])) for x in whr]
            da = [float(pct(x[f"w{k}_assign"])) - float(pct(x["abtt_assign"])) for x in whr]
            a(f"- k={k} minus ABTT at train-selected Task B layers (table cells): DirAcc@1 "
              f"{min(dd):+.1f} to {max(dd):+.1f} pts, assignment {min(da):+.1f} to {max(da):+.1f} pts")
        ab = res[res.method == "abtt_optimal"].set_index(["model", "layer"])
        for k in [str(k) for k in WH_K] + ["full"]:
            s = wh[wh.k == k]
            dd = np.array([100 * (x.dir_acc_at_1 - ab.loc[(x.model, x.layer), "dir_acc_at_1"])
                           for x in s.itertuples()])
            da = np.array([100 * (x.overall_assignment_acc - ab.loc[(x.model, x.layer), "overall_assignment_acc"])
                           for x in s.itertuples()])
            a(f"- k={k} minus ABTT, same layer, all 100: DirAcc@1 median {np.median(dd):+.1f} pts "
              f"(>= ABTT at {int((dd >= 0).sum())}/100), assignment median {np.median(da):+.1f} "
              f"(>= ABTT at {int((da >= 0).sum())}/100); tau range {s.tau.min():.3f}-{s.tau.max():.3f}")
    a("")
    return L


def cmd_render(args) -> None:
    h1 = pd.read_csv(args.h1_csv)
    e3 = pd.read_csv(args.e3_csv)
    wh = pd.read_csv(args.whiten_csv, dtype={"k": str})
    res = load_results(args.results_csv)
    w = attach_pc1(h1_wide(h1), args.geom_csv)
    pc1 = pd.read_csv(args.pc1_csv) if args.pc1_csv.exists() else None
    assert len(w) == 100, f"expected 100 model-layers, got {len(w)}"
    args.tab_dir.mkdir(parents=True, exist_ok=True)
    write_d_table(w, args.tab_dir / "d_ablation.tex")
    block1, block2 = e3_rows(e3)
    sel_ft = e3[(e3.D_rule == "selected") & e3.model.isin([f[0] for f in FT_MODELS])]
    ft_sel = []
    for mid, name, _ in FT_MODELS:
        s = sel_ft[sel_ft.model == mid]
        if len(s):
            layer = first_argmax_layer(s.assign(t=s.train_auc_retained), "t")
            ft_sel.append({"name": f"{name}, train-selected D", **s[s.layer == layer].iloc[0].to_dict()})
    e3r = block1 + block2 + ft_sel
    coll_keys = set(zip(w[w.collapsed].model, w[w.collapsed].layer))
    sel = e3[e3.D_rule == "selected"]
    coll = sel[[k in coll_keys for k in zip(sel.model, sel.layer)]]
    write_e3_table(block1, block2, coll, args.tab_dir / "e3_subspace_split.tex")
    whr = whiten_rows(wh, res)
    write_whiten_table(whr, args.tab_dir / "whiten_reduced.tex")
    if not args.no_figure:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        fig_d_ablation(w, args.fig_dir / "fig_d_ablation.pdf")
    if args.facts_md is not None:
        rep = reproduction(h1, wh, e3, res)
        ftrep = ft_reproduction(e3, args.ft_csv)
        args.facts_md.parent.mkdir(parents=True, exist_ok=True)
        facts(h1, w, e3, e3r, whr, wh, rep, ftrep, args.facts_md, res=res, pc1=pc1)
        rep.to_csv(args.facts_md.with_name("reproduction_check.csv"), index=False,
                   float_format="%.10g")
    print("wrote tables" + ("" if args.no_figure else ", figure") +
          ("" if args.facts_md is None else f" and {args.facts_md}"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("h1", "e3", "whiten", "pc1"):
        p = sub.add_parser(name)
        p.add_argument("--split_csv", type=Path, default=SPLIT_CSV)
        p.add_argument("--bases_root", type=Path, default=BASES_ROOT)
        p.add_argument("--ft_bases_root", type=Path, default=FT_BASES_ROOT)
        p.add_argument("--models", default="", help="comma list of ids or display names (default all)")
        p.add_argument("--layers", default="", help="comma list of layers (default all)")
        p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        p.add_argument("--out", type=Path, default=None)
    p = sub.add_parser("render")
    p.add_argument("--h1_csv", type=Path, default=H1_CSV)
    p.add_argument("--e3_csv", type=Path, default=E3_CSV)
    p.add_argument("--whiten_csv", type=Path, default=WH_CSV)
    p.add_argument("--pc1_csv", type=Path, default=PC1_CSV)
    p.add_argument("--results_csv", type=Path, default=RES_CSV)
    p.add_argument("--geom_csv", type=Path,
                   default=Path("runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv"))
    p.add_argument("--ft_csv", type=Path, action="append", default=None)
    p.add_argument("--tab_dir", type=Path, default=TAB_DIR)
    p.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    p.add_argument("--facts_md", type=Path, default=FACTS_MD)
    p.add_argument("--no_facts", dest="facts_md", action="store_const", const=None)
    p.add_argument("--no_figure", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "render":
        if args.ft_csv is None:
            ft_dir = Path("runs/active/resubmit/results/finetune")
            args.ft_csv = [ft_dir / f"finetune_{n}_layer_results.csv"
                           for n in ("lata", "qwen3_0.6b", "kalm_mini")]
        cmd_render(args)
    else:
        cmd_compute(args)


if __name__ == "__main__":
    main()
