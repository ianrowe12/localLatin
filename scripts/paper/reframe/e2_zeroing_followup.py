#!/usr/bin/env python3
"""Reframe experiment E2, zeroing follow-up (issue #252): what survives coordinate zeroing.

Question left open by E1 (issue #246): at the 26 collapsed T5 layers, zeroing the ten
top coordinates of the mean-pooled passage vectors restores none, although those ten
hold a median 92 percent of the squared loading of the first principal component (PC1).
After zeroing, one direction still holds 0.324 to 0.995 of the remaining variance. Is it
the same direction, and why does it still dominate? Hypothesis under test: zeroing fails
because PC2, PC3 and later components do not sit on those coordinates and so survive
zeroing; and/or PC1 itself survives.

  compute  per (model, layer, ranking, k), ranking in {mean_abs, variance}, k in
           {1,3,5,10}, every statistic fit on the 847 TRAIN passages only. With X the raw
           train vectors, w_1..w_10 the top principal components of the centered train
           vectors, S the zeroed coordinate set (E1's ranking and zeroing functions), Z
           the zeroed vectors and v_1..v_3 the top principal components of the centered
           zeroed train vectors:
             1. loading          sum over S of w_j[i]^2, j = 1..10;
             2. variance left    Var(Zc . w_j) / Var(Xc . w_j), j = 1..10, and
                                 Var(Zc . w_j) as a share of the total train variance of Z;
             3. angles           arccos |w_j . v_j| in degrees, j = 1,2,3; the three
                                 principal angles between span(w_1..w_3) and
                                 span(v_1..v_3); the remainder cosine
                                 |w_1 . v_1| / sqrt(1 - loading of PC1 on S), which is 1
                                 when v_1 is what is left of w_1 outside S; and
                                 |v_1 . w_j|, j = 1,2,3;
             4. score correlation  Pearson and Spearman between the passages' scores on
                                 the old PC_j (Xc . w_j) and on the new PC_j (Zc . v_j),
                                 j = 1,2,3, on train passages and on test passages (test
                                 centered with the train means);
             5. top-PC share and effective rank of the zeroed TRAIN vectors (pca_stats,
                                 as E1);
             6. intervention cells, Task A test AUROC through the paper's metric block
                (abtt_subspace_whiten._metrics, the path E1 uses):
                  (a) zero S, then ABTT with D in {1,2,3} fit on the zeroed train vectors;
                  (b) on the unzeroed vectors, center and remove original PCs 2 and 3
                      only, keeping PC1;
                with the reference cells alongside: base, zero only, ABTT D = 1,2,3,10 on
                the raw vectors.
  check    the reproduction gates on an existing CSV (no caches needed).
  render   the facts file for the prose.

Post hoc. The whole follow-up was designed after the E1 results were read. Measures 1 to
5 and the count thresholds of the facts file were written here before the first run. The
intervention cells of 6 carry no prediction: they are post hoc and are reported as numbers
without a verdict.

Order of operations and precision. Zeroing acts on the raw vectors (not centered, not
normalized), exactly as in E1; the centered zeroed vectors are then the centered raw
vectors with the columns of S set to 0, so every v_j is 0 on S. Cells that go through the
metric block, ABTT or pca_stats take the vectors in their cached float32, as E1 and H1
do, so that the gated cells are the same computation. The geometry of measures 1 to 4
(means, principal components, projections) is computed in float64. ABTT and cell (b) use
EmbeddingCleaner's components, the ones ABTT subtracts.

Reading measure 2. w_j is not renormalized: its entries on S multiply zeros, so
Var(Zc . w_j) understates the variance along the surviving direction (w_j outside S,
renormalized) by the factor 1 - loading. The CSV therefore also holds, for j = 1,2,3, the
variance of the zeroed train vectors along that renormalized remainder as a share of
their total (``remainder_share_pc{j}`` = the share above divided by 1 - loading). The
ratio Var(Zc . w_j) / Var(Xc . w_j) is not bounded by 1 either: the part due to PC j's own
scores is (1 - loading)^2, and the rest is variance of the other components, which w_j
with its S entries dropped is no longer orthogonal to (for j > 1, PC1 above all). The
remainder cosine and the angles say whether the dominant direction is the same one;
measure 2 says how much variance is left along the original axis.

Reproduction gates (``compute --check`` or ``check``; exit status 3 on failure, after the
CSV is written):
  1. the zeroed coordinate sets = the ``coords`` cells of the matching rows of
     runs/active/reframe/e1/e1_coordinate_ablation.csv, as strings;
  2. zero-only AUROC = E1's cells within 1e-6;
  3. top-PC share of the zeroed train vectors = E1's ``pc1_share_train`` within 1e-6;
  4. base and ABTT D = 1,2,3,10 AUROC on the raw vectors =
     runs/active/reframe/h1/h1_d_ablation.csv within 1e-6.
E1's looser 1e-5 applies to D=0 (centering) cells only; no such cell is gated here, so
every tolerance is 1e-6 (--tol). A NaN on either side of a comparison fails its gate, and
so does an expected model with no rows. compute --check gates the CSV it has just
written, so compute, check and render report identical numbers.

Outputs (small CSVs, force-added; embeddings are never written):
  runs/active/reframe/e2/e2_zeroing_followup.csv   one row per (model, layer, ranking,
                                                   k); per-layer reference cells
                                                   (``ref_*``) repeat on its rows
  runs/active/reframe/e2/e2_zeroing_gate_check.csv
  runs/active/reframe/e2/facts_e2_zeroing.md       (render)

Run from the repo root. The embedding caches are gitignored, so point --bases_root at a
checkout that has them; a model with no cache is an error unless --allow_missing:
  python scripts/paper/reframe/e2_zeroing_followup.py compute --check --workers 16 \
      --bases_root /u/irowerojas/localLatin/runs/active/resubmit_bases
  python scripts/paper/reframe/e2_zeroing_followup.py render
CPU only. Python 3.10, numpy / pandas / scipy / scikit-learn.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "resubmit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import abtt_subspace_whiten as asw  # noqa: E402
import e1_coordinate_ablation as e1  # noqa: E402
from sif_abtt import EmbeddingCleaner  # noqa: E402

SPLIT_CSV = asw.SPLIT_CSV
BASES_ROOT = asw.BASES_ROOT
H1_CSV = asw.H1_CSV
E1_CSV = e1.OUT_DIR / e1.ABL_NAME
E1_CONC_CSV = e1.OUT_DIR / e1.CONC_NAME
OUT_DIR = asw.OUT_ROOT / "e2"
CSV_NAME = "e2_zeroing_followup.csv"
GATE_NAME = "e2_zeroing_gate_check.csv"
FACTS_NAME = "facts_e2_zeroing.md"

MODELS = e1.MODELS
ALL_MODEL_IDS = e1.ALL_MODEL_IDS
DISP = e1.DISP
ORDER = e1.ORDER
T5 = e1.T5

RANKINGS = e1.RANKINGS  # ("mean_abs", "variance"), E1's functions and order
RANK_LABEL = e1.RANK_LABEL
KS = e1.KS  # (1, 3, 5, 10)
N_PC = 10  # original principal components followed (w_1..w_10)
N_TOP = 3  # components compared one to one and as a subspace (w_1..w_3 against v_1..v_3)
ZERO_ABTT_D = (1, 2, 3)  # ABTT fit on the zeroed train vectors
REF_ABTT_D = (1, 2, 3, 10)  # ABTT on the raw vectors, the H1 reference cells
RM_PCS = (2, 3)  # original components removed while PC1 is kept (1-indexed)
HEAD_RANKING, HEAD_K = "variance", KS[-1]  # the setting the facts file leads with
SINGLE_LAYERS = (("LaTa", 6), ("PhilTa", 10), ("mT5-base", 5))  # worst baseline layers (E1)

# Thresholds of the counts in the facts file, written here before the first run.
COLLAPSE_AUROC = e1.COLLAPSE_AUROC  # collapsed = T5 and baseline test AUROC below 0.70
RESTORE_AUROC = e1.RESTORE_AUROC  # 0.90, E1's "restored"
LOADING_BAR = 0.5  # "loading on the zeroed coordinates below 0.5"
VAR_LEFT_BAR = 0.5  # "variance left along the original PC above 50%"
ANGLE_BAR = 30.0  # "angle between old and new PC below 30 degrees"
CORR_BAR = 0.9  # "|Pearson| of old and new PC scores above 0.9"
GATE_TOL = 1e-6
GATE_EXIT = e1.GATE_EXIT  # 3: the CSV is written, the numbers disagree

LOAD_COLS = [f"load_pc{j}" for j in range(1, N_PC + 1)]
LEFT_COLS = [f"var_left_pc{j}" for j in range(1, N_PC + 1)]
ZSHARE_COLS = [f"var_share_zeroed_pc{j}" for j in range(1, N_PC + 1)]
REF_SHARE_COLS = [f"ref_var_share_pc{j}" for j in range(1, N_PC + 1)]
# reference cell of the follow-up CSV -> (variant, D) of the H1 CSV
H1_KEYS = {"ref_auc_base": ("raw", -1),
           **{f"ref_auc_abtt_D{D}": ("abtt", D) for D in REF_ABTT_D}}
H1_LABEL = {"ref_auc_base": "base", **{f"ref_auc_abtt_D{D}": f"ABTT D={D}" for D in REF_ABTT_D}}


# --------------------------------------------------------------------------- #
# Pure functions (unit-tested on synthetic arrays)
# --------------------------------------------------------------------------- #

def pc_basis(train: np.ndarray, n: int = N_PC) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train mean, top-n unit principal components as rows, all eigenvalues), float64.

    The components are the right singular vectors of the centered TRAIN vectors, as in
    e1.concentration; the eigenvalues are the squared singular values.
    """
    x = np.asarray(train, dtype=np.float64)
    mu = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:n], s ** 2


def loading(pcs: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """Squared loading of each unit component (row) on the coordinates in ``idx``."""
    return (np.asarray(pcs)[:, np.asarray(idx, dtype=np.int64)] ** 2).sum(axis=1)


def align_signs(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """``new`` with each row's sign flipped where its dot product with ``old`` is negative.

    The sign of a principal component is arbitrary; aligned, a passage's scores on an old
    and a new component correlate positively when the two are the same direction.
    """
    flip = np.where((old * new).sum(axis=1) < 0, -1.0, 1.0)
    return new * flip[:, None]


def pc_angles(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Angle in degrees between matching rows: arccos |old_j . new_j|, in [0, 90]."""
    c = np.clip(np.abs((old * new).sum(axis=1)), 0.0, 1.0)
    return np.degrees(np.arccos(c))


def principal_angles(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Principal angles in degrees, smallest first, between the row spaces of two
    orthonormal bases: arccos of the singular values of old @ new.T. A rotation of either
    basis within its own span leaves them unchanged."""
    s = np.linalg.svd(old @ new.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, 0.0, 1.0)))


def remainder_cosine(w: np.ndarray, v: np.ndarray, load: float) -> float:
    """|w . v| / sqrt(1 - load): cosine between v and the part of w outside S.

    ``load`` is the squared loading of the unit vector w on S, and v is 0 on S, so the
    value lies in [0, 1] and is 1 when v is that part of w, renormalized. NaN when w lies
    entirely on S (nothing of it is left).
    """
    rest = 1.0 - float(load)
    if not rest > 1e-12:
        return float("nan")
    return float(min(1.0, abs(float(w @ v)) / np.sqrt(rest)))


def score_corr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(Pearson, Spearman) between two score vectors; NaN when either is constant.

    Spearman is Pearson on average ranks (ties share their mean rank).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if not (a.std() > 0 and b.std() > 0):
        return float("nan"), float("nan")
    return (float(np.corrcoef(a, b)[0, 1]),
            float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]))


def remove_pcs(train: np.ndarray, test: np.ndarray, which: Sequence[int]
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Center on the train mean, then subtract the listed principal components (1-indexed)
    of the centered TRAIN vectors and keep the others.

    The components are EmbeddingCleaner's, so ``which = (1, .., D)`` is ABTT with D
    components (asw.abtt) and ``which = (2, 3)`` is ABTT D=3 with PC1 left in.
    """
    which = [int(j) for j in which]
    cleaner = EmbeddingCleaner(num_components=max(which), center=True).fit(train)
    pcs = cleaner.pcs[[j - 1 for j in which]]
    a, b = train - cleaner.mean_vec, test - cleaner.mean_vec
    return a - a @ pcs.T @ pcs, b - b @ pcs.T @ pcs


def zeroing_geometry(train: np.ndarray, test: np.ndarray, idx: Sequence[int],
                     basis: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                     ) -> Dict[str, float]:
    """Measures 1 to 4 of one zeroed coordinate set, in float64.

    ``train`` and ``test`` are the raw vectors, ``idx`` the coordinates set to 0 and
    ``basis`` may pass ``pc_basis(train)`` when the caller already has it. Everything is
    fit on train; the test rows enter the test correlations only. A component that does
    not exist (fewer than N_PC train passages or coordinates) leaves its columns NaN.
    """
    idx = np.asarray(idx, dtype=np.int64)
    mu, w, _ = basis if basis is not None else pc_basis(train)
    xc = np.asarray(train, dtype=np.float64) - mu
    tc = np.asarray(test, dtype=np.float64) - mu
    # Zeroing a raw coordinate and then centering on the train mean of the zeroed vectors
    # is the same as zeroing that column of the centered vectors.
    zc, zt = e1.zero_coords(xc, tc, idx)
    nan = float("nan")
    out: Dict[str, float] = {c: nan for c in LOAD_COLS + LEFT_COLS + ZSHARE_COLS}

    load = loading(w, idx)
    before = (xc @ w.T).var(axis=0, ddof=e1.SD_DDOF)
    after = (zc @ w.T).var(axis=0, ddof=e1.SD_DDOF)
    total_before = float(xc.var(axis=0, ddof=e1.SD_DDOF).sum())
    total_after = float(zc.var(axis=0, ddof=e1.SD_DDOF).sum())
    for j in range(len(w)):
        out[LOAD_COLS[j]] = float(load[j])
        out[LEFT_COLS[j]] = float(after[j] / before[j]) if before[j] > 0 else nan
        out[ZSHARE_COLS[j]] = float(after[j] / total_after) if total_after > 0 else nan
    out["total_var_left"] = total_after / total_before if total_before > 0 else nan
    # along w_j outside S, renormalized: Var(Zc . w_j) / (1 - loading), as a share
    for j in range(N_TOP):
        rest = 1.0 - float(load[j]) if j < len(w) else 0.0
        out[f"remainder_share_pc{j + 1}"] = (out[ZSHARE_COLS[j]] / rest if rest > 1e-12
                                             else nan)

    n_top = min(N_TOP, len(w))
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    old, new = w[:n_top], align_signs(w[:n_top], vt[:n_top])
    ang = pc_angles(old, new)
    pang = principal_angles(old, new)
    for j in range(N_TOP):
        out[f"angle_pc{j + 1}"] = float(ang[j]) if j < n_top else nan
    for j in range(N_TOP):
        out[f"pangle_{j + 1}"] = float(pang[j]) if j < n_top else nan
    out["remainder_cos_pc1"] = remainder_cosine(old[0], new[0], load[0])
    for j in range(N_TOP):
        out[f"abscos_newpc1_oldpc{j + 1}"] = (float(abs(new[0] @ old[j])) if j < n_top
                                              else nan)
    for split, a, b in (("train", xc, zc), ("test", tc, zt)):
        for j in range(N_TOP):
            p, s = score_corr(a @ old[j], b @ new[j]) if j < n_top else (nan, nan)
            out[f"pearson_{split}_pc{j + 1}"] = p
            out[f"spearman_{split}_pc{j + 1}"] = s
    return out


def layer_rows(model_id: str, layer: int, tr: np.ndarray, te: np.ndarray,
               metrics_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]]
               ) -> List[Dict]:
    """The follow-up rows of one model-layer, one per (ranking, k).

    ``tr`` and ``te`` are the raw pooled vectors as cached. ``metrics_fn(train, test)``
    returns the metric block of one intervention (it L2-normalizes internally). Two
    (ranking, k) cells that zero the same coordinates share one computation.
    """
    key = {"model": model_id, "layer": int(layer)}
    stats = e1.coord_stats(tr)
    order = {r: e1.rank_coords(tr, r, stats) for r in RANKINGS}
    basis = pc_basis(tr)
    ev = basis[2]

    def auc(a: np.ndarray, b: np.ndarray) -> float:
        return float(metrics_fn(a, b)["aucroc"])

    # per-layer reference cells, repeated on every row of the layer
    ref: Dict[str, float] = {"ref_auc_base": auc(tr, te)}
    for D in REF_ABTT_D:
        ref[f"ref_auc_abtt_D{D}"] = auc(*asw.abtt(tr, te, D))
    ref["ref_auc_rm_pc23"] = auc(*remove_pcs(tr, te, RM_PCS))
    total = float(ev.sum())
    for j, col in enumerate(REF_SHARE_COLS):
        ref[col] = float(ev[j] / total) if j < len(ev) and total > 0 else float("nan")

    info = {"dim": int(tr.shape[1]), "n_train": int(tr.shape[0]), "n_test": int(te.shape[0])}
    done: Dict[Tuple[int, ...], Dict[str, float]] = {}
    rows: List[Dict] = []
    for ranking in RANKINGS:
        for k in KS:
            idx = order[ranking][:k]
            same = tuple(sorted(int(i) for i in idx))
            if same not in done:
                a, b = e1.zero_coords(tr, te, idx)
                cells = {**zeroing_geometry(tr, te, idx, basis), **e1._geometry(a),
                         "auc_zero": auc(a, b)}
                for D in ZERO_ABTT_D:
                    cells[f"auc_zero_abtt_D{D}"] = auc(*asw.abtt(a, b, D))
                done[same] = cells
            rows.append({**key, "ranking": ranking, "k": int(k),
                         "coords": e1._coords_str(idx), **info, **done[same], **ref})
    return rows


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #

def task_e2(args) -> List[Dict]:
    bases_root, model_id, layer = args
    t0 = time.time()
    tr, te = asw._load(bases_root, asw.slug(model_id), layer)
    rows = layer_rows(model_id, layer, tr, te, asw._metrics)
    x = next(r for r in rows if r["ranking"] == HEAD_RANKING and r["k"] == HEAD_K)
    print(f"  e2z {DISP.get(model_id, model_id)} L{layer}: base {x['ref_auc_base']:.3f} "
          f"zero {x['auc_zero']:.3f} zero+D1/2/3 {x['auc_zero_abtt_D1']:.3f}/"
          f"{x['auc_zero_abtt_D2']:.3f}/{x['auc_zero_abtt_D3']:.3f} -PC2,3 "
          f"{x['ref_auc_rm_pc23']:.3f} load PC1/2/3 {x['load_pc1']:.2f}/{x['load_pc2']:.2f}/"
          f"{x['load_pc3']:.2f} angle {x['angle_pc1']:.1f} ({time.time() - t0:.1f}s)",
          flush=True)
    return rows


def read_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"coords": str})


def cmd_compute(args) -> int:
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    model_ids = asw.pick_models(args.models, MODELS)
    if not model_ids:
        raise SystemExit(f"--models {args.models!r} matches none of {ORDER}")
    t0 = time.time()
    tasks = e1.build_tasks(args.bases_root, model_ids, layers, args.allow_missing)
    if not tasks:
        raise SystemExit("ERROR: nothing to compute (no model has cached vectors)")
    rows: List[Dict] = []
    if args.workers <= 1:
        asw._init(str(args.split_csv))
        parts = map(task_e2, tasks)
    else:
        from multiprocessing import Pool

        pool = Pool(args.workers, initializer=asw._init, initargs=(str(args.split_csv),))
        parts = pool.imap(task_e2, tasks, chunksize=1)
    for part in parts:
        rows.extend(part)
    if args.workers > 1:
        pool.close()
        pool.join()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = e1._sorted(pd.DataFrame(rows), [])
    df.to_csv(args.out_dir / CSV_NAME, index=False, float_format="%.10g")
    print(f"wrote {args.out_dir / CSV_NAME} ({len(df)} rows, {len(tasks)} model-layers) "
          f"in {time.time() - t0:.0f}s")
    if not args.check:
        return 0
    # Gate the frame as re-read from the CSV (written at %.10g), not the in-memory one, so
    # that compute --check, check and render compute the gate numbers from the same values.
    return run_gates(read_rows(args.out_dir / CSV_NAME), args, complete=layers is None,
                     write=True)


# --------------------------------------------------------------------------- #
# Reproduction gates
# --------------------------------------------------------------------------- #

def _cell_label(layer: int, ranking: str, k: int) -> str:
    return f"L{int(layer)} {ranking} k={int(k)}"


def gate_table(rows: pd.DataFrame, e1_abl: pd.DataFrame, h1: pd.DataFrame,
               complete: bool = True, tol: float = GATE_TOL,
               expected: Optional[Sequence[str]] = ALL_MODEL_IDS) -> pd.DataFrame:
    """One row per (gate, model): cells compared, cells missing, max |difference|, and
    the cells over tolerance by name.

    Gate 1 compares the zeroed coordinate sets with E1's ``coords`` cells as strings (its
    ``n_over_tolerance`` is the number of sets that differ; it has no tolerance). Gates 2
    and 3 compare the zero-only AUROC and the top-PC share of the zeroed train vectors
    with E1's cells, gate 4 the base and ABTT cells on the raw vectors with the H1 CSV,
    all within ``tol``.

    A NaN on either side of a comparison counts as over tolerance. ``expected`` lists the
    model ids that must have rows (default: all six); one with none gets a failing gate-0
    row. Pass None to gate only the models that are present. ``complete`` also requires
    every zero row of E1 and every H1 layer of a computed model to be present (turn it
    off for a --layers subset).
    """
    ours = rows.set_index(["model", "layer", "ranking", "k"])
    if not ours.index.is_unique:
        raise ValueError("duplicate (model, layer, ranking, k) rows in the follow-up CSV")
    ref = e1_abl[e1_abl["intervention"] == "zero"].set_index(["model", "layer", "ranking", "k"])
    h1i = h1.set_index(["model", "layer", "variant", "D"])["aucroc"]
    h1_layers = h1.groupby("model")["layer"].unique()
    present = set(rows["model"])
    wanted = set(expected) if expected is not None else set()
    known = list(ALL_MODEL_IDS) + sorted((present | wanted) - set(ALL_MODEL_IDS))
    nan = float("nan")
    out = []
    for mid in [m for m in known if m in present or m in wanted]:
        if mid not in present:
            out.append({"gate": "0 expected model has rows in the follow-up CSV", "model": mid,
                        "n_cells": 0, "n_missing_reference": 0, "n_reference_rows_absent": 0,
                        "max_abs_diff": nan, "tolerance": nan, "n_over_tolerance": 0,
                        "cells_over_tolerance": "", "ok": False})
            continue
        keys = [k for k in ours.index if k[0] == mid]
        mine = set(keys)
        ref_keys = [k for k in ref.index if k[0] == mid]
        absent = [k for k in ref_keys if k not in mine] if complete else []
        layers = sorted({k[1] for k in keys})
        h1_absent = ([x for x in h1_layers.get(mid, []) if x not in layers] if complete
                     else [])

        def ref_cell(k, col):
            return ref.loc[k, col] if k in ref.index else None

        # gate 1: strings, so its own comparison
        pairs = [(_cell_label(*k[1:]), str(ours.loc[k, "coords"]), ref_cell(k, "coords"))
                 for k in keys]
        differ = [(lab, a, str(b)) for lab, a, b in pairs if b is not None and a != str(b)]
        n_missing = sum(1 for _, _, b in pairs if b is None)
        n_cmp = len(pairs) - n_missing
        out.append({"gate": "1 zeroed coordinate sets vs E1", "model": mid, "n_cells": n_cmp,
                    "n_missing_reference": n_missing, "n_reference_rows_absent": len(absent),
                    "max_abs_diff": nan, "tolerance": nan, "n_over_tolerance": len(differ),
                    "cells_over_tolerance": "; ".join(f"{lab} {a} (E1 {b})"
                                                      for lab, a, b in differ),
                    "ok": bool(n_cmp) and n_missing == 0 and not absent and not differ})

        first = {x: next(k for k in keys if k[1] == x) for x in layers}
        # each cell: (label, ours, reference or None)
        specs = [
            ("2 zero-only AUROC vs E1", absent,
             [(_cell_label(*k[1:]), float(ours.loc[k, "auc_zero"]), ref_cell(k, "aucroc"))
              for k in keys]),
            ("3 top-PC share of the zeroed train vectors vs E1", absent,
             [(_cell_label(*k[1:]), float(ours.loc[k, "pc1_share_train"]),
               ref_cell(k, "pc1_share_train")) for k in keys]),
            ("4 base and ABTT D=1,2,3,10 AUROC on the raw vectors vs H1", h1_absent,
             [(f"L{int(x)} {H1_LABEL[col]}", float(ours.loc[first[x], col]),
               h1i.get((mid, x, variant, D)))
              for x in layers for col, (variant, D) in H1_KEYS.items()]),
        ]
        for name, gone, cells in specs:
            diffs = [(lab, abs(a - float(b))) for lab, a, b in cells if b is not None]
            n_missing = sum(1 for _, _, b in cells if b is None)
            # np.max keeps a NaN difference visible (the builtin max can hide it), and
            # "not d <= tol" counts a NaN as over tolerance ("d > tol" would let it pass).
            mx = float(np.max([d for _, d in diffs])) if diffs else nan
            over = [(lab, d) for lab, d in diffs if not d <= tol]
            out.append({"gate": name, "model": mid, "n_cells": len(diffs),
                        "n_missing_reference": n_missing,
                        "n_reference_rows_absent": len(gone), "max_abs_diff": mx,
                        "tolerance": tol, "n_over_tolerance": len(over),
                        "cells_over_tolerance": "; ".join(f"{lab} {d:.2e}" for lab, d in over),
                        "ok": bool(diffs) and n_missing == 0 and not gone and not over})
    return pd.DataFrame(out)


def write_gates(gates: pd.DataFrame, path: Path) -> None:
    gates.to_csv(path, index=False, float_format="%.6g")


def gate_line(g) -> str:
    """One gate row in words, shared by the log and the facts file."""
    name = DISP.get(g.model, g.model)
    if g.gate.startswith("0"):
        return f"gate {g.gate}, {name}: no rows: FAIL"
    if g.gate.startswith("1"):
        text = (f"gate {g.gate}, {name}: {g.n_cells} sets, {g.n_over_tolerance} differ: "
                f"{'PASS' if g.ok else 'FAIL'}")
    else:
        text = (f"gate {g.gate}, {name}: {g.n_cells} cells, max |diff| {g.max_abs_diff:.2e} "
                f"(tolerance {g.tolerance:.0e}): {'PASS' if g.ok else 'FAIL'}")
    if g.n_missing_reference or g.n_reference_rows_absent:
        text += (f"; missing reference cells {g.n_missing_reference}, reference rows absent "
                 f"from the follow-up CSV {g.n_reference_rows_absent}")
    if g.n_over_tolerance:
        text += (f"; {'sets that differ' if g.gate.startswith('1') else 'cells over tolerance'}"
                 f": {g.cells_over_tolerance}")
    return text


def gates_for(rows: pd.DataFrame, args, complete: bool = True) -> pd.DataFrame:
    return gate_table(rows, e1.read_abl(args.e1_csv), pd.read_csv(args.h1_csv),
                      complete=complete, tol=args.tol, expected=e1.expected_models(args))


def run_gates(rows: pd.DataFrame, args, complete: bool, write: bool) -> int:
    gates = gates_for(rows, args, complete=complete)
    if write:
        write_gates(gates, args.out_dir / GATE_NAME)
    for g in gates.itertuples():
        print(gate_line(g))
    if gates.empty or not gates["ok"].all():
        print("REPRODUCTION GATES FAILED")
        return GATE_EXIT
    print("reproduction gates passed")
    return 0


def cmd_check(args) -> int:
    return run_gates(read_rows(args.out_dir / CSV_NAME), args, complete=True,
                     write=not args.no_write)


# --------------------------------------------------------------------------- #
# render: facts
# --------------------------------------------------------------------------- #

def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    """The CSV rows plus display name, collapsed flag and the derived columns of the facts
    file (absolute correlations, zero + ABTT minus ABTT on the raw vectors)."""
    d = rows.copy()
    d["m"] = d["model"].map(lambda x: DISP.get(x, x))
    d["is_t5"] = d["m"].isin(T5)
    d["collapsed"] = d["is_t5"] & (d["ref_auc_base"] < COLLAPSE_AUROC)
    for split in ("train", "test"):
        for j in range(1, N_TOP + 1):
            for kind in ("pearson", "spearman"):
                d[f"abs_{kind}_{split}_pc{j}"] = d[f"{kind}_{split}_pc{j}"].abs()
    for D in ZERO_ABTT_D:
        d[f"diff_zero_abtt_D{D}"] = d[f"auc_zero_abtt_D{D}"] - d[f"ref_auc_abtt_D{D}"]
    return d


def at(d: pd.DataFrame, ranking: str = HEAD_RANKING, k: int = HEAD_K) -> pd.DataFrame:
    """One row per model-layer: the rows of one (ranking, k)."""
    return d[(d["ranking"] == ranking) & (d["k"] == k)].reset_index(drop=True)


def _rng(s: pd.Series, fmt: str = ".3f") -> str:
    """median (min to max); NaN cells are left out and counted."""
    n_nan = int(s.isna().sum())
    if n_nan == len(s):
        return "nan"
    text = f"{format(s.median(), fmt)} ({format(s.min(), fmt)} to {format(s.max(), fmt)})"
    return text + (f" [{n_nan} NaN left out]" if n_nan else "")


def _cnt(mask: pd.Series) -> str:
    return f"{int(mask.sum())}/{len(mask)}"


def _by_model(t: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    return [(m, t[t["m"] == m]) for m in ORDER if (t["m"] == m).any()]


def _line(t: pd.DataFrame, col: str, fmt: str = ".3f") -> str:
    """median (min to max) over ``t``, then the median per model."""
    return _rng(t[col], fmt) + "; " + ", ".join(
        f"{m} {format(g[col].median(), fmt)}" for m, g in _by_model(t))


def _per_pc(t: pd.DataFrame, cols: Sequence[str], fmt: str = ".3f") -> List[str]:
    return [f"  - PC{j}: {_line(t, col, fmt)}" for j, col in enumerate(cols, start=1)]


AUC_CELLS = (("base", "ref_auc_base"), ("zero only", "auc_zero"),
             *((f"zero + ABTT D={D}", f"auc_zero_abtt_D{D}") for D in ZERO_ABTT_D),
             ("PCs 2 and 3 removed, PC1 kept", "ref_auc_rm_pc23"),
             *((f"ABTT D={D}", f"ref_auc_abtt_D{D}") for D in REF_ABTT_D))


def measure_lines(t: pd.DataFrame, n_label: str) -> List[str]:
    """Sections 1.1 to 1.6: every measure over the layers in ``t`` (one row per layer)."""
    L: List[str] = []
    a = L.append
    n = len(t)
    a("### 1.1 Loading of the original PCs on the zeroed coordinates")
    a(f"- median (min to max) over the {n_label}; then the median per model")
    L.extend(_per_pc(t, LOAD_COLS))
    a(f"- loading below {LOADING_BAR} at: " + ", ".join(
        f"PC{j} {_cnt(t[c] < LOADING_BAR)}" for j, c in enumerate(LOAD_COLS, start=1)))
    a("### 1.2 Variance left along each original PC")
    a("- Var(Zc . w_j) / Var(Xc . w_j):")
    L.extend(_per_pc(t, LEFT_COLS))
    a(f"- variance left above {VAR_LEFT_BAR:.0%} at: " + ", ".join(
        f"PC{j} {_cnt(t[c] > VAR_LEFT_BAR)}" for j, c in enumerate(LEFT_COLS, start=1)))
    a("- Var(Zc . w_j) as a share of the total train variance of the zeroed vectors:")
    L.extend(_per_pc(t, ZSHARE_COLS))
    a("- variance of the zeroed vectors along what is left of w_j outside S, renormalized, "
      "as a share of their total train variance (the share above divided by 1 - loading):")
    L.extend(_per_pc(t, [f"remainder_share_pc{j}" for j in range(1, N_TOP + 1)]))
    a("- for scale, the share of the total train variance of the unzeroed vectors on each "
      "original PC:")
    L.extend(_per_pc(t, REF_SHARE_COLS))
    a(f"- total train variance left after zeroing (zeroed / unzeroed): "
      f"{_line(t, 'total_var_left')}")
    a("### 1.3 Angles between old and new components")
    for j in range(1, N_TOP + 1):
        a(f"- angle between old and new PC{j}, degrees: {_line(t, f'angle_pc{j}', '.1f')}")
    a(f"- angle below {ANGLE_BAR:.0f} degrees at: " + ", ".join(
        f"PC{j} {_cnt(t[f'angle_pc{j}'] < ANGLE_BAR)}" for j in range(1, N_TOP + 1)))
    for j, name in zip(range(1, N_TOP + 1), ("smallest", "middle", "largest")):
        a(f"- {name} principal angle between span(old PCs 1 to 3) and span(new PCs 1 to 3), "
          f"degrees: {_line(t, f'pangle_{j}', '.1f')}")
    a(f"- largest principal angle below {ANGLE_BAR:.0f} degrees at "
      f"{_cnt(t[f'pangle_{N_TOP}'] < ANGLE_BAR)}")
    a(f"- remainder cosine of PC1: {_line(t, 'remainder_cos_pc1')}")
    a(f"- remainder cosine of PC1 above {CORR_BAR} at {_cnt(t['remainder_cos_pc1'] > CORR_BAR)}")
    for j in range(1, N_TOP + 1):
        a(f"- |cosine| between the new PC1 and the old PC{j}: "
          f"{_line(t, f'abscos_newpc1_oldpc{j}')}")
    cos = t[[f"abscos_newpc1_oldpc{j}" for j in range(1, N_TOP + 1)]].to_numpy()
    if n and np.isfinite(cos).all():
        best = cos.argmax(axis=1)
        a("- old PC (among 1 to 3) with the largest |cosine| to the new PC1: " + ", ".join(
            f"PC{j + 1} at {int((best == j).sum())}/{n}" for j in range(N_TOP)))
    a("### 1.4 Correlation of the passages' scores on the old and the new component")
    for split in ("train", "test"):
        for j in range(1, N_TOP + 1):
            a(f"- PC{j}, {split} passages: |Pearson| "
              f"{_line(t, f'abs_pearson_{split}_pc{j}')}")
            a(f"  - |Spearman| {_line(t, f'abs_spearman_{split}_pc{j}')}")
    for split in ("train", "test"):
        a(f"- |Pearson| above {CORR_BAR} on {split} passages at: " + ", ".join(
            f"PC{j} {_cnt(t[f'abs_pearson_{split}_pc{j}'] > CORR_BAR)}"
            for j in range(1, N_TOP + 1)))
    a("### 1.5 Top-PC share and effective rank of the zeroed training vectors")
    a(f"- top-PC share after zeroing: {_line(t, 'pc1_share_train')}")
    a(f"  - before zeroing (share of the unzeroed train variance on PC1): "
      f"{_line(t, 'ref_var_share_pc1')}")
    a(f"- effective rank after zeroing: {_line(t, 'eff_rank_train', '.2f')}")
    a("### 1.6 The counts in one place")
    a(f"- PC1 loading on the zeroed coordinates below {LOADING_BAR} at "
      f"{_cnt(t['load_pc1'] < LOADING_BAR)}; PC2 at {_cnt(t['load_pc2'] < LOADING_BAR)}; "
      f"PC3 at {_cnt(t['load_pc3'] < LOADING_BAR)}")
    a(f"- variance left along the original PC1 above {VAR_LEFT_BAR:.0%} at "
      f"{_cnt(t['var_left_pc1'] > VAR_LEFT_BAR)}; PC2 at "
      f"{_cnt(t['var_left_pc2'] > VAR_LEFT_BAR)}; PC3 at "
      f"{_cnt(t['var_left_pc3'] > VAR_LEFT_BAR)}")
    a(f"- angle between old and new PC1 below {ANGLE_BAR:.0f} degrees at "
      f"{_cnt(t['angle_pc1'] < ANGLE_BAR)}; remainder cosine of PC1 above {CORR_BAR} at "
      f"{_cnt(t['remainder_cos_pc1'] > CORR_BAR)}")
    a(f"- |Pearson| of old and new PC1 scores above {CORR_BAR} at "
      f"{_cnt(t['abs_pearson_train_pc1'] > CORR_BAR)} (train passages) and "
      f"{_cnt(t['abs_pearson_test_pc1'] > CORR_BAR)} (test passages)")
    return L


def auc_lines(t: pd.DataFrame, zero_only: bool = False) -> List[str]:
    """AUROC of every intervention cell over the layers in ``t`` (one row per layer)."""
    L = []
    for label, col in AUC_CELLS:
        if zero_only and not col.startswith("auc_zero"):
            continue
        s = t[col]
        L.append(f"- {label}: AUROC >= {RESTORE_AUROC:.2f} at {_cnt(s >= RESTORE_AUROC)}; "
                 f"{_rng(s)}; " + ", ".join(
                     f"{m} {_cnt(g[col] >= RESTORE_AUROC)}" for m, g in _by_model(t)))
    return L


def layer_lines(x: pd.Series) -> List[str]:
    """Every measure of one row (one model-layer at one ranking and k)."""

    def seq(cols: Sequence[str], fmt: str = ".3f") -> str:
        return " / ".join(format(x[c], fmt) for c in cols)

    top = range(1, N_TOP + 1)
    return [
        f"  - zeroed coordinates: {x['coords']}",
        f"  - loading of PC1..PC{N_PC} on them: {seq(LOAD_COLS)}",
        f"  - variance left along PC1..PC{N_PC}: {seq(LEFT_COLS)}",
        f"  - that variance as a share of the zeroed total, PC1..PC{N_PC}: {seq(ZSHARE_COLS)}",
        "  - share of the zeroed total along what is left of PC1/2/3 outside the zeroed "
        f"coordinates, renormalized: {seq([f'remainder_share_pc{j}' for j in top])}",
        f"  - share of the unzeroed total on PC1..PC{N_PC}: {seq(REF_SHARE_COLS)}; total "
        f"variance left after zeroing {x['total_var_left']:.3f}",
        f"  - angle old/new PC1/2/3, degrees: {seq([f'angle_pc{j}' for j in top], '.1f')}; "
        f"principal angles {seq([f'pangle_{j}' for j in top], '.1f')}; remainder cosine of "
        f"PC1 {x['remainder_cos_pc1']:.3f}; |cosine| of the new PC1 with old PC1/2/3 "
        f"{seq([f'abscos_newpc1_oldpc{j}' for j in top])}",
        f"  - |Pearson| old/new scores PC1/2/3: train "
        f"{seq([f'abs_pearson_train_pc{j}' for j in top])}, test "
        f"{seq([f'abs_pearson_test_pc{j}' for j in top])}; |Spearman|: train "
        f"{seq([f'abs_spearman_train_pc{j}' for j in top])}, test "
        f"{seq([f'abs_spearman_test_pc{j}' for j in top])}",
        f"  - zeroed train vectors: top-PC share {x['pc1_share_train']:.3f}, effective rank "
        f"{x['eff_rank_train']:.2f}",
        "  - AUROC: " + "; ".join(f"{label} {x[col]:.3f}" for label, col in AUC_CELLS),
    ]


TABLE_COLS = (("base", "ref_auc_base", ".3f"), ("load1", "load_pc1", ".3f"),
              ("load2", "load_pc2", ".3f"), ("load3", "load_pc3", ".3f"),
              ("left1", "var_left_pc1", ".3f"), ("left2", "var_left_pc2", ".3f"),
              ("left3", "var_left_pc3", ".3f"), ("ang1", "angle_pc1", ".1f"),
              ("pang3", f"pangle_{N_TOP}", ".1f"), ("rem", "remainder_cos_pc1", ".3f"),
              ("rsh1", "remainder_share_pc1", ".3f"),
              ("r1", "abs_pearson_train_pc1", ".3f"), ("share", "pc1_share_train", ".3f"),
              ("zero", "auc_zero", ".3f"), ("z+D1", "auc_zero_abtt_D1", ".3f"),
              ("z+D2", "auc_zero_abtt_D2", ".3f"), ("z+D3", "auc_zero_abtt_D3", ".3f"),
              ("-PC2,3", "ref_auc_rm_pc23", ".3f"), ("D1", "ref_auc_abtt_D1", ".3f"),
              ("D2", "ref_auc_abtt_D2", ".3f"), ("D3", "ref_auc_abtt_D3", ".3f"),
              ("D10", "ref_auc_abtt_D10", ".3f"))


def _md_table(t: pd.DataFrame) -> List[str]:
    head = ["L"] + [c[0] for c in TABLE_COLS]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, x in t.sort_values("layer").iterrows():
        out.append(f"| {int(x['layer'])}{'*' if x['collapsed'] else ''} | "
                   + " | ".join(format(x[col], fmt) for _, col, fmt in TABLE_COLS) + " |")
    return out


def facts(d: pd.DataFrame, gates: Optional[pd.DataFrame], path: Path,
          conc: Optional[pd.DataFrame] = None) -> None:
    """Write the facts file from the prepared rows (``prepare``)."""
    L: List[str] = []
    a = L.append
    head = at(d)
    present = [m for m in ORDER if (d["m"] == m).any()]
    omitted = [m for m in ORDER if m not in present]
    coll = head[head["collapsed"]]
    n_coll = len(coll)
    k_word = f"k={HEAD_K}, ranking by {RANK_LABEL[HEAD_RANKING]}"

    a("# E2 zeroing follow-up: facts (generated)")
    a("")
    a("Generated by `scripts/paper/reframe/e2_zeroing_followup.py render` from "
      f"`{CSV_NAME}` in this directory. Every number below is a cell of that CSV or a "
      "count, median, minimum, maximum or difference of such cells. AUROC is Task A test "
      "AUROC.")
    a("")
    a("Question. E1 found that zeroing the ten top coordinates restores none of the "
      "collapsed T5 layers, although those ten hold most of the squared loading of PC1, "
      "and that one direction still dominates the zeroed vectors. It did not test whether "
      "that is the same direction, or why it still dominates. Hypothesis under test: "
      "zeroing fails because PC2, PC3 and later components do not sit on those coordinates "
      "and so survive zeroing; and/or PC1 itself survives.")
    a("")
    a("What is post hoc. The whole follow-up was designed after the E1 results were read. "
      "The measures of sections 1, 2 and 4 to 6 and the thresholds of the counts (loading "
      f"below {LOADING_BAR}, variance left above {VAR_LEFT_BAR:.0%}, angle below "
      f"{ANGLE_BAR:.0f} degrees, |Pearson| above {CORR_BAR}, AUROC >= {RESTORE_AUROC:.2f}, "
      f"collapsed = T5 and baseline AUROC < {COLLAPSE_AUROC:.2f}) were written into the "
      "script before its first run. The intervention cells of section 3 (zeroing followed "
      "by ABTT; original PCs 2 and 3 removed with PC1 kept) were added post hoc: no "
      "prediction is attached to them and they are reported as numbers without a verdict. "
      "No section prints a verdict; the reproduction gates of section 0 are the only "
      "PASS or FAIL lines.")
    a("")
    a("## Definitions")
    a("- Vectors: mean-pooled hidden states (`hidden_mean_tokempty`), raw as cached, the "
      "vectors of E1. Every statistic is fit on the training passages "
      f"({'/'.join(str(int(v)) for v in sorted(d['n_train'].unique()))}) and applied to "
      "train and test.")
    a("- S: the top k coordinates under a ranking, chosen by E1's functions on the raw "
      "training vectors (`mean |x|` = mean absolute value over training passages, "
      "`variance` = variance across them). Zeroing sets them to 0 in the raw vectors "
      "(not centered, not normalized), exactly as E1.")
    a(f"- w_1..w_{N_PC}: the top principal components of the centered training vectors "
      f"(\"old\" or \"original\" PCs). v_1..v_{N_TOP}: the top principal components of the "
      "centered zeroed training vectors (\"new\" PCs). Xc, Zc: the centered unzeroed and "
      "zeroed vectors. Zc is Xc with the columns of S set to 0, so every v_j is 0 on S.")
    a("- Loading of PC j on S: sum over S of w_j[i]^2 (w_j has unit length, so 1 = the "
      "component lies entirely on S). For PC1 at k=10 under the variance ranking this is "
      "E1's `PC1 mass on top 10`.")
    a("- Variance left along PC j: Var(Zc . w_j) / Var(Xc . w_j) over training passages "
      f"(population variance, ddof={e1.SD_DDOF}). w_j is not renormalized: its entries on S "
      "multiply zeros, so this understates the variance along the surviving direction "
      "(w_j outside S, renormalized) by the factor 1 - loading. The same holds for "
      "Var(Zc . w_j) as a share of the zeroed total, so the share along the renormalized "
      "remainder (that share divided by 1 - loading) is given beside it for PCs 1 to 3; "
      "it is comparable with the top-PC share of the zeroed vectors, which bounds it. The "
      "ratio is not bounded by 1: the part due to PC j's own scores is (1 - loading)^2, "
      "and the rest is variance of the other components, which w_j with its S entries "
      "dropped is no longer orthogonal to (for j > 1, PC1 above all). Read the angles and "
      "the remainder cosine for whether the dominant direction is the same one.")
    a("- Angle between old and new PC j: arccos |w_j . v_j| in degrees (0 = same "
      "direction, 90 = orthogonal). Zeroing can reorder components and components of "
      "nearly equal variance can mix, so the per-component angles of PC2 and PC3 are read "
      "together with the principal angles between span(w_1..w_3) and span(v_1..v_3) "
      "(arccos of the singular values of the 3 x 3 matrix of dot products; all 0 = same "
      "subspace).")
    a("- Remainder cosine of PC1: |w_1 . v_1| / sqrt(1 - loading of PC1 on S), in [0, 1]. "
      "It is 1 when the new top direction is exactly what is left of the old one outside "
      "S, renormalized. NaN when PC1 lies entirely on S.")
    a("- Score correlation: Pearson and Spearman between the passages' scores on the old "
      "PC j (Xc . w_j) and on the new PC j (Zc . v_j), on train passages and on test "
      "passages (test centered with the train means). The sign of a component is "
      "arbitrary: v_j is flipped to have a non-negative dot product with w_j, and the "
      "counts and medians below use absolute values.")
    a("- Top-PC share and effective rank after zeroing: `pca_stats` on the zeroed TRAIN "
      "vectors, the cell E1 reports.")
    a("- Intervention cells (section 3): `zero + ABTT D` = zero S, then ABTT with D "
      "components fit on the zeroed training vectors; `PCs 2 and 3 removed, PC1 kept` = "
      "center the unzeroed vectors on the train mean and subtract original PCs 2 and 3 "
      "only (ABTT D=3 with PC1 left in; it does not depend on S). Reference cells: base, "
      "zero only (E1), ABTT D=1, 2, 3, 10 on the unzeroed vectors (H1). All through the "
      "paper's metric block, as E1.")
    a(f"- Collapsed = T5 model and base AUROC < {COLLAPSE_AUROC:.2f}. `x (y to z)` = median "
      "(min to max). Counts are `layers satisfying / layers`.")
    a("")

    a("## 0. Coverage and reproduction gates")
    a("- model-layers: " + ", ".join(f"{m} {int((head['m'] == m).sum())}" for m in present)
      + f" (total {len(head)}); rows: {len(d)}")
    if omitted:
        a("- ABSENT from the CSV, so omitted from every count below: " + ", ".join(omitted))
    if gates is None or gates.empty:
        a("- gates: reference CSVs not found, not evaluated")
    else:
        for g in gates.itertuples():
            a("- " + gate_line(g))
        if not gates["ok"].all():
            a("- A gate FAILED. Read the cells it names before quoting numbers that depend on "
              "them.")
    if conc is not None and not conc.empty and "pc1_mass_top10var" in conc.columns:
        both = head.merge(conc[["model", "layer", "pc1_mass_top10var"]],
                          on=["model", "layer"], how="inner")
        if len(both):
            diff = (both["load_pc1"] - both["pc1_mass_top10var"]).abs()
            a(f"- consistency with E1 (not a gate): the PC1 loading at {k_word} against "
              f"`pc1_mass_top10var` of `{e1.CONC_NAME}`: {len(both)} model-layers, max "
              f"|diff| {diff.max():.2e}")
    a("")

    a(f"## 1. Collapsed T5 layers, {k_word}")
    a(f"- {n_coll} layers: " + (", ".join(
        f"{m} {len(g)} ({', '.join(str(int(v)) for v in sorted(g['layer']))})"
        for m, g in _by_model(coll)) or "none"))
    if n_coll:
        L.extend(measure_lines(coll, f"{n_coll} collapsed layers"))
    a("")

    a(f"## 2. Collapsed T5 layers, k={HEAD_K}, ranking by {RANK_LABEL['mean_abs']} (brief)")
    other = at(d, "mean_abs", HEAD_K)
    other = other[other["collapsed"]]
    if len(other):
        pair = coll.merge(other[["model", "layer", "coords"]], on=["model", "layer"],
                          suffixes=("", "_other"))
        same = pair.apply(lambda x: set(x["coords"].split(";"))
                          == set(x["coords_other"].split(";")), axis=1)
        a(f"- the two rankings zero the same {HEAD_K} coordinates at {_cnt(same)} collapsed "
          "layers")
        for label, col, fmt in (
                ("loading of PC1", "load_pc1", ".3f"), ("loading of PC2", "load_pc2", ".3f"),
                ("loading of PC3", "load_pc3", ".3f"),
                ("variance left along PC1", "var_left_pc1", ".3f"),
                ("variance left along PC2", "var_left_pc2", ".3f"),
                ("variance left along PC3", "var_left_pc3", ".3f"),
                ("angle old/new PC1, degrees", "angle_pc1", ".1f"),
                ("largest principal angle, degrees", f"pangle_{N_TOP}", ".1f"),
                ("remainder cosine of PC1", "remainder_cos_pc1", ".3f"),
                ("share of the zeroed variance along the remainder of PC1",
                 "remainder_share_pc1", ".3f"),
                ("|Pearson| old/new PC1 scores, train", "abs_pearson_train_pc1", ".3f"),
                ("top-PC share after zeroing", "pc1_share_train", ".3f")):
            a(f"- {label}: {_line(other, col, fmt)}")
        a(f"- PC1 loading below {LOADING_BAR} at {_cnt(other['load_pc1'] < LOADING_BAR)}; PC2 "
          f"at {_cnt(other['load_pc2'] < LOADING_BAR)}; PC3 at "
          f"{_cnt(other['load_pc3'] < LOADING_BAR)}; variance left along PC1 above "
          f"{VAR_LEFT_BAR:.0%} at {_cnt(other['var_left_pc1'] > VAR_LEFT_BAR)}; angle old/new "
          f"PC1 below {ANGLE_BAR:.0f} degrees at {_cnt(other['angle_pc1'] < ANGLE_BAR)}; "
          f"|Pearson| of PC1 scores (train) above {CORR_BAR} at "
          f"{_cnt(other['abs_pearson_train_pc1'] > CORR_BAR)}")
        L.extend(auc_lines(other, zero_only=True))
    a("")

    a("## 3. Intervention cells at the collapsed T5 layers (post hoc, no prediction)")
    a("These cells were added post hoc. No prediction is attached to them and no verdict "
      "is drawn here.")
    if n_coll:
        a(f"### {k_word}")
        L.extend(auc_lines(coll))
        for D in ZERO_ABTT_D:
            s = coll[f"diff_zero_abtt_D{D}"]
            a(f"- zero + ABTT D={D} minus ABTT D={D} on the unzeroed vectors: median "
              f"{s.median():+.3f}, range {s.min():+.3f} to {s.max():+.3f}; higher at "
              f"{int((s > 1e-9).sum())}, lower at {int((s < -1e-9).sum())} of {n_coll}")
        s = coll["ref_auc_rm_pc23"] - coll["ref_auc_abtt_D3"]
        a(f"- PCs 2 and 3 removed (PC1 kept) minus ABTT D=3: median {s.median():+.3f}, range "
          f"{s.min():+.3f} to {s.max():+.3f}")
        s = coll["ref_auc_rm_pc23"] - coll["ref_auc_base"]
        a(f"- PCs 2 and 3 removed (PC1 kept) minus base: median {s.median():+.3f}, range "
          f"{s.min():+.3f} to {s.max():+.3f}")
    a("")

    a(f"## 4. Smaller k at the collapsed T5 layers, ranking by {RANK_LABEL[HEAD_RANKING]}")
    a("Median over the collapsed layers; for the AUROC cells the count of layers at or "
      f"above {RESTORE_AUROC:.2f} follows in parentheses.")
    cols = (("load PC1", "load_pc1", ".3f", False), ("load PC2", "load_pc2", ".3f", False),
            ("load PC3", "load_pc3", ".3f", False), ("left PC1", "var_left_pc1", ".3f", False),
            ("left PC2", "var_left_pc2", ".3f", False), ("angle PC1", "angle_pc1", ".1f", False),
            ("rem cos", "remainder_cos_pc1", ".3f", False),
            ("top-PC share", "pc1_share_train", ".3f", False),
            ("zero", "auc_zero", ".3f", True),
            *((f"zero+D{D}", f"auc_zero_abtt_D{D}", ".3f", True) for D in ZERO_ABTT_D))
    if n_coll:
        a("| k | " + " | ".join(c[0] for c in cols) + " |")
        a("|" + "---|" * (len(cols) + 1))
        for k in KS:
            t = at(d, HEAD_RANKING, k)
            t = t[t["collapsed"]]
            a(f"| {k} | " + " | ".join(
                format(t[col].median(), fmt)
                + (f" ({_cnt(t[col] >= RESTORE_AUROC)})" if is_auc else "")
                for _, col, fmt, is_auc in cols) + " |")
    a("")

    a(f"## 5. Single layers, {k_word}")
    for name, layer in SINGLE_LAYERS:
        x = head[(head["m"] == name) & (head["layer"] == layer)]
        if x.empty:
            a(f"- {name} L{layer}: no rows")
            continue
        a(f"- {name} L{layer}{' (collapsed)' if x.iloc[0]['collapsed'] else ''}:")
        L.extend(layer_lines(x.iloc[0]))
    a("")

    a(f"## 6. Layers that are not collapsed (contrast), {k_word}")
    rest = head[~head["collapsed"]]
    keys = (("loading of PC1/2/3", [f"load_pc{j}" for j in (1, 2, 3)], ".3f"),
            ("variance left along PC1/2/3", [f"var_left_pc{j}" for j in (1, 2, 3)], ".3f"),
            ("angle old/new PC1, degrees", ["angle_pc1"], ".1f"),
            ("largest principal angle, degrees", [f"pangle_{N_TOP}"], ".1f"),
            ("remainder cosine of PC1", ["remainder_cos_pc1"], ".3f"),
            ("share of the zeroed variance along the remainder of PC1/2/3",
             [f"remainder_share_pc{j}" for j in (1, 2, 3)], ".3f"),
            ("|Pearson| old/new PC1 scores, train", ["abs_pearson_train_pc1"], ".3f"),
            ("top-PC share before zeroing", ["ref_var_share_pc1"], ".3f"),
            ("top-PC share after zeroing", ["pc1_share_train"], ".3f"),
            ("AUROC base", ["ref_auc_base"], ".3f"), ("zero only", ["auc_zero"], ".3f"),
            ("zero + ABTT D=1/2/3", [f"auc_zero_abtt_D{D}" for D in ZERO_ABTT_D], ".3f"),
            ("PCs 2 and 3 removed, PC1 kept", ["ref_auc_rm_pc23"], ".3f"),
            ("ABTT D=1/2/3", [f"ref_auc_abtt_D{D}" for D in (1, 2, 3)], ".3f"))
    for label, t in [(f"all layers that are not collapsed ({len(rest)})", rest)] + [
            (f"{m} ({len(g)} layers: {', '.join(str(int(v)) for v in sorted(g['layer']))})", g)
            for m, g in _by_model(rest)]:
        if t.empty:
            continue
        a(f"- {label}:")
        for name, cs, fmt in keys:
            a(f"  - {name}: " + " / ".join(_rng(t[c], fmt) for c in cs))
        a(f"  - PC1 loading below {LOADING_BAR} at {_cnt(t['load_pc1'] < LOADING_BAR)}; "
          f"variance left along PC1 above {VAR_LEFT_BAR:.0%} at "
          f"{_cnt(t['var_left_pc1'] > VAR_LEFT_BAR)}; angle old/new PC1 below "
          f"{ANGLE_BAR:.0f} degrees at {_cnt(t['angle_pc1'] < ANGLE_BAR)}; |Pearson| of PC1 "
          f"scores (train) above {CORR_BAR} at "
          f"{_cnt(t['abs_pearson_train_pc1'] > CORR_BAR)}")
    a("")

    a(f"## 7. All layers, {k_word}")
    a("Columns: base AUROC; load j = loading of PC j on the zeroed coordinates; left j = "
      "variance left along PC j; ang1 = angle between old and new PC1 (degrees); pang3 = "
      "largest principal angle between the spans of old and new PCs 1 to 3 (degrees); rem = "
      "remainder cosine of PC1; rsh1 = share of the zeroed training variance along what is "
      "left of PC1 outside the zeroed coordinates, renormalized; r1 = |Pearson| of old and "
      "new PC1 scores on train passages; "
      "share = top-PC share of the zeroed training vectors; AUROC after: zero only; zero + "
      "ABTT D=1, 2, 3; PCs 2 and 3 removed with PC1 kept; ABTT D=1, 2, 3, 10 on the "
      "unzeroed vectors. * marks a collapsed layer.")
    for name in present:
        a(f"### {name}")
        L.extend(_md_table(head[head["m"] == name]))
        a("")
    path.write_text("\n".join(L) + "\n")


def cmd_render(args) -> int:
    rows = read_rows(args.out_dir / CSV_NAME)
    d = prepare(rows)
    for name in [m for m in ORDER if not (d["m"] == m).any()]:
        print(f"omitting {name}: no rows in {args.out_dir / CSV_NAME}")
    gates = None
    if args.e1_csv.exists() and args.h1_csv.exists():
        gates = gates_for(rows, args)
    conc = pd.read_csv(args.e1_conc_csv) if args.e1_conc_csv.exists() else None
    args.facts_md.parent.mkdir(parents=True, exist_ok=True)
    facts(d, gates, args.facts_md, conc=conc)
    print(f"wrote {args.facts_md}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def refs(p) -> None:
        p.add_argument("--out_dir", type=Path, default=OUT_DIR)
        p.add_argument("--e1_csv", type=Path, default=E1_CSV)
        p.add_argument("--h1_csv", type=Path, default=H1_CSV)
        p.add_argument("--tol", type=float, default=GATE_TOL,
                       help="tolerance of gates 2, 3 and 4 (default 1e-6)")
        p.add_argument("--models", default="",
                       help="comma list of ids or display names (default all six); the "
                            "gates fail if one of them has no rows")
        p.add_argument("--allow_missing", action="store_true",
                       help="compute: skip a model whose cache is missing instead of failing; "
                            "gates: do not require every model to have rows")

    p = sub.add_parser("compute", help="measure, score and write the result CSV")
    refs(p)
    p.add_argument("--split_csv", type=Path, default=SPLIT_CSV)
    p.add_argument("--bases_root", type=Path, default=BASES_ROOT)
    p.add_argument("--layers", default="", help="comma list of layers (default all)")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--check", action="store_true",
                   help="run the reproduction gates after writing; exit 3 if one fails")
    p = sub.add_parser("check", help="reproduction gates on an existing CSV")
    refs(p)
    p.add_argument("--no_write", action="store_true", help=f"do not rewrite {GATE_NAME}")
    p = sub.add_parser("render", help="facts file")
    refs(p)
    p.add_argument("--e1_conc_csv", type=Path, default=E1_CONC_CSV)
    p.add_argument("--facts_md", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "compute":
        return cmd_compute(args)
    if args.cmd == "check":
        return cmd_check(args)
    if args.facts_md is None:
        args.facts_md = args.out_dir / FACTS_NAME
    return cmd_render(args)


if __name__ == "__main__":
    sys.exit(main())
