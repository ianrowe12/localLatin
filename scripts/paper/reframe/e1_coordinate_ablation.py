#!/usr/bin/env python3
"""Reframe experiment E1 (issue #246): coordinate ablation on cached pooled vectors.

Question: do a few residual coordinates of the mean-pooled passage vectors explain the
collapse of retrieval at mid-depth T5 layers? At every layer of the six panel models the
script edits the raw pooled vectors (``hidden_mean_tokempty``) and rescores them.

  compute  per model-layer, every statistic fit on the 847 TRAIN passages only:
             base          the unmodified vectors;
             zero          the top k coordinates set to 0, k in {1,3,5,10}, ranked
                           (a) by mean |x| over training passages ("mean_abs") and
                           (b) by variance across training passages ("variance"), both
                           on the raw vectors (not normalized, not centered);
             standardize   (x - train mean) / train SD per coordinate;
             center        x - train mean (ABTT with D=0);
             abtt          ABTT with D in {1,3,10} (D=10 is the gain denominator).
           Readouts: the paper's metric block (Task A test and train AUROC through
           run_resubmit_evaluate.evaluate_from_similarity, exactly as
           abtt_subspace_whiten._metrics) and the top-PC share and effective rank of the
           intervened TRAIN vectors (run_layer_geometry_diagnostics.pca_stats, the
           function behind the "raw" view of geometry_per_layer.csv).
           Also: per-coordinate statistics of the top 10 coordinates under each ranking
           (mean, mean |x|, SD, variance, r = SD / |mean|), and the share of the pooled
           cosine carried by the top k coordinates (Timkey and van Schijndel 2021).
  check    the three reproduction gates on an existing CSV (no caches needed).
  render   the paper table and a facts file for the prose.

Order of operations: the intervention acts on the raw vectors; L2 normalization happens
afterwards inside the metric. Nothing is centered before zeroing.

Cosine shares. With U the L2-normalized raw vectors of one split, the contribution of
coordinate k to cos(a, b) is u_ak u_bk. Its sum over the distinct pairs is
((sum_i u_ik)^2 - sum_i u_ik^2) / 2, and the sum of that over all coordinates is the sum
of the pairwise cosines. The share of a coordinate set is its summed contribution divided
by the summed cosine. Coordinates with a negative contribution exist, so a share can
exceed 1. Coordinates are always chosen on train; shares are computed on train pairs and
on test pairs.

Reproduction gates (``compute --check`` or ``check``; exit status 3 on failure, after the
CSVs are written):
  1. base AUROC = the baseline cells (repr hidden, pooling mean) of
     runs/active/resubmit/results/phase_resubmit_results.csv, within 1e-6, every layer;
  2. center and ABTT D=1, 3, 10 = runs/active/reframe/h1/h1_d_ablation.csv within 1e-6;
  3. base top-PC share = the train raw-view value of
     runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv within 1e-4.

Outputs (small CSVs, force-added; embeddings are never written):
  runs/active/reframe/e1/e1_coordinate_ablation.csv   long format, one row per
                                                      (model, layer, intervention)
  runs/active/reframe/e1/e1_top_coordinates.csv
  runs/active/reframe/e1/e1_cosine_shares.csv
  runs/active/reframe/e1/e1_gate_check.csv
  runs/active/reframe/e1/facts_e1.md                  (render)
  overleaf_drafts/tables/e1_coordinate_ablation.tex   (tab:e1_coordinate_ablation)

Run from the repo root. The embedding caches are gitignored, so point --bases_root at a
checkout that has them; a model with no cache is an error unless --allow_missing:
  python scripts/paper/reframe/e1_coordinate_ablation.py compute --check --workers 16 \
      --bases_root /u/irowerojas/localLatin/runs/active/resubmit_bases
  python scripts/paper/reframe/e1_coordinate_ablation.py render
CPU only. Python 3.10, numpy / pandas / scikit-learn.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "resubmit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import abtt_subspace_whiten as asw  # noqa: E402
from canon_retrieval import l2_normalize  # noqa: E402
from run_layer_geometry_diagnostics import pca_stats  # noqa: E402

SPLIT_CSV = asw.SPLIT_CSV
RES_CSV = asw.RES_CSV
BASES_ROOT = asw.BASES_ROOT
H1_CSV = asw.H1_CSV
GEOM_CSV = Path("runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv")
SELECTED_TEX = Path("overleaf_drafts/tables/selected_layers.tex")
OUT_DIR = asw.OUT_ROOT / "e1"
TAB_DIR = asw.TAB_DIR
ABL_NAME = "e1_coordinate_ablation.csv"
COORD_NAME = "e1_top_coordinates.csv"
SHARE_NAME = "e1_cosine_shares.csv"
GATE_NAME = "e1_gate_check.csv"
FACTS_NAME = "facts_e1.md"
TABLE_NAME = "e1_coordinate_ablation.tex"

MODELS = asw.MODELS
DISP = {m[0]: m[1] for m in MODELS}
ORDER = [m[1] for m in MODELS]
T5 = [m[1] for m in MODELS if m[2]]
NON_T5 = [m[1] for m in MODELS if not m[2]]

RANKINGS = ("mean_abs", "variance")
RANK_LABEL = {"mean_abs": "mean |x|", "variance": "variance"}
KS = (1, 3, 5, 10)
ABTT_D = (1, 3, 10)
TOP_N = 10  # coordinates listed per ranking in e1_top_coordinates.csv
SD_DDOF = 0  # population SD and variance over the training passages

# Thresholds of the paper paragraph, fixed before any result was read.
COLLAPSE_AUROC = asw.COLLAPSE_AUROC  # collapsed = baseline test AUROC below 0.70
RESTORE_AUROC = 0.90  # "restores AUROC ... to at least 0.90"
RESTORE_K = 5  # "zeroing the top k <= 5 coordinates"
SHARE_BAR = 0.2  # "top-PC share < 0.2"
STABLE_DELTA = 0.03  # embedding-trained models: |change| at most 0.03
GATE_TOL_AUROC = 1e-6
GATE_TOL_PC1 = 1e-4
GATE_EXIT = 3  # exit status of a failed gate: the CSVs are written, the numbers disagree


def zero_tag(ranking: str, k: int) -> str:
    return f"zero_{ranking}_k{k}"


# Column order of the paper table: Base, zeroed by mean |x|, zeroed by variance, Std.,
# D=0, ABTT D=1, ABTT D=3. abtt_D10 is computed for the gain fractions only.
TABLE_TAGS = (["base"] + [zero_tag("mean_abs", k) for k in KS]
              + [zero_tag("variance", k) for k in KS]
              + ["standardize", "center", "abtt_D1", "abtt_D3"])
ALL_TAGS = TABLE_TAGS + ["abtt_D10"]
TAG_LABEL = {"base": "base", "standardize": "std", "center": "D=0", "abtt_D1": "ABTT D=1",
             "abtt_D3": "ABTT D=3", "abtt_D10": "ABTT D=10",
             **{zero_tag(r, k): f"zero {RANK_LABEL[r]} k={k}" for r in RANKINGS for k in KS}}


# --------------------------------------------------------------------------- #
# Pure functions (unit-tested on synthetic arrays)
# --------------------------------------------------------------------------- #

def coord_stats(train: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-coordinate statistics of the raw TRAIN vectors (float64).

    r = SD / |mean|: how much a coordinate varies across passages relative to how far it
    shifts all of them together. A coordinate with mean 0 gets r = inf.
    """
    x = np.asarray(train, dtype=np.float64)
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=SD_DDOF)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(mean) > 0, sd / np.abs(mean), np.inf)
    return {"mean": mean, "mean_abs": np.abs(x).mean(axis=0), "sd": sd,
            "variance": x.var(axis=0, ddof=SD_DDOF), "r": r}


def rank_coords(train: np.ndarray, ranking: str) -> np.ndarray:
    """All coordinate indices, largest score first, from TRAIN vectors only.

    ``mean_abs`` scores a coordinate by its mean absolute value over training passages,
    ``variance`` by its variance across them. Ties keep the lower index first.
    """
    if ranking not in RANKINGS:
        raise ValueError(f"unknown ranking {ranking!r}; expected one of {RANKINGS}")
    return np.argsort(-coord_stats(train)[ranking], kind="stable")


def zero_coords(train: np.ndarray, test: np.ndarray, idx: Sequence[int]
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Copies of both splits with the listed coordinates set to zero."""
    idx = np.asarray(idx, dtype=np.int64)
    a, b = np.array(train, copy=True), np.array(test, copy=True)
    a[:, idx] = 0
    b[:, idx] = 0
    return a, b


def standardize(train: np.ndarray, test: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, int]:
    """(x - train mean) / train SD per coordinate, applied to both splits.

    A coordinate with zero training SD is constant on train; it is divided by 1, so it is
    0 on train and keeps its offset from the train mean on test. Returns the number of
    such coordinates as the third value.
    """
    x = np.asarray(train, dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=SD_DDOF)
    flat = ~(sd > 0)
    div = np.where(flat, 1.0, sd)
    return (x - mu) / div, (np.asarray(test, dtype=np.float64) - mu) / div, int(flat.sum())


def cosine_contributions(x: np.ndarray) -> Tuple[np.ndarray, int]:
    """Summed contribution of every coordinate to the pairwise cosines of one split.

    Returns (c, n_pairs) with c[k] = sum over pairs i<j of u_ik u_jk, U the L2-normalized
    rows of x, in closed form. c.sum() is the sum of the pairwise cosines.
    """
    u = l2_normalize(np.asarray(x, dtype=np.float64))
    n = u.shape[0]
    c = (u.sum(axis=0) ** 2 - (u ** 2).sum(axis=0)) / 2.0
    return c, n * (n - 1) // 2


def cosine_share(contrib: np.ndarray, idx: Sequence[int]) -> float:
    """Share of the summed pairwise cosine carried by the coordinates in ``idx``."""
    total = float(contrib.sum())
    if total == 0.0:
        return float("nan")
    return float(contrib[np.asarray(idx, dtype=np.int64)].sum() / total)


def _coords_str(idx: Sequence[int]) -> str:
    return ";".join(str(int(i)) for i in idx)


def _geometry(train: np.ndarray) -> Dict[str, float]:
    g = pca_stats(train)
    return {"pc1_share_train": g["pc1_variance_ratio"],
            "pc10_share_train": g["pc10_cumulative_variance_ratio"],
            "eff_rank_train": g["effective_rank_entropy"]}


def layer_rows(model_id: str, layer: int, tr: np.ndarray, te: np.ndarray,
               metrics_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]]
               ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """All E1 rows of one model-layer: (ablation, top-coordinate, cosine-share) rows.

    ``tr`` and ``te`` are the raw pooled vectors as cached. ``metrics_fn(train, test)``
    returns the metric block of one intervention (it L2-normalizes internally).
    """
    key = {"model": model_id, "layer": int(layer)}
    stats = coord_stats(tr)
    order = {r: np.argsort(-stats[r], kind="stable") for r in RANKINGS}

    abl: List[Dict] = []

    def add(tag: str, intervention: str, a: np.ndarray, b: np.ndarray, ranking: str = "",
            k: int = -1, D: int = -1, coords: str = "", n_zero_sd: int = -1) -> None:
        m = metrics_fn(a, b)
        abl.append({**key, "tag": tag, "intervention": intervention, "ranking": ranking,
                    "k": k, "D": D, "coords": coords, "n_zero_sd": n_zero_sd,
                    **{c: m[c] for c in asw.KEEP if c in m}, **_geometry(a)})

    add("base", "base", tr, te)
    for ranking in RANKINGS:
        for k in KS:
            idx = order[ranking][:k]
            a, b = zero_coords(tr, te, idx)
            add(zero_tag(ranking, k), "zero", a, b, ranking=ranking, k=k,
                coords=_coords_str(idx))
    a, b, n_flat = standardize(tr, te)
    add("standardize", "standardize", a, b, n_zero_sd=n_flat)
    a, b = asw.center(tr, te)
    add("center", "center", a, b, D=0)
    for D in ABTT_D:
        a, b = asw.abtt(tr, te, D)
        add(f"abtt_D{D}", "abtt", a, b, D=D)

    # top coordinates under each ranking, with their rank under the other one
    rank_of = {r: np.empty(len(order[r]), dtype=np.int64) for r in RANKINGS}
    for r in RANKINGS:
        rank_of[r][order[r]] = np.arange(1, len(order[r]) + 1)
    layer_info = {"dim": int(tr.shape[1]),
                  "median_mean_abs": float(np.median(stats["mean_abs"])),
                  "median_abs_mean": float(np.median(np.abs(stats["mean"]))),
                  "median_sd": float(np.median(stats["sd"]))}
    coords: List[Dict] = []
    for r in RANKINGS:
        other = RANKINGS[1 - RANKINGS.index(r)]
        for pos, i in enumerate(order[r][:TOP_N], start=1):
            coords.append({**key, "ranking": r, "rank": pos, "coord": int(i),
                           "mean": float(stats["mean"][i]),
                           "mean_abs": float(stats["mean_abs"][i]),
                           "sd": float(stats["sd"][i]),
                           "variance": float(stats["variance"][i]),
                           "r": float(stats["r"][i]),
                           "rank_other": int(rank_of[other][i]), **layer_info})

    shares: List[Dict] = []
    for split, x in (("train", tr), ("test", te)):
        c, n_pairs = cosine_contributions(x)
        total = float(c.sum())
        neg = c < 0
        for r in RANKINGS:
            for k in KS:
                idx = order[r][:k]
                shares.append({**key, "ranking": r, "k": k, "split": split,
                               "coords": _coords_str(idx),
                               "share": cosine_share(c, idx),
                               "topk_mean_contribution": float(c[idx].sum() / n_pairs),
                               "mean_cosine": total / n_pairs,
                               "n_negative_topk": int(neg[idx].sum()),
                               "n_negative_all": int(neg.sum()),
                               "negative_share": (float(c[neg].sum() / total)
                                                  if total != 0.0 else float("nan"))})
    return abl, coords, shares


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #

def task_e1(args) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    bases_root, model_id, layer = args
    t0 = time.time()
    tr, te = asw._load(bases_root, asw.slug(model_id), layer)
    out = layer_rows(model_id, layer, tr, te, asw._metrics)
    a = {r["tag"]: r["aucroc"] for r in out[0]}
    print(f"  e1 {DISP.get(model_id, model_id)} L{layer}: base {a['base']:.3f} "
          f"|x| k5 {a[zero_tag('mean_abs', 5)]:.3f} var k5 {a[zero_tag('variance', 5)]:.3f} "
          f"std {a['standardize']:.3f} D1 {a['abtt_D1']:.3f} D10 {a['abtt_D10']:.3f} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return out


def build_tasks(bases_root: Path, model_ids: Sequence[str], layers: Optional[List[int]],
                allow_missing: bool) -> List[Tuple[str, str, int]]:
    tasks = []
    for mid in model_ids:
        found = asw.discover(bases_root, mid)
        if not found:
            where = bases_root / "phase9_bases" / asw.slug(mid) / asw.SUBDIR
            if not allow_missing:
                raise SystemExit(
                    f"ERROR: no cached vectors for {DISP.get(mid, mid)} under {where}. "
                    "Extract them first, or pass --allow_missing to skip this model.")
            print(f"WARNING: skipping {DISP.get(mid, mid)}: no cached vectors under {where}",
                  flush=True)
            continue
        tasks += [(str(bases_root), mid, layer) for layer in found
                  if layers is None or layer in layers]
    return tasks


def _sorted(df: pd.DataFrame, extra: Sequence[str]) -> pd.DataFrame:
    order = {m[0]: i for i, m in enumerate(MODELS)}
    df = df.assign(_m=df["model"].map(order), _i=np.arange(len(df)))
    return (df.sort_values(["_m", "layer", *extra, "_i"], kind="stable")
            .drop(columns=["_m", "_i"]).reset_index(drop=True))


def cmd_compute(args) -> int:
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    model_ids = asw.pick_models(args.models, MODELS)
    if not model_ids:
        raise SystemExit(f"--models {args.models!r} matches none of {ORDER}")
    t0 = time.time()
    tasks = build_tasks(args.bases_root, model_ids, layers, args.allow_missing)
    if not tasks:
        raise SystemExit("ERROR: nothing to compute (no model has cached vectors)")
    abl: List[Dict] = []
    coords: List[Dict] = []
    shares: List[Dict] = []
    if args.workers <= 1:
        asw._init(str(args.split_csv))
        parts = map(task_e1, tasks)
    else:
        from multiprocessing import Pool

        pool = Pool(args.workers, initializer=asw._init, initargs=(str(args.split_csv),))
        parts = pool.imap(task_e1, tasks, chunksize=1)
    for a, c, s in parts:
        abl.extend(a)
        coords.extend(c)
        shares.extend(s)
    if args.workers > 1:
        pool.close()
        pool.join()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    abl_df = _sorted(pd.DataFrame(abl), [])
    abl_df.to_csv(args.out_dir / ABL_NAME, index=False, float_format="%.10g")
    _sorted(pd.DataFrame(coords), []).to_csv(args.out_dir / COORD_NAME, index=False,
                                             float_format="%.10g")
    _sorted(pd.DataFrame(shares), []).to_csv(args.out_dir / SHARE_NAME, index=False,
                                             float_format="%.10g")
    print(f"wrote {args.out_dir}/{{{ABL_NAME},{COORD_NAME},{SHARE_NAME}}} "
          f"({len(abl_df)} ablation rows, {len(tasks)} model-layers) in {time.time() - t0:.0f}s")
    if not args.check:
        return 0
    complete = layers is None
    return run_gates(abl_df, args, complete=complete, write=True)


# --------------------------------------------------------------------------- #
# Reproduction gates
# --------------------------------------------------------------------------- #

def gate_table(abl: pd.DataFrame, res: pd.DataFrame, h1: pd.DataFrame, geom: pd.DataFrame,
               complete: bool = True) -> pd.DataFrame:
    """One row per (gate, model): cells compared, cells missing, max |difference|.

    ``complete`` also requires every published layer of a computed model to be present
    (turn it off for a --layers subset).
    """
    ours = abl.set_index(["model", "layer", "tag"])
    pub = res[(res["repr"] == "hidden") & (res["pooling"] == "mean")
              & (res["method"] == "baseline")].set_index(["model", "layer"])["aucroc"]
    h1i = h1.set_index(["model", "layer", "variant", "D"])["aucroc"]
    g = geom[(geom["split"] == "train") & (geom["view"] == "raw") & (geom["pooling"] == "mean")]
    gi = g.set_index(["model", "layer"])["pc1_variance_ratio"]
    h1_keys = {"center": ("center", 0), "abtt_D1": ("abtt", 1), "abtt_D3": ("abtt", 3),
               "abtt_D10": ("abtt", 10)}
    rows = []
    for mid in [m[0] for m in MODELS if m[0] in set(abl["model"])]:
        mine = sorted(abl.loc[abl["model"] == mid, "layer"].unique())
        published = sorted(pub.loc[mid].index) if mid in pub.index.get_level_values(0) else []
        absent = [x for x in published if x not in mine] if complete else []
        specs = [
            ("1 base AUROC vs published baseline", GATE_TOL_AUROC,
             [(float(ours.loc[(mid, x, "base"), "aucroc"]), pub.get((mid, x))) for x in mine]),
            ("2 center and ABTT D=1,3,10 AUROC vs H1", GATE_TOL_AUROC,
             [(float(ours.loc[(mid, x, tag), "aucroc"]), h1i.get((mid, x, v, D)))
              for x in mine for tag, (v, D) in h1_keys.items()]),
            ("3 base top-PC share vs geometry_per_layer (train, raw)", GATE_TOL_PC1,
             [(float(ours.loc[(mid, x, "base"), "pc1_share_train"]), gi.get((mid, x)))
              for x in mine]),
        ]
        for name, tol, cells in specs:
            diffs = [abs(a - float(b)) for a, b in cells if b is not None]
            n_missing = sum(1 for _, b in cells if b is None)
            mx = max(diffs) if diffs else float("nan")
            ok = bool(diffs) and n_missing == 0 and not absent and mx <= tol
            rows.append({"gate": name, "model": mid, "n_cells": len(diffs),
                         "n_missing_reference": n_missing,
                         "n_published_layers_absent": len(absent),
                         "max_abs_diff": mx, "tolerance": tol, "ok": ok})
    return pd.DataFrame(rows)


def run_gates(abl: pd.DataFrame, args, complete: bool, write: bool) -> int:
    gates = gate_table(abl, pd.read_csv(args.results_csv), pd.read_csv(args.h1_csv),
                       pd.read_csv(args.geom_csv), complete=complete)
    if write:
        gates.to_csv(args.out_dir / GATE_NAME, index=False, float_format="%.6g")
    for g in gates.itertuples():
        print(f"gate {g.gate} | {DISP.get(g.model, g.model)}: {g.n_cells} cells, "
              f"max |diff| {g.max_abs_diff:.2e} (tol {g.tolerance:.0e}), "
              f"missing reference {g.n_missing_reference}, published layers absent "
              f"{g.n_published_layers_absent}: {'ok' if g.ok else 'FAIL'}")
    if gates.empty or not gates["ok"].all():
        print("REPRODUCTION GATES FAILED")
        return GATE_EXIT
    print("reproduction gates passed")
    return 0


def cmd_check(args) -> int:
    abl = pd.read_csv(args.out_dir / ABL_NAME, keep_default_na=False,
                      na_values=["nan", "NaN"])
    return run_gates(abl, args, complete=True, write=not args.no_write)


# --------------------------------------------------------------------------- #
# render: table
# --------------------------------------------------------------------------- #

def read_abl(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=["nan", "NaN"])


def wide(abl: pd.DataFrame) -> pd.DataFrame:
    """One row per model-layer: auc_<tag>, trauc_<tag>, pc1_<tag>, erank_<tag>."""
    rows = []
    for (mid, layer), g in abl.groupby(["model", "layer"], sort=False):
        g = g.set_index("tag")
        r = {"model": mid, "m": DISP.get(mid, mid), "layer": int(layer)}
        for tag in ALL_TAGS:
            if tag not in g.index:
                continue
            x = g.loc[tag]
            r[f"auc_{tag}"], r[f"trauc_{tag}"] = float(x["aucroc"]), float(x["train_aucroc"])
            r[f"pc1_{tag}"], r[f"erank_{tag}"] = (float(x["pc1_share_train"]),
                                                  float(x["eff_rank_train"]))
            if x["coords"] != "":
                r[f"coords_{tag}"] = str(x["coords"])
        rows.append(r)
    w = pd.DataFrame(rows)
    w["is_t5"] = w["m"].isin(T5)
    w["collapsed"] = w["is_t5"] & (w["auc_base"] < COLLAPSE_AUROC)
    return w


def worst_layer(w: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """The model's row at its lowest base test AUROC (first layer on ties)."""
    s = w[w["m"] == name].sort_values("layer")
    return None if s.empty else s.loc[s["auc_base"].idxmin()]


def selected_layer(w: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """The model's row at the baseline train-selected layer: first argmax train AUROC."""
    s = w[w["m"] == name].sort_values("layer")
    return None if s.empty else s.loc[s["trauc_base"].idxmax()]


def f3(x) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


CAPTION = (
    r"\caption{Coordinate ablation at each model's worst baseline layer (L). Top block: "
    r"Task~A test AUROC for the unmodified mean-pooled vectors (Base); after zeroing the $k$ "
    r"residual coordinates with the largest mean absolute training value, or with the largest "
    r"variance across training passages; after per-coordinate standardization with training "
    r"statistics (Std.); after centering alone ($D{=}0$); and after ABTT with one or three "
    r"components (ABTT$_{D=1}$, ABTT$_{D=3}$), the two reference repairs. Bottom block: top-PC "
    r"share of the training vectors after the same interventions, the share of their centered "
    r"variance on the first principal component; centering leaves it unchanged by definition. "
    r"At the collapsed T5 layers, one component recovers a median 45 percent of the AUROC gain "
    r"of $D{=}10$, and three recover at least 80 percent at every layer "
    r"(Table~\ref{tab:d_ablation}). All statistics are fit on training embeddings only.}")


def write_table(w: pd.DataFrame, path: Path) -> List[str]:
    """Write tab:e1_coordinate_ablation. Returns the display names of omitted models."""
    ncol = 2 + len(TABLE_TAGS)
    rows = [(name, worst_layer(w, name)) for name in ORDER]
    omitted = [name for name, x in rows if x is None]
    rows = [(name, x) for name, x in rows if x is not None]
    lines = [asw.HEADER, r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccccccccccccc@{}}",
             r"\toprule",
             r" & & & \multicolumn{4}{c}{Zeroed, ranked by mean $|x|$} & "
             r"\multicolumn{4}{c}{Zeroed, ranked by variance} & & & & \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
             r"Model & L & Base & 1 & 3 & 5 & 10 & 1 & 3 & 5 & 10 & Std. & $D{=}0$ & "
             r"ABTT$_{D=1}$ & ABTT$_{D=3}$ \\",
             r"\midrule"]
    for i, (title, prefix) in enumerate((("Task~A test AUROC", "auc"),
                                         ("Top-PC share of the training vectors", "pc1"))):
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for name, x in rows:
            lines.append(f"{name} & {int(x['layer'])} & "
                         + " & ".join(f3(x.get(f"{prefix}_{tag}")) for tag in TABLE_TAGS)
                         + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", CAPTION, r"\label{tab:e1_coordinate_ablation}",
              r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")
    return omitted


# --------------------------------------------------------------------------- #
# render: facts
# --------------------------------------------------------------------------- #

def first_k(x: pd.Series, ranking: str, prefix: str, passes: Callable[[float], bool],
            ks: Sequence[int] = KS) -> Optional[int]:
    """Smallest k whose cell satisfies ``passes``, or None."""
    for k in ks:
        v = x.get(f"{prefix}_{zero_tag(ranking, k)}")
        if v is not None and np.isfinite(v) and passes(v):
            return k
    return None


def _k_str(k: Optional[int]) -> str:
    return f"k={k}" if k is not None else f"none up to {KS[-1]}"


def parse_selected_layers(path: Path) -> Dict[str, int]:
    """Base Task A layer per model from the generated tab:selected_layers."""
    out: Dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9.\- ]+?) & (\d+) & ", line)
        if m and m.group(1) in ORDER:
            out[m.group(1)] = int(m.group(2))
    return out


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _auc_line(x: pd.Series, prefix: str = "auc") -> str:
    za = " / ".join(f3(x.get(f"{prefix}_{zero_tag('mean_abs', k)}")) for k in KS)
    zv = " / ".join(f3(x.get(f"{prefix}_{zero_tag('variance', k)}")) for k in KS)
    return (f"base {f3(x.get(f'{prefix}_base'))}; zero mean |x| k=1/3/5/10 {za}; zero variance "
            f"k=1/3/5/10 {zv}; std {f3(x.get(f'{prefix}_standardize'))}; D=0 "
            f"{f3(x.get(f'{prefix}_center'))}; ABTT D=1 {f3(x.get(f'{prefix}_abtt_D1'))}; D=3 "
            f"{f3(x.get(f'{prefix}_abtt_D3'))}; D=10 {f3(x.get(f'{prefix}_abtt_D10'))}")


def _md_table(w: pd.DataFrame, name: str, prefix: str) -> List[str]:
    tags = ALL_TAGS
    head = ["L", "base"] + [f"mag {k}" for k in KS] + [f"var {k}" for k in KS] + [
        "std", "D=0", "D=1", "D=3", "D=10"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, x in w[w["m"] == name].sort_values("layer").iterrows():
        mark = "*" if x["collapsed"] else ""
        out.append(f"| {int(x['layer'])}{mark} | "
                   + " | ".join(f3(x.get(f"{prefix}_{t}")) for t in tags) + " |")
    return out


def _fmt(v: float) -> str:
    """Compact number for coordinate statistics of very different scales."""
    if not np.isfinite(v):
        return "inf" if v > 0 else "nan"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def coord_lines(coords: pd.DataFrame, shares: pd.DataFrame, mid: str, layer: int) -> List[str]:
    """Cosine shares and r of the top coordinates of one model-layer, both rankings."""
    out = []
    c = coords[(coords["model"] == mid) & (coords["layer"] == layer)]
    s = shares[(shares["model"] == mid) & (shares["layer"] == layer)]
    if c.empty or s.empty:
        return [f"  - (no rows for {DISP.get(mid, mid)} layer {layer})"]
    tr0 = s[s["split"] == "train"].iloc[0]
    te0 = s[s["split"] == "test"].iloc[0]
    out.append(f"  - mean pairwise cosine: train {tr0['mean_cosine']:.3f}, test "
               f"{te0['mean_cosine']:.3f}; coordinates with a negative summed contribution: "
               f"train {int(tr0['n_negative_all'])}, test {int(te0['n_negative_all'])} of "
               f"{int(c['dim'].iloc[0])} (their summed share: train {tr0['negative_share']:+.3f}, "
               f"test {te0['negative_share']:+.3f}); median over coordinates of mean |x| "
               f"{_fmt(c['median_mean_abs'].iloc[0])}, of SD {_fmt(c['median_sd'].iloc[0])}")
    for r in RANKINGS:
        cr = c[c["ranking"] == r].sort_values("rank")
        top = "; ".join(f"#{int(x.coord)} mean {_fmt(x['mean'])}, mean |x| {_fmt(x.mean_abs)}, "
                        f"SD {_fmt(x.sd)}, r {x.r:.3f}" for _, x in cr.head(3).iterrows())
        r10 = cr["r"].to_numpy()
        sh = []
        for k in KS:
            a = s[(s["ranking"] == r) & (s["k"] == k) & (s["split"] == "train")].iloc[0]
            b = s[(s["ranking"] == r) & (s["k"] == k) & (s["split"] == "test")].iloc[0]
            sh.append(f"k={k} {a['share']:.3f} ({b['share']:.3f})")
        out.append(f"  - by {RANK_LABEL[r]}: top 3 = {top}. r >= 1 among the top 1/3/10: "
                   f"{int((r10[:1] >= 1).sum())}/1, {int((r10[:3] >= 1).sum())}/3, "
                   f"{int((r10[:10] >= 1).sum())}/{min(10, len(r10))}. Cosine share, train "
                   f"(test): " + ", ".join(sh))
    return out


def top3_r(coords: pd.DataFrame, mid: str, layer: int, ranking: str, n: int = 3) -> np.ndarray:
    c = coords[(coords["model"] == mid) & (coords["layer"] == layer)
               & (coords["ranking"] == ranking)].sort_values("rank")
    return c["r"].to_numpy()[:n]


def facts(w: pd.DataFrame, coords: pd.DataFrame, shares: pd.DataFrame,
          gates: Optional[pd.DataFrame], selected_tex: Dict[str, int], path: Path,
          omitted: Sequence[str] = ()) -> None:
    L: List[str] = []
    a = L.append
    present = [m for m in ORDER if (w["m"] == m).any()]
    coll = w[w["collapsed"]]
    n_coll = len(coll)

    a("# E1 coordinate ablation: facts (generated)")
    a("")
    a("Generated by `scripts/paper/reframe/e1_coordinate_ablation.py render` from "
      f"`{ABL_NAME}`, `{COORD_NAME}` and `{SHARE_NAME}` in this directory. Every number below "
      "is a cell of one of those CSVs or a difference or count of such cells. AUROC is Task A "
      "test AUROC unless marked train. Nothing here was tuned after the results were read: "
      f"k in {list(KS)}, the two rankings and the thresholds ({RESTORE_AUROC:.2f} AUROC, "
      f"k <= {RESTORE_K}, top-PC share < {SHARE_BAR:.1f}, |change| <= {STABLE_DELTA:.2f}, collapsed = "
      f"baseline AUROC < {COLLAPSE_AUROC:.2f}) are the ones in the Sec. 5 paragraph.")
    a("")
    a("## Definitions")
    a("- Vectors: mean-pooled hidden states (`hidden_mean_tokempty`), raw as cached, from the "
      "cache passed as `--bases_root`. On Delta these are James's re-extraction, not Ian's "
      "files, so the reproduction gates below show differences near 1e-7 and not exactly 0.")
    a("- Every statistic (coordinate ranking, mean, SD, principal components) is fit on the "
      "847 training passages and applied to train and test. The intervention acts on the raw "
      "vectors; L2 normalization happens afterwards inside the metric. Nothing is centered "
      "before zeroing.")
    a("- Rankings: `mean |x|` = mean absolute value of a coordinate over training passages; "
      "`variance` = its variance across training passages. Both on the raw vectors.")
    a(f"- r = SD / |mean| of a coordinate over training passages (population SD, ddof={SD_DDOF}).")
    a("- Standardization: (x - train mean) / train SD per coordinate.")
    a("- D=0 is centering on the train mean; ABTT D removes the top D principal components of "
      "the centered training vectors (the functions of the H1 script).")
    a("- Top-PC share and effective rank: `pca_stats` of "
      "`scripts/resubmit/run_layer_geometry_diagnostics.py` on the intervened TRAIN vectors "
      "(re-centered on their own mean, not normalized), the definition of the raw view of "
      "`geometry_per_layer.csv`. Centering therefore leaves the share equal to the base share "
      "by construction; that column is not a finding.")
    a("- Cosine share (Timkey and van Schijndel 2021): summed contribution u_ik u_jk of the top "
      "k coordinates over all distinct passage pairs of a split, divided by the summed pairwise "
      "cosine of that split, on the L2-normalized raw vectors. Coordinates are chosen on train; "
      "the share is given for train pairs and, in parentheses, for test pairs. A share can "
      "exceed 1 when other coordinates contribute negatively.")
    a("- Worst layer = first argmin of base test AUROC. Train-selected layer = first argmax of "
      "base train AUROC.")
    a("")

    a("## 0. Coverage and reproduction gates")
    a("- model-layers: " + ", ".join(f"{m} {int((w['m'] == m).sum())}" for m in present)
      + f" (total {len(w)})")
    if omitted:
        a("- ABSENT from the CSV, so omitted from the table and from every count below: "
          + ", ".join(omitted))
    if gates is None or gates.empty:
        a("- gates: reference CSVs not found, not evaluated")
    else:
        for g in gates.itertuples():
            a(f"- gate {g.gate}, {DISP.get(g.model, g.model)}: {g.n_cells} cells, max |diff| "
              f"{g.max_abs_diff:.2e} (tolerance {g.tolerance:.0e}): {_verdict(bool(g.ok))}")
    a("")

    a("## 1. Table rows: each model at its worst baseline layer")
    for name in present:
        x = worst_layer(w, name)
        a(f"- {name} L{int(x['layer'])}, AUROC: {_auc_line(x)}")
        a(f"  - top-PC share: {_auc_line(x, 'pc1')}")
        a(f"  - effective rank: base {x['erank_base']:.2f}; zero variance k=5 "
          f"{x['erank_' + zero_tag('variance', 5)]:.2f}; std {x['erank_standardize']:.2f}; "
          f"ABTT D=3 {x['erank_abtt_D3']:.2f}")
        a("  - zeroed coordinates: mean |x| top 10 = "
          f"{x.get('coords_' + zero_tag('mean_abs', 10))}; variance top 10 = "
          f"{x.get('coords_' + zero_tag('variance', 10))}")
    a("")

    # ---------------------------------------------------------------- collapsed T5
    a(f"## 2. Collapsed T5 layers (baseline AUROC < {COLLAPSE_AUROC:.2f})")
    a(f"- {n_coll} layers: " + ", ".join(
        f"{m} {int((coll['m'] == m).sum())} "
        f"({', '.join(str(int(v)) for v in sorted(coll[coll['m'] == m]['layer']))})"
        for m in T5 if (coll["m"] == m).any()))
    verdicts: List[Tuple[str, str]] = []
    if n_coll:
        res_k = {r: [first_k(x, r, "auc", lambda v: v >= RESTORE_AUROC) for _, x in coll.iterrows()]
                 for r in RANKINGS}
        sh_k = {r: [first_k(x, r, "pc1", lambda v: v < SHARE_BAR) for _, x in coll.iterrows()]
                for r in RANKINGS}
        a(f"### Prediction: zeroing k <= {RESTORE_K} by variance restores AUROC >= {RESTORE_AUROC:.2f}")
        for r in RANKINGS:
            n5 = sum(1 for k in res_k[r] if k is not None and k <= RESTORE_K)
            n10 = sum(1 for k in res_k[r] if k is not None)
            a(f"- ranking by {RANK_LABEL[r]}: restored to >= {RESTORE_AUROC:.2f} with some "
              f"k <= {RESTORE_K} at {n5}/{n_coll} collapsed layers; with some k <= {KS[-1]} at "
              f"{n10}/{n_coll}. Per k: " + ", ".join(
                  f"k={k} {int((coll[f'auc_{zero_tag(r, k)}'] >= RESTORE_AUROC).sum())}/{n_coll}"
                  for k in KS))
            for m in T5:
                t = coll[coll["m"] == m]
                if t.empty:
                    continue
                ks = [first_k(x, r, "auc", lambda v: v >= RESTORE_AUROC) for _, x in t.iterrows()]
                a(f"  - {m}: " + ", ".join(
                    f"L{int(x['layer'])} {_k_str(k)}" for (_, x), k in zip(t.iterrows(), ks)))
        n_var5 = sum(1 for k in res_k["variance"] if k is not None and k <= RESTORE_K)
        n_ma5 = sum(1 for k in res_k["mean_abs"] if k is not None and k <= RESTORE_K)
        best = {r: np.array([max(x["auc_" + zero_tag(r, k)] for k in KS if k <= RESTORE_K)
                             for _, x in coll.iterrows()]) for r in RANKINGS}
        a(f"- best AUROC over k <= {RESTORE_K}, ranking by variance: median "
          f"{f3(np.median(best['variance']))}, max {f3(best['variance'].max())}; by mean |x|: "
          f"median {f3(np.median(best['mean_abs']))}, max {f3(best['mean_abs'].max())}")
        v = _verdict(n_var5 == n_coll)
        a(f"- **{v}**: zeroing k <= {RESTORE_K} by variance restores {n_var5} of {n_coll} "
          f"collapsed T5 layers to AUROC >= {RESTORE_AUROC:.2f} (by mean |x|: {n_ma5} of {n_coll}).")
        verdicts.append((f"zeroing k <= {RESTORE_K} by variance restores collapsed T5 layers to "
                         f"AUROC >= {RESTORE_AUROC:.2f}", f"{v} ({n_var5}/{n_coll})"))

        a(f"### Prediction: top-PC share falls below {SHARE_BAR:.1f}")
        for r in RANKINGS:
            n5 = sum(1 for k in sh_k[r] if k is not None and k <= RESTORE_K)
            a(f"- ranking by {RANK_LABEL[r]}: top-PC share < {SHARE_BAR:.1f} with some "
              f"k <= {RESTORE_K} at {n5}/{n_coll}; per k: " + ", ".join(
                  f"k={k} {int((coll[f'pc1_{zero_tag(r, k)}'] < SHARE_BAR).sum())}/{n_coll}"
                  for k in KS)
              + f"; share at k={RESTORE_K}: "
              f"{f3(coll[f'pc1_{zero_tag(r, RESTORE_K)}'].min())} to "
              f"{f3(coll[f'pc1_{zero_tag(r, RESTORE_K)}'].max())} (median "
              f"{f3(coll[f'pc1_{zero_tag(r, RESTORE_K)}'].median())}); at k={KS[-1]}: "
              f"{f3(coll[f'pc1_{zero_tag(r, KS[-1])}'].min())} to "
              f"{f3(coll[f'pc1_{zero_tag(r, KS[-1])}'].max())}")
        a(f"- base top-PC share at these layers: {f3(coll['pc1_base'].min())} to "
          f"{f3(coll['pc1_base'].max())} (median {f3(coll['pc1_base'].median())})")
        for tag in ("standardize", "abtt_D1", "abtt_D3", "abtt_D10"):
            a(f"- after {TAG_LABEL[tag]}: top-PC share < {SHARE_BAR:.1f} at "
              f"{int((coll[f'pc1_{tag}'] < SHARE_BAR).sum())}/{n_coll}; range "
              f"{f3(coll[f'pc1_{tag}'].min())} to {f3(coll[f'pc1_{tag}'].max())}")
        nt5 = w[~w["is_t5"]]
        if len(nt5):
            a("- range of the embedding-trained models (base top-PC share over all their "
              "layers): " + ", ".join(
                  f"{m} {f3(nt5[nt5['m'] == m]['pc1_base'].min())} to "
                  f"{f3(nt5[nt5['m'] == m]['pc1_base'].max())}" for m in NON_T5
                  if (nt5["m"] == m).any()))
        n_sh = sum(1 for k in sh_k["variance"] if k is not None and k <= RESTORE_K)
        both = sum(1 for _, x in coll.iterrows()
                   if any(x[f"auc_{zero_tag('variance', k)}"] >= RESTORE_AUROC
                          and x[f"pc1_{zero_tag('variance', k)}"] < SHARE_BAR
                          for k in KS if k <= RESTORE_K))
        v = _verdict(n_sh == n_coll)
        a(f"- **{v}**: zeroing k <= {RESTORE_K} by variance brings top-PC share below "
          f"{SHARE_BAR:.1f} at {n_sh} of {n_coll} collapsed T5 layers; AUROC >= {RESTORE_AUROC:.2f} and "
          f"share < {SHARE_BAR:.1f} at the same k at {both} of {n_coll}.")
        verdicts.append((f"zeroing k <= {RESTORE_K} by variance brings top-PC share below "
                         f"{SHARE_BAR:.1f} at collapsed T5 layers", f"{v} ({n_sh}/{n_coll})"))

        a("### Standardization, centering and the projections at the collapsed layers")
        for tag in ("standardize", "center", "abtt_D1", "abtt_D3", "abtt_D10"):
            s = coll[f"auc_{tag}"]
            a(f"- {TAG_LABEL[tag]}: AUROC >= {RESTORE_AUROC:.2f} at "
              f"{int((s >= RESTORE_AUROC).sum())}/{n_coll}; range {f3(s.min())} to {f3(s.max())}, "
              f"median {f3(s.median())}; change against base {(s - coll['auc_base']).min():+.3f} "
              f"to {(s - coll['auc_base']).max():+.3f}")
            for m in T5:
                t = coll[coll["m"] == m]
                if len(t):
                    a(f"  - {m}: >= {RESTORE_AUROC:.2f} at "
                      f"{int((t[f'auc_{tag}'] >= RESTORE_AUROC).sum())}/{len(t)}; range "
                      f"{f3(t[f'auc_{tag}'].min())} to {f3(t[f'auc_{tag}'].max())}")
        a("### The paragraph's decision rule, per collapsed layer")
        a(f"(repaired = AUROC >= {RESTORE_AUROC:.2f}; zeroing = ranking by variance, some "
          f"k <= {RESTORE_K}; projection = ABTT D=3 or D=10)")
        cls = {"zeroing k<=5 repairs": [], "standardization repairs, zeroing k<=5 does not": [],
               "neither repairs, a projection does": [], "nothing repairs": []}
        for (_, x), k in zip(coll.iterrows(), res_k["variance"]):
            lab = f"{x['m']} {int(x['layer'])}"
            if k is not None and k <= RESTORE_K:
                cls["zeroing k<=5 repairs"].append(lab)
            elif x["auc_standardize"] >= RESTORE_AUROC:
                cls["standardization repairs, zeroing k<=5 does not"].append(lab)
            elif max(x["auc_abtt_D3"], x["auc_abtt_D10"]) >= RESTORE_AUROC:
                cls["neither repairs, a projection does"].append(lab)
            else:
                cls["nothing repairs"].append(lab)
        for name, labs in cls.items():
            a(f"- {name}: {len(labs)}/{n_coll}" + (f" ({', '.join(labs)})" if labs else ""))
        a("- the same split when zeroing may use either ranking and any k <= 10: repaired by "
          "zeroing at "
          + str(sum(1 for i in range(n_coll)
                    if res_k["variance"][i] is not None or res_k["mean_abs"][i] is not None))
          + f"/{n_coll}")
        g = coll["auc_abtt_D10"] - coll["auc_base"]
        a("### Share of the ABTT D=10 gain recovered at the collapsed layers "
          "((AUROC - base) / (D=10 - base); median, min to max)")
        for tag in [t for t in ALL_TAGS if t not in ("base", "abtt_D10")]:
            f = (coll[f"auc_{tag}"] - coll["auc_base"]) / g
            a(f"- {TAG_LABEL[tag]}: {100 * f.median():.0f}% ({100 * f.min():.0f}% to "
              f"{100 * f.max():.0f}%)")
        f1 = (coll["auc_abtt_D1"] - coll["auc_base"]) / g
        f3_ = (coll["auc_abtt_D3"] - coll["auc_base"]) / g
        a(f"- check of the caption sentence kept from the placeholder (\"one component "
          f"recovers a median 45 percent ..., three recover at least 80 percent at every "
          f"layer\"): D=1 median {100 * f1.median():.1f}%, D=3 minimum {100 * f3_.min():.1f}%: "
          f"{_verdict(round(100 * f1.median()) == 45 and f3_.min() >= 0.80)}")
        a("### Per collapsed layer")
        for _, x in coll.iterrows():
            a(f"- {x['m']} L{int(x['layer'])}, AUROC: {_auc_line(x)}")
            a(f"  - top-PC share: {_auc_line(x, 'pc1')}")
            a(f"  - smallest k reaching AUROC >= {RESTORE_AUROC:.2f}: by mean |x| "
              f"{_k_str(first_k(x, 'mean_abs', 'auc', lambda v: v >= RESTORE_AUROC))}, by "
              f"variance {_k_str(first_k(x, 'variance', 'auc', lambda v: v >= RESTORE_AUROC))}")
    a("")

    # ---------------------------------------------------------------- rankings compared
    a("## 3. Ranking by variance against ranking by mean |x|")
    a("Prediction: zeroing by mean |x| \"should help less\" than zeroing by variance at the "
      "collapsed T5 layers.")

    def overlap(x: pd.Series, k: int) -> int:
        s1 = set(str(x.get("coords_" + zero_tag("mean_abs", KS[-1]), "")).split(";")[:k])
        s2 = set(str(x.get("coords_" + zero_tag("variance", KS[-1]), "")).split(";")[:k])
        return len(s1 & s2)

    for label, s in ([("collapsed T5 layers", coll)] if n_coll else []) + [
            (f"all layers of {m}", w[w["m"] == m]) for m in present]:
        a(f"- {label} (n={len(s)}):")
        for k in KS:
            d = s[f"auc_{zero_tag('variance', k)}"] - s[f"auc_{zero_tag('mean_abs', k)}"]
            ov = np.array([overlap(x, k) for _, x in s.iterrows()])
            a(f"  - k={k}: variance minus mean |x| AUROC median {d.median():+.3f}, range "
              f"{d.min():+.3f} to {d.max():+.3f}; variance higher at {int((d > 1e-9).sum())}, "
              f"equal at {int((d.abs() <= 1e-9).sum())}, lower at {int((d < -1e-9).sum())}; "
              f"identical top-{k} sets at {int((ov == k).sum())}/{len(s)}, mean shared "
              f"coordinates {ov.mean():.1f} of {k}")
    if n_coll:
        higher = sum(int(((coll[f"auc_{zero_tag('variance', k)}"]
                           - coll[f"auc_{zero_tag('mean_abs', k)}"]) > 1e-9).sum()) for k in KS)
        lower = sum(int(((coll[f"auc_{zero_tag('variance', k)}"]
                          - coll[f"auc_{zero_tag('mean_abs', k)}"]) < -1e-9).sum()) for k in KS)
        total = n_coll * len(KS)
        v = "PASS" if higher > lower else "FAIL" if lower > higher else "NO DIFFERENCE"
        a(f"- **{v}** (criterion: more cells higher than lower): over the {total} (collapsed "
          "layer, k) cells, variance ranking gives the "
          f"higher AUROC in {higher}, the lower in {lower}, the same in {total - higher - lower}.")
        verdicts.append(("zeroing by mean |x| helps less than zeroing by variance at collapsed "
                         "T5 layers", f"{v} (variance higher in {higher}, lower in {lower} of "
                         f"{total} cells)"))
        a("- per collapsed layer, shared coordinates among the top 1/3/5/10 of the two rankings:")
        for m in T5:
            t = coll[coll["m"] == m]
            if len(t):
                a(f"  - {m}: " + ", ".join(
                    f"L{int(x['layer'])} " + "/".join(str(overlap(x, k)) for k in KS)
                    for _, x in t.iterrows()))
    a("")

    # ---------------------------------------------------------------- embedding-trained
    a("## 4. Embedding-trained models")
    a(f"Prediction: at the baseline train-selected layer every zeroing and standardization "
      f"changes AUROC by at most {STABLE_DELTA:.2f} in absolute value; at the weakest layer they "
      "recover only a small part of the ABTT D=10 gain (the paragraph gives no threshold for "
      "\"small\", so no verdict is attached to the second part).")
    zero_tags = [zero_tag(r, k) for r in RANKINGS for k in KS]
    for name in [m for m in NON_T5 if m in present]:
        x = selected_layer(w, name)
        ref = selected_tex.get(name)
        a(f"### {name}")
        a(f"- train-selected layer (argmax base train AUROC): {int(x['layer'])}; "
          + ("selected_layers.tex not found" if ref is None else
             f"selected_layers.tex Base Task A layer: {ref} "
             f"({'match' if ref == int(x['layer']) else 'MISMATCH'})"))
        a(f"- at layer {int(x['layer'])}, AUROC: {_auc_line(x)}")
        d = {t: x[f"auc_{t}"] - x["auc_base"] for t in zero_tags + ["standardize"]}
        a("  - change against base: " + ", ".join(f"{TAG_LABEL[t]} {v:+.4f}" for t, v in d.items()))
        dz = max(abs(d[t]) for t in zero_tags)
        a(f"  - largest |change| over the eight zeroings: {dz:.4f} "
          f"({TAG_LABEL[max(zero_tags, key=lambda t: abs(d[t]))]}); standardization "
          f"{d['standardize']:+.4f}; for reference D=0 {x['auc_center'] - x['auc_base']:+.4f}, "
          f"ABTT D=1 {x['auc_abtt_D1'] - x['auc_base']:+.4f}, D=3 "
          f"{x['auc_abtt_D3'] - x['auc_base']:+.4f}, D=10 "
          f"{x['auc_abtt_D10'] - x['auc_base']:+.4f}")
        vz = _verdict(dz <= STABLE_DELTA)
        vs = _verdict(abs(d["standardize"]) <= STABLE_DELTA)
        a(f"  - **zeroing {vz}** (max |change| {dz:.4f} against {STABLE_DELTA:.2f}); "
          f"**standardization {vs}** (|change| {abs(d['standardize']):.4f})")
        verdicts.append((f"{name}: zeroing changes AUROC by at most {STABLE_DELTA:.2f} at the "
                         f"train-selected layer {int(x['layer'])}", f"{vz} (max {dz:.4f})"))
        verdicts.append((f"{name}: standardization changes AUROC by at most {STABLE_DELTA:.2f} at "
                         f"the train-selected layer {int(x['layer'])}",
                         f"{vs} ({d['standardize']:+.4f})"))
        a(f"  - top-PC share: {_auc_line(x, 'pc1')}")
        y = worst_layer(w, name)
        gain = y["auc_abtt_D10"] - y["auc_base"]
        a(f"- weakest layer {int(y['layer'])}, AUROC: {_auc_line(y)}")
        a(f"  - ABTT D=10 gain {gain:+.4f}; share of it recovered: " + ", ".join(
            f"{TAG_LABEL[t]} {100 * (y[f'auc_{t}'] - y['auc_base']) / gain:.0f}%"
            for t in zero_tags + ["standardize", "center", "abtt_D1", "abtt_D3"]))
        a(f"  - top-PC share: {_auc_line(y, 'pc1')}")
        s = w[w["m"] == name]
        dz_all = pd.concat([(s[f"auc_{t}"] - s["auc_base"]) for t in zero_tags])
        ds_all = s["auc_standardize"] - s["auc_base"]
        a(f"- over all {len(s)} layers: zeroing changes AUROC by {dz_all.min():+.4f} to "
          f"{dz_all.max():+.4f} (|change| > {STABLE_DELTA:.2f} in "
          f"{int((dz_all.abs() > STABLE_DELTA).sum())} of {len(dz_all)} cells); standardization "
          f"by {ds_all.min():+.4f} to {ds_all.max():+.4f} (median {ds_all.median():+.4f})")
    a("")

    # ---------------------------------------------------------------- other T5 layers
    a("## 5. T5 layers that are not collapsed")
    for m in T5:
        s = w[(w["m"] == m) & ~w["collapsed"]]
        if s.empty:
            continue
        a(f"- {m} (layers {', '.join(str(int(v)) for v in sorted(s['layer']))}):")
        for _, x in s.sort_values("layer").iterrows():
            a(f"  - L{int(x['layer'])}, AUROC: {_auc_line(x)}")
    a("")

    # ---------------------------------------------------------------- cosine shares and r
    a("## 6. Cosine shares and r of the top coordinates")
    a("Prediction: r >= 1 for the top coordinates at the collapsed T5 layers; r < 1 at "
      "mT5-base layer 1 and in Qwen3-0.6B. \"Top coordinates\" is read as the three with the "
      "largest mean |x| (the massive coordinates of the paragraph); the same counts under the "
      "variance ranking are given beside it.")
    id_of = {m[1]: m[0] for m in MODELS}
    if n_coll:
        for r in RANKINGS:
            alln = sum(1 for _, x in coll.iterrows()
                       if (top3_r(coords, x["model"], x["layer"], r) >= 1).all())
            anyn = sum(1 for _, x in coll.iterrows()
                       if (top3_r(coords, x["model"], x["layer"], r) >= 1).any())
            top1 = sum(1 for _, x in coll.iterrows()
                       if (top3_r(coords, x["model"], x["layer"], r, 1) >= 1).all())
            rs = np.concatenate([top3_r(coords, x["model"], x["layer"], r)
                                 for _, x in coll.iterrows()])
            a(f"- collapsed T5 layers, top 3 by {RANK_LABEL[r]}: all three have r >= 1 at "
              f"{alln}/{n_coll} layers, at least one at {anyn}/{n_coll}, the top one at "
              f"{top1}/{n_coll}; r over these {len(rs)} coordinates: median "
              f"{np.median(rs):.3f}, min {rs.min():.3f}, max {rs.max():.3f}")
            for m in T5:
                t = coll[coll["m"] == m]
                if len(t):
                    a(f"  - {m}: " + ", ".join(
                        f"L{int(x['layer'])} "
                        + "/".join(f"{v:.2f}" for v in top3_r(coords, x["model"], x["layer"], r))
                        for _, x in t.iterrows()))
        alln = sum(1 for _, x in coll.iterrows()
                   if (top3_r(coords, x["model"], x["layer"], "mean_abs") >= 1).all())
        v = _verdict(alln == n_coll)
        a(f"- **{v}**: the three largest-magnitude coordinates all have r >= 1 at {alln} of "
          f"{n_coll} collapsed T5 layers.")
        verdicts.append(("r >= 1 for the top 3 coordinates by mean |x| at collapsed T5 layers",
                         f"{v} ({alln}/{n_coll})"))
    if "mT5-base" in present:
        rr = top3_r(coords, id_of["mT5-base"], 1, "mean_abs")
        rv = top3_r(coords, id_of["mT5-base"], 1, "variance")
        v = _verdict(bool((rr < 1).all()))
        a("- mT5-base layer 1: r of the top 3 by mean |x| "
          + "/".join(f"{x:.3f}" for x in rr) + "; by variance "
          + "/".join(f"{x:.3f}" for x in rv) + f". **{v}** (prediction r < 1).")
        verdicts.append(("r < 1 for the top 3 coordinates by mean |x| at mT5-base layer 1",
                         f"{v} ({'/'.join(f'{x:.3f}' for x in rr)})"))
    if "Qwen3-0.6B" in present:
        q = w[w["m"] == "Qwen3-0.6B"].sort_values("layer")
        for r in RANKINGS:
            per = [top3_r(coords, id_of["Qwen3-0.6B"], int(layer), r) for layer in q["layer"]]
            alln = sum(1 for p in per if (p < 1).all())
            rs = np.concatenate(per)
            a(f"- Qwen3-0.6B, top 3 by {RANK_LABEL[r]}: all three have r < 1 at "
              f"{alln}/{len(per)} layers; r median {np.median(rs):.3f}, min {rs.min():.3f}, max "
              f"{rs.max():.3f}; layers with some r >= 1: "
              + (", ".join(str(int(layer)) for layer, p in zip(q["layer"], per)
                           if (p >= 1).any()) or "none"))
        per = [top3_r(coords, id_of["Qwen3-0.6B"], int(layer), "mean_abs") for layer in q["layer"]]
        alln = sum(1 for p in per if (p < 1).all())
        v = _verdict(alln == len(per))
        a(f"- **{v}**: the three largest-magnitude coordinates all have r < 1 at {alln} of "
          f"{len(per)} Qwen3-0.6B layers.")
        verdicts.append(("r < 1 for the top 3 coordinates by mean |x| in Qwen3-0.6B",
                         f"{v} ({alln}/{len(per)} layers)"))
    for name in present:
        s = w[w["m"] == name].sort_values("layer")
        per = [top3_r(coords, id_of[name], int(layer), "mean_abs") for layer in s["layer"]]
        a(f"- {name}, r of the top 3 by mean |x| per layer: " + ", ".join(
            f"L{int(layer)}{'*' if c else ''} " + "/".join(f"{v:.2f}" for v in p)
            for layer, c, p in zip(s["layer"], s["collapsed"], per)) + " (* = collapsed)")
    a("")
    a("### Key layers")
    keys: List[Tuple[str, int, str]] = []
    if "LaTa" in present:
        keys.append(("LaTa", 7, "the layer the attribution rule selects"))
    if "mT5-base" in present:
        keys.append(("mT5-base", 1, "massive coordinates without collapse"))
    for name in [m for m in NON_T5 if m in present]:
        lw, ls = int(worst_layer(w, name)["layer"]), int(selected_layer(w, name)["layer"])
        if lw == ls:
            keys.append((name, lw, "worst and train-selected layer"))
        else:
            keys.append((name, lw, "worst layer"))
            keys.append((name, ls, "train-selected layer"))
    for _, x in coll.iterrows():
        if (x["m"], int(x["layer"])) not in [(k[0], k[1]) for k in keys]:
            keys.append((x["m"], int(x["layer"]), "collapsed"))
    for name, layer, why in keys:
        x = w[(w["m"] == name) & (w["layer"] == layer)]
        if x.empty:
            continue
        x = x.iloc[0]
        a(f"- {name} L{layer} ({why}{', collapsed' if x['collapsed'] and why != 'collapsed' else ''}"
          f"): base AUROC {f3(x['auc_base'])}, top-PC share {f3(x['pc1_base'])}")
        L.extend(coord_lines(coords, shares, id_of[name], layer))
    a("")

    # ---------------------------------------------------------------- quoted numbers
    a("## 7. Numbers quoted in the Sec. 5 preamble")

    def share_of(name: str, layer: int, ranking: str, k: int, split: str) -> float:
        s = shares[(shares["model"] == id_of[name]) & (shares["layer"] == layer)
                   & (shares["ranking"] == ranking) & (shares["k"] == k)
                   & (shares["split"] == split)]
        return float(s["share"].iloc[0]) if len(s) else float("nan")

    if "LaTa" in present:
        c = coords[(coords["model"] == id_of["LaTa"]) & (coords["layer"] == 7)
                   & (coords["ranking"] == "mean_abs")].sort_values("rank")
        if len(c):
            top = c["mean_abs"].to_numpy()[:3]
            med = float(c["median_mean_abs"].iloc[0])
            ok = bool(np.all(np.abs(top - 5000) <= 1000) and round(med) == 36)
            a("- \"At LaTa layer 7 ... three coordinates have a mean magnitude of about 5,000 "
              "against a median of 36\": top 3 by mean |x| are coordinates "
              + ", ".join(f"#{int(i)}" for i in c["coord"].to_numpy()[:3])
              + " with mean |x| " + ", ".join(_fmt(v) for v in top)
              + " (|mean| " + ", ".join(_fmt(abs(v)) for v in c["mean"].to_numpy()[:3])
              + f"); the fourth has {_fmt(c['mean_abs'].to_numpy()[3])}; median over all "
              f"coordinates of mean |x| is {_fmt(med)} (of |mean|: "
              f"{_fmt(float(c['median_abs_mean'].iloc[0]))}). "
              f"**{'REPRODUCES' if ok else 'DOES NOT REPRODUCE'}** (criterion: each of the three "
              "within 1,000 of 5,000 and the median rounding to 36).")
            verdicts.append(("quoted: LaTa layer 7, three coordinates near 5,000 against a "
                             "median of 36", "REPRODUCES" if ok else "DOES NOT REPRODUCE"))
        a(f"- LaTa layer 7 cosine share of the top 3 coordinates by mean |x| (the first "
          f"\\pendingnum): train {share_of('LaTa', 7, 'mean_abs', 3, 'train'):.4f}, test "
          f"{share_of('LaTa', 7, 'mean_abs', 3, 'test'):.4f}; by variance: train "
          f"{share_of('LaTa', 7, 'variance', 3, 'train'):.4f}, test "
          f"{share_of('LaTa', 7, 'variance', 3, 'test'):.4f}")
    if "mT5-base" in present:
        a(f"- mT5-base layer 1 cosine share of the top 3 coordinates by mean |x| (the second "
          f"\\pendingnum): train {share_of('mT5-base', 1, 'mean_abs', 3, 'train'):.4f}, test "
          f"{share_of('mT5-base', 1, 'mean_abs', 3, 'test'):.4f}; by variance: train "
          f"{share_of('mT5-base', 1, 'variance', 3, 'train'):.4f}, test "
          f"{share_of('mT5-base', 1, 'variance', 3, 'test'):.4f}")
        x = w[(w["m"] == "mT5-base") & (w["layer"] == 1)]
        if len(x):
            x = x.iloc[0]
            ok = round(x["auc_base"], 2) == 0.82 and round(x["pc1_base"], 2) == 0.23
            a(f"- \"AUROC is 0.82 and top-PC share is 0.23\" at mT5-base layer 1: AUROC "
              f"{x['auc_base']:.4f}, top-PC share {x['pc1_base']:.4f}. "
              f"**{'REPRODUCES' if ok else 'DOES NOT REPRODUCE'}** (both rounded to 2 decimals).")
            verdicts.append(("quoted: mT5-base layer 1 AUROC 0.82 and top-PC share 0.23",
                             "REPRODUCES" if ok else "DOES NOT REPRODUCE"))
    a("")

    a("## 8. Verdicts in one place")
    for claim, v in verdicts:
        a(f"- {claim}: {v}")
    a("")

    a("## 9. All layers")
    a("Columns: base; mag k = zeroed top k by mean |x|; var k = zeroed top k by variance; "
      "standardized; D=0; ABTT D=1, 3, 10. * marks a collapsed layer.")
    for name in present:
        a(f"### {name}: Task A test AUROC")
        L.extend(_md_table(w, name, "auc"))
        a("")
        a(f"### {name}: top-PC share of the training vectors")
        L.extend(_md_table(w, name, "pc1"))
        a("")
    path.write_text("\n".join(L) + "\n")


def cmd_render(args) -> int:
    abl = read_abl(args.out_dir / ABL_NAME)
    coords = pd.read_csv(args.out_dir / COORD_NAME)
    shares = pd.read_csv(args.out_dir / SHARE_NAME)
    w = wide(abl)
    args.tab_dir.mkdir(parents=True, exist_ok=True)
    omitted = write_table(w, args.tab_dir / TABLE_NAME)
    for name in omitted:
        print(f"omitting {name}: no rows in {args.out_dir / ABL_NAME}")
    print(f"wrote {args.tab_dir / TABLE_NAME} ({len(ORDER) - len(omitted)} model rows per block)")
    for name in ORDER:
        x = worst_layer(w, name)
        if x is not None:
            print(f"  worst baseline layer {name}: {int(x['layer'])} (AUROC {x['auc_base']:.3f}); "
                  f"train-selected layer {int(selected_layer(w, name)['layer'])}")
    if args.facts_md is not None:
        gates = None
        if all(p.exists() for p in (args.results_csv, args.h1_csv, args.geom_csv)):
            gates = gate_table(abl, pd.read_csv(args.results_csv), pd.read_csv(args.h1_csv),
                               pd.read_csv(args.geom_csv))
        args.facts_md.parent.mkdir(parents=True, exist_ok=True)
        facts(w, coords, shares, gates, parse_selected_layers(args.selected_tex),
              args.facts_md, omitted=omitted)
        print(f"wrote {args.facts_md}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def refs(p) -> None:
        p.add_argument("--out_dir", type=Path, default=OUT_DIR)
        p.add_argument("--results_csv", type=Path, default=RES_CSV)
        p.add_argument("--h1_csv", type=Path, default=H1_CSV)
        p.add_argument("--geom_csv", type=Path, default=GEOM_CSV)

    p = sub.add_parser("compute", help="ablate, score and write the three CSVs")
    refs(p)
    p.add_argument("--split_csv", type=Path, default=SPLIT_CSV)
    p.add_argument("--bases_root", type=Path, default=BASES_ROOT)
    p.add_argument("--models", default="", help="comma list of ids or display names (default all six)")
    p.add_argument("--allow_missing", action="store_true",
                   help="skip a model whose cache is missing instead of failing")
    p.add_argument("--layers", default="", help="comma list of layers (default all)")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--check", action="store_true",
                   help="run the reproduction gates after writing; exit 3 if one fails")
    p = sub.add_parser("check", help="reproduction gates on an existing CSV")
    refs(p)
    p.add_argument("--no_write", action="store_true", help=f"do not rewrite {GATE_NAME}")
    p = sub.add_parser("render", help="paper table and facts file")
    refs(p)
    p.add_argument("--tab_dir", type=Path, default=TAB_DIR)
    p.add_argument("--selected_tex", type=Path, default=SELECTED_TEX)
    p.add_argument("--facts_md", type=Path, default=None)
    p.add_argument("--no_facts", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "compute":
        return cmd_compute(args)
    if args.cmd == "check":
        return cmd_check(args)
    if args.facts_md is None and not args.no_facts:
        args.facts_md = args.out_dir / FACTS_NAME
    if args.no_facts:
        args.facts_md = None
    return cmd_render(args)


if __name__ == "__main__":
    sys.exit(main())
