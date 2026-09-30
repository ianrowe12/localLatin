#!/usr/bin/env python3
"""E1 extension (issue #246, post hoc): zero many more top coordinates than k = 10.

Descriptive follow-up to ``e1_coordinate_ablation.py``, added after its results were
read. The pre-registered E1 grid k in {1, 3, 5, 10} tested the Sec. 5 prediction that
zeroing k <= 5 coordinates repairs the collapsed T5 layers; it does not change here, and
no verdict of ``facts_e1.md`` depends on this script. The question is different: how many
coordinates must be zeroed before zeroing matches standardization or ABTT, and is the gain
specific to the top coordinates or does zeroing any k coordinates do as well?

  compute  per model-layer, fit on the 847 TRAIN passages only, on the raw pooled vectors
           (``hidden_mean_tokempty``): zero the top k coordinates, k in K_SWEEP (capped
           below the width), ranked by mean |x| and by variance exactly as in E1, and, as
           the control, zero k coordinates drawn uniformly at random (N_SEEDS nested
           permutations, so the random k sets grow like the ranked ones).
           Readouts: Task A test and train AUROC (cosine on L2-normalized vectors, the
           same function as the paper's metric block), top-PC share and effective rank of
           the intervened TRAIN vectors (``pca_stats``, as in E1), and the share of the
           total training variance held by the zeroed coordinates.
  check    gate: the ranked rows at k in {1, 3, 5, 10} equal ``e1_coordinate_ablation.csv``
           (test AUROC, train AUROC, top-PC share) within 1e-6; exit 3 on failure.
  render   ``facts_e1_k_sweep.md`` and ``e1_k_sweep.pdf`` (AUROC and top-PC share
           against k at each model's worst baseline layer). Reference lines (base,
           standardization, ABTT D=3 and D=10) are read from ``e1_coordinate_ablation.csv``.

Outputs, in runs/active/reframe/e1/ (small, force-added): e1_k_sweep.csv,
e1_k_sweep_gate.csv, facts_e1_k_sweep.md, e1_k_sweep.pdf. Nothing is written to
overleaf_drafts/.

  python scripts/paper/reframe/e1_k_sweep.py compute --check --workers 16 \
      --bases_root runs/active/resubmit_bases
  python scripts/paper/reframe/e1_k_sweep.py render
CPU only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e1_coordinate_ablation as e1  # noqa: E402  (sets sys.path for src/ and resubmit/)

asw = e1.asw

OUT_DIR = e1.OUT_DIR
SWEEP_NAME = "e1_k_sweep.csv"
GATE_NAME = "e1_k_sweep_gate.csv"
FACTS_NAME = "facts_e1_k_sweep.md"
FIG_NAME = "e1_k_sweep.pdf"

K_SWEEP = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400)
N_SEEDS = 5
SEED_BASE = 246  # permutation seed s is SEED_BASE + s, independent of model and layer
SWEEP_RANKINGS = ("mean_abs", "variance", "random")
LABEL = {"mean_abs": "mean |x|", "variance": "variance", "random": "random"}
GATE_TOL = 1e-6
GATE_EXIT = 3


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #

def ks_for(dim: int, ks: Sequence[int] = K_SWEEP) -> List[int]:
    """The sweep's k values that leave at least one coordinate."""
    return [k for k in ks if k < dim]


def random_orders(dim: int, n_seeds: int = N_SEEDS, seed_base: int = SEED_BASE
                  ) -> List[np.ndarray]:
    """One permutation of the coordinates per seed; its first k entries are the random
    k-set, so the sets are nested in k. Depends only on the width and the seed."""
    return [np.random.default_rng(seed_base + s).permutation(dim) for s in range(n_seeds)]


def sweep_rows(model_id: str, layer: int, tr: np.ndarray, te: np.ndarray,
               auroc_fn, ks: Sequence[int] = K_SWEEP, n_seeds: int = N_SEEDS
               ) -> List[Dict]:
    """All sweep rows of one model-layer. ``auroc_fn(train, test)`` returns a dict with
    ``aucroc`` (test) and ``train_aucroc``; it L2-normalizes internally."""
    key = {"model": model_id, "layer": int(layer)}
    dim = int(tr.shape[1])
    stats = e1.coord_stats(tr)
    var = stats["variance"]
    total_var = float(var.sum())
    orders: List[Tuple[str, int, np.ndarray]] = [
        (r, -1, e1.rank_coords(tr, r, stats)) for r in e1.RANKINGS]
    orders += [("random", s, o) for s, o in enumerate(random_orders(dim, n_seeds))]
    rows: List[Dict] = []
    for ranking, seed, order in orders:
        for k in ks_for(dim, ks):
            idx = order[:k]
            a, b = e1.zero_coords(tr, te, idx)
            m = auroc_fn(a, b)
            rows.append({**key, "dim": dim, "ranking": ranking, "seed": seed, "k": k,
                         "aucroc": m["aucroc"], "train_aucroc": m["train_aucroc"],
                         "var_removed": (float(var[idx].sum() / total_var)
                                         if total_var > 0 else float("nan")),
                         **geometry(a, f" at {model_id} L{layer} {ranking} seed {seed} "
                                       f"k={k}")})
    return rows


def geometry(train: np.ndarray, where: str = "") -> Dict[str, float]:
    """E1's ``_geometry``; if LAPACK's SVD does not converge, the same spectrum from the
    eigenvalues of the Gram matrix of the centered vectors (float64), noted on stdout."""
    try:
        return e1._geometry(train)
    except np.linalg.LinAlgError:
        x = np.asarray(train, dtype=np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        eig = np.clip(np.linalg.eigvalsh(x @ x.T)[::-1], 0.0, None)
        p = eig / eig.sum()
        nz = p[p > 0]
        print(f"  note: SVD did not converge{where}; top-PC share from the Gram eigenvalues",
              flush=True)
        return {"pc1_share_train": float(p[0]),
                "pc10_share_train": float(p[:10].sum()),
                "eff_rank_train": float(np.exp(-np.sum(nz * np.log(nz))))}


def _auroc(train: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    return {"aucroc": asw.pair_auroc(test, asw._CTX["lab_te"]),
            "train_aucroc": asw.pair_auroc(train, asw._CTX["lab_tr"])}


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #

def task_sweep(args) -> List[Dict]:
    bases_root, model_id, layer = args
    t0 = time.time()
    tr, te = asw._load(bases_root, asw.slug(model_id), layer)
    rows = sweep_rows(model_id, layer, tr, te, _auroc)
    best = max((r for r in rows if r["ranking"] == "variance"), key=lambda r: r["aucroc"])
    print(f"  sweep {e1.DISP.get(model_id, model_id)} L{layer}: best variance k={best['k']} "
          f"AUROC {best['aucroc']:.3f} ({len(rows)} rows, {time.time() - t0:.1f}s)",
          flush=True)
    return rows


def cmd_compute(args) -> int:
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    model_ids = asw.pick_models(args.models, e1.MODELS)
    tasks = e1.build_tasks(args.bases_root, model_ids, layers, args.allow_missing)
    if not tasks:
        raise SystemExit("ERROR: nothing to compute (no model has cached vectors)")
    t0 = time.time()
    rows: List[Dict] = []
    if args.workers <= 1:
        asw._init(str(args.split_csv))
        for part in map(task_sweep, tasks):
            rows.extend(part)
    else:
        from multiprocessing import Pool

        with Pool(args.workers, initializer=asw._init,
                  initargs=(str(args.split_csv),)) as pool:
            for part in pool.imap(task_sweep, tasks, chunksize=1):
                rows.extend(part)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = e1._sorted(pd.DataFrame(rows), [])
    df.to_csv(args.out_dir / SWEEP_NAME, index=False, float_format="%.10g")
    print(f"wrote {args.out_dir / SWEEP_NAME} ({len(df)} rows, {len(tasks)} model-layers) "
          f"in {time.time() - t0:.0f}s")
    return cmd_check(args) if args.check else 0


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #

def gate_table(sweep: pd.DataFrame, abl: pd.DataFrame) -> pd.DataFrame:
    """Ranked sweep rows at E1's k values against the E1 ablation CSV."""
    ref = abl[abl["intervention"] == "zero"][
        ["model", "layer", "ranking", "k", "aucroc", "train_aucroc", "pc1_share_train"]]
    s = sweep[(sweep["ranking"].isin(e1.RANKINGS)) & (sweep["k"].isin(e1.KS))]
    m = s.merge(ref, on=["model", "layer", "ranking", "k"], how="outer",
                suffixes=("", "_e1"), indicator=True)
    out = []
    for (mid, ranking), g in m.groupby(["model", "ranking"], sort=False):
        both = g[g["_merge"] == "both"]
        diffs = [np.abs(both[c] - both[f"{c}_e1"]).max() if len(both) else np.nan
                 for c in ("aucroc", "train_aucroc", "pc1_share_train")]
        out.append({"model": mid, "ranking": ranking, "n_cells": len(both),
                    "n_unmatched": int((g["_merge"] != "both").sum()),
                    "max_diff_auroc": diffs[0], "max_diff_train_auroc": diffs[1],
                    "max_diff_pc1_share": diffs[2]})
    t = pd.DataFrame(out)
    worst = t[["max_diff_auroc", "max_diff_train_auroc", "max_diff_pc1_share"]].max(axis=1)
    t["pass"] = (t["n_unmatched"] == 0) & (t["n_cells"] > 0) & (worst <= GATE_TOL)
    return t


def cmd_check(args) -> int:
    sweep = pd.read_csv(args.out_dir / SWEEP_NAME)
    abl = e1.read_abl(args.out_dir / e1.ABL_NAME)
    abl = abl[abl["model"].isin(sweep["model"].unique())]
    t = gate_table(sweep, abl)
    t.to_csv(args.out_dir / GATE_NAME, index=False, float_format="%.3g")
    for _, g in t.iterrows():
        print(f"  gate {e1.DISP.get(g['model'], g['model'])} {g['ranking']}: {g['n_cells']} "
              f"cells, max |diff| AUROC {g['max_diff_auroc']:.2e}, train "
              f"{g['max_diff_train_auroc']:.2e}, top-PC share {g['max_diff_pc1_share']:.2e}: "
              f"{'PASS' if g['pass'] else 'FAIL'}")
    return 0 if bool(t["pass"].all()) else GATE_EXIT


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

def curves(sweep: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, layer, ranking, k); random = mean over seeds with min and max."""
    ranked = sweep[sweep["ranking"] != "random"].copy()
    for c in ("aucroc", "pc1_share_train"):
        ranked[f"{c}_min"] = ranked[c]
        ranked[f"{c}_max"] = ranked[c]
    g = sweep[sweep["ranking"] == "random"].groupby(["model", "layer", "ranking", "k"],
                                                     sort=False)
    rnd = g.agg(aucroc=("aucroc", "mean"), aucroc_min=("aucroc", "min"),
                aucroc_max=("aucroc", "max"), train_aucroc=("train_aucroc", "mean"),
                pc1_share_train=("pc1_share_train", "mean"),
                pc1_share_train_min=("pc1_share_train", "min"),
                pc1_share_train_max=("pc1_share_train", "max"),
                eff_rank_train=("eff_rank_train", "mean"),
                var_removed=("var_removed", "mean"), dim=("dim", "first")).reset_index()
    cols = ["model", "layer", "ranking", "k", "dim", "aucroc", "aucroc_min", "aucroc_max",
            "train_aucroc", "pc1_share_train", "pc1_share_train_min", "pc1_share_train_max",
            "eff_rank_train", "var_removed"]
    return pd.concat([ranked[cols], rnd[cols]], ignore_index=True)


def _curve(c: pd.DataFrame, mid: str, layer: int, ranking: str) -> pd.DataFrame:
    return c[(c["model"] == mid) & (c["layer"] == layer)
             & (c["ranking"] == ranking)].sort_values("k")


def first_k_reaching(cur: pd.DataFrame, col: str, bar: float, above: bool = True
                     ) -> Optional[int]:
    hit = cur[cur[col] >= bar] if above else cur[cur[col] < bar]
    return None if hit.empty else int(hit["k"].iloc[0])


def _kstr(k: Optional[int], kmax: int) -> str:
    return f"k={k}" if k is not None else f"none up to {kmax}"


def _med(values: Sequence[float]) -> str:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return "--"
    return f"{np.median(v):.3f} ({v.min():.3f} to {v.max():.3f})"


def facts(sweep: pd.DataFrame, w: pd.DataFrame, gates: pd.DataFrame, path: Path) -> None:
    c = curves(sweep)
    L: List[str] = []
    a = L.append
    kmax = max(K_SWEEP)
    a("# E1 k sweep: facts (generated, post hoc)")
    a("")
    a("Generated by `scripts/paper/reframe/e1_k_sweep.py render` from `e1_k_sweep.csv` and "
      "`e1_coordinate_ablation.csv`. This sweep was added after the E1 results were read. "
      "It attaches no prediction and changes no verdict of `facts_e1.md`, whose "
      "pre-registered grid stays k in [1, 3, 5, 10].")
    a("")
    a("## Definitions")
    a(f"- k in {list(K_SWEEP)}, dropping any k >= the width. Rankings `mean |x|` and "
      "`variance` are E1's, on the raw TRAIN vectors. `random` zeroes the first k "
      f"coordinates of a fixed permutation (seeds {SEED_BASE} to {SEED_BASE + N_SEEDS - 1}, "
      "the same permutations at every layer of a model width); its numbers are the mean "
      "over the seeds, with min to max where shown.")
    a("- AUROC is Task A test AUROC on L2-normalized vectors after zeroing. Top-PC share and "
      "effective rank are E1's, on the zeroed TRAIN vectors. `var removed` is the share of "
      "the total training variance held by the zeroed coordinates.")
    a("- References from `e1_coordinate_ablation.csv`: base, std (per-coordinate "
      "standardization), ABTT D=3 and D=10. Collapsed = T5 layer with base AUROC < "
      f"{e1.COLLAPSE_AUROC:.2f}. Repaired = AUROC >= {e1.RESTORE_AUROC:.2f}.")
    a("")
    a("## 0. Gate: sweep rows at E1's k values against `e1_coordinate_ablation.csv`")
    for _, g in gates.iterrows():
        a(f"- {e1.DISP.get(g['model'], g['model'])}, {LABEL[g['ranking']]}: {g['n_cells']} "
          f"cells, max |diff| test AUROC {g['max_diff_auroc']:.2e}, train AUROC "
          f"{g['max_diff_train_auroc']:.2e}, top-PC share {g['max_diff_pc1_share']:.2e} "
          f"(tolerance {GATE_TOL:g}): {'PASS' if g['pass'] else 'FAIL'}")
    a("")

    coll = w[w["collapsed"]].sort_values(["m", "layer"])
    coll = pd.concat([coll[coll["m"] == n] for n in e1.T5])
    n_coll = len(coll)
    a(f"## 1. Collapsed T5 layers (n = {n_coll}): median (min to max) over layers, per k")
    ks_all = sorted(sweep["k"].unique())
    for r in SWEEP_RANKINGS:
        a(f"### zero by {LABEL[r]}")
        a("| k | AUROC | top-PC share | var removed | repaired | >= std AUROC | "
          "share < 0.2 |")
        a("|---|---|---|---|---|---|---|")
        for k in ks_all:
            vals, shares, vr, rep, std_hit, sh = [], [], [], 0, 0, 0
            for _, x in coll.iterrows():
                cur = _curve(c, x["model"], int(x["layer"]), r)
                cur = cur[cur["k"] == k]
                if cur.empty:
                    continue
                y = cur.iloc[0]
                vals.append(y["aucroc"])
                shares.append(y["pc1_share_train"])
                vr.append(y["var_removed"])
                rep += int(y["aucroc"] >= e1.RESTORE_AUROC)
                std_hit += int(y["aucroc"] >= x["auc_standardize"])
                sh += int(y["pc1_share_train"] < e1.SHARE_BAR)
            if vals:
                a(f"| {k} | {_med(vals)} | {_med(shares)} | {_med(vr)} | {rep}/{len(vals)} "
                  f"| {std_hit}/{len(vals)} | {sh}/{len(vals)} |")
        a("")
    a("- references at the collapsed layers: base " + _med(coll["auc_base"]) + "; std "
      + _med(coll["auc_standardize"]) + "; ABTT D=3 " + _med(coll["auc_abtt_D3"])
      + "; ABTT D=10 " + _med(coll["auc_abtt_D10"]))
    a("")
    a("## 2. Per collapsed layer")
    a("Smallest k reaching AUROC >= 0.90, smallest k reaching the layer's std AUROC, "
      "smallest k with top-PC share < 0.2, and the best AUROC over the sweep with its k. "
      "Random uses the seed mean.")
    a("| model | L | base | std | D=3 | ranking | k to 0.90 | k to std | k to share<0.2 "
      "| best AUROC (k) |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for _, x in coll.iterrows():
        for r in SWEEP_RANKINGS:
            cur = _curve(c, x["model"], int(x["layer"]), r)
            b = cur.loc[cur["aucroc"].idxmax()]
            a(f"| {x['m']} | {int(x['layer'])} | {x['auc_base']:.3f} | "
              f"{x['auc_standardize']:.3f} | {x['auc_abtt_D3']:.3f} | {LABEL[r]} | "
              f"{_kstr(first_k_reaching(cur, 'aucroc', e1.RESTORE_AUROC), kmax)} | "
              f"{_kstr(first_k_reaching(cur, 'aucroc', x['auc_standardize']), kmax)} | "
              f"{_kstr(first_k_reaching(cur, 'pc1_share_train', e1.SHARE_BAR, False), kmax)}"
              f" | {b['aucroc']:.3f} ({int(b['k'])}) |")
    a("")
    a("## 3. Ranked minus random at the collapsed layers")
    a("AUROC of zeroing by variance minus the random seed mean, and the share of random "
      "seeds that the variance ranking beats, per k.")
    for k in ks_all:
        d, beat, n = [], 0, 0
        for _, x in coll.iterrows():
            v = _curve(c, x["model"], int(x["layer"]), "variance")
            v = v[v["k"] == k]
            s = sweep[(sweep["model"] == x["model"]) & (sweep["layer"] == x["layer"])
                      & (sweep["ranking"] == "random") & (sweep["k"] == k)]
            if v.empty or s.empty:
                continue
            d.append(float(v["aucroc"].iloc[0] - s["aucroc"].mean()))
            beat += int((v["aucroc"].iloc[0] > s["aucroc"]).sum())
            n += len(s)
        if d:
            a(f"- k={k}: variance minus random {_med(d)}; variance above {beat}/{n} "
              "(layer, seed) cells")
    a("")
    a("## 4. Each model at its worst baseline layer and at its train-selected layer")
    for name in e1.ORDER:
        for which, fn in (("worst", e1.worst_layer), ("train-selected", e1.selected_layer)):
            x = fn(w, name)
            if x is None:
                continue
            a(f"### {name} L{int(x['layer'])} ({which}): base {x['auc_base']:.3f}, std "
              f"{x['auc_standardize']:.3f}, ABTT D=3 {x['auc_abtt_D3']:.3f}, D=10 "
              f"{x['auc_abtt_D10']:.3f}; base top-PC share {x['pc1_base']:.3f}")
            a("| k | " + " | ".join(f"AUROC {LABEL[r]}" for r in SWEEP_RANKINGS) + " | "
              + " | ".join(f"share {LABEL[r]}" for r in SWEEP_RANKINGS) + " |")
            a("|---" * (1 + 2 * len(SWEEP_RANKINGS)) + "|")
            cs = {r: _curve(c, x["model"], int(x["layer"]), r).set_index("k")
                  for r in SWEEP_RANKINGS}
            for k in cs["variance"].index:
                a(f"| {k} | " + " | ".join(f"{cs[r].loc[k, 'aucroc']:.3f}"
                                           for r in SWEEP_RANKINGS) + " | "
                  + " | ".join(f"{cs[r].loc[k, 'pc1_share_train']:.3f}"
                               for r in SWEEP_RANKINGS) + " |")
            a("")
    path.write_text("\n".join(L) + "\n")


def figure(sweep: pd.DataFrame, w: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = curves(sweep)
    names = [n for n in e1.ORDER if e1.worst_layer(w, n) is not None]
    fig, axes = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5.2),
                             squeeze=False, sharex=True)
    style = {"variance": ("#0072B2", "-"), "mean_abs": ("#E69F00", "--"),
             "random": ("#777777", ":")}
    for j, name in enumerate(names):
        x = e1.worst_layer(w, name)
        for i, (col, ref) in enumerate((("aucroc", "auc"), ("pc1_share_train", "pc1"))):
            ax = axes[i, j]
            for r in SWEEP_RANKINGS:
                cur = _curve(c, x["model"], int(x["layer"]), r)
                colr, ls = style[r]
                ax.plot(cur["k"], cur[col], ls, color=colr, lw=1.4, label=LABEL[r])
                if r == "random":
                    ax.fill_between(cur["k"], cur[f"{col}_min"], cur[f"{col}_max"],
                                    color=colr, alpha=0.2, lw=0)
            for tag, lab, colr in (("base", "base", "k"), ("standardize", "std", "#009E73"),
                                   ("abtt_D3", "ABTT D=3", "#CC79A7")):
                ax.axhline(x[f"{ref}_{tag}"], color=colr, lw=0.8, alpha=0.7, ls="-.",
                           label=lab)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(f"{name} L{int(x['layer'])}", fontsize=9)
            else:
                ax.set_xlabel("k zeroed", fontsize=8)
            if j == 0:
                ax.set_ylabel("test AUROC" if i == 0 else "top-PC share (train)", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cmd_render(args) -> int:
    sweep = pd.read_csv(args.out_dir / SWEEP_NAME)
    abl = e1.read_abl(args.out_dir / e1.ABL_NAME)
    abl = abl[abl["model"].isin(sweep["model"].unique())]
    w = e1.wide(abl)
    gates = gate_table(sweep, abl)
    facts(sweep, w, gates, args.out_dir / FACTS_NAME)
    print(f"wrote {args.out_dir / FACTS_NAME}")
    figure(sweep, w, args.out_dir / FIG_NAME)
    print(f"wrote {args.out_dir / FIG_NAME}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compute", "check", "render"):
        p = sub.add_parser(name)
        p.add_argument("--out_dir", type=Path, default=OUT_DIR)
        if name == "compute":
            p.add_argument("--split_csv", type=Path, default=e1.SPLIT_CSV)
            p.add_argument("--bases_root", type=Path, default=e1.BASES_ROOT)
            p.add_argument("--models", default="")
            p.add_argument("--layers", default="")
            p.add_argument("--allow_missing", action="store_true")
            p.add_argument("--workers", type=int,
                           default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
            p.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    return {"compute": cmd_compute, "check": cmd_check, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
