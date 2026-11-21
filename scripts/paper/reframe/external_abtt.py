#!/usr/bin/env python3
"""Can ABTT's components come from other CCL text instead of the training split?

Siddique's question (2026-11-21): fit the top principal components on text that is not
the canon testbed, here the decretal excerpts (``decrExcerpt``, 439 labelled files,
cached by the April 2026 run under ``runs/active/decritals_bases``), and check whether
they are the components fit on the testbed's training split. If they are, ABTT needs no
training split of its own.

Per panel model and layer (hidden_mean_tokempty, the cache behind every paper number):

  overlap   subspace overlap ||U_a U_b^T||_F^2 / D of the top-D principal directions,
            D in {1, 3, 10}, for (canon train, decretals) and, as the within-testbed
            reference, (canon train, canon test). 1 = same subspace; a random D-subspace
            of R^d gives about D/d.
  scores    the paper's metric block (test AUROC, test and train DirAcc@1, tau learned on
            canon train) after ABTT with D in the paper grid, the components fit on
              canon_train            the paper's setting;
              decr                   decretals mean and components;
              decr_pcs_train_mean    decretals components, canon training mean;
              canon_all              all 1,705 testbed witnesses, train and test pooled
                                     (label-free, but it sees the test text).
            D is also selected per fit by training DirAcc@1, as the paper does; that
            selection still uses testbed training labels.
  leakage   how many decretal files repeat testbed text: exact matches, and files whose
            best character 5-gram Jaccard with a testbed witness is 0.5 or more.

Outputs under runs/active/reframe/external_abtt/: overlap.csv, scores.csv, leakage.json,
summary.md. CPU only; run from the data root:
  python scripts/paper/reframe/external_abtt.py --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtt_subspace_whiten as asw  # noqa: E402
from sif_abtt import EmbeddingCleaner  # noqa: E402

EXT_ROOT = Path("runs/active/decritals_bases/phase9_bases")
EXT_DATA = Path("data/decritals_labelled")
OUT = asw.OUT_ROOT / "external_abtt"
OVERLAP_D = (1, 3, 10)
GRID = asw.SEL_GRID
FITS = ("canon_train", "decr", "decr_pcs_train_mean", "canon_all")


def top_dirs(X: np.ndarray, D: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    _, _, vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    return vt[:D]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """||A B^T||_F^2 / D for two orthonormal D x d row bases: mean cos^2 of principal angles."""
    return float(np.sum((a @ b.T) ** 2) / a.shape[0])


def cleaner(fit_on: np.ndarray, D: int, mean=None) -> EmbeddingCleaner:
    c = EmbeddingCleaner(num_components=D, center=True).fit(fit_on)
    if mean is not None:
        c.mean_vec = mean
    return c


def task(args) -> Dict[str, List[Dict]]:
    bases_root, model_id, layer = args
    tr, te = asw._load(bases_root, asw.slug(model_id), layer)
    ext = np.load(EXT_ROOT / asw.slug(model_id) / asw.SUBDIR / f"hidden_layer{layer}_embeddings.npy")
    d = tr.shape[1]
    ov = []
    for D in OVERLAP_D:
        u_tr, u_te, u_ex = top_dirs(tr, D), top_dirs(te, D), top_dirs(ext, D)
        ov.append({"model": model_id, "layer": layer, "D": D, "d": d,
                   "train_vs_decr": overlap(u_tr, u_ex), "train_vs_test": overlap(u_tr, u_te),
                   "random": D / d})
    base = asw._metrics(tr, te)
    rows = [{"model": model_id, "layer": layer, "fit": "none", "D": 0,
             **{k: base[k] for k in asw.KEEP}}]
    both = np.vstack([tr, te])
    for D in GRID:
        cs = {"canon_train": cleaner(tr, D), "decr": cleaner(ext, D),
              "decr_pcs_train_mean": cleaner(ext, D, mean=tr.mean(axis=0)),
              "canon_all": cleaner(both, D)}
        for fit, c in cs.items():
            m = asw._metrics(c.transform(tr), c.transform(te))
            rows.append({"model": model_id, "layer": layer, "fit": fit, "D": D,
                         **{k: m[k] for k in asw.KEEP}})
    print(f"  {asw.DISP[model_id]} L{layer}: base {base['aucroc']:.3f} "
          + " ".join(f"{r['fit']}@{r['D']} {r['aucroc']:.3f}" for r in rows if r["D"] == 3),
          flush=True)
    return {"overlap": ov, "scores": rows}


def shingles(text: str, n: int = 5) -> set:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 1))}


def leakage() -> Dict:
    canon = [p.read_text(errors="ignore") for p in sorted(Path("data/canon_labelled").rglob("*.txt"))]
    decr = [p.read_text(errors="ignore") for p in sorted(EXT_DATA.rglob("*.txt"))]
    h = {hashlib.sha256(t.strip().encode()).hexdigest() for t in canon}
    exact = sum(hashlib.sha256(t.strip().encode()).hexdigest() in h for t in decr)
    cs = [shingles(t) for t in canon]
    best = []
    for t in decr:
        s = shingles(t)
        best.append(max((len(s & c) / len(s | c) for c in cs if c), default=0.0))
    best = np.array(best)
    return {"n_decr_files": len(decr), "n_canon_files": len(canon), "exact_duplicates": int(exact),
            "jaccard5_ge_0.5": int((best >= 0.5).sum()), "jaccard5_ge_0.3": int((best >= 0.3).sum()),
            "jaccard5_median": float(np.median(best))}


def select(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, layer, fit): D with the first-highest training DirAcc@1 on the grid."""
    out = []
    for (m, l, f), g in df[df.fit != "none"].groupby(["model", "layer", "fit"], sort=False):
        g = g.set_index("D").loc[GRID]
        D = int(g["train_dir_acc_at_1"].idxmax())
        out.append({"model": m, "layer": l, "fit": f, "D_sel": D, **g.loc[D].to_dict()})
    return pd.DataFrame(out)


def summarize(ov: pd.DataFrame, sc: pd.DataFrame, leak: Dict) -> str:
    base = sc[sc.fit == "none"].set_index(["model", "layer"])["aucroc"]
    collapsed = base[base < asw.COLLAPSE_AUROC].index
    sel = select(sc)
    lines = ["# External ABTT components (decretals)", "",
             f"Leakage: {json.dumps(leak)}", "",
             "## Subspace overlap with the canon-train components (median over layers)", "",
             "| model | D | train vs decretals | train vs test | random |", "|---|---|---|---|---|"]
    for (m, D), g in ov.groupby(["model", "D"], sort=False):
        lines.append(f"| {asw.DISP[m]} | {D} | {g.train_vs_decr.median():.3f} "
                     f"| {g.train_vs_test.median():.3f} | {g.random.iloc[0]:.4f} |")
    lines += ["", f"## Collapsed layers (baseline test AUROC < 0.70): {len(collapsed)}", "",
              "| fit | restored >= 0.90 at D=3 | at selected D | median AUROC, selected D |",
              "|---|---|---|---|"]
    for f in FITS:
        d3 = sc[(sc.fit == f) & (sc.D == 3)].set_index(["model", "layer"])["aucroc"]
        s = sel[sel.fit == f].set_index(["model", "layer"])["aucroc"]
        lines.append(f"| {f} | {(d3.loc[collapsed] >= 0.90).sum()} | {(s.loc[collapsed] >= 0.90).sum()} "
                     f"| {s.loc[collapsed].median():.3f} |")
    lines += ["", "## All layers, selected D: difference from canon_train", "",
              "| fit | median dAUROC | min dAUROC | median dDirAcc@1 | min dDirAcc@1 |", "|---|---|---|---|---|"]
    ref = sel[sel.fit == "canon_train"].set_index(["model", "layer"])
    for f in FITS[1:]:
        s = sel[sel.fit == f].set_index(["model", "layer"])
        da, dd = s.aucroc - ref.aucroc, s.dir_acc_at_1 - ref.dir_acc_at_1
        lines.append(f"| {f} | {da.median():+.4f} | {da.min():+.4f} | {dd.median():+.4f} | {dd.min():+.4f} |")
    lines += ["", "## Per model, best layer by training DirAcc@1 (selected D)", "",
              "| model | fit | layer | D | test AUROC | test DirAcc@1 |", "|---|---|---|---|---|---|"]
    for m in sel.model.unique():
        for f in FITS:
            g = sel[(sel.model == m) & (sel.fit == f)]
            r = g.loc[g.train_dir_acc_at_1.idxmax()]
            lines.append(f"| {asw.DISP[m]} | {f} | {int(r.layer)} | {int(r.D_sel)} "
                         f"| {r.aucroc:.3f} | {100 * r.dir_acc_at_1:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bases_root", default=str(asw.BASES_ROOT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = asw.build_tasks(Path(args.bases_root), asw.pick_models(args.models, asw.MODELS), None)
    tasks = [t for t in tasks if (EXT_ROOT / asw.slug(t[1]) / asw.SUBDIR /
                                  f"hidden_layer{t[2]}_embeddings.npy").exists()]
    ov, sc = [], []
    with Pool(args.workers, initializer=asw._init, initargs=(str(asw.SPLIT_CSV),)) as pool:
        for part in pool.imap(task, tasks, chunksize=1):
            ov += part["overlap"]
            sc += part["scores"]
    ov, sc = pd.DataFrame(ov), pd.DataFrame(sc)
    ov.to_csv(OUT / "overlap.csv", index=False)
    sc.to_csv(OUT / "scores.csv", index=False)
    leak = leakage()
    (OUT / "leakage.json").write_text(json.dumps(leak, indent=2))
    text = summarize(ov, sc, leak)
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
