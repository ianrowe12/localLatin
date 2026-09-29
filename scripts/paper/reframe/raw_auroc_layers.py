#!/usr/bin/env python3
"""Per-layer Task A test AUROC of unmodified mean-pooled vectors (issue #244).

Fills the T5-v1.1-base AUROC_min cell of ``tab:panel_2x2``: the lowest Task A test
AUROC over layers of the raw mean-pooled T5-v1.1-base vectors on the Latin corpus,
extracted by ``gen_extract.py`` (#238) into ``runs/active/reframe/gen/bases/``.

To prove the pipeline, the same code first scores the three raw T5 encoders of the
paper's panel from their cached vectors (``hidden_mean_tokempty``) and checks every
layer against the published baseline cells of
``runs/active/resubmit/results/phase_resubmit_results.csv`` and the per-model minima
the table prints (LaTa 0.496 at layer 6, PhilTa 0.538 at layer 10, mT5-base 0.654 at
layer 5).

Protocol (the paper's evaluator, ``scripts/resubmit/run_resubmit_evaluate.py``):
  * rows are aligned to the split by filename (``AlignmentResolver`` reads the
    ``meta.csv`` the extractor wrote beside the matrices), never by position;
  * no post-processing: cosine on the L2-normalized mean-pooled vectors;
  * Task A AUROC comes from ``run_resubmit_evaluate.evaluate_from_similarity``, the
    single definition of the reported metrics: sklearn roc_auc_score over the upper
    triangle of the 858 x 858 test cosine matrix, same- against different-directory.

Output: ``runs/active/reframe/p2x2/raw_auroc_layers.csv`` (one row per model-layer).
CPU only, well under a minute. Run from the repo root; the caches are gitignored, so
point ``--bases_root`` and ``--gen_bases`` at a checkout that has them:

  python scripts/paper/reframe/raw_auroc_layers.py \
      --bases_root /u/irowerojas/localLatin/runs/active/resubmit_bases \
      --gen_bases /u/irowerojas/localLatin/runs/active/reframe/gen/bases
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "resubmit"))

from canon_retrieval import l2_normalize, similarity_matrix  # noqa: E402
from embedding_alignment import AlignmentResolver  # noqa: E402
import run_resubmit_evaluate as paper_eval  # noqa: E402

SPLIT_CSV = Path("runs/active/resubmit/data/phase_resubmit_split.csv")
RES_CSV = Path("runs/active/resubmit/results/phase_resubmit_results.csv")
BASES_ROOT = Path("runs/active/resubmit_bases")
GEN_BASES = Path("runs/active/reframe/gen/bases")
OUT_CSV = Path("runs/active/reframe/p2x2/raw_auroc_layers.csv")
SUBDIR = "hidden_mean_tokempty"

# (display, HF id, source): "panel" reads the paper's cache, "gen" the #238 extraction.
MODELS = [
    ("LaTa", "bowphs/LaTa", "panel"),
    ("PhilTa", "bowphs/PhilTa", "panel"),
    ("mT5-base", "google/mt5-base", "panel"),
    ("T5-v1.1-base", "google/t5-v1_1-base", "gen"),
]
# The raw T5 minima tab:panel_2x2 already prints: (layer, AUROC to 3 decimals).
PRINTED_MIN = {"LaTa": (6, 0.496), "PhilTa": (10, 0.538), "mT5-base": (5, 0.654)}

LAYER_RE = re.compile(r"^hidden_layer(\d+)_embeddings\.npy$")


def run_dir(source: str, model_id: str, bases_root: Path, gen_bases: Path) -> Path:
    slug = model_id.replace("/", "_")
    if source == "panel":
        return bases_root / "phase9_bases" / slug / SUBDIR
    return gen_bases / slug / "latin"


def layers_in(directory: Path) -> List[int]:
    found = [int(m.group(1)) for p in directory.iterdir() if (m := LAYER_RE.match(p.name))]
    return sorted(found)


def task_a_auroc(emb: np.ndarray, split: pd.DataFrame) -> float:
    """Task A test AUROC of split-aligned vectors, through the paper's metric block."""
    tr = split["split"].values == "train"
    te = split["split"].values == "test"
    metrics = paper_eval.evaluate_from_similarity(
        train_sim=similarity_matrix(l2_normalize(emb[tr])),
        test_sim=similarity_matrix(l2_normalize(emb[te])),
        train_folder_ids=split.loc[tr, "folder_id"].values,
        test_folder_ids=split.loc[te, "folder_id"].values,
        test_has_partner=split.loc[te, "has_test_partner"].values.astype(bool),
    )
    return float(metrics["aucroc"])


def score_models(split: pd.DataFrame, bases_root: Path, gen_bases: Path,
                 names: Optional[Sequence[str]] = None,
                 layers: Optional[Sequence[int]] = None) -> pd.DataFrame:
    resolver = AlignmentResolver(split)
    rows: List[Dict] = []
    for name, model_id, source in MODELS:
        if names and name not in names:
            continue
        d = run_dir(source, model_id, bases_root, gen_bases)
        for layer in layers_in(d):
            if layers and layer not in layers:
                continue
            emb = resolver.load(d / f"hidden_layer{layer}_embeddings.npy")
            rows.append({"model": name, "model_id": model_id, "source": source,
                         "layer": layer, "aucroc": task_a_auroc(emb, split)})
            print(f"{name:13s} L{layer:<2d} AUROC {rows[-1]['aucroc']:.4f}", flush=True)
    print(resolver.summary())
    return pd.DataFrame(rows)


def minima(df: pd.DataFrame) -> pd.DataFrame:
    """Lowest AUROC per model and its layer (first layer on ties)."""
    idx = df.groupby("model", sort=False)["aucroc"].idxmin()
    return df.loc[idx, ["model", "layer", "aucroc"]].reset_index(drop=True)


def check_reproduction(df: pd.DataFrame, res_csv: Optional[Path]) -> List[str]:
    """Problems found against the printed minima and the published baseline cells."""
    problems: List[str] = []
    mins = minima(df).set_index("model")
    for name, (layer, value) in PRINTED_MIN.items():
        if name not in mins.index:
            continue
        got_layer, got = int(mins.loc[name, "layer"]), float(mins.loc[name, "aucroc"])
        if got_layer != layer or round(got, 3) != value:
            problems.append(f"{name}: printed {value} at L{layer}, got {got:.4f} at L{got_layer}")
    if res_csv is not None and res_csv.exists():
        pub = pd.read_csv(res_csv)
        pub = pub[(pub["repr"] == "hidden") & (pub["pooling"] == "mean")
                  & (pub["method"] == "baseline")].set_index(["model", "layer"])["aucroc"]
        for r in df[df["source"] == "panel"].itertuples():
            ref = float(pub.loc[(r.model_id, r.layer)])
            if abs(ref - r.aucroc) > 1e-6:
                problems.append(f"{r.model} L{r.layer}: published {ref:.6f}, got {r.aucroc:.6f}")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split_csv", type=Path, default=SPLIT_CSV)
    ap.add_argument("--res_csv", type=Path, default=RES_CSV)
    ap.add_argument("--bases_root", type=Path, default=BASES_ROOT)
    ap.add_argument("--gen_bases", type=Path, default=GEN_BASES)
    ap.add_argument("--models", nargs="*", default=None, help="display names, default all")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args(argv)

    split = pd.read_csv(args.split_csv)
    df = score_models(split, args.bases_root, args.gen_bases, args.models, args.layers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.10g")
    print(f"wrote {args.out} ({len(df)} rows)")
    for r in minima(df).itertuples():
        print(f"min {r.model}: {r.aucroc:.4f} (layer {r.layer})")
    problems = check_reproduction(df, args.res_csv)
    for p in problems:
        print("REPRODUCTION MISMATCH:", p)
    if not problems:
        print("reproduction: raw T5 panel cells match the published baseline and printed minima")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
