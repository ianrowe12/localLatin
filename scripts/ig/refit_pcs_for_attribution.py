"""Refit ABTT principal components at D=10 for the attribution pipeline.

The legacy PC files at ``runs/phase12_release/pcs/<slug>/layer{L}_pcs.npz``
were extracted from old phase12c NPZs (via
``scripts/ig/extract_pcs_from_npzs.py``). Two issues with those files were
found in the 2026-04-26 cosine investigation (see
``docs/analyses/cosine_inflation_investigation.md``):

1. **Wrong D**: only the per-model legacy D values were preserved (LaTa D=2,
   LaBSE D=1, Qwen3-0.6B D=3). Per the 2026-03-31 meeting decision, the
   universal D=10 default matches ``abtt_optimal`` across all models.
2. **Wrong distribution**: the stored mean_vec / PCs were fit on
   token-keep-filtered phase12c data, while ``run_attribution_metrics.py``
   computes embeddings via plain mean-pool with no token filter. The
   old PC1 has cosine ~0.10 with a fresh PC1 fit on canon train (LaTa L4),
   and the mean_vec L2 differs by ~5700. Using the stale PCs to clean
   freshly-pooled vectors actively *inflates* cosines instead of cleaning
   them, which is why ``abtt full_cos_mean`` was 0.985 vs baseline 0.926
   in the 200-pair output.

This script refits PCs from the canonical TRAIN embeddings using the same
mean-pool path that ``run_attribution_metrics.forward_pooled`` produces,
and overwrites the PC file at the path the IG NPZ generator reads from.

Default usage matches the 200-pair pipeline (CANON dataset, phase9 split,
encoder_bases cache)::

    python scripts/ig/refit_pcs_for_attribution.py

For the canon_labelled / resubmit pipeline use::

    python scripts/ig/refit_pcs_for_attribution.py \\
        --bases_root runs/active/resubmit_bases/phase9_bases \\
        --pooling hidden_mean_tokempty \\
        --split_csv runs/active/resubmit/data/phase_resubmit_split.csv

After running, ``run_phase12e_pair_explanations.py`` will load 10 PCs
(line 229 slices ``pc_data["pcs"][:d_value]``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from sif_abtt import remove_top_components  # noqa: E402


# Layer per slug (mirrors scripts/ig/extract_pcs_from_npzs.py:MODEL_LAYER and
# the FEATURED_MODELS dict in scripts/ig/sample_random_test_pairs.py).
SLUG_LAYER = {
    "bowphs_LaTa": 4,
    "bowphs_PhilTa": 6,
    "sentence-transformers_LaBSE": 12,
    "Qwen_Qwen3-Embedding-0.6B": 23,
}


def refit_one(
    slug: str,
    layer: int,
    bases_root: Path,
    pooling: str,
    train_idx: np.ndarray,
    n_split_rows: int,
    pc_root: Path,
    d: int,
) -> None:
    emb_path = bases_root / slug / pooling / f"hidden_layer{layer}_embeddings.npy"
    if not emb_path.exists():
        raise SystemExit(f"Embedding cache missing: {emb_path}")
    emb = np.load(emb_path)
    if emb.shape[0] != n_split_rows:
        raise SystemExit(
            f"[{slug}] cache rows {emb.shape[0]} != split rows {n_split_rows}. "
            f"Likely wrong --bases_root / --split_csv combination."
        )
    train_emb = emb[train_idx].astype(np.float32)
    print(f"[{slug}] fitting D={d} on layer {layer} train ({train_emb.shape})")
    _, mean_vec, pcs = remove_top_components(train_emb, num_components=d, center=True)
    out = pc_root / slug / f"layer{layer}_pcs.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pcs=pcs, mean_vec=mean_vec)
    print(f"[{slug}] wrote {out}  pcs.shape={pcs.shape}  mean.shape={mean_vec.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="+", default=["bowphs_LaTa", "bowphs_PhilTa"],
                    help="model slugs to refit (must be keys of SLUG_LAYER)")
    ap.add_argument("--bases_root",
                    default="runs/active/encoder_bases",
                    help="Default matches the canon (phase9) cache used by the "
                         "200-pair attribution pipeline. Use "
                         "runs/active/resubmit_bases/phase9_bases for canon_labelled.")
    ap.add_argument("--pooling", default="hidden_mean",
                    help="pooling subdir under <bases_root>/<slug>/. Mean-pool "
                         "without SIF matches forward_pooled in the metrics script.")
    ap.add_argument("--split_csv",
                    default="runs/phase9/phase9_split.csv",
                    help="Default phase9 split (canon, 1278 rows). "
                         "Use runs/active/resubmit/data/phase_resubmit_split.csv "
                         "for canon_labelled (1705 rows).")
    ap.add_argument("--pc_root", default="runs/phase12_release/pcs")
    ap.add_argument("--d", type=int, default=10,
                    help="number of top PCs to fit (universal default per "
                         "2026-03-31 meeting)")
    args = ap.parse_args()

    split = pd.read_csv(args.split_csv)
    train_idx = np.where(split["split"] == "train")[0]
    n_split_rows = len(split)

    bases_root = Path(args.bases_root)
    pc_root = Path(args.pc_root)

    for slug in args.slugs:
        if slug not in SLUG_LAYER:
            raise SystemExit(f"Unknown slug {slug!r}; add to SLUG_LAYER")
        refit_one(slug, SLUG_LAYER[slug], bases_root, args.pooling,
                  train_idx, n_split_rows, pc_root, args.d)


if __name__ == "__main__":
    main()
