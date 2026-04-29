"""Extract PCs + mean_vec from existing pair-example NPZs into the PC-root path
expected by run_phase12e_pair_explanations.py.

The NPZ-producing script loads PCs from
    <pc_root>/<model_slug>/layer{L}_pcs.npz

but that path no longer exists on disk (it was pruned). Every NPZ in
runs/active/ig_examples/artifacts/<slug>/ already stores the PCs used for
ABTT, so we just copy them out. All 20 NPZs per model share the same (model, layer)
PCs by construction, so we pick the first one we find per slug.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MODEL_LAYER = {
    "bowphs_LaTa": 4,
    "bowphs_PhilTa": 6,
    "sentence-transformers_LaBSE": 12,
    "Qwen_Qwen3-Embedding-0.6B": 23,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_root", default="runs/active/ig_examples/artifacts")
    ap.add_argument("--pc_root", default="runs/phase12_release/pcs")
    ap.add_argument("--slugs", nargs="*", default=list(MODEL_LAYER.keys()))
    args = ap.parse_args()

    artifacts_root = Path(args.artifacts_root)
    pc_root = Path(args.pc_root)

    for slug in args.slugs:
        if slug not in MODEL_LAYER:
            raise SystemExit(f"Unknown slug {slug!r}; add to MODEL_LAYER")
        layer = MODEL_LAYER[slug]
        model_dir = artifacts_root / slug
        candidates = sorted(model_dir.glob("example*_pair_example.npz"))
        if not candidates:
            raise SystemExit(f"No NPZs under {model_dir}")
        src = candidates[0]
        data = np.load(src)
        pcs = data["pcs"]
        mean_vec = data["mean_vec"]
        out = pc_root / slug / f"layer{layer}_pcs.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, pcs=pcs, mean_vec=mean_vec)
        print(f"{slug:45s} layer={layer} pcs={tuple(pcs.shape)} mean_vec={tuple(mean_vec.shape)} -> {out}")


if __name__ == "__main__":
    main()
