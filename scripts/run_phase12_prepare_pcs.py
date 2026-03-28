"""Phase 12 Step 1: Pre-compute PCs from train embeddings for each model/layer.

Reads pre-extracted .npy embeddings from phase9_bases, filters to train split,
computes SVD, and saves top-10 PCs + mean vector as .npz files.

Output: runs/phase12/pcs/<model_slug>/layer{L}_pcs.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


MODEL_CONFIGS = {
    "bowphs/LaTa":       {"slug": "bowphs_LaTa",       "layers": list(range(1, 13)), "subdir": "hidden_mean"},
    "bowphs/PhilTa":     {"slug": "bowphs_PhilTa",     "layers": list(range(1, 13)), "subdir": "hidden_mean"},
    "sentence-transformers/LaBSE": {"slug": "sentence-transformers_LaBSE", "layers": list(range(1, 13)), "subdir": "hidden_mean"},
    "Qwen/Qwen3-Embedding-0.6B":  {"slug": "Qwen_Qwen3-Embedding-0.6B",  "layers": list(range(1, 29)), "subdir": "hidden_mean"},
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": {
        "slug": "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5",
        "layers": list(range(1, 25)),
        "subdir": "hidden_mean",
    },
    "google/mt5-base":   {"slug": "google_mt5-base",   "layers": list(range(1, 13)), "subdir": "hidden_mean"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute PCs for Phase 12 attribution.")
    parser.add_argument("--split_csv", required=True, help="Path to phase9_split.csv.")
    parser.add_argument("--runs_root", required=True, help="Root for extracted embeddings.")
    parser.add_argument("--out_dir", required=True, help="Output directory for PC files.")
    parser.add_argument("--D_max", type=int, default=10, help="Max number of PCs to save.")
    parser.add_argument(
        "--models", default="",
        help="Comma-separated model names (default: all 6 models).",
    )
    parser.add_argument(
        "--hidden_subdir",
        default="hidden_mean",
        help="Embedding subdirectory to read from for hidden-state PCs.",
    )
    parser.add_argument(
        "--fail_if_missing",
        action="store_true",
        help="Exit non-zero if any requested layer embedding is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_df = pd.read_csv(args.split_csv)
    train_mask = (split_df["split"] == "train").values

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = list(MODEL_CONFIGS.keys())

    runs_root = Path(args.runs_root)
    out_root = Path(args.out_dir)
    saved_total = 0
    missing_by_model: dict[str, list[int]] = {}

    for model_name in model_names:
        cfg = MODEL_CONFIGS[model_name]
        slug = cfg["slug"]
        emb_dir = runs_root / "phase9_bases" / slug / args.hidden_subdir
        model_out = out_root / slug
        model_out.mkdir(parents=True, exist_ok=True)
        missing_layers: list[int] = []

        print(f"\n=== {model_name} ({slug}) ===")
        for layer in cfg["layers"]:
            emb_path = emb_dir / f"hidden_layer{layer}_embeddings.npy"
            if not emb_path.exists():
                print(f"  WARNING: {emb_path} not found, skipping layer {layer}")
                missing_layers.append(layer)
                continue

            emb = np.load(emb_path)  # (n_files, hidden_dim)
            train_emb = emb[train_mask]

            mean_vec = train_emb.mean(axis=0).astype(np.float32)
            X_centered = train_emb - mean_vec
            _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
            pcs = vt[: args.D_max].astype(np.float32)

            out_path = model_out / f"layer{layer}_pcs.npz"
            np.savez(out_path, pcs=pcs, mean_vec=mean_vec)
            print(f"  Layer {layer}: pcs={pcs.shape}, saved to {out_path}")
            saved_total += 1

        if missing_layers:
            missing_by_model[model_name] = missing_layers

    if saved_total == 0:
        raise SystemExit("No PC files were written.")

    if args.fail_if_missing and missing_by_model:
        print("\nMissing embeddings detected:")
        for model_name, layers in missing_by_model.items():
            print(f"  {model_name}: missing layers {layers}")
        raise SystemExit("Missing requested hidden embeddings; refusing to continue.")

    print(f"\nDone. Wrote {saved_total} PC files.")


if __name__ == "__main__":
    main()
