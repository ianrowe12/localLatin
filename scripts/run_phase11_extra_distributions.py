"""Compute cosine similarity distributions for Phase 11 density histogram figures.

Saves distributions for 6 conditions per model:
  1. Baseline (hidden, mean), last layer
  2. Baseline (hidden, mean), best middle layer (30-70% depth, by AUCROC)
  3. SIF+ABTT optimal (hidden, sif), last layer
  4. SIF+ABTT optimal (hidden, sif), best middle layer
  5. ABTT optimal (hidden, mean), last layer
  6. ABTT optimal (hidden, mean), best middle layer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_retrieval import l2_normalize, similarity_matrix, upper_triangle, upper_triangle_labels
from sif_abtt import EmbeddingCleaner

MAX_LAYERS = {
    "bowphs/LaTa": 12,
    "bowphs/PhilTa": 12,
    "sentence-transformers/LaBSE": 12,
    "Qwen/Qwen3-Embedding-0.6B": 28,
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": 24,
    "google/mt5-base": 12,
}


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def find_embedding_file(runs_root: Path, slug: str, repr_name: str, pooling: str, layer: int):
    suffix = "" if pooling == "mean" else "_sif"
    fname = f"{repr_name}_layer{layer}_embeddings{suffix}.npy"
    candidate = runs_root / "phase9_bases" / slug / f"{repr_name}_{pooling}" / fname
    if candidate.exists():
        return candidate
    candidate2 = runs_root / "phase9_bases" / slug / fname
    if candidate2.exists():
        return candidate2
    return None


def compute_and_save(emb_test, test_folder_ids, out_path):
    """Compute cosine sim on test set, split into same/diff, save as npz."""
    emb_norm = l2_normalize(emb_test)
    sim = similarity_matrix(emb_norm)
    sims = upper_triangle(sim)
    labels = upper_triangle_labels(test_folder_ids)
    pos = labels.astype(bool)
    np.savez_compressed(out_path, same_sims=sims[pos], diff_sims=sims[~pos])
    print(f"  Saved {out_path.name}  (same={pos.sum()}, diff={(~pos).sum()})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--dist_dir", required=True)
    parser.add_argument("--models", default="", help="Optional comma-separated model names.")
    parser.add_argument("--hidden_mean_subdir", default="hidden_mean")
    parser.add_argument("--hidden_sif_subdir", default="hidden_sif")
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    split_meta = pd.read_csv(args.split_csv)
    runs_root = Path(args.runs_root)
    dist_dir = Path(args.dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    test_mask = split_meta["split"].values == "test"
    train_mask = split_meta["split"].values == "train"
    test_folder_ids = split_meta.loc[test_mask, "folder_id"].values

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = list(dict.fromkeys(results["model"].tolist()))

    for model in models:
        slug = model_slug(model)
        max_l = MAX_LAYERS.get(model, int(results.loc[results["model"] == model, "layer"].max()))
        short = model.split("/")[-1]
        print(f"\n=== {short} ===")

        # Determine last layer and best middle layer
        base = results[
            (results["model"] == model)
            & (results["method"] == "baseline")
            & (results["repr"] == "hidden")
            & (results["pooling"] == "mean")
        ].copy()
        if base.empty:
            print(f"  No baseline rows found, skipping.")
            continue

        last_layer = int(base["layer"].max())
        base["pct"] = base["layer"] / max_l * 100
        middle = base[(base["pct"] >= 30) & (base["pct"] <= 70)]
        if middle.empty:
            mid_layer = max_l // 2
        else:
            mid_layer = int(middle.loc[middle["aucroc"].idxmax(), "layer"])

        # Get optimal D at those layers
        opt = results[
            (results["model"] == model)
            & (results["method"] == "sif_abtt_optimal")
            & (results["repr"] == "hidden")
        ]
        abtt_opt = results[
            (results["model"] == model)
            & (results["method"] == "abtt_optimal")
            & (results["repr"] == "hidden")
        ]

        def get_D(layer):
            row = opt[opt["layer"] == layer]
            return int(row["D"].iloc[0]) if not row.empty else 10

        def get_abtt_D(layer):
            row = abtt_opt[abtt_opt["layer"] == layer]
            return int(row["D"].iloc[0]) if not row.empty else 10

        print(f"  Last layer: L{last_layer}, Middle layer: L{mid_layer}")
        print(f"  D(last)={get_D(last_layer)}, D(mid)={get_D(mid_layer)}")

        # --- Condition 1: Baseline, last layer ---
        emb_path = runs_root / "phase9_bases" / slug / args.hidden_mean_subdir / f"hidden_layer{last_layer}_embeddings.npy"
        if not emb_path.exists():
            emb_path = find_embedding_file(runs_root, slug, "hidden", "mean", last_layer)
        if emb_path:
            emb = np.load(emb_path)
            compute_and_save(emb[test_mask], test_folder_ids,
                             dist_dir / f"{slug}_baseline_last.npz")

        # --- Condition 2: Baseline, middle layer ---
        emb_path = runs_root / "phase9_bases" / slug / args.hidden_mean_subdir / f"hidden_layer{mid_layer}_embeddings.npy"
        if not emb_path.exists():
            emb_path = find_embedding_file(runs_root, slug, "hidden", "mean", mid_layer)
        if emb_path:
            emb = np.load(emb_path)
            compute_and_save(emb[test_mask], test_folder_ids,
                             dist_dir / f"{slug}_baseline_middle.npz")

        # --- Condition 3: SIF+ABTT, last layer ---
        emb_path = runs_root / "phase9_bases" / slug / args.hidden_sif_subdir / f"hidden_layer{last_layer}_embeddings_sif.npy"
        if not emb_path.exists():
            emb_path = find_embedding_file(runs_root, slug, "hidden", "sif", last_layer)
        if emb_path:
            emb = np.load(emb_path)
            D = get_D(last_layer)
            cleaner = EmbeddingCleaner(num_components=D, center=True)
            cleaner.fit(emb[train_mask])
            test_cleaned = cleaner.transform(emb[test_mask])
            compute_and_save(test_cleaned, test_folder_ids,
                             dist_dir / f"{slug}_sif_abtt_last.npz")

        # --- Condition 4: SIF+ABTT, middle layer ---
        emb_path = runs_root / "phase9_bases" / slug / args.hidden_sif_subdir / f"hidden_layer{mid_layer}_embeddings_sif.npy"
        if not emb_path.exists():
            emb_path = find_embedding_file(runs_root, slug, "hidden", "sif", mid_layer)
        if emb_path:
            emb = np.load(emb_path)
            D = get_D(mid_layer)
            cleaner = EmbeddingCleaner(num_components=D, center=True)
            cleaner.fit(emb[train_mask])
            test_cleaned = cleaner.transform(emb[test_mask])
            compute_and_save(test_cleaned, test_folder_ids,
                             dist_dir / f"{slug}_sif_abtt_middle.npz")

        # --- Condition 5: ABTT-only, last layer ---
        emb_path = runs_root / "phase9_bases" / slug / args.hidden_mean_subdir / f"hidden_layer{last_layer}_embeddings.npy"
        if not emb_path.exists():
            emb_path = find_embedding_file(runs_root, slug, "hidden", "mean", last_layer)
        if emb_path:
            emb = np.load(emb_path)
            D = get_abtt_D(last_layer)
            cleaner = EmbeddingCleaner(num_components=D, center=True)
            cleaner.fit(emb[train_mask])
            test_cleaned = cleaner.transform(emb[test_mask])
            compute_and_save(test_cleaned, test_folder_ids,
                             dist_dir / f"{slug}_abtt_last.npz")

        # --- Condition 6: ABTT-only, middle layer ---
        emb_path = runs_root / "phase9_bases" / slug / args.hidden_mean_subdir / f"hidden_layer{mid_layer}_embeddings.npy"
        if not emb_path.exists():
            emb_path = find_embedding_file(runs_root, slug, "hidden", "mean", mid_layer)
        if emb_path:
            emb = np.load(emb_path)
            D = get_abtt_D(mid_layer)
            cleaner = EmbeddingCleaner(num_components=D, center=True)
            cleaner.fit(emb[train_mask])
            test_cleaned = cleaner.transform(emb[test_mask])
            compute_and_save(test_cleaned, test_folder_ids,
                             dist_dir / f"{slug}_abtt_middle.npz")

    print("\nDone.")


if __name__ == "__main__":
    main()
