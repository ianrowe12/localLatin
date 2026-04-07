"""Cross-dataset retrieval: predict directory labels for unlabelled files.

For each model, loads labelled embeddings as database and unlabelled as queries.
Reads the evaluation results CSV to find the best layer+method per model.
Applies ABTT post-processing (fitted on labelled train), computes cosine similarity,
and outputs top-K directory predictions per unlabelled file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from canon_retrieval import l2_normalize, upper_triangle, upper_triangle_labels, sweep_thresholds, similarity_matrix
from sif_abtt import EmbeddingCleaner

# Fallback configs if results CSV is missing a model
FALLBACK_CONFIGS = [
    ("bowphs/LaTa", 6, "hidden", "sif"),
    ("bowphs/PhilTa", 1, "hidden", "sif"),
    ("sentence-transformers/LaBSE", 12, "hidden", "sif"),
    ("Qwen/Qwen3-Embedding-0.6B", 21, "hidden", "sif"),
    ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", 1, "hidden", "sif"),
    ("google/mt5-base", 1, "hidden", "sif"),
]

ALL_MODELS = [c[0] for c in FALLBACK_CONFIGS]
D_VALUES = [1, 2, 3, 5, 7, 10]
TOP_K = 10


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def find_best_config_from_results(results_csv: str) -> Dict[str, Tuple[int, str, str]]:
    """Read results CSV and find best layer+method per model by overall_assignment_acc."""
    df = pd.read_csv(results_csv)
    # Exclude whitening (consistently fails)
    df = df[df["method"] != "whitening"]
    best = {}
    for model_name in ALL_MODELS:
        mdf = df[df["model"] == model_name]
        if len(mdf) == 0:
            continue
        idx = mdf["overall_assignment_acc"].idxmax()
        row = mdf.loc[idx]
        pooling = "sif" if row["method"] in ("abtt_fixed", "abtt_optimal") else row["pooling"]
        best[model_name] = (int(row["layer"]), row["repr"], pooling)
    return best


def find_optimal_D(train_emb, train_folder_ids, D_values):
    """Sweep D values on train set, return best D by assignment accuracy."""
    best_D = D_values[0]
    best_score = -1.0
    unique, counts = np.unique(train_folder_ids, return_counts=True)
    multi_folders = set(unique[counts > 1])
    has_partner = np.array([fid in multi_folders for fid in train_folder_ids])

    for D in D_values:
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_emb)
        cleaned = cleaner.transform(train_emb)
        cleaned_norm = l2_normalize(cleaned)
        sim = similarity_matrix(cleaned_norm)

        sims = upper_triangle(sim)
        labels = upper_triangle_labels(train_folder_ids)
        thresh_df = sweep_thresholds(sims, labels, np.linspace(0, 1, 200))
        best_f1_idx = thresh_df["f1"].idxmax()
        tau = float(thresh_df.loc[best_f1_idx, "threshold"])

        # Compute assignment accuracy
        n = sim.shape[0]
        existing_correct = 0
        existing_total = 0
        new_correct = 0
        new_total = 0
        for i in range(n):
            scores = sim[i].copy()
            scores[i] = -np.inf
            max_sim = float(np.max(scores))
            if has_partner[i]:
                existing_total += 1
                if max_sim >= tau:
                    existing_correct += 1
            else:
                new_total += 1
                if max_sim < tau:
                    new_correct += 1
        overall = (existing_correct + new_correct) / n if n > 0 else 0.0

        print(f"    D={D:>2d}  tau={tau:.4f}  assign_acc={overall:.4f}"
              f"  (exist={existing_correct}/{existing_total},"
              f" new={new_correct}/{new_total})")

        if overall > best_score:
            best_score = overall
            best_D = D

    return best_D


def predict_directories(
    labelled_emb_norm: np.ndarray,
    unlabelled_emb_norm: np.ndarray,
    folder_ids: np.ndarray,
    top_k: int = 10,
) -> List[List[Tuple[str, float]]]:
    """For each unlabelled file, find top-K directories by max cosine similarity."""
    # Build directory -> file indices mapping
    dir_to_indices: Dict[str, List[int]] = {}
    for i, fid in enumerate(folder_ids):
        dir_to_indices.setdefault(str(fid), []).append(i)

    # Cross-similarity: (n_unlabelled, n_labelled)
    cross_sim = unlabelled_emb_norm @ labelled_emb_norm.T

    predictions = []
    for q_idx in range(cross_sim.shape[0]):
        q_sims = cross_sim[q_idx]
        dir_scores = []
        for dir_name, file_indices in dir_to_indices.items():
            max_sim = float(np.max(q_sims[file_indices]))
            dir_scores.append((dir_name, max_sim))
        dir_scores.sort(key=lambda x: x[1], reverse=True)
        predictions.append(dir_scores[:top_k])

    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Unlabelled -> labelled cross-dataset retrieval.")
    parser.add_argument("--labelled_bases", default="runs/active/resubmit_bases/phase9_bases",
                        help="Root for labelled embeddings.")
    parser.add_argument("--unlabelled_bases", default="runs/active/resubmit/unlabelled/bases",
                        help="Root for unlabelled embeddings.")
    parser.add_argument("--split_csv", default="runs/active/resubmit/data/phase_resubmit_split.csv")
    parser.add_argument("--unlabelled_meta", default="runs/active/resubmit/data/unlabelled_meta.csv")
    parser.add_argument("--results_csv", default="runs/active/resubmit/results/phase_resubmit_results.csv",
                        help="Evaluation results CSV for finding best model+layer+method.")
    parser.add_argument("--out_dir", default="runs/active/resubmit/unlabelled")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    return parser.parse_args()


def main():
    args = parse_args()
    # Load metadata
    split_meta = pd.read_csv(args.split_csv)
    unlabelled_meta = pd.read_csv(args.unlabelled_meta)

    # Find best configs from results CSV
    results_path = Path(args.results_csv)
    if results_path.exists():
        print(f"Reading best configs from {results_path}")
        best_configs = find_best_config_from_results(args.results_csv)
    else:
        print("Results CSV not found, using fallback configs.")
        best_configs = {}

    # Build model configs list
    model_configs = []
    for model_name, fallback_layer, fallback_repr, fallback_pooling in FALLBACK_CONFIGS:
        if model_name in best_configs:
            layer, repr_name, pooling = best_configs[model_name]
            print(f"  {model_name}: layer={layer}, repr={repr_name}, pooling={pooling} (from results)")
        else:
            layer, repr_name, pooling = fallback_layer, fallback_repr, fallback_pooling
            print(f"  {model_name}: layer={layer}, repr={repr_name}, pooling={pooling} (fallback)")
        model_configs.append((model_name, layer, repr_name, pooling))

    # Filter to selected models
    if args.models != "all":
        selected = [m.strip() for m in args.models.split(",")]
        model_configs = [c for c in model_configs if c[0] in selected]

    all_predictions_dfs = []

    for model_name, opt_layer, repr_name, pooling in model_configs:
        slug = model_slug(model_name)
        print(f"\n{'='*60}")
        print(f"Model: {model_name} (layer {opt_layer}, {pooling})")

        # Determine suffix based on pooling
        suffix = "_sif" if pooling == "sif" else ""
        subdir = f"{repr_name}_{pooling}_tokempty"
        fname = f"{repr_name}_layer{opt_layer}_embeddings{suffix}.npy"

        lab_path = Path(args.labelled_bases) / slug / subdir / fname
        unlab_path = Path(args.unlabelled_bases) / slug / subdir / fname

        if not lab_path.exists():
            print(f"  Labelled embeddings not found: {lab_path}, skipping.")
            continue
        if not unlab_path.exists():
            print(f"  Unlabelled embeddings not found: {unlab_path}, skipping.")
            continue

        lab_emb = np.load(lab_path)
        unlab_emb = np.load(unlab_path)
        print(f"  Labelled: {lab_emb.shape}, Unlabelled: {unlab_emb.shape}")

        # Validate shapes
        if lab_emb.shape[0] != len(split_meta):
            print(f"  Shape mismatch: {lab_emb.shape[0]} vs {len(split_meta)} files, skipping.")
            continue
        if unlab_emb.shape[0] != len(unlabelled_meta):
            print(f"  Shape mismatch: {unlab_emb.shape[0]} vs {len(unlabelled_meta)} files, skipping.")
            continue

        # ABTT: fit on train, apply to all
        train_mask = split_meta["split"].values == "train"
        train_emb = lab_emb[train_mask]
        train_folder_ids = split_meta.loc[train_mask, "folder_id"].values

        best_D = find_optimal_D(train_emb, train_folder_ids, D_VALUES)
        print(f"  Optimal D: {best_D}")

        cleaner = EmbeddingCleaner(num_components=best_D, center=True)
        cleaner.fit(train_emb)
        lab_cleaned = cleaner.transform(lab_emb)
        unlab_cleaned = cleaner.transform(unlab_emb)

        lab_norm = l2_normalize(lab_cleaned)
        unlab_norm = l2_normalize(unlab_cleaned)

        # Predict top-K directories
        folder_ids = split_meta["folder_id"].values
        predictions = predict_directories(lab_norm, unlab_norm, folder_ids, top_k=args.top_k)

        # Build output DataFrame
        rows = []
        for i, preds in enumerate(predictions):
            row = {
                "file_id": int(unlabelled_meta.iloc[i]["file_id"]),
                "filename": unlabelled_meta.iloc[i]["filename"],
                "file_path": unlabelled_meta.iloc[i]["path"],
            }
            for rank, (dir_name, score) in enumerate(preds, 1):
                row[f"rank{rank}_dir"] = dir_name
                row[f"rank{rank}_score"] = round(float(score), 6)
            rows.append(row)

        # Save per-model predictions
        per_model_path = Path(args.out_dir) / f"unlabelled_predictions_{slug}.csv"
        per_model_path.parent.mkdir(parents=True, exist_ok=True)
        model_df = pd.DataFrame(rows)
        model_df.to_csv(per_model_path, index=False)
        print(f"  Saved {len(rows)} predictions -> {per_model_path}")

        # Also collect for combined output
        for row in rows:
            row["model"] = model_name
            row["layer"] = opt_layer
            row["pooling"] = pooling
        all_predictions_dfs.append(pd.DataFrame(rows))

    # Save combined predictions
    if all_predictions_dfs:
        combined = pd.concat(all_predictions_dfs, ignore_index=True)
        combined_path = Path(args.out_dir) / "unlabelled_predictions.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined predictions ({len(combined)} rows) -> {combined_path}")


if __name__ == "__main__":
    main()
