"""Cross-dataset retrieval: predict directory labels for unlabelled files.

For each model, loads labelled embeddings as database and unlabelled as queries,
computes cosine similarity, and outputs top-K directory predictions per
unlabelled file.

Variants (``--variant``)
------------------------
Each variant is a different embedding pipeline. The variant alone decides which
embedding files are read and whether ABTT (All-But-The-Top / principal-component
removal) is applied:

===========  ============  ==========  ====================================
variant      pooling       ABTT        embedding file loaded
===========  ============  ==========  ====================================
``raw``      mean          no          ``hidden_layer{N}_embeddings.npy``
``abtt``     mean          yes         ``hidden_layer{N}_embeddings.npy``
``sif``      SIF           no          ``hidden_layer{N}_embeddings_sif.npy``
``sif_abtt`` SIF           yes         ``hidden_layer{N}_embeddings_sif.npy``
===========  ============  ==========  ====================================

Layer selection
---------------
The per-model layer is read from the evaluation results CSV
(``runs/active/resubmit/results/phase_resubmit_results.csv``), picking the row
with the highest ``overall_assignment_acc`` *among the methods that correspond
to the requested variant*:

* ``raw``      -> ``baseline`` rows
* ``abtt``     -> ``abtt_fixed`` / ``abtt_optimal`` rows
* ``sif``      -> ``sif_only`` rows
* ``sif_abtt`` -> ``sif_abtt_fixed`` / ``sif_abtt_optimal`` rows

So every variant runs at the best layer *for that variant*, which makes the four
output CSVs directly comparable. ``--layer_overrides`` pins a layer per model
(e.g. ``"Qwen/Qwen3-Embedding-0.6B=5"``) for reproducing an older run.

Note: the ``pooling`` column of the results CSV is deliberately ignored. It is
implied by the variant, and trusting it used to mis-pair the two (the previous
version forced SIF files whenever the winning method was ``abtt_*``, which are
mean-pooled rows).

Leak-free protocol
------------------
``EmbeddingCleaner`` is fitted on the *train* split only (per ``--split_csv``)
and then applied to the labelled database and the unlabelled queries. The D
value is swept over ``D_VALUES`` and chosen by assignment accuracy on train.

Output schema
-------------
``{out_dir}/unlabelled_predictions_{variant}.csv`` (all models concatenated) and
``{out_dir}/unlabelled_predictions_{variant}_{model_slug}.csv`` (per model), with
one row per (model, query):

    file_id, filename, file_path,
    rank1_dir, rank1_score, ..., rank{K}_dir, rank{K}_score,
    model, variant, layer, pooling

``rank{i}_dir`` is the labelled directory name, ``rank{i}_score`` the max cosine
similarity between the query and any file in that directory.

The legacy ``unlabelled_predictions.csv`` / ``unlabelled_predictions_{slug}.csv``
files (no ``variant`` column) are the frozen webapp input and are no longer
written by this script; every output filename now carries the variant. To refresh
the webapp input, copy ``unlabelled_predictions_sif_abtt.csv`` over it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from canon_retrieval import l2_normalize, upper_triangle, upper_triangle_labels, sweep_thresholds, similarity_matrix
from sif_abtt import EmbeddingCleaner

# Fallback layers if the results CSV is missing a model (repr is always "hidden").
FALLBACK_CONFIGS = [
    ("bowphs/LaTa", 6, "hidden"),
    ("bowphs/PhilTa", 1, "hidden"),
    ("sentence-transformers/LaBSE", 12, "hidden"),
    ("Qwen/Qwen3-Embedding-0.6B", 21, "hidden"),
    ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", 1, "hidden"),
    ("google/mt5-base", 1, "hidden"),
]

ALL_MODELS = [c[0] for c in FALLBACK_CONFIGS]
D_VALUES = [1, 2, 3, 5, 7, 10]
TOP_K = 10

# variant -> (pooling, apply_abtt, results-CSV methods used for layer selection)
VARIANTS: Dict[str, Tuple[str, bool, Tuple[str, ...]]] = {
    "raw": ("mean", False, ("baseline",)),
    "abtt": ("mean", True, ("abtt_fixed", "abtt_optimal")),
    "sif": ("sif", False, ("sif_only",)),
    "sif_abtt": ("sif", True, ("sif_abtt_fixed", "sif_abtt_optimal")),
}


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def find_best_config_from_results(results_csv: str, methods: Tuple[str, ...]) -> Dict[str, Tuple[int, str]]:
    """Best layer per model by overall_assignment_acc, restricted to `methods`.

    Returns {model_name: (layer, repr)}. The results-CSV `pooling` column is not
    consulted: pooling is determined by the variant (see module docstring).
    """
    df = pd.read_csv(results_csv)
    df = df[df["method"].isin(methods)]
    best = {}
    for model_name in ALL_MODELS:
        mdf = df[df["model"] == model_name]
        if len(mdf) == 0:
            continue
        idx = mdf["overall_assignment_acc"].idxmax()
        row = mdf.loc[idx]
        best[model_name] = (int(row["layer"]), row["repr"])
    return best


def parse_layer_overrides(spec: str) -> Dict[str, int]:
    """Parse 'model=layer,model=layer' into {model: layer}."""
    overrides: Dict[str, int] = {}
    if not spec:
        return overrides
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --layer_overrides entry {item!r}; expected 'model=layer'.")
        name, layer = item.rsplit("=", 1)
        overrides[name.strip()] = int(layer)
    return overrides


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
                        help="Evaluation results CSV for finding the best layer per model.")
    parser.add_argument("--out_dir", default="runs/active/resubmit/unlabelled")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--variant", default="sif_abtt", choices=sorted(VARIANTS),
                        help="Embedding pipeline: raw | abtt | sif | sif_abtt (see module docstring).")
    parser.add_argument("--layer_overrides", default="",
                        help="Pin layers, e.g. 'Qwen/Qwen3-Embedding-0.6B=5,google/mt5-base=1'.")
    return parser.parse_args()


def main():
    args = parse_args()
    variant = args.variant
    pooling, apply_abtt, sel_methods = VARIANTS[variant]
    print(f"Variant: {variant} (pooling={pooling}, abtt={apply_abtt}, "
          f"layer selected from methods {list(sel_methods)})")

    # Load metadata
    split_meta = pd.read_csv(args.split_csv)
    unlabelled_meta = pd.read_csv(args.unlabelled_meta)

    # Find best layer per model from results CSV, restricted to this variant's methods
    results_path = Path(args.results_csv)
    if results_path.exists():
        print(f"Reading best layers from {results_path}")
        best_configs = find_best_config_from_results(args.results_csv, sel_methods)
    else:
        print("Results CSV not found, using fallback configs.")
        best_configs = {}

    layer_overrides = parse_layer_overrides(args.layer_overrides)

    # Build model configs list
    model_configs = []
    for model_name, fallback_layer, fallback_repr in FALLBACK_CONFIGS:
        if model_name in best_configs:
            layer, repr_name = best_configs[model_name]
            source = "from results"
        else:
            layer, repr_name = fallback_layer, fallback_repr
            source = "fallback"
        if model_name in layer_overrides:
            layer = layer_overrides[model_name]
            source = "override"
        print(f"  {model_name}: layer={layer}, repr={repr_name}, pooling={pooling} ({source})")
        model_configs.append((model_name, layer, repr_name))

    # Filter to selected models
    if args.models != "all":
        selected = [m.strip() for m in args.models.split(",")]
        model_configs = [c for c in model_configs if c[0] in selected]

    all_predictions_dfs = []

    for model_name, opt_layer, repr_name in model_configs:
        slug = model_slug(model_name)
        print(f"\n{'='*60}")
        print(f"Model: {model_name} (layer {opt_layer}, {pooling}, variant {variant})")

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

        if apply_abtt:
            # ABTT: fit the cleaner on the train split only, then apply to all.
            train_mask = split_meta["split"].values == "train"
            train_emb = lab_emb[train_mask]
            train_folder_ids = split_meta.loc[train_mask, "folder_id"].values

            best_D = find_optimal_D(train_emb, train_folder_ids, D_VALUES)
            print(f"  Optimal D: {best_D}")

            cleaner = EmbeddingCleaner(num_components=best_D, center=True)
            cleaner.fit(train_emb)
            lab_emb = cleaner.transform(lab_emb)
            unlab_emb = cleaner.transform(unlab_emb)
        else:
            print("  No ABTT for this variant (cosine on raw pooled vectors).")

        lab_norm = l2_normalize(lab_emb)
        unlab_norm = l2_normalize(unlab_emb)

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
            row["model"] = model_name
            row["variant"] = variant
            row["layer"] = opt_layer
            row["pooling"] = pooling
            rows.append(row)

        # Save per-model predictions
        per_model_path = Path(args.out_dir) / f"unlabelled_predictions_{variant}_{slug}.csv"
        per_model_path.parent.mkdir(parents=True, exist_ok=True)
        model_df = pd.DataFrame(rows)
        model_df.to_csv(per_model_path, index=False)
        print(f"  Saved {len(rows)} predictions -> {per_model_path}")

        all_predictions_dfs.append(model_df)

    # Save combined predictions
    if all_predictions_dfs:
        combined = pd.concat(all_predictions_dfs, ignore_index=True)
        combined_path = Path(args.out_dir) / f"unlabelled_predictions_{variant}.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined predictions ({len(combined)} rows) -> {combined_path}")


if __name__ == "__main__":
    main()
