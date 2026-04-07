"""Phase 9 Experiment 1: evaluate extracted embeddings with post-processing methods.

For each (model, repr, pooling, layer):
- Load raw embeddings
- Apply 5 post-processing methods (baseline, sif_only, sif_abtt_fixed,
  sif_abtt_optimal, whitening)
- Learn threshold on train, evaluate on test-only retrieval
- Compute: AUCROC, Spearman, cosine stats, Acc@1/3, assignment accuracy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from canon_retrieval import (
    accuracy_at_k,
    l2_normalize,
    similarity_matrix,
    sweep_thresholds,
    upper_triangle,
    upper_triangle_labels,
)
from pair_evaluation import safe_auc_roc, safe_spearman
from sif_abtt import EmbeddingCleaner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 Experiment 1 evaluation.")
    parser.add_argument("--split_csv", required=True, help="Path to phase9_split.csv.")
    parser.add_argument("--runs_root", required=True, help="Root for extracted embeddings.")
    parser.add_argument("--out_csv", required=True, help="Output results CSV path.")
    parser.add_argument(
        "--models", default="bowphs/LaTa,bowphs/PhilTa",
        help="Comma-separated seq2seq model names.",
    )
    parser.add_argument(
        "--encoder_models", default="sentence-transformers/LaBSE",
        help="Comma-separated encoder-only model names.",
    )
    parser.add_argument(
        "--D_values", default="1,2,3,5,7,10",
        help="Comma-separated D values for ABTT R-sweep.",
    )
    parser.add_argument("--sif_a", type=float, default=0.001)
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def find_embedding_file(
    runs_root: Path, slug: str, repr_name: str, pooling: str, layer: int,
) -> Optional[Path]:
    """Locate the raw (un-normalized) embedding .npy file."""
    suffix = "" if pooling == "mean" else "_sif"
    fname = f"{repr_name}_layer{layer}_embeddings{suffix}.npy"
    # Check under phase9_bases/{slug}/{repr}_{pooling}/
    candidate = runs_root / "phase9_bases" / slug / f"{repr_name}_{pooling}" / fname
    if candidate.exists():
        return candidate
    # Fallback: directly under phase9_bases/{slug}/
    candidate2 = runs_root / "phase9_bases" / slug / fname
    if candidate2.exists():
        return candidate2
    return None


def learn_threshold_on_train(
    emb: np.ndarray,
    folder_ids: np.ndarray,
) -> float:
    """Build train similarity matrix, sweep thresholds, return best F1 threshold."""
    emb_norm = l2_normalize(emb)
    sim = similarity_matrix(emb_norm)
    sims = upper_triangle(sim)
    labels = upper_triangle_labels(folder_ids)
    thresholds = np.linspace(0, 1, 200)
    df = sweep_thresholds(sims, labels, thresholds)
    best_idx = df["f1"].idxmax()
    return float(df.loc[best_idx, "threshold"])


def compute_assignment_acc(
    sim: np.ndarray,
    has_test_partner: np.ndarray,
    tau: float,
) -> Tuple[float, float, float]:
    """Compute new vs existing assignment accuracy using threshold tau.

    For each test file:
    - max_sim = max(sim[i,j] for j != i)
    - "Existing" (has_test_partner=True): correct if max_sim >= tau
    - "New" (has_test_partner=False): correct if max_sim < tau
    """
    n = sim.shape[0]
    has_partner = np.asarray(has_test_partner, dtype=bool)

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

    existing_acc = existing_correct / existing_total if existing_total > 0 else 0.0
    new_acc = new_correct / new_total if new_total > 0 else 0.0
    overall = (existing_correct + new_correct) / n if n > 0 else 0.0
    return existing_acc, new_acc, overall


def evaluate_query_vs_directory(
    query_emb_norm: np.ndarray,
    ref_emb_norm: np.ndarray,
    query_folder_ids: np.ndarray,
    ref_folder_ids: np.ndarray,
    query_has_reference: np.ndarray,
    tau: float,
    top_k: int = 5,
) -> Dict:
    """Evaluate queries against reference directories using max-sim aggregation.

    For each query, similarity to a reference directory = max cosine sim to any
    file in that directory. Queries with has_reference=False should match "__NEW__".
    """
    # Rectangular similarity: (n_query, n_ref)
    rect_sim = query_emb_norm @ ref_emb_norm.T

    # Build directory -> column indices mapping
    ref_dir_members: Dict[str, List[int]] = {}
    for j, fid in enumerate(ref_folder_ids):
        ref_dir_members.setdefault(str(fid), []).append(j)

    ref_dir_labels = sorted(ref_dir_members.keys())
    n_query = len(query_folder_ids)
    n_ref_dirs = len(ref_dir_labels)

    # Query-to-directory similarity: (n_query, n_ref_dirs)
    dir_sim = np.full((n_query, n_ref_dirs), -np.inf)
    for d_idx, d_label in enumerate(ref_dir_labels):
        members = ref_dir_members[d_label]
        dir_sim[:, d_idx] = rect_sim[:, members].max(axis=1)

    # Evaluate each query
    acc_at_k_correct = np.zeros(top_k, dtype=int)
    existing_correct = 0
    existing_total = 0
    new_correct = 0
    new_total = 0

    for i in range(n_query):
        my_dir = str(query_folder_ids[i])
        has_ref = bool(query_has_reference[i])
        correct_label = my_dir if has_ref else "__NEW__"

        # Build scored entries: all ref dirs + __NEW__ at tau
        entries = []
        for d_idx, d_label in enumerate(ref_dir_labels):
            entries.append((float(dir_sim[i, d_idx]), d_label))
        entries.append((float(tau), "__NEW__"))

        # Sort descending by score, then by label for tie-breaking
        entries.sort(key=lambda x: (-x[0], x[1]))

        # Find rank of correct label
        rank = None
        for r, (_, label) in enumerate(entries):
            if label == correct_label:
                rank = r + 1  # 1-indexed
                break

        if rank is not None:
            for k in range(top_k):
                if rank <= k + 1:
                    acc_at_k_correct[k] += 1

        # Assignment accuracy
        top1_score, top1_label = entries[0]
        if has_ref:
            existing_total += 1
            if top1_score >= tau and top1_label == my_dir:
                existing_correct += 1
        else:
            new_total += 1
            if top1_label == "__NEW__":
                # All dirs scored below tau
                new_correct += 1

    result = {}
    for k in range(top_k):
        result[f"dir_acc_at_{k+1}"] = acc_at_k_correct[k] / n_query if n_query > 0 else 0.0

    result["existing_acc"] = existing_correct / existing_total if existing_total > 0 else 0.0
    result["new_acc"] = new_correct / new_total if new_total > 0 else 0.0
    result["overall_assignment_acc"] = (
        (existing_correct + new_correct) / n_query if n_query > 0 else 0.0
    )
    result["n_query"] = n_query
    result["n_reference_files"] = len(ref_folder_ids)
    result["n_reference_dirs"] = n_ref_dirs
    result["n_query_existing"] = existing_total
    result["n_query_new"] = new_total
    return result


def find_optimal_D(
    train_emb: np.ndarray,
    train_folder_ids: np.ndarray,
    D_values: List[int],
) -> Tuple[int, float]:
    """Sweep D values on train set, return best D by Assignment Accuracy."""
    best_D = D_values[0]
    best_score = -1.0

    # Determine which train files have at least one same-folder partner in train
    unique, counts = np.unique(train_folder_ids, return_counts=True)
    multi_folders = set(unique[counts > 1])
    has_partner = np.array([fid in multi_folders for fid in train_folder_ids])

    for D in D_values:
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_emb)
        cleaned = cleaner.transform(train_emb)
        cleaned_norm = l2_normalize(cleaned)
        sim = similarity_matrix(cleaned_norm)

        # Learn threshold τ via best F1
        sims = upper_triangle(sim)
        labels = upper_triangle_labels(train_folder_ids)
        thresh_df = sweep_thresholds(sims, labels, np.linspace(0, 1, 200))
        best_f1_idx = thresh_df["f1"].idxmax()
        tau = float(thresh_df.loc[best_f1_idx, "threshold"])

        # Compute assignment accuracy on train
        _, _, overall_acc = compute_assignment_acc(sim, has_partner, tau)
        if overall_acc > best_score:
            best_score = overall_acc
            best_D = D

    return best_D, best_score


def evaluate_single(
    emb_all: np.ndarray,
    split_meta: pd.DataFrame,
    method: str,
    D: int,
    sif_a: float,
    model_name: str,
    repr_name: str,
    pooling_src: str,
    layer: int,
    D_values: List[int],
) -> Optional[Dict]:
    """Run evaluation for one (model, repr, layer, method) combo."""
    train_mask = split_meta["split"].values == "train"
    test_mask = split_meta["split"].values == "test"
    train_folder_ids = split_meta.loc[train_mask, "folder_id"].values
    test_folder_ids = split_meta.loc[test_mask, "folder_id"].values
    test_has_partner = split_meta.loc[test_mask, "has_test_partner"].values.astype(bool)
    test_is_query = split_meta.loc[test_mask, "is_test_query"].values.astype(bool)

    train_emb = emb_all[train_mask]
    test_emb = emb_all[test_mask]

    actual_D = D
    actual_pooling = pooling_src

    # Apply post-processing
    if method == "baseline":
        # No post-processing, use raw embeddings (expected to be mean-pooled)
        pass
    elif method == "sif_only":
        # No post-processing, embeddings already SIF-pooled
        pass
    elif method == "sif_abtt_fixed":
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_emb)
        train_emb = cleaner.transform(train_emb)
        test_emb = cleaner.transform(test_emb)
    elif method == "sif_abtt_optimal":
        best_D, _ = find_optimal_D(train_emb, train_folder_ids, D_values)
        actual_D = best_D
        cleaner = EmbeddingCleaner(num_components=best_D, center=True)
        cleaner.fit(train_emb)
        train_emb = cleaner.transform(train_emb)
        test_emb = cleaner.transform(test_emb)
    elif method == "whitening":
        pca = PCA(whiten=True)
        pca.fit(train_emb)
        train_emb = pca.transform(train_emb)
        test_emb = pca.transform(test_emb)
    else:
        return None

    # Normalize
    train_emb_norm = l2_normalize(train_emb)
    test_emb_norm = l2_normalize(test_emb)

    # Threshold learning on train
    train_sim = similarity_matrix(train_emb_norm)
    train_sims = upper_triangle(train_sim)
    train_labels = upper_triangle_labels(train_folder_ids)
    thresh_df = sweep_thresholds(train_sims, train_labels, np.linspace(0, 1, 200))
    best_f1_idx = thresh_df["f1"].idxmax()
    tau = float(thresh_df.loc[best_f1_idx, "threshold"])

    # Test-only evaluation
    test_sim = similarity_matrix(test_emb_norm)
    test_sims = upper_triangle(test_sim)
    test_labels = upper_triangle_labels(test_folder_ids)

    aucroc = safe_auc_roc(test_sims, test_labels)
    spearman = safe_spearman(test_sims, test_labels.astype(float))

    # Cosine stats
    pos_mask = test_labels.astype(bool)
    same_avg = float(np.mean(test_sims[pos_mask])) if pos_mask.any() else float("nan")
    diff_avg = float(np.mean(test_sims[~pos_mask])) if (~pos_mask).any() else float("nan")
    gap = same_avg - diff_avg

    # Acc@k (only for test files that have a partner)
    acc1 = accuracy_at_k(test_sim, test_folder_ids, test_is_query, k=1)
    acc3 = accuracy_at_k(test_sim, test_folder_ids, test_is_query, k=3)

    # Assignment accuracy (file-level, legacy)
    existing_acc, new_acc, overall_acc = compute_assignment_acc(
        test_sim, test_has_partner, tau
    )

    n_existing = int(test_has_partner.sum())
    n_new = int(test_mask.sum()) - n_existing

    result = {
        "model": model_name,
        "repr": repr_name,
        "pooling": actual_pooling,
        "layer": layer,
        "method": method,
        "D": actual_D,
        "sif_a": sif_a,
        "tau": tau,
        "aucroc": aucroc,
        "spearman": spearman,
        "same_avg": same_avg,
        "diff_avg": diff_avg,
        "gap": gap,
        "acc_at_1": acc1,
        "acc_at_3": acc3,
        "existing_acc": existing_acc,
        "new_acc": new_acc,
        "overall_assignment_acc": overall_acc,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_existing": n_existing,
        "n_new": n_new,
    }

    # Query-vs-directory evaluation (new Task B paradigm)
    if "taskb_role" in split_meta.columns:
        test_roles = split_meta.loc[test_mask, "taskb_role"].values
        query_in_test = test_roles == "query"
        ref_in_test = test_roles == "reference"

        if query_in_test.any() and ref_in_test.any():
            query_emb_norm = test_emb_norm[query_in_test]
            ref_emb_norm = test_emb_norm[ref_in_test]
            query_fids = test_folder_ids[query_in_test]
            ref_fids = test_folder_ids[ref_in_test]
            query_has_ref = split_meta.loc[test_mask, "has_reference_dir"].values[query_in_test].astype(bool)

            qvd = evaluate_query_vs_directory(
                query_emb_norm, ref_emb_norm,
                query_fids, ref_fids, query_has_ref,
                tau, top_k=5,
            )
            for key, val in qvd.items():
                result[f"qvd_{key}"] = val

    return result


def discover_layers(
    runs_root: Path, slug: str, repr_name: str, pooling: str,
) -> List[int]:
    """Find which layers have embeddings extracted."""
    suffix = "" if pooling == "mean" else "_sif"
    pattern = f"{repr_name}_layer*_embeddings{suffix}.npy"
    base_dir = runs_root / "phase9_bases" / slug / f"{repr_name}_{pooling}"
    if not base_dir.exists():
        base_dir = runs_root / "phase9_bases" / slug
    if not base_dir.exists():
        return []

    layers = []
    for f in sorted(base_dir.glob(pattern)):
        # Extract layer number from filename
        name = f.stem  # e.g. hidden_layer6_embeddings_sif
        parts = name.split("_layer")
        if len(parts) == 2:
            rest = parts[1].split("_")[0]
            try:
                layers.append(int(rest))
            except ValueError:
                continue
    return sorted(layers)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    split_meta = pd.read_csv(args.split_csv)

    seq2seq_models = [m.strip() for m in args.models.split(",") if m.strip()]
    encoder_models = [m.strip() for m in args.encoder_models.split(",") if m.strip()]
    D_values = [int(d.strip()) for d in args.D_values.split(",")]

    results: List[Dict] = []

    # Define what representations/poolings to evaluate per model type
    # seq2seq: hidden + ff1, mean + sif
    # encoder-only: hidden only, mean + sif
    model_configs = []
    for m in seq2seq_models:
        model_configs.append((m, "seq2seq", ["hidden", "ff1"]))
    for m in encoder_models:
        model_configs.append((m, "encoder", ["hidden", "ffn_int"]))

    for model_name, model_type, reprs in model_configs:
        slug = model_slug(model_name)
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_type})")
        print(f"{'='*60}")

        for repr_name in reprs:
            for pooling in ["mean", "sif"]:
                layers = discover_layers(runs_root, slug, repr_name, pooling)
                if not layers:
                    print(f"  [{repr_name}/{pooling}] No layers found, skipping.")
                    continue
                print(f"  [{repr_name}/{pooling}] Layers: {layers}")

                for layer in layers:
                    emb_path = find_embedding_file(
                        runs_root, slug, repr_name, pooling, layer
                    )
                    if emb_path is None:
                        print(f"    Layer {layer}: embedding file not found, skipping.")
                        continue

                    emb_all = np.load(emb_path)
                    if emb_all.shape[0] != len(split_meta):
                        print(
                            f"    Layer {layer}: shape mismatch "
                            f"({emb_all.shape[0]} vs {len(split_meta)}), skipping."
                        )
                        continue

                    # Determine which methods to run based on pooling
                    if pooling == "mean":
                        methods = ["baseline", "whitening"]
                    else:
                        methods = ["sif_only", "sif_abtt_fixed", "sif_abtt_optimal"]

                    for method in methods:
                        print(f"    Layer {layer}, {method}...", end=" ", flush=True)
                        row = evaluate_single(
                            emb_all=emb_all,
                            split_meta=split_meta,
                            method=method,
                            D=10,
                            sif_a=args.sif_a,
                            model_name=model_name,
                            repr_name=repr_name,
                            pooling_src=pooling,
                            layer=layer,
                            D_values=D_values,
                        )
                        if row is not None:
                            results.append(row)
                            print(
                                f"AUCROC={row['aucroc']:.4f} "
                                f"Spearman={row['spearman']:.4f} "
                                f"gap={row['gap']:.4f}"
                            )
                        else:
                            print("skipped")

    # Save results
    if results:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} results to {out_path}")

        # Print summary: best AUCROC per model
        print("\nBest AUCROC per model:")
        for model_name in df["model"].unique():
            mdf = df[df["model"] == model_name]
            best = mdf.loc[mdf["aucroc"].idxmax()]
            print(
                f"  {model_name}: AUCROC={best['aucroc']:.4f} "
                f"({best['repr']}/L{best['layer']}/{best['method']})"
            )
    else:
        print("\nNo results generated. Check embedding paths.")


if __name__ == "__main__":
    main()
