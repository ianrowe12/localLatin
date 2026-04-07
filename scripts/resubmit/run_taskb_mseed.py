"""Run Task B evaluation M times with different random seeds.

For each seed:
  1. Generate query/reference split (train/test split stays fixed)
  2. Run evaluation across all models
  3. Collect per-seed metrics

Then aggregate: mean +/- std for each metric across M seeds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_retrieval import l2_normalize, similarity_matrix, sweep_thresholds, upper_triangle, upper_triangle_labels
from canon_split_v2 import (
    build_meta_with_split_v2,
    canon_taskb_query_reference_split,
    taskb_split_summary,
)
from sif_abtt import EmbeddingCleaner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M-seed Task B evaluation with query/reference split."
    )
    parser.add_argument("--canon_root", default="data/canon_labelled")
    parser.add_argument("--out_dir", default="runs/active/resubmit/taskb_mseed")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="First seed; will use base_seed..base_seed+M-1")
    parser.add_argument("--M", type=int, default=5, help="Number of seed repetitions")
    parser.add_argument("--split_csv", default=None,
                        help="Pre-computed train/test split CSV (without taskb columns)")
    parser.add_argument("--runs_root", required=True,
                        help="Root for extracted embeddings (contains phase9_bases/)")
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
        help="Comma-separated D values for ABTT sweep.",
    )
    parser.add_argument("--sif_a", type=float, default=0.001)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--hidden_mean_subdir", default="hidden_mean_tokempty",
        help="Subdirectory for hidden/mean embeddings.",
    )
    parser.add_argument(
        "--hidden_sif_subdir", default="hidden_sif_tokempty",
        help="Subdirectory for hidden/sif embeddings.",
    )
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def find_embedding_file(
    runs_root: Path, slug: str, repr_name: str, pooling: str, layer: int,
    subdir_override: Optional[str] = None,
) -> Optional[Path]:
    suffix = "" if pooling == "mean" else "_sif"
    fname = f"{repr_name}_layer{layer}_embeddings{suffix}.npy"
    subdir = subdir_override or f"{repr_name}_{pooling}"
    candidate = runs_root / "phase9_bases" / slug / subdir / fname
    if candidate.exists():
        return candidate
    fallback = runs_root / "phase9_bases" / slug / fname
    if fallback.exists():
        return fallback
    return None


def discover_layers(
    runs_root: Path, slug: str, repr_name: str, pooling: str,
    subdir_override: Optional[str] = None,
) -> List[int]:
    suffix = "" if pooling == "mean" else "_sif"
    pattern = f"{repr_name}_layer*_embeddings{suffix}.npy"
    subdir = subdir_override or f"{repr_name}_{pooling}"
    base_dir = runs_root / "phase9_bases" / slug / subdir
    if not base_dir.exists():
        base_dir = runs_root / "phase9_bases" / slug
    if not base_dir.exists():
        return []
    layers = []
    for f in sorted(base_dir.glob(pattern)):
        parts = f.stem.split("_layer")
        if len(parts) == 2:
            rest = parts[1].split("_")[0]
            try:
                layers.append(int(rest))
            except ValueError:
                continue
    return sorted(layers)


def learn_threshold_on_train(emb: np.ndarray, folder_ids: np.ndarray) -> float:
    emb_norm = l2_normalize(emb)
    sim = similarity_matrix(emb_norm)
    sims = upper_triangle(sim)
    labels = upper_triangle_labels(folder_ids)
    thresholds = np.linspace(0, 1, 200)
    df = sweep_thresholds(sims, labels, thresholds)
    best_idx = df["f1"].idxmax()
    return float(df.loc[best_idx, "threshold"])


def find_optimal_D(
    train_emb: np.ndarray, train_folder_ids: np.ndarray, D_values: List[int],
) -> int:
    unique, counts = np.unique(train_folder_ids, return_counts=True)
    multi_folders = set(unique[counts > 1])
    has_partner = np.array([fid in multi_folders for fid in train_folder_ids])

    best_D = D_values[0]
    best_score = -1.0
    for D in D_values:
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_emb)
        cleaned = cleaner.transform(train_emb)
        cleaned_norm = l2_normalize(cleaned)
        sim = similarity_matrix(cleaned_norm)
        sims = upper_triangle(sim)
        labels = upper_triangle_labels(train_folder_ids)
        thresh_df = sweep_thresholds(sims, labels, np.linspace(0, 1, 200))
        tau = float(thresh_df.loc[thresh_df["f1"].idxmax(), "threshold"])

        # Assignment accuracy on train
        n = sim.shape[0]
        correct = 0
        for i in range(n):
            scores = sim[i].copy()
            scores[i] = -np.inf
            max_sim = float(np.max(scores))
            if has_partner[i]:
                if max_sim >= tau:
                    correct += 1
            else:
                if max_sim < tau:
                    correct += 1
        overall_acc = correct / n if n > 0 else 0.0
        if overall_acc > best_score:
            best_score = overall_acc
            best_D = D
    return best_D


def evaluate_query_vs_directory(
    query_emb_norm: np.ndarray,
    ref_emb_norm: np.ndarray,
    query_folder_ids: np.ndarray,
    ref_folder_ids: np.ndarray,
    query_has_reference: np.ndarray,
    tau: float,
    top_k: int = 5,
) -> Dict:
    """Evaluate queries against reference directories using max-sim aggregation."""
    rect_sim = query_emb_norm @ ref_emb_norm.T

    ref_dir_members: Dict[str, List[int]] = {}
    for j, fid in enumerate(ref_folder_ids):
        ref_dir_members.setdefault(str(fid), []).append(j)

    ref_dir_labels = sorted(ref_dir_members.keys())
    n_query = len(query_folder_ids)
    n_ref_dirs = len(ref_dir_labels)

    dir_sim = np.full((n_query, n_ref_dirs), -np.inf)
    for d_idx, d_label in enumerate(ref_dir_labels):
        members = ref_dir_members[d_label]
        dir_sim[:, d_idx] = rect_sim[:, members].max(axis=1)

    acc_at_k_correct = np.zeros(top_k, dtype=int)
    existing_correct = 0
    existing_total = 0
    new_correct = 0
    new_total = 0

    for i in range(n_query):
        my_dir = str(query_folder_ids[i])
        has_ref = bool(query_has_reference[i])
        correct_label = my_dir if has_ref else "__NEW__"

        entries = []
        for d_idx, d_label in enumerate(ref_dir_labels):
            entries.append((float(dir_sim[i, d_idx]), d_label))
        entries.append((float(tau), "__NEW__"))
        entries.sort(key=lambda x: (-x[0], x[1]))

        rank = None
        for r, (_, label) in enumerate(entries):
            if label == correct_label:
                rank = r + 1
                break

        if rank is not None:
            for k in range(top_k):
                if rank <= k + 1:
                    acc_at_k_correct[k] += 1

        top1_score, top1_label = entries[0]
        if has_ref:
            existing_total += 1
            if top1_score >= tau and top1_label == my_dir:
                existing_correct += 1
        else:
            new_total += 1
            if top1_label == "__NEW__":
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
    result["n_reference_dirs"] = n_ref_dirs
    result["n_query_existing"] = existing_total
    result["n_query_new"] = new_total
    return result


def evaluate_model_for_seed(
    emb_all: np.ndarray,
    meta: pd.DataFrame,
    model_name: str,
    repr_name: str,
    pooling: str,
    layer: int,
    method: str,
    D_values: List[int],
    sif_a: float,
    top_k: int,
) -> Optional[Dict]:
    """Run one (model, repr, layer, method) evaluation with query-vs-directory."""
    train_mask = meta["split"].values == "train"
    query_mask = meta["taskb_role"].values == "query"
    ref_mask = meta["taskb_role"].values == "reference"

    train_emb = emb_all[train_mask]
    query_emb = emb_all[query_mask]
    ref_emb = emb_all[ref_mask]
    train_folder_ids = meta.loc[train_mask, "folder_id"].values

    actual_D = 10

    # Apply post-processing: fit on train, transform query+ref together
    combined = np.concatenate([query_emb, ref_emb], axis=0)
    if method == "baseline" or method == "sif_only":
        pass
    elif method == "sif_abtt_fixed":
        cleaner = EmbeddingCleaner(num_components=10, center=True)
        cleaner.fit(train_emb)
        train_emb = cleaner.transform(train_emb)
        combined = cleaner.transform(combined)
    elif method in ("sif_abtt_optimal", "abtt_optimal"):
        best_D = find_optimal_D(train_emb, train_folder_ids, D_values)
        actual_D = best_D
        cleaner = EmbeddingCleaner(num_components=best_D, center=True)
        cleaner.fit(train_emb)
        train_emb = cleaner.transform(train_emb)
        combined = cleaner.transform(combined)
    elif method in ("abtt_fixed", "sif_abtt_fixed"):
        cleaner = EmbeddingCleaner(num_components=10, center=True)
        cleaner.fit(train_emb)
        train_emb = cleaner.transform(train_emb)
        combined = cleaner.transform(combined)
    elif method == "whitening":
        pca = PCA(whiten=True)
        pca.fit(train_emb)
        train_emb = pca.transform(train_emb)
        combined = pca.transform(combined)
    else:
        return None

    n_q = len(query_emb)
    query_emb_pp = combined[:n_q]
    ref_emb_pp = combined[n_q:]

    # Learn threshold on train
    tau = learn_threshold_on_train(train_emb, train_folder_ids)

    # Evaluate
    query_emb_norm = l2_normalize(query_emb_pp)
    ref_emb_norm = l2_normalize(ref_emb_pp)

    query_fids = meta.loc[query_mask, "folder_id"].values
    ref_fids = meta.loc[ref_mask, "folder_id"].values
    query_has_ref = meta.loc[query_mask, "has_reference_dir"].values.astype(bool)

    qvd = evaluate_query_vs_directory(
        query_emb_norm, ref_emb_norm,
        query_fids, ref_fids, query_has_ref,
        tau, top_k=top_k,
    )

    result = {
        "model": model_name,
        "repr": repr_name,
        "pooling": pooling,
        "layer": layer,
        "method": method,
        "D": actual_D,
        "tau": tau,
    }
    result.update(qvd)
    return result


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.base_seed, args.base_seed + args.M))
    D_values = [int(d.strip()) for d in args.D_values.split(",")]

    # Build or load the base train/test split (FIXED, same for all seeds)
    if args.split_csv:
        base_meta = pd.read_csv(args.split_csv)
    else:
        print("Building base train/test split (seed=42, fixed)...")
        base_meta = build_meta_with_split_v2(args.canon_root, random_seed=42)

    # Collect all model configs
    seq2seq_models = [m.strip() for m in args.models.split(",") if m.strip()]
    encoder_models = [m.strip() for m in args.encoder_models.split(",") if m.strip()]

    model_configs = []
    for m in seq2seq_models:
        model_configs.append((m, ["hidden", "ff1"]))
    for m in encoder_models:
        model_configs.append((m, ["hidden"]))

    all_seed_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({seeds.index(seed)+1}/{len(seeds)})")
        print(f"{'='*60}")

        # Apply query/reference split with this seed
        meta = canon_taskb_query_reference_split(base_meta, random_seed=seed)

        # Save per-seed split
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        meta.to_csv(seed_dir / "split.csv", index=False)

        tb_summary = taskb_split_summary(meta)
        print(f"  Queries: {tb_summary['n_queries']} "
              f"({tb_summary['n_queries_with_reference']} existing, "
              f"{tb_summary['n_queries_none']} none)")
        print(f"  Reference dirs: {tb_summary['n_reference_directories']}")

        seed_results = []

        for model_name, reprs in model_configs:
            slug = model_slug(model_name)
            print(f"\n  Model: {model_name}")

            for repr_name in reprs:
                for pooling in ["mean", "sif"]:
                    subdir = args.hidden_mean_subdir if pooling == "mean" else args.hidden_sif_subdir
                    layers = discover_layers(runs_root, slug, repr_name, pooling,
                                             subdir_override=subdir)
                    if not layers:
                        continue

                    # Methods based on pooling
                    if pooling == "mean":
                        methods = ["baseline"]
                    else:
                        methods = ["sif_only", "sif_abtt_fixed", "sif_abtt_optimal"]

                    for layer in layers:
                        emb_path = find_embedding_file(
                            runs_root, slug, repr_name, pooling, layer,
                            subdir_override=subdir,
                        )
                        if emb_path is None:
                            continue

                        emb_all = np.load(emb_path)
                        if emb_all.shape[0] != len(meta):
                            print(f"    Layer {layer}: shape mismatch, skipping.")
                            continue

                        for method in methods:
                            row = evaluate_model_for_seed(
                                emb_all=emb_all,
                                meta=meta,
                                model_name=model_name,
                                repr_name=repr_name,
                                pooling=pooling,
                                layer=layer,
                                method=method,
                                D_values=D_values,
                                sif_a=args.sif_a,
                                top_k=args.top_k,
                            )
                            if row is not None:
                                seed_results.append(row)
                                print(
                                    f"    L{layer}/{method}: "
                                    f"dir@1={row['dir_acc_at_1']:.3f} "
                                    f"assign={row['overall_assignment_acc']:.3f}"
                                )

        # Save per-seed results
        if seed_results:
            seed_df = pd.DataFrame(seed_results)
            seed_df.to_csv(seed_dir / "results.csv", index=False)

        for row in seed_results:
            row["seed"] = seed
        all_seed_results.extend(seed_results)

    # Aggregate across seeds
    if all_seed_results:
        all_df = pd.DataFrame(all_seed_results)
        all_df.to_csv(out_dir / "all_seeds_results.csv", index=False)

        agg_df = aggregate_results(all_df, args.top_k)
        agg_df.to_csv(out_dir / "aggregated_results.csv", index=False)

        print(f"\n{'='*60}")
        print(f"Aggregated results across {len(seeds)} seeds")
        print(f"{'='*60}")
        print(f"Saved {len(all_df)} rows to {out_dir / 'all_seeds_results.csv'}")
        print(f"Saved {len(agg_df)} aggregated rows to {out_dir / 'aggregated_results.csv'}")

        # Print summary table
        print(f"\nBest overall_assignment_acc per model (mean ± std):")
        for model_name in agg_df["model"].unique():
            mdf = agg_df[agg_df["model"] == model_name]
            best = mdf.loc[mdf["overall_assignment_acc_mean"].idxmax()]
            print(
                f"  {model_name}: "
                f"{best['overall_assignment_acc_mean']:.3f} ± {best['overall_assignment_acc_std']:.3f} "
                f"({best['repr']}/L{best['layer']}/{best['method']})"
            )
    else:
        print("\nNo results generated. Check embedding paths.")


def aggregate_results(all_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Compute mean +/- std for each metric, grouped by configuration."""
    metric_cols = [
        f"dir_acc_at_{k}" for k in range(1, top_k + 1)
    ] + [
        "existing_acc", "new_acc", "overall_assignment_acc", "tau",
    ]

    group_cols = ["model", "method", "repr", "pooling", "layer", "D"]

    rows = []
    for group_key, group_df in all_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, group_key))
        row["n_seeds"] = len(group_df)
        for col in metric_cols:
            if col in group_df.columns:
                values = group_df[col].dropna()
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_std"] = float(values.std()) if len(values) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
