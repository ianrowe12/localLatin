"""Resubmit Task A evaluator: evaluate extracted embeddings with post-processing methods.

(Renamed from run_phase11_evaluate.py — same code.)

Notable behavior:
- Excludes layer 0 from all processing
- Directory-level assignment accuracy at k (Task B redefinition)
- Optimizes D on train via exact rank-1 directory assignment accuracy
- Reports train_aucroc and train_dir_acc_at_1 for best-config selection
- Saves similarity distributions for the density-histogram figures
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
    l2_normalize,
    similarity_matrix,
    sweep_thresholds,
    upper_triangle,
    upper_triangle_labels,
)
from pair_evaluation import safe_auc_roc, safe_spearman
from sif_abtt import EmbeddingCleaner
from embedding_alignment import AlignmentResolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 11 evaluation.")
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
        "--reprs",
        default="",
        help="Optional comma-separated reprs to evaluate, e.g. 'hidden' or 'hidden,ff1'.",
    )
    parser.add_argument(
        "--hidden_mean_subdir",
        default="hidden_mean",
        help="Subdirectory to use for hidden/mean embeddings.",
    )
    parser.add_argument(
        "--hidden_sif_subdir",
        default="hidden_sif",
        help="Subdirectory to use for hidden/sif embeddings.",
    )
    parser.add_argument(
        "--hidden_lasttok_subdir",
        default="hidden_lasttok",
        help="Subdirectory to use for hidden/lasttok embeddings.",
    )
    parser.add_argument(
        "--D_values", default="1,2,3,5,7,10",
        help="Comma-separated D values for ABTT sweep.",
    )
    parser.add_argument(
        "--poolings", default="mean,sif",
        help="Comma-separated pooling variants to evaluate.",
    )
    parser.add_argument(
        "--methods", default="",
        help="Comma-separated method filter (empty=all). Intersects with pooling-specific method list.",
    )
    parser.add_argument("--sif_a", type=float, default=0.001)
    parser.add_argument(
        "--dist_dir", default=None,
        help="Output directory for similarity distribution .npz files (Figure 1).",
    )
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def _pooling_suffix(pooling: str) -> str:
    if pooling == "mean":
        return ""
    if pooling == "lasttok":
        return "_lasttok"
    if pooling == "sif":
        return "_sif"
    raise ValueError(f"Unknown pooling: {pooling}")


def find_embedding_file(
    runs_root: Path,
    slug: str,
    repr_name: str,
    pooling: str,
    layer: int,
    subdir_override: Optional[str] = None,
) -> Optional[Path]:
    """Locate the raw (un-normalized) embedding .npy file."""
    suffix = _pooling_suffix(pooling)
    fname = f"{repr_name}_layer{layer}_embeddings{suffix}.npy"
    subdir = subdir_override or f"{repr_name}_{pooling}"
    candidate = runs_root / "phase9_bases" / slug / subdir / fname
    if candidate.exists():
        return candidate
    candidate2 = runs_root / "phase9_bases" / slug / fname
    if candidate2.exists():
        return candidate2
    return None


def discover_layers(
    runs_root: Path,
    slug: str,
    repr_name: str,
    pooling: str,
    subdir_override: Optional[str] = None,
) -> List[int]:
    """Find which layers have embeddings extracted, excluding layer 0."""
    suffix = _pooling_suffix(pooling)
    pattern = f"{repr_name}_layer*_embeddings{suffix}.npy"
    subdir = subdir_override or f"{repr_name}_{pooling}"
    base_dir = runs_root / "phase9_bases" / slug / subdir
    if not base_dir.exists():
        base_dir = runs_root / "phase9_bases" / slug
    if not base_dir.exists():
        return []

    layers = []
    for f in sorted(base_dir.glob(pattern)):
        name = f.stem
        parts = name.split("_layer")
        if len(parts) == 2:
            rest = parts[1].split("_")[0]
            try:
                l = int(rest)
                if l >= 1:  # Phase 11: exclude layer 0
                    layers.append(l)
            except ValueError:
                continue
    return sorted(layers)


def learn_tau_from_similarity(
    sim: np.ndarray,
    folder_ids: np.ndarray,
) -> float:
    """Sweep thresholds on an already-built similarity matrix, return best-F1 tau.

    Split out of :func:`learn_threshold_on_train` so that scorers which produce a
    similarity matrix directly (the lexical baselines in
    ``scripts/resubmit/lexical_baselines.py``) learn tau through exactly the same
    code path as the embedding methods.
    """
    sims = upper_triangle(sim)
    labels = upper_triangle_labels(folder_ids)
    thresholds = np.linspace(0, 1, 200)
    df = sweep_thresholds(sims, labels, thresholds)
    best_idx = df["f1"].idxmax()
    return float(df.loc[best_idx, "threshold"])


def learn_threshold_on_train(
    emb: np.ndarray,
    folder_ids: np.ndarray,
) -> float:
    """Build train similarity matrix, sweep thresholds, return best F1 threshold."""
    emb_norm = l2_normalize(emb)
    sim = similarity_matrix(emb_norm)
    return learn_tau_from_similarity(sim, folder_ids)


def has_partner_flags(folder_ids: np.ndarray) -> np.ndarray:
    """Return a boolean mask marking rows whose directory has another member."""
    unique_ids, counts = np.unique(folder_ids, return_counts=True)
    multi_ids = set(unique_ids[counts > 1])
    return np.array([fid in multi_ids for fid in folder_ids], dtype=bool)


def compute_assignment_acc(
    sim: np.ndarray,
    has_test_partner: np.ndarray,
    tau: float,
) -> Tuple[float, float, float]:
    """Compute new vs existing assignment accuracy using threshold tau."""
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


def directory_assignment_accuracy_at_k(
    sim: np.ndarray,
    folder_ids: np.ndarray,
    has_partner: np.ndarray,
    tau: float,
    k: int,
) -> float:
    """Directory-level assignment accuracy at k.

    For each file i:
    1. Compute dir_score[d] = max(sim[i,j] for j in d, j != i) for each directory d
    2. Add "__NEW__" pseudo-directory with score = tau
    3. Rank all entries descending by score, breaking ties lexically by label
    4. If has_partner[i]: correct if folder_ids[i] in top-k
    5. If not has_partner[i]: correct if "__NEW__" in top-k
    6. Return fraction correct over all files
    """
    n = sim.shape[0]
    folder_ids = np.asarray(folder_ids)
    has_partner = np.asarray(has_partner, dtype=bool)

    # Precompute directory membership: {dir_id -> list of indices}
    dir_members: Dict[str, List[int]] = {}
    for idx, fid in enumerate(folder_ids):
        fid_str = str(fid)
        if fid_str not in dir_members:
            dir_members[fid_str] = []
        dir_members[fid_str].append(idx)

    correct = 0
    for i in range(n):
        my_dir = str(folder_ids[i])

        # Compute score per directory (max sim to any member, excluding self)
        dir_scores: Dict[str, float] = {}
        for d, members in dir_members.items():
            member_sims = []
            for j in members:
                if j != i:
                    member_sims.append(sim[i, j])
            if member_sims:
                dir_scores[d] = max(member_sims)
            # If no members left (only self), skip this directory

        # Add __NEW__ pseudo-directory
        dir_scores["__NEW__"] = tau

        # Rank descending by score
        ranked = sorted(dir_scores.keys(), key=lambda d: (-dir_scores[d], d))
        top_k = set(ranked[:k])

        if has_partner[i]:
            if my_dir in top_k:
                correct += 1
        else:
            if "__NEW__" in top_k:
                correct += 1

    return correct / n if n > 0 else 0.0


def find_optimal_D_phase11(
    train_emb: np.ndarray,
    train_folder_ids: np.ndarray,
    D_values: List[int],
) -> Tuple[int, float]:
    """Sweep D values on train set, return best D by exact rank-1 directory accuracy."""
    best_D = D_values[0]
    best_score = -1.0
    train_has_partner = has_partner_flags(train_folder_ids)

    for D in D_values:
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_emb)
        cleaned = cleaner.transform(train_emb)
        cleaned_norm = l2_normalize(cleaned)
        sim = similarity_matrix(cleaned_norm)
        tau = learn_threshold_on_train(cleaned, train_folder_ids)
        score = directory_assignment_accuracy_at_k(
            sim,
            train_folder_ids,
            train_has_partner,
            tau,
            k=1,
        )
        if score > best_score:
            best_score = score
            best_D = D

    return best_D, best_score


def evaluate_from_similarity(
    train_sim: np.ndarray,
    test_sim: np.ndarray,
    train_folder_ids: np.ndarray,
    test_folder_ids: np.ndarray,
    test_has_partner: np.ndarray,
) -> Dict:
    """Task A + Task B metric block for a pair of (train, test) similarity matrices.

    This is the single definition of the reported metrics. :func:`evaluate_single`
    feeds it cosine matrices built from post-processed embeddings;
    ``scripts/resubmit/lexical_baselines.py`` feeds it BM25 / TF-IDF / Levenshtein
    score matrices, so both go through identical metric code.

    tau is learned on ``train_sim`` only (best-F1 cut over same-directory vs.
    different-directory train pairs) and then applied to the test matrix.
    """
    # ── Train metrics ──────────────────────────────────────────────
    train_sims = upper_triangle(train_sim)
    train_labels = upper_triangle_labels(train_folder_ids)
    tau = learn_tau_from_similarity(train_sim, train_folder_ids)

    train_aucroc = safe_auc_roc(train_sims, train_labels)

    train_has_partner = has_partner_flags(train_folder_ids)
    train_dir_acc_at_1 = directory_assignment_accuracy_at_k(
        train_sim, train_folder_ids, train_has_partner, tau, k=1
    )

    # ── Test metrics ───────────────────────────────────────────────
    test_sims = upper_triangle(test_sim)
    test_labels = upper_triangle_labels(test_folder_ids)
    test_has_partner = np.asarray(test_has_partner, dtype=bool)

    aucroc = safe_auc_roc(test_sims, test_labels)
    spearman = safe_spearman(test_sims, test_labels.astype(float))

    # Cosine stats
    pos_mask = test_labels.astype(bool)
    same_avg = float(np.mean(test_sims[pos_mask])) if pos_mask.any() else float("nan")
    diff_avg = float(np.mean(test_sims[~pos_mask])) if (~pos_mask).any() else float("nan")
    gap = same_avg - diff_avg

    # Directory assignment acc@k (new Task B)
    dir_acc_1 = directory_assignment_accuracy_at_k(
        test_sim, test_folder_ids, test_has_partner, tau, k=1
    )
    dir_acc_3 = directory_assignment_accuracy_at_k(
        test_sim, test_folder_ids, test_has_partner, tau, k=3
    )

    # Assignment accuracy (diagnostic, carried over from Phase 10)
    existing_acc, new_acc, overall_acc = compute_assignment_acc(
        test_sim, test_has_partner, tau
    )

    n_existing = int(test_has_partner.sum())
    n_new = int(len(test_folder_ids)) - n_existing

    return {
        "tau": tau,
        "aucroc": aucroc,
        "spearman": spearman,
        "same_avg": same_avg,
        "diff_avg": diff_avg,
        "gap": gap,
        "dir_acc_at_1": dir_acc_1,
        "dir_acc_at_3": dir_acc_3,
        "existing_acc": existing_acc,
        "new_acc": new_acc,
        "overall_assignment_acc": overall_acc,
        "train_aucroc": train_aucroc,
        "train_dir_acc_at_1": train_dir_acc_at_1,
        "n_train": int(len(train_folder_ids)),
        "n_test": int(len(test_folder_ids)),
        "n_existing": n_existing,
        "n_new": n_new,
    }


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

    train_emb = emb_all[train_mask]
    test_emb = emb_all[test_mask]

    actual_D = D
    actual_pooling = pooling_src

    # Apply post-processing
    if method == "baseline":
        pass
    elif method == "sif_only":
        pass
    elif method == "sif_abtt_fixed":
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_emb)
        train_emb = cleaner.transform(train_emb)
        test_emb = cleaner.transform(test_emb)
    elif method == "sif_abtt_optimal":
        best_D, _ = find_optimal_D_phase11(train_emb, train_folder_ids, D_values)
        actual_D = best_D
        cleaner = EmbeddingCleaner(num_components=best_D, center=True)
        cleaner.fit(train_emb)
        train_emb = cleaner.transform(train_emb)
        test_emb = cleaner.transform(test_emb)
    elif method == "abtt_fixed":
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_emb)
        train_emb = cleaner.transform(train_emb)
        test_emb = cleaner.transform(test_emb)
    elif method == "abtt_optimal":
        best_D, _ = find_optimal_D_phase11(train_emb, train_folder_ids, D_values)
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

    metrics = evaluate_from_similarity(
        train_sim=similarity_matrix(train_emb_norm),
        test_sim=similarity_matrix(test_emb_norm),
        train_folder_ids=train_folder_ids,
        test_folder_ids=test_folder_ids,
        test_has_partner=test_has_partner,
    )

    row = {
        "model": model_name,
        "repr": repr_name,
        "pooling": actual_pooling,
        "layer": layer,
        "method": method,
        "D": actual_D,
        "sif_a": sif_a,
    }
    row.update(metrics)
    return row


def save_distributions(
    emb_all: np.ndarray,
    split_meta: pd.DataFrame,
    model_name: str,
    dist_dir: Path,
) -> None:
    """Save same/diff cosine similarity distributions for the baseline last layer."""
    test_mask = split_meta["split"].values == "test"
    test_emb = emb_all[test_mask]
    test_folder_ids = split_meta.loc[test_mask, "folder_id"].values

    test_emb_norm = l2_normalize(test_emb)
    sim = similarity_matrix(test_emb_norm)
    sims = upper_triangle(sim)
    labels = upper_triangle_labels(test_folder_ids)

    pos_mask = labels.astype(bool)
    same_sims = sims[pos_mask]
    diff_sims = sims[~pos_mask]

    dist_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(model_name)
    np.savez_compressed(
        dist_dir / f"{slug}_baseline_distributions.npz",
        same_sims=same_sims,
        diff_sims=diff_sims,
    )
    print(f"    Saved distributions for {model_name}")


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    split_meta = pd.read_csv(args.split_csv)
    aligner = AlignmentResolver(split_meta)
    dist_dir = Path(args.dist_dir) if args.dist_dir else None

    seq2seq_models = [m.strip() for m in args.models.split(",") if m.strip()]
    encoder_models = [m.strip() for m in args.encoder_models.split(",") if m.strip()]
    D_values = [int(d.strip()) for d in args.D_values.split(",")]
    requested_reprs = {r.strip() for r in args.reprs.split(",") if r.strip()}
    poolings = [p.strip() for p in args.poolings.split(",") if p.strip()]

    results: List[Dict] = []

    model_configs = []
    for m in seq2seq_models:
        reprs = ["hidden", "ff1"]
        if requested_reprs:
            reprs = [r for r in reprs if r in requested_reprs]
        model_configs.append((m, "seq2seq", reprs))
    for m in encoder_models:
        reprs = ["hidden", "ffn_int"]
        if requested_reprs:
            reprs = [r for r in reprs if r in requested_reprs]
        model_configs.append((m, "encoder", reprs))

    # Track which models have had distributions saved (baseline, hidden, mean, last layer)
    dist_saved: set = set()

    for model_name, model_type, reprs in model_configs:
        slug = model_slug(model_name)
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_type})")
        print(f"{'='*60}")

        for repr_name in reprs:
            for pooling in poolings:
                subdir_override = None
                if repr_name == "hidden":
                    if pooling == "mean":
                        subdir_override = args.hidden_mean_subdir
                    elif pooling == "sif":
                        subdir_override = args.hidden_sif_subdir
                    elif pooling == "lasttok":
                        subdir_override = args.hidden_lasttok_subdir
                layers = discover_layers(
                    runs_root,
                    slug,
                    repr_name,
                    pooling,
                    subdir_override=subdir_override,
                )
                if not layers:
                    print(f"  [{repr_name}/{pooling}] No layers found, skipping.")
                    continue
                print(f"  [{repr_name}/{pooling}] Layers: {layers}")

                for layer in layers:
                    emb_path = find_embedding_file(
                        runs_root,
                        slug,
                        repr_name,
                        pooling,
                        layer,
                        subdir_override=subdir_override,
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
                    emb_all = aligner.aligner_for(emb_path).apply(emb_all)

                    # Save distributions for baseline (hidden, mean, last layer)
                    if (
                        dist_dir is not None
                        and repr_name == "hidden"
                        and pooling == "mean"
                        and layer == layers[-1]  # last layer
                        and model_name not in dist_saved
                    ):
                        save_distributions(emb_all, split_meta, model_name, dist_dir)
                        dist_saved.add(model_name)

                    # Determine which methods to run based on pooling
                    if pooling in ("mean", "lasttok"):
                        methods = ["baseline", "abtt_fixed", "abtt_optimal", "whitening"]
                    else:
                        methods = ["sif_only", "sif_abtt_fixed", "sif_abtt_optimal"]
                    if args.methods.strip():
                        wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
                        methods = [m for m in methods if m in wanted]

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
                                f"DirAcc@1={row['dir_acc_at_1']:.4f} "
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

        print("\nBest AUCROC per model:")
        for model_name in df["model"].unique():
            mdf = df[df["model"] == model_name]
            best = mdf.loc[mdf["aucroc"].idxmax()]
            print(
                f"  {model_name}: AUCROC={best['aucroc']:.4f} "
                f"({best['repr']}/L{best['layer']}/{best['method']})"
            )

        print("\nBest Dir Acc@1 per model:")
        for model_name in df["model"].unique():
            mdf = df[df["model"] == model_name]
            best = mdf.loc[mdf["dir_acc_at_1"].idxmax()]
            print(
                f"  {model_name}: DirAcc@1={best['dir_acc_at_1']:.4f} "
                f"({best['repr']}/L{best['layer']}/{best['method']})"
            )
    else:
        print("\nNo results generated. Check embedding paths.")


if __name__ == "__main__":
    main()
