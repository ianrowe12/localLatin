from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from canon_retrieval import l2_normalize
from cli_utils import extract_layer_numbers, parse_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter or reduce cached embeddings into derived run dirs."
    )
    parser.add_argument("--base_run_dir", required=True, help="Base run directory.")
    parser.add_argument("--out_run_dir", required=True, help="Output derived run directory.")
    parser.add_argument("--repr", choices=["ff1", "hidden"], required=True)
    parser.add_argument("--pooling", choices=["mean", "lasttok"], default="mean")
    parser.add_argument("--layers", default="", help="Layer list, e.g. 1-12 or 0,6,12.")
    parser.add_argument(
        "--method",
        choices=["variance", "corr", "pca", "pc_remove", "center", "standardize", "lda", "random"],
        required=True,
        help="Filtering/reduction method.",
    )

    parser.add_argument("--keep_top_k", type=int, default=0)
    parser.add_argument("--keep_ratio", type=float, default=0.0)
    parser.add_argument("--drop_lowest_percent", type=float, default=0.0)

    parser.add_argument("--corr_threshold", type=float, default=0.99)
    parser.add_argument("--corr_sample_n", type=int, default=512)
    parser.add_argument("--corr_target_dims", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=13)

    parser.add_argument("--pca_components", default="256")
    parser.add_argument("--pca_solver", choices=["randomized", "full"], default="randomized")
    parser.add_argument("--pc_remove_components", type=int, default=0)
    parser.add_argument(
        "--pc_remove_center",
        action="store_true",
        help="Center embeddings before removing top PCs.",
    )
    parser.add_argument("--standardize_eps", type=float, default=1e-6)

    parser.add_argument("--lda_test_fraction", type=float, default=0.2)
    parser.add_argument("--lda_solver", choices=["svd", "lsqr", "eigen"], default="lsqr")
    parser.add_argument("--lda_shrinkage", default="auto")
    return parser.parse_args()


def infer_layers(run_dir: Path, repr_name: str, pooling: str) -> List[int]:
    suffix = "" if pooling == "mean" else "_lasttok"
    pattern = rf"{repr_name}_layer(\\d+)_embeddings{suffix}\\.npy$"
    paths = [p.as_posix() for p in run_dir.glob(f"{repr_name}_layer*_embeddings{suffix}.npy")]
    return extract_layer_numbers(paths, pattern)


def embedding_path(run_dir: Path, repr_name: str, pooling: str, layer: int) -> Path:
    suffix = "" if pooling == "mean" else "_lasttok"
    return run_dir / f"{repr_name}_layer{layer}_embeddings{suffix}.npy"


def select_by_variance(
    emb: np.ndarray,
    keep_top_k: int,
    keep_ratio: float,
    drop_lowest_percent: float,
) -> np.ndarray:
    dims = emb.shape[1]
    var = np.var(emb, axis=0)
    order = np.argsort(var)[::-1]

    if keep_top_k > 0:
        k = min(keep_top_k, dims)
    elif keep_ratio > 0:
        k = max(int(round(keep_ratio * dims)), 1)
    elif drop_lowest_percent > 0:
        k = max(int(round((1.0 - drop_lowest_percent / 100.0) * dims)), 1)
    else:
        raise ValueError("Provide one of keep_top_k, keep_ratio, drop_lowest_percent.")

    return order[:k]


def select_by_correlation(
    emb: np.ndarray,
    corr_threshold: float,
    corr_sample_n: int,
    corr_target_dims: int,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    n = emb.shape[0]
    sample_n = min(corr_sample_n, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    sample = emb[sample_idx]

    corr = np.corrcoef(sample, rowvar=False).astype(np.float32)
    var = np.var(emb, axis=0)
    order = np.argsort(var)[::-1]

    target = int(corr_target_dims) if corr_target_dims > 0 else 0
    if target:
        target = min(target, emb.shape[1])

    selected: List[int] = []
    for idx in order:
        if not selected:
            selected.append(int(idx))
            continue
        max_corr = float(np.max(np.abs(corr[idx, selected])))
        if max_corr < corr_threshold:
            selected.append(int(idx))
        if target and len(selected) >= target:
            break

    if not selected:
        selected.append(int(order[0]))
    return np.array(selected, dtype=np.int64)


def select_random_dims(
    emb: np.ndarray,
    keep_top_k: int,
    keep_ratio: float,
    drop_lowest_percent: float,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    dims = emb.shape[1]
    if keep_top_k > 0:
        k = min(keep_top_k, dims)
    elif keep_ratio > 0:
        k = max(int(round(keep_ratio * dims)), 1)
    elif drop_lowest_percent > 0:
        k = max(int(round((1.0 - drop_lowest_percent / 100.0) * dims)), 1)
    else:
        raise ValueError("Provide one of keep_top_k, keep_ratio, drop_lowest_percent.")
    return rng.choice(dims, size=k, replace=False).astype(np.int64)


def parse_components(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def save_config(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_meta(out_dir: Path, meta: pd.DataFrame) -> None:
    meta.to_csv(out_dir / "meta.csv", index=False)


def derive_output_dirs(base_out_dir: Path, method: str, components: List[int]) -> List[Path]:
    if method != "pca" or len(components) <= 1:
        return [base_out_dir]
    return [base_out_dir / f"pca_{k}" for k in components]


def stratified_split_winnable(
    meta: pd.DataFrame, test_fraction: float, random_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    is_train = np.zeros(len(meta), dtype=bool)
    is_eval_query = np.zeros(len(meta), dtype=bool)
    winnable = meta["is_winnable"].to_numpy(dtype=bool)
    meta_win = meta[winnable]
    for folder_id, group in meta_win.groupby("folder_id"):
        idx = group.index.to_numpy()
        n = len(idx)
        if n <= 1:
            continue
        test_n = max(int(round(n * test_fraction)), 1)
        test_n = min(test_n, n - 1)
        test_idx = rng.choice(idx, size=test_n, replace=False)
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)
        is_eval_query[test_idx] = True
        is_train[train_idx] = True
    return is_train, is_eval_query


def main() -> None:
    args = parse_args()
    base_run_dir = Path(args.base_run_dir)
    out_run_dir = Path(args.out_run_dir)

    meta = pd.read_csv(base_run_dir / "meta.csv")
    layers = parse_layers(args.layers)
    if not layers:
        layers = infer_layers(base_run_dir, args.repr, args.pooling)
    if not layers:
        raise FileNotFoundError("No embeddings found for given repr/pooling.")

    components = parse_components(args.pca_components) if args.method == "pca" else []
    out_dirs = derive_output_dirs(out_run_dir, args.method, components)

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_out = meta.copy()
        method_config: Dict[str, dict] = {}

        if args.method == "lda":
            is_train, is_eval_query = stratified_split_winnable(
                meta_out, args.lda_test_fraction, args.random_seed
            )
            meta_out["is_train"] = is_train
            meta_out["is_eval_query"] = is_eval_query

        write_meta(out_dir, meta_out)

        for layer in layers:
            emb_path = embedding_path(base_run_dir, args.repr, args.pooling, layer)
            emb = np.load(emb_path)

            if args.method == "variance":
                selected = select_by_variance(
                    emb, args.keep_top_k, args.keep_ratio, args.drop_lowest_percent
                )
                out_emb = emb[:, selected]
                emb_norm = l2_normalize(out_emb)
                dims_path = out_dir / f"{args.repr}_layer{layer}_dims_variance.npy"
                np.save(dims_path, selected)
                method_config[str(layer)] = {
                    "selected_dims": dims_path.name,
                    "num_dims": int(len(selected)),
                }

            elif args.method == "corr":
                selected = select_by_correlation(
                    emb,
                    args.corr_threshold,
                    args.corr_sample_n,
                    args.corr_target_dims,
                    args.random_seed,
                )
                out_emb = emb[:, selected]
                emb_norm = l2_normalize(out_emb)
                dims_path = out_dir / f"{args.repr}_layer{layer}_dims_corr.npy"
                np.save(dims_path, selected)
                method_config[str(layer)] = {
                    "selected_dims": dims_path.name,
                    "num_dims": int(len(selected)),
                    "corr_threshold": args.corr_threshold,
                    "corr_sample_n": args.corr_sample_n,
                    "corr_target_dims": args.corr_target_dims,
                }

            elif args.method == "pca":
                from sklearn.decomposition import PCA

                if len(components) > 1:
                    k = int(out_dir.name.split("_")[-1])
                else:
                    k = components[0]
                pca = PCA(n_components=k, svd_solver=args.pca_solver, random_state=args.random_seed)
                out_emb = pca.fit_transform(emb)
                emb_norm = l2_normalize(out_emb)
                pca_path = out_dir / f"{args.repr}_layer{layer}_pca.pkl"
                with open(pca_path, "wb") as f:
                    pickle.dump(pca, f)
                evr_path = out_dir / f"{args.repr}_layer{layer}_pca_evr.npy"
                np.save(evr_path, pca.explained_variance_ratio_.astype(np.float32))
                method_config[str(layer)] = {
                    "pca_model": pca_path.name,
                    "explained_variance_ratio": evr_path.name,
                    "num_dims": int(k),
                    "solver": args.pca_solver,
                }

            elif args.method == "pc_remove":
                from sklearn.decomposition import PCA

                if args.pc_remove_components <= 0:
                    raise ValueError("--pc_remove_components must be > 0 for pc_remove.")

                if args.pc_remove_center:
                    mean_vec = emb.mean(axis=0)
                    emb_centered = emb - mean_vec
                else:
                    mean_vec = None
                    emb_centered = emb

                pca = PCA(
                    n_components=args.pc_remove_components,
                    svd_solver=args.pca_solver,
                    random_state=args.random_seed,
                )
                pca.fit(emb_centered)
                proj = emb_centered @ pca.components_.T @ pca.components_
                out_emb = emb_centered - proj
                emb_norm = l2_normalize(out_emb)

                pca_path = out_dir / f"{args.repr}_layer{layer}_pc_remove.pkl"
                with open(pca_path, "wb") as f:
                    pickle.dump(pca, f)
                evr_path = out_dir / f"{args.repr}_layer{layer}_pc_remove_evr.npy"
                np.save(evr_path, pca.explained_variance_ratio_.astype(np.float32))
                mean_path = None
                if mean_vec is not None:
                    mean_path = out_dir / f"{args.repr}_layer{layer}_pc_remove_mean.npy"
                    np.save(mean_path, mean_vec.astype(np.float32))
                method_config[str(layer)] = {
                    "pc_remove_components": int(args.pc_remove_components),
                    "pc_remove_center": bool(args.pc_remove_center),
                    "pca_model": pca_path.name,
                    "explained_variance_ratio": evr_path.name,
                    "mean_vector": mean_path.name if mean_path else "",
                    "num_dims": int(out_emb.shape[1]),
                    "solver": args.pca_solver,
                }

            elif args.method == "center":
                mean_vec = emb.mean(axis=0)
                out_emb = emb - mean_vec
                emb_norm = l2_normalize(out_emb)
                mean_path = out_dir / f"{args.repr}_layer{layer}_center_mean.npy"
                np.save(mean_path, mean_vec.astype(np.float32))
                method_config[str(layer)] = {
                    "mean_vector": mean_path.name,
                    "num_dims": int(out_emb.shape[1]),
                }

            elif args.method == "standardize":
                mean_vec = emb.mean(axis=0)
                std_vec = emb.std(axis=0)
                std_vec = np.maximum(std_vec, args.standardize_eps)
                out_emb = (emb - mean_vec) / std_vec
                emb_norm = l2_normalize(out_emb)
                mean_path = out_dir / f"{args.repr}_layer{layer}_standardize_mean.npy"
                std_path = out_dir / f"{args.repr}_layer{layer}_standardize_std.npy"
                np.save(mean_path, mean_vec.astype(np.float32))
                np.save(std_path, std_vec.astype(np.float32))
                method_config[str(layer)] = {
                    "mean_vector": mean_path.name,
                    "std_vector": std_path.name,
                    "standardize_eps": float(args.standardize_eps),
                    "num_dims": int(out_emb.shape[1]),
                }

            elif args.method == "random":
                selected = select_random_dims(
                    emb,
                    args.keep_top_k,
                    args.keep_ratio,
                    args.drop_lowest_percent,
                    args.random_seed,
                )
                out_emb = emb[:, selected]
                emb_norm = l2_normalize(out_emb)
                dims_path = out_dir / f"{args.repr}_layer{layer}_dims_random.npy"
                np.save(dims_path, selected)
                method_config[str(layer)] = {
                    "selected_dims": dims_path.name,
                    "num_dims": int(len(selected)),
                    "random_seed": args.random_seed,
                }

            elif args.method == "lda":
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

                is_train = meta_out["is_train"].to_numpy(dtype=bool)
                labels = meta_out["folder_id"].to_numpy()
                train_x = emb[is_train]
                train_y = labels[is_train]
                shrinkage: Optional[float | str]
                if args.lda_solver == "svd":
                    shrinkage = None
                elif args.lda_shrinkage == "auto":
                    shrinkage = "auto"
                elif args.lda_shrinkage:
                    shrinkage = float(args.lda_shrinkage)
                else:
                    shrinkage = None
                lda = LinearDiscriminantAnalysis(
                    solver=args.lda_solver, shrinkage=shrinkage
                )
                lda.fit(train_x, train_y)
                out_emb = lda.transform(emb)
                emb_norm = l2_normalize(out_emb)
                lda_path = out_dir / f"{args.repr}_layer{layer}_lda.pkl"
                with open(lda_path, "wb") as f:
                    pickle.dump(lda, f)
                method_config[str(layer)] = {
                    "lda_model": lda_path.name,
                    "num_dims": int(out_emb.shape[1]),
                    "solver": args.lda_solver,
                    "shrinkage": args.lda_shrinkage,
                }

            else:
                raise ValueError(f"Unknown method: {args.method}")

            suffix = "" if args.pooling == "mean" else "_lasttok"
            np.save(out_dir / f"{args.repr}_layer{layer}_embeddings{suffix}.npy", out_emb)
            np.save(out_dir / f"{args.repr}_layer{layer}_embeddings{suffix}_norm.npy", emb_norm)

        config = {
            "base_run_dir": str(base_run_dir),
            "repr": args.repr,
            "pooling": args.pooling,
            "layers": layers,
            "method": args.method,
            "random_seed": args.random_seed,
            "params": {
                "keep_top_k": args.keep_top_k,
                "keep_ratio": args.keep_ratio,
                "drop_lowest_percent": args.drop_lowest_percent,
                "corr_threshold": args.corr_threshold,
                "corr_sample_n": args.corr_sample_n,
                "corr_target_dims": args.corr_target_dims,
                "pca_components": components,
                "pca_solver": args.pca_solver,
                "pc_remove_components": args.pc_remove_components,
                "pc_remove_center": args.pc_remove_center,
                "standardize_eps": args.standardize_eps,
                "lda_test_fraction": args.lda_test_fraction,
                "lda_solver": args.lda_solver,
                "lda_shrinkage": args.lda_shrinkage,
            },
            "layer_outputs": method_config,
        }
        save_config(out_dir / "config.json", config)

    print(f"Derived runs written under: {out_run_dir}")


if __name__ == "__main__":
    main()
