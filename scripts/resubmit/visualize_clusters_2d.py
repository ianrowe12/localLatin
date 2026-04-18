"""2D cluster visualizations (t-SNE + UMAP) per model x post-processing method.

Produces two figures (`fig_tsne_per_model.pdf`, `fig_umap_per_model.pdf`) showing
how ABTT tightens same-directory fragment clusters compared to baseline, SIF-only,
and whitening. Each model is evaluated at its best-performing layer as recorded in
``runs/active/resubmit/results/taskA_per_model_summary.csv``.

Pipeline per (model, method):
  1. Load the appropriate pooled embedding (mean for baseline/whitening, SIF for
     sif_only/sif_abtt_optimal).
  2. Apply the train-only fit protocol from ``evaluate_vectors.py`` (EmbeddingCleaner
     for sif_abtt_optimal with D selected via ``find_optimal_D``; PCA whitening for
     whitening; no-op for baseline/sif_only).
  3. L2-normalize; this matches the cosine-similarity regime used for evaluation.
  4. Project to 2D with t-SNE and UMAP (seeded, with cached outputs under
     ``runs/active/resubmit/cluster_viz/``).

The final figures place models as rows (6) and methods as columns (4), with
winnable fragments (folder size >= 2) colored by folder and singletons shown in
light gray. Dense same-color clumps under ``sif_abtt_optimal`` are the intended
visual evidence that complements the cosine-gap line plots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT / "scripts" / "resubmit"))

from canon_retrieval import l2_normalize  # noqa: E402
from sif_abtt import EmbeddingCleaner  # noqa: E402
from evaluate_vectors import find_optimal_D  # noqa: E402


MODELS: List[Tuple[str, str, str]] = [
    ("LaTa", "bowphs_LaTa", "hidden"),
    ("PhilTa", "bowphs_PhilTa", "hidden"),
    ("LaBSE", "sentence-transformers_LaBSE", "hidden"),
    ("Qwen3-0.6B", "Qwen_Qwen3-Embedding-0.6B", "hidden"),
    ("KaLM-mini", "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5", "hidden"),
    ("mT5-base", "google_mt5-base", "hidden"),
]

METHOD_DISPLAY = {
    "baseline": "Baseline",
    "sif_only": "SIF",
    "sif_abtt_optimal": "SIF+ABTT",
    "whitening": "Whitening",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split_csv",
        default=str(REPO_ROOT / "runs/active/resubmit/data/phase_resubmit_split.csv"),
    )
    parser.add_argument(
        "--bases_root",
        default=str(REPO_ROOT / "runs/active/resubmit_bases/phase9_bases"),
    )
    parser.add_argument(
        "--best_layer_csv",
        default=str(REPO_ROOT / "runs/active/resubmit/results/taskA_per_model_summary.csv"),
    )
    parser.add_argument(
        "--intermediate_dir",
        default=str(REPO_ROOT / "runs/active/resubmit/cluster_viz"),
    )
    parser.add_argument(
        "--figures_dir",
        default=str(REPO_ROOT / "overleaf_drafts/figures"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "sif_only", "sif_abtt_optimal", "whitening"],
    )
    parser.add_argument("--projections", nargs="+", default=["tsne", "umap"])
    parser.add_argument("--D_values", default="1,2,3,5,7,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to these display names (default: all 6).",
    )
    parser.add_argument(
        "--render_only",
        action="store_true",
        help="Skip computation; assume caches exist and just render figures.",
    )
    return parser.parse_args()


def load_best_layers(csv_path: str) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    return dict(zip(df["model_display"], df["best_layer"].astype(int)))


def embedding_file(
    bases_root: Path, slug: str, repr_name: str, pooling: str, layer: int
) -> Path:
    suffix = "" if pooling == "mean" else "_sif"
    subdir_tag = "mean_tokempty" if pooling == "mean" else "sif_tokempty"
    fname = f"{repr_name}_layer{layer}_embeddings{suffix}.npy"
    return bases_root / slug / f"{repr_name}_{subdir_tag}" / fname


def method_pooling(method: str) -> str:
    if method in {"sif_only", "sif_abtt_optimal"}:
        return "sif"
    return "mean"


def apply_method(
    emb_all: np.ndarray,
    train_mask: np.ndarray,
    train_folder_ids: np.ndarray,
    method: str,
    D_values: List[int],
) -> Tuple[np.ndarray, Optional[int]]:
    """Return post-processed embeddings (all rows) and actual_D if applicable."""
    if method == "baseline" or method == "sif_only":
        return emb_all.astype(np.float32, copy=False), None

    train_emb = emb_all[train_mask]

    if method == "sif_abtt_optimal":
        best_D, _ = find_optimal_D(train_emb, train_folder_ids, D_values)
        cleaner = EmbeddingCleaner(num_components=int(best_D), center=True)
        cleaner.fit(train_emb)
        return cleaner.transform(emb_all).astype(np.float32, copy=False), int(best_D)

    if method == "whitening":
        from sklearn.decomposition import PCA

        n_comp = min(train_emb.shape[0] - 1, train_emb.shape[1])
        pca = PCA(n_components=n_comp, whiten=True, random_state=0)
        pca.fit(train_emb)
        return pca.transform(emb_all).astype(np.float32, copy=False), n_comp

    raise ValueError(f"Unknown method: {method}")


def project_tsne(emb: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
        n_jobs=-1,
    )
    return tsne.fit_transform(emb)


def project_umap(
    emb: np.ndarray, n_neighbors: int, min_dist: float, seed: int
) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=int(seed),
    )
    return reducer.fit_transform(emb)


def compute_or_load_projection(
    cache_path: Path,
    proj_name: str,
    emb_norm: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if cache_path.exists() and not args.force_recompute:
        arr = np.load(cache_path)["coords"]
        if arr.shape == (emb_norm.shape[0], 2):
            return arr
    if proj_name == "tsne":
        coords = project_tsne(emb_norm, args.tsne_perplexity, args.seed)
    elif proj_name == "umap":
        coords = project_umap(
            emb_norm, args.umap_n_neighbors, args.umap_min_dist, args.seed
        )
    else:
        raise ValueError(f"Unknown projection: {proj_name}")
    np.savez_compressed(cache_path, coords=coords.astype(np.float32))
    return coords


HIGHLIGHT_K = 12  # top-K most-populated winnable folders receive distinct colors


def build_folder_color_map(
    folder_ids: np.ndarray, is_winnable: np.ndarray, top_k: int = HIGHLIGHT_K
) -> np.ndarray:
    """Assign a color bucket to each row.

    - Non-winnable (singleton) rows: bucket -2 (very light gray, background)
    - Winnable rows not in top-K largest folders: bucket -1 (mid gray)
    - Winnable rows in the top-K largest folders: bucket 0..top_k-1 (distinct)

    Restricting the colored set to the K largest folders gives a cleaner visual
    signal: if ABTT really pulls related fragments together, the handful of
    distinctly-colored clusters should be compact and separated, while baseline
    and whitening should scatter them across the plane.
    """
    n = len(folder_ids)
    color_idx = np.full(n, -2, dtype=np.int32)
    winnable_mask = is_winnable.astype(bool)
    if not winnable_mask.any():
        return color_idx
    color_idx[winnable_mask] = -1

    winnable_df = pd.DataFrame({"folder_id": folder_ids[winnable_mask]})
    sizes = winnable_df.groupby("folder_id").size().sort_values(ascending=False)
    top_folders = sizes.index[:top_k].tolist()
    rank_map = {fid: i for i, fid in enumerate(top_folders)}

    for i, fid in enumerate(folder_ids):
        if fid in rank_map:
            color_idx[i] = rank_map[fid]
    return color_idx


def render_grid(
    proj_name: str,
    model_order: List[str],
    methods: List[str],
    coord_cache: Dict[Tuple[str, str], np.ndarray],
    best_layers: Dict[str, int],
    color_idx: np.ndarray,
    is_winnable: np.ndarray,
    figures_dir: Path,
    stem: str,
) -> None:
    n_rows = len(model_order)
    n_cols = len(methods)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows), squeeze=False
    )

    # High-contrast palette for the top-K highlighted folders.
    palette = np.array(
        [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#e7298a", "#17becf", "#8c564b",
            "#bcbd22", "#2ca02c", "#d62728", "#1f77b4",
        ]
    )[:HIGHLIGHT_K]

    background_mask = color_idx == -2  # singletons
    mid_mask = color_idx == -1  # winnable but not in top-K
    highlight_mask = color_idx >= 0  # top-K colored
    highlight_colors = palette[color_idx[highlight_mask]]

    for r, model_name in enumerate(model_order):
        for c, method in enumerate(methods):
            ax = axes[r][c]
            coords = coord_cache.get((model_name, method))
            if coords is None:
                ax.set_facecolor("#f2f2f2")
                ax.text(
                    0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes
                )
            else:
                # Background: singletons, very faint.
                ax.scatter(
                    coords[background_mask, 0],
                    coords[background_mask, 1],
                    s=3,
                    c="#d9d9d9",
                    alpha=0.35,
                    linewidths=0,
                )
                # Mid-layer: winnable points not in the top-K focus set.
                ax.scatter(
                    coords[mid_mask, 0],
                    coords[mid_mask, 1],
                    s=6,
                    c="#9e9e9e",
                    alpha=0.45,
                    linewidths=0,
                )
                # Foreground: top-K folders, large + opaque + black edge for
                # contrast against the gray background.
                ax.scatter(
                    coords[highlight_mask, 0],
                    coords[highlight_mask, 1],
                    s=40,
                    c=highlight_colors,
                    alpha=0.95,
                    linewidths=0.4,
                    edgecolors="black",
                )

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#333333")

            if r == 0:
                ax.set_title(METHOD_DISPLAY.get(method, method), fontsize=13)

            if c == 0:
                ax.set_ylabel(
                    f"{model_name}",
                    fontsize=13,
                    fontweight="bold",
                    rotation=90,
                    labelpad=6,
                )
                layer_id = best_layers.get(model_name)
                if layer_id is not None:
                    ax.text(
                        0.02,
                        0.02,
                        f"L={layer_id}",
                        transform=ax.transAxes,
                        fontsize=9,
                        color="#444444",
                        ha="left",
                        va="bottom",
                    )

    fig.suptitle(
        f"2D {proj_name.upper()} of labelled fragments at each model's best layer",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def compute_silhouettes(
    coord_cache: Dict[Tuple[str, str], np.ndarray],
    folder_ids: np.ndarray,
    is_winnable: np.ndarray,
    out_csv: Path,
    projection: str,
) -> None:
    """Write a small sanity-check CSV: silhouette score (winnable rows only)."""
    try:
        from sklearn.metrics import silhouette_score
    except Exception:  # pragma: no cover
        return

    winnable_mask = is_winnable.astype(bool)
    rows = []
    for (model_name, method), coords in coord_cache.items():
        sub = coords[winnable_mask]
        labels = folder_ids[winnable_mask]
        # silhouette needs at least 2 labels, each with at least 2 members.
        unique, counts = np.unique(labels, return_counts=True)
        keep = unique[counts >= 2]
        if len(keep) < 2:
            rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "projection": projection,
                    "silhouette": None,
                    "n_points": int(sub.shape[0]),
                }
            )
            continue
        mask = np.isin(labels, keep)
        if mask.sum() < 10:
            rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "projection": projection,
                    "silhouette": None,
                    "n_points": int(mask.sum()),
                }
            )
            continue
        try:
            score = float(silhouette_score(sub[mask], labels[mask]))
        except Exception:
            score = None
        rows.append(
            {
                "model": model_name,
                "method": method,
                "projection": projection,
                "silhouette": score,
                "n_points": int(mask.sum()),
            }
        )

    out_df = pd.DataFrame(rows)
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        existing = existing[existing["projection"] != projection]
        out_df = pd.concat([existing, out_df], ignore_index=True)
    out_df.to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()

    split_csv = Path(args.split_csv)
    bases_root = Path(args.bases_root)
    intermediate_dir = Path(args.intermediate_dir)
    figures_dir = Path(args.figures_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(split_csv)
    train_mask = (split_df["split"].values == "train")
    folder_ids = split_df["folder_id"].values
    is_winnable = split_df["is_winnable"].values.astype(bool)
    train_folder_ids = folder_ids[train_mask]

    best_layers = load_best_layers(args.best_layer_csv)
    D_values = [int(x) for x in args.D_values.split(",") if x.strip()]

    if args.models is not None:
        requested = set(args.models)
        active_models = [m for m in MODELS if m[0] in requested]
    else:
        active_models = list(MODELS)

    model_order = [m[0] for m in active_models]

    color_idx = build_folder_color_map(folder_ids, is_winnable)

    proj_caches: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {
        p: {} for p in args.projections
    }

    for display, slug, repr_name in active_models:
        if display not in best_layers:
            print(f"[skip] no best_layer entry for {display}")
            continue
        layer = best_layers[display]
        for method in args.methods:
            pooling = method_pooling(method)
            emb_path = embedding_file(bases_root, slug, repr_name, pooling, layer)
            if not emb_path.exists():
                print(f"[skip] missing embeddings: {emb_path}")
                continue

            cache_base = intermediate_dir / f"{slug}_L{layer}_{method}"

            for proj in args.projections:
                proj_cache = cache_base.with_name(cache_base.name + f"_{proj}.npz")

                if args.render_only or (
                    proj_cache.exists() and not args.force_recompute
                ):
                    if proj_cache.exists():
                        arr = np.load(proj_cache)["coords"]
                        if arr.shape[1] == 2 and arr.shape[0] == len(folder_ids):
                            proj_caches[proj][(display, method)] = arr
                            print(f"[cache] {display} | {method} | {proj}: loaded")
                            continue

                if args.render_only:
                    continue

                # (Re)compute: load raw, apply method, normalize, project.
                emb = np.load(emb_path)
                if emb.shape[0] != len(folder_ids):
                    print(
                        f"[skip] row mismatch at {emb_path}: "
                        f"{emb.shape[0]} vs {len(folder_ids)}"
                    )
                    continue

                post, actual_D = apply_method(
                    emb, train_mask, train_folder_ids, method, D_values
                )
                post_norm = l2_normalize(post)
                coords = compute_or_load_projection(
                    proj_cache, proj, post_norm, args
                )
                proj_caches[proj][(display, method)] = coords
                suffix = f" (D={actual_D})" if actual_D is not None else ""
                print(f"[ok] {display} | {method}{suffix} | {proj}: {coords.shape}")

    # Render and write silhouette CSV per projection.
    silhouette_csv = intermediate_dir / "cluster_silhouette.csv"
    for proj in args.projections:
        stem = f"fig_{proj}_per_model"
        render_grid(
            proj_name=proj,
            model_order=model_order,
            methods=list(args.methods),
            coord_cache=proj_caches[proj],
            best_layers=best_layers,
            color_idx=color_idx,
            is_winnable=is_winnable,
            figures_dir=figures_dir,
            stem=stem,
        )
        print(f"[render] {figures_dir / stem}.pdf")

        compute_silhouettes(
            coord_cache=proj_caches[proj],
            folder_ids=folder_ids,
            is_winnable=is_winnable,
            out_csv=silhouette_csv,
            projection=proj,
        )

    print(f"[done] silhouette sanity check: {silhouette_csv}")


if __name__ == "__main__":
    main()
