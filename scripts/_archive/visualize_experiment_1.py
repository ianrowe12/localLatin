"""Phase 9 Experiment 1 visualization.

Figure 1: "The Dip Graph" — AUCROC vs. Layer Depth (one subplot per model)
Figure 2: Threshold Histogram — cosine similarity distributions with decision boundary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_retrieval import l2_normalize, similarity_matrix, upper_triangle, upper_triangle_labels


METHOD_COLORS = {
    "baseline": "#999999",
    "sif_only": "#4daf4a",
    "sif_abtt_fixed": "#377eb8",
    "sif_abtt_optimal": "#e41a1c",
    "whitening": "#ff7f00",
}

METHOD_LABELS = {
    "baseline": "Baseline (mean)",
    "sif_only": "SIF only",
    "sif_abtt_fixed": "SIF+ABTT (D=10)",
    "sif_abtt_optimal": "SIF+ABTT (optimal D)",
    "whitening": "PCA Whitening",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 Experiment 1 visualization.")
    parser.add_argument("--results_csv", required=True, help="Output from evaluate_vectors.py.")
    parser.add_argument("--out_dir", required=True, help="Directory for saved figures.")
    parser.add_argument(
        "--split_csv", default="",
        help="phase9_split.csv (for threshold histogram data).",
    )
    parser.add_argument(
        "--emb_dir", default="",
        help="Path to embeddings directory (for threshold histogram).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def plot_dip_graph(df: pd.DataFrame, out_dir: Path, dpi: int):
    """Figure 1: AUCROC vs. Layer Depth, one subplot per model."""
    models = df["model"].unique()
    methods_to_plot = ["baseline", "sif_only", "sif_abtt_optimal", "whitening"]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        mdf = df[df["model"] == model_name]
        # Filter to hidden repr (primary comparison)
        repr_filter = mdf[mdf["repr"] == "hidden"]
        if repr_filter.empty:
            repr_filter = mdf

        for method in methods_to_plot:
            subset = repr_filter[repr_filter["method"] == method].sort_values("layer")
            if subset.empty:
                continue
            ax.plot(
                subset["layer"], subset["aucroc"],
                marker="o", markersize=4,
                color=METHOD_COLORS.get(method, "#000"),
                label=METHOD_LABELS.get(method, method),
                linewidth=1.5,
            )

        ax.set_xlabel("Layer")
        slug = model_name.split("/")[-1]
        ax.set_title(slug)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    axes[0].set_ylabel("AUCROC")
    # Place legend outside last subplot
    axes[-1].legend(loc="lower right", fontsize=8)

    fig.suptitle("AUCROC vs. Layer Depth (hidden states)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_dip_graph.png", dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig1_dip_graph.png'}")

    # Also make one with ff1 if available
    ff1_data = df[df["repr"] == "ff1"]
    if not ff1_data.empty:
        ff1_models = ff1_data["model"].unique()
        n_ff1 = len(ff1_models)
        fig2, axes2 = plt.subplots(1, n_ff1, figsize=(5 * n_ff1, 4.5), sharey=True)
        if n_ff1 == 1:
            axes2 = [axes2]

        for ax, model_name in zip(axes2, ff1_models):
            mdf = ff1_data[ff1_data["model"] == model_name]
            for method in methods_to_plot:
                subset = mdf[mdf["method"] == method].sort_values("layer")
                if subset.empty:
                    continue
                ax.plot(
                    subset["layer"], subset["aucroc"],
                    marker="s", markersize=4,
                    color=METHOD_COLORS.get(method, "#000"),
                    label=METHOD_LABELS.get(method, method),
                    linewidth=1.5,
                )
            ax.set_xlabel("Layer")
            slug = model_name.split("/")[-1]
            ax.set_title(slug)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.05)

        axes2[0].set_ylabel("AUCROC")
        axes2[-1].legend(loc="lower right", fontsize=8)
        fig2.suptitle("AUCROC vs. Layer Depth (FF1)", fontsize=14, y=1.02)
        fig2.tight_layout()
        fig2.savefig(out_dir / "fig1_dip_graph_ff1.png", dpi=dpi)
        plt.close(fig2)
        print(f"Saved: {out_dir / 'fig1_dip_graph_ff1.png'}")


def plot_threshold_histogram(
    results_df: pd.DataFrame,
    split_csv: str,
    emb_dir: str,
    out_dir: Path,
    dpi: int,
):
    """Figure 2: Histogram of cosine similarities with threshold line.

    Uses train-set pair similarities from a representative (model, layer, method).
    """
    if not split_csv or not emb_dir:
        print("Skipping threshold histogram (no split_csv or emb_dir provided).")
        return

    split_meta = pd.read_csv(split_csv)
    emb_path = Path(emb_dir)

    # Pick best representative: highest AUCROC from sif_abtt_optimal on hidden
    candidates = results_df[
        (results_df["method"] == "sif_abtt_optimal") & (results_df["repr"] == "hidden")
    ]
    if candidates.empty:
        candidates = results_df[results_df["method"] == "sif_abtt_optimal"]
    if candidates.empty:
        candidates = results_df
    if candidates.empty:
        print("No results to pick representative from. Skipping histogram.")
        return

    best = candidates.loc[candidates["aucroc"].idxmax()]
    model_name = best["model"]
    repr_name = best["repr"]
    layer = int(best["layer"])
    tau = best["tau"]

    # Try to load the SIF embeddings for ABTT processing
    slug = model_name.replace("/", "_")
    suffix = "_sif"
    fname = f"{repr_name}_layer{layer}_embeddings{suffix}.npy"

    candidate_paths = [
        emb_path / slug / f"{repr_name}_sif" / fname,
        emb_path / slug / fname,
    ]
    raw_emb_path = None
    for p in candidate_paths:
        if p.exists():
            raw_emb_path = p
            break

    if raw_emb_path is None:
        print(f"Could not find embedding file for histogram. Tried: {candidate_paths}")
        return

    emb_all = np.load(raw_emb_path)
    train_mask = split_meta["split"].values == "train"
    train_emb = emb_all[train_mask]
    train_folder_ids = split_meta.loc[train_mask, "folder_id"].values

    # Apply ABTT with the optimal D
    from sif_abtt import EmbeddingCleaner
    D = int(best["D"])
    cleaner = EmbeddingCleaner(num_components=D, center=True)
    cleaner.fit(train_emb)
    cleaned = cleaner.transform(train_emb)
    cleaned_norm = l2_normalize(cleaned)

    sim = similarity_matrix(cleaned_norm)
    sims = upper_triangle(sim)
    labels = upper_triangle_labels(train_folder_ids)

    pos_sims = sims[labels.astype(bool)]
    neg_sims = sims[~labels.astype(bool)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.2, 1.0, 100)

    ax.hist(neg_sims, bins=bins, alpha=0.6, color="#e41a1c", label="Distinct pairs", density=True)
    ax.hist(pos_sims, bins=bins, alpha=0.6, color="#377eb8", label="Semantically equal pairs", density=True)
    ax.axvline(x=tau, color="black", linestyle="--", linewidth=1.5, label=f"Threshold (tau={tau:.3f})")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    slug_short = model_name.split("/")[-1]
    ax.set_title(f"Train-set similarity distribution — {slug_short} L{layer} SIF+ABTT(D={D})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_threshold_histogram.png", dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig2_threshold_histogram.png'}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("Loading results...")
    df = pd.read_csv(args.results_csv)
    print(f"  {len(df)} rows, models: {df['model'].unique()}")

    print("\nGenerating Figure 1: Dip Graph...")
    plot_dip_graph(df, out_dir, args.dpi)

    print("\nGenerating Figure 2: Threshold Histogram...")
    plot_threshold_histogram(
        df, args.split_csv, args.emb_dir, out_dir, args.dpi
    )

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
