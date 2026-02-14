"""Phase 9 Experiment 2: Visualization — 3 publication figures.

Figure 1: Layer-wise Spearman (3x5 grid: models x languages)
Figure 2: Optimal D heatmap (models x languages)
Figure 3: Lexical vs Semantic scatter (low-resource languages)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "baseline_mean": "#377eb8",
    "whitening": "#984ea3",
    "sif_abtt_optimal": "#e41a1c",
}

METHOD_LABELS = {
    "baseline_mean": "Baseline (mean)",
    "whitening": "PCA Whitening",
    "sif_abtt_optimal": "SIF+ABTT (optimal D)",
}

LANG_COLORS = {
    "tamil": "#e41a1c",
    "sinhala": "#377eb8",
    "serbian": "#4daf4a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 Exp 2 visualization.")
    parser.add_argument("--results_csv", required=True, help="results.csv from evaluate.")
    parser.add_argument("--diagnostics_csv", default="", help="diagnostics.csv from diagnostics.")
    parser.add_argument("--cache_dir", default="", help="Embedding cache dir (for pair sims).")
    parser.add_argument("--out_dir", required=True)
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


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def short_name(model: str) -> str:
    return model.split("/")[-1]


# ── Figure 1: Layer-wise Spearman ───────────────────────────────

def plot_layerwise_spearman(df: pd.DataFrame, out_dir: Path, dpi: int):
    """3x5 grid: rows=models, columns=languages."""
    models = df["model"].unique()
    languages = df["language"].unique()
    methods = ["baseline_mean", "whitening", "sif_abtt_optimal"]

    n_models = len(models)
    n_langs = len(languages)

    fig, axes = plt.subplots(
        n_models, n_langs,
        figsize=(3.5 * n_langs, 3.5 * n_models),
        sharex=False, sharey=True,
        squeeze=False,
    )

    for i, model_name in enumerate(models):
        for j, lang in enumerate(languages):
            ax = axes[i, j]
            subset = df[(df["model"] == model_name) & (df["language"] == lang)]

            for method in methods:
                mdata = subset[subset["method"] == method].sort_values("layer")
                if mdata.empty:
                    continue
                ax.plot(
                    mdata["layer"], mdata["spearman_test"],
                    marker="o", markersize=3,
                    color=METHOD_COLORS.get(method, "#000"),
                    label=METHOD_LABELS.get(method, method),
                    linewidth=1.5,
                )

            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(lang.capitalize())
            if j == 0:
                ax.set_ylabel(f"{short_name(model_name)}\nSpearman (test)")
            if i == n_models - 1:
                ax.set_xlabel("Layer")

    # Single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(methods),
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Layer-wise Spearman by Model and Language", fontsize=14, y=1.01)
    fig.tight_layout()
    path = out_dir / "fig1_layerwise_spearman.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: Optimal D Heatmap ─────────────────────────────────

def plot_optimal_d_heatmap(df: pd.DataFrame, out_dir: Path, dpi: int):
    """Heatmap of optimal D at the best-performing layer per (model, language)."""
    abtt = df[df["method"] == "sif_abtt_optimal"].copy()
    if abtt.empty:
        print("No sif_abtt_optimal data for heatmap.")
        return

    models = abtt["model"].unique()
    languages = abtt["language"].unique()

    # For each (model, lang): find layer with best test Spearman, get its D
    heatmap = np.full((len(models), len(languages)), np.nan)
    for i, model_name in enumerate(models):
        for j, lang in enumerate(languages):
            subset = abtt[(abtt["model"] == model_name) & (abtt["language"] == lang)]
            if subset.empty:
                continue
            best_row = subset.loc[subset["spearman_test"].idxmax()]
            heatmap[i, j] = best_row["D"]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(languages)), max(3, 1.2 * len(models))))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels([l.capitalize() for l in languages], rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_name(m) for m in models])

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(languages)):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_title("Optimal D at Best Layer (SIF+ABTT)")
    fig.colorbar(im, ax=ax, label="D")
    fig.tight_layout()
    path = out_dir / "fig2_optimal_d_heatmap.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: Lexical vs Semantic Scatter ───────────────────────

def plot_lexical_vs_semantic(
    results_df: pd.DataFrame,
    diagnostics_csv: str,
    cache_dir: str,
    out_dir: Path,
    dpi: int,
):
    """Scatter: X=ROUGE-L, Y=cosine sim, colored by language (low-resource only)."""
    if not diagnostics_csv or not cache_dir:
        print("Skipping Figure 3 (no diagnostics_csv or cache_dir).")
        return

    diag = pd.read_csv(diagnostics_csv)
    cache_path = Path(cache_dir)
    low_resource = ["tamil", "sinhala", "serbian"]
    diag_lr = diag[diag["language"].isin(low_resource)]

    if diag_lr.empty:
        print("No low-resource diagnostics data for scatter.")
        return

    # For each low-resource language, find the best (model, layer, method)
    abtt = results_df[results_df["method"] == "sif_abtt_optimal"]

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False

    for lang in low_resource:
        lang_diag = diag_lr[diag_lr["language"] == lang]
        if lang_diag.empty:
            continue

        # Find best model+layer for this language
        lang_results = abtt[abtt["language"] == lang]
        if lang_results.empty:
            continue
        best = lang_results.loc[lang_results["spearman_test"].idxmax()]
        slug = model_slug(best["model"])
        layer = int(best["layer"])

        sims_path = cache_path / slug / lang / f"layer{layer}_sif_abtt_optimal_pair_sims.npy"
        if not sims_path.exists():
            print(f"  Pair sims not found for {lang}: {sims_path}")
            continue

        pair_sims = np.load(sims_path)
        # Diagnostics are test-only; pair_sims should align
        n = min(len(lang_diag), len(pair_sims))
        rouge_vals = lang_diag["rouge_l"].values[:n]
        cos_vals = pair_sims[:n]

        ax.scatter(
            rouge_vals, cos_vals,
            alpha=0.4, s=15,
            color=LANG_COLORS.get(lang, "#000"),
            label=f"{lang.capitalize()} ({short_name(best['model'])} L{layer})",
        )
        plotted = True

    if not plotted:
        print("No data to scatter for Figure 3.")
        plt.close(fig)
        return

    ax.set_xlabel("ROUGE-L (lexical overlap)")
    ax.set_ylabel("Cosine Similarity (SIF+ABTT)")
    ax.set_title("Lexical vs Semantic Similarity (Low-Resource)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "fig3_lexical_vs_semantic.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("Loading results...")
    df = pd.read_csv(args.results_csv)
    print(f"  {len(df)} rows, models: {list(df['model'].unique())}")

    print("\nGenerating Figure 1: Layer-wise Spearman...")
    plot_layerwise_spearman(df, out_dir, args.dpi)

    print("\nGenerating Figure 2: Optimal D Heatmap...")
    plot_optimal_d_heatmap(df, out_dir, args.dpi)

    print("\nGenerating Figure 3: Lexical vs Semantic Scatter...")
    plot_lexical_vs_semantic(df, args.diagnostics_csv, args.cache_dir, out_dir, args.dpi)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
