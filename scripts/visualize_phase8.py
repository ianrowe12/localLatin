"""Phase 8 visualization: paper-quality figures and LaTeX table.

Figures:
1. Layer-wise Spearman (the dip) — one subplot per language
2. R-value sensitivity heatmap
3. Global anisotropy — off-diag mean before/after PC removal
4. Canon clean vs Phase 7 — side-by-side layer accuracy
5. Table 1 (LaTeX) — best Spearman per language per method
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 visualizations.")
    parser.add_argument(
        "--musts_csv", default="",
        help="Path to phase8_musts_sweep.csv.",
    )
    parser.add_argument(
        "--canon_csv", default="",
        help="Path to phase8_canon_sweep.csv.",
    )
    parser.add_argument(
        "--phase7_canon_csv", default="",
        help="Path to phase7_layer_sweep.csv (for comparison).",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figsize_w", type=float, default=12)
    parser.add_argument("--figsize_h", type=float, default=8)
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


METHOD_COLORS = {
    "baseline_mean": "#999999",
    "sif_only": "#4daf4a",
    "sif_abtt_fixed": "#377eb8",
    "sif_abtt_optimal": "#e41a1c",
    "whitening": "#ff7f00",
    "variance_filter": "#984ea3",
}

METHOD_LABELS = {
    "baseline_mean": "Baseline (mean)",
    "sif_only": "SIF only",
    "sif_abtt_fixed": "SIF+ABTT (D=10)",
    "sif_abtt_optimal": "SIF+ABTT (optimal D)",
    "whitening": "PCA Whitening",
    "variance_filter": "Variance Filter",
}


# --- Figure 1: Layer-wise Spearman ---

def plot_layerwise_spearman(df: pd.DataFrame, out_dir: Path, dpi: int):
    """X=layer, Y=Spearman. Lines: methods. Subplots: languages."""
    models = df["model"].unique()
    languages = df["language"].unique()
    methods_to_plot = ["baseline_mean", "sif_only", "sif_abtt_optimal"]

    for model_name in models:
        mdf = df[df["model"] == model_name]
        n_langs = len(languages)
        fig, axes = plt.subplots(1, n_langs, figsize=(5 * n_langs, 4), sharey=True)
        if n_langs == 1:
            axes = [axes]

        for ax, lang in zip(axes, languages):
            ldf = mdf[mdf["language"] == lang]
            for method in methods_to_plot:
                subset = ldf[ldf["method"] == method].sort_values("layer")
                if subset.empty:
                    continue
                ax.plot(
                    subset["layer"], subset["spearman_test"],
                    marker="o", markersize=4,
                    color=METHOD_COLORS.get(method, "#000"),
                    label=METHOD_LABELS.get(method, method),
                )
            ax.set_xlabel("Layer")
            ax.set_title(lang.capitalize())
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Spearman (test)")
        axes[-1].legend(loc="best")
        slug = model_name.replace("/", "_")
        fig.suptitle(f"Layer-wise Spearman — {model_name}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig1_layerwise_spearman_{slug}.png", dpi=dpi)
        plt.close(fig)
        print(f"  Saved fig1 for {model_name}")


# --- Figure 2: R-value Sensitivity Heatmap ---

def plot_r_heatmap(df: pd.DataFrame, out_dir: Path, dpi: int):
    """X=D value, Y=language. Color=best test Spearman across layers."""
    models = df["model"].unique()
    # Only look at sif_abtt methods with different D values
    abtt_rows = df[df["method"].isin(["sif_abtt_fixed", "sif_abtt_optimal"])]
    if abtt_rows.empty:
        # Try to reconstruct from train sweep data if available
        print("  Skipping R heatmap (no ABTT results with varying D)")
        return

    for model_name in models:
        mdf = abtt_rows[abtt_rows["model"] == model_name]
        languages = mdf["language"].unique()
        D_values = sorted(mdf["D"].unique())

        if len(D_values) < 2:
            continue

        matrix = np.full((len(languages), len(D_values)), np.nan)
        for i, lang in enumerate(languages):
            for j, d in enumerate(D_values):
                subset = mdf[(mdf["language"] == lang) & (mdf["D"] == d)]
                if not subset.empty:
                    matrix[i, j] = subset["spearman_test"].max()

        fig, ax = plt.subplots(figsize=(max(6, len(D_values)), max(3, len(languages) * 0.8)))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(D_values)))
        ax.set_xticklabels(D_values)
        ax.set_yticks(range(len(languages)))
        ax.set_yticklabels([l.capitalize() for l in languages])
        ax.set_xlabel("D (components removed)")
        ax.set_title(f"R-value Sensitivity — {model_name}")
        plt.colorbar(im, ax=ax, label="Spearman (test)")

        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

        fig.tight_layout()
        slug = model_name.replace("/", "_")
        fig.savefig(out_dir / f"fig2_r_heatmap_{slug}.png", dpi=dpi)
        plt.close(fig)
        print(f"  Saved fig2 for {model_name}")


# --- Figure 3: Global Anisotropy ---

def plot_anisotropy(df: pd.DataFrame, out_dir: Path, dpi: int):
    """X=layer, Y=off-diag mean. Before/after PC removal. Per language."""
    if "off_diag_mean" not in df.columns:
        print("  Skipping anisotropy (no off_diag_mean column)")
        return

    models = df["model"].unique()
    for model_name in models:
        mdf = df[df["model"] == model_name]
        languages = mdf["language"].unique() if "language" in mdf.columns else ["all"]
        n_langs = len(languages)

        fig, axes = plt.subplots(1, n_langs, figsize=(5 * n_langs, 4), sharey=True)
        if n_langs == 1:
            axes = [axes]

        for ax, lang in zip(axes, languages):
            if "language" in mdf.columns:
                ldf = mdf[mdf["language"] == lang]
            else:
                ldf = mdf

            for method, label, color, ls in [
                ("baseline_mean", "Before (baseline)", "#999999", "-"),
                ("sif_abtt_optimal", "After (SIF+ABTT)", "#e41a1c", "--"),
            ]:
                subset = ldf[ldf["method"] == method].sort_values("layer")
                if subset.empty:
                    continue
                ax.plot(
                    subset["layer"], subset["off_diag_mean"],
                    marker="o", markersize=3, color=color, linestyle=ls, label=label,
                )

            ax.set_xlabel("Layer")
            ax.set_title(lang.capitalize() if lang != "all" else model_name)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Off-diagonal mean (anisotropy)")
        axes[-1].legend(loc="best")
        fig.suptitle(f"Anisotropy — {model_name}", fontsize=14)
        fig.tight_layout()
        slug = model_name.replace("/", "_")
        fig.savefig(out_dir / f"fig3_anisotropy_{slug}.png", dpi=dpi)
        plt.close(fig)
        print(f"  Saved fig3 for {model_name}")


# --- Figure 4: Canon Clean vs Phase 7 ---

def plot_canon_comparison(
    clean_df: pd.DataFrame,
    phase7_df: Optional[pd.DataFrame],
    out_dir: Path,
    dpi: int,
):
    """Side-by-side layer accuracy: Phase 7 (leaked) vs Phase 8 (clean)."""
    if phase7_df is None or phase7_df.empty:
        print("  Skipping canon comparison (no Phase 7 data)")
        return

    models = clean_df["model"].unique()
    for model_name in models:
        cdf = clean_df[clean_df["model"] == model_name]
        pdf = phase7_df[phase7_df["model"] == model_name]
        if pdf.empty:
            continue

        reprs = cdf["repr"].unique()
        fig, axes = plt.subplots(1, len(reprs), figsize=(6 * len(reprs), 4), sharey=True)
        if len(reprs) == 1:
            axes = [axes]

        for ax, repr_name in zip(axes, reprs):
            # Phase 7 baseline
            p7_base = pdf[(pdf["repr"] == repr_name) & (pdf["method"] == "baseline")].sort_values("layer")
            if not p7_base.empty:
                ax.plot(
                    p7_base["layer"], p7_base["acc@1_winnable"],
                    marker="s", markersize=4, color="#377eb8", linestyle="--",
                    label="Phase 7 baseline",
                )

            # Phase 8 baseline
            p8_base = cdf[(cdf["repr"] == repr_name) & (cdf["method"] == "baseline") & (cdf["pooling"] == "mean")].sort_values("layer")
            if not p8_base.empty:
                ax.plot(
                    p8_base["layer"], p8_base["acc@1_winnable"],
                    marker="o", markersize=4, color="#e41a1c", linestyle="-",
                    label="Phase 8 baseline (clean)",
                )

            # Phase 8 ABTT
            p8_abtt = cdf[(cdf["repr"] == repr_name) & (cdf["method"] == "abtt") & (cdf["pooling"] == "sif")].sort_values("layer")
            if not p8_abtt.empty:
                ax.plot(
                    p8_abtt["layer"], p8_abtt["acc@1_winnable"],
                    marker="^", markersize=4, color="#4daf4a", linestyle="-",
                    label="Phase 8 SIF+ABTT (clean)",
                )

            ax.set_xlabel("Layer")
            ax.set_title(f"{repr_name}")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Acc@1 (winnable)")
        axes[-1].legend(loc="best")
        slug = model_name.replace("/", "_")
        fig.suptitle(f"Canon: Phase 7 vs Phase 8 — {model_name}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig4_canon_comparison_{slug}.png", dpi=dpi)
        plt.close(fig)
        print(f"  Saved fig4 for {model_name}")


# --- Table 1: LaTeX ---

def generate_latex_table(df: pd.DataFrame, out_dir: Path):
    """Best Spearman per language per method."""
    methods = [
        "baseline_mean", "sif_only", "sif_abtt_fixed",
        "sif_abtt_optimal", "whitening", "variance_filter",
    ]
    languages = sorted(df["language"].unique())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Best test Spearman correlation per method and language.}")
    lines.append(r"\label{tab:phase8_spearman}")
    cols = "l" + "c" * len(languages)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(l.capitalize() for l in languages) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in methods:
        mdf = df[df["method"] == method]
        vals = []
        for lang in languages:
            subset = mdf[mdf["language"] == lang]
            if subset.empty:
                vals.append("--")
            else:
                best = subset["spearman_test"].max()
                if np.isnan(best):
                    vals.append("--")
                else:
                    vals.append(f"{best:.3f}")
        label = METHOD_LABELS.get(method, method)
        row = f"{label} & " + " & ".join(vals) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    tex_path = out_dir / "table1_spearman.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Saved LaTeX table: {tex_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    if args.musts_csv:
        print("Loading MUSTS results...")
        musts_df = pd.read_csv(args.musts_csv)

        print("Generating MUSTS figures...")
        plot_layerwise_spearman(musts_df, out_dir, args.dpi)
        plot_r_heatmap(musts_df, out_dir, args.dpi)
        plot_anisotropy(musts_df, out_dir, args.dpi)
        generate_latex_table(musts_df, out_dir)

    if args.canon_csv:
        print("Loading canon results...")
        canon_df = pd.read_csv(args.canon_csv)
        phase7_df = None
        if args.phase7_canon_csv:
            phase7_df = pd.read_csv(args.phase7_canon_csv)
        plot_canon_comparison(canon_df, phase7_df, out_dir, args.dpi)

    print("Visualization complete.")


if __name__ == "__main__":
    main()
