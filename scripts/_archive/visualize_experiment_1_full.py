"""Phase 9 Experiment 1 — comprehensive publication-quality figures.

Figures:
1. The Dip Graph: AUCROC vs Layer (baseline vs SIF+ABTT vs SIF-only), per model
2. Method comparison heatmap: best AUCROC per (model, method)
3. Cosine gap across layers: same_avg - diff_avg per method
4. Assignment accuracy breakdown: existing vs new vs overall, grouped by method
5. Threshold histogram: cosine similarity distributions (train set)
6. Acc@1 vs Layer: retrieval accuracy across depths
7. Optimal D distribution: which D was selected per model/layer
8. Compact summary table saved as figure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


# ── Style ────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "baseline": "#888888",
    "sif_only": "#4daf4a",
    "sif_abtt_fixed": "#377eb8",
    "sif_abtt_optimal": "#e41a1c",
    "whitening": "#ff7f00",
}
METHOD_LABELS = {
    "baseline": "Baseline (mean pool)",
    "sif_only": "SIF only",
    "sif_abtt_fixed": "SIF + ABTT (D=10)",
    "sif_abtt_optimal": "SIF + ABTT (optimal D)",
    "whitening": "PCA Whitening",
}
METHOD_ORDER = ["baseline", "sif_only", "sif_abtt_fixed", "sif_abtt_optimal", "whitening"]
MODEL_SHORT = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def short(model: str) -> str:
    return MODEL_SHORT.get(model, model.split("/")[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", required=True)
    p.add_argument("--split_csv", default="")
    p.add_argument("--emb_dir", default="", help="phase9_bases dir for histogram")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# ── Figure 1: The Dip ────────────────────────────────────────────────────
def fig1_dip_graph(df: pd.DataFrame, out: Path, dpi: int):
    """AUCROC vs layer for hidden states, one subplot per model."""
    hidden = df[df["repr"] == "hidden"]
    models = list(hidden["model"].unique())
    methods = ["baseline", "sif_only", "sif_abtt_optimal"]

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = hidden[hidden["model"] == model]
        for method in methods:
            sub = mdf[mdf["method"] == method].sort_values("layer")
            if sub.empty:
                continue
            ax.plot(
                sub["layer"], sub["aucroc"],
                marker="o", markersize=5, linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_xlabel("Layer")
        ax.set_title(short(model), fontweight="bold")
        ax.set_xticks(range(0, 13))
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_ylim(0.40, 1.02)

    axes[0].set_ylabel("AUCROC (test-only)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("The Anisotropy Dip: AUCROC vs. Layer Depth", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out / "fig1_dip_graph.png", dpi=dpi)
    fig.savefig(out / "fig1_dip_graph.pdf", dpi=dpi)
    plt.close(fig)
    print(f"  Fig 1 saved")


# ── Figure 1b: FF1 Dip ──────────────────────────────────────────────────
def fig1b_ff1_dip(df: pd.DataFrame, out: Path, dpi: int):
    ff1 = df[df["repr"] == "ff1"]
    if ff1.empty:
        return
    models = list(ff1["model"].unique())
    methods = ["baseline", "sif_only", "sif_abtt_optimal"]

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = ff1[ff1["model"] == model]
        for method in methods:
            sub = mdf[mdf["method"] == method].sort_values("layer")
            if sub.empty:
                continue
            ax.plot(
                sub["layer"], sub["aucroc"],
                marker="s", markersize=5, linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_xlabel("Layer")
        ax.set_title(short(model), fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_ylim(0.40, 1.02)

    axes[0].set_ylabel("AUCROC (test-only)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("AUCROC vs. Layer Depth (FF1 Representations)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out / "fig1b_dip_ff1.png", dpi=dpi)
    fig.savefig(out / "fig1b_dip_ff1.pdf", dpi=dpi)
    plt.close(fig)
    print(f"  Fig 1b saved")


# ── Figure 2: Method comparison heatmap ──────────────────────────────────
def fig2_method_heatmap(df: pd.DataFrame, out: Path, dpi: int):
    """Best AUCROC per (model x repr, method) as heatmap."""
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    model_reprs = []
    for model in df["model"].unique():
        for repr_name in df[df["model"] == model]["repr"].unique():
            model_reprs.append((model, repr_name))

    matrix = np.full((len(model_reprs), len(methods)), np.nan)
    for i, (model, repr_name) in enumerate(model_reprs):
        for j, method in enumerate(methods):
            sub = df[(df["model"] == model) & (df["repr"] == repr_name) & (df["method"] == method)]
            if not sub.empty:
                matrix[i, j] = sub["aucroc"].max()

    y_labels = [f"{short(m)} / {r}" for m, r in model_reprs]
    x_labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_reprs)))
    ax.set_yticklabels(y_labels, fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                color = "white" if val < 0.65 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Best AUCROC", shrink=0.8)
    ax.set_title("Best AUCROC by Model/Representation and Method", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig2_method_heatmap.png", dpi=dpi)
    fig.savefig(out / "fig2_method_heatmap.pdf", dpi=dpi)
    plt.close(fig)
    print(f"  Fig 2 saved")


# ── Figure 3: Cosine gap across layers ───────────────────────────────────
def fig3_cosine_gap(df: pd.DataFrame, out: Path, dpi: int):
    """Gap (same_avg - diff_avg) vs layer for each method, per model."""
    hidden = df[df["repr"] == "hidden"]
    models = list(hidden["model"].unique())
    methods = ["baseline", "sif_only", "sif_abtt_optimal"]

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = hidden[hidden["model"] == model]
        for method in methods:
            sub = mdf[mdf["method"] == method].sort_values("layer")
            if sub.empty:
                continue
            ax.plot(
                sub["layer"], sub["gap"],
                marker="o", markersize=4, linewidth=1.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_xlabel("Layer")
        ax.set_title(short(model), fontweight="bold")
        ax.set_xticks(range(0, 13))
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    axes[0].set_ylabel("Cosine Gap (same_avg - diff_avg)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Separability: Cosine Gap Between Same vs. Different Pairs", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out / "fig3_cosine_gap.png", dpi=dpi)
    fig.savefig(out / "fig3_cosine_gap.pdf", dpi=dpi)
    plt.close(fig)
    print(f"  Fig 3 saved")


# ── Figure 4: Assignment accuracy bars ───────────────────────────────────
def fig4_assignment_accuracy(df: pd.DataFrame, out: Path, dpi: int):
    """Grouped bar chart: existing_acc, new_acc, overall per method (best layer per model)."""
    hidden = df[df["repr"] == "hidden"]
    methods = [m for m in METHOD_ORDER if m in hidden["method"].unique()]

    # For each model/method, pick the layer with best overall_assignment_acc
    rows = []
    for model in hidden["model"].unique():
        for method in methods:
            sub = hidden[(hidden["model"] == model) & (hidden["method"] == method)]
            if sub.empty:
                continue
            best = sub.loc[sub["overall_assignment_acc"].idxmax()]
            rows.append({
                "model": short(model),
                "method": method,
                "existing_acc": best["existing_acc"],
                "new_acc": best["new_acc"],
                "overall": best["overall_assignment_acc"],
                "layer": int(best["layer"]),
            })

    adf = pd.DataFrame(rows)
    models = adf["model"].unique()
    n_methods = len(methods)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    bar_width = 0.22
    x = np.arange(n_methods)

    for ax, model_name in zip(axes, models):
        msub = adf[adf["model"] == model_name]
        existing = []
        new = []
        overall = []
        for method in methods:
            row = msub[msub["method"] == method]
            if row.empty:
                existing.append(0)
                new.append(0)
                overall.append(0)
            else:
                existing.append(row.iloc[0]["existing_acc"])
                new.append(row.iloc[0]["new_acc"])
                overall.append(row.iloc[0]["overall"])

        ax.bar(x - bar_width, existing, bar_width, label="Existing (has partner)", color="#377eb8", alpha=0.85)
        ax.bar(x, new, bar_width, label="New (singleton in test)", color="#e41a1c", alpha=0.85)
        ax.bar(x + bar_width, overall, bar_width, label="Overall", color="#4daf4a", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m).replace(" (", "\n(") for m in methods], fontsize=8, rotation=30, ha="right")
        ax.set_title(model_name, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle("Directory Assignment: Existing vs. New File Classification", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out / "fig4_assignment_accuracy.png", dpi=dpi)
    fig.savefig(out / "fig4_assignment_accuracy.pdf", dpi=dpi)
    plt.close(fig)
    print(f"  Fig 4 saved")


# ── Figure 5: Threshold histogram ────────────────────────────────────────
def fig5_threshold_histogram(
    results_df: pd.DataFrame, split_csv: str, emb_dir: str, out: Path, dpi: int,
):
    """Cosine similarity histograms for best configuration."""
    if not split_csv or not emb_dir:
        print("  Skipping Fig 5 (no split_csv or emb_dir)")
        return

    from canon_retrieval import l2_normalize, similarity_matrix, upper_triangle, upper_triangle_labels
    from sif_abtt import EmbeddingCleaner

    split_meta = pd.read_csv(split_csv)
    emb_root = Path(emb_dir)

    # Pick best sif_abtt_optimal hidden
    cands = results_df[(results_df["method"] == "sif_abtt_optimal") & (results_df["repr"] == "hidden")]
    if cands.empty:
        print("  Skipping Fig 5 (no sif_abtt_optimal results)")
        return

    best = cands.loc[cands["aucroc"].idxmax()]
    slug = best["model"].replace("/", "_")
    layer = int(best["layer"])
    D = int(best["D"])
    tau = best["tau"]

    fname = f"hidden_layer{layer}_embeddings_sif.npy"
    for candidate in [
        emb_root / slug / f"hidden_sif" / fname,
        emb_root / slug / fname,
    ]:
        if candidate.exists():
            emb_all = np.load(candidate)
            break
    else:
        print(f"  Skipping Fig 5 (embedding file not found)")
        return

    train_mask = split_meta["split"].values == "train"
    train_emb = emb_all[train_mask]
    train_fids = split_meta.loc[train_mask, "folder_id"].values

    cleaner = EmbeddingCleaner(num_components=D, center=True)
    cleaner.fit(train_emb)
    cleaned = cleaner.transform(train_emb)
    cleaned_norm = l2_normalize(cleaned)

    sim = similarity_matrix(cleaned_norm)
    sims = upper_triangle(sim)
    labels = upper_triangle_labels(train_fids).astype(bool)

    pos_sims = sims[labels]
    neg_sims = sims[~labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-0.3, 1.0, 120)

    ax.hist(neg_sims, bins=bins, alpha=0.55, color="#e41a1c", label=f"Different directory (n={len(neg_sims):,})", density=True)
    ax.hist(pos_sims, bins=bins, alpha=0.65, color="#377eb8", label=f"Same directory (n={len(pos_sims):,})", density=True)
    ax.axvline(x=tau, color="black", linestyle="--", linewidth=2, label=f"Threshold $\\tau$={tau:.3f}")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    model_short = short(best["model"])
    ax.set_title(f"Similarity Distribution (Train) — {model_short} L{layer}, SIF+ABTT(D={D})", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle="--")

    fig.tight_layout()
    fig.savefig(out / "fig5_threshold_histogram.png", dpi=dpi)
    fig.savefig(out / "fig5_threshold_histogram.pdf", dpi=dpi)
    plt.close(fig)
    print(f"  Fig 5 saved")


# ── Figure 6: Acc@1 vs Layer ─────────────────────────────────────────────
def fig6_acc_at_1(df: pd.DataFrame, out: Path, dpi: int):
    """Acc@1 vs layer for hidden states."""
    hidden = df[df["repr"] == "hidden"]
    models = list(hidden["model"].unique())
    methods = ["baseline", "sif_only", "sif_abtt_optimal"]

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = hidden[hidden["model"] == model]
        for method in methods:
            sub = mdf[mdf["method"] == method].sort_values("layer")
            if sub.empty:
                continue
            ax.plot(
                sub["layer"], sub["acc_at_1"],
                marker="o", markersize=4, linewidth=1.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_xlabel("Layer")
        ax.set_title(short(model), fontweight="bold")
        ax.set_xticks(range(0, 13))
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_ylim(0, 1.02)

    axes[0].set_ylabel("Acc@1 (test queries)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Retrieval Accuracy (Acc@1) vs. Layer Depth", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out / "fig6_acc_at_1.png", dpi=dpi)
    fig.savefig(out / "fig6_acc_at_1.pdf", dpi=dpi)
    plt.close(fig)
    print(f"  Fig 6 saved")


# ── Figure 7: Optimal D distribution ─────────────────────────────────────
def fig7_optimal_d(df: pd.DataFrame, out: Path, dpi: int):
    """What D was chosen per model and layer."""
    opt = df[(df["method"] == "sif_abtt_optimal") & (df["repr"] == "hidden")]
    if opt.empty:
        return

    models = list(opt["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = opt[opt["model"] == model].sort_values("layer")
        ax.bar(sub["layer"], sub["D"], color="#377eb8", alpha=0.8, edgecolor="white")
        for _, row in sub.iterrows():
            ax.text(row["layer"], row["D"] + 0.15, str(int(row["D"])),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_title(short(model), fontweight="bold")
        ax.set_xticks(range(0, 13))
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")

    axes[0].set_ylabel("Optimal D (PCs removed)")
    fig.suptitle("Optimal Number of Principal Components Removed per Layer", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig7_optimal_d.png", dpi=dpi)
    fig.savefig(out / "fig7_optimal_d.pdf", dpi=dpi)
    plt.close(fig)
    print(f"  Fig 7 saved")


# ── Figure 8: Summary table ──────────────────────────────────────────────
def fig8_summary_table(df: pd.DataFrame, out: Path, dpi: int):
    """Best results per model as a rendered table figure."""
    hidden = df[df["repr"] == "hidden"]
    rows = []
    for model in hidden["model"].unique():
        for method in METHOD_ORDER:
            sub = hidden[(hidden["model"] == model) & (hidden["method"] == method)]
            if sub.empty:
                continue
            best = sub.loc[sub["aucroc"].idxmax()]
            rows.append({
                "Model": short(model),
                "Method": METHOD_LABELS.get(method, method),
                "Layer": f"L{int(best['layer'])}",
                "D": int(best["D"]) if method.startswith("sif_abtt") else "-",
                "AUCROC": f"{best['aucroc']:.4f}",
                "Spearman": f"{best['spearman']:.4f}",
                "Gap": f"{best['gap']:.3f}",
                "Acc@1": f"{best['acc_at_1']:.3f}",
                "Assign": f"{best['overall_assignment_acc']:.3f}",
                "tau": f"{best['tau']:.3f}",
            })

    tdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 0.4 * len(rows) + 1.2))
    ax.axis("off")

    table = ax.table(
        cellText=tdf.values,
        colLabels=tdf.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Header styling
    for j in range(len(tdf.columns)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(rows)):
        color = "#f7f7f7" if i % 2 == 0 else "white"
        for j in range(len(tdf.columns)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title("Best Results per Model and Method (Hidden States)", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out / "fig8_summary_table.png", dpi=dpi)
    fig.savefig(out / "fig8_summary_table.pdf", dpi=dpi)
    plt.close(fig)
    print(f"  Fig 8 saved")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} results from {args.results_csv}")
    print(f"Models: {list(df['model'].unique())}")
    print()

    print("Generating figures...")
    fig1_dip_graph(df, out, args.dpi)
    fig1b_ff1_dip(df, out, args.dpi)
    fig2_method_heatmap(df, out, args.dpi)
    fig3_cosine_gap(df, out, args.dpi)
    fig4_assignment_accuracy(df, out, args.dpi)
    fig5_threshold_histogram(df, args.split_csv, args.emb_dir, out, args.dpi)
    fig6_acc_at_1(df, out, args.dpi)
    fig7_optimal_d(df, out, args.dpi)
    fig8_summary_table(df, out, args.dpi)

    print(f"\nAll figures saved to {out}")


if __name__ == "__main__":
    main()
