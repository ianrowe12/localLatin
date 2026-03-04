"""Phase 11 Visualization: 6 figures + 2 tables for the paper.

Figures:
  1. Cosine similarity distribution (histogram, 1x5 subplots) — baseline last layer
  2. Layer-wise performance with normalized x-axis (1x2 panels)
  3. Baselines vs middle layers (grouped bar, 1x2 panels)
  4. SIF+ABTT method comparison (grouped bar, 1x2 panels)
  5. 4-condition density histograms (4x5 grid):
     baseline last, baseline middle, SIF+ABTT last, SIF+ABTT middle
  6. Per-model layer-wise cosine gap — all methods compared (1x5 subplots)

Tables:
  A. Task A — pairwise classification (best AUCROC)
  B. Task B — directory assignment (best Dir Acc@1/3)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

SHORT = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
}
COLORS = {
    "LaTa": "#1f77b4",
    "PhilTa": "#ff7f0e",
    "LaBSE": "#2ca02c",
    "Qwen3-0.6B": "#d62728",
    "KaLM-mini": "#9467bd",
}
METHOD_COLORS = {
    "baseline": "#bdbdbd",
    "sif_only": "#fdae6b",
    "abtt_optimal": "#756bb1",
    "sif_abtt_fixed": "#e6550d",
    "sif_abtt_optimal": "#d62728",
    "whitening": "#6baed6",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "sif_only": "SIF only",
    "abtt_optimal": "ABTT only (opt D)",
    "sif_abtt_fixed": "SIF+ABTT (D=10)",
    "sif_abtt_optimal": "SIF+ABTT (opt D)",
    "whitening": "Whitening",
}

MAX_LAYERS = {
    "bowphs/LaTa": 12,
    "bowphs/PhilTa": 12,
    "sentence-transformers/LaBSE": 12,
    "Qwen/Qwen3-Embedding-0.6B": 28,
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": 24,
}

MODELS = list(SHORT.keys())


def sn(model: str) -> str:
    return SHORT.get(model, model)


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def _save(fig, out: Path, name: str, dpi: int):
    fig.savefig(out / f"{name}.png", dpi=dpi)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Cosine Similarity Distribution
# ═══════════════════════════════════════════════════════════════════════

def fig1_distributions(dist_dir: Path, out: Path, dpi: int):
    """1x5 histograms of same vs different directory cosine similarities."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for ax, m in zip(axes, MODELS):
        slug = model_slug(m)
        npz_path = dist_dir / f"{slug}_baseline_distributions.npz"
        if not npz_path.exists():
            ax.set_title(sn(m), fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            continue

        data = np.load(npz_path)
        same = data["same_sims"]
        diff = data["diff_sims"]

        bins = np.linspace(-0.2, 1.0, 80)
        ax.hist(diff, bins=bins, alpha=0.5, color="#d62728", label="Diff Dir",
                density=True, edgecolor="none")
        ax.hist(same, bins=bins, alpha=0.5, color="#1f77b4", label="Same Dir",
                density=True, edgecolor="none")

        # Means
        same_mean = np.mean(same)
        diff_mean = np.mean(diff)
        ax.axvline(same_mean, color="#1f77b4", ls="--", lw=1.5)
        ax.axvline(diff_mean, color="#d62728", ls="--", lw=1.5)
        ax.text(same_mean, ax.get_ylim()[1] * 0.9,
                f"{same_mean:.2f}", color="#1f77b4", fontsize=8, ha="center")
        ax.text(diff_mean, ax.get_ylim()[1] * 0.8,
                f"{diff_mean:.2f}", color="#d62728", fontsize=8, ha="center")

        ax.set_title(sn(m), fontweight="bold")
        ax.set_xlabel("Cosine Similarity")
        ax.legend(fontsize=7, loc="upper left")

    axes[0].set_ylabel("Density")
    fig.suptitle("Fig 1: Cosine Similarity Distributions — Baseline (Hidden, Last Layer)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig1_distributions", dpi)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Layer-wise Performance (Normalized X-axis)
# ═══════════════════════════════════════════════════════════════════════

def fig2_layerwise(df: pd.DataFrame, out: Path, dpi: int):
    """1x2 panels: (A) AUCROC, (B) Dir Acc@1 — normalized layer percentage."""
    sub = df[(df["method"] == "sif_abtt_optimal") & (df["repr"] == "hidden")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = [("aucroc", "AUCROC"), ("dir_acc_at_1", "Dir Acc@1")]
    panel_labels = ["(A)", "(B)"]

    for ax, (metric, ylabel), panel in zip(axes, metrics, panel_labels):
        for m in MODELS:
            msub = sub[sub["model"] == m].sort_values("layer").copy()
            if msub.empty:
                continue
            max_l = MAX_LAYERS.get(m, msub["layer"].max())
            msub["layer_pct"] = msub["layer"] / max_l * 100
            ax.plot(msub["layer_pct"], msub[metric],
                    marker="o", ms=3, lw=2,
                    label=sn(m), color=COLORS[sn(m)])

        ax.set_xlabel("Layer Percentage (%)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 105)
        if metric == "aucroc":
            ax.set_ylim(0.4, 1.02)
        else:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title(f"{panel} {ylabel}", fontweight="bold")

    fig.suptitle("Fig 2: Layer-wise Performance — SIF+ABTT (opt D), Hidden",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig2_layerwise", dpi)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Baselines vs Middle Layers
# ═══════════════════════════════════════════════════════════════════════

def fig3_baseline_vs_middle(df: pd.DataFrame, out: Path, dpi: int):
    """Grouped bar: Last Layer vs Best Middle Layer (baseline, hidden, mean)."""
    sub = df[(df["method"] == "baseline") & (df["repr"] == "hidden") & (df["pooling"] == "mean")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = [("aucroc", "AUCROC"), ("dir_acc_at_1", "Dir Acc@1")]
    panel_labels = ["(A)", "(B)"]

    for ax, (metric, ylabel), panel in zip(axes, metrics, panel_labels):
        last_vals = []
        middle_vals = []
        last_layers_list = []
        middle_layers_list = []

        for m in MODELS:
            msub = sub[sub["model"] == m].copy()
            if msub.empty:
                last_vals.append(0)
                middle_vals.append(0)
                last_layers_list.append("")
                middle_layers_list.append("")
                continue

            max_l = MAX_LAYERS.get(m, msub["layer"].max())

            # Last layer
            last_row = msub.loc[msub["layer"].idxmax()]
            last_vals.append(last_row[metric])
            last_layers_list.append(f"L{int(last_row['layer'])}")

            # Best middle layer (30-70% range)
            msub["pct"] = msub["layer"] / max_l * 100
            middle = msub[(msub["pct"] >= 30) & (msub["pct"] <= 70)]
            if middle.empty:
                middle_vals.append(0)
                middle_layers_list.append("")
            else:
                best_mid = middle.loc[middle[metric].idxmax()]
                middle_vals.append(best_mid[metric])
                middle_layers_list.append(f"L{int(best_mid['layer'])}")

        x = np.arange(len(MODELS))
        w = 0.35
        bars1 = ax.bar(x - w / 2, last_vals, w, label="Last Layer", color="#2166ac")
        bars2 = ax.bar(x + w / 2, middle_vals, w, label="Best Middle Layer", color="#b2182b")

        # Annotate layer numbers
        for i, (bar, lbl) in enumerate(zip(bars1, last_layers_list)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    lbl, ha="center", va="bottom", fontsize=7)
        for i, (bar, lbl) in enumerate(zip(bars2, middle_layers_list)):
            if lbl:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        lbl, ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([sn(m) for m in MODELS], fontsize=9)
        ax.set_ylabel(ylabel)
        if metric != "aucroc":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
        ax.legend(fontsize=8)
        ax.set_title(f"{panel} {ylabel}", fontweight="bold")

    fig.suptitle("Fig 3: Last Layer vs Best Middle Layer — Baseline (Hidden, Mean)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig3_baseline_vs_middle", dpi)


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: SIF+ABTT Method Comparison
# ═══════════════════════════════════════════════════════════════════════

def fig4_method_comparison(df: pd.DataFrame, out: Path, dpi: int):
    """Grouped bar: 4 bars per model comparing methods/layers."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    metrics = [("aucroc", "AUCROC"), ("dir_acc_at_1", "Dir Acc@1")]
    panel_labels = ["(A)", "(B)"]

    bar_configs = [
        ("SIF+ABTT Last", "sif_abtt_optimal", "last", "#d62728"),
        ("SIF+ABTT Middle", "sif_abtt_optimal", "middle", "#ff7f0e"),
        ("Baseline Last", "baseline", "last", "#bdbdbd"),
        ("Whitening Last", "whitening", "last", "#6baed6"),
    ]

    for ax, (metric, ylabel), panel in zip(axes, metrics, panel_labels):
        x = np.arange(len(MODELS))
        n_bars = len(bar_configs)
        w = 0.8 / n_bars

        for j, (label, method, layer_sel, color) in enumerate(bar_configs):
            vals = []
            for m in MODELS:
                msub = df[(df["model"] == m) & (df["method"] == method)
                          & (df["repr"] == "hidden")]
                if msub.empty:
                    vals.append(0)
                    continue

                max_l = MAX_LAYERS.get(m, msub["layer"].max())
                if layer_sel == "last":
                    row = msub.loc[msub["layer"].idxmax()]
                    vals.append(row[metric])
                else:  # middle
                    msub = msub.copy()
                    msub["pct"] = msub["layer"] / max_l * 100
                    middle = msub[(msub["pct"] >= 30) & (msub["pct"] <= 70)]
                    if middle.empty:
                        vals.append(0)
                    else:
                        vals.append(middle[metric].max())

            offset = (j - n_bars / 2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=label, color=color,
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([sn(m) for m in MODELS], fontsize=9)
        ax.set_ylabel(ylabel)
        if metric != "aucroc":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{panel} {ylabel}", fontweight="bold")

    fig.suptitle("Fig 4: SIF+ABTT vs Baseline vs Whitening — Method Comparison",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig4_method_comparison", dpi)


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: 4-Condition Density Histograms (4×5 grid)
# ═══════════════════════════════════════════════════════════════════════

_COND_SUFFIXES = [
    ("baseline_last",   "Baseline — Last Layer"),
    ("baseline_middle", "Baseline — Best Middle Layer"),
    ("sif_abtt_last",   "SIF+ABTT — Last Layer"),
    ("sif_abtt_middle", "SIF+ABTT — Best Middle Layer"),
]


def fig5_density_grid(dist_dir: Path, out: Path, dpi: int):
    """4×5 grid: rows = conditions, columns = models."""
    n_cond = len(_COND_SUFFIXES)
    n_models = len(MODELS)
    fig, axes = plt.subplots(n_cond, n_models, figsize=(20, 14),
                             sharex=True, sharey="row")

    bins = np.linspace(-0.2, 1.0, 80)

    for row, (suffix, row_label) in enumerate(_COND_SUFFIXES):
        for col, m in enumerate(MODELS):
            ax = axes[row, col]
            slug = model_slug(m)
            npz_path = dist_dir / f"{slug}_{suffix}.npz"

            if not npz_path.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            else:
                data = np.load(npz_path)
                same = data["same_sims"]
                diff = data["diff_sims"]

                ax.hist(diff, bins=bins, alpha=0.5, color="#d62728",
                        label="Diff Dir", density=True, edgecolor="none")
                ax.hist(same, bins=bins, alpha=0.5, color="#1f77b4",
                        label="Same Dir", density=True, edgecolor="none")

                # Mean lines
                same_mean = np.mean(same)
                diff_mean = np.mean(diff)
                ax.axvline(same_mean, color="#1f77b4", ls="--", lw=1.2)
                ax.axvline(diff_mean, color="#d62728", ls="--", lw=1.2)
                gap = same_mean - diff_mean
                ax.text(0.97, 0.95, f"gap={gap:.2f}",
                        transform=ax.transAxes, fontsize=7,
                        ha="right", va="top",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

            # Labels
            if row == 0:
                ax.set_title(sn(m), fontweight="bold", fontsize=11)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight="bold")
            if row == n_cond - 1:
                ax.set_xlabel("Cosine Similarity", fontsize=9)
            if row == 0 and col == n_models - 1:
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Fig 5: Cosine Similarity Distributions — 4 Conditions Compared",
                 y=1.01, fontweight="bold", fontsize=14)
    fig.tight_layout()
    _save(fig, out, "fig5_density_grid", dpi)


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: Per-Model Layer-wise Cosine Gap — All Methods
# ═══════════════════════════════════════════════════════════════════════

_GAP_METHODS = [
    ("baseline",         "mean", "--",  1.5, "#bdbdbd", "Baseline"),
    ("sif_only",         "sif",  "-.",  1.5, "#fdae6b", "SIF only"),
    ("abtt_optimal",     "mean", "-.",  2.0, "#756bb1", "ABTT only (opt D)"),
    ("sif_abtt_fixed",   "sif",  ":",   1.8, "#e6550d", "SIF+ABTT (D=10)"),
    ("sif_abtt_optimal", "sif",  "-",   2.2, "#d62728", "SIF+ABTT (opt D)"),
    ("whitening",        "mean", "-",   1.5, "#6baed6", "Whitening"),
]


def fig6_gap_per_model(df: pd.DataFrame, out: Path, dpi: int):
    """1×5 subplots: cosine gap across layers for each model, all methods."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    for ax, m in zip(axes, MODELS):
        max_l = MAX_LAYERS.get(m, 12)
        for method, pooling, ls, lw, color, label in _GAP_METHODS:
            msub = df[(df["model"] == m) & (df["method"] == method)
                      & (df["repr"] == "hidden") & (df["pooling"] == pooling)]
            if msub.empty:
                continue
            msub = msub.sort_values("layer").copy()
            msub["layer_pct"] = msub["layer"] / max_l * 100
            ax.plot(msub["layer_pct"], msub["gap"],
                    ls=ls, lw=lw, color=color,
                    marker="o", ms=3, label=label)

        ax.set_title(sn(m), fontweight="bold")
        ax.set_xlabel("Layer (%)")
        ax.set_xlim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best")

    axes[0].set_ylabel("Cosine Gap (same - diff)")
    fig.suptitle("Fig 6: Layer-wise Cosine Gap — All Methods Compared (Hidden)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig6_gap_per_model", dpi)


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Per-Model Layer-wise AUCROC — All Methods
# ═══════════════════════════════════════════════════════════════════════

def fig7_aucroc_per_model(df: pd.DataFrame, out: Path, dpi: int):
    """1×5 subplots: AUCROC across layers for each model, all methods."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    for ax, m in zip(axes, MODELS):
        max_l = MAX_LAYERS.get(m, 12)
        for method, pooling, ls, lw, color, label in _GAP_METHODS:
            msub = df[(df["model"] == m) & (df["method"] == method)
                      & (df["repr"] == "hidden") & (df["pooling"] == pooling)]
            if msub.empty:
                continue
            msub = msub.sort_values("layer").copy()
            msub["layer_pct"] = msub["layer"] / max_l * 100
            ax.plot(msub["layer_pct"], msub["aucroc"],
                    ls=ls, lw=lw, color=color,
                    marker="o", ms=3, label=label)

        ax.set_title(sn(m), fontweight="bold")
        ax.set_xlabel("Layer (%)")
        ax.set_xlim(0, 105)
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best")

    axes[0].set_ylabel("AUCROC")
    fig.suptitle("Fig 7: Layer-wise AUCROC — All Methods Compared (Hidden)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig7_aucroc_per_model", dpi)


# ═══════════════════════════════════════════════════════════════════════
# Table A: Task A (Pairwise Classification)
# ═══════════════════════════════════════════════════════════════════════

def table_a_task_a(df: pd.DataFrame, out: Path, dpi: int):
    """Select best layer per (model, method, repr) by train_aucroc, report test aucroc."""
    methods = ["baseline", "sif_only", "sif_abtt_optimal", "whitening"]
    rows = []
    for m in MODELS:
        for method in methods:
            msub = df[(df["model"] == m) & (df["method"] == method)]
            if msub.empty:
                continue
            for repr_name in msub["repr"].unique():
                rsub = msub[msub["repr"] == repr_name]
                if rsub.empty:
                    continue
                best_idx = rsub["train_aucroc"].idxmax()
                best = rsub.loc[best_idx]
                rows.append({
                    "Model": sn(m),
                    "Method": METHOD_LABELS.get(method, method),
                    "Repr": repr_name,
                    "Best Layer": int(best["layer"]),
                    "AUCROC": f"{best['aucroc']:.4f}",
                })

    table_df = pd.DataFrame(rows)

    # Save CSV
    table_df.to_csv(out / "table_a_task_a.csv", index=False)

    # Matplotlib table figure
    fig, ax = plt.subplots(figsize=(12, max(3, 0.4 * len(rows) + 1.5)))
    ax.axis("off")
    col_labels = list(table_df.columns)
    cell_text = table_df.values.tolist()
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2166ac")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best AUCROC per model
    aucroc_vals = [float(r["AUCROC"]) for r in rows]
    model_names = [r["Model"] for r in rows]
    for model_short in set(model_names):
        idxs = [i for i, mn in enumerate(model_names) if mn == model_short]
        best_i = max(idxs, key=lambda i: aucroc_vals[i])
        table[best_i + 1, 4].set_facecolor("#d4edda")

    ax.set_title("Table A: Task A — Pairwise Classification (AUCROC)",
                 fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, out, "table_a_task_a", dpi)
    print(f"  Saved table_a_task_a.csv")


# ═══════════════════════════════════════════════════════════════════════
# Table B: Task B (Directory Assignment)
# ═══════════════════════════════════════════════════════════════════════

def table_b_task_b(df: pd.DataFrame, out: Path, dpi: int):
    """Select best layer per (model, method, repr) by train_dir_acc_at_1."""
    methods = ["baseline", "sif_only", "sif_abtt_optimal", "whitening"]
    rows = []
    for m in MODELS:
        for method in methods:
            msub = df[(df["model"] == m) & (df["method"] == method)]
            if msub.empty:
                continue
            for repr_name in msub["repr"].unique():
                rsub = msub[msub["repr"] == repr_name]
                if rsub.empty:
                    continue
                best_idx = rsub["train_dir_acc_at_1"].idxmax()
                best = rsub.loc[best_idx]
                rows.append({
                    "Model": sn(m),
                    "Method": METHOD_LABELS.get(method, method),
                    "Repr": repr_name,
                    "Best Layer": int(best["layer"]),
                    "Dir Acc@1": f"{best['dir_acc_at_1']:.4f}",
                    "Dir Acc@3": f"{best['dir_acc_at_3']:.4f}",
                })

    table_df = pd.DataFrame(rows)

    # Save CSV
    table_df.to_csv(out / "table_b_task_b.csv", index=False)

    # Matplotlib table figure
    fig, ax = plt.subplots(figsize=(14, max(3, 0.4 * len(rows) + 1.5)))
    ax.axis("off")
    col_labels = list(table_df.columns)
    cell_text = table_df.values.tolist()
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2166ac")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best Dir Acc@1 per model
    acc_vals = [float(r["Dir Acc@1"]) for r in rows]
    model_names = [r["Model"] for r in rows]
    for model_short in set(model_names):
        idxs = [i for i, mn in enumerate(model_names) if mn == model_short]
        best_i = max(idxs, key=lambda i: acc_vals[i])
        table[best_i + 1, 4].set_facecolor("#d4edda")

    ax.set_title("Table B: Task B — Directory Assignment (Dir Acc@k)",
                 fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, out, "table_b_task_b", dpi)
    print(f"  Saved table_b_task_b.csv")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 11 visualization.")
    parser.add_argument("--results_csv", required=True, help="Phase 11 results CSV.")
    parser.add_argument("--dist_dir", default=None, help="Directory with distribution .npz files.")
    parser.add_argument("--out_dir", default="runs/phase11/figures", help="Output directory.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results_csv)

    print("Generating Phase 11 figures...")

    dist_dir = Path(args.dist_dir) if args.dist_dir else None

    if dist_dir and dist_dir.exists():
        fig1_distributions(dist_dir, out, args.dpi)
        fig5_density_grid(dist_dir, out, args.dpi)
    else:
        print(f"  Distribution dir not found, skipping Fig 1 & 5.")

    fig2_layerwise(df, out, args.dpi)
    fig3_baseline_vs_middle(df, out, args.dpi)
    fig4_method_comparison(df, out, args.dpi)
    fig6_gap_per_model(df, out, args.dpi)
    fig7_aucroc_per_model(df, out, args.dpi)

    print("\nGenerating tables...")
    table_a_task_a(df, out, args.dpi)
    table_b_task_b(df, out, args.dpi)

    print(f"\nAll figures and tables saved to: {out}")


if __name__ == "__main__":
    main()
