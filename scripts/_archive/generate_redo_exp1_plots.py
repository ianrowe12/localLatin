"""Generate all plots for Experiment 1 (Redo): Canon Embedding Analysis."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS = Path("runs/redo_results")
CSV = RESULTS / "experiment1_canon.csv"
PLOT_DIR = RESULTS / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SHORT = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
}
MODEL_COLORS = {
    "LaTa": "#1f77b4",
    "PhilTa": "#ff7f0e",
    "LaBSE": "#2ca02c",
}
METHOD_STYLES = {
    "baseline_mean": {"ls": "-", "marker": "o", "label": "Baseline Mean"},
    "sif_abtt_optimal": {"ls": "-", "marker": "s", "label": "SIF + ABTT"},
    "whitening": {"ls": "--", "marker": "^", "label": "Whitening"},
}


def load():
    df = pd.read_csv(CSV)
    df["model_short"] = df["model"].map(MODEL_SHORT)
    return df


# ---------------------------------------------------------------------------
# Figure 1: Hidden-state baseline AUC – the U-shape
# ---------------------------------------------------------------------------
def fig01_hidden_baseline_ushape(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    base = df[(df["method"] == "baseline_mean") & (df["repr"].isin(["hidden"]))]
    for model, grp in base.groupby("model_short"):
        grp = grp.sort_values("layer")
        ax.plot(grp["layer"], grp["auc_roc_test"],
                color=MODEL_COLORS[model], marker="o", linewidth=2, label=model)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="Random (0.5)")
    ax.set_xlabel("Hidden-State Layer (0 = embedding output)", fontsize=12)
    ax.set_ylabel("AUC-ROC (test)", fontsize=12)
    ax.set_title("The Anisotropy Dip: Baseline Mean Pooling on Hidden States", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    # Annotate the dip
    ax.annotate("Anisotropy\nDip", xy=(6, 0.47), fontsize=10,
                ha="center", color="#d62728", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.8))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp1_01_hidden_baseline_uShape.png", dpi=150)
    plt.close(fig)
    print("  fig_exp1_01")


# ---------------------------------------------------------------------------
# Figure 2: Baseline vs SIF+ABTT (hidden states) – the rescue
# ---------------------------------------------------------------------------
def fig02_hidden_sif_rescue(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    hidden = df[df["repr"] == "hidden"]
    for ax, model in zip(axes, ["LaTa", "PhilTa", "LaBSE"]):
        sub = hidden[hidden["model_short"] == model]
        for method in ["baseline_mean", "sif_abtt_optimal"]:
            m = sub[sub["method"] == method].sort_values("layer")
            st = METHOD_STYLES[method]
            ax.plot(m["layer"], m["auc_roc_test"],
                    ls=st["ls"], marker=st["marker"], linewidth=2, label=st["label"],
                    color=MODEL_COLORS[model] if method == "baseline_mean" else "#d62728")
        ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylim(0.4, 1.02)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("AUC-ROC (test)", fontsize=12)
    fig.suptitle("SIF + ABTT Rescues All Layers (Hidden States)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp1_02_hidden_sif_rescue.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_exp1_02")


# ---------------------------------------------------------------------------
# Figure 3: All 3 methods compared (hidden states)
# ---------------------------------------------------------------------------
def fig03_all_methods(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    hidden = df[df["repr"] == "hidden"]
    colors = {
        "baseline_mean": "#1f77b4",
        "sif_abtt_optimal": "#d62728",
        "whitening": "#9467bd",
    }
    for ax, model in zip(axes, ["LaTa", "PhilTa", "LaBSE"]):
        sub = hidden[hidden["model_short"] == model]
        for method in ["baseline_mean", "sif_abtt_optimal", "whitening"]:
            m = sub[sub["method"] == method].sort_values("layer")
            st = METHOD_STYLES[method]
            ax.plot(m["layer"], m["auc_roc_test"],
                    ls=st["ls"], marker=st["marker"], linewidth=2,
                    label=st["label"], color=colors[method], markersize=5)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylim(0.35, 1.02)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("AUC-ROC (test)", fontsize=12)
    fig.suptitle("Three Methods Compared: Hidden States", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp1_03_all_methods_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_exp1_03")


# ---------------------------------------------------------------------------
# Figure 4: FF1 / FFN intermediate layer-wise AUC
# ---------------------------------------------------------------------------
def fig04_ff1_ffn(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    ff = df[df["repr"].isin(["ff1", "ffn_intermediate"])]
    colors = {
        "baseline_mean": "#1f77b4",
        "sif_abtt_optimal": "#d62728",
        "whitening": "#9467bd",
    }
    for ax, model in zip(axes, ["LaTa", "PhilTa", "LaBSE"]):
        sub = ff[ff["model_short"] == model]
        repr_name = sub["repr"].iloc[0] if len(sub) > 0 else "ff1"
        for method in ["baseline_mean", "sif_abtt_optimal", "whitening"]:
            m = sub[sub["method"] == method].sort_values("layer")
            st = METHOD_STYLES[method]
            ax.plot(m["layer"], m["auc_roc_test"],
                    ls=st["ls"], marker=st["marker"], linewidth=2,
                    label=st["label"], color=colors[method], markersize=5)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
        title_repr = "FF1" if repr_name == "ff1" else "FFN Intermediate"
        ax.set_title(f"{model} ({title_repr})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylim(0.35, 1.02)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("AUC-ROC (test)", fontsize=12)
    fig.suptitle("FF1 / FFN Intermediate Layers", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp1_04_ff1_ffn_layers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_exp1_04")


# ---------------------------------------------------------------------------
# Figure 5: Optimal D heatmap
# ---------------------------------------------------------------------------
def fig05_optimal_D(df):
    sif = df[df["method"] == "sif_abtt_optimal"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    combos = [
        ("LaTa", "hidden"), ("PhilTa", "hidden"), ("LaBSE", "hidden"),
        ("LaTa", "ff1"), ("PhilTa", "ff1"), ("LaBSE", "ffn_intermediate"),
    ]
    for ax, (model, repr_name) in zip(axes.flat, combos):
        sub = sif[(sif["model_short"] == model) & (sif["repr"] == repr_name)].sort_values("layer")
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        layers = sub["layer"].values
        d_vals = sub["D"].values
        # Single-row heatmap
        data = d_vals.reshape(1, -1)
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=10)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8)
        ax.set_yticks([])
        for i, d in enumerate(d_vals):
            ax.text(i, 0, str(int(d)), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if d > 6 else "black")
        title_repr = "FF1" if repr_name == "ff1" else ("FFN Int." if repr_name == "ffn_intermediate" else "Hidden")
        ax.set_title(f"{model} ({title_repr})", fontsize=11)
        ax.set_xlabel("Layer", fontsize=10)

    fig.suptitle("Optimal D (PCs Removed) by Model and Layer", fontsize=14, y=1.0)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="D", shrink=0.6, pad=0.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp1_05_optimal_D_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_exp1_05")


# ---------------------------------------------------------------------------
# Figure 6: Best-config retrieval accuracy bar chart
# ---------------------------------------------------------------------------
def fig06_best_config_retrieval(df):
    sif = df[df["method"] == "sif_abtt_optimal"]
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["LaTa", "PhilTa", "LaBSE"]
    ks = ["acc_at_1_test", "acc_at_3_test", "acc_at_5_test"]
    k_labels = ["Acc@1", "Acc@3", "Acc@5"]
    x = np.arange(len(models))
    width = 0.25

    for i, (k_col, k_label) in enumerate(zip(ks, k_labels)):
        vals = []
        for model in models:
            sub = sif[sif["model_short"] == model]
            best_idx = sub["auc_roc_test"].idxmax()
            vals.append(sub.loc[best_idx, k_col])
        ax.bar(x + i * width, vals, width, label=k_label, alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Retrieval Accuracy", fontsize=12)
    ax.set_title("Best-Config Retrieval Accuracy (SIF + ABTT)", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0.85, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (k_col, _) in enumerate(zip(ks, k_labels)):
        for j, model in enumerate(models):
            sub = sif[sif["model_short"] == model]
            best_idx = sub["auc_roc_test"].idxmax()
            v = sub.loc[best_idx, k_col]
            ax.text(j + i * width, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp1_06_best_config_retrieval.png", dpi=150)
    plt.close(fig)
    print("  fig_exp1_06")


# ---------------------------------------------------------------------------
# Figure 7: Method improvement over baseline
# ---------------------------------------------------------------------------
def fig07_method_improvement(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    models = ["LaTa", "PhilTa", "LaBSE"]
    reprs_per_model = {
        "LaTa": ["hidden", "ff1"],
        "PhilTa": ["hidden", "ff1"],
        "LaBSE": ["hidden", "ffn_intermediate"],
    }

    rows = []
    for model in models:
        for repr_name in reprs_per_model[model]:
            base_sub = df[(df["model_short"] == model) & (df["repr"] == repr_name) & (df["method"] == "baseline_mean")]
            base_best = base_sub["auc_roc_test"].max() if len(base_sub) > 0 else 0

            for method in ["sif_abtt_optimal", "whitening"]:
                m_sub = df[(df["model_short"] == model) & (df["repr"] == repr_name) & (df["method"] == method)]
                m_best = m_sub["auc_roc_test"].max() if len(m_sub) > 0 else 0
                repr_label = "FF1" if repr_name == "ff1" else ("FFN Int." if repr_name == "ffn_intermediate" else "Hidden")
                rows.append({
                    "config": f"{model}\n{repr_label}",
                    "method": METHOD_STYLES[method]["label"],
                    "improvement": m_best - base_best,
                })

    plot_df = pd.DataFrame(rows)
    configs = plot_df["config"].unique()
    x = np.arange(len(configs))
    width = 0.35

    for i, method_label in enumerate(["SIF + ABTT", "Whitening"]):
        vals = plot_df[plot_df["method"] == method_label]["improvement"].values
        color = "#d62728" if method_label == "SIF + ABTT" else "#9467bd"
        bars = ax.bar(x + i * width, vals, width, label=method_label, color=color, alpha=0.85)

    ax.set_xlabel("Model / Representation", fontsize=11)
    ax.set_ylabel("AUC Improvement over Baseline", fontsize=11)
    ax.set_title("Improvement over Baseline Mean (Best Layer Each)", fontsize=13)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(configs, fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp1_07_method_improvement_bars.png", dpi=150)
    plt.close(fig)
    print("  fig_exp1_07")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load()
    print(f"  {len(df)} rows loaded")
    print("Generating plots...")
    fig01_hidden_baseline_ushape(df)
    fig02_hidden_sif_rescue(df)
    fig03_all_methods(df)
    fig04_ff1_ffn(df)
    fig05_optimal_D(df)
    fig06_best_config_retrieval(df)
    fig07_method_improvement(df)
    print(f"Done. Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
