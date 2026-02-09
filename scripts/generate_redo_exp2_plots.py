"""Generate all plots for Experiment 2 (Redo): MUSTS Multilingual Analysis."""
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
CSV = RESULTS / "experiment2_musts.csv"
PLOT_DIR = RESULTS / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SHORT = {
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM",
    "sentence-transformers/LaBSE": "LaBSE",
}
MODEL_COLORS = {
    "Qwen3-0.6B": "#1f77b4",
    "KaLM": "#ff7f0e",
    "LaBSE": "#2ca02c",
}
METHOD_STYLES = {
    "baseline_mean": {"ls": "-", "marker": "o", "label": "Baseline Mean"},
    "sif_abtt_optimal": {"ls": "-", "marker": "s", "label": "SIF + ABTT"},
    "whitening": {"ls": "--", "marker": "^", "label": "Whitening"},
}
METHOD_COLORS = {
    "baseline_mean": "#1f77b4",
    "sif_abtt_optimal": "#d62728",
    "whitening": "#9467bd",
}
LANGUAGES = ["english", "french", "sinhala", "tamil"]
LANG_TITLES = {"english": "English", "french": "French", "sinhala": "Sinhala", "tamil": "Tamil"}


def load():
    df = pd.read_csv(CSV)
    df["model_short"] = df["model"].map(MODEL_SHORT)
    return df


# ---------------------------------------------------------------------------
# Figures 1-4: Layer-wise Spearman per language
# ---------------------------------------------------------------------------
def fig_layer_curves(df, lang, fig_num):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    lang_df = df[df["language"] == lang]

    for ax, model in zip(axes, ["Qwen3-0.6B", "KaLM", "LaBSE"]):
        sub = lang_df[lang_df["model_short"] == model]
        for method in ["baseline_mean", "sif_abtt_optimal", "whitening"]:
            m = sub[sub["method"] == method].sort_values("layer")
            st = METHOD_STYLES[method]
            ax.plot(m["layer"], m["spearman_test"],
                    ls=st["ls"], marker=st["marker"], linewidth=2,
                    label=st["label"], color=METHOD_COLORS[method], markersize=4)
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    y_min = max(0, lang_df["spearman_test"].min() - 0.05)
    y_max = min(1.0, lang_df["spearman_test"].max() + 0.05)
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel("Spearman (test)", fontsize=12)
    fig.suptitle(f"Layer-wise Spearman: {LANG_TITLES[lang]}", fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f"fig_exp2_{fig_num:02d}_{lang}_layer_curves.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_exp2_{fig_num:02d}")


# ---------------------------------------------------------------------------
# Figure 5: SIF improvement heatmap
# ---------------------------------------------------------------------------
def fig05_sif_improvement(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["Qwen3-0.6B", "KaLM", "LaBSE"]
    langs = LANGUAGES

    data = np.zeros((len(models), len(langs)))
    annot = np.empty((len(models), len(langs)), dtype=object)

    for i, model in enumerate(models):
        for j, lang in enumerate(langs):
            base = df[(df["model_short"] == model) & (df["language"] == lang) & (df["method"] == "baseline_mean")]
            sif = df[(df["model_short"] == model) & (df["language"] == lang) & (df["method"] == "sif_abtt_optimal")]
            b_best = base["spearman_test"].max() if len(base) > 0 else 0
            s_best = sif["spearman_test"].max() if len(sif) > 0 else 0
            diff = s_best - b_best
            data[i, j] = diff
            annot[i, j] = f"{diff:+.3f}"

    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.07, vmax=0.07, aspect="auto")
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([LANG_TITLES[l] for l in langs], fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(langs)):
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if abs(data[i, j]) > 0.04 else "black")

    fig.colorbar(im, ax=ax, label="Spearman Improvement", shrink=0.8)
    ax.set_title("SIF + ABTT Improvement over Baseline\n(Best Layer Each)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp2_05_sif_improvement_heatmap.png", dpi=150)
    plt.close(fig)
    print("  fig_exp2_05")


# ---------------------------------------------------------------------------
# Figure 6: Best Spearman per model/language
# ---------------------------------------------------------------------------
def fig06_best_spearman(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    models = ["Qwen3-0.6B", "KaLM", "LaBSE"]
    x = np.arange(len(LANGUAGES))
    width = 0.25

    for i, model in enumerate(models):
        vals = []
        for lang in LANGUAGES:
            sub = df[(df["model_short"] == model) & (df["language"] == lang)]
            vals.append(sub["spearman_test"].max() if len(sub) > 0 else 0)
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=MODEL_COLORS[model], alpha=0.85)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.008, f"{v:.3f}",
                    ha="center", fontsize=8, rotation=0)

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Best Spearman (test)", fontsize=12)
    ax.set_title("Best Spearman per Model and Language (Any Method)", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels([LANG_TITLES[l] for l in LANGUAGES], fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp2_06_best_spearman_bars.png", dpi=150)
    plt.close(fig)
    print("  fig_exp2_06")


# ---------------------------------------------------------------------------
# Figure 7: Language difficulty
# ---------------------------------------------------------------------------
def fig07_language_difficulty(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["Qwen3-0.6B", "KaLM", "LaBSE"]
    lang_avgs = {}
    for lang in LANGUAGES:
        vals = []
        for model in models:
            sub = df[(df["model_short"] == model) & (df["language"] == lang)]
            if len(sub) > 0:
                vals.append(sub["spearman_test"].max())
        lang_avgs[lang] = np.mean(vals)

    langs_sorted = sorted(LANGUAGES, key=lambda l: lang_avgs[l], reverse=True)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    bars = ax.barh(range(len(langs_sorted)),
                   [lang_avgs[l] for l in langs_sorted],
                   color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(langs_sorted)))
    ax.set_yticklabels([LANG_TITLES[l] for l in langs_sorted], fontsize=12)
    ax.set_xlabel("Average Best Spearman (across 3 models)", fontsize=11)
    ax.set_title("Language Difficulty Ranking", fontsize=13)
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="x")

    for i, lang in enumerate(langs_sorted):
        ax.text(lang_avgs[lang] + 0.01, i, f"{lang_avgs[lang]:.3f}", va="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp2_07_language_difficulty.png", dpi=150)
    plt.close(fig)
    print("  fig_exp2_07")


# ---------------------------------------------------------------------------
# Figure 8: Optimal D heatmap (English)
# ---------------------------------------------------------------------------
def fig08_optimal_D(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={"hspace": 0.5})

    sif = df[(df["method"] == "sif_abtt_optimal") & (df["language"] == "english")]
    models = ["Qwen3-0.6B", "KaLM", "LaBSE"]

    for ax, model in zip(axes, models):
        sub = sif[sif["model_short"] == model].sort_values("layer")
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        layers = sub["layer"].values
        d_vals = sub["D"].values.astype(int)
        data = d_vals.reshape(1, -1)
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=10)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=7)
        ax.set_yticks([])
        for i, d in enumerate(d_vals):
            ax.text(i, 0, str(d), ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white" if d > 6 else "black")
        ax.set_title(f"{model} (English)", fontsize=11)
        ax.set_xlabel("Layer", fontsize=10)

    fig.suptitle("Optimal D by Model and Layer (English)", fontsize=14)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="D", shrink=0.4, pad=0.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp2_08_optimal_D_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_exp2_08")


# ---------------------------------------------------------------------------
# Figure 9: Grouped bar chart - all methods x models x languages
# ---------------------------------------------------------------------------
def fig09_method_comparison(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = ["Qwen3-0.6B", "KaLM", "LaBSE"]
    methods = ["baseline_mean", "sif_abtt_optimal", "whitening"]
    method_labels = ["Baseline", "SIF+ABTT", "Whitening"]

    for ax, lang in zip(axes.flat, LANGUAGES):
        x = np.arange(len(models))
        width = 0.25

        for i, (method, mlabel) in enumerate(zip(methods, method_labels)):
            vals = []
            for model in models:
                sub = df[(df["model_short"] == model) & (df["language"] == lang) & (df["method"] == method)]
                vals.append(sub["spearman_test"].max() if len(sub) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=mlabel,
                   color=METHOD_COLORS[method], alpha=0.85)
            for j, v in enumerate(vals):
                ax.text(x[j] + i * width, v + 0.008, f"{v:.2f}",
                        ha="center", fontsize=7, rotation=45)

        ax.set_title(LANG_TITLES[lang], fontsize=13, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel("Best Spearman", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Method Comparison Across Models and Languages", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_exp2_09_method_comparison_grouped.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_exp2_09")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load()
    print(f"  {len(df)} rows loaded")
    print("Generating plots...")
    for i, lang in enumerate(LANGUAGES, start=1):
        fig_layer_curves(df, lang, i)
    fig05_sif_improvement(df)
    fig06_best_spearman(df)
    fig07_language_difficulty(df)
    fig08_optimal_D(df)
    fig09_method_comparison(df)
    print(f"Done. Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
