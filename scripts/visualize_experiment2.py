"""Phase 9 Experiment 2: Comprehensive Visualization Suite.

Generates publication-quality figures for the Multilingual Representation
Analysis & Pooling Dynamics experiment.

Figures:
  1. Layer-wise Spearman (3×5 grid: models × languages, method lines)
  2. Best-layer method comparison (grouped bar chart)
  3. Optimal D heatmap (models × languages)
  4. Lexical overlap distributions (ROUGE-L / BLEU-1 by language)
  5. Lexical vs Semantic scatter (ROUGE-L vs cosine sim, low-resource)
  6. LaBSE vs KaLM gap analysis (paired bar chart)
  7. Contingency: alternative pooling comparison
  8. The "Dip" phenomenon — baseline only, all languages overlaid per model
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
from matplotlib.patches import Patch

# ── Constants ────────────────────────────────────────────────────

METHOD_COLORS = {
    "baseline_mean": "#377eb8",
    "sif_abtt_optimal": "#e41a1c",
    "whitening": "#ff7f00",
}

METHOD_LABELS = {
    "baseline_mean": "Baseline (Mean Pool)",
    "sif_abtt_optimal": "SIF + ABTT (opt. D)",
    "whitening": "PCA Whitening",
}

LANG_ORDER = ["english", "french", "serbian", "sinhala", "tamil"]

LANG_COLORS = {
    "english": "#1b9e77",
    "french": "#d95f02",
    "serbian": "#7570b3",
    "sinhala": "#e7298a",
    "tamil": "#66a61e",
}

LANG_LABELS = {
    "english": "English",
    "french": "French",
    "serbian": "Serbian",
    "sinhala": "Sinhala",
    "tamil": "Tamil",
}

MODEL_LABELS = {
    "sentence-transformers/LaBSE": "LaBSE (Encoder)",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini (Decoder)",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B (Decoder)",
}


def short_model(m: str) -> str:
    if "LaBSE" in m:
        return "LaBSE"
    if "KaLM" in m:
        return "KaLM-mini"
    if "Qwen" in m:
        return "Qwen3-0.6B"
    return m.split("/")[-1]


def model_slug(m: str) -> str:
    return m.replace("/", "_")


# ── Args & Style ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", required=True)
    p.add_argument("--diagnostics_csv", default="")
    p.add_argument("--contingency_csv", default="")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ── Figure 1: Layer-wise Spearman (3×5 grid) ────────────────────

def fig1_layerwise_spearman(df: pd.DataFrame, out_dir: Path, dpi: int):
    models = [m for m in df["model"].unique() if m in MODEL_LABELS]
    languages = [l for l in LANG_ORDER if l in df["language"].unique()]
    methods = ["baseline_mean", "sif_abtt_optimal", "whitening"]

    n_m, n_l = len(models), len(languages)
    fig, axes = plt.subplots(n_m, n_l, figsize=(3.2 * n_l, 3.0 * n_m),
                             sharex=False, sharey=True, squeeze=False)

    for i, model in enumerate(models):
        for j, lang in enumerate(languages):
            ax = axes[i, j]
            sub = df[(df["model"] == model) & (df["language"] == lang)]
            for method in methods:
                mdata = sub[sub["method"] == method].sort_values("layer")
                if mdata.empty:
                    continue
                ax.plot(mdata["layer"], mdata["spearman_test"],
                        marker="o", markersize=2.5, linewidth=1.4,
                        color=METHOD_COLORS[method],
                        label=METHOD_LABELS[method] if i == 0 and j == 0 else None)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.set_ylim(0.0, 1.0)
            if i == 0:
                ax.set_title(LANG_LABELS.get(lang, lang))
            if j == 0:
                ax.set_ylabel(f"{short_model(model)}\nSpearman ($\\rho$)")
            if i == n_m - 1:
                ax.set_xlabel("Layer")

    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2, marker="o",
                           markersize=4, label=METHOD_LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(methods),
               fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("Layer-wise Spearman Correlation by Model and Language",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig1_layerwise_spearman.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Best-layer method comparison (grouped bars) ────────

def fig2_method_comparison(df: pd.DataFrame, out_dir: Path, dpi: int):
    models = [m for m in df["model"].unique() if m in MODEL_LABELS]
    languages = [l for l in LANG_ORDER if l in df["language"].unique()]
    methods = ["baseline_mean", "sif_abtt_optimal", "whitening"]

    n_groups = len(languages)
    n_bars = len(methods)
    bar_w = 0.22
    x = np.arange(n_groups)

    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        for k, method in enumerate(methods):
            vals = []
            for lang in languages:
                sub = df[(df["model"] == model) & (df["language"] == lang)
                         & (df["method"] == method)]
                vals.append(sub["spearman_test"].max() if len(sub) > 0 else 0)
            offset = (k - (n_bars - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, color=METHOD_COLORS[method],
                          label=METHOD_LABELS[method] if ax_idx == 0 else None,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages], rotation=30, ha="right")
        ax.set_title(short_model(model))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
        if ax_idx == 0:
            ax.set_ylabel("Best Spearman ($\\rho$)")

    fig.legend(loc="upper center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Best-Layer Spearman by Method (Test Set)", fontsize=13,
                 fontweight="bold", y=1.12)
    fig.tight_layout()
    path = out_dir / "fig2_method_comparison.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Optimal D Heatmap ─────────────────────────────────

def fig3_optimal_d_heatmap(df: pd.DataFrame, out_dir: Path, dpi: int):
    abtt = df[df["method"] == "sif_abtt_optimal"].copy()
    if abtt.empty:
        return

    models = [m for m in df["model"].unique() if m in MODEL_LABELS]
    languages = [l for l in LANG_ORDER if l in df["language"].unique()]

    # For each (model, lang): find best test layer -> get D and best layer
    heatmap = np.full((len(models), len(languages)), np.nan)
    layer_map = np.full((len(models), len(languages)), np.nan)
    for i, model in enumerate(models):
        for j, lang in enumerate(languages):
            sub = abtt[(abtt["model"] == model) & (abtt["language"] == lang)]
            if sub.empty:
                continue
            best = sub.loc[sub["spearman_test"].idxmax()]
            heatmap[i, j] = best["D"]
            layer_map[i, j] = best["layer"]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)

    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_model(m) for m in models])

    for i in range(len(models)):
        for j in range(len(languages)):
            d = heatmap[i, j]
            l = layer_map[i, j]
            if not np.isnan(d):
                ax.text(j, i, f"D={int(d)}\nL{int(l)}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if d > 5 else "black")

    ax.set_title("Optimal D and Best Layer (SIF+ABTT)")
    fig.colorbar(im, ax=ax, label="D (PCs removed)", shrink=0.8)
    fig.tight_layout()
    path = out_dir / "fig3_optimal_d_heatmap.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Lexical overlap distributions ──────────────────────

def fig4_lexical_distributions(diagnostics_csv: str, out_dir: Path, dpi: int):
    if not diagnostics_csv:
        print("  Skipping Fig 4 (no diagnostics_csv).")
        return
    diag = pd.read_csv(diagnostics_csv)
    languages = [l for l in LANG_ORDER if l in diag["language"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: ROUGE-L distributions (violin-like box plots)
    for metric, ax, title in [("rouge_l", axes[0], "ROUGE-L"),
                               ("bleu_1", axes[1], "BLEU-1")]:
        # Split into positive (sim>3) and all
        data_pos = []
        data_all = []
        labels = []
        for lang in languages:
            ldf = diag[diag["language"] == lang]
            pos = ldf[ldf["similarity"] > 3.0][metric].values
            data_pos.append(pos)
            data_all.append(ldf[metric].values)
            labels.append(LANG_LABELS.get(lang, lang))

        positions = np.arange(len(languages))
        bp_all = ax.boxplot(data_all, positions=positions - 0.18, widths=0.3,
                            patch_artist=True, showfliers=False, showmeans=True,
                            meanprops=dict(marker="D", markerfacecolor="black", markersize=4))
        bp_pos = ax.boxplot(data_pos, positions=positions + 0.18, widths=0.3,
                            patch_artist=True, showfliers=False, showmeans=True,
                            meanprops=dict(marker="D", markerfacecolor="black", markersize=4))

        for patch in bp_all["boxes"]:
            patch.set_facecolor("#b3cde3")
            patch.set_alpha(0.8)
        for patch in bp_pos["boxes"]:
            patch.set_facecolor("#fbb4ae")
            patch.set_alpha(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} Distribution by Language")
        ax.grid(axis="y", alpha=0.2)
        ax.legend([bp_all["boxes"][0], bp_pos["boxes"][0]],
                  ["All pairs", "Positive pairs (sim > 3)"],
                  fontsize=8, loc="upper right")

    fig.suptitle("Lexical Overlap: ROUGE-L and BLEU-1 Across Languages",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig4_lexical_distributions.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Lexical vs Semantic scatter ────────────────────────

def fig5_lexical_vs_semantic(df: pd.DataFrame, diagnostics_csv: str,
                              cache_dir: str, out_dir: Path, dpi: int):
    if not diagnostics_csv or not cache_dir:
        print("  Skipping Fig 5 (no diagnostics or cache).")
        return

    diag = pd.read_csv(diagnostics_csv)
    cache_p = Path(cache_dir)
    low_resource = ["sinhala", "tamil", "serbian"]
    abtt = df[df["method"] == "baseline_mean"]

    fig, axes = plt.subplots(1, len(low_resource), figsize=(5 * len(low_resource), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for idx, lang in enumerate(low_resource):
        ax = axes[idx]
        lang_diag = diag[diag["language"] == lang]
        if lang_diag.empty:
            continue

        # Find best model+layer for this language (use baseline to show the raw failure)
        lang_res = abtt[abtt["language"] == lang]
        if lang_res.empty:
            continue
        best = lang_res.loc[lang_res["spearman_test"].idxmax()]
        slug = model_slug(best["model"])
        layer = int(best["layer"])

        sims_path = cache_p / slug / lang / f"layer{layer}_baseline_mean_pair_sims.npy"
        if not sims_path.exists():
            # Fallback: try sif_abtt_optimal
            abtt2 = df[(df["method"] == "sif_abtt_optimal") & (df["language"] == lang)]
            if not abtt2.empty:
                best2 = abtt2.loc[abtt2["spearman_test"].idxmax()]
                slug = model_slug(best2["model"])
                layer = int(best2["layer"])
                sims_path = cache_p / slug / lang / f"layer{layer}_sif_abtt_optimal_pair_sims.npy"
            if not sims_path.exists():
                print(f"    No pair sims for {lang}")
                continue

        pair_sims = np.load(sims_path)
        n = min(len(lang_diag), len(pair_sims))
        rouge = lang_diag["rouge_l"].values[:n]
        cosine = pair_sims[:n]
        sim_scores = lang_diag["similarity"].values[:n]

        # Color by human similarity score
        sc = ax.scatter(rouge, cosine, c=sim_scores, cmap="RdYlGn",
                       alpha=0.5, s=12, vmin=0, vmax=5, edgecolors="none")
        ax.set_xlabel("ROUGE-L")
        ax.set_title(f"{LANG_LABELS.get(lang, lang)}\n({short_model(best['model'])} L{layer})")
        ax.grid(True, alpha=0.15)

        # Mark the Theory A zone (high ROUGE, low cosine)
        ax.axhspan(ymin=ax.get_ylim()[0], ymax=0.5, xmin=0.6, xmax=1.0,
                   alpha=0.05, color="red")
        if idx == 0:
            ax.annotate("Theory A zone\n(high lexical,\nlow semantic)", xy=(0.7, 0.3),
                       fontsize=7, color="red", alpha=0.6, style="italic")

    axes[0].set_ylabel("Cosine Similarity")
    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, label="Human Score (0-5)")
    fig.suptitle("Lexical vs Semantic Similarity (Low-Resource Languages)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    path = out_dir / "fig5_lexical_vs_semantic.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: Encoder vs Decoder gap ─────────────────────────────

def fig6_encoder_vs_decoder_gap(df: pd.DataFrame, out_dir: Path, dpi: int):
    languages = [l for l in LANG_ORDER if l in df["language"].unique()]
    models = [m for m in df["model"].unique() if m in MODEL_LABELS]

    if len(models) < 2:
        print("  Skipping Fig 6 (need 2+ models).")
        return

    # Get best Spearman per (model, language) across all methods
    best_by_ml = (
        df.groupby(["model", "language"])["spearman_test"]
        .max()
        .reset_index()
    )

    # Identify LaBSE and decoder models
    labse_m = [m for m in models if "LaBSE" in m]
    decoder_models = [m for m in models if "LaBSE" not in m]

    # Panel count: 1 (grouped bars) + 1 per decoder model (gap chart)
    n_panels = 1 + len(decoder_models)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Panel A: side-by-side bars (best Spearman by model)
    ax = axes[0]
    x = np.arange(len(languages))
    n_models = len(models)
    bar_w = 0.8 / n_models

    model_colors = {
        m: c for m, c in zip(
            models,
            ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"][:len(models)]
        )
    }
    # Ensure LaBSE is always blue
    for m in models:
        if "LaBSE" in m:
            model_colors[m] = "#377eb8"

    for k, model in enumerate(models):
        vals = []
        for lang in languages:
            sub = best_by_ml[(best_by_ml["model"] == model) & (best_by_ml["language"] == lang)]
            vals.append(sub["spearman_test"].values[0] if len(sub) > 0 else 0)
        offset = (k - (n_models - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=short_model(model),
                      color=model_colors[model], edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages])
    ax.set_ylabel("Best Spearman ($\\rho$)")
    ax.set_title("Best Performance by Model")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, 1.08)

    # Panels B+: gap (LaBSE - decoder) for each decoder model
    if labse_m:
        labse_name = labse_m[0]
        for panel_idx, dec_model in enumerate(decoder_models):
            ax = axes[1 + panel_idx]
            gaps = []
            for lang in languages:
                l_sub = best_by_ml[(best_by_ml["model"] == labse_name) & (best_by_ml["language"] == lang)]
                d_sub = best_by_ml[(best_by_ml["model"] == dec_model) & (best_by_ml["language"] == lang)]
                l_val = l_sub["spearman_test"].values[0] if len(l_sub) > 0 else 0
                d_val = d_sub["spearman_test"].values[0] if len(d_sub) > 0 else 0
                gaps.append(l_val - d_val)

            colors = ["#2166ac" if g > 0 else "#b2182b" for g in gaps]
            bars = ax.bar(x, gaps, 0.6, color=colors, edgecolor="white")
            for bar, g in zip(bars, gaps):
                y = g + 0.005 if g > 0 else g - 0.015
                ax.text(bar.get_x() + bar.get_width() / 2, y,
                        f"{g:+.3f}", ha="center",
                        va="bottom" if g > 0 else "top", fontsize=8.5)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages])
            dec_short = short_model(dec_model)
            ax.set_ylabel(f"LaBSE $-$ {dec_short} ($\\Delta\\rho$)")
            ax.set_title(f"Gap: LaBSE vs {dec_short}")
            ax.grid(axis="y", alpha=0.2)
            ax.annotate("Blue = LaBSE wins", xy=(0.02, 0.95), xycoords="axes fraction",
                        fontsize=7.5, color="#2166ac", fontweight="bold")
            ax.annotate("Red = Decoder wins", xy=(0.02, 0.88), xycoords="axes fraction",
                        fontsize=7.5, color="#b2182b", fontweight="bold")

    fig.suptitle("Encoder (LaBSE) vs Decoder Models: Performance Comparison",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    path = out_dir / "fig6_encoder_vs_decoder.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 7: Contingency results ───────────────────────────────

def fig7_contingency(contingency_csv: str, out_dir: Path, dpi: int):
    if not contingency_csv:
        print("  Skipping Fig 7 (no contingency_csv).")
        return

    cont = pd.read_csv(contingency_csv)
    if cont.empty:
        return

    cont_models = cont["model"].unique()
    languages = [l for l in LANG_ORDER if l in cont["language"].unique()]

    contingency_colors = {
        "baseline": "#999999",
        "abtt_optimal": "#e41a1c",
        "optimal_transport": "#7570b3",
    }
    contingency_labels = {
        "baseline": "Last-Token (raw)",
        "abtt_optimal": "Last-Token + ABTT",
        "optimal_transport": "Optimal Transport",
    }
    c_methods = ["baseline", "abtt_optimal", "optimal_transport"]

    n_cont_models = len(cont_models)
    fig, axes = plt.subplots(n_cont_models, 2, figsize=(14, 5 * n_cont_models),
                             squeeze=False)

    for row_idx, model_name in enumerate(cont_models):
        model_cont = cont[cont["model"] == model_name]
        model_short = short_model(model_name)
        x = np.arange(len(languages))
        bar_w = 0.25

        # Panel A: Spearman by method
        ax = axes[row_idx, 0]
        for k, method in enumerate(c_methods):
            vals = []
            for lang in languages:
                sub = model_cont[(model_cont["language"] == lang) & (model_cont["method"] == method)]
                vals.append(sub["spearman_test"].values[0] if len(sub) > 0 else 0)
            offset = (k - 1) * bar_w
            ax.bar(x + offset, vals, bar_w,
                   color=contingency_colors.get(method, "#000"),
                   label=contingency_labels.get(method, method) if row_idx == 0 else None,
                   edgecolor="white")

        # Reference lines per language
        for i, lang in enumerate(languages):
            sub = model_cont[model_cont["language"] == lang]
            if len(sub) > 0:
                ref = sub["baseline_spearman"].values[0]
                ax.hlines(ref, i - 0.4, i + 0.4, colors="#377eb8",
                         linestyles="dashed", linewidth=1.5,
                         label="Mean Pool baseline" if i == 0 and row_idx == 0 else None)
                labse = sub["labse_spearman"].values[0]
                ax.hlines(labse, i - 0.4, i + 0.4, colors="#2ca02c",
                         linestyles="dotted", linewidth=1.5,
                         label="LaBSE baseline" if i == 0 and row_idx == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages], rotation=25, ha="right")
        ax.set_ylabel(f"{model_short}\nSpearman ($\\rho$)")
        ax.set_title(f"Alternative Pooling ({model_short})")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(-0.1, 1.0)

        # Panel B: Performance deltas
        ax = axes[row_idx, 1]
        for k, method in enumerate(c_methods):
            vals = []
            for lang in languages:
                sub = model_cont[(model_cont["language"] == lang) & (model_cont["method"] == method)]
                vals.append(sub["performance_delta"].values[0] if len(sub) > 0 else 0)
            offset = (k - 1) * bar_w
            ax.bar(x + offset, vals, bar_w,
                   color=contingency_colors.get(method, "#000"),
                   label=contingency_labels.get(method, method) if row_idx == 0 else None,
                   edgecolor="white")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages], rotation=25, ha="right")
        ax.set_ylabel("$\\Delta$ Spearman (vs Mean Pool)")
        ax.set_title(f"Performance Delta ({model_short})")
    # Legend from first row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    handles.extend(h2)
    labels.extend(l2)
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)),
               fontsize=8, bbox_to_anchor=(0.5, -0.03), frameon=False)

    fig.suptitle("Phase 9.5 Contingency: Alternative Pooling Strategies",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig7_contingency.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 8: The "Dip" phenomenon (baseline only) ──────────────

def fig8_dip_phenomenon(df: pd.DataFrame, out_dir: Path, dpi: int):
    models = [m for m in df["model"].unique() if m in MODEL_LABELS]
    languages = [l for l in LANG_ORDER if l in df["language"].unique()]
    baseline = df[df["method"] == "baseline_mean"]

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        for lang in languages:
            sub = baseline[(baseline["model"] == model) & (baseline["language"] == lang)]
            sub = sub.sort_values("layer")
            if sub.empty:
                continue
            ax.plot(sub["layer"], sub["spearman_test"],
                    marker="o", markersize=3, linewidth=1.5,
                    color=LANG_COLORS[lang], label=LANG_LABELS.get(lang, lang))
        ax.set_xlabel("Layer")
        ax.set_title(f"{short_model(model)}")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("Spearman ($\\rho$) — Baseline Mean Pool")
    fig.suptitle("The \"Dip\" Phenomenon: Raw Baseline Across Layers",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig8_dip_phenomenon.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 9: Summary dashboard ─────────────────────────────────

def fig9_summary_table(df: pd.DataFrame, diagnostics_csv: str,
                        contingency_csv: str, out_dir: Path, dpi: int):
    """A visual summary table as a figure for quick reference."""
    models = [m for m in df["model"].unique() if m in MODEL_LABELS]
    languages = [l for l in LANG_ORDER if l in df["language"].unique()]
    methods = ["baseline_mean", "sif_abtt_optimal", "whitening"]

    # Build summary data
    rows = []
    for model in models:
        for lang in languages:
            row = {"Model": short_model(model), "Language": LANG_LABELS.get(lang, lang)}
            for method in methods:
                sub = df[(df["model"] == model) & (df["language"] == lang)
                         & (df["method"] == method)]
                row[METHOD_LABELS[method]] = sub["spearman_test"].max() if len(sub) > 0 else np.nan
            # Best method
            vals = {m: row[METHOD_LABELS[m]] for m in methods}
            best_m = max(vals, key=lambda k: vals[k] if not np.isnan(vals[k]) else -1)
            row["Best"] = METHOD_LABELS[best_m]
            rows.append(row)

    tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 3 + 0.4 * len(rows)))
    ax.axis("off")

    cols = list(tbl.columns)
    cell_text = []
    cell_colors = []
    for _, r in tbl.iterrows():
        text_row = []
        color_row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and not np.isnan(v):
                text_row.append(f"{v:.3f}")
                # Color code by value
                if v >= 0.8:
                    color_row.append("#c7e9c0")
                elif v >= 0.6:
                    color_row.append("#ffffcc")
                elif v >= 0.4:
                    color_row.append("#fee8c8")
                else:
                    color_row.append("#fdd49e")
            else:
                text_row.append(str(v))
                color_row.append("white")
        cell_text.append(text_row)
        cell_colors.append(color_row)

    table = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colors,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(range(len(cols)))
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#d9d9d9")

    fig.suptitle("Experiment 2 Summary: Best Spearman ($\\rho$) by Model × Language × Method",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.95)
    path = out_dir / "fig9_summary_table.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("Loading data...")
    df = pd.read_csv(args.results_csv)
    print(f"  results.csv: {len(df)} rows")

    print("\nFigure 1: Layer-wise Spearman (3×5 grid)")
    fig1_layerwise_spearman(df, out_dir, args.dpi)

    print("Figure 2: Best-layer method comparison")
    fig2_method_comparison(df, out_dir, args.dpi)

    print("Figure 3: Optimal D heatmap")
    fig3_optimal_d_heatmap(df, out_dir, args.dpi)

    print("Figure 4: Lexical overlap distributions")
    fig4_lexical_distributions(args.diagnostics_csv, out_dir, args.dpi)

    print("Figure 5: Lexical vs Semantic scatter")
    fig5_lexical_vs_semantic(df, args.diagnostics_csv, args.cache_dir, out_dir, args.dpi)

    print("Figure 6: Encoder vs Decoder gap")
    fig6_encoder_vs_decoder_gap(df, out_dir, args.dpi)

    print("Figure 7: Contingency results")
    fig7_contingency(args.contingency_csv, out_dir, args.dpi)

    print("Figure 8: The Dip phenomenon")
    fig8_dip_phenomenon(df, out_dir, args.dpi)

    print("Figure 9: Summary table")
    fig9_summary_table(df, args.diagnostics_csv, args.contingency_csv, out_dir, args.dpi)

    print("\nAll figures saved to:", out_dir)


if __name__ == "__main__":
    main()
