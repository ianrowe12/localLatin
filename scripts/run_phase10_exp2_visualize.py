"""Phase 10 Experiment 2: Comprehensive Visualization Suite.

Copy of visualize_experiment2.py with:
  - Spanish added to LANG_ORDER, LANG_COLORS, LANG_LABELS
  - New large model labels added to MODEL_LABELS
  - --bias_csv arg
  - Fig 10: Bias analysis (new)

Figures:
  1. Layer-wise Spearman (N_models × 6 grid: models × languages, method lines)
  2. Best-layer method comparison (grouped bar chart)
  3. Optimal D heatmap (models × languages)
  4. Lexical overlap distributions (ROUGE-L / BLEU-1 by language)
  5. Lexical vs Semantic scatter (ROUGE-L vs cosine sim, low-resource)
  6. LaBSE vs Decoder gap analysis (paired bar chart)
  7. Contingency: alternative pooling comparison
  8. The "Dip" phenomenon — baseline only, all languages overlaid per model
  9. Summary table
  10. Bias analysis — mean residual by resource level (new)
  11. Cross-phase scaling comparison (Phase 9 small → Phase 10 large)
  12. Per-language bias breakdown (individual language residuals)
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

LANG_ORDER = ["english", "french", "spanish", "serbian", "sinhala", "tamil"]

LANG_COLORS = {
    "english": "#1b9e77",
    "french": "#d95f02",
    "spanish": "#a6761d",
    "serbian": "#7570b3",
    "sinhala": "#e7298a",
    "tamil": "#66a61e",
}

LANG_LABELS = {
    "english": "English",
    "french": "French",
    "spanish": "Spanish",
    "serbian": "Serbian",
    "sinhala": "Sinhala",
    "tamil": "Tamil",
}

MODEL_LABELS = {
    "sentence-transformers/LaBSE": "LaBSE (Encoder)",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini (Decoder)",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B (Decoder)",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-8B (Dec)",
    "tencent/KaLM-Embedding-Gemma3-12B-2511": "KaLM-Gemma3-12B (Dec)",
}


def short_model(m: str) -> str:
    if m in MODEL_LABELS:
        return MODEL_LABELS[m]
    if "LaBSE" in m:
        return "LaBSE"
    if "KaLM" in m:
        return "KaLM"
    if "Qwen" in m:
        return "Qwen"
    return m.split("/")[-1]


def model_slug(m: str) -> str:
    return m.replace("/", "_")


# ── Args & Style ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", required=True)
    p.add_argument("--diagnostics_csv", default="")
    p.add_argument("--contingency_csv", default="")
    p.add_argument("--bias_csv", default="")
    p.add_argument("--phase9_results_csv", default="")
    p.add_argument("--phase9_contingency_csv", default="")
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


# ── Figure 1: Layer-wise Spearman ───────────────────────────────

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

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(languages)), 2.8))
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

    for metric, ax, title in [("rouge_l", axes[0], "ROUGE-L"),
                               ("bleu_1", axes[1], "BLEU-1")]:
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

        lang_res = abtt[abtt["language"] == lang]
        if lang_res.empty:
            continue
        best = lang_res.loc[lang_res["spearman_test"].idxmax()]
        slug = model_slug(best["model"])
        layer = int(best["layer"])

        sims_path = cache_p / slug / lang / f"layer{layer}_baseline_mean_pair_sims.npy"
        if not sims_path.exists():
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

        sc = ax.scatter(rouge, cosine, c=sim_scores, cmap="RdYlGn",
                       alpha=0.5, s=12, vmin=0, vmax=5, edgecolors="none")
        ax.set_xlabel("ROUGE-L")
        ax.set_title(f"{LANG_LABELS.get(lang, lang)}\n({short_model(best['model'])} L{layer})")
        ax.grid(True, alpha=0.15)

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

def fig6_encoder_vs_decoder_gap(df: pd.DataFrame, out_dir: Path, dpi: int,
                                phase9_df: pd.DataFrame | None = None):
    """Compare Phase 10 decoders against Phase 9 LaBSE reference."""
    languages_5 = ["english", "french", "serbian", "sinhala", "tamil"]
    p10_models = [m for m in df["model"].unique() if m in MODEL_LABELS]

    # Build best-performance table combining Phase 9 LaBSE + Phase 10 decoders
    all_best = (
        df.groupby(["model", "language"])["spearman_test"]
        .max().reset_index()
    )

    labse_best = None
    if phase9_df is not None:
        labse_rows = phase9_df[phase9_df["model"].str.contains("LaBSE")]
        if not labse_rows.empty:
            labse_best = (
                labse_rows.groupby(["model", "language"])["spearman_test"]
                .max().reset_index()
            )
            all_best = pd.concat([all_best, labse_best], ignore_index=True)

    all_models = list(all_best["model"].unique())
    # Order: LaBSE first, then Phase 10 models
    labse_m = [m for m in all_models if "LaBSE" in m]
    decoder_models = [m for m in all_models if "LaBSE" not in m]
    plot_models = labse_m + decoder_models
    languages = languages_5  # Only 5 languages (Spanish has no LaBSE ref)

    n_panels = 1 + (len(decoder_models) if labse_m else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5),
                             squeeze=False)
    axes = axes[0]

    # Panel 1: grouped bars
    ax = axes[0]
    x = np.arange(len(languages))
    n_m = len(plot_models)
    bar_w = 0.8 / max(n_m, 1)
    palette = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"]

    for k, model in enumerate(plot_models):
        vals = []
        for lang in languages:
            sub = all_best[(all_best["model"] == model) & (all_best["language"] == lang)]
            vals.append(sub["spearman_test"].values[0] if len(sub) > 0 else 0)
        offset = (k - (n_m - 1) / 2) * bar_w
        color = "#377eb8" if "LaBSE" in model else palette[min(k, len(palette)-1)]
        bars = ax.bar(x + offset, vals, bar_w, label=short_model(model),
                      color=color, edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages])
    ax.set_ylabel("Best Spearman ($\\rho$)")
    ax.set_title("Best Performance by Model")
    note = " (LaBSE from Phase 9)" if labse_m else ""
    ax.legend(fontsize=7, loc="lower left", title=note, title_fontsize=6)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, 1.08)

    # Gap panels
    if labse_m:
        labse_name = labse_m[0]
        for panel_idx, dec_model in enumerate(decoder_models):
            ax = axes[1 + panel_idx]
            gaps = []
            for lang in languages:
                l_sub = all_best[(all_best["model"] == labse_name) & (all_best["language"] == lang)]
                d_sub = all_best[(all_best["model"] == dec_model) & (all_best["language"] == lang)]
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
            ax.set_title(f"Gap: LaBSE (Phase 9) vs {dec_short}")
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    handles.extend(h2)
    labels.extend(l2)
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)),
               fontsize=8, bbox_to_anchor=(0.5, -0.03), frameon=False)

    fig.suptitle("Phase 10 Contingency: Alternative Pooling Strategies",
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
    models = [m for m in df["model"].unique() if m in MODEL_LABELS]
    languages = [l for l in LANG_ORDER if l in df["language"].unique()]
    methods = ["baseline_mean", "sif_abtt_optimal", "whitening"]

    rows = []
    for model in models:
        for lang in languages:
            row = {"Model": short_model(model), "Language": LANG_LABELS.get(lang, lang)}
            for method in methods:
                sub = df[(df["model"] == model) & (df["language"] == lang)
                         & (df["method"] == method)]
                row[METHOD_LABELS[method]] = sub["spearman_test"].max() if len(sub) > 0 else np.nan
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


# ── Figure 10: Bias Analysis ─────────────────────────────────────

def fig10_bias_analysis(bias_csv: str, out_dir: Path, dpi: int):
    if not bias_csv:
        print("  Skipping Fig 10 (no bias_csv).")
        return

    bias_df = pd.read_csv(bias_csv)
    if bias_df.empty:
        print("  Skipping Fig 10 (bias_csv is empty).")
        return

    models = bias_df["model"].unique()
    methods = ["baseline_mean", "sif_abtt_optimal", "whitening"]
    method_xlabels = ["Baseline\n(Mean Pool)", "SIF+ABTT\n(opt. D)", "PCA\nWhitening"]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    bar_w = 0.3
    x = np.arange(len(methods))

    for ax_idx, model_name in enumerate(models):
        ax = axes[ax_idx]
        model_data = bias_df[bias_df["model"] == model_name]

        high_means = []
        high_stds = []
        low_means = []
        low_stds = []

        for method in methods:
            md = model_data[model_data["method"] == method]
            high = md[md["resource_level"] == "high"]["mean_residual"]
            low = md[md["resource_level"] == "low"]["mean_residual"]
            high_means.append(high.mean() if len(high) > 0 else 0.0)
            high_stds.append(high.std() if len(high) > 1 else 0.0)
            low_means.append(low.mean() if len(low) > 0 else 0.0)
            low_stds.append(low.std() if len(low) > 1 else 0.0)

        bars_high = ax.bar(x - bar_w / 2, high_means, bar_w,
                           yerr=high_stds, capsize=4,
                           color="#2166ac", alpha=0.8, label="High-resource",
                           edgecolor="white", error_kw={"linewidth": 1.2})
        bars_low = ax.bar(x + bar_w / 2, low_means, bar_w,
                          yerr=low_stds, capsize=4,
                          color="#b2182b", alpha=0.8, label="Low-resource",
                          edgecolor="white", error_kw={"linewidth": 1.2})

        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(method_xlabels)
        ax.set_title(short_model(model_name))
        ax.grid(axis="y", alpha=0.2)

        if ax_idx == 0:
            ax.set_ylabel("Mean Residual (predicted cosine − GT/5)")
            ax.legend(fontsize=9)

    fig.suptitle(
        "Bias Analysis: Systematic Over/Under-estimation by Resource Level",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = out_dir / "fig10_bias_analysis.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 11: Cross-phase scaling comparison ──────────────────

def fig11_scaling_comparison(df: pd.DataFrame, phase9_df: pd.DataFrame | None,
                              out_dir: Path, dpi: int):
    """Compare small (Phase 9) vs large (Phase 10) model variants."""
    if phase9_df is None:
        print("  Skipping Fig 11 (no phase9_results_csv).")
        return

    # Define scaling pairs: (Phase9 small, Phase10 large, short label)
    pairs = [
        ("Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B", "Qwen3"),
        ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
         "tencent/KaLM-Embedding-Gemma3-12B-2511", "KaLM"),
    ]

    languages = ["english", "french", "serbian", "sinhala", "tamil"]
    methods = ["baseline_mean", "sif_abtt_optimal"]
    method_labels = {"baseline_mean": "Baseline", "sif_abtt_optimal": "SIF+ABTT"}

    fig, axes = plt.subplots(len(pairs), len(methods),
                              figsize=(7 * len(methods), 5 * len(pairs)),
                              squeeze=False)

    for row, (small_m, large_m, fam_label) in enumerate(pairs):
        small_df = phase9_df[phase9_df["model"] == small_m]
        large_df = df[df["model"] == large_m]

        if small_df.empty and large_df.empty:
            continue

        for col, method in enumerate(methods):
            ax = axes[row, col]
            x = np.arange(len(languages))
            bar_w = 0.35

            # Small model
            small_vals = []
            for lang in languages:
                sub = small_df[(small_df["language"] == lang) & (small_df["method"] == method)]
                small_vals.append(sub["spearman_test"].max() if len(sub) > 0 else 0)

            # Large model
            large_vals = []
            for lang in languages:
                sub = large_df[(large_df["language"] == lang) & (large_df["method"] == method)]
                large_vals.append(sub["spearman_test"].max() if len(sub) > 0 else 0)

            bars1 = ax.bar(x - bar_w / 2, small_vals, bar_w,
                           color="#74add1", edgecolor="white",
                           label=f"{short_model(small_m)}")
            bars2 = ax.bar(x + bar_w / 2, large_vals, bar_w,
                           color="#d73027", edgecolor="white",
                           label=f"{short_model(large_m)}")

            # Annotate deltas
            for i, (s, l) in enumerate(zip(small_vals, large_vals)):
                if s > 0 and l > 0:
                    delta = l - s
                    color = "#2ca02c" if delta > 0 else "#d62728"
                    top = max(s, l)
                    ax.text(x[i], top + 0.02, f"{delta:+.3f}",
                            ha="center", va="bottom", fontsize=7.5,
                            color=color, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages],
                               rotation=25, ha="right")
            ax.set_ylim(0, 1.08)
            ax.set_title(f"{fam_label} — {method_labels[method]}")
            ax.grid(axis="y", alpha=0.2)
            if col == 0:
                ax.set_ylabel("Best Spearman ($\\rho$)")
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Scaling Effect: Small (Phase 9) vs Large (Phase 10) Models",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig11_scaling_comparison.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 12: Per-language bias breakdown ─────────────────────

def fig12_bias_per_language(bias_csv: str, out_dir: Path, dpi: int):
    """Show residuals for each language individually (not just high/low)."""
    if not bias_csv:
        print("  Skipping Fig 12 (no bias_csv).")
        return

    bias_df = pd.read_csv(bias_csv).drop_duplicates(
        subset=["model", "language", "method", "layer"]
    )
    if bias_df.empty:
        return

    # Focus on Qwen3-8B (complete data)
    qwen = bias_df[bias_df["model"].str.contains("Qwen3-Embedding-8B")]
    if qwen.empty:
        print("  Skipping Fig 12 (no Qwen3-8B bias data).")
        return

    methods = ["baseline_mean", "sif_abtt_optimal", "whitening"]
    method_short = {"baseline_mean": "Baseline", "sif_abtt_optimal": "SIF+ABTT",
                    "whitening": "Whitening"}
    languages = [l for l in LANG_ORDER if l in qwen["language"].unique()]

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5),
                              sharey=True, squeeze=False)
    axes = axes[0]

    resource_colors = {"high": "#74add1", "low": "#f46d43"}
    x = np.arange(len(languages))

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        mdata = qwen[qwen["method"] == method]

        residuals = []
        stds = []
        colors = []
        for lang in languages:
            sub = mdata[mdata["language"] == lang]
            if len(sub) > 0:
                residuals.append(sub["mean_residual"].values[0])
                stds.append(sub["std_residual"].values[0])
                rl = sub["resource_level"].values[0]
                colors.append(resource_colors.get(rl, "#999"))
            else:
                residuals.append(0)
                stds.append(0)
                colors.append("#999")

        bars = ax.bar(x, residuals, 0.6, color=colors, edgecolor="white",
                      yerr=stds, capsize=3, error_kw={"linewidth": 1.0, "alpha": 0.5})
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

        # Annotate overestimation fraction
        for i, lang in enumerate(languages):
            sub = mdata[mdata["language"] == lang]
            if len(sub) > 0:
                frac = sub["frac_overestimated"].values[0]
                y_pos = residuals[i] + stds[i] + 0.02 if residuals[i] > 0 else residuals[i] - stds[i] - 0.04
                ax.text(x[i], y_pos, f"{frac:.0%}",
                        ha="center", va="bottom" if residuals[i] > 0 else "top",
                        fontsize=7, color="#555")

        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages],
                           rotation=30, ha="right")
        ax.set_title(method_short[method])
        ax.grid(axis="y", alpha=0.2)
        if ax_idx == 0:
            ax.set_ylabel("Mean Residual (cos − GT/5)")
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(color="#74add1", label="High-resource"),
                Patch(color="#f46d43", label="Low-resource"),
            ], fontsize=8, loc="upper right")

    fig.suptitle("Qwen3-8B: Per-Language Bias (% = overestimation rate)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig12_bias_per_language.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 13: Cross-phase contingency comparison ──────────────

def fig13_contingency_scaling(contingency_csv: str, phase9_contingency_csv: str,
                               out_dir: Path, dpi: int):
    """Compare contingency (alt pooling) results: 0.6B vs 8B Qwen."""
    if not contingency_csv or not phase9_contingency_csv:
        print("  Skipping Fig 13 (missing contingency CSVs).")
        return

    c10 = pd.read_csv(contingency_csv).drop_duplicates(
        subset=["model", "language", "method", "pooling"]
    )
    c9 = pd.read_csv(phase9_contingency_csv)

    # Only Qwen models
    q9 = c9[c9["model"].str.contains("Qwen3-Embedding-0.6B")]
    q10 = c10[c10["model"].str.contains("Qwen3-Embedding-8B")]
    if q9.empty or q10.empty:
        print("  Skipping Fig 13 (missing Qwen contingency data).")
        return

    languages = ["english", "french", "serbian", "sinhala", "tamil"]
    c_methods = ["baseline", "abtt_optimal", "optimal_transport"]
    c_labels = {"baseline": "Last-Tok", "abtt_optimal": "Last-Tok+ABTT",
                "optimal_transport": "OT"}

    fig, axes = plt.subplots(1, len(c_methods), figsize=(6 * len(c_methods), 5),
                              sharey=True, squeeze=False)
    axes = axes[0]

    x = np.arange(len(languages))
    bar_w = 0.35

    for col, method in enumerate(c_methods):
        ax = axes[col]
        v9 = []
        v10 = []
        for lang in languages:
            s9 = q9[(q9["language"] == lang) & (q9["method"] == method)]
            s10 = q10[(q10["language"] == lang) & (q10["method"] == method)]
            v9.append(s9["spearman_test"].values[0] if len(s9) > 0 else np.nan)
            v10.append(s10["spearman_test"].values[0] if len(s10) > 0 else np.nan)

        ax.bar(x - bar_w / 2, v9, bar_w, color="#74add1", edgecolor="white",
               label="Qwen3-0.6B (Phase 9)")
        ax.bar(x + bar_w / 2, v10, bar_w, color="#d73027", edgecolor="white",
               label="Qwen3-8B (Phase 10)")

        for i in range(len(languages)):
            if not np.isnan(v9[i]) and not np.isnan(v10[i]):
                delta = v10[i] - v9[i]
                top = max(v9[i], v10[i])
                color = "#2ca02c" if delta > 0 else "#d62728"
                ax.text(x[i], top + 0.02, f"{delta:+.3f}", ha="center",
                        va="bottom", fontsize=7, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages],
                           rotation=25, ha="right")
        ax.set_title(c_labels[method])
        ax.set_ylim(-0.1, 1.05)
        ax.grid(axis="y", alpha=0.2)
        if col == 0:
            ax.set_ylabel("Spearman ($\\rho$)")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Contingency Scaling: Qwen3-0.6B vs 8B (Alternative Pooling)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig13_contingency_scaling.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("Loading data...")
    df = pd.read_csv(args.results_csv)
    print(f"  results.csv: {len(df)} rows")

    # Load Phase 9 reference data
    phase9_df = None
    if args.phase9_results_csv:
        phase9_df = pd.read_csv(args.phase9_results_csv)
        print(f"  phase9_results.csv: {len(phase9_df)} rows")

    print("\nFigure 1: Layer-wise Spearman")
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
    fig6_encoder_vs_decoder_gap(df, out_dir, args.dpi, phase9_df=phase9_df)

    print("Figure 7: Contingency results")
    fig7_contingency(args.contingency_csv, out_dir, args.dpi)

    print("Figure 8: The Dip phenomenon")
    fig8_dip_phenomenon(df, out_dir, args.dpi)

    print("Figure 9: Summary table")
    fig9_summary_table(df, args.diagnostics_csv, args.contingency_csv, out_dir, args.dpi)

    print("Figure 10: Bias analysis")
    fig10_bias_analysis(args.bias_csv, out_dir, args.dpi)

    print("Figure 11: Scaling comparison")
    fig11_scaling_comparison(df, phase9_df, out_dir, args.dpi)

    print("Figure 12: Per-language bias breakdown")
    fig12_bias_per_language(args.bias_csv, out_dir, args.dpi)

    print("Figure 13: Contingency scaling")
    fig13_contingency_scaling(args.contingency_csv, args.phase9_contingency_csv,
                               out_dir, args.dpi)

    print("\nAll figures saved to:", out_dir)


if __name__ == "__main__":
    main()
