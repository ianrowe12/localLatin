"""Generate Phase 8 Experiment 2 (MUSTS) analysis plots."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

OUT = Path("runs/phase8_results/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("runs/phase8_results/phase8_musts_sweep.csv")

# Consistent style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

MODEL_LABELS = {
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen2-7B": "Qwen2-7B",
    "google/gemma-7b": "Gemma-7B",
}

LANG_COLORS = {
    "english": "#377eb8",
    "french": "#e41a1c",
    "sinhala": "#4daf4a",
    "tamil": "#ff7f00",
}

METHOD_COLORS = {
    "baseline_mean": "#888888",
    "sif_only": "#4daf4a",
    "sif_abtt_fixed": "#377eb8",
    "sif_abtt_optimal": "#e41a1c",
    "whitening": "#984ea3",
    "variance_filter": "#ff7f00",
}

METHOD_LABELS = {
    "baseline_mean": "Baseline (mean)",
    "sif_only": "SIF only",
    "sif_abtt_fixed": "SIF+ABTT (D=10)",
    "sif_abtt_optimal": "SIF+ABTT (optimal D)",
    "whitening": "Whitening",
    "variance_filter": "Variance filter",
}

LANGUAGES = ["english", "french", "sinhala", "tamil"]
MODELS = list(MODEL_LABELS.keys())


# ============================================================
# FIGURE 8: Layer-wise Spearman profiles (baseline vs SIF+ABTT)
# 3 columns (models) x 4 rows (languages)
# ============================================================
fig, axes = plt.subplots(4, 3, figsize=(16, 16), sharey="row")

for col, model in enumerate(MODELS):
    mdf = df[df["model"] == model]
    for row, lang in enumerate(LANGUAGES):
        ax = axes[row, col]
        ldf = mdf[mdf["language"] == lang]

        for method, label, color, ls, marker, lw in [
            ("baseline_mean", "Baseline (mean)", "#888888", "-", "o", 1.5),
            ("sif_only", "SIF only", "#4daf4a", "-", "s", 1.5),
            ("sif_abtt_optimal", "SIF+ABTT (opt D)", "#e41a1c", "--", "D", 2.0),
            ("whitening", "Whitening", "#984ea3", ":", "^", 1.5),
        ]:
            s = ldf[ldf["method"] == method].sort_values("layer")
            if not s.empty:
                ax.plot(s["layer"], s["spearman_test"], marker=marker, markersize=3,
                        color=color, linestyle=ls, linewidth=lw, label=label)

        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)

        if row == 0:
            ax.set_title(MODEL_LABELS[model], fontweight="bold", fontsize=13)
        if col == 0:
            ax.set_ylabel(f"{lang.capitalize()}\nSpearman (test)")
        if row == len(LANGUAGES) - 1:
            ax.set_xlabel("Layer")

        # Only show legend on top-right panel
        if row == 0 and col == 2:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

fig.suptitle("Figure 8: Layer-Wise Spearman Profiles — Baseline vs SIF+ABTT",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "fig8_musts_layer_profiles.png", dpi=300)
plt.close(fig)
print("Saved fig8_musts_layer_profiles.png")


# ============================================================
# FIGURE 9: Method comparison heatmap
# Best test Spearman per model x language for each method
# ============================================================
methods_order = ["baseline_mean", "sif_only", "sif_abtt_fixed",
                 "sif_abtt_optimal", "whitening", "variance_filter"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax_idx, model in enumerate(MODELS):
    ax = axes[ax_idx]
    mdf = df[df["model"] == model]

    matrix = np.zeros((len(LANGUAGES), len(methods_order)))
    for i, lang in enumerate(LANGUAGES):
        for j, method in enumerate(methods_order):
            vals = mdf[(mdf["language"] == lang) & (mdf["method"] == method)]["spearman_test"]
            matrix[i, j] = vals.max() if not vals.empty else np.nan

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(len(LANGUAGES)):
        for j in range(len(methods_order)):
            val = matrix[i, j]
            color = "white" if val < 0.35 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(methods_order)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods_order], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(LANGUAGES)))
    ax.set_yticklabels([l.capitalize() for l in LANGUAGES])
    ax.set_title(MODEL_LABELS[model], fontweight="bold")

fig.colorbar(im, ax=axes, shrink=0.8, label="Best Test Spearman")
fig.suptitle("Figure 9: Best Test Spearman by Method (Best Layer)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 0.92, 0.94])
fig.savefig(OUT / "fig9_method_comparison.png", dpi=300)
plt.close(fig)
print("Saved fig9_method_comparison.png")


# ============================================================
# FIGURE 10: Cross-model comparison — peak Spearman per language
# Grouped bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(LANGUAGES))
width = 0.22
model_colors = ["#377eb8", "#e41a1c", "#4daf4a"]

for i, model in enumerate(MODELS):
    mdf = df[(df["model"] == model) & (df["method"] == "sif_abtt_optimal")]
    peaks = []
    for lang in LANGUAGES:
        vals = mdf[mdf["language"] == lang]["spearman_test"]
        peaks.append(vals.max() if not vals.empty else 0)

    bars = ax.bar(x + i * width - width, peaks, width,
                  label=MODEL_LABELS[model], color=model_colors[i], alpha=0.85)
    for j, val in enumerate(peaks):
        ax.text(x[j] + i * width - width, val + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([l.capitalize() for l in LANGUAGES], fontsize=12)
ax.set_ylabel("Peak Test Spearman (SIF+ABTT optimal)")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.2, axis="y")

fig.suptitle("Figure 10: Best Achievable Spearman per Language (SIF+ABTT)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig10_best_spearman_comparison.png", dpi=300)
plt.close(fig)
print("Saved fig10_best_spearman_comparison.png")


# ============================================================
# FIGURE 11: SIF+ABTT lift over baseline at best layer
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

groups = []
for model in MODELS:
    for lang in LANGUAGES:
        base_all = df[(df["model"] == model) & (df["language"] == lang) &
                       (df["method"] == "baseline_mean")]
        abtt_all = df[(df["model"] == model) & (df["language"] == lang) &
                       (df["method"] == "sif_abtt_optimal")]
        if base_all.empty or abtt_all.empty:
            continue

        # Find the best ABTT layer, then get the baseline at that same layer
        best_abtt_row = abtt_all.loc[abtt_all["spearman_test"].idxmax()]
        best_layer = int(best_abtt_row["layer"])
        base_at_layer = base_all[base_all["layer"] == best_layer]["spearman_test"].values
        base_val = base_at_layer[0] if len(base_at_layer) > 0 else 0

        slug = MODEL_LABELS[model]
        groups.append({
            "label": f"{slug}\n{lang.capitalize()}",
            "baseline": base_val,
            "abtt": best_abtt_row["spearman_test"],
            "lift": best_abtt_row["spearman_test"] - base_val,
            "layer": best_layer,
        })

gdf = pd.DataFrame(groups)
x_pos = np.arange(len(gdf))
w = 0.35

bars1 = ax.bar(x_pos - w/2, gdf["baseline"], w, label="Baseline (mean)", color="#888888", alpha=0.85)
bars2 = ax.bar(x_pos + w/2, gdf["abtt"], w, label="SIF+ABTT (optimal D)", color="#e41a1c", alpha=0.85)

for i, row in gdf.iterrows():
    lift = row["lift"]
    sign = "+" if lift >= 0 else ""
    ax.annotate(f"{sign}{lift:.2f}\n(L{int(row['layer'])})",
                xy=(i + w/2, row["abtt"]),
                ha="center", va="bottom", fontsize=7, fontweight="bold", color="#e41a1c")

ax.set_xticks(x_pos)
ax.set_xticklabels(gdf["label"], fontsize=8)
ax.set_ylabel("Spearman (test)")
ax.set_ylim(-0.1, 1.1)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.2, axis="y")
ax.axhline(y=0, color="black", linewidth=0.5)

fig.suptitle("Figure 11: SIF+ABTT Lift Over Baseline (at Best ABTT Layer)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig11_lift_bar_chart.png", dpi=300)
plt.close(fig)
print("Saved fig11_lift_bar_chart.png")


# ============================================================
# FIGURE 12: Optimal D heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))

opt = df[df["method"] == "sif_abtt_optimal"]
d_matrix = np.zeros((len(MODELS), len(LANGUAGES)))

for i, model in enumerate(MODELS):
    for j, lang in enumerate(LANGUAGES):
        sub = opt[(opt["model"] == model) & (opt["language"] == lang)]
        if not sub.empty:
            best = sub.loc[sub["spearman_test"].idxmax()]
            d_matrix[i, j] = best["D"]

im = ax.imshow(d_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)

for i in range(len(MODELS)):
    for j in range(len(LANGUAGES)):
        ax.text(j, i, f"D={int(d_matrix[i, j])}", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if d_matrix[i, j] > 6 else "black")

ax.set_xticks(range(len(LANGUAGES)))
ax.set_xticklabels([l.capitalize() for l in LANGUAGES], fontsize=11)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8, label="Optimal D (# PCs removed)")

fig.suptitle("Figure 12: Optimal D at Best Layer (Train-Fitted)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig12_optimal_D_heatmap.png", dpi=300)
plt.close(fig)
print("Saved fig12_optimal_D_heatmap.png")


# ============================================================
# FIGURE 13: Whitening vs ABTT — scatter plot
# Each dot = one (model, language, layer) setting
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

for ax_idx, model in enumerate(MODELS):
    ax = axes[ax_idx]
    mdf = df[df["model"] == model]

    for lang in LANGUAGES:
        ldf = mdf[mdf["language"] == lang]
        wh = ldf[ldf["method"] == "whitening"].sort_values("layer")["spearman_test"].values
        ab = ldf[ldf["method"] == "sif_abtt_optimal"].sort_values("layer")["spearman_test"].values
        n = min(len(wh), len(ab))
        if n > 0:
            ax.scatter(wh[:n], ab[:n], s=25, alpha=0.6,
                      color=LANG_COLORS[lang], label=lang.capitalize(), edgecolors="none")

    # Diagonal line
    ax.plot([-0.2, 1.0], [-0.2, 1.0], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Whitening (Spearman)")
    if ax_idx == 0:
        ax.set_ylabel("SIF+ABTT (Spearman)")
    ax.set_title(MODEL_LABELS[model], fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.15, 1.0)
    ax.set_ylim(-0.15, 1.0)
    ax.set_aspect("equal")

    # Count wins
    all_wh = []
    all_ab = []
    for lang in LANGUAGES:
        ldf = mdf[mdf["language"] == lang]
        w = ldf[ldf["method"] == "whitening"].sort_values("layer")["spearman_test"].values
        a = ldf[ldf["method"] == "sif_abtt_optimal"].sort_values("layer")["spearman_test"].values
        n = min(len(w), len(a))
        all_wh.extend(w[:n])
        all_ab.extend(a[:n])
    wins_abtt = sum(a > w for a, w in zip(all_ab, all_wh))
    total = len(all_wh)
    ax.text(0.05, 0.95, f"ABTT wins: {wins_abtt}/{total}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    if ax_idx == 2:
        ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

fig.suptitle("Figure 13: Whitening vs SIF+ABTT (Each Dot = One Layer)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig13_whitening_vs_abtt.png", dpi=300)
plt.close(fig)
print("Saved fig13_whitening_vs_abtt.png")


print(f"\nAll MUSTS plots saved to: {OUT}")
