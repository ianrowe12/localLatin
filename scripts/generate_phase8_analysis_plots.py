"""Generate all Phase 8 analysis plots."""
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

df = pd.read_csv("runs/phase8_results/phase8_canon_sweep.csv")

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

COLORS = {
    "baseline_mean": "#888888",
    "baseline_sif": "#4daf4a",
    "abtt_mean": "#377eb8",
    "abtt_sif": "#e41a1c",
}

# ============================================================
# PLOT 1: The Middle-Layer Dip (LaTa & PhilTa hidden states)
# Shows baseline collapses in middle layers, ABTT rescues it
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, model, title in zip(axes, ["bowphs/LaTa", "bowphs/PhilTa"], ["LaTa", "PhilTa"]):
    mdf = df[(df["model"] == model) & (df["repr"] == "hidden")]

    for pooling, method, label, color, ls, marker in [
        ("mean", "baseline", "Baseline (mean pool)", COLORS["baseline_mean"], "-", "o"),
        ("sif",  "baseline", "SIF pooling only",     COLORS["baseline_sif"], "-", "s"),
        ("mean", "abtt",     "ABTT (D=10, mean)",    COLORS["abtt_mean"], "--", "^"),
        ("sif",  "abtt",     "SIF + ABTT (D=10)",    COLORS["abtt_sif"], "--", "D"),
    ]:
        s = mdf[(mdf["pooling"] == pooling) & (mdf["method"] == method)].sort_values("layer")
        ax.plot(s["layer"], s["acc@1_winnable"], marker=marker, markersize=5,
                color=color, linestyle=ls, linewidth=1.8, label=label)

    ax.set_xlabel("Layer")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(0, 13))
    ax.grid(True, alpha=0.25)
    ax.axhline(y=0.1, color="gray", linestyle=":", alpha=0.4)

axes[0].set_ylabel("Acc@1 (winnable test queries)")
axes[1].legend(loc="lower right", framealpha=0.9)
fig.suptitle("Figure 1: The Middle-Layer Performance Dip (Hidden States)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig1_middle_layer_dip.png", dpi=300)
plt.close(fig)
print("Saved fig1_middle_layer_dip.png")


# ============================================================
# PLOT 2: FF1 representations — same story but even more dramatic
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, model, title in zip(axes, ["bowphs/LaTa", "bowphs/PhilTa"], ["LaTa", "PhilTa"]):
    mdf = df[(df["model"] == model) & (df["repr"] == "ff1")]

    for pooling, method, label, color, ls, marker in [
        ("mean", "baseline", "Baseline (mean pool)", COLORS["baseline_mean"], "-", "o"),
        ("sif",  "baseline", "SIF pooling only",     COLORS["baseline_sif"], "-", "s"),
        ("mean", "abtt",     "ABTT (D=10, mean)",    COLORS["abtt_mean"], "--", "^"),
        ("sif",  "abtt",     "SIF + ABTT (D=10)",    COLORS["abtt_sif"], "--", "D"),
    ]:
        s = mdf[(mdf["pooling"] == pooling) & (mdf["method"] == method)].sort_values("layer")
        ax.plot(s["layer"], s["acc@1_winnable"], marker=marker, markersize=5,
                color=color, linestyle=ls, linewidth=1.8, label=label)

    ax.set_xlabel("Layer")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("Acc@1 (winnable test queries)")
axes[1].legend(loc="lower right", framealpha=0.9)
fig.suptitle("Figure 2: FF1 (Post-Activation) Representations", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig2_ff1_representations.png", dpi=300)
plt.close(fig)
print("Saved fig2_ff1_representations.png")


# ============================================================
# PLOT 3: Anisotropy — off-diagonal mean across layers
# Shows WHY the dip happens (high anisotropy) and ABTT fixes it
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, model, title in zip(axes, ["bowphs/LaTa", "bowphs/PhilTa"], ["LaTa", "PhilTa"]):
    mdf = df[(df["model"] == model) & (df["repr"] == "hidden")]

    for pooling, method, label, color, ls in [
        ("mean", "baseline", "Baseline (mean)", "#888888", "-"),
        ("mean", "abtt",     "After ABTT (mean)", "#377eb8", "--"),
        ("sif",  "baseline", "SIF pooling", "#4daf4a", "-"),
        ("sif",  "abtt",     "SIF + ABTT", "#e41a1c", "--"),
    ]:
        s = mdf[(mdf["pooling"] == pooling) & (mdf["method"] == method)].sort_values("layer")
        ax.plot(s["layer"], s["off_diag_mean"], marker="o", markersize=4,
                color=color, linestyle=ls, linewidth=1.5, label=label)

    ax.set_xlabel("Layer")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(0, 13))
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("Off-diagonal mean (anisotropy proxy)")
axes[1].legend(loc="best", framealpha=0.9)
fig.suptitle("Figure 3: Embedding Anisotropy Across Layers", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig3_anisotropy.png", dpi=300)
plt.close(fig)
print("Saved fig3_anisotropy.png")


# ============================================================
# PLOT 4: LaBSE layer-wise results (hidden + FFN intermediate)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

labse = df[df["model"] == "sentence-transformers/LaBSE"]

# Hidden states
ax = axes[0]
for pooling, method, label, color, ls, marker in [
    ("mean", "baseline", "Baseline (mean)", COLORS["baseline_mean"], "-", "o"),
    ("sif",  "baseline", "SIF only",        COLORS["baseline_sif"], "-", "s"),
    ("mean", "abtt",     "ABTT (mean)",     COLORS["abtt_mean"], "--", "^"),
    ("sif",  "abtt",     "SIF + ABTT",      COLORS["abtt_sif"], "--", "D"),
]:
    s = labse[(labse["repr"] == "hidden") & (labse["pooling"] == pooling) &
              (labse["method"] == method)].sort_values("layer")
    if not s.empty:
        ax.plot(s["layer"], s["acc@1_winnable"], marker=marker, markersize=5,
                color=color, linestyle=ls, linewidth=1.8, label=label)
ax.set_xlabel("Layer")
ax.set_ylabel("Acc@1 (winnable)")
ax.set_title("Hidden States", fontweight="bold")
ax.set_xticks(range(0, 13))
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", framealpha=0.9)

# FFN intermediate
ax = axes[1]
for pooling, method, label, color, ls, marker in [
    ("mean", "baseline", "Baseline (mean)", COLORS["baseline_mean"], "-", "o"),
    ("sif",  "baseline", "SIF only",        COLORS["baseline_sif"], "-", "s"),
    ("mean", "abtt",     "ABTT (mean)",     COLORS["abtt_mean"], "--", "^"),
    ("sif",  "abtt",     "SIF + ABTT",      COLORS["abtt_sif"], "--", "D"),
]:
    s = labse[(labse["repr"] == "ffn_intermediate") & (labse["pooling"] == pooling) &
              (labse["method"] == method)].sort_values("layer")
    if not s.empty:
        ax.plot(s["layer"], s["acc@1_winnable"], marker=marker, markersize=5,
                color=color, linestyle=ls, linewidth=1.8, label=label)
ax.set_xlabel("Layer")
ax.set_title("FFN Intermediate", fontweight="bold")
ax.set_xticks(range(0, 12))
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", framealpha=0.9)

fig.suptitle("Figure 4: LaBSE on Latin Canon", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig4_labse.png", dpi=300)
plt.close(fig)
print("Saved fig4_labse.png")


# ============================================================
# PLOT 5: ABTT lift — how much does ABTT improve over baseline?
# Grouped bar chart: best layer per model/repr/pooling
# ============================================================
groups = []
for model in df["model"].unique():
    for repr_name in df[df["model"] == model]["repr"].unique():
        for pooling in ["mean", "sif"]:
            base = df[(df["model"] == model) & (df["repr"] == repr_name) &
                       (df["pooling"] == pooling) & (df["method"] == "baseline")]
            abtt = df[(df["model"] == model) & (df["repr"] == repr_name) &
                       (df["pooling"] == pooling) & (df["method"] == "abtt")]
            if base.empty or abtt.empty:
                continue
            best_base = base["acc@1_winnable"].max()
            best_abtt = abtt["acc@1_winnable"].max()
            slug = model.split("/")[-1]
            groups.append({
                "label": f"{slug}\n{repr_name}/{pooling}",
                "baseline": best_base,
                "abtt": best_abtt,
                "lift": best_abtt - best_base,
            })

gdf = pd.DataFrame(groups)
x = np.arange(len(gdf))
w = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
bars1 = ax.bar(x - w/2, gdf["baseline"], w, label="Best Baseline", color="#888888", alpha=0.85)
bars2 = ax.bar(x + w/2, gdf["abtt"], w, label="Best ABTT (D=10)", color="#e41a1c", alpha=0.85)

for i, row in gdf.iterrows():
    lift = row["lift"]
    ax.annotate(f"+{lift:.1%}", xy=(i + w/2, row["abtt"]),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#e41a1c")

ax.set_xticks(x)
ax.set_xticklabels(gdf["label"], fontsize=8)
ax.set_ylabel("Best Acc@1 (winnable)")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.2, axis="y")
fig.suptitle("Figure 5: ABTT Improvement Over Baseline (Best Layer)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig5_abtt_lift.png", dpi=300)
plt.close(fig)
print("Saved fig5_abtt_lift.png")


# ============================================================
# PLOT 6: Gap metric (same_avg - diff_avg) across layers
# Shows the separability of same-folder vs different-folder pairs
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

for ax, model, title in zip(axes,
    ["bowphs/LaTa", "bowphs/PhilTa", "sentence-transformers/LaBSE"],
    ["LaTa", "PhilTa", "LaBSE"]):
    mdf = df[(df["model"] == model) & (df["repr"].isin(["hidden"]))]

    for pooling, method, label, color, ls in [
        ("mean", "baseline", "Baseline", "#888888", "-"),
        ("mean", "abtt",     "ABTT", "#377eb8", "--"),
        ("sif",  "abtt",     "SIF + ABTT", "#e41a1c", "--"),
    ]:
        s = mdf[(mdf["pooling"] == pooling) & (mdf["method"] == method)].sort_values("layer")
        if not s.empty:
            ax.plot(s["layer"], s["gap"], marker="o", markersize=4,
                    color=color, linestyle=ls, linewidth=1.5, label=label)

    ax.set_xlabel("Layer")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("Gap (same_avg - diff_avg)")
axes[-1].legend(loc="best", framealpha=0.9)
fig.suptitle("Figure 6: Cosine Similarity Gap Across Layers", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig6_gap.png", dpi=300)
plt.close(fig)
print("Saved fig6_gap.png")


# ============================================================
# PLOT 7: All three models comparison — hidden baseline vs SIF+ABTT
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

model_styles = {
    "bowphs/LaTa": ("#377eb8", "o", "LaTa"),
    "bowphs/PhilTa": ("#4daf4a", "s", "PhilTa"),
    "sentence-transformers/LaBSE": ("#e41a1c", "D", "LaBSE"),
}

for model, (color, marker, label) in model_styles.items():
    # Baseline mean
    s = df[(df["model"] == model) & (df["repr"] == "hidden") &
           (df["pooling"] == "mean") & (df["method"] == "baseline")].sort_values("layer")
    ax.plot(s["layer"], s["acc@1_winnable"], marker=marker, markersize=5,
            color=color, linestyle="-", linewidth=1.2, alpha=0.5, label=f"{label} baseline")

    # SIF + ABTT
    s2 = df[(df["model"] == model) & (df["repr"] == "hidden") &
            (df["pooling"] == "sif") & (df["method"] == "abtt")].sort_values("layer")
    if not s2.empty:
        ax.plot(s2["layer"], s2["acc@1_winnable"], marker=marker, markersize=6,
                color=color, linestyle="--", linewidth=2.0, label=f"{label} SIF+ABTT")

ax.set_xlabel("Layer")
ax.set_ylabel("Acc@1 (winnable)")
ax.set_xticks(range(0, 13))
ax.grid(True, alpha=0.25)
ax.legend(loc="lower left", framealpha=0.9, ncol=2)
fig.suptitle("Figure 7: All Models — Baseline vs SIF+ABTT (Hidden States)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig7_all_models.png", dpi=300)
plt.close(fig)
print("Saved fig7_all_models.png")


print("\nAll plots saved to:", OUT)
