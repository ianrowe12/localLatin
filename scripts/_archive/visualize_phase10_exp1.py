"""Phase 10 Experiment 1: Visualization script.

Generates figures for both the STS (Phase 8 extension) and Assignment
(Phase 9 extension) analyses across all 5 models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

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

# Friendly short names and colours
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
    "sif_abtt_fixed": "#e6550d",
    "sif_abtt_optimal": "#d62728",
    "whitening": "#6baed6",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "sif_only": "SIF only",
    "sif_abtt_fixed": "SIF+ABTT (D=10)",
    "sif_abtt_optimal": "SIF+ABTT (opt D)",
    "whitening": "Whitening",
}
ARCH_LABELS = {
    "bowphs/LaTa": "T5 (Seq2Seq)",
    "bowphs/PhilTa": "T5 (Seq2Seq)",
    "sentence-transformers/LaBSE": "BERT (Encoder)",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen (Decoder)",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "Gemma (Decoder)",
}


def sn(model: str) -> str:
    return SHORT.get(model, model)


# ═══════════════════════════════════════════════════════════════════════
# STS Figures (Phase 8 canon sweep)
# ═══════════════════════════════════════════════════════════════════════

def fig_sts_1_dip(p8: pd.DataFrame, out: Path, dpi: int):
    """Layer-wise Acc@1 for hidden/mean: baseline vs ABTT, all 5 models."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    models = list(SHORT.keys())
    for ax, m in zip(axes, models):
        for method, ls, lw in [("baseline", "--", 1.5), ("abtt", "-", 2.2)]:
            sub = p8[(p8["model"] == m) & (p8["repr"] == "hidden")
                     & (p8["pooling"] == "mean") & (p8["method"] == method)]
            if sub.empty:
                continue
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub["acc@1_all"],
                    ls=ls, lw=lw, color=COLORS[sn(m)],
                    marker="o", ms=3, label=method.upper())
        ax.set_title(sn(m), fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Acc@1 (all queries)")
    fig.suptitle("Fig 1: The Anisotropy Dip — Hidden-State Acc@1 by Layer", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig1_sts_dip_hidden.png", dpi=dpi)
    fig.savefig(out / "fig1_sts_dip_hidden.pdf")
    plt.close(fig)
    print(f"  Saved fig1_sts_dip_hidden")


def fig_sts_2_dip_ffn(p8: pd.DataFrame, out: Path, dpi: int):
    """Layer-wise Acc@1 for FFN representations: baseline vs ABTT."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    models = list(SHORT.keys())
    for ax, m in zip(axes, models):
        repr_name = "ff1" if "bowphs" in m else "ffn_intermediate"
        for method, ls, lw in [("baseline", "--", 1.5), ("abtt", "-", 2.2)]:
            sub = p8[(p8["model"] == m) & (p8["repr"] == repr_name)
                     & (p8["pooling"] == "mean") & (p8["method"] == method)]
            if sub.empty:
                continue
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub["acc@1_all"],
                    ls=ls, lw=lw, color=COLORS[sn(m)],
                    marker="o", ms=3, label=method.upper())
        ax.set_title(sn(m), fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Acc@1 (all queries)")
    fig.suptitle("Fig 2: FFN Representations — Acc@1 by Layer", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig2_sts_dip_ffn.png", dpi=dpi)
    fig.savefig(out / "fig2_sts_dip_ffn.pdf")
    plt.close(fig)
    print(f"  Saved fig2_sts_dip_ffn")


def fig_sts_3_best_bar(p8: pd.DataFrame, out: Path, dpi: int):
    """Grouped bar chart: best Acc@1/3/5 per model (ABTT only)."""
    models = list(SHORT.keys())
    abtt = p8[p8["method"] == "abtt"]
    data = []
    for m in models:
        sub = abtt[abtt["model"] == m]
        best_row = sub.loc[sub["acc@1_all"].idxmax()]
        data.append({
            "model": sn(m),
            "acc@1": best_row["acc@1_all"],
            "acc@3": best_row["acc@3_all"],
            "acc@5": best_row["acc@5_all"],
            "config": f"{best_row['repr']} L{int(best_row['layer'])}",
        })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.25
    bars1 = ax.bar(x - w, df["acc@1"], w, label="Acc@1", color="#2166ac")
    bars2 = ax.bar(x, df["acc@3"], w, label="Acc@3", color="#67a9cf")
    bars3 = ax.bar(x + w, df["acc@5"], w, label="Acc@5", color="#d1e5f0")

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['model']}\n({r['config']})" for _, r in df.iterrows()], fontsize=10)
    ax.set_ylim(0.88, 0.98)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
    ax.legend()
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                    f"{h:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Fig 3: Best STS Retrieval Accuracy per Model (with ABTT)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig3_sts_best_bar.png", dpi=dpi)
    fig.savefig(out / "fig3_sts_best_bar.pdf")
    plt.close(fig)
    print(f"  Saved fig3_sts_best_bar")


def fig_sts_4_gap_profile(p8: pd.DataFrame, out: Path, dpi: int):
    """Cosine gap: baseline vs ABTT side-by-side, all 5 models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    models = list(SHORT.keys())
    for ax, (method, title) in zip(axes, [
        ("baseline", "(A) Baseline — No Correction"),
        ("abtt", "(B) With ABTT"),
    ]):
        for m in models:
            sub = p8[(p8["model"] == m) & (p8["repr"] == "hidden")
                     & (p8["pooling"] == "mean") & (p8["method"] == method)]
            if sub.empty:
                continue
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub["gap"], marker="o", ms=4,
                    label=sn(m), color=COLORS[sn(m)], lw=2)
        ax.set_xlabel("Layer")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cosine Gap (same - diff)")
    axes[0].set_ylim(-0.05, 0.65)
    fig.suptitle("Fig 4: Cosine Gap by Layer — Baseline vs ABTT", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig4_sts_gap.png", dpi=dpi)
    fig.savefig(out / "fig4_sts_gap.pdf")
    plt.close(fig)
    print(f"  Saved fig4_sts_gap")


def fig_sts_5_architecture_comparison(p8: pd.DataFrame, out: Path, dpi: int):
    """Baseline dip magnitude by architecture type."""
    models = list(SHORT.keys())
    archs = [ARCH_LABELS[m] for m in models]
    baseline = p8[(p8["method"] == "baseline") & (p8["repr"] == "hidden")
                  & (p8["pooling"] == "mean")]
    drops = []
    bests = []
    worsts = []
    for m in models:
        sub = baseline[baseline["model"] == m]
        if sub.empty:
            drops.append(0)
            bests.append(0)
            worsts.append(0)
            continue
        best = sub["acc@1_all"].max()
        worst = sub["acc@1_all"].min()
        drops.append(best - worst)
        bests.append(best)
        worsts.append(worst)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    ax.bar(x, bests, 0.35, label="Best layer", color=[COLORS[sn(m)] for m in models], alpha=0.8)
    ax.bar(x, worsts, 0.35, label="Worst layer", color=[COLORS[sn(m)] for m in models], alpha=0.3)
    for i, (b, w) in enumerate(zip(bests, worsts)):
        ax.annotate("", xy=(i, w), xytext=(i, b),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
        ax.text(i + 0.15, (b + w) / 2, f"{b-w:.0%}", va="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{sn(m)}\n({archs[i]})" for i, m in enumerate(models)], fontsize=10)
    ax.set_ylabel("Acc@1 (baseline)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
    ax.set_title("Fig 5: Anisotropy Dip Severity by Architecture", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig5_sts_dip_severity.png", dpi=dpi)
    fig.savefig(out / "fig5_sts_dip_severity.pdf")
    plt.close(fig)
    print(f"  Saved fig5_sts_dip_severity")


# ═══════════════════════════════════════════════════════════════════════
# Assignment Figures (Phase 10 Experiment 1)
# ═══════════════════════════════════════════════════════════════════════

def fig_assign_1_dip(p10: pd.DataFrame, out: Path, dpi: int):
    """AUCROC dip: baseline vs SIF vs SIF+ABTT vs whitening, all 5 models."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    models = list(SHORT.keys())
    method_styles = [
        ("baseline",         "--", 1.5, "#bdbdbd", "Baseline"),
        ("sif_only",         "-.", 1.5, "#fdae6b", "SIF only"),
        ("whitening",        ":",  1.8, "#6baed6", "Whitening"),
        ("sif_abtt_optimal", "-",  2.2, "#d62728", "SIF+ABTT (opt)"),
    ]
    for ax, m in zip(axes, models):
        for method, ls, lw, color, label in method_styles:
            sub = p10[(p10["model"] == m) & (p10["method"] == method)
                      & (p10["repr"] == "hidden")]
            if sub.empty:
                continue
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub["aucroc"],
                    ls=ls, lw=lw, color=color,
                    marker="o", ms=2, label=label)
        ax.set_title(sn(m), fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("AUCROC")
    fig.suptitle("Fig 1: AUCROC by Layer — All Methods Compared (Assignment Task)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig1_assign_dip.png", dpi=dpi)
    fig.savefig(out / "fig1_assign_dip.pdf")
    plt.close(fig)
    print(f"  Saved fig1_assign_dip")


def fig_assign_2_method_heatmap(p10: pd.DataFrame, out: Path, dpi: int):
    """Heatmap: best Assignment Accuracy per (model x method)."""
    models = list(SHORT.keys())
    methods = ["baseline", "sif_only", "whitening", "sif_abtt_fixed", "sif_abtt_optimal"]

    data = np.zeros((len(models), len(methods)))
    for i, m in enumerate(models):
        for j, method in enumerate(methods):
            sub = p10[(p10["model"] == m) & (p10["method"] == method)]
            if not sub.empty:
                data[i, j] = sub["overall_assignment_acc"].max()

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.45, vmax=0.92, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([sn(m) for m in models])
    for i in range(len(models)):
        for j in range(len(methods)):
            ax.text(j, i, f"{data[i,j]:.1%}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if data[i, j] < 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Best Assignment Accuracy")
    ax.set_title("Fig 2: Best Assignment Accuracy per Model and Method", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig2_assign_method_heatmap.png", dpi=dpi)
    fig.savefig(out / "fig2_assign_method_heatmap.pdf")
    plt.close(fig)
    print(f"  Saved fig2_assign_method_heatmap")


def fig_assign_3_assignment_accuracy(p10: pd.DataFrame, out: Path, dpi: int):
    """Method comparison: best assignment accuracy per method, averaged across models."""
    methods = ["baseline", "sif_only", "whitening", "sif_abtt_fixed", "sif_abtt_optimal"]
    models = list(SHORT.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Panel A: Best overall assignment accuracy per (model x method) -- grouped bar
    ax = axes[0]
    x = np.arange(len(models))
    n_methods = len(methods)
    w = 0.8 / n_methods
    for j, method in enumerate(methods):
        vals = []
        for m in models:
            sub = p10[(p10["model"] == m) & (p10["method"] == method)]
            vals.append(sub["overall_assignment_acc"].max() if len(sub) else 0)
        offset = (j - n_methods / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([sn(m) for m in models], fontsize=9)
    ax.set_ylabel("Assignment Accuracy")
    ax.set_ylim(0.45, 0.95)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(A) Best Assignment Accuracy by Method", fontweight="bold")

    # Panel B: existing vs new breakdown for best config per model (all methods)
    ax = axes[1]
    data = []
    for m in models:
        sub = p10[p10["model"] == m]
        best = sub.loc[sub["overall_assignment_acc"].idxmax()]
        data.append({
            "model": sn(m),
            "existing": best["existing_acc"],
            "new": best["new_acc"],
            "overall": best["overall_assignment_acc"],
            "method": METHOD_LABELS.get(best["method"], best["method"]),
        })
    df = pd.DataFrame(data)
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["existing"], w, label="Existing files", color="#2166ac")
    ax.bar(x, df["new"], w, label="New files", color="#b2182b")
    ax.bar(x + w, df["overall"], w, label="Overall", color="#4d4d4d")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['model']}\n({r['method']})" for _, r in df.iterrows()], fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.7, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(B) Existing vs New Breakdown (Best Config)", fontweight="bold")

    fig.suptitle("Fig 3: Assignment Accuracy — Method Comparison", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig3_assign_accuracy.png", dpi=dpi)
    fig.savefig(out / "fig3_assign_accuracy.pdf")
    plt.close(fig)
    print(f"  Saved fig3_assign_accuracy")


def fig_assign_4_gap(p10: pd.DataFrame, out: Path, dpi: int):
    """Cosine gap across layers: all methods compared."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    models = list(SHORT.keys())
    method_styles = [
        ("baseline",         "--", 1.5, "#bdbdbd", "Baseline"),
        ("sif_only",         "-.", 1.5, "#fdae6b", "SIF only"),
        ("whitening",        ":",  1.8, "#6baed6", "Whitening"),
        ("sif_abtt_optimal", "-",  2.2, "#d62728", "SIF+ABTT"),
    ]
    for ax, m in zip(axes, models):
        for method, ls, lw, color, label in method_styles:
            sub = p10[(p10["model"] == m) & (p10["method"] == method)
                      & (p10["repr"] == "hidden")]
            if sub.empty:
                continue
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub["gap"], ls=ls, lw=lw, color=color,
                    marker="o", ms=2, label=label)
        ax.set_title(sn(m), fontweight="bold")
        ax.set_xlabel("Layer")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("Cosine Gap (same - diff)")
    fig.suptitle("Fig 4: Cosine Gap by Layer — All Methods (Assignment Task)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig4_assign_gap.png", dpi=dpi)
    fig.savefig(out / "fig4_assign_gap.pdf")
    plt.close(fig)
    print(f"  Saved fig4_assign_gap")


def fig_assign_5_whitening_failure(p10: pd.DataFrame, out: Path, dpi: int):
    """Existing vs New breakdown: SIF vs Whitening vs SIF+ABTT (4 panels)."""
    models = list(SHORT.keys())
    methods = ["baseline", "sif_only", "whitening", "sif_abtt_optimal"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, method in zip(axes, methods):
        exist_vals = []
        new_vals = []
        for m in models:
            sub = p10[(p10["model"] == m) & (p10["method"] == method)]
            if sub.empty:
                exist_vals.append(0)
                new_vals.append(0)
                continue
            best = sub.loc[sub["overall_assignment_acc"].idxmax()]
            exist_vals.append(best["existing_acc"])
            new_vals.append(best["new_acc"])

        x = np.arange(len(models))
        ax.bar(x - 0.17, exist_vals, 0.34, label="Existing", color="#2166ac")
        ax.bar(x + 0.17, new_vals, 0.34, label="New", color="#b2182b")
        ax.set_xticks(x)
        ax.set_xticklabels([sn(m) for m in models], rotation=25, ha="right", fontsize=8)
        ax.set_title(METHOD_LABELS.get(method, method), fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
        ax.legend(fontsize=8)
    fig.suptitle("Fig 5: Existing vs New Accuracy — Method Comparison",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig5_assign_whitening.png", dpi=dpi)
    fig.savefig(out / "fig5_assign_whitening.pdf")
    plt.close(fig)
    print(f"  Saved fig5_assign_whitening")


def fig_assign_6_optimal_d(p10: pd.DataFrame, out: Path, dpi: int):
    """Optimal D heatmap per model and layer (sif_abtt_optimal, hidden only)."""
    opt = p10[(p10["method"] == "sif_abtt_optimal") & (p10["repr"] == "hidden")]
    models = list(SHORT.keys())

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
    for ax, m in zip(axes, models):
        sub = opt[opt["model"] == m].sort_values("layer")
        if sub.empty:
            continue
        ax.bar(sub["layer"], sub["D"], color=COLORS[sn(m)], alpha=0.8)
        ax.set_title(sn(m), fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylim(0, 11)
    axes[0].set_ylabel("Optimal D")
    fig.suptitle("Fig 6: Optimal D by Layer (Assignment Task, Hidden)", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig6_assign_optimal_d.png", dpi=dpi)
    fig.savefig(out / "fig6_assign_optimal_d.pdf")
    plt.close(fig)
    print(f"  Saved fig6_assign_optimal_d")


def fig_assign_7_summary_table(p10: pd.DataFrame, out: Path, dpi: int):
    """Visual summary table: best result per model for AUCROC, Acc@1, Assignment."""
    models = list(SHORT.keys())
    rows = []
    for m in models:
        sub = p10[p10["model"] == m]
        best_auc = sub.loc[sub["aucroc"].idxmax()]
        best_acc = sub.loc[sub["acc_at_1"].idxmax()]
        best_assign = sub.loc[sub["overall_assignment_acc"].idxmax()]
        rows.append([
            sn(m),
            f"{best_auc['aucroc']:.4f}",
            f"{best_auc['repr']} L{int(best_auc['layer'])}",
            f"{best_acc['acc_at_1']:.4f}",
            f"{best_acc['repr']} L{int(best_acc['layer'])}",
            f"{best_assign['overall_assignment_acc']:.4f}",
            f"{best_assign['repr']} L{int(best_assign['layer'])}",
        ])

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")
    col_labels = ["Model", "AUCROC", "Config", "Acc@1", "Config", "Assignment", "Config"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2166ac")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight best per metric
    auc_vals = [float(r[1]) for r in rows]
    acc_vals = [float(r[3]) for r in rows]
    assign_vals = [float(r[5]) for r in rows]
    for vals, col_idx in [(auc_vals, 1), (acc_vals, 3), (assign_vals, 5)]:
        best_i = np.argmax(vals)
        table[best_i + 1, col_idx].set_facecolor("#d4edda")

    ax.set_title("Fig 7: Summary — Best Result per Model (Assignment Task)",
                 fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out / "fig7_assign_summary.png", dpi=dpi)
    fig.savefig(out / "fig7_assign_summary.pdf")
    plt.close(fig)
    print(f"  Saved fig7_assign_summary")


def fig_assign_8_acc1_layerwise(p10: pd.DataFrame, out: Path, dpi: int):
    """Acc@1 by layer, SIF+ABTT optimal, all models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = list(SHORT.keys())
    opt = p10[(p10["method"] == "sif_abtt_fixed") & (p10["repr"] == "hidden")]
    for m in models:
        sub = opt[opt["model"] == m].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["acc_at_1"], marker="o", ms=4,
                label=sn(m), color=COLORS[sn(m)], lw=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Acc@1")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
    ax.set_title("Fig 8: Retrieval Acc@1 by Layer (SIF+ABTT, Hidden)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig8_assign_acc1.png", dpi=dpi)
    fig.savefig(out / "fig8_assign_acc1.pdf")
    plt.close(fig)
    print(f"  Saved fig8_assign_acc1")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p8_csv", default="runs/phase8_results/phase8_canon_sweep.csv")
    parser.add_argument("--p10_csv", default="runs/phase9/phase10_experiment1_results.csv")
    parser.add_argument("--out_dir", default="runs/phase10/experiment1/figures")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p8 = pd.read_csv(args.p8_csv)
    p10 = pd.read_csv(args.p10_csv)

    print("Generating STS figures...")
    fig_sts_1_dip(p8, out, args.dpi)
    fig_sts_2_dip_ffn(p8, out, args.dpi)
    fig_sts_3_best_bar(p8, out, args.dpi)
    fig_sts_4_gap_profile(p8, out, args.dpi)
    fig_sts_5_architecture_comparison(p8, out, args.dpi)

    print("Generating Assignment figures...")
    fig_assign_1_dip(p10, out, args.dpi)
    fig_assign_2_method_heatmap(p10, out, args.dpi)
    fig_assign_3_assignment_accuracy(p10, out, args.dpi)
    fig_assign_4_gap(p10, out, args.dpi)
    fig_assign_5_whitening_failure(p10, out, args.dpi)
    fig_assign_6_optimal_d(p10, out, args.dpi)
    fig_assign_7_summary_table(p10, out, args.dpi)
    fig_assign_8_acc1_layerwise(p10, out, args.dpi)

    print(f"\nAll figures saved to: {out}")


if __name__ == "__main__":
    main()
