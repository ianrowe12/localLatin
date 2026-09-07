"""Render resubmit figures: AUROC per model, Gap per model, density 2x2.

Combines the release line plots from visualize_phase11_release.py with the
density histogram from package_paper_release_assets.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHORT = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "google/mt5-base": "mT5-base",
}

MAX_LAYERS = {
    "bowphs/LaTa": 12,
    "bowphs/PhilTa": 12,
    "sentence-transformers/LaBSE": 12,
    "Qwen/Qwen3-Embedding-0.6B": 28,
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": 24,
    "google/mt5-base": 12,
}

METHOD_ORDER = ["baseline", "abtt_optimal", "sif_only", "whitening"]
METHOD_LABELS = {
    "baseline": "Baseline",
    "abtt_optimal": "ABTT",
    "sif_only": "SIF",
    "whitening": "Whitening",
}
METHOD_COLORS = {
    "baseline": "#8a8a8a",
    "abtt_optimal": "#1f77b4",
    "sif_only": "#e67e22",
    "whitening": "#2ca6a4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize resubmit figures.")
    parser.add_argument("--results_csv", required=True, help="Phase resubmit results CSV.")
    parser.add_argument("--dist_dir", required=True, help="Distribution .npz directory.")
    parser.add_argument("--out_dir", required=True, help="Figure output directory.")
    parser.add_argument(
        "--repr_name",
        default="hidden",
        help="Representation to plot (paper default: hidden).",
    )
    parser.add_argument(
        "--feature_model",
        default="bowphs/PhilTa",
        help="Featured model for density 2x2 figure.",
    )
    parser.add_argument(
        "--taskb_mseed_csv",
        default=None,
        help="Optional: aggregated_results.csv from M-seed Task B evaluation.",
    )
    return parser.parse_args()


def normalize_layer_depth(model_name: str, layer: pd.Series) -> pd.Series:
    max_layers = MAX_LAYERS.get(model_name, int(layer.max()))
    return layer.astype(float) / float(max_layers) * 100.0


def filter_release_rows(results: pd.DataFrame, repr_name: str) -> pd.DataFrame:
    keep_rows = []
    for method in METHOD_ORDER:
        sub = results[(results["repr"] == repr_name) & (results["method"] == method)].copy()
        if method in {"baseline", "abtt_optimal", "whitening"}:
            sub = sub[sub["pooling"] == "mean"]
        elif method == "sif_only":
            sub = sub[sub["pooling"] == "sif"]
        keep_rows.append(sub)
    if not keep_rows:
        return pd.DataFrame()
    return pd.concat(keep_rows, ignore_index=True)


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    # ``CreationDate: None`` drops the wall-clock stamp matplotlib otherwise
    # writes into the PDF trailer. Without it every re-run of the figure job
    # rewrites five bytes in each PDF and dirties the tree, which makes it
    # impossible to tell a real figure change from a re-run. PNGs carry no
    # such stamp and were already byte-reproducible.
    fig.savefig(
        out_dir / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(fig)


def plot_metric(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    models: list[str],
    methods: list[str],
) -> None:
    n_models = len(models)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(4.3 * n_models, 4.4),
        sharex=False,
        sharey=(metric in ("aucroc", "overall_assignment_acc")),
        squeeze=False,
    )
    axes = axes.ravel()

    if metric == "gap":
        finite_values = results[metric].replace([np.inf, -np.inf], np.nan).dropna()
        y_min = float(finite_values.min()) if not finite_values.empty else -0.1
        y_max = float(finite_values.max()) if not finite_values.empty else 0.8
        pad = max(0.03, (y_max - y_min) * 0.08)
        y_limits = (y_min - pad, y_max + pad)
    else:
        y_limits = (0.3, 1.02)

    handles = []
    labels = []
    for ax, model_name in zip(axes, models):
        model_rows = results[results["model"] == model_name].copy()
        for method in methods:
            method_rows = model_rows[model_rows["method"] == method].sort_values("layer")
            if method_rows.empty:
                continue
            x = normalize_layer_depth(model_name, method_rows["layer"])
            line = ax.plot(
                x,
                method_rows[metric],
                color=METHOD_COLORS[method],
                marker="o",
                markersize=4.5,
                linewidth=2.3,
                label=METHOD_LABELS[method],
            )[0]
            if METHOD_LABELS[method] not in labels:
                handles.append(line)
                labels.append(METHOD_LABELS[method])

        ax.set_title(SHORT.get(model_name, model_name), fontsize=15, fontweight="bold")
        ax.set_xlabel("Layer Depth (%)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlim(0, 100)
        ax.set_ylim(*y_limits)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.25, linewidth=0.8)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out_dir, stem)


def plot_metric_grid_6model(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    models: list[str],
    methods: list[str],
) -> None:
    """Two rows of three models, one panel per model, shared axes.

    This is the shape the paper's two headline figures use, and it differs from
    ``plot_metric`` in ways that matter and must not be merged into it: the y
    axis autoscales instead of being pinned to a hardcoded range, the axis
    labels are sentence case, and only the outer panels carry labels. The
    single-row figures were built against the other conventions, so changing
    them there would silently move six other figures.
    """
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        # 4.6 rather than 5.3 inches tall: these are line plots with five x ticks
        # and four y ticks, so the shorter panels stay legible at column width
        # and give the main text back about a third of a column.
        figsize=(12.6, 4.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat = axes.ravel()

    handles: list = []
    labels: list[str] = []
    for idx, (ax, model_name) in enumerate(zip(flat, models)):
        model_rows = results[results["model"] == model_name].copy()
        for method in methods:
            method_rows = model_rows[model_rows["method"] == method].sort_values("layer")
            if method_rows.empty:
                continue
            x = normalize_layer_depth(model_name, method_rows["layer"])
            line = ax.plot(
                x,
                method_rows[metric],
                color=METHOD_COLORS[method],
                marker="o",
                markersize=4.5,
                linewidth=2.3,
                label=METHOD_LABELS[method],
            )[0]
            if METHOD_LABELS[method] not in labels:
                handles.append(line)
                labels.append(METHOD_LABELS[method])

        ax.set_title(SHORT.get(model_name, model_name), fontsize=15, fontweight="bold")
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=13)
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel("Layer depth (%)", fontsize=13)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.25, linewidth=0.8)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out_dir, stem)


def compute_collapsed_layer(
    results: pd.DataFrame, model: str, max_layer: int
) -> tuple[int, int]:
    """Return (last_layer, collapsed_layer), the rule run_resubmit_distributions.py uses.

    ``last`` is the deepest layer with a baseline row. The collapsed layer is the
    baseline layer with the *highest* AUROC in the 30-70% depth band, which reads
    backwards until you see the profile: on the T5 encoders every layer in that
    band sits near chance, so taking the strongest of them makes the
    before-and-after panels a conservative illustration rather than a
    best-case one. Falls back to (max_layer, max_layer // 2) if no rows match.
    """
    base = results[
        (results["model"] == model)
        & (results["method"] == "baseline")
        & (results["repr"] == "hidden")
        & (results["pooling"] == "mean")
    ].copy()
    if base.empty:
        return max_layer, max_layer // 2
    last_layer = int(base["layer"].max())
    base["pct"] = base["layer"] / max_layer * 100
    middle = base[(base["pct"] >= 30) & (base["pct"] <= 70)]
    if middle.empty:
        return last_layer, max_layer // 2
    collapsed_layer = int(middle.loc[middle["aucroc"].idxmax(), "layer"])
    return last_layer, collapsed_layer


def _draw_density_panel(
    ax: plt.Axes,
    npz_path: Path,
    title: str,
    bins: np.ndarray,
) -> float:
    """Draw a single same-vs-different cosine-similarity histogram. Returns the
    panel's top y-limit so callers can equalize per row.
    """
    if not npz_path.exists():
        ax.axis("off")
        ax.text(0.5, 0.5, f"Missing {npz_path.name}", ha="center", va="center")
        return 0.0

    data = np.load(npz_path)
    same = data["same_sims"]
    diff = data["diff_sims"]
    ax.hist(
        diff,
        bins=bins,
        alpha=0.55,
        color="#c0392b",
        density=True,
        edgecolor="#7f241a",
        linewidth=0.6,
        label="Different",
    )
    ax.hist(
        same,
        bins=bins,
        alpha=0.55,
        color="#2471a3",
        density=True,
        edgecolor="#17436b",
        linewidth=0.6,
        label="Same",
    )
    same_mean = float(np.mean(same))
    diff_mean = float(np.mean(diff))
    ax.axvline(same_mean, color="#2471a3", linestyle="--", linewidth=1.3)
    ax.axvline(diff_mean, color="#c0392b", linestyle="--", linewidth=1.3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.text(
        0.98,
        0.95,
        f"gap={same_mean - diff_mean:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    return float(ax.get_ylim()[1])


def plot_density_2x2(
    feature_model: str,
    dist_dir: Path,
    out_dir: Path,
    results: pd.DataFrame | None = None,
) -> None:
    """PhilTa-style 2x2: rows = {Last layer, Collapsed layer}, cols = {Baseline, ABTT}.
    Uses pure `abtt_optimal` distributions (mean-pooled + ABTT), not SIF+ABTT.
    """
    slug = feature_model.replace("/", "_")
    max_layer = MAX_LAYERS.get(feature_model, 12)
    if results is not None:
        last_layer, collapsed_layer = compute_collapsed_layer(results, feature_model, max_layer)
    else:
        last_layer, collapsed_layer = max_layer, max_layer // 2

    conds = [
        ("baseline_last", f"Baseline · Last layer (L{last_layer})"),
        ("baseline_middle", f"Baseline · Collapsed layer (L{collapsed_layer})"),
        ("abtt_last", f"ABTT · Last layer (L{last_layer})"),
        ("abtt_middle", f"ABTT · Collapsed layer (L{collapsed_layer})"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.ravel()
    bins = np.linspace(-0.2, 1.0, 48)
    row_ymax = [0.0, 0.0]

    for idx, (suffix, title) in enumerate(conds):
        ymax = _draw_density_panel(
            axes[idx], dist_dir / f"{slug}_{suffix}.npz", title, bins
        )
        row = idx // 2
        row_ymax[row] = max(row_ymax[row], ymax)

    for row in range(2):
        ymax = row_ymax[row] * 1.05 if row_ymax[row] > 0 else 1.0
        for col in range(2):
            axes[row * 2 + col].set_ylim(0, ymax)

    axes[0].set_ylabel("Density", fontsize=12)
    axes[2].set_ylabel("Density", fontsize=12)
    axes[2].set_xlabel("Cosine Similarity", fontsize=12)
    axes[3].set_xlabel("Cosine Similarity", fontsize=12)
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    save(fig, out_dir, "paper_fig_density_2x2")


def plot_density_2x2_models(
    feature_models: list[str],
    dist_dir: Path,
    out_dir: Path,
    results: pd.DataFrame,
) -> None:
    """2x2 variant focused on the collapsed-layer repair: rows = models, cols = {Baseline, ABTT}
    at each model's collapsed layer. Sharpens the "ABTT repairs the collapsed layer" story across
    two models instead of four panels of one model. Uses pure `abtt_optimal` distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    bins = np.linspace(-0.2, 1.0, 48)

    for row_idx, model in enumerate(feature_models[:2]):
        slug = model.replace("/", "_")
        max_layer = MAX_LAYERS.get(model, 12)
        _, collapsed_layer = compute_collapsed_layer(results, model, max_layer)
        short = SHORT.get(model, model)
        panels = [
            ("baseline_middle", f"{short} · Baseline (L{collapsed_layer})"),
            ("abtt_middle", f"{short} · ABTT (L{collapsed_layer})"),
        ]
        row_ymax = 0.0
        for col_idx, (suffix, title) in enumerate(panels):
            ymax = _draw_density_panel(
                axes[row_idx, col_idx],
                dist_dir / f"{slug}_{suffix}.npz",
                title,
                bins,
            )
            row_ymax = max(row_ymax, ymax)
        scaled = row_ymax * 1.05 if row_ymax > 0 else 1.0
        for col_idx in range(2):
            axes[row_idx, col_idx].set_ylim(0, scaled)
        axes[row_idx, 0].set_ylabel("Density", fontsize=12)

    axes[1, 0].set_xlabel("Cosine Similarity", fontsize=12)
    axes[1, 1].set_xlabel("Cosine Similarity", fontsize=12)
    axes[0, 1].legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    save(fig, out_dir, "paper_fig_density_2x2_2models")


def plot_taskb_mseed_bar(csv_path: Path, out_dir: Path) -> None:
    """Bar chart of M-seed Task B results: best config per model at K=1,3,5."""
    agg_df = pd.read_csv(csv_path)

    model_order = [m for m in SHORT.keys() if m in agg_df["model"].values]
    rows = []
    for model in model_order:
        sub = agg_df[(agg_df["model"] == model) & (agg_df["repr"] == "hidden")]
        if sub.empty:
            continue
        rows.append(sub.loc[sub["dir_acc_at_1_mean"].idxmax()])
    if not rows:
        print("Warning: no Task B M-seed data to plot.")
        return
    best_df = pd.DataFrame(rows).reset_index(drop=True)

    models = [SHORT.get(r["model"], r["model"]) for _, r in best_df.iterrows()]
    ks = [1, 3, 5]
    n_models = len(models)
    n_k = len(ks)
    x = np.arange(n_models)
    width = 0.8 / n_k
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(ks):
        means = best_df[f"dir_acc_at_{k}_mean"].values * 100
        stds = best_df[f"dir_acc_at_{k}_std"].values * 100
        offset = (i - n_k / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=f"Top-{k}", color=colors[i], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Task B: Directory Accuracy (M=5 seeds)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ymin = max(0, best_df["dir_acc_at_1_mean"].min() * 100 - 15)
    ax.set_ylim(ymin, 105)
    fig.tight_layout()
    save(fig, out_dir, "fig_taskb_mseed_bar")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results_csv)
    release_rows = filter_release_rows(results, args.repr_name)
    if release_rows.empty:
        raise SystemExit("No release rows matched the requested representation.")

    main_models = ["bowphs/LaTa", "bowphs/PhilTa", "google/mt5-base"]
    appendix_models = [
        "sentence-transformers/LaBSE",
        "Qwen/Qwen3-Embedding-0.6B",
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    ]

    # The paper's two headline figures: all six models on one grid.
    for metric, ylabel, stem in (
        ("aucroc", "AUROC", "fig_release_aucroc_6model"),
        ("gap", "Cosine gap", "fig_release_gap_6model"),
    ):
        plot_metric_grid_6model(
            release_rows,
            metric=metric,
            ylabel=ylabel,
            out_dir=out_dir,
            stem=stem,
            models=main_models + appendix_models,
            methods=["baseline", "sif_only", "abtt_optimal"],
        )

    plot_metric(
        release_rows,
        metric="aucroc",
        ylabel="AUROC",
        out_dir=out_dir,
        stem="fig_release_aucroc_per_model",
        models=main_models,
        methods=["baseline", "abtt_optimal"],
    )
    plot_metric(
        release_rows,
        metric="aucroc",
        ylabel="AUROC",
        out_dir=out_dir,
        stem="fig_appendix_aucroc_per_model",
        models=appendix_models,
        methods=["baseline", "abtt_optimal"],
    )
    plot_metric(
        release_rows,
        metric="overall_assignment_acc",
        ylabel="Assignment Accuracy",
        out_dir=out_dir,
        stem="fig_release_taskb_per_model",
        models=main_models,
        methods=["baseline", "abtt_optimal"],
    )
    plot_metric(
        release_rows,
        metric="overall_assignment_acc",
        ylabel="Assignment Accuracy",
        out_dir=out_dir,
        stem="fig_appendix_taskb_per_model",
        models=appendix_models,
        methods=["baseline", "abtt_optimal"],
    )
    plot_metric(
        release_rows,
        metric="gap",
        ylabel="Cosine Gap",
        out_dir=out_dir,
        stem="fig_release_gap_per_model",
        models=main_models,
        methods=["baseline", "abtt_optimal", "whitening"],
    )
    plot_metric(
        release_rows,
        metric="gap",
        ylabel="Cosine Gap",
        out_dir=out_dir,
        stem="fig_appendix_gap_per_model",
        models=appendix_models,
        methods=["baseline", "abtt_optimal", "whitening", "sif_only"],
    )

    dist_dir = Path(args.dist_dir)
    if dist_dir.exists():
        plot_density_2x2(args.feature_model, dist_dir, out_dir, results=results)
        print(f"Saved density 2x2 to {out_dir}")
        plot_density_2x2_models(
            ["bowphs/LaTa", "bowphs/PhilTa"],
            dist_dir,
            out_dir,
            results=results,
        )
        print(f"Saved density 2x2 (2-model variant) to {out_dir}")
    else:
        print(f"Warning: dist_dir {dist_dir} not found, skipping density figure.")

    if args.taskb_mseed_csv:
        mseed_path = Path(args.taskb_mseed_csv)
        if mseed_path.exists():
            plot_taskb_mseed_bar(mseed_path, out_dir)
            print(f"Saved Task B M-seed bar chart to {out_dir}")
        else:
            print(f"Warning: --taskb_mseed_csv {mseed_path} not found, skipping.")

    print(f"Saved release figures to {out_dir}")


if __name__ == "__main__":
    main()
