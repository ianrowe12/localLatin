"""Phase 12c Analysis: Visualize retrieval-target token attribution results.

Four analyses:
1. Retrieval Attribution Delta (Δ_retrieval) by token category
2. Positive vs Negative partner contrast (selectivity)
3. Token-level heatmaps for exemplar documents
4. Per-token retrieval importance share (content vs subword vs empty)

Reads: runs/phase12c/attributions/{slug}/layer{L}_retrieval_attr.npz
Outputs: runs/phase12c/figures/ + runs/phase12c/analysis_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from token_filtering import classify_token_for_filtering

MODELS = {
    "LaTa": {
        "slug": "bowphs_LaTa",
        "hf": "bowphs/LaTa",
        "trust": False,
        "selected_layers": [1, 4, 8, 12],
        "labels": ["L1 early", "L4 max dip", "L8 mid dip", "L12 best"],
    },
    "PhilTa": {
        "slug": "bowphs_PhilTa",
        "hf": "bowphs/PhilTa",
        "trust": False,
        "selected_layers": [1, 6, 9, 12],
        "labels": ["L1 early", "L6 worst", "L9 mid dip", "L12 best"],
    },
    "LaBSE": {
        "slug": "sentence-transformers_LaBSE",
        "hf": "sentence-transformers/LaBSE",
        "trust": False,
        "selected_layers": [1, 6, 9, 11],
        "labels": ["L1 worst", "L6 mid", "L9 late", "L11 best"],
    },
    "Qwen3-0.6B": {
        "slug": "Qwen_Qwen3-Embedding-0.6B",
        "hf": "Qwen/Qwen3-Embedding-0.6B",
        "trust": True,
        "selected_layers": [1, 4, 14, 28],
        "labels": ["L1 early", "L4 worst", "L14 mid", "L28 best"],
    },
    "KaLM-mini": {
        "slug": "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5",
        "hf": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "trust": True,
        "selected_layers": [1, 5, 13, 23],
        "labels": ["L1 early", "L5 worst", "L13 mid", "L23 best"],
    },
    "mt5-base": {
        "slug": "google_mt5-base",
        "hf": "google/mt5-base",
        "trust": False,
        "selected_layers": [1, 4, 8, 12],
        "labels": ["L1 early", "L4 worst", "L8 mid", "L12 best"],
    },
}
CATEGORIES = ["content", "short_subword", "empty"]


def load_attribution_data(attr_dir, slug, layer):
    """Load .npz file for a model/layer. Returns dict or None."""
    path = Path(attr_dir) / slug / f"layer{layer}_retrieval_attr.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def decode_tokens(input_ids_row, seq_len, tokenizer):
    """Decode token IDs to strings."""
    ids = input_ids_row[:seq_len].tolist()
    return [tokenizer.decode([tid]) for tid in ids]


# ──────────────────────────────────────────────────────────────
# Analysis 1: Δ_retrieval by token category
# ──────────────────────────────────────────────────────────────
def analysis1_delta_retrieval(attr_dir, fig_dir, tokenizers):
    """Compute Δ = |IG_abtt_pos| − |IG_baseline_pos| per token, grouped by category."""
    print("\n=== Analysis 1: Δ_retrieval by token category ===")

    rows = []
    for mname, cfg in MODELS.items():
        tok = tokenizers[mname]
        for li, layer in enumerate(cfg["selected_layers"]):
            data = load_attribution_data(attr_dir, cfg["slug"], layer)
            if data is None:
                continue

            n_q = data["seq_lengths"].shape[0]
            for qi in range(n_q):
                sl = int(data["seq_lengths"][qi])
                if sl == 0:
                    continue

                ig_bp = np.abs(data["ig_baseline_pos"][qi, :sl])
                ig_ap = np.abs(data["ig_abtt_pos"][qi, :sl])
                delta = ig_ap - ig_bp  # positive = ABTT helped this token

                tokens = decode_tokens(data["input_ids"][qi], sl, tok)
                cats = [classify_token_for_filtering(t) for t in tokens]

                for ci, cat in enumerate(cats):
                    if cat in CATEGORIES:
                        rows.append({
                            "model": mname, "layer": layer,
                            "label": cfg["labels"][li],
                            "category": cat, "delta": delta[ci],
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No data")
        return df

    # Grouped bar chart
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 5), sharey=True)
    colors = {"content": "#2196F3", "short_subword": "#FF9800", "empty": "#9E9E9E"}
    bar_w = 0.25

    for ax, (mname, cfg) in zip(axes, MODELS.items()):
        mdf = df[df["model"] == mname]
        if mdf.empty:
            ax.set_title(mname)
            continue

        labels = cfg["labels"]
        x = np.arange(len(labels))

        for ci, cat in enumerate(CATEGORIES):
            means = []
            for lab in labels:
                sub = mdf[(mdf["label"] == lab) & (mdf["category"] == cat)]
                means.append(sub["delta"].mean() if len(sub) > 0 else 0)
            ax.bar(x + ci * bar_w, means, bar_w, label=cat, color=colors[cat], alpha=0.85)

        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(mname, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Δ_retrieval (ABTT − baseline)")

    axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Retrieval Attribution Delta by Token Category\n"
                 "(positive = ABTT helped this token match partner)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_analysis1_delta_retrieval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Summary CSV
    summary = df.groupby(["model", "label", "category"])["delta"].agg(["mean", "std", "count"])
    summary.to_csv(fig_dir / "analysis1_delta_summary.csv")
    print(f"  Saved figure + CSV ({len(df)} token observations)")
    return df


# ──────────────────────────────────────────────────────────────
# Analysis 2: Positive vs Negative partner contrast
# ──────────────────────────────────────────────────────────────
def analysis2_selectivity(attr_dir, fig_dir, tokenizers):
    """Measure how well ABTT separates positive vs negative partner attribution."""
    print("\n=== Analysis 2: Positive vs Negative Selectivity ===")

    rows = []
    for mname, cfg in MODELS.items():
        for li, layer in enumerate(cfg["selected_layers"]):
            data = load_attribution_data(attr_dir, cfg["slug"], layer)
            if data is None:
                continue

            n_q = data["seq_lengths"].shape[0]
            for qi in range(n_q):
                sl = int(data["seq_lengths"][qi])
                if sl == 0:
                    continue

                # Mean absolute IG per query
                mean_bp = np.abs(data["ig_baseline_pos"][qi, :sl]).mean()
                mean_bn = np.abs(data["ig_baseline_neg"][qi, :sl]).mean()
                mean_ap = np.abs(data["ig_abtt_pos"][qi, :sl]).mean()
                mean_an = np.abs(data["ig_abtt_neg"][qi, :sl]).mean()

                # Selectivity = |IG_pos| / (|IG_pos| + |IG_neg|)
                sel_base = mean_bp / max(mean_bp + mean_bn, 1e-12)
                sel_abtt = mean_ap / max(mean_ap + mean_an, 1e-12)

                rows.append({
                    "model": mname, "layer": layer, "label": cfg["labels"][li],
                    "base_pos": mean_bp, "base_neg": mean_bn,
                    "abtt_pos": mean_ap, "abtt_neg": mean_an,
                    "selectivity_baseline": sel_base,
                    "selectivity_abtt": sel_abtt,
                    "selectivity_gain": sel_abtt - sel_base,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No data")
        return df

    # Scatter: selectivity baseline vs ABTT
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.5 * len(MODELS), 4.5), sharey=True)
    for ax, (mname, cfg) in zip(axes, MODELS.items()):
        mdf = df[df["model"] == mname]
        if mdf.empty:
            ax.set_title(mname)
            continue

        for li, lab in enumerate(cfg["labels"]):
            sub = mdf[mdf["label"] == lab]
            ax.scatter(sub["selectivity_baseline"], sub["selectivity_abtt"],
                       s=15, alpha=0.6, label=lab)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
        ax.set_xlim(0.3, 0.8)
        ax.set_ylim(0.3, 0.8)
        ax.set_xlabel("Selectivity (baseline)")
        ax.set_title(mname, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Selectivity (ABTT)")
        ax.legend(fontsize=6, loc="lower right")

    fig.suptitle("Partner Selectivity: Baseline vs ABTT\n"
                 "(above diagonal = ABTT improved selectivity)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_analysis2_selectivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Summary
    summary = df.groupby(["model", "label"]).agg({
        "selectivity_baseline": "mean",
        "selectivity_abtt": "mean",
        "selectivity_gain": "mean",
    }).round(4)
    summary.to_csv(fig_dir / "analysis2_selectivity_summary.csv")
    pct_improved = (df["selectivity_gain"] > 0).mean() * 100
    print(f"  {pct_improved:.1f}% of queries had improved selectivity after ABTT")
    return df


# ──────────────────────────────────────────────────────────────
# Analysis 3: Exemplar heatmaps
# ──────────────────────────────────────────────────────────────
def analysis3_heatmaps(attr_dir, fig_dir, tokenizers, n_examples=5):
    """Heatmap comparing baseline vs ABTT IG for exemplar queries."""
    print("\n=== Analysis 3: Exemplar Token Heatmaps ===")

    for mname, cfg in MODELS.items():
        tok = tokenizers[mname]
        # Use the worst dip layer
        dip_layer = cfg["selected_layers"][1]  # second layer = worst/dip

        data = load_attribution_data(attr_dir, cfg["slug"], dip_layer)
        if data is None:
            continue

        n_q = data["seq_lengths"].shape[0]
        # Pick queries with highest delta (ABTT helped most)
        deltas = []
        for qi in range(n_q):
            sl = int(data["seq_lengths"][qi])
            if sl == 0:
                deltas.append(0)
                continue
            d = np.abs(data["ig_abtt_pos"][qi, :sl]).mean() - \
                np.abs(data["ig_baseline_pos"][qi, :sl]).mean()
            deltas.append(d)

        top_qi = np.argsort(deltas)[-n_examples:][::-1]

        fig, axes = plt.subplots(n_examples, 2, figsize=(16, 2.5 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, 2)

        for row_idx, qi in enumerate(top_qi):
            sl = int(data["seq_lengths"][qi])
            tokens = decode_tokens(data["input_ids"][qi], sl, tok)
            # Truncate display to first 40 tokens for readability
            show_len = min(sl, 40)
            tokens_show = tokens[:show_len]

            ig_base = data["ig_baseline_pos"][qi, :show_len]
            ig_abtt = data["ig_abtt_pos"][qi, :show_len]

            # Normalize to same scale
            vmax = max(np.abs(ig_base).max(), np.abs(ig_abtt).max(), 1e-8)

            for col, (ig_vals, title) in enumerate(
                [(ig_base, "Baseline"), (ig_abtt, "ABTT")]
            ):
                ax = axes[row_idx, col]
                im = ax.imshow(
                    ig_vals.reshape(1, -1),
                    aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                )
                ax.set_xticks(range(show_len))
                ax.set_xticklabels(tokens_show, rotation=90, fontsize=5, fontfamily="monospace")
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"Q{qi}", fontsize=8)
                if row_idx == 0:
                    ax.set_title(title, fontweight="bold")

        fig.suptitle(f"{mname} L{dip_layer} — Token Attribution to Positive Partner\n"
                     f"(red = pushes toward partner, blue = pushes away)",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig_analysis3_heatmap_{mname}.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  {mname}: heatmap saved")


# ──────────────────────────────────────────────────────────────
# Analysis 4: Importance share by token category
# ──────────────────────────────────────────────────────────────
def analysis4_importance_share(attr_dir, fig_dir, tokenizers):
    """What fraction of total |IG| comes from content vs subword vs empty tokens?"""
    print("\n=== Analysis 4: Importance Share by Token Category ===")

    rows = []
    for mname, cfg in MODELS.items():
        tok = tokenizers[mname]
        for li, layer in enumerate(cfg["selected_layers"]):
            data = load_attribution_data(attr_dir, cfg["slug"], layer)
            if data is None:
                continue

            # Accumulate total |IG| per category across all queries
            cat_total_base = {c: 0.0 for c in CATEGORIES}
            cat_total_abtt = {c: 0.0 for c in CATEGORIES}

            n_q = data["seq_lengths"].shape[0]
            for qi in range(n_q):
                sl = int(data["seq_lengths"][qi])
                if sl == 0:
                    continue
                tokens = decode_tokens(data["input_ids"][qi], sl, tok)
                cats = [classify_token_for_filtering(t) for t in tokens]

                ig_base = np.abs(data["ig_baseline_pos"][qi, :sl])
                ig_abtt = np.abs(data["ig_abtt_pos"][qi, :sl])

                for ci, cat in enumerate(cats):
                    if cat in CATEGORIES:
                        cat_total_base[cat] += ig_base[ci]
                        cat_total_abtt[cat] += ig_abtt[ci]

            total_base = sum(cat_total_base.values())
            total_abtt = sum(cat_total_abtt.values())

            for cat in CATEGORIES:
                rows.append({
                    "model": mname, "layer": layer, "label": cfg["labels"][li],
                    "category": cat,
                    "share_baseline": cat_total_base[cat] / max(total_base, 1e-12),
                    "share_abtt": cat_total_abtt[cat] / max(total_abtt, 1e-12),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No data")
        return df

    # Stacked bar chart: share by category, baseline vs ABTT
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 5), sharey=True)
    colors = {"content": "#2196F3", "short_subword": "#FF9800", "empty": "#9E9E9E"}

    for ax, (mname, cfg) in zip(axes, MODELS.items()):
        mdf = df[df["model"] == mname]
        if mdf.empty:
            ax.set_title(mname)
            continue

        labels = cfg["labels"]
        x = np.arange(len(labels))
        w = 0.35

        for side, (offset, method) in enumerate(
            [(-w/2, "share_baseline"), (w/2, "share_abtt")]
        ):
            bottom = np.zeros(len(labels))
            for cat in CATEGORIES:
                vals = []
                for lab in labels:
                    sub = mdf[(mdf["label"] == lab) & (mdf["category"] == cat)]
                    vals.append(sub[method].values[0] if len(sub) > 0 else 0)
                vals = np.array(vals)
                label_str = f"{cat}" if side == 0 else None
                ax.bar(x + offset, vals, w, bottom=bottom,
                       color=colors[cat], label=label_str, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
                bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(mname, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Share of total |IG|")

        # Add "Base" / "ABTT" labels
        ax.text(0.25, -0.15, "B  A", transform=ax.transAxes, fontsize=6,
                ha="center", color="gray")

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Token Category Importance Share: Baseline (left bar) vs ABTT (right bar)\n"
                 "(higher content share after ABTT = unmasking)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_analysis4_importance_share.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(fig_dir / "analysis4_importance_share.csv", index=False)
    print(f"  Saved figure + CSV")
    return df


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attr_dir", default="runs/phase12c/attributions")
    p.add_argument("--fig_dir", default="runs/phase12c/figures")
    p.add_argument(
        "--fail_if_empty",
        action="store_true",
        help="Exit non-zero if no analysis data is found.",
    )
    args = p.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizers for all models
    from transformers import AutoTokenizer
    print("Loading tokenizers...")
    tokenizers = {}
    for mname, cfg in MODELS.items():
        tokenizers[mname] = AutoTokenizer.from_pretrained(
            cfg["hf"], trust_remote_code=cfg["trust"]
        )
    print(f"  Loaded {len(tokenizers)} tokenizers")

    # Run all analyses
    df1 = analysis1_delta_retrieval(args.attr_dir, fig_dir, tokenizers)
    df2 = analysis2_selectivity(args.attr_dir, fig_dir, tokenizers)
    analysis3_heatmaps(args.attr_dir, fig_dir, tokenizers)
    df4 = analysis4_importance_share(args.attr_dir, fig_dir, tokenizers)

    has_data = any(not df.empty for df in (df1, df2, df4))
    if args.fail_if_empty and not has_data:
        raise SystemExit("No phase12c analysis data found.")

    print("\n=== All analyses complete ===")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
