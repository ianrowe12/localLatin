"""Phase 12b Deep Analysis: Five methods to prove ABTT 'unmasks' token signals.

Method 1 — Signal-to-Noise Ratio (SNR) Shift:
    SNR = ||v_ABTT|| / |v · PC1|,  computed per-document (mean-pooled hidden state).
    Shows PC1 eclipses semantic content in dip layers.

Method 2 — Attribution Delta (ΔIG):
    ΔIG = |IG_abtt_norm(t)| − |IG_pc1_dot(t)|,  per-token, grouped by category.
    Positive ΔIG = token is MORE important for clean signal than for noise.
    Negative ΔIG = token is noise-dominated.

Method 3 — Cosine Resolution Gap:
    Density of pairwise cosine similarities before/after ABTT for
    same-folder (similar) vs cross-folder (dissimilar) document pairs.
    Shows ABTT restores discrimination by separating the distributions.

Method 4 — Effective Dimensionality:
    Participation ratio of eigenvalues: d_eff = (Σλ)² / Σ(λ²).
    Shows embedding space literally collapses to ~1 dimension in dip layers,
    and ABTT restores it to usable dimensionality.

Method 5 — Retrieval Rank Shift:
    For each 'winnable' test query, find its best same-folder partner's rank
    in the cosine-sorted list before and after ABTT. Shows file-by-file how
    ABTT pushes true partners from rank ~300 to rank ~1.

Usage:
  python scripts/run_phase12b_deep_analysis.py \\
    --emb_base runs/phase9_bases \\
    --pc_dir  runs/phase12/pcs \\
    --attr_dir runs/phase12/attributions \\
    --split_csv runs/phase9/phase9_split.csv \\
    --phase11_csv runs/phase11/phase11_results.csv \\
    --out_dir runs/phase12b/deep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer

# ── Model configs with selected layers ────────────────────────────────────
MODELS = {
    "LaTa": {
        "slug": "bowphs_LaTa",
        "hf": "bowphs/LaTa",
        "tok": "bowphs/LaTa",
        "trust": False,
        "selected_layers": [1, 4, 8, 12],
        "labels": ["L1\nearly", "L4\nmax dip", "L8\nmid dip", "L12\nbest"],
    },
    "PhilTa": {
        "slug": "bowphs_PhilTa",
        "hf": "bowphs/PhilTa",
        "tok": "bowphs/PhilTa",
        "trust": False,
        "selected_layers": [1, 6, 9, 12],
        "labels": ["L1\nearly", "L6\nworst", "L9\nmid dip", "L12\nbest"],
    },
    "LaBSE": {
        "slug": "sentence-transformers_LaBSE",
        "hf": "sentence-transformers/LaBSE",
        "tok": "sentence-transformers/LaBSE",
        "trust": False,
        "selected_layers": [1, 6, 9, 11],
        "labels": ["L1\nworst", "L6\nmid", "L9\nlate", "L11\nbest"],
    },
    "Qwen3-0.6B": {
        "slug": "Qwen_Qwen3-Embedding-0.6B",
        "hf": "Qwen/Qwen3-Embedding-0.6B",
        "tok": "Qwen/Qwen3-Embedding-0.6B",
        "trust": True,
        "selected_layers": [1, 4, 14, 28],
        "labels": ["L1\nearly", "L4\nworst", "L14\nmid", "L28\nbest"],
    },
    "KaLM-mini": {
        "slug": "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5",
        "hf": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "tok": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "trust": True,
        "selected_layers": [1, 5, 13, 23],
        "labels": ["L1\nearly", "L5\nworst", "L13\nmid", "L23\nbest"],
    },
}

LATIN_PUNCTUATION = set(".,;:!?()[]{}\"'-/\\@#$%^&*+=<>~`")
CATEGORIES = ["content", "short_subword", "empty"]  # focus on these three


def classify_token(s: str) -> str:
    s = s.strip()
    if not s:
        return "empty"
    clean = s.lstrip("▁##Ġ")
    if not clean:
        return "empty"
    if all(c in LATIN_PUNCTUATION for c in clean):
        return "other"
    if clean.isdigit():
        return "other"
    if len(clean) <= 2:
        return "short_subword"
    return "content"


# ── data loaders ──────────────────────────────────────────────────────────
def load_emb(emb_base: Path, slug: str, layer: int) -> np.ndarray | None:
    p = emb_base / slug / "hidden_mean" / f"hidden_layer{layer}_embeddings.npy"
    return np.load(p) if p.exists() else None


def load_pcs(pc_dir: Path, slug: str, layer: int):
    p = pc_dir / slug / f"layer{layer}_pcs.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["pcs"], d["mean_vec"]


def load_attr(attr_dir: Path, slug: str, layer: int):
    p = attr_dir / slug / f"layer{layer}_attributions.npz"
    return dict(np.load(p)) if p.exists() else None


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    Xn = X / norms
    return Xn @ Xn.T


def apply_abtt(X: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    Xc = X - mean_vec
    return Xc - Xc @ pcs.T @ pcs


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  METHOD 1: SIGNAL-TO-NOISE RATIO (SNR)                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def method1_snr(emb_base, pc_dir, split_df, test_mask, fig_dir):
    """Compute SNR = ||v_ABTT|| / |v·PC1| per document, split by similar/dissimilar."""

    print("\n" + "="*70)
    print("  METHOD 1: Signal-to-Noise Ratio (SNR)")
    print("="*70)

    winnable = split_df["is_winnable"].values[test_mask]
    rows = []

    for mname, cfg in MODELS.items():
        slug = cfg["slug"]
        for li, layer in enumerate(cfg["selected_layers"]):
            X = load_emb(emb_base, slug, layer)
            pcs, mean_vec = load_pcs(pc_dir, slug, layer)
            if X is None or pcs is None:
                continue

            X_test = X[test_mask]
            Xc = X_test - mean_vec
            pc1 = pcs[0]  # first PC only

            # Per-document: noise magnitude = |v · PC1|, signal = ||v_ABTT||
            noise = np.abs(Xc @ pc1)                         # (n,)
            signal = np.linalg.norm(Xc - np.outer(Xc @ pc1, pc1), axis=1)  # |v - proj_PC1|
            snr = signal / noise.clip(1e-12)

            for is_w in [True, False]:
                mask = winnable == is_w
                cat = "Similar" if is_w else "Dissimilar"
                rows.append({
                    "model": mname, "layer": layer, "layer_label": cfg["labels"][li],
                    "category": cat,
                    "mean_snr": float(np.mean(snr[mask])),
                    "median_snr": float(np.median(snr[mask])),
                    "mean_noise": float(np.mean(noise[mask])),
                    "mean_signal": float(np.mean(signal[mask])),
                    "n": int(mask.sum()),
                })

        print(f"  {mname}: done")

    df = pd.DataFrame(rows)
    df.to_csv(fig_dir.parent / "snr_per_layer.csv", index=False)

    # ── Visualization: Dual-bar (log scale noise vs signal) per model ─────
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 3.8 * len(MODELS)))
    bar_width = 0.18

    for ax, (mname, cfg) in zip(axes, MODELS.items()):
        sub = df[df["model"] == mname]
        layers = cfg["selected_layers"]
        labels = cfg["labels"]
        x = np.arange(len(layers))

        # Bars: noise (red) and signal (blue) for similar and dissimilar
        sim_noise = sub[sub["category"] == "Similar"]["mean_noise"].values
        sim_signal = sub[sub["category"] == "Similar"]["mean_signal"].values
        dis_noise = sub[sub["category"] == "Dissimilar"]["mean_noise"].values
        dis_signal = sub[sub["category"] == "Dissimilar"]["mean_signal"].values

        ax.bar(x - 1.5*bar_width, sim_noise, bar_width, color="#e74c3c", alpha=0.85,
               label="|v·PC₁| (noise) — similar", edgecolor="k", linewidth=0.5)
        ax.bar(x - 0.5*bar_width, sim_signal, bar_width, color="#3498db", alpha=0.85,
               label="‖v_clean‖ (signal) — similar", edgecolor="k", linewidth=0.5)
        ax.bar(x + 0.5*bar_width, dis_noise, bar_width, color="#e74c3c", alpha=0.45,
               label="|v·PC₁| (noise) — dissimilar", edgecolor="k", linewidth=0.5,
               hatch="//")
        ax.bar(x + 1.5*bar_width, dis_signal, bar_width, color="#3498db", alpha=0.45,
               label="‖v_clean‖ (signal) — dissimilar", edgecolor="k", linewidth=0.5,
               hatch="//")

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Magnitude (log)", fontsize=10)
        ax.set_title(f"{mname} — PC₁ Noise vs Clean Signal", fontsize=12, fontweight="bold")
        if ax == axes[0]:
            ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_method1_snr_bars.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_method1_snr_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Visualization 1b: SNR ratio itself, split bars ────────────────────
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.5 * len(MODELS), 4.5),
                             sharey=True)
    for ax, (mname, cfg) in zip(axes, MODELS.items()):
        sub = df[df["model"] == mname]
        layers = cfg["selected_layers"]
        labels = cfg["labels"]
        x = np.arange(len(layers))

        sim_snr = sub[sub["category"] == "Similar"]["mean_snr"].values
        dis_snr = sub[sub["category"] == "Dissimilar"]["mean_snr"].values

        ax.bar(x - 0.15, sim_snr, 0.28, color="#2ecc71", alpha=0.85,
               label="Similar", edgecolor="k", linewidth=0.5)
        ax.bar(x + 0.15, dis_snr, 0.28, color="#e67e22", alpha=0.85,
               label="Dissimilar", edgecolor="k", linewidth=0.5)

        ax.axhline(1.0, color="red", ls="--", lw=1.2, alpha=0.7)
        ax.annotate("SNR = 1\n(noise = signal)", xy=(0.5, 1.0),
                    xycoords=("axes fraction", "data"),
                    fontsize=7, color="red", ha="center", va="bottom")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(mname, fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("SNR  =  ‖signal‖ / |noise|", fontsize=10)
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Signal-to-Noise Ratio per Layer — Below 1 Means PC₁ Dominates",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_method1_snr_ratio.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_method1_snr_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  SNR figures saved")
    return df


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  METHOD 2: ATTRIBUTION DELTA (ΔIG)                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def method2_delta_ig(attr_dir, fig_dir):
    """ΔIG = |IG_abtt_norm(t)| − |IG_pc1_dot(t)|, grouped by token category."""

    print("\n" + "="*70)
    print("  METHOD 2: Attribution Delta (ΔIG)")
    print("="*70)

    rows = []

    for mname, cfg in MODELS.items():
        slug = cfg["slug"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg["tok"], trust_remote_code=cfg["trust"])
        except Exception as e:
            print(f"  WARNING: cannot load tokenizer for {mname}: {e}")
            continue

        for li, layer in enumerate(cfg["selected_layers"]):
            data = load_attr(attr_dir, slug, layer)
            if data is None:
                continue

            n_files = data["ig_pc1_dot"].shape[0]
            cat_deltas = {c: [] for c in CATEGORIES}

            for fi in range(n_files):
                sl = int(data["seq_lengths"][fi])
                if sl < 1:
                    continue
                ids = data["input_ids"][fi, :sl]
                ig_pc1 = np.abs(data["ig_pc1_dot"][fi, :sl])
                ig_abtt = np.abs(data["ig_abtt_norm"][fi, :sl])

                # Raw delta: positive = more useful for ABTT signal, negative = noise
                delta = ig_abtt - ig_pc1
                tokens = tokenizer.convert_ids_to_tokens(ids.tolist())

                for tok_str, d in zip(tokens, delta):
                    cat = classify_token(tok_str)
                    if cat in cat_deltas:
                        cat_deltas[cat].append(float(d))

            for cat in CATEGORIES:
                if cat_deltas[cat]:
                    rows.append({
                        "model": mname, "layer": layer,
                        "layer_label": cfg["labels"][li],
                        "category": cat,
                        "mean_delta_ig": np.mean(cat_deltas[cat]),
                        "median_delta_ig": np.median(cat_deltas[cat]),
                        "std_delta_ig": np.std(cat_deltas[cat]),
                        "n_tokens": len(cat_deltas[cat]),
                    })

        print(f"  {mname}: done")

    df = pd.DataFrame(rows)
    df.to_csv(fig_dir.parent / "delta_ig_by_type.csv", index=False)

    # ── Visualization: Grouped bar chart per model ────────────────────────
    cat_colors = {"content": "#2ecc71", "short_subword": "#e67e22", "empty": "#95a5a6"}
    cat_labels = {"content": "Content\n(roots, words)", "short_subword": "Short Subword\n(≤2 chars)",
                  "empty": "Empty/Marker\n(tokenizer artifacts)"}
    bar_width = 0.22

    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.8 * len(MODELS), 5), sharey=True)
    for ax, (mname, cfg) in zip(axes, MODELS.items()):
        sub = df[df["model"] == mname]
        layers = cfg["selected_layers"]
        labels = cfg["labels"]
        x = np.arange(len(layers))

        for ci, cat in enumerate(CATEGORIES):
            csub = sub[sub["category"] == cat].sort_values("layer")
            if csub.empty:
                continue
            vals = csub["mean_delta_ig"].values
            offset = (ci - 1) * bar_width
            ax.bar(x + offset, vals, bar_width, color=cat_colors[cat],
                   label=cat_labels[cat] if ax == axes[0] else "",
                   edgecolor="k", linewidth=0.5, alpha=0.9)

        ax.axhline(0, color="black", lw=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(mname, fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("ΔIG  =  |IG_abtt| − |IG_pc1|", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Attribution Delta: Positive = Token Serves ABTT Signal, "
                 "Negative = Token Is PC₁ Noise",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_method2_delta_ig_grouped.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_method2_delta_ig_grouped.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Visualization 2b: Heatmap per model (layer × category) ───────────
    for mname, cfg in MODELS.items():
        sub = df[df["model"] == mname]
        if sub.empty:
            continue
        layers = cfg["selected_layers"]
        labels = cfg["labels"]
        heatmap = np.full((len(layers), len(CATEGORIES)), np.nan)
        for li, layer in enumerate(layers):
            for ci, cat in enumerate(CATEGORIES):
                row = sub[(sub["layer"] == layer) & (sub["category"] == cat)]
                if not row.empty:
                    heatmap[li, ci] = row.iloc[0]["mean_delta_ig"]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        vmax = max(abs(np.nanmin(heatmap)), abs(np.nanmax(heatmap)))
        im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels([cat_labels[c] for c in CATEGORIES], fontsize=9)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"ΔIG Heatmap — {mname}", fontsize=12, fontweight="bold")
        for li in range(len(layers)):
            for ci in range(len(CATEGORIES)):
                v = heatmap[li, ci]
                if not np.isnan(v):
                    ax.text(ci, li, f"{v:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if abs(v) > vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax, label="ΔIG  (green = useful, red = noise)", shrink=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig_method2_heatmap_{cfg['slug']}.pdf", dpi=200)
        fig.savefig(fig_dir / f"fig_method2_heatmap_{cfg['slug']}.png", dpi=200)
        plt.close(fig)

    print("  ΔIG figures saved")
    return df


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  METHOD 3: COSINE RESOLUTION GAP                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def method3_cosine_gap(emb_base, pc_dir, split_df, test_mask, fig_dir):
    """Density plots of pairwise cosine similarity before/after ABTT,
    split by same-folder vs cross-folder pairs."""

    print("\n" + "="*70)
    print("  METHOD 3: Cosine Resolution Gap")
    print("="*70)

    fids_test = split_df["folder_id"].values[test_mask]
    fid_arr = np.array(fids_test)
    n = len(fid_arr)
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    same_mask = np.equal.outer(fid_arr, fid_arr) & triu
    diff_mask = (~np.equal.outer(fid_arr, fid_arr)) & triu
    n_same = same_mask.sum()
    n_diff = diff_mask.sum()
    print(f"  Pairs: {n_same} same-folder, {n_diff} cross-folder")

    # Sample cross-folder pairs if too many (for density estimation)
    max_pairs = 50_000
    np.random.seed(42)
    diff_indices = np.argwhere(diff_mask)
    if len(diff_indices) > max_pairs:
        sel = np.random.choice(len(diff_indices), max_pairs, replace=False)
        diff_sample_idx = diff_indices[sel]
    else:
        diff_sample_idx = diff_indices
    same_indices = np.argwhere(same_mask)

    rows = []

    for mname, cfg in MODELS.items():
        slug = cfg["slug"]
        fig, axes = plt.subplots(1, len(cfg["selected_layers"]),
                                 figsize=(4.5 * len(cfg["selected_layers"]), 4),
                                 sharey=True)

        for ax, (li, layer) in zip(axes, enumerate(cfg["selected_layers"])):
            X = load_emb(emb_base, slug, layer)
            pcs, mean_vec = load_pcs(pc_dir, slug, layer)
            if X is None or pcs is None:
                continue

            X_test = X[test_mask]
            sim_base = cosine_sim_matrix(X_test)
            X_abtt = apply_abtt(X_test, pcs, mean_vec)
            sim_abtt = cosine_sim_matrix(X_abtt)

            # Extract pair values
            same_base = sim_base[same_mask]
            same_abtt = sim_abtt[same_mask]
            diff_base = np.array([sim_base[i, j] for i, j in diff_sample_idx])
            diff_abtt = np.array([sim_abtt[i, j] for i, j in diff_sample_idx])

            # Store stats
            rows.append({
                "model": mname, "layer": layer,
                "same_base_mean": float(np.mean(same_base)),
                "same_abtt_mean": float(np.mean(same_abtt)),
                "diff_base_mean": float(np.mean(diff_base)),
                "diff_abtt_mean": float(np.mean(diff_abtt)),
                "gap_base": float(np.mean(same_base) - np.mean(diff_base)),
                "gap_abtt": float(np.mean(same_abtt) - np.mean(diff_abtt)),
                "cgs": float((np.mean(same_abtt) - np.mean(diff_abtt)) -
                             (np.mean(same_base) - np.mean(diff_base))),
            })

            # ── Density plots ─────────────────────────
            bins = np.linspace(-0.2, 1.0, 80)

            # Baseline distributions
            ax.hist(diff_base, bins=bins, density=True, alpha=0.35, color="#e74c3c",
                    label="Diff-folder (baseline)")
            ax.hist(same_base, bins=bins, density=True, alpha=0.35, color="#3498db",
                    label="Same-folder (baseline)")

            # ABTT distributions (step histograms for separation)
            ax.hist(diff_abtt, bins=bins, density=True, alpha=0.7, color="#e74c3c",
                    histtype="step", linewidth=2, linestyle="--",
                    label="Diff-folder (ABTT)")
            ax.hist(same_abtt, bins=bins, density=True, alpha=0.7, color="#3498db",
                    histtype="step", linewidth=2, linestyle="--",
                    label="Same-folder (ABTT)")

            ax.set_xlabel("Cosine Similarity", fontsize=10)
            ax.set_title(cfg["labels"][li].replace("\n", " — "), fontsize=10,
                         fontweight="bold")
            if ax == axes[0]:
                ax.set_ylabel("Density", fontsize=10)
            if li == len(cfg["selected_layers"]) - 1:
                ax.legend(fontsize=7, loc="upper left")

        fig.suptitle(f"{mname} — Cosine Similarity Distributions: "
                     f"Baseline (filled) vs ABTT (outline)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(fig_dir / f"fig_method3_cosine_gap_{cfg['slug']}.pdf",
                    dpi=200, bbox_inches="tight")
        fig.savefig(fig_dir / f"fig_method3_cosine_gap_{cfg['slug']}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {mname}: density plots saved")

    df = pd.DataFrame(rows)
    df.to_csv(fig_dir.parent / "cosine_resolution_gap.csv", index=False)

    # ── Summary figure: gap_baseline vs gap_ABTT per model/layer ──────────
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = {"LaTa": "o", "PhilTa": "s", "LaBSE": "^", "Qwen3-0.6B": "D", "KaLM-mini": "v"}
    colors = {"LaTa": "#1f77b4", "PhilTa": "#ff7f0e", "LaBSE": "#2ca02c",
              "Qwen3-0.6B": "#d62728", "KaLM-mini": "#9467bd"}
    for mname in MODELS:
        sub = df[df["model"] == mname]
        ax.scatter(sub["gap_base"], sub["gap_abtt"],
                   marker=markers[mname], color=colors[mname], s=70,
                   label=mname, edgecolors="k", linewidths=0.5, alpha=0.85)
    # Diagonal line (no change)
    lim = [0, max(df["gap_abtt"].max(), df["gap_base"].max()) * 1.1]
    ax.plot(lim, lim, "k--", alpha=0.4, label="No improvement")
    ax.set_xlabel("Cosine Gap — Baseline (same − diff)", fontsize=11)
    ax.set_ylabel("Cosine Gap — ABTT (same − diff)", fontsize=11)
    ax.set_title("Cosine Resolution: ABTT Always Above the Diagonal",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_method3_gap_scatter.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_method3_gap_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Gap scatter saved")
    return df


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  METHOD 4: EFFECTIVE DIMENSIONALITY                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def method4_effective_dim(emb_base, pc_dir, split_df, test_mask, fig_dir):
    """Participation ratio: d_eff = (Σλ)² / Σ(λ²).

    Shows the embedding space collapses to ~1D in dip layers.
    ABTT restores usable dimensionality by removing the dominant direction(s).
    """

    print("\n" + "="*70)
    print("  METHOD 4: Effective Dimensionality")
    print("="*70)

    rows = []

    for mname, cfg in MODELS.items():
        slug = cfg["slug"]
        for li, layer in enumerate(cfg["selected_layers"]):
            X = load_emb(emb_base, slug, layer)
            pcs, mean_vec = load_pcs(pc_dir, slug, layer)
            if X is None or pcs is None:
                continue

            X_test = X[test_mask]

            # Baseline: center by sample mean, compute eigenvalues via SVD
            Xc_base = X_test - X_test.mean(axis=0)
            _, S_base, _ = np.linalg.svd(Xc_base, full_matrices=False)
            lam_base = S_base ** 2
            d_eff_base = float(lam_base.sum() ** 2 / (lam_base ** 2).sum())

            # After ABTT
            X_abtt = apply_abtt(X_test, pcs, mean_vec)
            Xc_abtt = X_abtt - X_abtt.mean(axis=0)
            _, S_abtt, _ = np.linalg.svd(Xc_abtt, full_matrices=False)
            lam_abtt = S_abtt ** 2
            d_eff_abtt = float(lam_abtt.sum() ** 2 / (lam_abtt ** 2).sum())

            rows.append({
                "model": mname, "layer": layer,
                "layer_label": cfg["labels"][li],
                "d_eff_baseline": d_eff_base,
                "d_eff_abtt": d_eff_abtt,
                "dim_gain": d_eff_abtt / d_eff_base if d_eff_base > 0 else float("inf"),
            })

        print(f"  {mname}: done")

    df = pd.DataFrame(rows)
    df.to_csv(fig_dir.parent / "effective_dimensionality.csv", index=False)

    # ── Visualization: Side-by-side bars, baseline vs ABTT ────────────────
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.5 * len(MODELS), 5),
                             sharey=True)
    bar_width = 0.32

    for ax, (mname, cfg) in zip(axes, MODELS.items()):
        sub = df[df["model"] == mname].sort_values("layer")
        x = np.arange(len(sub))

        ax.bar(x - bar_width/2, sub["d_eff_baseline"].values, bar_width,
               color="#e74c3c", alpha=0.85, label="Baseline",
               edgecolor="k", linewidth=0.5)
        ax.bar(x + bar_width/2, sub["d_eff_abtt"].values, bar_width,
               color="#2ecc71", alpha=0.85, label="After ABTT",
               edgecolor="k", linewidth=0.5)

        # Annotate gain
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.annotate(f"{row['dim_gain']:.0f}×",
                        xy=(i + bar_width/2, row["d_eff_abtt"]),
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color="#27ae60")

        ax.axhline(1, color="red", ls=":", lw=1, alpha=0.5)
        ax.annotate("d=1\n(collapsed)", xy=(0.02, 1), xycoords=("axes fraction", "data"),
                    fontsize=7, color="red", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", " ") for l in cfg["labels"]], fontsize=8)
        ax.set_title(mname, fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Effective Dimensionality\n(participation ratio)", fontsize=10)
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Effective Dimensionality: ABTT Restores the Embedding Space",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_method4_effective_dim.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_method4_effective_dim.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Effective dimensionality figures saved")
    return df


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  METHOD 5: RETRIEVAL RANK SHIFT                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def method5_rank_shift(emb_base, pc_dir, split_df, test_mask, fig_dir):
    """For each 'winnable' test query, compute its true partner's retrieval rank
    before and after ABTT. Show the distribution of rank improvements."""

    print("\n" + "="*70)
    print("  METHOD 5: Retrieval Rank Shift")
    print("="*70)

    fids_test = split_df["folder_id"].values[test_mask]
    winnable_test = split_df["is_winnable"].values[test_mask]
    n_test = int(test_mask.sum())

    rows = []
    rank_data = {}  # for violin plot: {(model, layer): [rank_base, rank_abtt]}

    for mname, cfg in MODELS.items():
        slug = cfg["slug"]
        for li, layer in enumerate(cfg["selected_layers"]):
            X = load_emb(emb_base, slug, layer)
            pcs, mean_vec = load_pcs(pc_dir, slug, layer)
            if X is None or pcs is None:
                continue

            X_test = X[test_mask]
            sim_base = cosine_sim_matrix(X_test)
            X_abtt = apply_abtt(X_test, pcs, mean_vec)
            sim_abtt = cosine_sim_matrix(X_abtt)

            ranks_base = []
            ranks_abtt = []

            for qi in range(n_test):
                if not winnable_test[qi]:
                    continue  # skip singletons — no partner to find
                fid_q = fids_test[qi]
                # Find indices of same-folder documents (excluding self)
                partners = [j for j in range(n_test)
                            if j != qi and fids_test[j] == fid_q]
                if not partners:
                    continue

                # Rank = position of best partner in cosine-sorted list
                # (lower = better, 1 = best possible)
                order_base = np.argsort(-sim_base[qi])  # descending sim
                order_abtt = np.argsort(-sim_abtt[qi])

                # Best partner rank (1-indexed)
                best_rank_base = min(int(np.where(order_base == p)[0][0]) + 1
                                     for p in partners)
                best_rank_abtt = min(int(np.where(order_abtt == p)[0][0]) + 1
                                     for p in partners)
                ranks_base.append(best_rank_base)
                ranks_abtt.append(best_rank_abtt)

            if ranks_base:
                rb = np.array(ranks_base)
                ra = np.array(ranks_abtt)
                improvement = rb - ra  # positive = ABTT improved rank
                rows.append({
                    "model": mname, "layer": layer,
                    "layer_label": cfg["labels"][li],
                    "mean_rank_base": float(np.mean(rb)),
                    "mean_rank_abtt": float(np.mean(ra)),
                    "median_rank_base": float(np.median(rb)),
                    "median_rank_abtt": float(np.median(ra)),
                    "mean_improvement": float(np.mean(improvement)),
                    "pct_improved": float((improvement > 0).mean() * 100),
                    "pct_at_rank1_base": float((rb == 1).mean() * 100),
                    "pct_at_rank1_abtt": float((ra == 1).mean() * 100),
                    "n_queries": len(rb),
                })
                rank_data[(mname, layer)] = (rb, ra)

        print(f"  {mname}: done")

    df = pd.DataFrame(rows)
    df.to_csv(fig_dir.parent / "retrieval_rank_shift.csv", index=False)

    # ── Visualization 5a: Rank@1 improvement bar chart ────────────────────
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.5 * len(MODELS), 5),
                             sharey=True)
    bar_width = 0.32

    for ax, (mname, cfg) in zip(axes, MODELS.items()):
        sub = df[df["model"] == mname].sort_values("layer")
        if sub.empty:
            continue
        x = np.arange(len(sub))

        ax.bar(x - bar_width/2, sub["pct_at_rank1_base"].values, bar_width,
               color="#e74c3c", alpha=0.85, label="Baseline",
               edgecolor="k", linewidth=0.5)
        ax.bar(x + bar_width/2, sub["pct_at_rank1_abtt"].values, bar_width,
               color="#2ecc71", alpha=0.85, label="After ABTT",
               edgecolor="k", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", " ") for l in cfg["labels"]], fontsize=8)
        ax.set_title(mname, fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("% Queries with Partner at Rank 1", fontsize=10)
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Retrieval at Rank 1: ABTT Puts the True Partner on Top",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_method5_rank1_bars.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_method5_rank1_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Visualization 5b: Rank scatter (before vs after) per model ────────
    for mname, cfg in MODELS.items():
        n_layers = len(cfg["selected_layers"])
        fig, axes_row = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4),
                                     sharex=True, sharey=True)
        for ax, (li, layer) in zip(axes_row, enumerate(cfg["selected_layers"])):
            key = (mname, layer)
            if key not in rank_data:
                continue
            rb, ra = rank_data[key]
            # Jitter for visibility
            jx = rb + np.random.normal(0, 0.3, len(rb))
            jy = ra + np.random.normal(0, 0.3, len(ra))
            ax.scatter(jx, jy, s=8, alpha=0.3, c="#3498db", edgecolors="none")
            lim = max(rb.max(), ra.max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", alpha=0.4, lw=1)
            ax.set_xlabel("Rank (baseline)", fontsize=9)
            ax.set_title(cfg["labels"][li].replace("\n", " "), fontsize=10,
                         fontweight="bold")
            if ax == axes_row[0]:
                ax.set_ylabel("Rank (ABTT)", fontsize=9)
            # Annotate % improved
            sub = df[(df["model"]==mname) & (df["layer"]==layer)]
            if not sub.empty:
                pct = sub.iloc[0]["pct_improved"]
                ax.annotate(f"{pct:.0f}% improved",
                            xy=(0.95, 0.95), xycoords="axes fraction",
                            ha="right", va="top", fontsize=9,
                            fontweight="bold", color="#27ae60",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      alpha=0.8))

        fig.suptitle(f"{mname} — Retrieval Rank: Baseline vs ABTT\n"
                     f"(points below diagonal = ABTT improved)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(fig_dir / f"fig_method5_rank_scatter_{cfg['slug']}.pdf",
                    dpi=200, bbox_inches="tight")
        fig.savefig(fig_dir / f"fig_method5_rank_scatter_{cfg['slug']}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("  Rank shift figures saved")
    return df


# ═══════════════════════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Phase 12b deep analysis — 5 methods.")
    p.add_argument("--emb_base", required=True)
    p.add_argument("--pc_dir", required=True)
    p.add_argument("--attr_dir", required=True)
    p.add_argument("--split_csv", required=True)
    p.add_argument("--phase11_csv", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    emb_base = Path(args.emb_base)
    pc_dir = Path(args.pc_dir)
    attr_dir = Path(args.attr_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(args.split_csv)
    test_mask = (split_df["split"] == "test").values

    print("Phase 12b Deep Analysis — 5 Methods\n")

    df_snr = method1_snr(emb_base, pc_dir, split_df, test_mask, fig_dir)
    df_dig = method2_delta_ig(attr_dir, fig_dir)
    df_cos = method3_cosine_gap(emb_base, pc_dir, split_df, test_mask, fig_dir)
    df_dim = method4_effective_dim(emb_base, pc_dir, split_df, test_mask, fig_dir)
    df_rank = method5_rank_shift(emb_base, pc_dir, split_df, test_mask, fig_dir)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY TABLE")
    print("="*70)
    for mname, cfg in MODELS.items():
        print(f"\n  {mname}:")
        for li, layer in enumerate(cfg["selected_layers"]):
            snr_row = df_snr[(df_snr["model"]==mname) & (df_snr["layer"]==layer) &
                             (df_snr["category"]=="Similar")]
            cos_row = df_cos[(df_cos["model"]==mname) & (df_cos["layer"]==layer)]
            dim_row = df_dim[(df_dim["model"]==mname) & (df_dim["layer"]==layer)]
            rank_row = df_rank[(df_rank["model"]==mname) & (df_rank["layer"]==layer)]

            snr_val = snr_row["mean_snr"].values[0] if not snr_row.empty else float("nan")
            gap_b = cos_row["gap_base"].values[0] if not cos_row.empty else float("nan")
            gap_a = cos_row["gap_abtt"].values[0] if not cos_row.empty else float("nan")
            d_base = dim_row["d_eff_baseline"].values[0] if not dim_row.empty else float("nan")
            d_abtt = dim_row["d_eff_abtt"].values[0] if not dim_row.empty else float("nan")
            r1b = rank_row["pct_at_rank1_base"].values[0] if not rank_row.empty else float("nan")
            r1a = rank_row["pct_at_rank1_abtt"].values[0] if not rank_row.empty else float("nan")

            print(f"    {cfg['labels'][li].replace(chr(10),' '):<15s}  "
                  f"SNR={snr_val:.3f}  "
                  f"d_eff={d_base:.1f}→{d_abtt:.1f}  "
                  f"R@1={r1b:.0f}→{r1a:.0f}%  "
                  f"CGS={gap_a-gap_b:+.4f}")

    print("\n\nPhase 12b deep analysis complete.")
    print(f"All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
