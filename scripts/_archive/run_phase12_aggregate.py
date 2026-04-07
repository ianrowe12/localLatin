"""Phase 12 Step 3: Aggregate attribution results and generate figures.

Analyses:
  a) Token-type attribution profiles (heatmap)
  b) PC1 loading vs ABTT importance correlation
  c) IG vs FA agreement
  d) Architecture comparison
  e) Layer-wise attribution evolution
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from token_filtering import classify_token


# Model configs: name → (slug, layers, tokenizer_name, trust_remote_code)
MODEL_CONFIGS = {
    "LaTa": ("bowphs_LaTa", range(1, 13), "bowphs/LaTa", False),
    "PhilTa": ("bowphs_PhilTa", range(1, 13), "bowphs/PhilTa", False),
    "LaBSE": ("sentence-transformers_LaBSE", range(1, 13), "sentence-transformers/LaBSE", False),
    "Qwen3-0.6B": ("Qwen_Qwen3-Embedding-0.6B", range(1, 29), "Qwen/Qwen3-Embedding-0.6B", True),
    "KaLM-mini": (
        "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5",
        range(1, 25),
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        True,
    ),
    "mt5-base": ("google_mt5-base", range(1, 13), "google/mt5-base", False),
}

ARCH_GROUPS = {
    "T5 Encoder": ["LaTa", "PhilTa", "mt5-base"],
    "BERT": ["LaBSE"],
    "Decoder": ["Qwen3-0.6B", "KaLM-mini"],
}


def load_attributions(attr_dir: Path, model_slug: str, layer: int) -> dict:
    """Load attribution .npz for a model/layer."""
    path = attr_dir / model_slug / f"layer{layer}_attributions.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def compute_per_file_spearman(a: np.ndarray, b: np.ndarray, seq_lengths: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation between a and b for each file, respecting seq lengths."""
    n_files = a.shape[0]
    corrs = np.full(n_files, np.nan)
    for i in range(n_files):
        sl = seq_lengths[i]
        if sl < 3:
            continue
        a_i = a[i, :sl]
        b_i = b[i, :sl]
        if np.std(a_i) < 1e-12 or np.std(b_i) < 1e-12:
            continue
        corrs[i] = spearmanr(a_i, b_i).correlation
    return corrs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 12 attribution aggregation.")
    parser.add_argument("--attr_dir", required=True, help="Attribution results directory.")
    parser.add_argument("--results_csv", default="", help="Phase 11 results CSV for AUCROC overlay.")
    parser.add_argument("--out_dir", required=True, help="Output directory for figures and stats.")
    parser.add_argument(
        "--models", default="",
        help="Comma-separated short model names (default: all 5).",
    )
    return parser.parse_args()


def analysis_b_pc1_vs_abtt(attr_dir: Path, out_dir: Path, model_names: List[str]) -> pd.DataFrame:
    """Analysis B: PC1 loading vs ABTT importance correlation per layer."""
    rows = []
    for mname in model_names:
        slug, layers, _, _ = MODEL_CONFIGS[mname]
        for layer in layers:
            data = load_attributions(attr_dir, slug, layer)
            if data is None:
                continue
            corrs = compute_per_file_spearman(
                data["ig_pc1_dot"], data["ig_abtt_norm"], data["seq_lengths"]
            )
            valid = corrs[~np.isnan(corrs)]
            if len(valid) == 0:
                continue
            rows.append({
                "model": mname,
                "layer": layer,
                "mean_spearman": np.mean(valid),
                "median_spearman": np.median(valid),
                "std_spearman": np.std(valid),
                "n_valid": len(valid),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Plot: line per model, x=layer, y=mean_spearman
    fig, ax = plt.subplots(figsize=(10, 5))
    for mname in model_names:
        sub = df[df["model"] == mname]
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["mean_spearman"], marker="o", markersize=3, label=mname)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Spearman(IG_pc1, IG_abtt)")
    ax.set_title("PC1 Loading vs ABTT Importance Correlation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pc1_vs_abtt_correlation.pdf", dpi=150)
    plt.close(fig)
    return df


def analysis_c_ig_fa_agreement(attr_dir: Path, out_dir: Path, model_names: List[str]) -> pd.DataFrame:
    """Analysis C: IG vs FA agreement per model/layer."""
    rows = []
    for mname in model_names:
        slug, layers, _, _ = MODEL_CONFIGS[mname]
        for layer in layers:
            data = load_attributions(attr_dir, slug, layer)
            if data is None:
                continue
            for target in ["pc1_dot", "abtt_norm"]:
                corrs = compute_per_file_spearman(
                    data[f"ig_{target}"], data[f"fa_{target}"], data["seq_lengths"]
                )
                valid = corrs[~np.isnan(corrs)]
                if len(valid) == 0:
                    continue
                rows.append({
                    "model": mname, "layer": layer, "target": target,
                    "mean_ig_fa_spearman": np.mean(valid),
                    "median_ig_fa_spearman": np.median(valid),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Box plot per model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, target in zip(axes, ["pc1_dot", "abtt_norm"]):
        sub = df[df["target"] == target]
        data_per_model = [sub[sub["model"] == m]["mean_ig_fa_spearman"].values for m in model_names]
        ax.boxplot(data_per_model, labels=model_names, vert=True)
        ax.set_title(f"IG-FA Agreement: {target}")
        ax.set_ylabel("Spearman(IG, FA)")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ig_fa_agreement.pdf", dpi=150)
    plt.close(fig)
    return df


def analysis_a_token_profiles(
    attr_dir: Path, out_dir: Path, model_names: List[str]
) -> None:
    """Analysis A: Token-type attribution profiles (heatmap per model)."""
    categories = ["content", "short_subword", "punctuation", "number", "empty"]

    for mname in model_names:
        slug, layers, tok_name, trust = MODEL_CONFIGS[mname]
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust)
        except Exception as e:
            print(f"  WARNING: Cannot load tokenizer for {mname}: {e}")
            continue

        layer_list = list(layers)
        heatmap = np.full((len(layer_list), len(categories)), np.nan)

        for li, layer in enumerate(layer_list):
            data = load_attributions(attr_dir, slug, layer)
            if data is None:
                continue

            # Aggregate mean |attribution| per token category
            cat_sums = {c: [] for c in categories}
            n_files = data["ig_pc1_dot"].shape[0]

            for fi in range(n_files):
                sl = data["seq_lengths"][fi]
                if sl < 1:
                    continue
                ids = data["input_ids"][fi, :sl]
                attrs = np.abs(data["ig_pc1_dot"][fi, :sl])
                tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
                for tok_str, attr_val in zip(tokens, attrs):
                    cat = classify_token(tok_str)
                    if cat in cat_sums:
                        cat_sums[cat].append(float(attr_val))

            for ci, cat in enumerate(categories):
                if cat_sums[cat]:
                    heatmap[li, ci] = np.mean(cat_sums[cat])

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, max(4, len(layer_list) * 0.35)))
        im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_yticks(range(len(layer_list)))
        ax.set_yticklabels(layer_list)
        ax.set_ylabel("Layer")
        ax.set_title(f"Mean |IG_pc1| by Token Type: {mname}")
        fig.colorbar(im, ax=ax, label="Mean |attribution|")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_token_profile_{slug}.pdf", dpi=150)
        plt.close(fig)
        print(f"  Token profile heatmap for {mname} saved")


def analysis_e_layerwise_evolution(
    attr_dir: Path, out_dir: Path, model_names: List[str],
    results_csv: str,
) -> None:
    """Analysis E: Layer-wise attribution of frequent tokens overlaid with AUCROC."""
    # Load AUCROC results if available
    aucroc_data = {}
    if results_csv and Path(results_csv).exists():
        rdf = pd.read_csv(results_csv)
        for mname in model_names:
            slug, _, _, _ = MODEL_CONFIGS[mname]
            model_hf = {
                "LaTa": "bowphs/LaTa", "PhilTa": "bowphs/PhilTa",
                "LaBSE": "sentence-transformers/LaBSE",
                "Qwen3-0.6B": "Qwen/Qwen3-Embedding-0.6B",
                "KaLM-mini": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
                "mt5-base": "google/mt5-base",
            }.get(mname, "")
            sub = rdf[(rdf["model"] == model_hf) & (rdf["method"] == "baseline") & (rdf["repr"] == "hidden")]
            if not sub.empty:
                aucroc_data[mname] = sub.set_index("layer")["aucroc"].to_dict()

    for mname in model_names:
        slug, layers, tok_name, trust = MODEL_CONFIGS[mname]
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust)
        except Exception:
            continue

        layer_list = list(layers)
        mean_attr_content = []
        mean_attr_short = []

        for layer in layer_list:
            data = load_attributions(attr_dir, slug, layer)
            if data is None:
                mean_attr_content.append(np.nan)
                mean_attr_short.append(np.nan)
                continue

            content_attrs = []
            short_attrs = []
            n_files = data["ig_pc1_dot"].shape[0]

            for fi in range(n_files):
                sl = data["seq_lengths"][fi]
                if sl < 1:
                    continue
                ids = data["input_ids"][fi, :sl]
                attrs = np.abs(data["ig_pc1_dot"][fi, :sl])
                tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
                for tok_str, attr_val in zip(tokens, attrs):
                    cat = classify_token(tok_str)
                    if cat == "content":
                        content_attrs.append(float(attr_val))
                    elif cat == "short_subword":
                        short_attrs.append(float(attr_val))

            mean_attr_content.append(np.mean(content_attrs) if content_attrs else np.nan)
            mean_attr_short.append(np.mean(short_attrs) if short_attrs else np.nan)

        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(layer_list, mean_attr_content, "b-o", markersize=3, label="Content tokens |IG_pc1|")
        ax1.plot(layer_list, mean_attr_short, "r-s", markersize=3, label="Short subword |IG_pc1|")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Mean |IG attribution|", color="black")
        ax1.legend(loc="upper left", fontsize=8)

        if mname in aucroc_data:
            ax2 = ax1.twinx()
            auc_layers = sorted(aucroc_data[mname].keys())
            auc_vals = [aucroc_data[mname][l] for l in auc_layers]
            ax2.plot(auc_layers, auc_vals, "k--", alpha=0.5, label="Baseline AUCROC")
            ax2.set_ylabel("AUCROC", color="gray")
            ax2.legend(loc="upper right", fontsize=8)

        ax1.set_title(f"Layer-wise Token Attribution: {mname}")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_layerwise_{slug}.pdf", dpi=150)
        plt.close(fig)
        print(f"  Layer-wise plot for {mname} saved")


def analysis_d_architecture_comparison(
    attr_dir: Path, out_dir: Path, model_names: List[str]
) -> None:
    """Analysis D: Compare mean |IG_pc1| on short subwords across architectures."""
    arch_data = {}
    for arch_name, members in ARCH_GROUPS.items():
        vals = []
        for mname in members:
            if mname not in model_names:
                continue
            slug, layers, tok_name, trust = MODEL_CONFIGS[mname]
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust)
            except Exception:
                continue

            for layer in layers:
                data = load_attributions(attr_dir, slug, layer)
                if data is None:
                    continue
                n_files = data["ig_pc1_dot"].shape[0]
                for fi in range(n_files):
                    sl = data["seq_lengths"][fi]
                    if sl < 1:
                        continue
                    ids = data["input_ids"][fi, :sl]
                    attrs = np.abs(data["ig_pc1_dot"][fi, :sl])
                    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
                    for tok_str, attr_val in zip(tokens, attrs):
                        if classify_token(tok_str) == "short_subword":
                            vals.append(float(attr_val))
        if vals:
            arch_data[arch_name] = vals

    if not arch_data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(arch_data.keys())
    means = [np.mean(v) for v in arch_data.values()]
    stds = [np.std(v) / np.sqrt(len(v)) for v in arch_data.values()]
    ax.bar(labels, means, yerr=stds, capsize=5, color=["#1f77b4", "#2ca02c", "#d62728"])
    ax.set_ylabel("Mean |IG_pc1| on short subwords")
    ax.set_title("PC1 Loading of Short Subwords by Architecture")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_arch_comparison.pdf", dpi=150)
    plt.close(fig)
    print("  Architecture comparison saved")


def main() -> None:
    args = parse_args()
    attr_dir = Path(args.attr_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = list(MODEL_CONFIGS.keys())

    print("=== Analysis A: Token-type profiles ===")
    analysis_a_token_profiles(attr_dir, out_dir, model_names)

    print("\n=== Analysis B: PC1 vs ABTT correlation ===")
    df_b = analysis_b_pc1_vs_abtt(attr_dir, out_dir, model_names)
    if not df_b.empty:
        df_b.to_csv(out_dir / "stats_pc1_vs_abtt.csv", index=False)

    print("\n=== Analysis C: IG-FA agreement ===")
    df_c = analysis_c_ig_fa_agreement(attr_dir, out_dir, model_names)
    if not df_c.empty:
        df_c.to_csv(out_dir / "stats_ig_fa_agreement.csv", index=False)

    print("\n=== Analysis D: Architecture comparison ===")
    analysis_d_architecture_comparison(attr_dir, out_dir, model_names)

    print("\n=== Analysis E: Layer-wise evolution ===")
    analysis_e_layerwise_evolution(attr_dir, out_dir, model_names, args.results_csv)

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
