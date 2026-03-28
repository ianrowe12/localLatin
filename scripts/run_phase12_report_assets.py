"""Generate report assets for analysis_phase12.md.

This script regenerates inline PNG figures and summary CSV tables from
Phase 12 outputs so the analysis markdown stays reproducible.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from token_filtering import classify_token


MODEL_ORDER = ["LaTa", "PhilTa", "LaBSE", "Qwen3-0.6B", "KaLM-mini", "mt5-base"]

MODEL_CONFIGS = {
    "LaTa": {
        "slug": "bowphs_LaTa",
        "hf_model": "bowphs/LaTa",
        "tokenizer": "bowphs/LaTa",
        "trust_remote_code": False,
        "report_layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "arch": "T5 Encoder",
    },
    "PhilTa": {
        "slug": "bowphs_PhilTa",
        "hf_model": "bowphs/PhilTa",
        "tokenizer": "bowphs/PhilTa",
        "trust_remote_code": False,
        "report_layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "arch": "T5 Encoder",
    },
    "LaBSE": {
        "slug": "sentence-transformers_LaBSE",
        "hf_model": "sentence-transformers/LaBSE",
        "tokenizer": "sentence-transformers/LaBSE",
        "trust_remote_code": False,
        "report_layers": [1, 6, 9, 11],
        "arch": "BERT",
    },
    "Qwen3-0.6B": {
        "slug": "Qwen_Qwen3-Embedding-0.6B",
        "hf_model": "Qwen/Qwen3-Embedding-0.6B",
        "tokenizer": "Qwen/Qwen3-Embedding-0.6B",
        "trust_remote_code": True,
        "report_layers": [1, 4, 14, 28],
        "arch": "Decoder",
    },
    "KaLM-mini": {
        "slug": "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5",
        "hf_model": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "tokenizer": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "trust_remote_code": True,
        "report_layers": [1, 5, 13, 23],
        "arch": "Decoder",
    },
    "mt5-base": {
        "slug": "google_mt5-base",
        "hf_model": "google/mt5-base",
        "tokenizer": "google/mt5-base",
        "trust_remote_code": False,
        "report_layers": [1, 4, 8, 12],
        "arch": "T5 Encoder",
    },
}

ARCH_ORDER = ["T5 Encoder", "BERT", "Decoder"]
TOKEN_CATEGORIES = ["content", "short_subword", "punctuation", "number", "empty"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 12 report assets.")
    parser.add_argument("--attr_dir", required=True, help="runs/phase12/attributions directory")
    parser.add_argument("--figures_dir", required=True, help="runs/phase12/figures directory")
    parser.add_argument("--stats_pc1_csv", required=True, help="stats_pc1_vs_abtt.csv")
    parser.add_argument("--stats_igfa_csv", required=True, help="stats_ig_fa_agreement.csv")
    parser.add_argument("--phase11_csv", required=True, help="runs/phase11/phase11_results.csv")
    parser.add_argument("--sample_csv", required=True, help="runs/phase12/phase12_samples.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for report assets")
    return parser.parse_args()


def parse_layer(path: Path) -> int:
    match = re.search(r"layer(\d+)_attributions\.npz$", path.name)
    if match is None:
        raise ValueError(f"Could not parse layer from {path}")
    return int(match.group(1))


def model_attr_files(attr_dir: Path, model_name: str) -> List[Path]:
    slug = MODEL_CONFIGS[model_name]["slug"]
    return sorted((attr_dir / slug).glob("layer*_attributions.npz"), key=parse_layer)


def load_attr_npz(path: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(path))


def collect_coverage(attr_dir: Path, sample_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_df = pd.read_csv(sample_csv)
    sample_category = Counter(sample_df["category"])
    sample_split = Counter(sample_df["split"])
    expected_layers = {m: len(MODEL_CONFIGS[m]["report_layers"]) for m in MODEL_ORDER}

    rows = []
    for model_name in MODEL_ORDER:
        files = model_attr_files(attr_dir, model_name)
        layers = [parse_layer(p) for p in files]
        n_files = np.nan
        seq_len = np.nan
        if files:
            data0 = load_attr_npz(files[0])
            n_files = int(data0["ig_pc1_dot"].shape[0])
            seq_len = int(data0["ig_pc1_dot"].shape[1])
        rows.append(
            {
                "model": model_name,
                "architecture": MODEL_CONFIGS[model_name]["arch"],
                "layers_expected": expected_layers[model_name],
                "layers_found": len(files),
                "layers_missing": expected_layers[model_name] - len(files),
                "layer_list_found": ",".join(str(x) for x in layers),
                "files_per_layer": n_files,
                "max_seq_len": seq_len,
            }
        )

    coverage_df = pd.DataFrame(rows)

    table01_rows = [
        {
            "metric": "n_models",
            "value": len(MODEL_ORDER),
            "notes": ", ".join(MODEL_ORDER),
        },
        {"metric": "n_samples_total", "value": int(len(sample_df)), "notes": "From phase12_samples.csv"},
        {
            "metric": "n_samples_similar",
            "value": int(sample_category.get("similar", 0)),
            "notes": "Query set with same-folder partners",
        },
        {
            "metric": "n_samples_dissimilar",
            "value": int(sample_category.get("dissimilar", 0)),
            "notes": "Singleton/no-partner sample set",
        },
        {"metric": "n_samples_train", "value": int(sample_split.get("train", 0)), "notes": "Train split"},
        {"metric": "n_samples_test", "value": int(sample_split.get("test", 0)), "notes": "Test split"},
    ]

    for _, row in coverage_df.iterrows():
        table01_rows.append(
            {
                "metric": f"{row['model']}_layers_found",
                "value": int(row["layers_found"]),
                "notes": f"Expected {int(row['layers_expected'])}; found layers {row['layer_list_found']}",
            }
        )
        table01_rows.append(
            {
                "metric": f"{row['model']}_files_per_layer",
                "value": int(row["files_per_layer"]),
                "notes": "Number of texts attributed at each saved layer",
            }
        )

    table01 = pd.DataFrame(table01_rows)
    return coverage_df, table01


def plot_fig01_completion(coverage_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(coverage_df))
    w = 0.35

    axes[0].bar(x - w / 2, coverage_df["layers_expected"], width=w, label="Expected", color="#9aa0a6")
    axes[0].bar(x + w / 2, coverage_df["layers_found"], width=w, label="Found", color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(coverage_df["model"], rotation=25, ha="right")
    axes[0].set_ylabel("Layer files")
    axes[0].set_title("Layer Coverage by Model")
    axes[0].legend()
    for i, val in enumerate(coverage_df["layers_missing"]):
        if int(val) == 0:
            axes[0].text(i + w / 2, coverage_df["layers_found"].iloc[i] + 0.1, "OK", ha="center", va="bottom", fontsize=8)
        else:
            axes[0].text(i + w / 2, coverage_df["layers_found"].iloc[i] + 0.1, f"-{int(val)}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, coverage_df["files_per_layer"], color="#2ca02c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(coverage_df["model"], rotation=25, ha="right")
    axes[1].set_ylabel("Texts per layer")
    axes[1].set_title("Attribution Scope (Files per Saved Layer)")
    for i, val in enumerate(coverage_df["files_per_layer"]):
        axes[1].text(i, float(val) + max(5, float(val) * 0.01), f"{int(val)}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Phase 12 Completion and Coverage", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_table02_pc1(stats_pc1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        sub = stats_pc1[stats_pc1["model"] == model_name].sort_values("layer")
        if sub.empty:
            continue
        min_row = sub.loc[sub["mean_spearman"].idxmin()]
        max_row = sub.loc[sub["mean_spearman"].idxmax()]
        rows.append(
            {
                "model": model_name,
                "n_layers": int(len(sub)),
                "layers": ",".join(str(int(v)) for v in sub["layer"].tolist()),
                "mean_spearman_avg": float(sub["mean_spearman"].mean()),
                "mean_spearman_median": float(sub["mean_spearman"].median()),
                "n_negative_layers": int((sub["mean_spearman"] < 0).sum()),
                "n_positive_layers": int((sub["mean_spearman"] > 0).sum()),
                "min_layer": int(min_row["layer"]),
                "min_mean_spearman": float(min_row["mean_spearman"]),
                "max_layer": int(max_row["layer"]),
                "max_mean_spearman": float(max_row["mean_spearman"]),
                "n_valid_per_layer": int(sub["n_valid"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_fig02_pc1_vs_abtt(stats_pc1: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for model_name in MODEL_ORDER:
        sub = stats_pc1[stats_pc1["model"] == model_name].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(
            sub["layer"],
            sub["mean_spearman"],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=model_name,
        )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.65, linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Spearman(IG_pc1_dot, IG_abtt_norm)")
    ax.set_title("PC1-Loading vs ABTT-Surviving Token Importance")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_table03_ig_fa(stats_igfa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        for target in ["pc1_dot", "abtt_norm"]:
            sub = stats_igfa[(stats_igfa["model"] == model_name) & (stats_igfa["target"] == target)].sort_values("layer")
            if sub.empty:
                continue
            min_row = sub.loc[sub["mean_ig_fa_spearman"].idxmin()]
            max_row = sub.loc[sub["mean_ig_fa_spearman"].idxmax()]
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "n_layers": int(len(sub)),
                    "mean_spearman_avg": float(sub["mean_ig_fa_spearman"].mean()),
                    "median_spearman_avg": float(sub["median_ig_fa_spearman"].mean()),
                    "min_layer": int(min_row["layer"]),
                    "min_mean_spearman": float(min_row["mean_ig_fa_spearman"]),
                    "max_layer": int(max_row["layer"]),
                    "max_mean_spearman": float(max_row["mean_ig_fa_spearman"]),
                }
            )
    return pd.DataFrame(rows)


def plot_fig03_ig_fa(stats_igfa: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, target in zip(axes, ["pc1_dot", "abtt_norm"]):
        subset = stats_igfa[stats_igfa["target"] == target]
        x = np.arange(len(MODEL_ORDER))
        means = []
        mins = []
        maxs = []
        for model_name in MODEL_ORDER:
            sub = subset[subset["model"] == model_name]
            if sub.empty:
                means.append(np.nan)
                mins.append(np.nan)
                maxs.append(np.nan)
            else:
                means.append(float(sub["mean_ig_fa_spearman"].mean()))
                mins.append(float(sub["mean_ig_fa_spearman"].min()))
                maxs.append(float(sub["mean_ig_fa_spearman"].max()))
        means_arr = np.array(means)
        err_low = means_arr - np.array(mins)
        err_high = np.array(maxs) - means_arr
        ax.bar(x, means_arr, color="#1f77b4" if target == "pc1_dot" else "#ff7f0e")
        ax.errorbar(x, means_arr, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
        ax.set_title(f"IG vs FA: {target}")
        ax.set_ylabel("Spearman (layer-mean with min/max whiskers)")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.6, linewidth=1)

    fig.suptitle("Attribution Method Agreement by Target")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def iter_tokenized_attributions(
    tokenizer: AutoTokenizer, data: Dict[str, np.ndarray]
) -> Iterable[Tuple[str, float]]:
    n_files = data["ig_pc1_dot"].shape[0]
    for fi in range(n_files):
        seq_len = int(data["seq_lengths"][fi])
        if seq_len < 1:
            continue
        ids = data["input_ids"][fi, :seq_len]
        attrs = np.abs(data["ig_pc1_dot"][fi, :seq_len])
        toks = tokenizer.convert_ids_to_tokens(ids.tolist())
        for tok, attr in zip(toks, attrs):
            yield classify_token(tok), float(attr)


def compute_token_profiles(attr_dir: Path) -> Tuple[Dict[str, Dict[int, Dict[str, float]]], Dict[str, Dict[str, float]]]:
    """Return per-model-per-layer category means and per-arch short_subword stats."""
    model_layer_means: Dict[str, Dict[int, Dict[str, float]]] = {}
    arch_short_vals: Dict[str, List[float]] = {k: [] for k in ARCH_ORDER}

    for model_name in MODEL_ORDER:
        cfg = MODEL_CONFIGS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["tokenizer"],
            trust_remote_code=cfg["trust_remote_code"],
        )
        layer_means: Dict[int, Dict[str, float]] = {}
        for npz_path in model_attr_files(attr_dir, model_name):
            layer = parse_layer(npz_path)
            data = load_attr_npz(npz_path)
            sums = {cat: 0.0 for cat in TOKEN_CATEGORIES}
            counts = {cat: 0 for cat in TOKEN_CATEGORIES}
            for cat, val in iter_tokenized_attributions(tokenizer, data):
                if cat not in sums:
                    continue
                sums[cat] += val
                counts[cat] += 1
            means = {}
            for cat in TOKEN_CATEGORIES:
                means[cat] = sums[cat] / counts[cat] if counts[cat] > 0 else np.nan
            layer_means[layer] = means
            if counts["short_subword"] > 0:
                arch_short_vals[cfg["arch"]].append(means["short_subword"])
        model_layer_means[model_name] = layer_means

    arch_summary = {}
    for arch in ARCH_ORDER:
        vals = np.array(arch_short_vals[arch], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            arch_summary[arch] = {"mean": np.nan, "sem": np.nan}
        else:
            arch_summary[arch] = {
                "mean": float(np.mean(vals)),
                "sem": float(np.std(vals) / np.sqrt(len(vals))),
            }
    return model_layer_means, arch_summary


def plot_fig04_token_profiles(
    model_layer_means: Dict[str, Dict[int, Dict[str, float]]], out_path: Path
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes_list = axes.flatten()
    vmax = 0.0
    for model_name in MODEL_ORDER:
        layers = sorted(model_layer_means[model_name].keys())
        if not layers:
            continue
        mat = np.array([[model_layer_means[model_name][l][c] for c in TOKEN_CATEGORIES] for l in layers], dtype=float)
        vmax = max(vmax, np.nanmax(mat))

    for i, model_name in enumerate(MODEL_ORDER):
        ax = axes_list[i]
        layers = sorted(model_layer_means[model_name].keys())
        if not layers:
            ax.axis("off")
            continue
        mat = np.array([[model_layer_means[model_name][l][c] for c in TOKEN_CATEGORIES] for l in layers], dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax if vmax > 0 else None)
        ax.set_title(model_name)
        ax.set_xticks(range(len(TOKEN_CATEGORIES)))
        ax.set_xticklabels(TOKEN_CATEGORIES, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=8)
        ax.set_ylabel("Layer")

    axes_list[-1].axis("off")
    cbar_ax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    fig.colorbar(im, cax=cbar_ax, label="Mean |IG_pc1_dot|")
    fig.suptitle("Token-Type Attribution Profiles (Mean |IG_pc1_dot|)", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_fig05_arch(arch_summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    labels = ARCH_ORDER
    means = [arch_summary[l]["mean"] for l in labels]
    sems = [arch_summary[l]["sem"] for l in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4e79a7", "#59a14f", "#e15759"]
    ax.bar(labels, means, yerr=sems, capsize=6, color=colors)
    ax.set_ylabel("Mean |IG_pc1_dot| on short_subword tokens")
    ax.set_title("Architecture Comparison: Short-Subword PC1 Loading")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_fig06_layerwise(
    model_layer_means: Dict[str, Dict[int, Dict[str, float]]],
    phase11_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes_list = axes.flatten()

    for i, model_name in enumerate(MODEL_ORDER):
        ax = axes_list[i]
        cfg = MODEL_CONFIGS[model_name]
        layer_map = model_layer_means[model_name]
        if not layer_map:
            ax.axis("off")
            continue
        layers = sorted(layer_map.keys())
        content = [layer_map[l]["content"] for l in layers]
        short = [layer_map[l]["short_subword"] for l in layers]
        ax.plot(layers, content, "o-", color="#1f77b4", label="Content |IG_pc1|", markersize=3)
        ax.plot(layers, short, "s-", color="#d62728", label="Short-subword |IG_pc1|", markersize=3)
        ax.set_title(model_name)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean |IG_pc1_dot|")

        auc_sub = phase11_df[
            (phase11_df["model"] == cfg["hf_model"])
            & (phase11_df["method"] == "baseline")
            & (phase11_df["repr"] == "hidden")
        ].sort_values("layer")
        if not auc_sub.empty:
            ax2 = ax.twinx()
            ax2.plot(auc_sub["layer"], auc_sub["aucroc"], "--", color="black", alpha=0.6, label="Baseline AUCROC")
            ax2.set_ylabel("AUCROC")
            ax2.set_ylim(0.4, 1.0)

        h1, l1 = ax.get_legend_handles_labels()
        if i == 0:
            ax.legend(h1, l1, loc="upper left", fontsize=8)

    axes_list[-1].axis("off")
    fig.suptitle("Layer-wise Mechanism Signals vs Baseline Retrieval (Phase 11 AUCROC)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def normalize_token_for_report(token: str) -> str:
    token = token.strip()
    if not token:
        return "<empty>"
    specials = {"<pad>", "</s>", "<s>", "[CLS]", "[SEP]", "[PAD]", "<|endoftext|>"}
    if token in specials:
        return token
    token = token.lstrip("▁Ġ")
    if token.startswith("##"):
        token = token[2:]
    token = token.strip()
    return token if token else "<empty>"


def compute_layer_token_rankings(
    tokenizer: AutoTokenizer,
    data: Dict[str, np.ndarray],
    attr_key: str,
    top_k: int = 12,
) -> List[Dict[str, object]]:
    token_sum = {}
    token_count = {}
    n_files = data[attr_key].shape[0]
    for fi in range(n_files):
        seq_len = int(data["seq_lengths"][fi])
        if seq_len < 1:
            continue
        ids = data["input_ids"][fi, :seq_len]
        attrs = np.abs(data[attr_key][fi, :seq_len])
        tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
        for tok_raw, attr in zip(tokens, attrs):
            tok = normalize_token_for_report(tok_raw)
            if tok in {"<pad>", "</s>", "<s>", "[CLS]", "[SEP]", "[PAD]", "<empty>"}:
                continue
            token_sum[tok] = token_sum.get(tok, 0.0) + float(attr)
            token_count[tok] = token_count.get(tok, 0) + 1

    rows = []
    for tok, s in token_sum.items():
        c = token_count[tok]
        rows.append(
            {
                "token": tok,
                "category": classify_token(tok),
                "mean_abs_attr": s / c if c > 0 else 0.0,
                "count": c,
            }
        )
    rows = sorted(rows, key=lambda x: x["mean_abs_attr"], reverse=True)[:top_k]
    return rows


def build_token_granularity_tables(
    attr_dir: Path, phase11_df: pd.DataFrame, out_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for model_name in MODEL_ORDER:
        cfg = MODEL_CONFIGS[model_name]
        auc_sub = phase11_df[
            (phase11_df["model"] == cfg["hf_model"])
            & (phase11_df["method"] == "baseline")
            & (phase11_df["repr"] == "hidden")
        ].sort_values("aucroc")
        if auc_sub.empty:
            focus_layer = cfg["report_layers"][0]
        else:
            focus_layer = int(auc_sub.iloc[0]["layer"])
        npz_path = attr_dir / cfg["slug"] / f"layer{focus_layer}_attributions.npz"
        if not npz_path.exists():
            files = model_attr_files(attr_dir, model_name)
            if not files:
                continue
            npz_path = files[0]
            focus_layer = parse_layer(npz_path)
        data = load_attr_npz(npz_path)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["tokenizer"],
            trust_remote_code=cfg["trust_remote_code"],
        )

        for target in ["ig_pc1_dot", "ig_abtt_norm"]:
            top_rows = compute_layer_token_rankings(tokenizer, data, target, top_k=12)
            target_name = "pc1_dot" if target == "ig_pc1_dot" else "abtt_norm"
            for rank, r in enumerate(top_rows, start=1):
                rows.append(
                    {
                        "model": model_name,
                        "focus_layer": focus_layer,
                        "target": target_name,
                        "rank": rank,
                        "token": r["token"],
                        "category": r["category"],
                        "mean_abs_attr": r["mean_abs_attr"],
                        "count": r["count"],
                    }
                )
            cat_counts = Counter(x["category"] for x in top_rows)
            summary_rows.append(
                {
                    "model": model_name,
                    "focus_layer": focus_layer,
                    "target": target_name,
                    "n_top_tokens": len(top_rows),
                    "n_content": cat_counts.get("content", 0),
                    "n_short_subword": cat_counts.get("short_subword", 0),
                    "n_punctuation": cat_counts.get("punctuation", 0),
                    "n_number": cat_counts.get("number", 0),
                    "n_empty": cat_counts.get("empty", 0),
                }
            )

    detailed = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    overlap_rows = []
    for model_name in detailed["model"].unique():
        sub = detailed[detailed["model"] == model_name]
        pc1_tokens = set(sub[sub["target"] == "pc1_dot"]["token"].tolist())
        abtt_tokens = set(sub[sub["target"] == "abtt_norm"]["token"].tolist())
        inter = len(pc1_tokens & abtt_tokens)
        union = len(pc1_tokens | abtt_tokens)
        jaccard = (inter / union) if union > 0 else np.nan
        focus_layers = sorted(sub["focus_layer"].unique().tolist())
        overlap_rows.append(
            {
                "model": model_name,
                "focus_layer": ",".join(str(int(x)) for x in focus_layers),
                "pc1_top_token_count": len(pc1_tokens),
                "abtt_top_token_count": len(abtt_tokens),
                "intersection_count": inter,
                "union_count": union,
                "jaccard_overlap": jaccard,
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    detailed.to_csv(out_dir / "table05_token_level_top_tokens.csv", index=False)
    summary.to_csv(out_dir / "table06_token_level_category_summary.csv", index=False)
    overlap.to_csv(out_dir / "table07_token_overlap_pc1_vs_abtt.csv", index=False)
    return detailed, summary, overlap


def build_table04_guardrails(table02: pd.DataFrame, table03: pd.DataFrame) -> pd.DataFrame:
    t2 = table02.set_index("model")
    t3 = table03.set_index(["model", "target"])
    rows = [
        {
            "claim_id": "C1",
            "claim": "ABTT is a robust performance fix across models/layers (from Phase 11).",
            "status": "Supported",
            "evidence": "Phase 11 showed universal ABTT lift; Phase 12 run-complete coverage supports mechanism follow-up.",
            "wording_guideline": "Use strong wording for retrieval performance, not for universal token mechanism.",
        },
        {
            "claim_id": "C2",
            "claim": "PC1 vs ABTT importance is consistently anti-correlated across all architectures.",
            "status": "Not supported",
            "evidence": (
                f"LaTa avg={t2.at['LaTa', 'mean_spearman_avg']:.3f}, PhilTa avg={t2.at['PhilTa', 'mean_spearman_avg']:.3f} "
                f"but Qwen avg={t2.at['Qwen3-0.6B', 'mean_spearman_avg']:.3f}, KaLM avg={t2.at['KaLM-mini', 'mean_spearman_avg']:.3f}."
            ),
            "wording_guideline": "State mechanism as architecture- and layer-conditional.",
        },
        {
            "claim_id": "C3",
            "claim": "T5 dip-layer behavior shows strongest anti-correlation pattern.",
            "status": "Supported",
            "evidence": (
                f"PhilTa min layer {int(t2.at['PhilTa', 'min_layer'])} = {t2.at['PhilTa', 'min_mean_spearman']:.3f}; "
                f"LaTa min layer {int(t2.at['LaTa', 'min_layer'])} = {t2.at['LaTa', 'min_mean_spearman']:.3f}."
            ),
            "wording_guideline": "Use as central mechanistic evidence for T5 layers.",
        },
        {
            "claim_id": "C4",
            "claim": "IG and FA are sufficiently aligned for the PC1 target.",
            "status": "Supported",
            "evidence": (
                "Mean IG-FA(pc1_dot): "
                f"LaTa={t3.at[('LaTa', 'pc1_dot'), 'mean_spearman_avg']:.3f}, "
                f"PhilTa={t3.at[('PhilTa', 'pc1_dot'), 'mean_spearman_avg']:.3f}, "
                f"LaBSE={t3.at[('LaBSE', 'pc1_dot'), 'mean_spearman_avg']:.3f}."
            ),
            "wording_guideline": "Treat PC1-target interpretations as higher confidence.",
        },
        {
            "claim_id": "C5",
            "claim": "IG and FA strongly validate ABTT-norm attributions.",
            "status": "Partially supported",
            "evidence": (
                "Mean IG-FA(abtt_norm) is low/moderate: "
                f"LaTa={t3.at[('LaTa', 'abtt_norm'), 'mean_spearman_avg']:.3f}, "
                f"PhilTa={t3.at[('PhilTa', 'abtt_norm'), 'mean_spearman_avg']:.3f}, "
                f"Qwen={t3.at[('Qwen3-0.6B', 'abtt_norm'), 'mean_spearman_avg']:.3f}."
            ),
            "wording_guideline": "Use cautious language for ABTT-norm token mechanism.",
        },
        {
            "claim_id": "C6",
            "claim": "One universal token-importance story explains all models.",
            "status": "Not supported",
            "evidence": "Sign flips across layers in Qwen/KaLM and mixed LaBSE trends.",
            "wording_guideline": "Frame paper contribution as mechanism taxonomy, not single mechanism.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    attr_dir = Path(args.attr_dir)
    stats_pc1_csv = Path(args.stats_pc1_csv)
    stats_igfa_csv = Path(args.stats_igfa_csv)
    phase11_csv = Path(args.phase11_csv)
    sample_csv = Path(args.sample_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_pc1 = pd.read_csv(stats_pc1_csv)
    stats_igfa = pd.read_csv(stats_igfa_csv)
    phase11_df = pd.read_csv(phase11_csv)

    coverage_df, table01 = collect_coverage(attr_dir, sample_csv)
    table02 = build_table02_pc1(stats_pc1)
    table03 = build_table03_ig_fa(stats_igfa)
    table04 = build_table04_guardrails(table02, table03)

    table01.to_csv(out_dir / "table01_dataset_and_run_scope.csv", index=False)
    table02.to_csv(out_dir / "table02_pc1_abtt_key_stats.csv", index=False)
    table03.to_csv(out_dir / "table03_ig_fa_summary.csv", index=False)
    table04.to_csv(out_dir / "table04_claim_guardrails.csv", index=False)
    build_token_granularity_tables(attr_dir, phase11_df, out_dir)

    plot_fig01_completion(coverage_df, out_dir / "fig01_phase12_completion_and_coverage.png")
    plot_fig02_pc1_vs_abtt(stats_pc1, out_dir / "fig02_pc1_vs_abtt_by_layer.png")
    plot_fig03_ig_fa(stats_igfa, out_dir / "fig03_ig_fa_agreement_targets.png")

    model_layer_means, arch_summary = compute_token_profiles(attr_dir)
    plot_fig04_token_profiles(model_layer_means, out_dir / "fig04_token_profile_selected_models.png")
    plot_fig05_arch(arch_summary, out_dir / "fig05_architecture_short_subword_loading.png")
    plot_fig06_layerwise(model_layer_means, phase11_df, out_dir / "fig06_layerwise_mechanism_vs_auc.png")

    print(f"Wrote report assets to: {out_dir}")
    for name in [
        "table01_dataset_and_run_scope.csv",
        "table02_pc1_abtt_key_stats.csv",
        "table03_ig_fa_summary.csv",
        "table04_claim_guardrails.csv",
        "table05_token_level_top_tokens.csv",
        "table06_token_level_category_summary.csv",
        "table07_token_overlap_pc1_vs_abtt.csv",
        "fig01_phase12_completion_and_coverage.png",
        "fig02_pc1_vs_abtt_by_layer.png",
        "fig03_ig_fa_agreement_targets.png",
        "fig04_token_profile_selected_models.png",
        "fig05_architecture_short_subword_loading.png",
        "fig06_layerwise_mechanism_vs_auc.png",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
