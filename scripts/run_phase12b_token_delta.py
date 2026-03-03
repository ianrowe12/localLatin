"""Phase 12b: Token-level attribution delta analysis.

Reuses existing Phase 12 attribution .npz files (no new Captum runs).
Computes:
  4) Attribution Delta = (|IG_pc1| - |IG_abtt|) / (|IG_pc1| + |IG_abtt| + eps)
     per token, then aggregated by token category and layer.

The key insight: unlike the flawed Spearman(pc1_dot, abtt_norm), the attribution
delta captures *relative* importance differences. In dip layers, non-content
tokens should have positive delta (they contribute more to noise than signal).

Usage:
  python scripts/run_phase12b_token_delta.py \
    --attr_dir runs/phase12/attributions \
    --phase11_csv runs/phase11/phase11_results.csv \
    --out_dir runs/phase12b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Model configs — must match run_phase12_aggregate.py
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "LaTa": {
        "slug": "bowphs_LaTa",
        "hf": "bowphs/LaTa",
        "tokenizer": "bowphs/LaTa",
        "trust": False,
        "layers": list(range(1, 13)),
    },
    "PhilTa": {
        "slug": "bowphs_PhilTa",
        "hf": "bowphs/PhilTa",
        "tokenizer": "bowphs/PhilTa",
        "trust": False,
        "layers": list(range(1, 13)),
    },
    "LaBSE": {
        "slug": "sentence-transformers_LaBSE",
        "hf": "sentence-transformers/LaBSE",
        "tokenizer": "sentence-transformers/LaBSE",
        "trust": False,
        "layers": list(range(1, 13)),
    },
    "Qwen3-0.6B": {
        "slug": "Qwen_Qwen3-Embedding-0.6B",
        "hf": "Qwen/Qwen3-Embedding-0.6B",
        "tokenizer": "Qwen/Qwen3-Embedding-0.6B",
        "trust": True,
        "layers": list(range(1, 29)),
    },
    "KaLM-mini": {
        "slug": "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5",
        "hf": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "tokenizer": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "trust": True,
        "layers": list(range(1, 25)),
    },
}

COLORS = {"LaTa": "#1f77b4", "PhilTa": "#ff7f0e", "LaBSE": "#2ca02c",
           "Qwen3-0.6B": "#d62728", "KaLM-mini": "#9467bd"}

# Token classification — identical to run_phase12_aggregate.py
LATIN_PUNCTUATION = set(".,;:!?()[]{}\"'-/\\@#$%^&*+=<>~`")
CATEGORIES = ["content", "short_subword", "punctuation", "number", "empty"]


def classify_token(token_str: str) -> str:
    s = token_str.strip()
    if not s:
        return "empty"
    clean = s.lstrip("▁##Ġ")
    if not clean:
        return "empty"  # subword marker only → treat as empty
    if all(c in LATIN_PUNCTUATION for c in clean):
        return "punctuation"
    if clean.isdigit():
        return "number"
    if len(clean) <= 2:
        return "short_subword"
    return "content"


def load_attributions(attr_dir: Path, slug: str, layer: int):
    p = attr_dir / slug / f"layer{layer}_attributions.npz"
    if not p.exists():
        return None
    return dict(np.load(p))


def parse_args():
    p = argparse.ArgumentParser(description="Phase 12b token-level attribution delta.")
    p.add_argument("--attr_dir", required=True)
    p.add_argument("--phase11_csv", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    attr_dir = Path(args.attr_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 11 AUCROC
    p11 = pd.read_csv(args.phase11_csv)
    p11_base = p11[(p11["method"] == "baseline") & (p11["repr"] == "hidden")]
    p11_abtt = p11[(p11["method"] == "abtt_optimal") & (p11["repr"] == "hidden")]

    def _get_dauc(hf, layer):
        b = p11_base[(p11_base["model"] == hf) & (p11_base["layer"] == layer)]
        a = p11_abtt[(p11_abtt["model"] == hf) & (p11_abtt["layer"] == layer)]
        if b.empty or a.empty:
            return None
        return float(a.iloc[0]["aucroc"] - b.iloc[0]["aucroc"])

    all_rows = []

    for mname, cfg in MODEL_CONFIGS.items():
        slug = cfg["slug"]
        print(f"\n{'='*50}\n  {mname}\n{'='*50}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], trust_remote_code=cfg["trust"])
        except Exception as e:
            print(f"  WARNING: cannot load tokenizer: {e}")
            continue

        for layer in cfg["layers"]:
            data = load_attributions(attr_dir, slug, layer)
            if data is None:
                continue

            n_files = data["ig_pc1_dot"].shape[0]
            # Per-category accumulators
            cat_deltas = {c: [] for c in CATEGORIES}
            cat_abs_pc1 = {c: [] for c in CATEGORIES}
            cat_abs_abtt = {c: [] for c in CATEGORIES}

            for fi in range(n_files):
                sl = int(data["seq_lengths"][fi])
                if sl < 1:
                    continue
                ids = data["input_ids"][fi, :sl]
                ig_pc1 = np.abs(data["ig_pc1_dot"][fi, :sl])
                ig_abtt = np.abs(data["ig_abtt_norm"][fi, :sl])
                tokens = tokenizer.convert_ids_to_tokens(ids.tolist())

                # Normalised attribution delta per token
                denom = ig_pc1 + ig_abtt + 1e-12
                delta = (ig_pc1 - ig_abtt) / denom  # range [-1, +1]

                for tok_str, d, p1, ab in zip(tokens, delta, ig_pc1, ig_abtt):
                    cat = classify_token(tok_str)
                    if cat in cat_deltas:
                        cat_deltas[cat].append(float(d))
                        cat_abs_pc1[cat].append(float(p1))
                        cat_abs_abtt[cat].append(float(ab))

            for cat in CATEGORIES:
                if cat_deltas[cat]:
                    all_rows.append({
                        "model": mname,
                        "layer": layer,
                        "category": cat,
                        "mean_delta": np.mean(cat_deltas[cat]),
                        "median_delta": np.median(cat_deltas[cat]),
                        "std_delta": np.std(cat_deltas[cat]),
                        "mean_abs_pc1": np.mean(cat_abs_pc1[cat]),
                        "mean_abs_abtt": np.mean(cat_abs_abtt[cat]),
                        "n_tokens": len(cat_deltas[cat]),
                    })

            print(f"  L{layer}: processed {n_files} files")

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "token_delta_by_type.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved token delta data to {csv_path}")

    # ── Figure A: Heatmap per model (categories × layers) ─────────────
    for mname, cfg in MODEL_CONFIGS.items():
        sub = df[df["model"] == mname]
        if sub.empty:
            continue
        layers_present = sorted(sub["layer"].unique())
        cats_present = [c for c in CATEGORIES if c in sub["category"].values]
        heatmap = np.full((len(layers_present), len(cats_present)), np.nan)
        for li, layer in enumerate(layers_present):
            for ci, cat in enumerate(cats_present):
                row = sub[(sub["layer"] == layer) & (sub["category"] == cat)]
                if not row.empty:
                    heatmap[li, ci] = row.iloc[0]["mean_delta"]

        fig, ax = plt.subplots(figsize=(8, max(4, len(layers_present) * 0.4)))
        vmax = max(abs(np.nanmin(heatmap)), abs(np.nanmax(heatmap)))
        im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(cats_present)))
        ax.set_xticklabels(cats_present, rotation=45, ha="right")
        ax.set_yticks(range(len(layers_present)))
        ax.set_yticklabels(layers_present)
        ax.set_ylabel("Layer")
        ax.set_title(f"Attribution Delta by Token Type: {mname}\n"
                     f"(+) = more PC1 noise, (−) = more ABTT signal", fontsize=11)
        fig.colorbar(im, ax=ax, label="Mean Δ = (|IG_pc1|−|IG_abtt|)/(|IG_pc1|+|IG_abtt|)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig_attr_delta_heatmap_{cfg['slug']}.pdf", dpi=150)
        fig.savefig(fig_dir / f"fig_attr_delta_heatmap_{cfg['slug']}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved heatmap for {mname}")

    # ── Figure B: Attribution delta (short_subword) vs ΔAUCROC ────────
    fig, axes = plt.subplots(len(MODEL_CONFIGS), 1,
                             figsize=(10, 3.5 * len(MODEL_CONFIGS)), sharex=False)
    for ax, (mname, cfg) in zip(axes, MODEL_CONFIGS.items()):
        sub = df[(df["model"] == mname) & (df["category"] == "short_subword")].sort_values("layer")
        if sub.empty:
            ax.set_title(f"{mname}: no short_subword data")
            continue

        ax.plot(sub["layer"], sub["mean_delta"], "b-o", markersize=4,
                label="short_subword Δ")
        ax.set_ylabel("Mean Attr. Delta", color="b", fontsize=10)
        ax.tick_params(axis="y", labelcolor="b")
        ax.axhline(0, color="gray", ls="--", alpha=0.4)

        # Overlay ΔAUCROC
        dauc_vals = []
        for _, r in sub.iterrows():
            d = _get_dauc(cfg["hf"], int(r["layer"]))
            dauc_vals.append(d if d is not None else np.nan)

        ax2 = ax.twinx()
        ax2.plot(sub["layer"].values, dauc_vals, "r--s", markersize=4,
                 label="ΔAUCROC", alpha=0.7)
        ax2.set_ylabel("ΔAUCROC", color="r", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="r")
        ax.set_xlabel("Layer")
        ax.set_title(mname, fontsize=11)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_attr_delta_vs_aucroc.pdf", dpi=150)
    fig.savefig(fig_dir / "fig_attr_delta_vs_aucroc.png", dpi=150)
    plt.close(fig)
    print("  Saved attribution delta vs ΔAUCROC panel")

    # ── Figure C: Content vs short_subword delta comparison ───────────
    fig, axes = plt.subplots(len(MODEL_CONFIGS), 1,
                             figsize=(10, 3.5 * len(MODEL_CONFIGS)), sharex=False)
    for ax, (mname, cfg) in zip(axes, MODEL_CONFIGS.items()):
        content_sub = df[(df["model"] == mname) & (df["category"] == "content")].sort_values("layer")
        short_sub = df[(df["model"] == mname) & (df["category"] == "short_subword")].sort_values("layer")
        empty_sub = df[(df["model"] == mname) & (df["category"] == "empty")].sort_values("layer")

        if not content_sub.empty:
            ax.plot(content_sub["layer"], content_sub["mean_delta"], "g-o",
                    markersize=4, label="content Δ")
        if not short_sub.empty:
            ax.plot(short_sub["layer"], short_sub["mean_delta"], "r-s",
                    markersize=4, label="short_subword Δ")
        if not empty_sub.empty:
            ax.plot(empty_sub["layer"], empty_sub["mean_delta"], "k-^",
                    markersize=4, label="empty Δ", alpha=0.7)

        ax.axhline(0, color="gray", ls="--", alpha=0.4)
        ax.set_ylabel("Mean Attribution Delta")
        ax.set_xlabel("Layer")
        ax.set_title(mname, fontsize=11)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_attr_delta_content_vs_short.pdf", dpi=150)
    fig.savefig(fig_dir / "fig_attr_delta_content_vs_short.png", dpi=150)
    plt.close(fig)
    print("  Saved content vs short_subword delta panel")

    print("\nPhase 12b token delta analysis complete.")


if __name__ == "__main__":
    main()
