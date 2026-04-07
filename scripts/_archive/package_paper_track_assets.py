from __future__ import annotations

import argparse
import json
import shutil
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
    "google/mt5-base": "mt5-base",
}

DIP_LAYERS = {
    "bowphs/LaTa": 4,
    "bowphs/PhilTa": 6,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package filtered paper-track assets.")
    parser.add_argument("--phase11_csv", required=True)
    parser.add_argument("--dist_dir", required=True)
    parser.add_argument("--phase12c_figures_dir", required=True)
    parser.add_argument("--phase12d_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def choose_postprocess_method(results: pd.DataFrame) -> tuple[str, dict]:
    score_rows = []
    for model_name in SHORT:
        for method in ["abtt_optimal", "sif_abtt_optimal"]:
            sub = results[
                (results["model"] == model_name)
                & (results["method"] == method)
                & (results["repr"] == "hidden")
            ]
            if sub.empty:
                continue
            best = sub.loc[sub["train_aucroc"].idxmax()]
            score_rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "train_aucroc": float(best["train_aucroc"]),
                    "test_aucroc": float(best["aucroc"]),
                    "layer": int(best["layer"]),
                }
            )

    score_df = pd.DataFrame(score_rows)
    sif_wins = 0
    for model_name in SHORT:
        sub = score_df[score_df["model"] == model_name]
        if {"abtt_optimal", "sif_abtt_optimal"} - set(sub["method"]):
            continue
        abtt = float(sub[sub["method"] == "abtt_optimal"]["test_aucroc"].iloc[0])
        sif = float(sub[sub["method"] == "sif_abtt_optimal"]["test_aucroc"].iloc[0])
        if sif > abtt:
            sif_wins += 1

    feature_model = choose_feature_model(results)
    feature_sub = score_df[score_df["model"] == feature_model]
    sif_loss = 0.0
    if {"abtt_optimal", "sif_abtt_optimal"} <= set(feature_sub["method"]):
        abtt = float(feature_sub[feature_sub["method"] == "abtt_optimal"]["test_aucroc"].iloc[0])
        sif = float(feature_sub[feature_sub["method"] == "sif_abtt_optimal"]["test_aucroc"].iloc[0])
        sif_loss = abtt - sif

    preferred = "sif_abtt_optimal" if sif_wins >= 4 and sif_loss <= 0.01 else "abtt_optimal"
    return preferred, {
        "preferred_method": preferred,
        "sif_wins": int(sif_wins),
        "feature_model": feature_model,
        "feature_model_short": SHORT[feature_model],
        "feature_model_sif_loss_vs_abtt": float(sif_loss),
        "decision_rule": "Prefer sif_abtt_optimal only if it wins on >=4 models and does not lose >0.01 on the featured T5 model.",
    }


def choose_feature_model(results: pd.DataFrame) -> str:
    lifts = []
    for model_name, dip_layer in DIP_LAYERS.items():
        baseline = results[
            (results["model"] == model_name)
            & (results["method"] == "baseline")
            & (results["repr"] == "hidden")
            & (results["layer"] == dip_layer)
        ]
        abtt = results[
            (results["model"] == model_name)
            & (results["method"] == "abtt_optimal")
            & (results["repr"] == "hidden")
            & (results["layer"] == dip_layer)
        ]
        if baseline.empty or abtt.empty:
            continue
        lifts.append(
            (
                model_name,
                float(abtt["aucroc"].iloc[0] - baseline["aucroc"].iloc[0]),
            )
        )
    if not lifts:
        return "bowphs/LaTa"
    lifts.sort(key=lambda item: item[1], reverse=True)
    if len(lifts) > 1 and abs(lifts[0][1] - lifts[1][1]) <= 0.02:
        return "bowphs/LaTa"
    return lifts[0][0]


def write_density_figure(
    feature_model: str,
    dist_dir: Path,
    out_dir: Path,
    preferred_method: str,
) -> Path:
    slug = model_slug(feature_model)
    post_prefix = "sif_abtt" if preferred_method == "sif_abtt_optimal" else "abtt"
    cond_suffixes = [
        ("baseline_last", "Baseline - Last Layer"),
        ("baseline_middle", "Baseline - Dip Layer"),
        (f"{post_prefix}_last", "ABTT - Last Layer"),
        (f"{post_prefix}_middle", "ABTT - Dip Layer"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    bins = np.linspace(-0.2, 1.0, 80)

    # --- First pass: draw histograms and record per-row Y maxima ---
    row_ymax = [0.0, 0.0]  # one per row
    for idx, (ax, (suffix, title)) in enumerate(zip(axes.ravel(), cond_suffixes)):
        npz_path = dist_dir / f"{slug}_{suffix}.npz"
        data = np.load(npz_path)
        same = data["same_sims"]
        diff = data["diff_sims"]
        ax.hist(
            diff, bins=bins, alpha=0.5, color="#d62728",
            density=True, edgecolor="none", label="Different",
        )
        ax.hist(
            same, bins=bins, alpha=0.5, color="#1f77b4",
            density=True, edgecolor="none", label="Same",
        )
        same_mean = float(np.mean(same))
        diff_mean = float(np.mean(diff))
        ax.axvline(same_mean, color="#1f77b4", ls="--", lw=1.2)
        ax.axvline(diff_mean, color="#d62728", ls="--", lw=1.2)
        ax.set_title(title)
        ax.text(
            0.98,
            0.96,
            f"gap={same_mean - diff_mean:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        row = idx // 2
        cur_ymax = ax.get_ylim()[1]
        if cur_ymax > row_ymax[row]:
            row_ymax[row] = cur_ymax

    # --- Second pass: enforce row-wise Y-axis sharing ---
    for row in range(2):
        ymax = row_ymax[row] * 1.05  # 5 % headroom
        for col in range(2):
            axes[row, col].set_ylim(0, ymax)

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_xlabel("Cosine Similarity")
    axes[1, 1].set_xlabel("Cosine Similarity")
    # Add a shared legend in the top-right panel
    axes[0, 1].legend(loc="upper left", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    png_path = out_dir / "paper_fig_density_2x2.png"
    pdf_path = out_dir / "paper_fig_density_2x2.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def build_main_results_table(results: pd.DataFrame, preferred_method: str, out_dir: Path) -> Path:
    rows = []
    for model_name in SHORT:
        sub = results[
            (results["model"] == model_name)
            & (results["method"] == preferred_method)
            & (results["repr"] == "hidden")
        ]
        if sub.empty:
            continue
        best = sub.loc[sub["train_aucroc"].idxmax()]
        rows.append(
            {
                "model": SHORT[model_name],
                "layer": int(best["layer"]),
                "test_aucroc": float(best["aucroc"]),
                "test_gap": float(best["gap"]),
                "dir_acc_at_1": float(best["dir_acc_at_1"]),
                "pooling": str(best["pooling"]),
                "method": preferred_method,
            }
        )
    table = pd.DataFrame(rows).sort_values("test_aucroc", ascending=False)
    out_path = out_dir / "paper_table_main_results.csv"
    table.to_csv(out_path, index=False)
    return out_path


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    args = parse_args()
    phase11_csv = Path(args.phase11_csv)
    dist_dir = Path(args.dist_dir)
    phase12c_figures_dir = Path(args.phase12c_figures_dir)
    phase12d_dir = Path(args.phase12d_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(phase11_csv)
    feature_model = choose_feature_model(results)
    preferred_method, decision = choose_postprocess_method(results)

    manifest_rows = []

    decision_path = out_dir / "method_decision.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    manifest_rows.append(
        {
            "asset_id": "method_decision",
            "path": str(decision_path),
            "kind": "json",
            "purpose": "Preferred paper post-processing method and decision evidence.",
        }
    )

    density_path = write_density_figure(feature_model, dist_dir, out_dir, preferred_method)
    manifest_rows.append(
        {
            "asset_id": "paper_fig_density_2x2",
            "path": str(density_path),
            "kind": "figure",
            "purpose": f"Featured 2x2 density figure for {SHORT[feature_model]}.",
        }
    )

    table_path = build_main_results_table(results, preferred_method, out_dir)
    manifest_rows.append(
        {
            "asset_id": "paper_table_main_results",
            "path": str(table_path),
            "kind": "table",
            "purpose": "Train-selected best-layer results for the preferred hidden-state method.",
        }
    )

    copy_specs = [
        ("phase11_layerwise_auc", phase11_csv.parent / "figures" / "fig7_aucroc_per_model.pdf"),
        ("phase11_layerwise_gap", phase11_csv.parent / "figures" / "fig6_gap_per_model.pdf"),
        ("phase12c_delta", phase12c_figures_dir / "fig_analysis1_delta_retrieval.png"),
        ("phase12c_selectivity", phase12c_figures_dir / "fig_analysis2_selectivity.png"),
        ("phase12c_importance_share", phase12c_figures_dir / "fig_analysis4_importance_share.png"),
        ("phase12c_t5_content_share", phase12c_figures_dir / "fig_t5_content_share.pdf"),
        ("phase12d_aucroc", phase12d_dir / "fig_filtering_vs_abtt.png"),
        ("phase12d_acc1", phase12d_dir / "fig_filtering_vs_abtt_acc1.png"),
    ]
    for asset_id, src in copy_specs:
        dst = out_dir / src.name
        if copy_if_exists(src, dst):
            manifest_rows.append(
                {
                    "asset_id": asset_id,
                    "path": str(dst),
                    "kind": "copied_asset",
                    "purpose": f"Paper-track supporting asset copied from {src.parent.name}.",
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "asset_manifest.csv", index=False)


if __name__ == "__main__":
    main()
