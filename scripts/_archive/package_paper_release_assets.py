"""Package the revised paper-release assets into one manifest-backed bundle."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package revised paper-release assets.")
    parser.add_argument("--phase11_csv", required=True, help="Phase 11 results CSV.")
    parser.add_argument("--phase11_fig_dir", required=True, help="Phase 11 release figures.")
    parser.add_argument("--dist_dir", required=True, help="Phase 11 distribution directory.")
    parser.add_argument("--taskb_dir", required=True, help="Task B output directory.")
    parser.add_argument("--phase12e_fig_dir", required=True, help="Phase 12e figures directory.")
    parser.add_argument("--out_dir", required=True, help="Final packaged asset directory.")
    parser.add_argument(
        "--feature_model",
        default="bowphs/PhilTa",
        help="Featured model for the 2x2 density figure.",
    )
    parser.add_argument(
        "--overleaf_figures_dir",
        default="",
        help="Optional figures directory to sync paper-facing figures into.",
    )
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_density_figure(feature_model: str, dist_dir: Path, out_dir: Path) -> Path:
    slug = model_slug(feature_model)
    conds = [
        ("baseline_last", "Baseline - Last Layer"),
        ("baseline_middle", "Baseline - Dip Layer"),
        ("abtt_last", "ABTT - Last Layer"),
        ("abtt_middle", "ABTT - Dip Layer"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.ravel()
    bins = np.linspace(-0.2, 1.0, 80)
    row_ymax = [0.0, 0.0]

    for idx, (suffix, title) in enumerate(conds):
        ax = axes[idx]
        npz_path = dist_dir / f"{slug}_{suffix}.npz"
        if not npz_path.exists():
            ax.axis("off")
            ax.text(0.5, 0.5, f"Missing {npz_path.name}", ha="center", va="center")
            continue

        data = np.load(npz_path)
        same = data["same_sims"]
        diff = data["diff_sims"]
        ax.hist(
            diff,
            bins=bins,
            alpha=0.5,
            color="#c0392b",
            density=True,
            edgecolor="none",
            label="Different",
        )
        ax.hist(
            same,
            bins=bins,
            alpha=0.5,
            color="#2471a3",
            density=True,
            edgecolor="none",
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
        row = idx // 2
        row_ymax[row] = max(row_ymax[row], ax.get_ylim()[1])

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

    png_path = out_dir / "paper_fig_density_2x2.png"
    pdf_path = out_dir / "paper_fig_density_2x2.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def build_release_summary(phase11_df: pd.DataFrame, taskb_dir: Path, feature_model: str) -> dict:
    release = phase11_df[
        (phase11_df["repr"] == "hidden")
        & (
            ((phase11_df["method"].isin(["baseline", "abtt_optimal", "whitening"])) & (phase11_df["pooling"] == "mean"))
            | ((phase11_df["method"] == "sif_only") & (phase11_df["pooling"] == "sif"))
        )
    ].copy()
    summary = {
        "feature_model": feature_model,
        "feature_model_short": SHORT.get(feature_model, feature_model),
        "n_phase11_rows": int(len(phase11_df)),
        "n_release_rows": int(len(release)),
    }
    taskb_csv = taskb_dir / "taskb_rank_distribution.csv"
    if taskb_csv.exists():
        taskb_df = pd.read_csv(taskb_csv)
        summary["n_taskb_rows"] = int(len(taskb_df))
        summary["taskb_models"] = sorted(taskb_df["model_short"].unique().tolist())
    return summary


def sync_overleaf(figures_dir: Path, packaged_files: list[Path]) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    keep_suffixes = {".pdf", ".png"}
    for src in packaged_files:
        if src.suffix.lower() not in keep_suffixes:
            continue
        shutil.copy2(src, figures_dir / src.name)


def main() -> None:
    args = parse_args()
    phase11_csv = Path(args.phase11_csv)
    phase11_fig_dir = Path(args.phase11_fig_dir)
    dist_dir = Path(args.dist_dir)
    taskb_dir = Path(args.taskb_dir)
    phase12e_fig_dir = Path(args.phase12e_fig_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    packaged_files: list[Path] = []

    phase11_df = pd.read_csv(phase11_csv)
    summary = build_release_summary(phase11_df, taskb_dir, args.feature_model)
    summary_path = out_dir / "release_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    manifest_rows.append(
        {
            "asset_id": "release_summary",
            "path": str(summary_path),
            "kind": "json",
            "purpose": "High-level summary of the revised paper-release bundle.",
        }
    )
    packaged_files.append(summary_path)

    density_pdf = write_density_figure(args.feature_model, dist_dir, out_dir)
    packaged_files.extend([density_pdf, density_pdf.with_suffix(".png")])
    manifest_rows.append(
        {
            "asset_id": "paper_fig_density_2x2",
            "path": str(density_pdf),
            "kind": "figure",
            "purpose": f"Featured density comparison for {SHORT.get(args.feature_model, args.feature_model)}.",
        }
    )

    copy_specs = [
        ("fig_release_aucroc_per_model", phase11_fig_dir / "fig_release_aucroc_per_model.pdf", "figure", "Layerwise AUCROC figure."),
        ("fig_release_aucroc_per_model_png", phase11_fig_dir / "fig_release_aucroc_per_model.png", "figure", "Layerwise AUCROC figure PNG."),
        ("fig_release_gap_per_model", phase11_fig_dir / "fig_release_gap_per_model.pdf", "figure", "Layerwise cosine-gap figure."),
        ("fig_release_gap_per_model_png", phase11_fig_dir / "fig_release_gap_per_model.png", "figure", "Layerwise cosine-gap figure PNG."),
        ("taskb_selected_configs", taskb_dir / "taskb_selected_configs.csv", "table_data", "Selected per-model Task B configurations."),
        ("taskb_top5_predictions", taskb_dir / "taskb_top5_predictions.csv", "table_data", "Per-query Task B top-5 predictions."),
        ("taskb_rank_distribution", taskb_dir / "taskb_rank_distribution.csv", "table_data", "Exact-rank Task B summary."),
        ("taskb_rank_distribution_table", taskb_dir / "taskb_rank_distribution_table.csv", "table_data", "Paper-facing Task B table values."),
        ("taskb_rank_distribution_table_pdf", taskb_dir / "taskb_rank_distribution_table.pdf", "table", "Paper-facing Task B table PDF."),
        ("taskb_rank_distribution_table_png", taskb_dir / "taskb_rank_distribution_table.png", "table", "Paper-facing Task B table PNG."),
        ("fig_pair_matrix_philta", phase12e_fig_dir / "fig_pair_matrix_philta.pdf", "figure", "Final derived pair-matrix figure."),
        ("fig_pair_matrix_philta_png", phase12e_fig_dir / "fig_pair_matrix_philta.png", "figure", "Final derived pair-matrix figure PNG."),
        ("fig_attention_philta", phase12e_fig_dir / "fig_attention_philta.pdf", "figure", "Final companion attention figure."),
        ("fig_attention_philta_png", phase12e_fig_dir / "fig_attention_philta.png", "figure", "Final companion attention figure PNG."),
        ("phase12e_review_manifest", phase12e_fig_dir / "phase12e_review_manifest.csv", "metadata", "Review-manifest for candidate explanation figures."),
    ]
    for asset_id, src, kind, purpose in copy_specs:
        dst = out_dir / src.name
        if copy_if_exists(src, dst):
            packaged_files.append(dst)
            manifest_rows.append(
                {
                    "asset_id": asset_id,
                    "path": str(dst),
                    "kind": kind,
                    "purpose": purpose,
                }
            )

    manifest_path = out_dir / "asset_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    packaged_files.append(manifest_path)

    if args.overleaf_figures_dir:
        sync_overleaf(Path(args.overleaf_figures_dir), packaged_files)

    print(f"Packaged revised paper assets in {out_dir}")


if __name__ == "__main__":
    main()
