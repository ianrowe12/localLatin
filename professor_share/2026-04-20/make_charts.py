#!/usr/bin/env python
"""Generate two shareable bar charts from the attribution_metrics summary CSV:
  - rho_LOO (the robust signal that ABTT helps universally)
  - Suff@25% ratio (the mixed signal that undercuts the uniform-improvement story)

Reads: runs/active/ig_examples/attribution_metrics/summary.csv
Writes: professor_share/chart_rho_loo.pdf, chart_suff25.pdf, chart_rho_loo_all4.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/projects/beto/irowerojas/localLatin")
SUMMARY = REPO / "runs/active/ig_examples/attribution_metrics/summary.csv"
OUT = REPO / "professor_share"

MODEL_LABEL = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
}
METHOD_LABEL = {"ig": "IG", "retrieval_mark": "MaRC"}

COLOR_BASE = "#6c757d"
COLOR_ABTT = "#e63946"


def load() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY)
    return df[
        df["method"].isin(["ig", "retrieval_mark"])
        & df["variant"].isin(["baseline", "abtt"])
    ].copy()


def grouped_bar(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    direction: str,
    models: list[str],
    title: str,
    out_path: Path,
    y_lims: tuple[float, float] | None = None,
    se_col: str | None = None,
) -> None:
    methods = ["ig", "retrieval_mark"]
    groups = [(m, mt) for m in models for mt in methods]
    n = len(groups)
    x = np.arange(n)
    width = 0.38

    base_vals, abtt_vals, base_se, abtt_se = [], [], [], []
    for model, method in groups:
        row_b = df[
            (df["model"] == model) & (df["method"] == method) & (df["variant"] == "baseline")
        ]
        row_a = df[
            (df["model"] == model) & (df["method"] == method) & (df["variant"] == "abtt")
        ]
        base_vals.append(row_b[metric_col].iloc[0] if len(row_b) else np.nan)
        abtt_vals.append(row_a[metric_col].iloc[0] if len(row_a) else np.nan)
        if se_col:
            base_se.append(row_b[se_col].iloc[0] if len(row_b) else 0.0)
            abtt_se.append(row_a[se_col].iloc[0] if len(row_a) else 0.0)

    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * n), 4.0))
    b1 = ax.bar(
        x - width / 2,
        base_vals,
        width,
        yerr=base_se if se_col else None,
        label="baseline",
        color=COLOR_BASE,
        capsize=3,
    )
    b2 = ax.bar(
        x + width / 2,
        abtt_vals,
        width,
        yerr=abtt_se if se_col else None,
        label="ABTT",
        color=COLOR_ABTT,
        capsize=3,
    )

    labels = [f"{MODEL_LABEL[m]}\n{METHOD_LABEL[mt]}" for m, mt in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"{metric_label} ({direction})")
    ax.set_title(title, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(frameon=False, loc="best")
    if y_lims:
        ax.set_ylim(*y_lims)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load()

    t5 = ["bowphs/LaTa", "bowphs/PhilTa"]
    all4 = ["bowphs/LaTa", "bowphs/PhilTa", "sentence-transformers/LaBSE", "Qwen/Qwen3-Embedding-0.6B"]

    # Chart 1: rho_LOO, T5 only (matches main-text candidate scope)
    grouped_bar(
        df,
        metric_col="loo_rho_mean",
        metric_label=r"$\rho_{\mathrm{LOO}}$",
        direction="higher is better",
        models=t5,
        title=r"$\rho_{\mathrm{LOO}}$ improves under ABTT (T5 models; main-text scope)",
        out_path=OUT / "chart_rho_loo_t5.pdf",
        se_col="loo_rho_se",
    )

    # Chart 2: Suff@25% ratio, T5 only (the mixed / honest picture)
    grouped_bar(
        df,
        metric_col="suff@0.25_ratio_mean",
        metric_label="Suff@25% ratio",
        direction="higher is better",
        models=t5,
        title="Sufficiency@25%: mixed picture under ABTT (T5 models)",
        out_path=OUT / "chart_suff25_t5.pdf",
        se_col="suff@0.25_ratio_se",
    )

    # Chart 3: rho_LOO, 3 models where metric is defined across both methods
    # (Qwen3-0.6B x MaRC is excluded: decoder-only fallback to attention masking,
    # rho_LOO undefined for that cell.)
    valid3 = ["bowphs/LaTa", "bowphs/PhilTa", "sentence-transformers/LaBSE"]
    grouped_bar(
        df,
        metric_col="loo_rho_mean",
        metric_label=r"$\rho_{\mathrm{LOO}}$",
        direction="higher is better",
        models=valid3,
        title=r"$\rho_{\mathrm{LOO}}$: ABTT $>$ baseline on every cell where metric is defined (6/6)",
        out_path=OUT / "chart_rho_loo_all.pdf",
        se_col="loo_rho_se",
    )


if __name__ == "__main__":
    main()
