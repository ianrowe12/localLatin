#!/usr/bin/env python3
"""Build ``tables/layer_diagnostics_main.tex`` (``tab:layer_diagnostics_main``).

The table in Appendix ``app:layer_diagnostics`` was hand-written (issue #235
item 17). This regenerates it from the committed layer-diagnostics CSV, so a
re-run of the diagnostics cannot leave the appendix quoting old values:

  runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv
      per model/layer/split/view geometry; this table reads the test split,
      views ``raw`` (baseline) and ``abtt_d10`` (ABTT with D=10)

The table used to carry a second column, the operational token-attribution
layer read from ``layer_rule_candidates.csv``. Token attribution left the paper
(paper spine, 2 October 2026), so that column and its caption sentence are gone.

Per T5 encoder: the most anisotropic layer, the argmax of top-PC share (``pc1_variance_ratio``) over layers on the test
split, baseline view (the first layer on a tie); and, at that layer, top-PC
share and entropy effective rank before (baseline) and after ABTT-D10.

    python scripts/paper/reframe/layer_diagnostics_table.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DIAG = Path("runs/active/resubmit/layer_diagnostics")
MODELS = [("bowphs/LaTa", "LaTa"), ("bowphs/PhilTa", "PhilTa"), ("google/mt5-base", "mT5-base")]

CAPTION = (
    r"Most anisotropic layer of each raw T5-family panel encoder, by top-PC share (PC1) "
    r"on test passages; Table~\ref{tab:geometry_regimes} reads top-PC share on training "
    r"passages. PC1 and effective rank before and after ABTT with $D=10$, fit on training "
    r"passages."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--geometry_csv", default=str(DIAG / "geometry_per_layer.csv"))
    p.add_argument("--out", default="overleaf_drafts/tables/layer_diagnostics_main.tex")
    return p.parse_args()


def _view(geom: pd.DataFrame, model: str, split: str, view: str) -> pd.DataFrame:
    sub = geom[
        (geom["model"] == model) & (geom["split"] == split) & (geom["view"] == view)
        & (geom["repr"] == "hidden") & (geom["pooling"] == "mean")
    ].sort_values("layer")
    if sub.empty:
        raise SystemExit(f"no {split}/{view} geometry rows for {model}")
    return sub


def peak_layer(rows: pd.DataFrame) -> int:
    """Layer of the largest top-PC share; the first such layer on a tie."""
    return int(rows.loc[rows["pc1_variance_ratio"].idxmax(), "layer"])


def render(geom: pd.DataFrame) -> str:
    rows = []
    for model, display in MODELS:
        raw = _view(geom, model, "test", "raw")
        abtt = _view(geom, model, "test", "abtt_d10")
        peak = peak_layer(raw)
        before = raw[raw["layer"] == peak].iloc[0]
        after = abtt[abtt["layer"] == peak].iloc[0]
        rows.append(
            f"{display} & {peak} & {before['pc1_variance_ratio']:.3f} & "
            f"{after['pc1_variance_ratio']:.3f} & "
            f"{before['effective_rank_entropy']:.2f}$\\rightarrow$"
            f"{after['effective_rank_entropy']:.2f} \\\\"
        )
    caption = CAPTION
    lines = [
        "% generated table",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Most anisotropic layer} & "
        r"\textbf{PC1 before} & \textbf{PC1 after} & "
        r"\textbf{Eff. rank before$\rightarrow$after} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        r"\label{tab:layer_diagnostics_main}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tex = render(pd.read_csv(args.geometry_csv))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
