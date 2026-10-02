#!/usr/bin/env python3
"""The three main-text figures of the paper-spine rewrite (October 2026).

Reads only committed result CSVs (no embeddings, CPU only, a few seconds):
  runs/active/resubmit/results/phase_resubmit_results.csv   baseline and ABTT per layer
  runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv  top-PC share, mean cosine
  runs/active/reframe/p2x2/p2x2_layers.csv                   T5-base and T5-v1.1-base layers
  runs/active/reframe/e1/e1_k_sweep.csv                      coordinate zeroing sweep
  runs/active/reframe/h1/h1_d_ablation.csv                   components-removed sweep

Writes, in overleaf_drafts/figures/:
  fig_depth.pdf        (fig:depth, figure*)   baseline and ABTT test AUROC by layer, six panel
                                              models, one small panel each
  fig_diagnostics.pdf  (fig:diagnostics, figure, one column) baseline test AUROC against
                                              top-PC share and mean pairwise cosine (both on
                                              the training passages), one point per layer;
                                              the panel plus T5-base and T5-v1.1-base
  fig_localize.pdf     (fig:localize, figure*) AUROC at the 26 collapsed layers (a) as the k
                                              largest coordinates (mean |x|) are zeroed, with
                                              the random-coordinate control, and (b) as the
                                              top D principal components are removed
  fig_localize_col.pdf (fig:localize, figure, one column) the same two panels stacked

Collapsed layer = baseline test AUROC below 0.70 (paper Sec. 3). Each line in fig_localize
is the median over that model's collapsed layers; the band spans their minimum to maximum.

  python scripts/paper/reframe/paper_figures.py            # all three
  python scripts/paper/reframe/paper_figures.py --only depth,localize
Python 3.10, pandas / numpy / matplotlib.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

RES_CSV = Path("runs/active/resubmit/results/phase_resubmit_results.csv")
GEOM_CSV = Path("runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv")
P2X2_CSV = Path("runs/active/reframe/p2x2/p2x2_layers.csv")
KSWEEP_CSV = Path("runs/active/reframe/e1/e1_k_sweep.csv")
H1_CSV = Path("runs/active/reframe/h1/h1_d_ablation.csv")
FIG_DIR = Path("overleaf_drafts/figures")

COLLAPSE_AUROC = 0.70
REPAIR_AUROC = 0.90
TEXT_WIDTH_IN = 6.3  # ACL two-column \textwidth (455 pt)
COLUMN_WIDTH_IN = 3.03  # ACL \columnwidth (219 pt)
DEPTH_HEIGHT_IN = 1.55  # fig_depth, about 112 pt
DIAG_HEIGHT_IN = 2.15  # fig_diagnostics
LOCALIZE_HEIGHT_IN = 1.6  # fig_localize (full width), about 115 pt
LOCALIZE_COL_HEIGHT_IN = 3.2  # fig_localize_col (one column, panels stacked)

# id, display, is_t5, colour (Okabe-Ito), marker, line style. Same colours and markers as
# geometry_vs_retrieval.py, abtt_subspace_whiten.py and e1_k_sweep.py.
PANEL = [
    ("bowphs/LaTa", "LaTa", True, "#0072B2", "o", "-"),
    ("bowphs/PhilTa", "PhilTa", True, "#E69F00", "s", "--"),
    ("google/mt5-base", "mT5-base", True, "#009E73", "^", "-."),
    ("sentence-transformers/LaBSE", "LaBSE", False, "#CC79A7", "D", "-"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B", False, "#D55E00", "v", "-"),
    ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", "KaLM-mini", False,
     "#56B4E9", "P", "-"),
]
# The two within-family controls of the diagnostics figure (p2x2_layers.csv display names).
CONTROLS = [
    ("T5-v1.1-base", "#000000", "X"),
    ("T5-base", "#888888", "*"),
]
DISP = {m[0]: m[1] for m in PANEL}
T5_COLLAPSING = [m[1] for m in PANEL if m[2]]
H1_DS = (0, 1, 2, 3, 5, 7, 10)
K_MAX = 400
ZERO_RANKING = "mean_abs"  # "largest coordinates" = largest mean |x| on training passages


def style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "grid.linewidth": 0.4, "grid.color": "#dddddd", "font.family": "DejaVu Sans",
        "lines.linewidth": 1.1, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


# --------------------------------------------------------------------------- #
# Data (pure functions; tested in tests/test_paper_figures.py)
# --------------------------------------------------------------------------- #

def depth_data(res: pd.DataFrame) -> pd.DataFrame:
    """One row per panel model-layer: baseline and ABTT (train-selected D) test AUROC."""
    b = res[res.method == "baseline"][["model", "layer", "aucroc"]]
    a = res[res.method == "abtt_optimal"][["model", "layer", "aucroc"]]
    d = b.merge(a, on=["model", "layer"], suffixes=("_base", "_abtt"), validate="1:1")
    d = d[d.model.isin(DISP)].copy()
    d["m"] = d.model.map(DISP)
    return d.sort_values(["model", "layer"]).reset_index(drop=True)


def diagnostics_data(res: pd.DataFrame, geom: pd.DataFrame,
                     p2x2: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Baseline test AUROC, top-PC share and mean pairwise cosine (both on the training
    passages, raw mean-pooled vectors), one row per layer. Panel from geometry_per_layer;
    T5-base and T5-v1.1-base from p2x2_layers (same definitions, train split)."""
    b = res[res.method == "baseline"][["model", "layer", "aucroc"]]
    g = geom[(geom.split == "train") & (geom.view == "raw")][
        ["model", "layer", "pc1_variance_ratio", "anisotropy_mean_cosine"]]
    d = b.merge(g, on=["model", "layer"], validate="1:1").rename(columns={
        "pc1_variance_ratio": "pc1", "anisotropy_mean_cosine": "mean_cos"})
    d = d[d.model.isin(DISP)].copy()
    d["m"] = d.model.map(DISP)
    parts = [d[["m", "layer", "aucroc", "pc1", "mean_cos"]]]
    if p2x2 is not None:
        names = [c[0] for c in CONTROLS]
        parts.append(p2x2[p2x2.model.isin(names)].rename(columns={"model": "m"})[
            ["m", "layer", "aucroc", "pc1", "mean_cos"]])
    return pd.concat(parts, ignore_index=True)


def collapsed_layers(res: pd.DataFrame) -> pd.DataFrame:
    """Panel model-layers with baseline test AUROC below the collapse cutoff."""
    b = res[(res.method == "baseline") & (res.aucroc < COLLAPSE_AUROC)]
    return b[["model", "layer"]].reset_index(drop=True)


def _band(df: pd.DataFrame, x: str) -> pd.DataFrame:
    return (df.groupby(["model", x])["aucroc"].agg(["median", "min", "max"])
            .reset_index().sort_values(["model", x]))


def zeroing_bands(sweep: pd.DataFrame, coll: pd.DataFrame,
                  ranking: str = ZERO_RANKING) -> Dict[str, pd.DataFrame]:
    """Per model, median/min/max over its collapsed layers of test AUROC against k, for the
    ranked zeroing and for the random control (seed mean first, then over layers)."""
    s = sweep.merge(coll, on=["model", "layer"])
    s = s[s.k <= K_MAX]
    ranked = s[s.ranking == ranking]
    rnd = (s[s.ranking == "random"].groupby(["model", "layer", "k"])["aucroc"].mean()
           .reset_index())
    return {"ranked": _band(ranked, "k"), "random": _band(rnd, "k")}


def components_band(h1: pd.DataFrame, coll: pd.DataFrame) -> pd.DataFrame:
    """Per model, median/min/max over its collapsed layers of test AUROC after removing the
    top D components (D = 0 is centering only)."""
    h = h1.merge(coll, on=["model", "layer"])
    h = h[h.variant.isin(["center", "abtt"]) & h.D.isin(H1_DS)]
    return _band(h, "D")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})


def fig_depth(d: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style()
    fig, axes = plt.subplots(1, 6, figsize=(TEXT_WIDTH_IN, DEPTH_HEIGHT_IN), sharey=True)
    for ax, (mid, name, t5, c, mk, _ls) in zip(axes, PANEL):
        s = d[d.model == mid]
        ms = 2.6 if s.layer.max() <= 12 else 1.8
        ax.axhspan(0.45, COLLAPSE_AUROC, color="#f2f2f2", lw=0, zorder=0)
        ax.axhline(0.5, color="#888888", lw=0.6, zorder=1)
        ax.plot(s.layer, s.aucroc_abtt, color=c, lw=1.1, ls="-", marker=mk, markersize=ms,
                markerfacecolor=c, markeredgecolor=c, zorder=3)
        ax.plot(s.layer, s.aucroc_base, color=c, lw=1.1, ls=":", marker=mk, markersize=ms,
                markerfacecolor="white", markeredgecolor=c, markeredgewidth=0.7, zorder=4)
        n = int(s.layer.max())
        ax.set_xlim(0.3, n + 0.7)
        step = 4 if n <= 12 else 8
        ax.set_xticks([1] + list(range(step, n + 1, step)))
        ax.set_title(name, pad=2)
        ax.grid(True, axis="y", zorder=0)
        ax.set_xlabel("Layer")
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_ylabel("Test AUROC")
    key = [Line2D([0], [0], color="#444444", ls=":", marker="o", markersize=3,
                  markerfacecolor="white", markeredgecolor="#444444", label="Baseline"),
           Line2D([0], [0], color="#444444", ls="-", marker="o", markersize=3,
                  markerfacecolor="#444444", label="ABTT (fit on training embeddings)")]
    fig.legend(handles=key, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.10), handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(w_pad=0.5)
    _save(fig, path)
    plt.close(fig)


def fig_diagnostics(d: pd.DataFrame, path: Path) -> None:
    """(a) mean pairwise cosine, (b) top-PC share, against baseline test AUROC. The six
    panel models are in the legend above; the two T5 controls in a legend inside the empty
    lower left of panel (b)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style()
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, DIAG_HEIGHT_IN), sharey=True)
    marks = [(name, c, mk, t5) for _mid, name, t5, c, mk, _ls in PANEL]
    marks += [(name, c, mk, True) for name, c, mk in CONTROLS]
    ctrl_names = {x[0] for x in CONTROLS}
    for ax, col, xl in ((axes[0], "mean_cos", "(a) Mean pairwise cosine"),
                        (axes[1], "pc1", "(b) Top-PC share")):
        ax.axhspan(0.45, COLLAPSE_AUROC, color="#f2f2f2", lw=0, zorder=0)
        for name, c, mk, filled in marks:
            s = d[d.m == name]
            if s.empty:
                continue
            ctrl = name in ctrl_names
            ax.scatter(s[col], s.aucroc, marker=mk, s=(34 if mk == "*" else 22) if ctrl else
                       (13 if filled else 11), linewidths=0 if ctrl else 0.8,
                       facecolors=c if filled else "none", edgecolors="none" if ctrl else c,
                       zorder=4 if ctrl else 3, alpha=0.9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel(xl)
        ax.grid(True, zorder=0)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_ylabel("Baseline test AUROC")

    def handle(name: str, c: str, mk: str, filled: bool) -> Line2D:
        ctrl = name in ctrl_names
        return Line2D([0], [0], ls="none", marker=mk, markersize=6 if mk == "*" else 4.5,
                      markerfacecolor=c if filled else "white", markeredgecolor=c,
                      markeredgewidth=0 if ctrl else 0.8, label=name)

    present = [m for m in marks if not d[d.m == m[0]].empty]
    top = [handle(*m) for m in present if m[0] not in ctrl_names]
    inner = [handle(*m) for m in present if m[0] in ctrl_names]
    fig.legend(handles=top, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.53, 1.0), handletextpad=0.1, columnspacing=0.6)
    if inner:
        # the empty lower left of panel (b) (no point below AUROC 0.70 left of share 0.76);
        # T5-base on top so the longer label sits on the bottom row, below every point
        axes[1].legend(handles=inner[::-1], loc="lower left", bbox_to_anchor=(-0.02, 0.0),
                       frameon=False, handletextpad=0.0, borderaxespad=0.0, labelspacing=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.83), w_pad=0.6)
    _save(fig, path)
    plt.close(fig)


def fig_localize(zb: Dict[str, pd.DataFrame], cb: pd.DataFrame, path: Path,
                 column: bool = False) -> None:
    """Two panels side by side at text width, or (``column``) stacked at column width."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter

    style()
    if column:
        fig, (axk, axd) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, LOCALIZE_COL_HEIGHT_IN))
    else:
        fig, (axk, axd) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, LOCALIZE_HEIGHT_IN),
                                       sharey=True, gridspec_kw={"width_ratios": [1.15, 1.0]})
    t5 = [m for m in PANEL if m[2]]
    xpos = {D: i for i, D in enumerate(H1_DS)}
    for mid, name, _t5, c, mk, ls in t5:
        r = zb["ranked"][zb["ranked"].model == mid]
        q = zb["random"][zb["random"].model == mid]
        axk.fill_between(r.k, r["min"], r["max"], color=c, alpha=0.13, lw=0, zorder=2)
        axk.plot(q.k, q["median"], color=c, lw=0.9, ls=(0, (1, 1.5)), zorder=3)
        axk.plot(r.k, r["median"], color=c, lw=1.2, ls=ls, marker=mk, markersize=3,
                 markeredgewidth=0.6, zorder=4)
        b = cb[cb.model == mid]
        xs = [xpos[int(D)] for D in b.D]
        axd.fill_between(xs, b["min"], b["max"], color=c, alpha=0.13, lw=0, zorder=2)
        axd.plot(xs, b["median"], color=c, lw=1.2, ls=ls, marker=mk, markersize=3,
                 markeredgewidth=0.6, zorder=4)
    for ax in (axk, axd):
        ax.axhline(0.5, color="#888888", lw=0.6, zorder=1)
        ax.axhline(REPAIR_AUROC, color="#888888", lw=0.6, ls="--", zorder=1)
        ax.grid(True, zorder=0)
    axk.axvline(10, color="#888888", lw=0.6, ls=":", zorder=1)
    axk.set_xscale("log")
    ticks = (1, 3, 10, 30, 100, 400)
    axk.set_xticks(ticks)
    axk.set_xticklabels([str(k) for k in ticks])
    axk.xaxis.set_minor_formatter(NullFormatter())
    axk.set_xlim(0.85, 470)
    axk.set_xlabel("Coordinates zeroed (k)")
    axk.set_ylabel("Test AUROC")
    axk.set_ylim(0.45, 1.0)
    axk.set_title("(a) Zeroing the largest coordinates", loc="left", pad=3)
    axd.set_xticks(list(xpos.values()))
    axd.set_xticklabels([str(D) for D in H1_DS])
    axd.set_xlim(-0.3, len(H1_DS) - 0.7)
    axd.set_xlabel("Components removed (D)")
    axd.set_title("(b) Removing top principal components", loc="left", pad=3)
    if column:
        axd.set_ylim(0.45, 1.0)
        axd.set_ylabel("Test AUROC")
    hs = [Line2D([0], [0], color=c, lw=1.2, ls=ls, marker=mk, markersize=3.5, label=name)
          for _mid, name, _t5, c, mk, ls in t5]
    hs.append(Line2D([0], [0], color="#444444", lw=0.9, ls=(0, (1, 1.5)),
                     label="Random coordinates" + ("" if column else " (seed mean)")))
    if column:
        fig.legend(handles=hs, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=0.8)
        fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.6)
    else:
        fig.legend(handles=hs, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.12), handlelength=2.6, columnspacing=1.6)
        fig.tight_layout(w_pad=1.0)
    _save(fig, path)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="depth,diagnostics,localize")
    ap.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    args = ap.parse_args(argv)
    which = set(args.only.split(","))
    res = pd.read_csv(RES_CSV)
    if "depth" in which:
        fig_depth(depth_data(res), args.fig_dir / "fig_depth.pdf")
        print(f"wrote {args.fig_dir / 'fig_depth.pdf'}")
    if "diagnostics" in which:
        p2x2 = pd.read_csv(P2X2_CSV) if P2X2_CSV.exists() else None
        if p2x2 is None:
            print(f"note: {P2X2_CSV} missing; T5-base and T5-v1.1-base left out")
        fig_diagnostics(diagnostics_data(res, pd.read_csv(GEOM_CSV), p2x2),
                        args.fig_dir / "fig_diagnostics.pdf")
        print(f"wrote {args.fig_dir / 'fig_diagnostics.pdf'}")
    if "localize" in which:
        coll = collapsed_layers(res)
        zb = zeroing_bands(pd.read_csv(KSWEEP_CSV), coll)
        cb = components_band(pd.read_csv(H1_CSV), coll)
        fig_localize(zb, cb, args.fig_dir / "fig_localize.pdf")
        print(f"wrote {args.fig_dir / 'fig_localize.pdf'}")
        fig_localize(zb, cb, args.fig_dir / "fig_localize_col.pdf", column=True)
        print(f"wrote {args.fig_dir / 'fig_localize_col.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
