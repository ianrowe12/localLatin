#!/usr/bin/env python3
"""GEN and FT (issue #234): label-free layer geometry across model x text, and the
fine-tuned LaTa layerwise row.

Definitions are the paper's (Section "Two Geometries", geometry_vs_retrieval.py, computed
upstream by scripts/resubmit/run_layer_geometry_diagnostics.py, whose `pca_stats` and
`cosine_stats` are imported here unchanged):
  top-PC share   first eigenvalue share of the centered (not L2-normalized) covariance of the
                 mean-pooled vectors
  effective rank exp(entropy) of the same eigenvalue shares
  mean cosine    mean cosine over all pairs of the same rows
There is no fit/apply step: the statistics are read directly on a set of rows. The paper
reads them on the 847 training passages. Here the primary subset is the same: the Latin
training passages, and for English the 847 passages whose Latin partner is a training
passage (so n and the length profile are identical). Every cell is also reported on all
1,705 rows as a sensitivity check.

Stages:
  compute  reads embeddings (CPU), writes CSVs under --out_dir:
             gen_geometry.csv      model, text, layer, subset, n, pc1, erank, pc10, mean_cos
             gen_repro.csv         mT5-base / PhilTa Latin: new vs published geometry, and
                                   new vs cached vectors (max |diff|, min cosine)
             ft_layerwise.csv      LaTa pre-trained vs fine-tuned: per-layer baseline test
                                   AUROC (from the run records) and train top-PC share,
                                   effective rank (recomputed on the cached vectors; the train
                                   rows include the fine-tuning data, as in the paper)
             gen_rematch.csv       sensitivity: train rows re-matched on each model's own
                                   tokenizer lengths (also runnable alone: --stage rematch)
  render   reads those CSVs only, writes
             overleaf_drafts/tables/gen_geometry.tex       (label tab:gen_geometry)
             overleaf_drafts/tables/ft_lata_layerwise.tex  (label tab:ft_lata_layerwise)
             overleaf_drafts/figures/fig_gen_geometry.{pdf,png}
             <out_dir>/gen_ft_facts.md

Run from the repo root:
  python scripts/paper/reframe/gen_ft_geometry.py --stage compute --runs_root runs/active
  python scripts/paper/reframe/gen_ft_geometry.py --stage render
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HEADER = "% generated table"
TAB_DIR = Path("overleaf_drafts/tables")
FIG_DIR = Path("overleaf_drafts/figures")
OUT_DIR = Path("runs/active/reframe/gen")

GEN_MODELS = [  # id, display, colour (Okabe-Ito, as geometry_vs_retrieval.py), marker
    ("google/mt5-base", "mT5-base", "#009E73", "^"),
    ("bowphs/PhilTa", "PhilTa", "#E69F00", "s"),
    ("google/t5-v1_1-base", "T5-v1.1-base", "#882255", "o"),
]
TEXTS = [("latin", "Latin"), ("english", "English")]
DISP = {m[0]: m[1] for m in GEN_MODELS}
COLLAPSE_PC1 = 0.6  # the paper's separation threshold (Sec. 4: marks all 26 collapsed layers)


# --------------------------------------------------------------------------- compute
def _geometry_fns():
    for p in (REPO / "src", REPO / "scripts" / "resubmit"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from run_layer_geometry_diagnostics import cosine_stats, pca_stats  # noqa: E402
    return pca_stats, cosine_stats


def layer_stats(x: np.ndarray) -> dict:
    pca_stats, cosine_stats = _geometry_fns()
    p = pca_stats(x)
    c = cosine_stats(x)
    return dict(n=int(x.shape[0]), pc1=p["pc1_variance_ratio"], erank=p["effective_rank_entropy"],
                pc10=p["pc10_cumulative_variance_ratio"], mean_cos=c["anisotropy_mean_cosine"])


def compute(args) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from embedding_alignment import AlignmentResolver

    runs = Path(args.runs_root)
    split = pd.read_csv(args.split_csv)
    train = split["split"].to_numpy() == "train"
    resolver = AlignmentResolver(split)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    new_latin = {}
    for mid, disp, _, _ in GEN_MODELS:
        for text, _ in TEXTS:
            base = runs / "reframe/gen/bases" / mid.replace("/", "_") / text
            for layer in range(1, 13):
                x = resolver.load(base / f"hidden_layer{layer}_embeddings.npy")
                if text == "latin":
                    new_latin[(mid, layer)] = x
                for subset, mask in [("train", train), ("all", np.ones_like(train))]:
                    rows.append(dict(model=disp, text=text, layer=layer, subset=subset,
                                     **layer_stats(x[mask])))
                print(f"{disp} {text} L{layer}: pc1 {rows[-2]['pc1']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "gen_geometry.csv", index=False, float_format="%.6g")

    # Latin reproduction: published geometry (train, raw view) and the cached vectors.
    pub = pd.read_csv(runs / "resubmit/layer_diagnostics/geometry_per_layer.csv")
    pub = pub[(pub.split == "train") & (pub.view == "raw")]
    rep = []
    for mid in ["google/mt5-base", "bowphs/PhilTa"]:
        for layer in range(1, 13):
            p = pub[(pub.model == mid) & (pub.layer == layer)].iloc[0]
            x_new = new_latin[(mid, layer)]
            x_old = resolver.load(runs / "resubmit_bases/phase9_bases" / mid.replace("/", "_")
                                  / "hidden_mean_tokempty" / f"hidden_layer{layer}_embeddings.npy")
            s = layer_stats(x_new[train])
            cos = np.sum(x_new * x_old, 1) / (np.linalg.norm(x_new, axis=1) * np.linalg.norm(x_old, axis=1))
            rep.append(dict(model=DISP[mid], layer=layer, pc1_new=s["pc1"],
                            pc1_published=p.pc1_variance_ratio, erank_new=s["erank"],
                            erank_published=p.effective_rank_entropy,
                            vec_max_abs_diff=float(np.abs(x_new - x_old).max()),
                            vec_max_rel_diff=float((np.linalg.norm(x_new - x_old, axis=1)
                                                    / np.linalg.norm(x_old, axis=1)).max()),
                            vec_min_cosine=float(cos.min())))
    pd.DataFrame(rep).to_csv(out_dir / "gen_repro.csv", index=False, float_format="%.6g")

    # FT: fine-tuned LaTa vs pre-trained LaTa, baseline view.
    res = pd.read_csv(runs / "resubmit/results/phase_resubmit_results.csv")
    ftr = pd.read_csv(runs / "resubmit/results/finetune/finetune_lata_layer_results.csv")
    ft = []
    for tag, bases, table, model in [
        ("pretrained", runs / "resubmit_bases/phase9_bases/bowphs_LaTa", res, "bowphs/LaTa"),
        ("finetuned", runs / "resubmit_finetune_bases/phase9_bases/bowphs_LaTa-ft", ftr, "bowphs/LaTa-ft"),
    ]:
        for layer in range(1, 13):
            x = resolver.load(bases / "hidden_mean_tokempty" / f"hidden_layer{layer}_embeddings.npy")
            s = layer_stats(x[train])
            r = table[(table.model == model) & (table.method == "baseline") & (table.layer == layer)].iloc[0]
            a = table[(table.model == model) & (table.method == "abtt_optimal") & (table.layer == layer)].iloc[0]
            ft.append(dict(model=tag, layer=layer, auroc=r.aucroc, gap=r.gap, auroc_abtt=a.aucroc,
                           D_abtt=int(a.D), pc1=s["pc1"], erank=s["erank"], mean_cos_train=s["mean_cos"]))
    pd.DataFrame(ft).to_csv(out_dir / "ft_layerwise.csv", index=False, float_format="%.6g")
    print(resolver.summary())
    rematch(args)


LEN_KEY = {"mT5-base": "mt5", "PhilTa": "philta", "T5-v1.1-base": "t5v11"}


def rematch_pairs(lat_len, eng_len, max_length: int = 512):
    """Length-matched Latin/English index pairs under one tokenizer.

    Lengths are clipped at the encoder's max_length (what the model sees). Both sides are
    sorted and paired greedily with a two-pointer sweep, accepting a pair when the English
    length is within max(2, 3%) tokens of the Latin one (gen_english_sample.tolerance).
    Returns (latin_idx, english_idx), each sorted by the Latin length.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_english_sample import tolerance

    a = np.minimum(np.asarray(lat_len), max_length)
    b = np.minimum(np.asarray(eng_len), max_length)
    ia, ib = np.argsort(a, kind="stable"), np.argsort(b, kind="stable")
    i = j = 0
    li, ej = [], []
    while i < len(ia) and j < len(ib):
        la, lb = int(a[ia[i]]), int(b[ib[j]])
        if abs(lb - la) <= tolerance(la):
            li.append(int(ia[i]))
            ej.append(int(ib[j]))
            i += 1
            j += 1
        elif lb < la:
            j += 1
        else:
            i += 1
    return np.array(li, dtype=int), np.array(ej, dtype=int)


def rematch(args) -> None:
    """Sensitivity: re-match the train rows on each model's own tokenizer lengths.

    The sample is length-matched under the mT5 tokenizer only. For each model, Latin and
    English training rows are re-paired on that model's token lengths (rematch_pairs) and
    the geometry is recomputed on the matched rows. Writes gen_rematch.csv.
    """
    sys.path.insert(0, str(REPO / "src"))
    from embedding_alignment import AlignmentResolver

    runs, out_dir = Path(args.runs_root), Path(args.out_dir)
    split = pd.read_csv(args.split_csv)
    lens = pd.read_csv(out_dir / "english_sample_lengths.csv")
    if list(lens.latin_filename) != list(split.filename):
        raise SystemExit("english_sample_lengths.csv is not in split-CSV order")
    tr = np.flatnonzero(split["split"].to_numpy() == "train")
    resolver = AlignmentResolver(split)
    rows = []
    for mid, disp, _, _ in GEN_MODELS:
        k = LEN_KEY[disp]
        li, ej = rematch_pairs(lens[f"latin_{k}"].to_numpy()[tr], lens[f"english_{k}"].to_numpy()[tr])
        base = runs / "reframe/gen/bases" / mid.replace("/", "_")
        for layer in range(1, 13):
            for text, idx in [("latin", tr[li]), ("english", tr[ej])]:
                x = resolver.load(base / text / f"hidden_layer{layer}_embeddings.npy")
                rows.append(dict(model=disp, text=text, layer=layer, subset="rematch_train",
                                 tokenizer=k, **layer_stats(x[idx])))
        print(f"rematch {disp} ({k}): n = {len(li)}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "gen_rematch.csv", index=False, float_format="%.6g")


# --------------------------------------------------------------------------- render
def summarize(geo: pd.DataFrame, subset: str = "train") -> pd.DataFrame:
    """One row per model x text: peak top-PC share, minimum effective rank, collapsed layers."""
    out = []
    g = geo[geo.subset == subset]
    for _, disp, _, _ in GEN_MODELS:
        for text, tdisp in TEXTS:
            s = g[(g.model == disp) & (g.text == text)].sort_values("layer")
            if s.empty:
                continue
            pk = s.loc[s.pc1.idxmax()]
            lo = s.loc[s.erank.idxmin()]
            hs = s[s.pc1 >= COLLAPSE_PC1]
            hi = hs.layer.tolist()
            out.append(dict(model=disp, text=tdisp, n=int(s.n.iloc[0]), pc1_max=pk.pc1,
                            pc1_max_layer=int(pk.layer), erank_min=lo.erank,
                            erank_min_layer=int(lo.layer), n_high=len(hi),
                            high_layers=_ranges(hi), pc1_hi_min=hs.pc1.min(),
                            pc1_hi_max=hs.pc1.max(), erank_hi_min=hs.erank.min(),
                            erank_hi_max=hs.erank.max(),
                            pc1_L1=s.pc1.iloc[0], pc1_L12=s.pc1.iloc[-1]))
    return pd.DataFrame(out)


def _ranges(layers) -> str:
    """[2,3,4,7] -> '2--4, 7' (LaTeX en dash); [] -> '--'."""
    if not layers:
        return "none"
    parts, start, prev = [], layers[0], layers[0]
    for x in list(layers[1:]) + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        parts.append(f"{start}" if start == prev else f"{start}--{prev}")
        if x is not None:
            start = prev = x
    return ", ".join(parts)


def write_gen_table(summ: pd.DataFrame, path: Path) -> None:
    lines = [HEADER, r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"& & & \multicolumn{3}{c}{Layers with PC1 $\ge$ " + f"{COLLAPSE_PC1:.1f}" + r"} \\",
             r"\cmidrule(lr){4-6}",
             r"Model & Text & PC1$_{\max}$ ($\ell$) & Layers & PC1 & Eff.\ rank \\", r"\midrule"]
    prev = None
    for _, x in summ.iterrows():
        if prev is not None and x.model != prev:
            lines.append(r"\addlinespace")
        name = x.model if x.model != prev else ""
        lines.append(f"{name} & {x.text} & {x.pc1_max:.3f} ({x.pc1_max_layer}) & {x.high_layers} & "
                     f"{x.pc1_hi_min:.3f}--{x.pc1_hi_max:.3f} & {x.erank_hi_min:.2f}--{x.erank_hi_max:.2f} \\\\")
        prev = x.model
    n = int(summ.n.iloc[0]) if len(summ) else 0
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Label-free layer geometry crossed by model and text. Each cell reads the "
              r"mean-pooled hidden states of " + f"{n:,}".replace(",", "{,}") + r" passages: the Latin "
              r"training passages, or the English passages paired with them one to one and matched "
              r"in mT5 token length (US court opinions, Caselaw Access Project). PC1$_{\max}$: peak "
              r"top-PC share over the 12 layers, the share of centered variance on the first "
              r"principal component, with its layer $\ell$. The last three columns cover the layers "
              r"whose top-PC share is at least " + f"{COLLAPSE_PC1:.1f}" + r", the threshold that "
              r"marks every collapsed Latin layer (one false alarm, mT5-base layer~4), and give the "
              r"range of top-PC share and of entropy effective rank over them. No labels are used, so the "
              r"English rows say nothing about retrieval.}",
              r"\label{tab:gen_geometry}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_ft_table(ft: pd.DataFrame, path: Path) -> None:
    p = ft[ft.model == "pretrained"].set_index("layer")
    f = ft[ft.model == "finetuned"].set_index("layer")
    lines = [HEADER, r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}rcccccc@{}}", r"\toprule",
             r"& \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{Top-PC share} & "
             r"\multicolumn{2}{c}{Eff.\ rank} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Layer & PT & FT & PT & FT & PT & FT \\", r"\midrule"]
    for layer in p.index:
        a, b = p.loc[layer], f.loc[layer]
        lines.append(f"{layer} & {a.auroc:.3f} & {b.auroc:.3f} & {a.pc1:.3f} & {b.pc1:.3f} & "
                     f"{a.erank:.2f} & {b.erank:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{LaTa before (PT) and after (FT) contrastive fine-tuning on training pairs, "
              r"per layer, with no post-hoc correction. AUROC: Task~A test pairwise AUROC of cosine on "
              r"mean-pooled vectors. Top-PC share and effective rank: label-free geometry of the "
              r"centered training vectors, which include the passages of the fine-tuning pairs. Fine-tuning lifts the last layer and leaves the middle "
              r"layers collapsed.}",
              r"\label{tab:ft_lata_layerwise}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def fig_gen(geo: pd.DataFrame, out: Path, subset: str = "train") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "grid.linewidth": 0.4, "grid.color": "#dddddd", "svg.hashsalt": "gen234"})
    g = geo[geo.subset == subset]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35))
    for ax, col, yl, title in [(axes[0], "pc1", "Top-PC share", "(a) Top-PC share"),
                               (axes[1], "erank", "Effective rank (log)", "(b) Effective rank")]:
        for _, disp, c, mk in GEN_MODELS:
            for text, _ in TEXTS:
                s = g[(g.model == disp) & (g.text == text)].sort_values("layer")
                latin = text == "latin"
                ax.plot(s.layer, s[col], color=c, lw=1.2, linestyle="-" if latin else "--",
                        marker=mk, markersize=3.4, markerfacecolor=c if latin else "white",
                        markeredgecolor=c, markeredgewidth=0.9, zorder=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(1, 13))
        ax.grid(True, zorder=0)
        if col == "pc1":
            ax.axhline(COLLAPSE_PC1, color="#888888", lw=0.7, ls=":", zorder=1)
            ax.set_ylim(0, 1.03)
        else:
            ax.set_yscale("log")
    hs = [Line2D([0], [0], color=c, marker=mk, markersize=4, lw=1.2, label=d) for _, d, c, mk in GEN_MODELS]
    hs += [Line2D([0], [0], color="#444444", lw=1.2, ls="-", marker="o", markersize=4, label="Latin"),
           Line2D([0], [0], color="#444444", lw=1.2, ls="--", marker="o", markersize=4,
                  markerfacecolor="white", label="English")]
    fig.legend(handles=hs, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.2)
    meta = {"CreationDate": None, "ModDate": None}
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", metadata=meta)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def rematch_summary(geo: pd.DataFrame, rem: pd.DataFrame) -> pd.DataFrame:
    """Per model x text: re-matched n, max |change| of top-PC share and effective rank against
    the primary train cells, whether the set of layers with share >= threshold is unchanged,
    and in how many layers the Latin-minus-English sign of the share is unchanged."""
    g = geo[geo.subset == "train"].set_index(["model", "text", "layer"]).sort_index()
    r = rem.set_index(["model", "text", "layer"]).sort_index()
    out = []
    for _, disp, _, _ in GEN_MODELS:
        if disp not in r.index.get_level_values(0):
            continue
        sign_same = 0
        for layer in range(1, 13):
            d0 = g.loc[(disp, "latin", layer), "pc1"] - g.loc[(disp, "english", layer), "pc1"]
            d1 = r.loc[(disp, "latin", layer), "pc1"] - r.loc[(disp, "english", layer), "pc1"]
            sign_same += int(np.sign(d0) == np.sign(d1))
        for text, tdisp in TEXTS:
            a = g.loc[(disp, text)].sort_index()
            b = r.loc[(disp, text)].sort_index()
            hi_a = a.index[a.pc1 >= COLLAPSE_PC1].tolist()
            hi_b = b.index[b.pc1 >= COLLAPSE_PC1].tolist()
            out.append(dict(model=disp, text=tdisp, tokenizer=b.tokenizer.iloc[0], n=int(b.n.iloc[0]),
                            max_dpc1=float((b.pc1 - a.pc1).abs().max()),
                            max_dpc1_high=float((b.pc1 - a.pc1).abs()[hi_a].max()) if hi_a else 0.0,
                            max_derank_rel=float((b.erank / a.erank - 1).abs().max()),
                            high_layers=_ranges(hi_b), same_high_layers=hi_a == hi_b,
                            pc1_max=float(b.pc1.max()), sign_same_layers=sign_same))
    return pd.DataFrame(out)


def facts(geo, summ, summ_all, rep, ft, path: Path, rem: pd.DataFrame | None = None) -> None:
    L = ["# GEN and FT numbers (generated)", "",
         "Generated by `scripts/paper/reframe/gen_ft_geometry.py --stage render`. Geometry on the "
         "847-row subset (Latin training passages / their English partners) unless marked 'all'.", ""]
    w = L.append
    for title, sm in [("train subset (primary)", summ), ("all 1,705 rows (sensitivity)", summ_all)]:
        w(f"## Summary, {title}")
        w("| model | text | n | PC1 max (layer) | layers PC1>=0.6 | PC1 range there | eff rank range there | min eff rank (layer) | PC1 L1 | PC1 L12 |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for _, x in sm.iterrows():
            w(f"| {x.model} | {x.text} | {x.n} | {x.pc1_max:.3f} ({x.pc1_max_layer}) | {x.high_layers.replace('--', '-')} | "
              f"{x.pc1_hi_min:.3f}-{x.pc1_hi_max:.3f} | {x.erank_hi_min:.2f}-{x.erank_hi_max:.2f} | "
              f"{x.erank_min:.2f} ({x.erank_min_layer}) | "
              f"{x.pc1_L1:.3f} | {x.pc1_L12:.3f} |")
        w("")
    w("## Per layer (train subset): top-PC share / effective rank / mean pairwise cosine (all on train rows)")
    g = geo[geo.subset == "train"]
    for _, disp, _, _ in GEN_MODELS:
        for text, _ in TEXTS:
            s = g[(g.model == disp) & (g.text == text)].sort_values("layer")
            w(f"- {disp} {text}: " + "; ".join(
                f"{int(x.layer)}: {x.pc1:.3f} / {x.erank:.2f} / {x.mean_cos:.3f}" for _, x in s.iterrows()))
    w("")
    if rem is not None:
        rs = rematch_summary(geo, rem)
        w("## Sensitivity: train rows re-matched on each model's own tokenizer lengths")
        w("Latin and English training rows re-paired greedily on the model's own token lengths "
          "(clipped at 512, tolerance max(2, 3%)); geometry recomputed on the matched rows.")
        w("| model | text | tokenizer | n | max abs change in PC1 vs primary | same, layers PC1>=0.6 only | max rel change in eff rank | layers PC1>=0.6 | same layers as primary | peak PC1 | layers with Latin-English sign unchanged |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, x in rs.iterrows():
            w(f"| {x.model} | {x.text} | {x.tokenizer} | {x.n} | {x.max_dpc1:.3f} | {x.max_dpc1_high:.3f} | {x.max_derank_rel:.3f} | "
              f"{x.high_layers.replace('--', '-')} | {x.same_high_layers} | {x.pc1_max:.3f} | {x.sign_same_layers}/12 |")
        w(f"- overall max abs change in top-PC share: {rs.max_dpc1.max():.3f} (over layers with "
          f"share >= {COLLAPSE_PC1}: {rs.max_dpc1_high.max():.3f})")
        w("")
    if rep is not None:
        w("## Latin reproduction (mT5-base, PhilTa): new pipeline vs published geometry_per_layer.csv")
        w(f"- max |pc1_new - pc1_published| = {np.abs(rep.pc1_new - rep.pc1_published).max():.2e}")
        w(f"- max |erank_new - erank_published| = {np.abs(rep.erank_new - rep.erank_published).max():.2e}"
          f" (max relative {np.abs(rep.erank_new / rep.erank_published - 1).max():.2e})")
        w(f"- vectors vs cached .npy: max |diff| {rep.vec_max_abs_diff.max():.2e}, max relative L2 "
          f"diff {rep.vec_max_rel_diff.max():.2e}, min cosine {rep.vec_min_cosine.min():.8f}")
        for _, x in rep.iterrows():
            w(f"  - {x.model} L{int(x.layer)}: pc1 {x.pc1_new:.4f} vs {x.pc1_published:.4f}; erank "
              f"{x.erank_new:.3f} vs {x.erank_published:.3f}; max |diff| {x.vec_max_abs_diff:.1e}")
        w("")
    if ft is not None:
        p = ft[ft.model == "pretrained"].set_index("layer")
        f = ft[ft.model == "finetuned"].set_index("layer")
        mid = list(range(2, 12))
        w("## FT: LaTa pre-trained vs fine-tuned (baseline view; AUROC from the run records)")
        for tag, d in [("pre-trained", p), ("fine-tuned", f)]:
            w(f"- {tag}: AUROC layers 2-11 {d.loc[mid, 'auroc'].min():.3f}-{d.loc[mid, 'auroc'].max():.3f} "
              f"(min layer {int(d.loc[mid, 'auroc'].idxmin())}); min over all layers {d.auroc.min():.3f} "
              f"(layer {int(d.auroc.idxmin())}); layer 1 {d.loc[1, 'auroc']:.3f}; layer 12 {d.loc[12, 'auroc']:.3f}; "
              f"peak top-PC share {d.pc1.max():.3f} (layer {int(d.pc1.idxmax())}); top-PC share layers 2-11 "
              f"{d.loc[mid, 'pc1'].min():.3f}-{d.loc[mid, 'pc1'].max():.3f}; layer 12 {d.loc[12, 'pc1']:.3f}; "
              f"min eff rank {d.erank.min():.2f} (layer {int(d.erank.idxmin())}); ABTT-optimal AUROC "
              f"{d.auroc_abtt.min():.4f}-{d.auroc_abtt.max():.4f}")
        w("- per layer (layer: AUROC PT -> FT / PC1 PT -> FT / rank PT -> FT):")
        for layer in p.index:
            w(f"  - {layer}: {p.loc[layer, 'auroc']:.3f} -> {f.loc[layer, 'auroc']:.3f} / "
              f"{p.loc[layer, 'pc1']:.3f} -> {f.loc[layer, 'pc1']:.3f} / "
              f"{p.loc[layer, 'erank']:.2f} -> {f.loc[layer, 'erank']:.2f}")
        w("")
    path.write_text("\n".join(L) + "\n")


def render(args) -> None:
    out_dir = Path(args.out_dir)
    geo = pd.read_csv(out_dir / "gen_geometry.csv")
    rep_p, ft_p = out_dir / "gen_repro.csv", out_dir / "ft_layerwise.csv"
    rep = pd.read_csv(rep_p) if rep_p.exists() else None
    ft = pd.read_csv(ft_p) if ft_p.exists() else None
    rem_p = out_dir / "gen_rematch.csv"
    rem = pd.read_csv(rem_p) if rem_p.exists() else None
    tab_dir, fig_dir = Path(args.tab_dir), Path(args.fig_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)
    summ, summ_all = summarize(geo, "train"), summarize(geo, "all")
    write_gen_table(summ, tab_dir / "gen_geometry.tex")
    if ft is not None:
        write_ft_table(ft, tab_dir / "ft_lata_layerwise.tex")
    if not args.no_figure:
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_gen(geo, fig_dir / "fig_gen_geometry")
    facts(geo, summ, summ_all, rep, ft, out_dir / "gen_ft_facts.md", rem)
    print("rendered tables, figure and", out_dir / "gen_ft_facts.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["compute", "rematch", "render", "all"], default="all")
    ap.add_argument("--runs_root", default="runs/active")
    ap.add_argument("--split_csv", default="runs/active/resubmit/data/phase_resubmit_split.csv")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--tab_dir", default=str(TAB_DIR))
    ap.add_argument("--fig_dir", default=str(FIG_DIR))
    ap.add_argument("--no_figure", action="store_true")
    args = ap.parse_args()
    if args.stage in ("compute", "all"):
        compute(args)
    if args.stage == "rematch":
        rematch(args)
    if args.stage in ("render", "all"):
        render(args)


if __name__ == "__main__":
    main()
