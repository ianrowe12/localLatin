#!/usr/bin/env python3
"""The main-text tables of the paper-spine rewrite (2 October 2026), and the appendix table of
all thirteen models, rendered from committed result files only (no embeddings, no GPU).

Outputs (``overleaf_drafts/tables/``):

``models_main.tex``    T1, label ``tab:models``: the ten main-text models, grouped by embedding
                       training and, for the raw T5 encoders, by layout; lowest
                       baseline AUROC (layer), peak top-PC share, number of collapsed layers.
``all_models.tex``     appendix, label ``tab:all_models``: all thirteen models with the fuller
                       statistics and the checkpoint facts.
``predictions.tex``    T2, label ``tab:predictions`` (one column, terse cells): each
                       localization test of Section 6, what
                       the account under test expects, the outcome with its key number, and
                       whether the expectation held. A dagger marks tests written down before
                       the run, as an expectation or as a rival account's decision rule
                       (``reframe_handoff_20260927.md`` and the frozen rules of
                       ``reframe_e2.md``); the other rows are controls or observations added after.
``headline_main.tex``  T3, label ``tab:headline``: ranking AUROC and routing DirAcc@1, baseline and
                       ABTT for the six panel encoders (with bootstrap intervals), the three
                       fine-tuned encoders (intervals only in tab:headline_ci) and the
                       character n-gram reference (with its interval).

Inputs (all committed):
  runs/active/reframe/p2x2/p2x2_layers.csv                 the twelve-layer controls, per layer
  runs/active/reframe/d2/p2x2_layers.csv                   T5-efficient-base (D2), per layer
  runs/active/resubmit/results/phase_resubmit_results.csv  panel baseline AUROC, per layer
  runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv  panel train geometry, per layer
  runs/active/reframe/e1/e1_k_sweep.csv                    coordinate zeroing and its random control
  runs/active/reframe/h1/h1_d_ablation.csv                 centering and ABTT with D components
  runs/active/reframe/e3/e3_subspace_split.csv             removed against retained subspace
  runs/active/reframe/e2/e2_token_ablation.csv             token ablation and mass-matched control
  runs/active/reframe/e2/e2_direction_audit.csv            Spearman of the PC1 score with log length
  docs/research/data/reframe_ci_pq/headline_ci.csv, run_info.json   headline cells and intervals
plus ``CHECKPOINTS`` below: feed-forward type and width and embedding tying, read from each
checkpoint's Hugging Face ``config.json`` on 2026-10-02 (not stored in any repo file).

For the four models in both sources (LaTa, PhilTa, mT5-base, LaBSE) the panel CSVs and
``p2x2_layers.csv`` must agree (AUROC and top-PC share to 1e-6); rendering stops otherwise.

Run from the repo root:
  python scripts/paper/reframe/spine_tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HEADER = "% generated table\n% python scripts/paper/reframe/spine_tables.py\n"
TAB_DIR = Path("overleaf_drafts/tables")

P2X2_CSV = Path("runs/active/reframe/p2x2/p2x2_layers.csv")
D2_CSV = Path("runs/active/reframe/d2/p2x2_layers.csv")  # T5-efficient-base (reframe_d2_controls.md)
RES_CSV = Path("runs/active/resubmit/results/phase_resubmit_results.csv")
GEO_CSV = Path("runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv")
KSWEEP_CSV = Path("runs/active/reframe/e1/e1_k_sweep.csv")
DABL_CSV = Path("runs/active/reframe/h1/h1_d_ablation.csv")
SPLIT3_CSV = Path("runs/active/reframe/e3/e3_subspace_split.csv")
TOKABL_CSV = Path("runs/active/reframe/e2/e2_token_ablation.csv")
AUDIT_CSV = Path("runs/active/reframe/e2/e2_direction_audit.csv")
CI_DIR = Path("docs/research/data/reframe_ci_pq")

COLLAPSE = 0.70      # a collapsed layer: baseline test AUROC below this
HIGH_PC1 = 0.76      # the top-PC share every collapsed layer reaches (min 0.764); appendix column
REPAIRED = 0.90      # the repair bar of the zeroing and token-ablation rules
AGREE_TOL = 1e-6

# display name -> (HF id, type, embedding objective, group, source)
# type: T5 = the encoder of a T5 encoder-decoder; Enc. = encoder-only; Dec. = decoder.
# group: v11 = raw T5, T5 v1.1 layout; orig = raw T5, original layout; enc = raw encoder-only;
#        emb = embedding-trained.  source: panel = the published panel CSVs; p2x2 = p2x2_layers.csv;
#        d2 = the D2 run's p2x2_layers.csv (T5-efficient-base only)
MODELS: Dict[str, Tuple[str, str, str, str, str]] = {
    "LaTa": ("bowphs/LaTa", "T5", "none", "v11", "panel"),
    "PhilTa": ("bowphs/PhilTa", "T5", "none", "v11", "panel"),
    "mT5-base": ("google/mt5-base", "T5", "none", "v11", "panel"),
    "T5-v1.1-base": ("google/t5-v1_1-base", "T5", "none", "v11", "p2x2"),
    "T5-base": ("google-t5/t5-base", "T5", "none", "orig", "p2x2"),
    "T5-efficient-base": ("google/t5-efficient-base", "T5", "none", "orig", "d2"),
    "LaBERTa": ("bowphs/LaBerta", "Enc.", "none", "enc", "p2x2"),
    "PhilBERTa": ("bowphs/PhilBerta", "Enc.", "none", "enc", "p2x2"),
    "LaBSE": ("sentence-transformers/LaBSE", "Enc.", "contrastive", "emb", "panel"),
    "Qwen3-0.6B": ("Qwen/Qwen3-Embedding-0.6B", "Dec.", "contrastive", "emb", "panel"),
    "KaLM-mini": ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", "Dec.",
                  "contrastive", "emb", "panel"),
    "Sentence-T5": ("sentence-transformers/sentence-t5-base", "T5", "contrastive", "emb", "p2x2"),
    "SPhilBERTa": ("bowphs/SPhilBerta", "Enc.", "distillation", "emb", "p2x2"),
}
MAIN_MODELS = ("LaTa", "PhilTa", "mT5-base", "T5-v1.1-base", "T5-base", "LaBERTa", "PhilBERTa",
               "LaBSE", "Qwen3-0.6B", "KaLM-mini")
ALL_MODELS = ("LaTa", "PhilTa", "mT5-base", "T5-v1.1-base", "T5-base", "LaBERTa", "PhilBERTa",
              "LaBSE", "Qwen3-0.6B", "KaLM-mini", "Sentence-T5", "SPhilBERTa", "T5-efficient-base")
PANEL = ("LaTa", "PhilTa", "mT5-base", "LaBSE", "Qwen3-0.6B", "KaLM-mini")
T5_PANEL_IDS = ("bowphs/LaTa", "bowphs/PhilTa", "google/mt5-base")

# Feed-forward activation, inner width d_ff (intermediate_size), and whether the input and
# output embeddings are tied, from each checkpoint's config.json (fetched 2026-10-02).
# T5-base, T5-efficient-base, LaBERTa and PhilBERTa leave tie_word_embeddings unset, so the transformers default
# (True) applies; "n/a": the checkpoint has no output head (T5EncoderModel, BertModel,
# RobertaModel). Qwen3 and KaLM-mini (Qwen2) use a gated SiLU MLP (SwiGLU).
CHECKPOINTS: Dict[str, Tuple[str, int, str]] = {
    "LaTa": ("gated GELU", 2048, "no"),
    "PhilTa": ("gated GELU", 2048, "no"),
    "mT5-base": ("gated GELU", 2048, "no"),
    "T5-v1.1-base": ("gated GELU", 2048, "no"),
    "T5-base": ("ReLU", 3072, "yes"),
    "T5-efficient-base": ("ReLU", 3072, "yes"),
    "Sentence-T5": ("ReLU", 3072, "n/a"),
    "LaBERTa": ("GELU", 3072, "yes"),
    "PhilBERTa": ("GELU", 3072, "yes"),
    "LaBSE": ("GELU", 3072, "n/a"),
    "SPhilBERTa": ("GELU", 3072, "n/a"),
    "Qwen3-0.6B": ("gated SiLU", 3072, "yes"),
    "KaLM-mini": ("gated SiLU", 4864, "yes"),
}

GROUPS_MAIN = [
    ("v11", "Raw T5 encoders, T5 v1.1 layout"),
    ("orig", "Raw T5 encoder, original layout"),
    ("enc", "Raw encoder-only siblings"),
    ("emb", "Embedding-trained"),
]


# --------------------------------------------------------------------------- loading
def per_layer(root: Path) -> pd.DataFrame:
    """One row per (model, layer) for all thirteen models: test AUROC of the unmodified
    mean-pooled vectors, and top-PC share, effective rank and mean pairwise cosine of the
    training passages. Panel models come from the published CSVs, T5-efficient-base from the
    D2 run's p2x2_layers.csv, the others from the P2x2 p2x2_layers.csv; where the panel CSVs
    and the P2x2 file both hold a model they must agree."""
    root = Path(root)
    p2 = pd.read_csv(root / P2X2_CSV)
    d2 = pd.read_csv(root / D2_CSV)
    res = pd.read_csv(root / RES_CSV)
    res = res[(res["repr"] == "hidden") & (res["pooling"] == "mean") & (res["method"] == "baseline")]
    geo = pd.read_csv(root / GEO_CSV)
    geo = geo[(geo["split"] == "train") & (geo["view"] == "raw") & (geo["repr"] == "hidden")
              & (geo["pooling"] == "mean")]
    pub = res[["model", "layer", "aucroc"]].merge(
        geo[["model", "layer", "pc1_variance_ratio", "effective_rank_entropy",
             "anisotropy_mean_cosine"]], on=["model", "layer"], how="inner", validate="1:1")
    pub = pub.rename(columns={"model": "model_id", "pc1_variance_ratio": "pc1",
                              "effective_rank_entropy": "erank",
                              "anisotropy_mean_cosine": "mean_cos"})
    rows = []
    for name in ALL_MODELS:
        model_id, _, _, _, source = MODELS[name]
        table = {"panel": pub, "p2x2": p2, "d2": d2}[source]
        src = table[table["model_id"] == model_id]
        if src.empty:
            raise SystemExit(f"no per-layer rows for {name} ({model_id}) in the {source} source")
        s = src[["layer", "aucroc", "pc1", "erank", "mean_cos"]].copy()
        s.insert(0, "model", name)
        rows.append(s.sort_values("layer"))
    df = pd.concat(rows, ignore_index=True)
    # agreement where both sources hold the model
    both = p2[p2["source"] == "panel"].merge(pub, on=["model_id", "layer"], suffixes=("_p2", "_pub"))
    for col in ("aucroc", "pc1"):
        diff = float((both[f"{col}_p2"] - both[f"{col}_pub"]).abs().max()) if len(both) else 0.0
        if diff > AGREE_TOL:
            raise SystemExit(f"p2x2_layers.csv and the published panel CSVs disagree on {col} "
                             f"(max difference {diff:.2e})")
    return df


def _ranges(layers: Sequence[int]) -> str:
    """[2,3,4,7] -> '2--4, 7'; [] -> 'none'."""
    layers = sorted(int(x) for x in layers)
    if not layers:
        return "none"
    parts, start, prev = [], layers[0], layers[0]
    for x in layers[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        parts.append(f"{start}" if start == prev else f"{start}--{prev}")
        if x is not None:
            start = prev = x
    return ", ".join(parts)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model. Minima and maxima take the first layer on ties."""
    out = []
    for name in dict.fromkeys(df["model"]):
        s = df[df["model"] == name].sort_values("layer").reset_index(drop=True)
        lo = s.loc[s["aucroc"].idxmin()]
        pk = s.loc[s["pc1"].idxmax()]
        rk = s.loc[s["erank"].idxmin()]
        coll = s.loc[s["aucroc"] < COLLAPSE, "layer"].tolist()
        high = s.loc[s["pc1"] >= HIGH_PC1, "layer"].tolist()
        out.append({
            "model": name, "n_layers": len(s),
            "auroc_min": float(lo["aucroc"]), "auroc_min_layer": int(lo["layer"]),
            "pc1_max": float(pk["pc1"]), "pc1_max_layer": int(pk["layer"]),
            "erank_min": float(rk["erank"]), "erank_min_layer": int(rk["layer"]),
            "cos_min": float(s["mean_cos"].min()), "cos_max": float(s["mean_cos"].max()),
            "n_collapsed": len(coll), "collapsed_layers": _ranges(coll),
            "n_high_pc1": len(high), "high_pc1_layers": _ranges(high),
            "pc1_min_collapsed": float(s.loc[s["aucroc"] < COLLAPSE, "pc1"].min()) if coll else float("nan"),
        })
    return pd.DataFrame(out).set_index("model")


# --------------------------------------------------------------------------- T1 and appendix
def _f3(x: float) -> str:
    return f"{x:.3f}"


_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def render_models_main(summ: pd.DataFrame) -> str:
    v11 = [m for m in MAIN_MODELS if MODELS[m][3] == "v11"]
    others = [m for m in MAIN_MODELS if MODELS[m][3] != "v11"]
    n_v11 = [int(summ.loc[m, "n_collapsed"]) for m in v11]
    if min(n_v11) == 0 or any(summ.loc[m, "n_collapsed"] for m in others):
        raise SystemExit("the T1 caption assumes that exactly the T5 v1.1-layout rows collapse")
    lines = [HEADER.rstrip("\n"), r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}llrccr@{}}", r"\toprule",
             r"Model & Type & L & AUROC$_{\min}$ ($\ell$) & PC1$_{\max}$ & Coll. \\",
             r"\midrule"]
    for gi, (group, label) in enumerate(GROUPS_MAIN):
        names = [m for m in MAIN_MODELS if MODELS[m][3] == group]
        if gi:
            lines.append(r"\addlinespace")
        lines.append(r"\multicolumn{6}{@{}l}{\emph{" + label + r"}} \\")
        for m in names:
            x = summ.loc[m]
            lines.append(f"{m} & {MODELS[m][1]} & {int(x['n_layers'])} & "
                         f"{_f3(x['auroc_min'])} ({int(x['auroc_min_layer'])}) & "
                         f"{_f3(x['pc1_max'])} & {int(x['n_collapsed'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The ten main-text models, read as mean-pooled hidden states with no "
              r"correction. Type: T5 encoder (T5), encoder-only (Enc.) or decoder (Dec.; "
              r"KaLM-mini with bidirectional attention). L: blocks read. AUROC$_{\min}$: "
              r"lowest test ranking AUROC over layers, at layer $\ell$. PC1$_{\max}$: peak "
              r"top-PC share of training embeddings. Coll.: collapsed layers (AUROC below "
              r"0.70). Only the " + _WORDS[len(v11)] + r" raw T5 encoders with the T5 v1.1 layout "
              r"(gated-GELU feed-forward, untied embeddings) collapse.}",
              r"\label{tab:models}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_all_models(summ: pd.DataFrame) -> str:
    lines = [HEADER.rstrip("\n"), r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{1.8pt}", r"\begin{tabular}{@{}lllllrccccll@{}}", r"\toprule",
             r"& & & \multicolumn{3}{c}{Checkpoint} & \multicolumn{4}{c}{Over layers} & "
             r"\multicolumn{2}{c}{Layers} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-10}\cmidrule(lr){11-12}",
             r"Model & Type & Emb.\ obj. & Feed-forward & $d_\mathrm{ff}$ & Tied & "
             r"AUROC$_{\min}$ & PC1$_{\max}$ & Rank$_{\min}$ & Mean cos. & "
             r"Coll. & PC1$\ge$" + f"{HIGH_PC1:.2f}" + r" \\",
             r"\midrule"]
    blocks = [("Main text", [m for m in ALL_MODELS if m in MAIN_MODELS]),
              ("Appendix only", [m for m in ALL_MODELS if m not in MAIN_MODELS])]
    for bi, (label, names) in enumerate(blocks):
        if bi:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{12}{@{}l}{\emph{" + label + r"}} \\")
        for m in names:
            x = summ.loc[m]
            ff, dff, tied = CHECKPOINTS[m]
            lines.append(
                f"{m} & {MODELS[m][1]} & {MODELS[m][2]} & {ff} & {dff} & {tied} & "
                f"{_f3(x['auroc_min'])} ({int(x['auroc_min_layer'])}) & "
                f"{_f3(x['pc1_max'])} ({int(x['pc1_max_layer'])}) & "
                f"{x['erank_min']:.2f} ({int(x['erank_min_layer'])}) & "
                f"{x['cos_min']:.2f}--{x['cos_max']:.2f} & "
                f"{x['collapsed_layers']} & {x['high_pc1_layers']} \\\\")
    def tt(hf_id: str) -> str:  # break points after "/" and "-" for the long ids
        t = hf_id.replace("_", r"\_").replace("/", r"/\allowbreak{}")
        return r"\texttt{" + t.replace("-", r"-\allowbreak{}") + "}"
    ids = "; ".join(f"{m}, {tt(MODELS[m][0])}" for m in ALL_MODELS)
    coll = summ.loc[[m for m in ALL_MODELS if summ.loc[m, "n_collapsed"]]]
    n_coll = int(coll["n_collapsed"].sum())
    n_panel = int(sum(summ.loc[m, "n_collapsed"] for m in PANEL))
    rest = [m for m in coll.index if m not in PANEL]
    coll_split = f"{n_panel} in the panel, " + ", ".join(
        f"{int(summ.loc[m, 'n_collapsed'])} in {m}" for m in rest)
    floor = float(summ["pc1_min_collapsed"].min())
    if floor < HIGH_PC1:
        raise SystemExit(f"a collapsed layer has top-PC share {floor:.3f} < {HIGH_PC1}")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{All thirteen models, read on the Latin corpus as mean-pooled hidden "
              r"states with no post-hoc correction. Type: the encoder of a T5 encoder-decoder "
              r"(T5), an encoder-only model (Enc.), or a decoder (Dec.). Emb.\ obj.: the "
              r"embedding objective trained after pretraining, if any. Checkpoint: the "
              r"feed-forward activation and inner width $d_\mathrm{ff}$, and whether input and "
              r"output embeddings are tied, as the checkpoint's configuration file states them "
              r"(T5-base, T5-efficient-base, LaBERTa and PhilBERTa leave tying at the library "
              r"default, tied; n/a: "
              r"the checkpoint has no output head). Over layers: the lowest test ranking AUROC, "
              r"the peak top-PC share and the lowest entropy effective rank, each with its "
              r"layer, and the range of mean pairwise cosine; top-PC share, effective rank and "
              r"mean cosine are computed on the training embeddings. Layers: the collapsed layers "
              r"(Coll.), with AUROC below 0.70, and those with top-PC share of at least "
              + f"{HIGH_PC1:.2f}" + r". "
              r"Every one of the " + f"{n_coll}" + r" collapsed layers (" + coll_split
              + r") has a top-PC share of "
              r"at least " + f"{floor:.2f}" + r", but a high share also occurs without collapse "
              r"(mT5-base layer 4, Sentence-T5). Checkpoints: " + ids + r".}",
              r"\label{tab:all_models}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- T2
def collapsed_index(root: Path) -> pd.MultiIndex:
    """(model id, layer) of the collapsed T5 layers: published baseline AUROC below 0.70."""
    h = pd.read_csv(Path(root) / DABL_CSV)
    raw = h[h["variant"] == "raw"].set_index(["model", "layer"])["aucroc"]
    col = raw[raw < COLLAPSE]
    if not set(col.index.get_level_values(0)) <= set(T5_PANEL_IDS):
        raise SystemExit("a non-T5 layer is below 0.70; the predictions table assumes otherwise")
    return col.index


def prediction_numbers(root: Path) -> Dict[str, float]:
    """Every number that T2 prints, from the committed CSVs."""
    root = Path(root)
    idx = collapsed_index(root)
    n = len(idx)
    out: Dict[str, float] = {"n_collapsed": n}

    h = pd.read_csv(root / DABL_CSV).set_index(["model", "layer"])
    raw = h[h["variant"] == "raw"]["aucroc"].loc[idx]

    def at(D: int) -> pd.Series:
        return h[h["D"] == D]["aucroc"].loc[idx]

    center, d1, d3, d10 = at(0), at(1), at(3), at(10)
    share1 = (d1 - raw) / (d10 - raw)
    out.update(raw_median=float(raw.median()), center_median=float(center.median()),
               center_repaired=int((center >= REPAIRED).sum()),
               d1_share_median=float(share1.median()), d1_share_80=int((share1 >= 0.8).sum()),
               d3_min=float(d3.min()), d3_n_091=int((d3 >= 0.91).sum()),
               d3_median=float(d3.median()))
    mt5 = [i for i in idx if i[0] == "google/mt5-base"]
    out.update(mt5_d3_median=float(at(3).loc[mt5].median()),
               mt5_d10_median=float(at(10).loc[mt5].median()), n_mt5=len(mt5))

    k = pd.read_csv(root / KSWEEP_CSV).set_index(["model", "layer"])
    k = k[k.index.isin(idx)]
    z = k[(k["ranking"] != "random") & (k["k"] <= 10)]
    out.update(zero_best=float(z["aucroc"].max()),
               zero_repaired=int((z.groupby(level=[0, 1])["aucroc"].max() >= REPAIRED).sum()))
    r = k[k["ranking"] == "random"].reset_index()
    rm = r.groupby(["model", "layer", "k"])["aucroc"].mean()  # mean over seeds
    r10 = rm.xs(10, level="k")
    out.update(rand_k10_shift=float((r10 - raw.reindex(r10.index)).abs().max()),
               rand_repaired=int((rm.groupby(level=[0, 1]).max() >= REPAIRED).sum()),
               rand_kmax=int(r["k"].max()))

    e3 = pd.read_csv(root / SPLIT3_CSV)
    e3 = e3[e3["D_rule"] == "selected"].set_index(["model", "layer"])
    e3 = e3[~e3.index.duplicated()].loc[idx]
    if set(e3["D"].astype(int)) != {10}:  # the row text says "the ten components"
        raise SystemExit(f"ABTT removes D in {sorted(set(e3['D']))} at the collapsed layers, not 10")
    out.update(removed_median=float(e3["auc_removed"].median()),
               retained_median=float(e3["auc_retained"].median()),
               removed_below=int((e3["auc_removed"] < e3["auc_retained"]).sum()))

    t = pd.read_csv(root / TOKABL_CSV).set_index(["model", "layer"])
    t = t[t.index.isin(idx)]
    lata = t.loc["bowphs/LaTa"]
    cell = lata[(lata["ranking"] == "pc123") & (lata["m"] == 3)]
    car = cell[cell["kind"] == "carrier"]
    ctrl = cell[cell["kind"] == "control_mass"].groupby(level=0)["aucroc"].mean()
    out.update(lata_n=len(car), lata_restored=int((car["aucroc"] >= REPAIRED).sum()),
               lata_median=float(car["aucroc"].median()),
               lata_mass_share=float(car["dropped_mass_test"].median()),
               lata_ctrl_median=float(ctrl.median()))
    types = set(car["types"].map(lambda s: frozenset(str(s).split(";"))))
    if types != {frozenset({"1", "4", "5"})}:
        raise SystemExit(f"LaTa's three dropped types are not </s>, comma, period: {types}")
    mt = t.loc["google/mt5-base"]
    mt = mt[mt["kind"] == "carrier"]
    out.update(mt5_restored=int((mt.groupby(level=0)["aucroc"].max() >= REPAIRED).sum()),
               mt5_best=float(mt["aucroc"].max()), mt5_max_m=int(mt["m"].max()))

    a = pd.read_csv(root / AUDIT_CSV)
    a = a[(a["direction"] == "pc1") & (a["split"] == "test")].set_index(["model", "layer"])
    a = a[a.index.isin(idx)]
    out.update(len_n05=int((a["rho_logn"].abs() >= 0.5).sum()),
               len_max=float(a["rho_logn"].abs().max()))
    return out


def prediction_rows(x: Dict[str, float]) -> List[Tuple[str, str, str, str, str]]:
    """(section, test, expectation, outcome, verdict). Expectations marked \\dag were written
    down before the run. Verdicts follow the numbers, so a changed CSV changes them."""
    n = int(x["n_collapsed"])
    dag = r"$^\dagger$"

    def verdict(ok: bool) -> str:
        return "held" if ok else r"\textbf{failed}"

    rows = [
        ("Coordinates",
         "Zero top $k$ coordinates",
         "$k{\\le}5$ restores" + dag,
         f"{x['zero_repaired']}/{n} restored at $k{{\\le}}10$, either ranking; "
         f"best {x['zero_best']:.3f}",
         verdict(x["zero_repaired"] > 0)),
        ("Coordinates",
         "Zero random ones",
         "no change",
         f"$\\le${x['rand_k10_shift']:.4f} at 10; {x['rand_repaired']}/{n} restored "
         f"up to {x['rand_kmax']}",
         verdict(x["rand_repaired"] == 0 and x["rand_k10_shift"] < 0.01)),
        ("Directions",
         "Center only",
         "little gain" + dag,
         f"median {x['center_median']:.3f} (raw {x['raw_median']:.3f}); "
         f"{x['center_repaired']}/{n} restored",
         verdict(x["center_repaired"] == 0 and x["center_median"] < x["raw_median"] + 0.05)),
        ("Directions",
         "Remove PC1 only",
         "$\\ge$80\\% of gain" + dag,
         f"median {100 * x['d1_share_median']:.0f}\\%; $\\ge$80\\% at {x['d1_share_80']}/{n}",
         verdict(x["d1_share_median"] >= 0.8)),
        ("Directions",
         "Remove the top 3 PCs",
         "--",
         f"{x['d3_n_091']}/{n} $\\ge$0.91 (lowest {x['d3_min']:.3f}); mT5-base "
         f"{x['mt5_d3_median']:.3f}, {x['mt5_d10_median']:.3f} at 10",
         "--"),
        ("Directions",
         "Rank on the 10 removed PCs",
         "below retained" + dag,
         f"{x['removed_median']:.3f} vs.\\ {x['retained_median']:.3f}; below at "
         f"{x['removed_below']}/{n}",
         verdict(x["removed_below"] == n)),
        ("Tokens",
         "LaTa: drop top types",
         "restores" + dag,
         f"comma, period, \\texttt{{</s>}} ({100 * x['lata_mass_share']:.0f}\\% of tokens): "
         f"{x['lata_restored']}/{x['lata_n']}, median {x['lata_median']:.3f}; mass-matched "
         f"random types {x['lata_ctrl_median']:.3f}",
         verdict(x["lata_restored"] == x["lata_n"])),
        ("Tokens",
         f"mT5-base: drop $\\le${x['mt5_max_m']} types",
         "restores" + dag,
         f"{x['mt5_restored']}/{x['n_mt5']}; best {x['mt5_best']:.3f}",
         verdict(x["mt5_restored"] > 0)),
        ("Tokens",
         "PC1 score vs.\\ log length",
         "$|\\rho|{\\ge}0.5^\\dagger$",
         f"{x['len_n05']}/{n}; largest $|\\rho|$ {x['len_max']:.2f}",
         verdict(x["len_n05"] > 0)),
    ]
    return rows


def render_predictions(x: Dict[str, float]) -> str:
    """One-column table: terse cells, every number of the outcome kept."""
    n = int(x["n_collapsed"])
    lines = [HEADER.rstrip("\n"), r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.27\columnwidth}"
             r">{\raggedright\arraybackslash}p{0.17\columnwidth}"
             r">{\raggedright\arraybackslash}p{0.385\columnwidth}l@{}}",
             r"\toprule",
             r"Test & Expected & Outcome & Result \\",
             r"\midrule"]
    last = None
    for sec, test, exp, outcome, res in prediction_rows(x):
        if sec != last:
            if last is not None:
                lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{4}{@{}l}{\emph{" + {
                "Coordinates": "Coordinates",
                "Directions": "Principal directions",
                "Tokens": "Tokens"}[sec] + r"}} \\")
            last = sec
        lines.append(f"{test} & {exp} & {outcome} & {res} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Localization tests at the " + f"{n}" + r" collapsed layers (AUROC "
              r"below 0.70) of LaTa, PhilTa and mT5-base, unless a row names one model. Outcomes are test ranking "
              r"AUROC or medians over layers; ``restores'' means AUROC $\ge$ 0.90. Every intervention uses "
              r"training embeddings only. Coordinates are ranked by variance "
              r"or mean $|x|$. PC: principal component of the centered training embeddings; PC1: "
              r"the first. Gain: that of removing ten PCs; top types: those that feed the top "
              r"PCs. "
              r"$^\dagger$: written down before the run, as our expectation or as the decision "
              r"rule of a rival account (passage length); the other rows are a control and an "
              r"observation added afterwards.}",
              r"\label{tab:predictions}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- T3
FT_ROWS = [("LaTa (fine-tuned)", "LaTa"), ("Qwen3-0.6B (fine-tuned)", "Qwen3-0.6B"),
           ("KaLM-mini (fine-tuned)", "KaLM-mini")]
LEX_ROW = ("TF-IDF char 3-5", "Char.\\ n-grams")


def _cell(ci: pd.DataFrame, task: str, row: str, setting: str, metric: str) -> pd.Series:
    sub = ci[(ci["task"] == task) & (ci["row"] == row) & (ci["setting"] == setting)
             & (ci["metric"] == metric)]
    if len(sub) != 1:
        raise SystemExit(f"headline_ci.csv: {len(sub)} rows for {task}/{row}/{setting}/{metric}")
    return sub.iloc[0]


def _num(v: float, metric: str) -> str:
    return f"{v:.3f}" if metric == "auroc" else f"{100 * v:.1f}"


def _int(lo: float, hi: float, metric: str) -> str:
    a, b = _num(lo, metric), _num(hi, metric)
    if metric == "auroc":
        a, b = a[1:] if a.startswith("0.") else a, b[1:] if b.startswith("0.") else b
    return f"[{a},{b}]"


def headline_numbers(ci: pd.DataFrame) -> Dict[str, float]:
    """Spreads across the six panel encoders, from the printed (rounded) values."""
    out = {}
    for task, metric in (("A", "auroc"), ("B", "dir1")):
        for setting in ("Base", "ABTT"):
            vals = [float(_num(_cell(ci, task, m, setting, metric)["estimate"], metric))
                    for m in PANEL]
            out[f"{metric}_{setting}_spread"] = round(max(vals) - min(vals), 3 if metric == "auroc" else 1)
            out[f"{metric}_{setting}_min"] = min(vals)
            out[f"{metric}_{setting}_max"] = max(vals)
    lex = {metric: float(_num(_cell(ci, task, LEX_ROW[0], "ref", metric)["estimate"], metric))
           for task, metric in (("A", "auroc"), ("B", "dir1"))}
    out.update(lex_auroc=lex["auroc"], lex_dir1=lex["dir1"])
    return out


def render_headline(ci: pd.DataFrame, info: Dict) -> str:
    def pair(row: str, setting: str) -> Tuple[List[str], List[str]]:
        vals, ints = [], []
        for task, metric in (("A", "auroc"), ("B", "dir1")):
            c = _cell(ci, task, row, setting, metric)
            vals.append(_num(c["estimate"], metric))
            ints.append(_int(c["ci_lo"], c["ci_hi"], metric))
        return vals, ints

    def body(label: str, row: str, intervals: bool = True) -> List[str]:
        (a0, b0), (ia0, ib0) = pair(row, "Base")
        (a1, b1), (ia1, ib1) = pair(row, "ABTT")
        out = [f"{label} & {a0} & {a1} & {b0} & {b1} \\\\"]
        if intervals:  # the fine-tuned rows print none; Table tab:headline_ci has them
            out.append(r" & {\scriptsize " + ia0 + r"} & {\scriptsize " + ia1
                       + r"} & {\scriptsize " + ib0 + r"} & {\scriptsize " + ib1 + r"} \\")
        return out

    x = headline_numbers(ci)
    lines = [HEADER.rstrip("\n"), r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"& \multicolumn{2}{c}{Ranking AUROC} & \multicolumn{2}{c}{Routing DirAcc@1} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Model & Base & ABTT & Base & ABTT \\", r"\midrule"]
    for m in PANEL:
        lines += body(m, m)
    lines.append(r"\addlinespace")
    lines.append(
        r"\emph{Spread} & " + f"{x['auroc_Base_spread']:.3f} & "
        f"{x['auroc_ABTT_spread']:.3f} & {x['dir1_Base_spread']:.1f} & "
        f"{x['dir1_ABTT_spread']:.1f} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l}{\emph{Fine-tuned; Base = fine-tuned, ABTT = fine-tuned "
                 r"+ ABTT}} \\")
    for row, label in FT_ROWS:
        lines += body(label, row, intervals=False)
    lines.append(r"\midrule")
    ca = _cell(ci, "A", LEX_ROW[0], "ref", "auroc")
    cb = _cell(ci, "B", LEX_ROW[0], "ref", "dir1")
    lines.append(LEX_ROW[1] + r" & \multicolumn{2}{c}{" + _num(ca["estimate"], "auroc")
                 + r"} & \multicolumn{2}{c}{" + _num(cb["estimate"], "dir1") + r"} \\")
    lines.append(r" & \multicolumn{2}{c}{\scriptsize " + _int(ca["ci_lo"], ca["ci_hi"], "auroc")
                 + r"} & \multicolumn{2}{c}{\scriptsize " + _int(cb["ci_lo"], cb["ci_hi"], "dir1")
                 + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Ranking (test AUROC) and routing (test DirAcc@1, \%) without "
              r"correction (Base) and after ABTT, each at its train-selected layer "
              r"(Table~\ref{tab:selected_layers}); ABTT is fit on training embeddings only. "
              r"Fine-tuned: contrastive fine-tuning on 499 training pairs (intervals in "
              r"Table~\ref{tab:headline_ci}). Char.\ n-grams: TF-IDF over character "
              r"3--5-grams. Brackets: 95\% bootstrap intervals over the "
              + f"{int(info['n_test_dirs'])}" + r" test directories. Spread: largest minus "
              r"smallest printed value across the six encoders.}",
              r"\label{tab:headline}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- driver
def render_all(root: Path, tab_dir: Optional[Path] = None) -> Dict[str, str]:
    """Render the four tables from the files under ``root``; write them when ``tab_dir``."""
    root = Path(root)
    summ = summarize(per_layer(root))
    ci = pd.read_csv(root / CI_DIR / "headline_ci.csv")
    info = json.loads((root / CI_DIR / "run_info.json").read_text())
    x = headline_numbers(ci)
    frozen = [m for m in PANEL]
    for m in frozen:  # the caption's claim: no frozen cell above the reference
        for task, metric, key in (("A", "auroc", "lex_auroc"), ("B", "dir1", "lex_dir1")):
            for setting in ("Base", "ABTT"):
                v = float(_num(_cell(ci, task, m, setting, metric)["estimate"], metric))
                if v > x[key]:
                    raise SystemExit(f"{m} {setting} {metric} {v} exceeds the n-gram reference")
    out = {
        "models_main.tex": render_models_main(summ),
        "all_models.tex": render_all_models(summ),
        "predictions.tex": render_predictions(prediction_numbers(root)),
        "headline_main.tex": render_headline(ci, info),
    }
    if tab_dir is not None:
        tab_dir = Path(tab_dir)
        tab_dir.mkdir(parents=True, exist_ok=True)
        for name, text in out.items():
            (tab_dir / name).write_text(text)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root (default: cwd)")
    ap.add_argument("--tab_dir", type=Path, default=TAB_DIR)
    args = ap.parse_args(argv)
    out = render_all(args.root, args.root / args.tab_dir)
    for name in out:
        print(f"rendered {args.tab_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
