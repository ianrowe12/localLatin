#!/usr/bin/env python3
"""P2x2 (issue #248): the architecture-by-objective panel, per layer.

For each of ten 12-layer encoders, on the UNMODIFIED mean-pooled vectors of the 1,705-file
Latin corpus, per layer 1 to 12:
  aucroc    Task A test AUROC (``raw_auroc_layers.task_a_auroc``, the paper's metric block)
  pc1       top-PC share: first eigenvalue share of the centered (not L2-normalized)
            covariance of the 847 training passages
  erank     effective rank: exp(entropy) of the same eigenvalue shares
  mean_cos  mean pairwise cosine of the same rows
The geometry is ``gen_ft_geometry.layer_stats``, which wraps ``pca_stats`` / ``cosine_stats``
of scripts/resubmit/run_layer_geometry_diagnostics.py unchanged. Rows are aligned to the split
by filename (``AlignmentResolver`` reads the ``meta.csv`` beside each cache), never by position.

The 2x2: T5 encoder against encoder-only model, crossed with raw pre-training against an
embedding objective. LaTa, PhilTa, mT5-base and LaBSE come from the paper's cache
(``--bases_root``); the other six were extracted for this issue (``--p2x2_bases``).

Stages:
  compute  reads embeddings (CPU), runs the reproduction gate, writes under --out_dir:
             p2x2_layers.csv   model, model_id, cell, emb_objective, source, layer, aucroc,
                               n_train, pc1, erank, pc10, mean_cos
             p2x2_repro.csv    every gated cell: value, reference, difference
           The gate is a hard failure (exit 1). When it fails the rows go to
           p2x2_layers.rejected.csv instead, so render never reads ungated numbers.
             a. AUROC of LaTa, PhilTa, mT5-base, LaBSE = published baseline cells (1e-6)
             b. their train top-PC share and effective rank = geometry_per_layer.csv
             c. T5-v1.1-base AUROC = the committed #244 CPU extraction (1e-4)
             d. the cells the paper already prints (PRINTED, PRINTED_RANGES)
           A model left out by --models is reported as SKIPPED, not as a failure.
  render   reads those CSVs and the tracked fine-tuned LaTa table only, writes
             overleaf_drafts/tables/panel_2x2.tex       (label tab:panel_2x2)
             overleaf_drafts/tables/p2x2_layerwise.tex  (label tab:p2x2_layerwise)
             <out_dir>/p2x2_facts.md

Run from the repo root; the caches are gitignored, so point the two roots at a checkout that
has them:
  python scripts/paper/reframe/p2x2_panel.py --stage compute \
      --bases_root <root>/runs/active/resubmit_bases \
      --p2x2_bases <root>/runs/active/reframe/p2x2/bases
  python scripts/paper/reframe/p2x2_panel.py --stage render
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_ft_geometry import COLLAPSE_PC1, HEADER, _ranges, layer_stats  # noqa: E402

SPLIT_CSV = Path("runs/active/resubmit/data/phase_resubmit_split.csv")
RES_CSV = Path("runs/active/resubmit/results/phase_resubmit_results.csv")
GEO_CSV = Path("runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv")
T5V11_CSV = Path("runs/active/reframe/p2x2/raw_auroc_layers.csv")
BASES_ROOT = Path("runs/active/resubmit_bases")
P2X2_BASES = Path("runs/active/reframe/p2x2/bases")
OUT_DIR = Path("runs/active/reframe/p2x2")
TAB_DIR = Path("overleaf_drafts/tables")
FT_TEX = TAB_DIR / "ft_lata_layerwise.tex"
SUBDIR = "hidden_mean_tokempty"
LAYERS = tuple(range(1, 13))
AUROC_FLOOR = 0.80  # the paper's "stays above 0.80 at every layer" reading

# display, HF id, cell, embedding objective, source ("panel": the paper's cache under
# --bases_root/phase9_bases; "p2x2": this issue's extraction under --p2x2_bases).
MODELS = [
    ("LaTa", "bowphs/LaTa", "T5, raw", "none", "panel"),
    ("PhilTa", "bowphs/PhilTa", "T5, raw", "none", "panel"),
    ("mT5-base", "google/mt5-base", "T5, raw", "none", "panel"),
    ("T5-v1.1-base", "google/t5-v1_1-base", "T5, raw", "none", "p2x2"),
    ("T5-base", "google-t5/t5-base", "T5, raw", "none", "p2x2"),
    ("Sentence-T5", "sentence-transformers/sentence-t5-base", "T5, emb.", "contrastive", "p2x2"),
    ("LaBERTa", "bowphs/LaBerta", "Enc., raw", "none", "p2x2"),
    ("PhilBERTa", "bowphs/PhilBerta", "Enc., raw", "none", "p2x2"),
    ("LaBSE", "sentence-transformers/LaBSE", "Enc., emb.", "contrastive", "panel"),
    ("SPhilBERTa", "bowphs/SPhilBerta", "Enc., emb.", "distillation", "p2x2"),
]

# Reproduction gate.
PUBLISHED_TOL = 1e-6   # a: AUROC against phase_resubmit_results.csv
GEOMETRY_TOL = 1e-6    # b: top-PC share (absolute) and effective rank (relative)
T5V11_TOL = 1e-4       # c: a re-extraction on other hardware against the #244 CPU extraction
T5V11 = "T5-v1.1-base"
# d: cells the paper already prints, to three decimals. A layer is checked where the paper
# (or tab:gen_geometry, for the T5-v1.1-base peak) names it.
PRINTED: Dict[str, Dict[str, float]] = {
    "LaTa": {"auroc_min": 0.496, "auroc_min_layer": 6},
    "PhilTa": {"auroc_min": 0.538, "auroc_min_layer": 10},
    "mT5-base": {"auroc_min": 0.654, "auroc_min_layer": 5},
    "LaBSE": {"auroc_min": 0.806, "pc1_max": 0.524},
    T5V11: {"auroc_min": 0.489, "auroc_min_layer": 2, "pc1_max": 0.975, "pc1_max_layer": 9},
}
RAW_T5_PANEL = ("LaTa", "PhilTa", "mT5-base")
PRINTED_RANGES = {RAW_T5_PANEL: {"auroc_min": "0.50--0.65", "pc1_max": "0.86--1.00"}}

# tab:panel_2x2 rows: (cell, models sharing the row). FT_ROW is fine-tuned LaTa, whose vectors
# are not in the P2x2 caches: its cells are read from the tracked ft_lata_layerwise.tex, a
# generated table of issue #238 (gen_ft_geometry.py), and must equal what Sec. 6 prints.
FT_ROW = "LaTa (fine-tuned)"
MIDRULE = ("midrule", ())
PANEL_ROWS: List[Tuple[str, Tuple[str, ...]]] = [
    ("T5, raw", RAW_T5_PANEL),
    ("T5, raw", (T5V11,)),
    ("T5, raw", ("T5-base",)),
    ("T5, raw+FT", (FT_ROW,)),
    ("T5, emb.", ("Sentence-T5",)),
    MIDRULE,
    ("Enc., raw", ("LaBERTa", "PhilBERTa")),
    ("Enc., emb.", ("LaBSE", "SPhilBERTa")),
]
FT_LAYERS = tuple(range(2, 12))  # the fine-tuned LaTa AUROC range is over layers 2 to 11
FT_PRINTED = {"auroc": "0.50--0.57", "pc1_max": "0.945"}
PENDING = r"\pendingnum{P2x2}"  # a cell whose model is not in the CSV (partial runs)

# tab:p2x2_layerwise: the models no other per-layer table of the paper covers.
LAYERWISE_MODELS = (T5V11, "T5-base", "Sentence-T5", "LaBERTa", "PhilBERTa", "SPhilBERTa")


# --------------------------------------------------------------------------- compute
def run_dir(source: str, model_id: str, bases_root: Path, p2x2_bases: Path) -> Path:
    slug = model_id.replace("/", "_")
    if source == "panel":
        return Path(bases_root) / "phase9_bases" / slug / SUBDIR
    return Path(p2x2_bases) / slug / SUBDIR


def score_models(split: pd.DataFrame, bases_root: Path, p2x2_bases: Path,
                 names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per model-layer: test AUROC on all rows, geometry on the train rows."""
    for p in (REPO / "src",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from embedding_alignment import AlignmentResolver
    from raw_auroc_layers import task_a_auroc

    known = [m[0] for m in MODELS]
    unknown = [n for n in (names or []) if n not in known]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {known}")
    chosen = [m for m in MODELS if not names or m[0] in names]
    missing = []
    for name, model_id, _, _, source in chosen:
        d = run_dir(source, model_id, bases_root, p2x2_bases)
        missing += [str(d / f"hidden_layer{layer}_embeddings.npy") for layer in LAYERS
                    if not (d / f"hidden_layer{layer}_embeddings.npy").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} embedding file(s) missing, e.g. {missing[:3]}. "
                         "Extract them first or restrict the run with --models.")

    train = split["split"].to_numpy() == "train"
    resolver = AlignmentResolver(split)
    rows: List[Dict] = []
    for name, model_id, cell, objective, source in chosen:
        d = run_dir(source, model_id, bases_root, p2x2_bases)
        for layer in LAYERS:
            emb = resolver.load(d / f"hidden_layer{layer}_embeddings.npy")
            s = layer_stats(emb[train])
            rows.append({"model": name, "model_id": model_id, "cell": cell,
                         "emb_objective": objective, "source": source, "layer": layer,
                         "aucroc": task_a_auroc(emb, split), "n_train": s["n"], "pc1": s["pc1"],
                         "erank": s["erank"], "pc10": s["pc10"], "mean_cos": s["mean_cos"]})
            print(f"{name:13s} L{layer:<2d} AUROC {rows[-1]['aucroc']:.4f}  PC1 {s['pc1']:.4f}  "
                  f"rank {s['erank']:.2f}", flush=True)
    print(resolver.summary())
    return pd.DataFrame(rows)


def _compare(gate: str, metric: str, got: pd.Series, ref: pd.Series, tol: float,
             relative: bool, records: List[Dict], problems: List[str]) -> Optional[float]:
    """Compare two (model, layer)-indexed series; append records and problems; return the
    largest difference (relative when ``relative``), or None when ``got`` is empty."""
    worst = None
    for (model, layer), value in got.items():
        if (model, layer) not in ref.index:
            problems.append(f"gate {gate}: {model} L{layer} {metric} has no reference cell")
            continue
        reference = float(ref.loc[(model, layer)])
        diff = abs(value - reference)
        shown = diff / abs(reference) if relative else diff
        ok = bool(shown <= tol)
        records.append({"gate": gate, "model": model, "layer": layer, "metric": metric,
                        "value": value, "reference": reference, "abs_diff": diff,
                        "rel_diff": diff / abs(reference) if reference else float("nan"),
                        "tol": tol, "tol_kind": "relative" if relative else "absolute", "ok": ok})
        worst = shown if worst is None else max(worst, shown)
        if not ok:
            kind = "relative" if relative else "absolute"
            problems.append(f"gate {gate}: {model} L{layer} {metric} {value:.8f}, reference "
                            f"{reference:.8f} ({kind} difference {shown:.2e} > {tol:.0e})")
    return worst


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model. Minima and maxima take the first layer on ties."""
    order = {m[0]: i for i, m in enumerate(MODELS)}
    names = sorted(df["model"].unique(), key=lambda n: (order.get(n, len(order)), n))
    out = []
    for name in names:
        s = df[df["model"] == name].sort_values("layer").reset_index(drop=True)
        lo, hi = s.loc[s["aucroc"].idxmin()], s.loc[s["aucroc"].idxmax()]
        pk, rk = s.loc[s["pc1"].idxmax()], s.loc[s["erank"].idxmin()]
        by_layer = s.set_index("layer")
        high = s.loc[s["pc1"] >= COLLAPSE_PC1, "layer"].tolist()
        low = s.loc[s["aucroc"] < AUROC_FLOOR, "layer"].tolist()
        out.append({
            "model": name, "model_id": s["model_id"].iloc[0], "cell": s["cell"].iloc[0],
            "emb_objective": s["emb_objective"].iloc[0], "source": s["source"].iloc[0],
            "n_layers": len(s), "n_train": int(s["n_train"].iloc[0]),
            "auroc_min": float(lo["aucroc"]), "auroc_min_layer": int(lo["layer"]),
            "auroc_max": float(hi["aucroc"]), "auroc_max_layer": int(hi["layer"]),
            "auroc_first": float(s["aucroc"].iloc[0]), "auroc_last": float(s["aucroc"].iloc[-1]),
            "first_layer": int(s["layer"].iloc[0]), "last_layer": int(s["layer"].iloc[-1]),
            "pc1_max": float(pk["pc1"]), "pc1_max_layer": int(pk["layer"]),
            "erank_at_pc1_max": float(pk["erank"]),
            "erank_min": float(rk["erank"]), "erank_min_layer": int(rk["layer"]),
            "auroc_at_pc1_max": float(by_layer.loc[int(pk["layer"]), "aucroc"]),
            "n_high_pc1": len(high), "high_pc1_layers": _ranges(high),
            "n_low_auroc": len(low), "low_auroc_layers": _ranges(low),
        })
    return pd.DataFrame(out)


def reproduction_gate(df: pd.DataFrame, res_csv: Optional[Path], geo_csv: Optional[Path],
                      t5v11_csv: Optional[Path]) -> Tuple[List[str], List[Dict], List[str]]:
    """Returns (problems, per-cell records, one status line per gate)."""
    problems: List[str] = []
    records: List[Dict] = []
    lines: List[str] = []
    panel = df[df["source"] == "panel"]
    by_id = panel.set_index(["model_id", "layer"])
    disp = dict(zip(panel["model_id"], panel["model"]))

    def relabel(series: pd.Series) -> pd.Series:
        """Index a (model id, layer) series by (display name, layer)."""
        series.index = pd.MultiIndex.from_tuples(
            [(disp.get(model_id, model_id), int(layer)) for model_id, layer in series.index])
        return series

    def status(gate: str, what: str, n0: int, worst: Optional[float], tol: float, kind: str) -> None:
        bad = len(problems) - n0
        shown = "n/a (no cell compared)" if worst is None else f"{worst:.2e}"
        lines.append(f"gate {gate} ({what}): max {kind} difference {shown}, tolerance "
                     f"{tol:.0e}: {'FAIL, ' + str(bad) + ' problem(s)' if bad else 'PASS'}")

    # a. published baseline AUROC
    if panel.empty:
        lines.append("gate a (published AUROC): SKIPPED (no panel model computed)")
    elif res_csv is None or not Path(res_csv).exists():
        problems.append(f"gate a: published results CSV not found ({res_csv})")
    else:
        pub = pd.read_csv(res_csv)
        pub = pub[(pub["repr"] == "hidden") & (pub["pooling"] == "mean")
                  & (pub["method"] == "baseline")].set_index(["model", "layer"])["aucroc"]
        n0 = len(problems)
        worst = _compare("a", "aucroc", relabel(by_id["aucroc"].copy()), relabel(pub.copy()),
                         PUBLISHED_TOL, False, records, problems)
        status("a", f"published AUROC, {panel['model'].nunique()} models", n0, worst,
               PUBLISHED_TOL, "absolute")

    # b. published train geometry
    if panel.empty:
        lines.append("gate b (published geometry): SKIPPED (no panel model computed)")
    elif geo_csv is None or not Path(geo_csv).exists():
        problems.append(f"gate b: published geometry CSV not found ({geo_csv})")
    else:
        geo = pd.read_csv(geo_csv)
        geo = geo[(geo["split"] == "train") & (geo["view"] == "raw") & (geo["repr"] == "hidden")
                  & (geo["pooling"] == "mean")].set_index(["model", "layer"])
        for metric, col, relative in [("pc1", "pc1_variance_ratio", False),
                                      ("erank", "effective_rank_entropy", True)]:
            n0 = len(problems)
            worst = _compare("b", metric, relabel(by_id[metric].copy()), relabel(geo[col].copy()),
                             GEOMETRY_TOL, relative, records, problems)
            status("b", f"published train {metric}, {panel['model'].nunique()} models", n0, worst,
                   GEOMETRY_TOL, "relative" if relative else "absolute")

    # c. T5-v1.1-base against the committed CPU extraction (#244)
    new = df[df["model"] == T5V11].set_index(["model", "layer"])["aucroc"]
    if new.empty:
        lines.append(f"gate c ({T5V11} AUROC): SKIPPED ({T5V11} not computed)")
    elif t5v11_csv is None or not Path(t5v11_csv).exists():
        problems.append(f"gate c: committed {T5V11} AUROC CSV not found ({t5v11_csv})")
    else:
        old = pd.read_csv(t5v11_csv)
        old = old[(old["model"] == T5V11) & (old["source"] == "gen")]
        n0 = len(problems)
        worst = _compare("c", "aucroc", new, old.set_index(["model", "layer"])["aucroc"],
                         T5V11_TOL, False, records, problems)
        status("c", f"{T5V11} AUROC against the #244 extraction", n0, worst, T5V11_TOL, "absolute")

    # d. cells the paper already prints
    summ = summarize(df).set_index("model") if len(df) else pd.DataFrame()
    n0, checked, skipped = len(problems), 0, []
    for name, cells in PRINTED.items():
        if name not in summ.index:
            skipped.append(name)
            continue
        for key, want in cells.items():
            got = summ.loc[name, key]
            checked += 1
            if key.endswith("_layer"):
                if int(got) != int(want):
                    problems.append(f"gate d: {name} {key} printed {int(want)}, got {int(got)}")
            elif f"{got:.3f}" != f"{want:.3f}":
                problems.append(f"gate d: {name} {key} printed {want:.3f}, got {got:.4f}")
    for group, cells in PRINTED_RANGES.items():
        if not all(g in summ.index for g in group):
            skipped.append("range of " + ", ".join(group))
            continue
        for key, want in cells.items():
            got = fmt_cell([summ.loc[g, key] for g in group])
            checked += 1
            if got != want:
                problems.append(f"gate d: {', '.join(group)} {key} printed {want}, got {got}")
    bad = len(problems) - n0
    lines.append(f"gate d (printed cells): {checked} checked: "
                 f"{'FAIL, ' + str(bad) + ' problem(s)' if bad else 'PASS'}"
                 + (f"; SKIPPED (not computed): {'; '.join(skipped)}" if skipped else ""))
    return problems, records, lines


def compute(args) -> int:
    split = pd.read_csv(args.split_csv)
    df = score_models(split, Path(args.bases_root), Path(args.p2x2_bases), args.models)
    problems, records, lines = reproduction_gate(df, args.res_csv, args.geo_csv, args.t5v11_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_dir / "p2x2_repro.csv", index=False, float_format="%.10g")
    out = out_dir / ("p2x2_layers.rejected.csv" if problems else "p2x2_layers.csv")
    df.to_csv(out, index=False, float_format="%.10g")
    print(f"wrote {out} ({len(df)} rows) and {out_dir / 'p2x2_repro.csv'}")
    for x in summarize(df).itertuples():
        print(f"{x.model:13s} AUROC min {x.auroc_min:.4f} (layer {x.auroc_min_layer})  "
              f"PC1 max {x.pc1_max:.4f} (layer {x.pc1_max_layer})  "
              f"rank min {x.erank_min:.2f} (layer {x.erank_min_layer})")
    for line in lines:
        print(line)
    for p in problems:
        print("REPRODUCTION MISMATCH:", p)
    if problems:
        print(f"reproduction gate FAILED with {len(problems)} problem(s); "
              f"p2x2_layers.csv was not written")
        return 1
    print("reproduction gate passed")
    return 0


# --------------------------------------------------------------------------- render
def fmt_cell(values: Sequence[float], single_nd: int = 3, range_nd: int = 2) -> str:
    """One value: ``single_nd`` decimals. Several: the range ``lo--hi`` at ``range_nd``
    decimals, or at one more decimal when the two ends would print alike."""
    values = [float(v) for v in values]
    if len(values) == 1:
        return f"{values[0]:.{single_nd}f}"
    lo, hi = min(values), max(values)
    for nd in (range_nd, range_nd + 1):
        a, b = f"{lo:.{nd}f}", f"{hi:.{nd}f}"
        if a != b:
            return f"{a}--{b}"
    return a


FT_LINE = re.compile(r"^\s*(\d+)((?:\s*&\s*[0-9.]+){6})\s*\\\\\s*$")


def read_ft_cells(path: Path) -> Dict:
    """Fine-tuned LaTa cells from the generated tab:ft_lata_layerwise (columns: layer, AUROC
    PT, AUROC FT, top-PC share PT, FT, effective rank PT, FT)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found: the fine-tuned LaTa row of tab:panel_2x2 is read "
                         "from that generated table (pass --ft_tex)")
    rows = {}
    for line in path.read_text().splitlines():
        m = FT_LINE.match(line)
        if m:
            rows[int(m.group(1))] = [float(v) for v in m.group(2).replace("&", " ").split()]
    absent = [layer for layer in LAYERS if layer not in rows]
    if absent:
        raise SystemExit(f"{path}: no table row for layer(s) {absent}")
    auroc = {layer: rows[layer][1] for layer in LAYERS}
    pc1 = {layer: rows[layer][3] for layer in LAYERS}
    erank = {layer: rows[layer][5] for layer in LAYERS}
    mid = [auroc[layer] for layer in FT_LAYERS]
    pk = max(LAYERS, key=lambda layer: (pc1[layer], -layer))
    rk = min(LAYERS, key=lambda layer: (erank[layer], layer))
    cells = {"auroc_lo": min(mid), "auroc_hi": max(mid), "auroc": fmt_cell([min(mid), max(mid)]),
             "pc1_max": pc1[pk], "pc1_max_layer": pk, "erank_min": erank[rk],
             "erank_min_layer": rk, "source": str(path)}
    got = {"auroc": cells["auroc"], "pc1_max": f"{cells['pc1_max']:.3f}"}
    if got != FT_PRINTED:
        raise SystemExit(f"{path}: fine-tuned LaTa cells {got} differ from the cells the paper "
                         f"prints {FT_PRINTED}")
    return cells


def _models_tex(names: Sequence[str]) -> str:
    """Two names on one line; three or more break after the second (makecell)."""
    if len(names) <= 2:
        return ", ".join(names)
    return r"\makecell[l]{" + ", ".join(names[:2]) + r",\\" + ", ".join(names[2:]) + "}"


def panel_rows(summ: pd.DataFrame, ft: Dict) -> List[Optional[Dict]]:
    """The rows of tab:panel_2x2 (None marks the rule between the two architectures)."""
    s = summ.set_index("model") if len(summ) else summ
    out: List[Optional[Dict]] = []
    for cell, names in PANEL_ROWS:
        if (cell, names) == MIDRULE:
            out.append(None)
        elif names == (FT_ROW,):
            out.append({"cell": cell, "models": FT_ROW, "auroc": ft["auroc"],
                        "pc1": f"{ft['pc1_max']:.3f}", "erank": f"{ft['erank_min']:.2f}"})
        elif all(n in s.index for n in names):
            out.append({"cell": cell, "models": _models_tex(names),
                        "auroc": fmt_cell([s.loc[n, "auroc_min"] for n in names]),
                        "pc1": fmt_cell([s.loc[n, "pc1_max"] for n in names]),
                        "erank": fmt_cell([s.loc[n, "erank_min"] for n in names], 2, 2)})
        else:
            out.append({"cell": cell, "models": _models_tex(names), "auroc": PENDING,
                        "pc1": PENDING, "erank": PENDING})
    return out


def write_panel_table(summ: pd.DataFrame, ft: Dict, path: Path) -> None:
    n = f"{int(summ['n_train'].iloc[0]):,}".replace(",", "{,}") if len(summ) else "847"
    lines = [HEADER, "% python scripts/paper/reframe/p2x2_panel.py --stage render",
             r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"Cell & Models & AUROC$_{\min}$ & PC1$_{\max}$ \\", r"\midrule"]
    # No effective-rank column: with it the table overflows one ACL column by about 50pt.
    # Effective rank is in tab:p2x2_layerwise and p2x2_facts.md.
    for row in panel_rows(summ, ft):
        if row is None:
            lines.append(r"\midrule")
        else:
            lines.append(f"{row['cell']} & {row['models']} & {row['auroc']} & {row['pc1']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Architecture-by-objective panel, read on the Latin corpus with no "
              r"post-hoc correction. AUROC$_{\min}$ is the lowest Task~A test AUROC over the 12 "
              r"layers of the unmodified mean-pooled vectors. PC1$_{\max}$ is the peak top-PC "
              r"share over layers, the largest share of centered variance on the first principal "
              r"component of the " + n + r" training passages. A row with one model gives its "
              r"value, to three decimals. A row with several "
              r"models gives the range of the per-model values, to two decimals (three if the "
              r"two ends would print alike). For fine-tuned "
              r"LaTa, the AUROC entry is the range over layers 2--11. Enc.\ is encoder-only, "
              r"emb.\ is embedding-trained, and FT is contrastive fine-tuning on training pairs. "
              r"T5-base is the raw partner of Sentence-T5, which starts from the original T5 "
              r"checkpoint.}",
              r"\label{tab:panel_2x2}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_layerwise_table(df: pd.DataFrame, path: Path) -> bool:
    """Per-layer AUROC, top-PC share and effective rank of LAYERWISE_MODELS present in df."""
    names = [n for n in LAYERWISE_MODELS if n in set(df["model"])]
    if not names:
        return False
    by = df.set_index(["model", "layer"]).sort_index()
    layers = sorted(df.loc[df["model"].isin(names), "layer"].unique())
    n = f"{int(df.loc[df['model'].isin(names), 'n_train'].iloc[0]):,}".replace(",", "{,}")
    head = " & ".join(r"\multicolumn{3}{c}{" + name + "}" for name in names)
    rules = "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(names)))
    lines = [HEADER, "% python scripts/paper/reframe/p2x2_panel.py --stage render",
             r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{2pt}",  # 3pt overflows the ACL text width by about 20pt
             r"\begin{tabular}{@{}r" + "ccc" * len(names) + r"@{}}", r"\toprule",
             "& " + head + r" \\", rules,
             "Layer & " + " & ".join(["AUROC & PC1 & Rank"] * len(names)) + r" \\", r"\midrule"]
    for layer in layers:
        cells = []
        for name in names:
            x = by.loc[(name, layer)]
            cells.append(f"{x['aucroc']:.3f} & {x['pc1']:.3f} & {x['erank']:.2f}")
        lines.append(f"{layer} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Per-layer readouts of the models added for the architecture-by-objective "
              r"panel (Table~\ref{tab:panel_2x2}), on the Latin corpus with no post-hoc "
              r"correction. AUROC: Task~A test pairwise AUROC of cosine on mean-pooled vectors. "
              r"PC1: top-PC share, the share of centered variance on the first principal "
              r"component of the " + n + r" training passages. Rank: entropy effective rank of "
              r"the same passages. T5-v1.1-base, T5-base, LaBERTa and PhilBERTa are raw "
              r"pre-trained encoders; Sentence-T5 and SPhilBERTa are embedding-trained.}",
              r"\label{tab:p2x2_layerwise}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")
    return True


def facts(df: pd.DataFrame, summ: pd.DataFrame, ft: Dict, rep: Optional[pd.DataFrame],
          path: Path) -> None:
    L = ["# P2x2 numbers (generated)", "",
         "Generated by `scripts/paper/reframe/p2x2_panel.py --stage render` from "
         "`p2x2_layers.csv`. Unmodified mean-pooled vectors, Latin corpus. AUROC: Task A test "
         "AUROC. Top-PC share (PC1), effective rank and mean pairwise cosine: the training "
         "passages. Minima and maxima take the first layer on ties.", ""]
    w = L.append
    absent = [m[0] for m in MODELS if m[0] not in set(summ["model"])]
    if absent:
        w(f"**Partial run: no rows for {', '.join(absent)}.**")
        w("")
    w("## Per model")
    w("| model | HF id | cell | emb. objective | source | layers | n train | AUROC min (layer) | "
      "AUROC max (layer) | AUROC first layer | AUROC last layer | PC1 max (layer) | "
      "eff rank at PC1 max | AUROC at PC1 max | eff rank min (layer) | "
      f"layers PC1 >= {COLLAPSE_PC1} | which | layers AUROC < {AUROC_FLOOR:.2f} | which |")
    w("|" + "---|" * 19)
    for x in summ.itertuples():
        w(f"| {x.model} | `{x.model_id}` | {x.cell} | {x.emb_objective} | {x.source} | "
          f"{x.first_layer}-{x.last_layer} | {x.n_train} | "
          f"{x.auroc_min:.3f} ({x.auroc_min_layer}) | {x.auroc_max:.3f} ({x.auroc_max_layer}) | "
          f"{x.auroc_first:.3f} | {x.auroc_last:.3f} | {x.pc1_max:.3f} ({x.pc1_max_layer}) | "
          f"{x.erank_at_pc1_max:.2f} | {x.auroc_at_pc1_max:.3f} | "
          f"{x.erank_min:.2f} ({x.erank_min_layer}) | {x.n_high_pc1} | "
          f"{x.high_pc1_layers.replace('--', '-')} | {x.n_low_auroc} | "
          f"{x.low_auroc_layers.replace('--', '-')} |")
    w("")
    w("## Table rows (tab:panel_2x2)")
    w("| cell | models | AUROC min | PC1 max | eff rank min |")
    w("|---|---|---|---|---|")
    for (cell, names), row in zip(PANEL_ROWS, panel_rows(summ, ft)):
        if row is None:
            continue
        shown = [("not computed" if row[k] == PENDING else row[k].replace("--", " to "))
                 for k in ("auroc", "pc1", "erank")]
        w(f"| {cell} | {', '.join(names)} | {shown[0]} | {shown[1]} | {shown[2]} |")
    w("")
    w(f"- {FT_ROW}: read from `{ft['source']}` (generated by gen_ft_geometry.py, issue #238), "
      f"not recomputed here. AUROC over layers {FT_LAYERS[0]}-{FT_LAYERS[-1]}: "
      f"{ft['auroc_lo']:.3f} to {ft['auroc_hi']:.3f}; peak top-PC share {ft['pc1_max']:.3f} "
      f"(layer {ft['pc1_max_layer']}); lowest effective rank {ft['erank_min']:.2f} "
      f"(layer {ft['erank_min_layer']}).")
    w("- Range rule: a row with several models gives the lowest and highest per-model value at "
      "two decimals (three when the two ends agree at two).")
    w("")
    w("## Per layer: AUROC / top-PC share / effective rank / mean pairwise cosine")
    for x in summ.itertuples():
        s = df[df["model"] == x.model].sort_values("layer")
        w(f"- {x.model}: " + "; ".join(
            f"{int(r.layer)}: {r.aucroc:.3f} / {r.pc1:.3f} / {r.erank:.2f} / {r.mean_cos:.3f}"
            for r in s.itertuples()))
    w("")
    if rep is not None and len(rep):
        w("## Reproduction gate (p2x2_repro.csv)")
        w("| gate | metric | models | cells | max abs difference | max relative difference | "
          "tolerance | all within |")
        w("|---|---|---|---|---|---|---|---|")
        for (gate, metric), g in rep.groupby(["gate", "metric"], sort=True):
            w(f"| {gate} | {metric} | {', '.join(dict.fromkeys(g['model']))} | {len(g)} | "
              f"{g['abs_diff'].max():.2e} | {g['rel_diff'].max():.2e} | "
              f"{g['tol'].iloc[0]:.0e} ({g['tol_kind'].iloc[0]}) | {bool(g['ok'].all())} |")
        w("")
        w("Gate a: AUROC against `phase_resubmit_results.csv` (hidden, mean, baseline). Gate b: "
          "train top-PC share and effective rank against `geometry_per_layer.csv` (train, raw). "
          "Gate c: T5-v1.1-base AUROC against the committed `raw_auroc_layers.csv` (source gen, "
          "a CPU extraction by gen_extract.py). Gate d, the printed cells, is checked at compute "
          "time and has no row here.")
        w("")
    path.write_text("\n".join(L) + "\n")


def render(args) -> int:
    out_dir, tab_dir = Path(args.out_dir), Path(args.tab_dir)
    df = pd.read_csv(out_dir / "p2x2_layers.csv")
    rep_p = out_dir / "p2x2_repro.csv"
    rep = None
    if rep_p.exists() and rep_p.stat().st_size > 1:
        try:
            rep = pd.read_csv(rep_p)
        except pd.errors.EmptyDataError:
            rep = None
    ft = read_ft_cells(args.ft_tex)
    summ = summarize(df)
    tab_dir.mkdir(parents=True, exist_ok=True)
    write_panel_table(summ, ft, tab_dir / "panel_2x2.tex")
    wrote = write_layerwise_table(df, tab_dir / "p2x2_layerwise.tex")
    facts(df, summ, ft, rep, out_dir / "p2x2_facts.md")
    absent = [m[0] for m in MODELS if m[0] not in set(df["model"])]
    if absent:
        print(f"WARNING: partial CSV, no rows for {', '.join(absent)}; their table cells "
              f"print {PENDING}")
    print(f"rendered {tab_dir / 'panel_2x2.tex'}"
          + (f", {tab_dir / 'p2x2_layerwise.tex'}" if wrote else " (no layerwise table: none of "
             "its models is in the CSV)") + f" and {out_dir / 'p2x2_facts.md'}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["compute", "render", "all"], default="all")
    ap.add_argument("--split_csv", type=Path, default=SPLIT_CSV)
    ap.add_argument("--res_csv", type=Path, default=RES_CSV)
    ap.add_argument("--geo_csv", type=Path, default=GEO_CSV)
    ap.add_argument("--t5v11_csv", type=Path, default=T5V11_CSV)
    ap.add_argument("--bases_root", type=Path, default=BASES_ROOT,
                    help="holds phase9_bases/<slug>/hidden_mean_tokempty (the paper's cache)")
    ap.add_argument("--p2x2_bases", type=Path, default=P2X2_BASES,
                    help="holds <slug>/hidden_mean_tokempty (the issue #248 extraction)")
    ap.add_argument("--models", nargs="*", default=None, help="display names, default all ten")
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tab_dir", type=Path, default=TAB_DIR)
    ap.add_argument("--ft_tex", type=Path, default=FT_TEX)
    args = ap.parse_args(argv)
    if args.stage in ("compute", "all"):
        status = compute(args)
        if status:
            return status
    if args.stage in ("render", "all"):
        return render(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
