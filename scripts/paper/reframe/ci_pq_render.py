"""LaTeX tables for issue #233 (CI + PQ), rendered from the published CSVs.

Inputs (committed, ``docs/research/data/reframe_ci_pq/``): ``headline_ci.csv``,
``headline_ci_diffs.csv``, ``pq_cells.csv``, ``sweep_selected.csv``,
``run_info.json``; plus the committed headline tables
``overleaf_drafts/tables/task{A,B}_headline.tex``.

Outputs (``overleaf_drafts/tables/``):

``headline_ci.tex``           appendix table, every headline cell with its interval
``headline_ci_diffs.tex``     paired differences the prose relies on
``pq_routing.tex``            the five routing checks at the Task B cells
``taskA_headline_ci.tex``     compact drop-in for ``taskA_headline.tex``
``taskB_headline_ci.tex``     compact drop-in for ``taskB_headline.tex``

The compact tables copy the committed headline table line for line and insert
one interval row under every row that has cells, so their point estimates are
the printed ones. Rendering stops if a printed cell and the CSV estimate
disagree at printed precision, which is what a stale CSV would look like.

Everything here is a pure function of those files, so ``tests/test_reframe_ci_pq.py``
regenerates the tables and compares them byte for byte.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

HEADER = "% generated table\n"
SETTINGS = ["Base", "SIF", "ABTT", "SIF+ABTT"]
ZS_ROWS = ["LaTa", "PhilTa", "mT5-base", "LaBSE", "Qwen3-0.6B", "KaLM-mini"]
FT_ROWS = [f"{m} (fine-tuned)" for m in ("LaTa", "Qwen3-0.6B", "KaLM-mini")]
LEX_ROWS = [("BM25 (word)", "BM25 (word)"), ("TF-IDF char 3-5", "TF-IDF char 3--5"),
            ("Levenshtein", "Levenshtein")]
LEX_TEX = dict(LEX_ROWS)
LEX_CSV = {tex: csv for csv, tex in LEX_ROWS}
FMT = {"auroc": (".3f", 1.0), "gap": (".3f", 1.0), "assign": (".1f", 100.0),
       "dir1": (".1f", 100.0)}


def _num(x: float, metric: str, drop_zero: bool = False) -> str:
    fmt, scale = FMT[metric]
    s = format(scale * float(x), fmt)
    if drop_zero:
        s = re.sub(r"^(-?)0\.", r"\1.", s)
    return s.replace("-", "$-$")


def _ci(lo: float, hi: float, metric: str, drop_zero: bool = False, tight: bool = False) -> str:
    sep = "," if tight else ", "
    return f"[{_num(lo, metric, drop_zero)}{sep}{_num(hi, metric, drop_zero)}]"


def _signed(x: float, metric: str) -> str:
    fmt, scale = FMT[metric]
    v = scale * float(x)
    s = format(abs(v), fmt)
    if s.strip("0.") == "":
        return s
    return ("$+$" if v > 0 else "$-$") + s


def _load(in_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict,
                                 Optional[pd.DataFrame]]:
    ci = pd.read_csv(in_dir / "headline_ci.csv")
    diffs = pd.read_csv(in_dir / "headline_ci_diffs.csv")
    pq = pd.read_csv(in_dir / "pq_cells.csv")
    info = json.loads((in_dir / "run_info.json").read_text())
    sweep_path = in_dir / "sweep_selected.csv"
    sweep = pd.read_csv(sweep_path) if sweep_path.exists() else None
    return ci, diffs, pq, info, sweep


def _lookup(ci: pd.DataFrame, task: str, row: str, setting: str, metric: str
            ) -> Optional[pd.Series]:
    sub = ci[(ci["task"] == task) & (ci["row"] == row) & (ci["setting"] == setting)
             & (ci["metric"] == metric)]
    return sub.iloc[0] if len(sub) else None


def _B(info: Dict) -> str:
    return f"{int(info['B']):,}".replace(",", "{,}")


def _boot_clause(info: Dict, n_dirs: int) -> str:
    return (
        f"95\\% percentile intervals from a directory-level bootstrap "
        f"($B={_B(info)}$ resamples of the {n_dirs} test directories with "
        f"replacement, seed {info['seed']})"
    )


FIXED_CLAUSE = (
    "The threshold $\\tau$, the ABTT components and $D$, the SIF weights and the "
    "layer stay fit on the fixed training split; only the test evaluation is "
    "resampled, so the intervals cover test-set sampling and not the choice of "
    "split."
)


# --------------------------------------------------------------------------- #
# Appendix table: every cell
# --------------------------------------------------------------------------- #


def render_appendix(ci: pd.DataFrame, info: Dict) -> str:
    lines = [
        HEADER.rstrip("\n"),
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Task A}} & \multicolumn{2}{c}{\textbf{Task B}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"\textbf{Model} & \textbf{Setting} & AUROC & Cosine gap & Assignment acc. "
        r"& DirAcc@1 \\",
        r"\midrule",
    ]

    def row(label: str, key: str, setting: str, show_label: bool, set_label: str) -> str:
        cells = []
        for task, metric in (("A", "auroc"), ("A", "gap"), ("B", "assign"), ("B", "dir1")):
            r = _lookup(ci, task, key, setting, metric)
            cells.append("--" if r is None else
                         f"{_num(r['estimate'], metric)} {_ci(r['ci_lo'], r['ci_hi'], metric)}")
        return " & ".join([label if show_label else "", set_label] + cells) + r" \\"

    for m in ZS_ROWS:
        for k, s in enumerate(SETTINGS):
            lines.append(row(m, m, s, k == 0, s))
        lines.append(r"\addlinespace")
    lines[-1] = r"\midrule"
    for m in FT_ROWS:
        if _lookup(ci, "A", m, "Base", "auroc") is None:
            continue
        for k, s in enumerate(("Base", "ABTT")):
            lines.append(row(m, m, s, k == 0, s))
    lex = [(csv, tex) for csv, tex in LEX_ROWS if _lookup(ci, "A", csv, "ref", "auroc") is not None]
    if lex:
        lines.append(r"\midrule")
        for csv, tex in lex:
            lines.append(row(tex, csv, "ref", True, "--"))
    n_dirs = _n_dirs(info)
    caption = (
        "Every headline cell of Tables~\\ref{tab:taskA_headline} and~\\ref{tab:taskB_headline} "
        "with its " + _boot_clause(info, n_dirs) + ". A replicate keeps every file and pair of a drawn "
        "directory; a directory drawn twice counts twice, and pairs between its two copies are "
        "not formed. " + FIXED_CLAUSE + " Each cell is read at its train-selected layer "
        "(Table~\\ref{tab:selected_layers}); Task B in percent. The lexical rows are the "
        "reference systems of Table~\\ref{tab:lexical_baselines}."
    )
    lines += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}",
              r"\label{tab:headline_ci}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def _n_dirs(info: Dict) -> int:
    return int(info.get("n_test_dirs", 514))


# --------------------------------------------------------------------------- #
# Paired differences
# --------------------------------------------------------------------------- #

DIFF_GROUPS: List[Tuple[str, str]] = [
    ("abtt_minus_base", "ABTT $-$ Base"),
    ("ft_abtt_minus_ft_base", "Fine-tuned: ABTT $-$ Base"),
    ("ft_abtt_minus_zs_abtt", "Fine-tuned ABTT $-$ zero-shot ABTT"),
    ("tfidf_minus_abtt", "TF-IDF char 3--5 $-$ ABTT"),
    ("center_minus_base_at_abtt_layer", "Centering $-$ Base (ABTT layer)"),
    ("abtt_minus_center_at_abtt_layer", "ABTT $-$ centering (ABTT layer)"),
]


def _diff_cell(d: pd.DataFrame, group: str, label: str, task: str, metric: str) -> str:
    sub = d[(d["group"] == group) & (d["label"] == label) & (d["task"] == task)
            & (d["metric"] == metric)]
    if not len(sub):
        return "--"
    r = sub.iloc[0]
    return f"{_signed(r['estimate'], metric)} {_ci(r['ci_lo'], r['ci_hi'], metric)}"


def render_diffs(diffs: pd.DataFrame, info: Dict) -> str:
    lines = [
        HEADER.rstrip("\n"),
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"\textbf{Contrast} & \textbf{Model} & $\Delta$ AUROC & $\Delta$ Assignment acc. "
        r"& $\Delta$ DirAcc@1 \\",
        r"\midrule",
    ]
    first_group = True
    for group, title in DIFF_GROUPS:
        labels = [lab for lab in dict.fromkeys(diffs.loc[diffs["group"] == group, "label"])]
        if not labels:
            continue
        if not first_group:
            lines.append(r"\midrule")
        first_group = False
        for k, lab in enumerate(labels):
            cells = [_diff_cell(diffs, group, lab, "A", "auroc"),
                     _diff_cell(diffs, group, lab, "B", "assign"),
                     _diff_cell(diffs, group, lab, "B", "dir1")]
            lines.append(" & ".join([title if k == 0 else "", lab] + cells) + r" \\")
    spread = diffs[diffs["group"] == "spread_across_models"]
    if len(spread):
        lines.append(r"\midrule")
        for k, setting in enumerate(("Base", "ABTT")):
            cells = []
            for task, metric in (("A", "auroc"), ("B", "assign"), ("B", "dir1")):
                sub = spread[(spread["label"] == setting) & (spread["task"] == task)
                             & (spread["metric"] == metric)]
                cells.append("--" if not len(sub) else
                             f"{_num(sub.iloc[0]['estimate'], metric)} "
                             f"{_ci(sub.iloc[0]['ci_lo'], sub.iloc[0]['ci_hi'], metric)}")
            lines.append(" & ".join(["Spread over six models" if k == 0 else "", setting]
                                    + cells) + r" \\")
    n_dirs = _n_dirs(info)
    caption = (
        "Paired differences between headline cells, with 95\\% intervals from the same "
        f"directory-level bootstrap replicates as Table~\\ref{{tab:headline_ci}} ($B={_B(info)}$, "
        f"{n_dirs} test directories, seed {info['seed']}). Both cells of a contrast are "
        "recomputed on each replicate, so the interval reflects their correlation. Each cell "
        "sits at its own train-selected layer. The two centering contrasts compare baseline, "
        "centering alone ($D=0$) and ABTT at the layer of the ABTT cell, the Task A ABTT layer "
        "for $\\Delta$ AUROC and the Task B ABTT layer for the two routing columns. Spread: the largest "
        "minus the smallest of the six zero-shot models. Task B in points. Differences are "
        "computed at full precision, so they can differ by one unit in the last digit from "
        "the difference of the rounded cells. "
        + FIXED_CLAUSE
    )
    lines += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}",
              r"\label{tab:headline_ci_diffs}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Routing checks
# --------------------------------------------------------------------------- #


def render_pq(pq: pd.DataFrame, info: Dict, sweep: Optional[pd.DataFrame]) -> str:
    lines = [
        HEADER.rstrip("\n"),
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llrrlrlrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Setting} & \textbf{L} & Cos.\ SD & Exist./new AUROC "
        r"& Assign. & Oracle gap & $\tau$ & Exact & Skew \\",
        r"\midrule",
    ]
    order = [("Base", "baseline"), ("Base @ ABTT L", "baseline"),
             ("Centering @ ABTT L", "center"), ("ABTT", "abtt_optimal")]
    for m in ZS_ROWS:
        rows = pq[pq["row"] == m]
        head = rows[rows["headline"].astype(bool)]
        abtt = head[head["setting"] == "ABTT"]
        if not len(abtt):
            continue
        L_abtt = int(abtt.iloc[0]["layer"])
        picked = []
        for name, method in order:
            if name == "Base":
                sub = head[head["setting"] == "Base"]
            elif name == "ABTT":
                sub = abtt
            else:
                sub = rows[(~rows["headline"].astype(bool)) & (rows["method"] == method)
                           & (rows["layer"] == L_abtt)]
                if method == "baseline" and not len(sub):
                    sub = head[(head["setting"] == "Base") & (head["layer"] == L_abtt)]
            if len(sub):
                picked.append((name, sub.iloc[0]))
        # The baseline at the ABTT layer repeats the Base cell when both share a layer.
        if len(picked) > 1 and picked[0][0] == "Base" and picked[1][0] == "Base @ ABTT L" \
                and int(picked[0][1]["layer"]) == int(picked[1][1]["layer"]):
            picked.pop(1)
        for k, (name, r) in enumerate(picked):
            cells = [
                m if k == 0 else "", name, str(int(r["layer"])),
                format(r["pair_cos_sd"], ".3f"),
                f"{r['ev_auroc']:.3f} {_ci(r['ev_auroc_lo'], r['ev_auroc_hi'], 'auroc')}",
                _num(r["assign"], "assign"),
                f"{_signed(r['oracle_gap_assign'], 'assign')} "
                f"{_ci(r['oracle_gap_assign_lo'], r['oracle_gap_assign_hi'], 'assign')}",
                format(r["tau_paper"], ".3f"),
                _num(r["assign_exact"], "assign"),
                format(r["hub_skew"], ".2f"),
            ]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    moved = _moved_clause(pq, sweep)
    caption = (
        "Routing checks at the Task B cells (Table~\\ref{tab:taskB_headline}) of the six "
        "zero-shot models: the Base and ABTT cells at their train-selected layers (L), and "
        "baseline and centering alone ($D=0$, subtract the training mean) at the ABTT cell's "
        "layer (a second baseline row only where that layer differs). Cos.\\ SD: standard deviation of all test pairwise cosines. Exist./new AUROC: "
        "threshold-free AUROC of each test file's maximum cosine for existing against new "
        "files. Assign.: assignment accuracy at the train-fit $\\tau$ (the printed cell). "
        "Oracle gap: best assignment accuracy over every test threshold minus Assign. "
        "Exact: assignment accuracy with $\\tau$ re-fit as the exact best-F1 cut over "
        "all training pair scores instead of the 200-point grid. Skew: skewness of the "
        "$k$-occurrence distribution $N_{10}$ over test files \\citep{radovanovic2010hubs}. "
        + f"Brackets: 95\\% directory-bootstrap intervals ($B={_B(info)}$). " + moved
    )
    lines += [r"\end{tabular}", f"\\caption{{{caption}}}", r"\label{tab:pq_routing}",
              r"\end{table*}"]
    return "\n".join(lines) + "\n"


def _moved_clause(pq: pd.DataFrame, sweep: Optional[pd.DataFrame]) -> str:
    head = pq[pq["headline"].astype(bool)]
    n = len(head)
    fixed = {g: int(head[f"assign_moves_{g}"].astype(bool).sum()
                    + head[f"dir1_moves_{g}"].astype(bool).sum()) for g in ("fine", "exact")}
    s = (f"Holding layer and $D$ fixed, the fine grid (step $10^{{-4}}$) moves "
         f"{fixed['fine']} and the exact cut {fixed['exact']} of the {2 * n} printed Task B "
         "numbers")
    if sweep is not None:
        sub = sweep[(sweep["task"] == "B") & sweep["published_layer"].notna()]
        parts = []
        for g in ("fine", "exact"):
            g_sub = sub[sub["grid"] == g]
            n_moves = 0
            for m in ("assign", "dir1"):
                n_moves += int((g_sub[f"{m}_printed"].astype(str)
                                != g_sub[f"{m}_published"].astype(str)).sum())
            parts.append(f"{n_moves} ({g})")
        s += ("; re-selecting layer and $D$ under the new grid as well moves "
              + " and ".join(parts) + f" of {2 * len(sub[sub['grid'] == 'fine'])}")
    return s + "."


# --------------------------------------------------------------------------- #
# Compact headline tables
# --------------------------------------------------------------------------- #


def _data_row_label(line: str) -> Optional[str]:
    s = line.strip()
    if not s.endswith(r"\\") or "&" not in s or s.startswith(r"\textbf{Model}"):
        return None
    if s.startswith("&"):
        return None
    return s.split("&")[0].strip()


def _cell_text(cell: str) -> str:
    cell = cell.strip()
    m = re.fullmatch(r"\\textbf\{(.*)\}", cell)
    return m.group(1) if m else cell


def render_compact(head_tex: str, ci: pd.DataFrame, task: str, info: Dict) -> str:
    metrics = ("auroc", "gap") if task == "A" else ("assign", "dir1")
    out: List[str] = []
    for line in head_tex.splitlines():
        if line.startswith(r"\caption{"):
            line = line[:-1] + " Bracketed rows: " + _boot_clause(info, _n_dirs(info)) + \
                "; all cells with intervals are in Table~\\ref{tab:headline_ci}.}"
        if line.startswith(r"\setlength{\tabcolsep}"):
            line = r"\setlength{\tabcolsep}{1.8pt}"
        out.append(line)
        label = _data_row_label(line)
        if label is None:
            continue
        body = line.strip()[:-2]
        if "multicolumn" in body:
            key = LEX_CSV.get(label, label)
            vals = re.findall(r"\\multicolumn\{4\}\{c\}\{([^}]*)\}", body)
            cells = []
            for metric, printed in zip(metrics, vals):
                r = _lookup(ci, task, key, "ref", metric)
                _check(r, printed, metric, label)
                cells.append("\\multicolumn{4}{c}{{\\scriptsize "
                             + _ci(r["ci_lo"], r["ci_hi"], metric, True, True) + "}}")
            out.append(" & " + " & ".join(cells) + r" \\")
            continue
        parts = [p.strip() for p in body.split("&")]
        if len(parts) != 9:
            continue
        cells = []
        for k, printed in enumerate(parts[1:]):
            metric = metrics[0] if k < 4 else metrics[1]
            setting = SETTINGS[k % 4]
            printed = _cell_text(printed)
            if printed == "--":
                cells.append("")
                continue
            r = _lookup(ci, task, label, setting, metric)
            _check(r, printed, metric, f"{label}/{setting}")
            cells.append("{\\scriptsize " + _ci(r["ci_lo"], r["ci_hi"], metric, True, True) + "}")
        out.append(" & " + " & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


def _check(r: Optional[pd.Series], printed: str, metric: str, where: str) -> None:
    if r is None:
        raise SystemExit(f"no interval for printed cell {where} ({metric})")
    got = _num(r["estimate"], metric).replace("$-$", "-")
    if got != printed:
        raise SystemExit(
            f"{where} {metric}: table prints {printed} but the CI CSV estimate is {got}; "
            "the CSV is stale for this table"
        )


# --------------------------------------------------------------------------- #


def render_all(in_dir: Path, table_dir: Path, head_dir: Optional[Path] = None) -> Dict[str, str]:
    """Render every table into ``table_dir``; returns ``{filename: text}``."""
    head_dir = head_dir or table_dir
    ci, diffs, pq, info, sweep = _load(in_dir)
    out = {
        "headline_ci.tex": render_appendix(ci, info),
        "headline_ci_diffs.tex": render_diffs(diffs, info),
        "pq_routing.tex": render_pq(pq, info, sweep),
        "taskA_headline_ci.tex": render_compact(
            (head_dir / "taskA_headline.tex").read_text(), ci, "A", info),
        "taskB_headline_ci.tex": render_compact(
            (head_dir / "taskB_headline.tex").read_text(), ci, "B", info),
    }
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, text in out.items():
        (table_dir / name).write_text(text)
        print(f"wrote {table_dir / name}")
    return out
