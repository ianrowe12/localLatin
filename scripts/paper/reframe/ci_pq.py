#!/usr/bin/env python3
"""Directory-level bootstrap CIs (CI) and per-query routing checks (PQ), issue #233.

Three subcommands, all CPU-only on cached embeddings:

``compute``
    For every headline cell of ``tab:taskA_headline`` and ``tab:taskB_headline``
    (six zero-shot models x four settings, the three fine-tuned reference rows,
    and the three lexical reference systems that ``--lexical_csv`` adds):

    1. Reproduce the cell with the paper's own evaluator
       (``run_resubmit_evaluate.evaluate_single``, or ``evaluate_from_similarity``
       for the lexical scorers) at the train-selected layer, compare it with the
       results CSV behind the table and with the printed string, and stop if any
       cell fails at printed precision.
    2. Rebuild the same scores through ``ci_pq_core`` and check that the
       per-pair and per-file units aggregate to exactly the evaluator's numbers.
    3. Directory-level bootstrap: resample the 514 test directories with
       replacement, B times, and recompute every cell from the fixed units. tau,
       the ABTT components, D and the SIF probabilities stay fit on the fixed
       train split; only the test evaluation is resampled. Percentile intervals.
    4. Paired differences over the same replicates (ABTT minus baseline, the
       fine-tuned contrasts, the lexical reference against ABTT, across-model
       spreads, and the centering decomposition of PQ check 4).
    5. The PQ checks at the Task B cells: threshold-free existing-vs-new AUROC
       of the max cosine, oracle (test-optimal) vs train threshold, hubness
       (k-occurrence skewness, k = 10), centering only (D = 0), and the
       threshold refit on a fine grid and the exact best-F1 cut.

``sweep``
    Every layer of every model: baseline, SIF, centering (D = 0), ABTT and
    SIF+ABTT with D and tau re-fit under the paper grid, the fine grid and the
    exact cut. Applies the paper's train-only selection rule, so it answers
    whether any printed cell moves when the grid changes the selected layer or
    D too, and gives centering its own train-selected cells. The paper-grid
    pass must reproduce the published selection; the script checks it.

``render``
    LaTeX tables from the published CSVs (``docs/research/data/reframe_ci_pq``):
    an appendix table of every cell with its interval, a table of paired
    differences, the routing-check table, and compact drop-in versions of the
    two headline tables with an interval row under each model row, built from
    the committed headline ``.tex`` so the point estimates are the printed ones.

Paths are relative to ``--data_root`` (inputs under ``runs/``) or to this repo
(tables, published CSVs). Outputs under ``runs/active/reframe/ci_pq*/`` are
never overwritten; pick a new ``--out_dir`` instead.

    python scripts/paper/reframe/ci_pq.py compute --data_root /path/to/repo
    python scripts/paper/reframe/ci_pq.py sweep --data_root /path/to/repo
    python scripts/paper/reframe/ci_pq.py render
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for _p in (REPO / "src", REPO / "scripts" / "resubmit", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ci_pq_core as C  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixed inputs
# --------------------------------------------------------------------------- #

SPLIT_CSV = "runs/active/resubmit/data/phase_resubmit_split.csv"
RESULTS_CSV = "runs/active/resubmit/results/phase_resubmit_results.csv"
LEXICAL_CSV = "runs/active/resubmit/results/lexical_baselines.csv"
BASES = "runs/active/resubmit_bases/phase9_bases"
FT_BASES = "runs/active/resubmit_finetune_bases/phase9_bases"
FT_RESULTS_DIR = "runs/active/resubmit/results/finetune"
DEFAULT_OUT = "runs/active/reframe/ci_pq"
PUBLISH_DIR = REPO / "docs/research/data/reframe_ci_pq"
TABLE_DIR = REPO / "overleaf_drafts/tables"

MODELS: List[Tuple[str, str]] = [
    ("bowphs/LaTa", "LaTa"),
    ("bowphs/PhilTa", "PhilTa"),
    ("google/mt5-base", "mT5-base"),
    ("sentence-transformers/LaBSE", "LaBSE"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B"),
    ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", "KaLM-mini"),
]
DISPLAY = dict(MODELS)
SETTINGS: List[Tuple[str, str]] = [
    ("baseline", "Base"),
    ("sif_only", "SIF"),
    ("abtt_optimal", "ABTT"),
    ("sif_abtt_optimal", "SIF+ABTT"),
]
SETTING_OF = dict(SETTINGS)
# display name, pre-trained id, fine-tuned bases slug, results prefix
FINETUNED: List[Tuple[str, str, str, str]] = [
    ("LaTa", "bowphs/LaTa", "bowphs_LaTa-ft", "finetune_lata"),
    ("Qwen3-0.6B", "Qwen/Qwen3-Embedding-0.6B", "Qwen_Qwen3-Embedding-0.6B-ft",
     "finetune_qwen3_0.6b"),
    ("KaLM-mini", "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
     "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5-ft",
     "finetune_kalm_mini"),
]
LEXICAL: List[Tuple[str, str, str]] = [  # CSV label, scorer key, LaTeX label
    ("BM25 (word)", "bm25_word", "BM25 (word)"),
    ("TF-IDF char 3-5", "tfidf_char35", "TF-IDF char 3--5"),
    ("Levenshtein", "levenshtein", "Levenshtein"),
]
D_VALUES = [1, 2, 3, 5, 7, 10]
SIF_A = 0.001
GRIDS = ("paper", "fine", "exact")

# metric key -> (results-CSV column, printed format, scale)
METRICS = {
    "auroc": ("aucroc", ".3f", 1.0),
    "gap": ("gap", ".3f", 1.0),
    "assign": ("overall_assignment_acc", ".1f", 100.0),
    "dir1": ("dir_acc_at_1", ".1f", 100.0),
}
TASK_METRICS = {"A": ("auroc", "gap"), "B": ("assign", "dir1")}
FT_COLS = {  # comparison-CSV column per metric
    "auroc": "taskA_aucroc",
    "gap": "taskA_cosine_gap",
    "assign": "taskB_assignment_acc",
    "dir1": "taskB_dir_acc_at_1",
}


def slug(model_id: str) -> str:
    return model_id.replace("/", "_")


# --------------------------------------------------------------------------- #
# Cells and configurations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """One set of test scores: a source at a layer under a method.

    ``kind``: ``zs`` (zero-shot model id), ``ft`` (fine-tuned display name) or
    ``lex`` (lexical CSV label; no layer or method).
    """

    kind: str
    source: str
    layer: int = -1
    method: str = "lexical"

    @property
    def pooling(self) -> str:
        return "sif" if self.method.startswith("sif") else "mean"

    def tag(self) -> str:
        if self.kind == "lex":
            return f"lex:{self.source}"
        return f"{self.kind}:{self.source}:L{self.layer}:{self.method}"


@dataclass
class Cell:
    """One printed headline cell."""

    task: str          # "A" or "B"
    row: str           # row label as printed
    setting: str       # Base / SIF / ABTT / SIF+ABTT, or "ref" for lexical
    kind: str
    config: Config
    reference: Dict[str, float] = field(default_factory=dict)  # CSV values


def headline_cells(data_root: Path, results: pd.DataFrame) -> List[Cell]:
    """Every cell of the two headline tables, with the CSV values behind it.

    Layers come from ``build_headline_tables.best_rows``, the function that
    renders the tables, so the selection is the published one by construction.
    """
    import build_headline_tables as bht

    cells: List[Cell] = []
    for task, criterion in (("A", "train_aucroc"), ("B", "train_dir_acc_at_1")):
        best = bht.best_rows(results, "hidden", criterion)
        for model_id, display in MODELS:
            for method, setting in SETTINGS:
                row = best[(best["_model_id"] == model_id) & (best["_method"] == method)].iloc[0]
                cfg = Config("zs", model_id, int(row["layer"]), method)
                ref = {m: float(row[METRICS[m][0]]) for m in TASK_METRICS[task]}
                ref.update(tau=float(row["tau"]), D=int(row["D"]))
                cells.append(Cell(task, display, setting, "zs", cfg, ref))
        for display, _, _, prefix in FINETUNED:
            comp = pd.read_csv(data_root / FT_RESULTS_DIR / f"{prefix}_ceiling_comparison.csv")
            for suffix, method, setting in (("", "baseline", "Base"),
                                            (" + ABTT", "abtt_optimal", "ABTT")):
                row = comp[comp["system"] == f"{display} (fine-tuned){suffix}"].iloc[0]
                assert row["method"] == method
                layer = int(row[f"task{task}_layer"])
                ref = {m: float(row[FT_COLS[m]]) for m in TASK_METRICS[task]}
                ref.update(D=int(row[f"task{task}_D"]))
                if task == "B":
                    ref["tau"] = float(row["taskB_tau"])
                cells.append(Cell(task, f"{display} (fine-tuned)", setting, "ft",
                                  Config("ft", display, layer, method), ref))
        lex = pd.read_csv(data_root / LEXICAL_CSV)
        for label, _, _ in LEXICAL:
            row = lex[lex["model"] == label].iloc[0]
            ref = {m: float(row[METRICS[m][0]]) for m in TASK_METRICS[task]}
            ref["tau"] = float(row["tau"])
            cells.append(Cell(task, label, "ref", "lex", Config("lex", label), ref))
    return cells


# --------------------------------------------------------------------------- #
# Printed values in the committed headline tables
# --------------------------------------------------------------------------- #


def _strip_tex(cell: str) -> str:
    cell = cell.strip()
    m = re.fullmatch(r"\\textbf\{(.*)\}", cell)
    return m.group(1) if m else cell


def printed_cells(tex_path: Path) -> Dict[Tuple[str, str, str], str]:
    """``(row label, setting, block)`` -> printed string, block ``left``/``right``.

    Lexical rows (a ``\\multicolumn`` spanning each block) are keyed with
    setting ``ref``.
    """
    out: Dict[Tuple[str, str, str], str] = {}
    if not tex_path.exists():
        return out
    settings = [s for _, s in SETTINGS]
    for line in tex_path.read_text().splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or "&" not in line or line.startswith(r"\textbf{Model}"):
            continue
        parts = [p.strip() for p in line[:-2].split("&")]
        label = parts[0]
        if "multicolumn" in line:
            vals = re.findall(r"\\multicolumn\{4\}\{c\}\{([^}]*)\}", line)
            if len(vals) == 2:
                out[(label, "ref", "left")] = vals[0]
                out[(label, "ref", "right")] = vals[1]
            continue
        if len(parts) != 9:
            continue
        for k, s in enumerate(settings):
            out[(label, s, "left")] = _strip_tex(parts[1 + k])
            out[(label, s, "right")] = _strip_tex(parts[5 + k])
    return out


LEX_TEX_LABEL = {csv: tex for csv, _, tex in LEXICAL}


def printed_value(printed: Dict, cell: Cell, metric: str) -> Optional[str]:
    block = "left" if metric in ("auroc", "assign") else "right"
    label = LEX_TEX_LABEL.get(cell.row, cell.row) if cell.kind == "lex" else cell.row
    return printed.get((label, cell.setting, block))


# --------------------------------------------------------------------------- #
# Loading and processing
# --------------------------------------------------------------------------- #


class Sources:
    """Loads embedding matrices (aligned by filename) and lexical score matrices."""

    def __init__(self, data_root: Path, split: pd.DataFrame, workers: int = -1):
        from embedding_alignment import AlignmentResolver

        self.data_root = data_root
        self.split = split
        self.resolver = AlignmentResolver(split)
        self.workers = workers
        self._lex: Dict[str, np.ndarray] = {}

    def emb_path(self, cfg: Config) -> Path:
        sub = "hidden_sif_tokempty" if cfg.pooling == "sif" else "hidden_mean_tokempty"
        suffix = "_sif" if cfg.pooling == "sif" else ""
        if cfg.kind == "zs":
            base = self.data_root / BASES / slug(cfg.source)
        else:
            ft_slug = {d: s for d, _, s, _ in FINETUNED}[cfg.source]
            base = self.data_root / FT_BASES / ft_slug
        return base / sub / f"hidden_layer{cfg.layer}_embeddings{suffix}.npy"

    def embeddings(self, cfg: Config) -> np.ndarray:
        return self.resolver.load(self.emb_path(cfg))

    def lexical(self, label: str) -> np.ndarray:
        if label not in self._lex:
            import lexical_baselines as lb

            key = {csv: k for csv, k, _ in LEXICAL}[label]
            texts = lb.load_texts(self.split, self.data_root)
            train_idx = np.flatnonzero(self.split["split"].values == "train")
            t0 = time.time()
            self._lex[label] = lb.build_score_matrix(key, texts, train_idx, workers=self.workers)
            print(f"  [lexical] {label}: {time.time() - t0:.1f}s", flush=True)
        return self._lex[label]


def process(emb_all: np.ndarray, split: pd.DataFrame, method: str, D: int
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized train and test embeddings under ``method``, fit on train.

    Mirrors ``run_resubmit_evaluate.evaluate_single``; ``center`` is the D = 0
    check (subtract the train mean, remove no component).
    """
    from canon_retrieval import l2_normalize
    from sif_abtt import EmbeddingCleaner

    train = emb_all[split["split"].values == "train"]
    test = emb_all[split["split"].values == "test"]
    if method in ("baseline", "sif_only"):
        pass
    elif method in ("abtt_optimal", "sif_abtt_optimal", "abtt_fixed", "sif_abtt_fixed"):
        cleaner = EmbeddingCleaner(num_components=D, center=True).fit(train)
        train, test = cleaner.transform(train), cleaner.transform(test)
    elif method in ("center", "sif_center"):
        mu = train.mean(axis=0)
        train, test = train - mu, test - mu
    else:
        raise ValueError(method)
    return l2_normalize(train), l2_normalize(test)


def select_D(train: np.ndarray, fids: np.ndarray, grid: str) -> Tuple[int, float]:
    """``find_optimal_D_phase11`` with a choice of threshold grid."""
    from canon_retrieval import l2_normalize
    from sif_abtt import EmbeddingCleaner

    best_D, best = D_VALUES[0], -1.0
    for D in D_VALUES:
        cleaned = EmbeddingCleaner(num_components=D, center=True).fit(train).transform(train)
        tn = l2_normalize(cleaned)
        sim = tn @ tn.T
        tau = C.fit_tau(*C.pair_arrays(sim, fids), grid=grid)
        score = C.train_dir_acc_at_1(sim, fids, tau)
        if score > best:
            best, best_D = score, D
    return best_D, best


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #


@dataclass
class Units:
    """Everything the bootstrap needs from one configuration."""

    cfg: Config
    D: Optional[int]
    tau: float
    test_sim: np.ndarray
    train_sim: np.ndarray
    pair_scores: np.ndarray = None
    ri_pairs: C.RankIndex = None
    tb: C.TaskBUnits = None
    ri_files: C.RankIndex = None


def paper_reproduce(src: Sources, cfg: Config, results: pd.DataFrame,
                    ft_layers: Dict[str, pd.DataFrame], lex: pd.DataFrame) -> Tuple[Dict, Dict]:
    """(paper evaluator row, reference CSV row) for one configuration."""
    import run_resubmit_evaluate as E

    split = src.split
    if cfg.kind == "lex":
        import lexical_baselines as lb

        row = lb.evaluate_single_split(src.lexical(cfg.source), split, cfg.source)
        ref = lex[lex["model"] == cfg.source].iloc[0].to_dict()
        return row, ref
    emb = src.embeddings(cfg)
    name = cfg.source if cfg.kind == "zs" else cfg.source + "-ft"
    row = E.evaluate_single(
        emb_all=emb, split_meta=split, method=cfg.method, D=10, sif_a=SIF_A,
        model_name=name, repr_name="hidden", pooling_src=cfg.pooling,
        layer=cfg.layer, D_values=D_VALUES,
    )
    if cfg.kind == "zs":
        sub = results[(results["model"] == cfg.source) & (results["repr"] == "hidden")
                      & (results["layer"] == cfg.layer) & (results["method"] == cfg.method)]
    else:
        lr = ft_layers[cfg.source]
        sub = lr[(lr["layer"] == cfg.layer) & (lr["method"] == cfg.method)]
    ref = sub.iloc[0].to_dict() if len(sub) == 1 else {}
    return row, ref


CHECK_KEYS = ("tau", "D", "aucroc", "gap", "overall_assignment_acc", "dir_acc_at_1",
              "train_aucroc", "train_dir_acc_at_1")
# Counts, tau and D must agree exactly. Continuous scores may differ in the
# eighth digit: cosines are float32 matrix products, and a different BLAS
# thread count (the paper's run vs this one) changes their last bit, which can
# reorder a few near-tied pairs. 1e-6 is far below the printed precision,
# which is checked separately cell by cell.
TOL = {"tau": 0.0, "D": 0.0, "overall_assignment_acc": 0.0, "dir_acc_at_1": 0.0,
       "train_dir_acc_at_1": 0.0, "aucroc": 1e-6, "gap": 1e-6, "train_aucroc": 1e-6}


def build_units(src: Sources, cfg: Config, D: Optional[int], tau: Optional[float],
                need_pairs: bool, need_files: bool) -> Units:
    split = src.split
    train_mask = split["split"].values == "train"
    test_mask = ~train_mask
    if cfg.kind == "lex":
        sim = src.lexical(cfg.source)
        train_sim = sim[np.ix_(train_mask, train_mask)]
        test_sim = sim[np.ix_(test_mask, test_mask)]
    else:
        tr, te = process(src.embeddings(cfg), split, cfg.method, D or 0)
        train_sim, test_sim = tr @ tr.T, te @ te.T
    train_f = split.loc[train_mask, "folder_id"].values
    test_f = split.loc[test_mask, "folder_id"].values
    if tau is None:
        tau = C.fit_tau(*C.pair_arrays(train_sim, train_f), grid="paper")
    u = Units(cfg, D, tau, test_sim, train_sim)
    if need_pairs:
        s, lab = C.pair_arrays(test_sim, test_f)
        u.pair_scores = s
        u.ri_pairs = C.rank_index(s, lab)
    if need_files:
        hp = split.loc[test_mask, "has_test_partner"].values.astype(bool)
        u.tb = C.taskb_units(test_sim, test_f, hp, tau)
        u.ri_files = C.rank_index(u.tb.max_cos, u.tb.existing)
    return u


def _close(a, b, tol=1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


_BOOT: Dict = {}


def _boot_range(rng: Tuple[int, int]) -> Tuple[int, int, Dict[Tuple[str, str], np.ndarray]]:
    """Bootstrap replicates ``rng[0]:rng[1]`` for every configuration.

    Runs in a forked worker; the units and the draw counts come from ``_BOOT``.
    """
    a, b = rng
    counts, pdirs, code = _BOOT["counts"], _BOOT["pdirs"], _BOOT["code"]
    pos_mask = pdirs.is_pos
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for tag, u, pairs, files in _BOOT["jobs"]:
        if pairs:
            out[(tag, "auroc")] = np.empty(b - a)
            out[(tag, "gap")] = np.empty(b - a)
        if files:
            for m in ("assign", "dir1", "ev_auroc", "oracle_assign", "oracle_dir1"):
                out[(tag, m)] = np.empty(b - a)
    for start in range(a, b, _BOOT["chunk"]):
        cnt = counts[start:min(start + _BOOT["chunk"], b)]
        sl = slice(start - a, start - a + len(cnt))
        Wp = pdirs.weights(cnt)
        Wp_pos, Wp_neg = Wp[:, pos_mask], Wp[:, ~pos_mask]
        wp_pos, wp_neg = Wp_pos.sum(axis=1), Wp_neg.sum(axis=1)
        Wf = cnt[:, code]
        wf_sum = Wf.sum(axis=1)
        for tag, u, pairs, files in _BOOT["jobs"]:
            if pairs:
                out[(tag, "auroc")][sl] = C.weighted_auroc(u.ri_pairs, Wp)
                sc = u.pair_scores.astype(np.float64)
                out[(tag, "gap")][sl] = (Wp_pos @ sc[pos_mask]) / wp_pos \
                    - (Wp_neg @ sc[~pos_mask]) / wp_neg
            if not files:
                continue
            tb = u.tb
            new = (~tb.existing).astype(float)
            out[(tag, "assign")][sl] = (Wf @ tb.assign_correct.astype(float)) / wf_sum
            out[(tag, "dir1")][sl] = (Wf @ tb.dir1_correct.astype(float)) / wf_sum
            out[(tag, "ev_auroc")][sl] = C.weighted_auroc(u.ri_files, Wf)
            out[(tag, "oracle_assign")][sl] = C.oracle_accuracy(
                tb.max_cos, tb.existing.astype(float), new, Wf)[0]
            out[(tag, "oracle_dir1")][sl] = C.oracle_accuracy(
                tb.max_cos, (tb.existing & tb.top_dir_correct).astype(float), new, Wf)[0]
    return a, b, out


def cmd_compute(args: argparse.Namespace) -> None:
    t_start = time.time()
    cpu_start = time.process_time()
    data_root = Path(args.data_root).resolve()
    out_dir = data_root / args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} exists and is not empty; results are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(data_root / SPLIT_CSV)
    results = pd.read_csv(data_root / RESULTS_CSV)
    lex = pd.read_csv(data_root / LEXICAL_CSV)
    ft_layers = {d: pd.read_csv(data_root / FT_RESULTS_DIR / f"{p}_layer_results.csv")
                 for d, _, _, p in FINETUNED}
    src = Sources(data_root, split, workers=args.workers)
    cells = headline_cells(data_root, results)
    if args.only_models:
        keep = set(args.only_models.split("|"))
        cells = [c for c in cells if c.row.replace(" (fine-tuned)", "") in keep]
    printed = {"A": printed_cells(TABLE_DIR / "taskA_headline.tex"),
               "B": printed_cells(TABLE_DIR / "taskB_headline.tex")}

    test_mask = split["split"].values == "test"
    test_f = split.loc[test_mask, "folder_id"].values
    train_f = split.loc[~test_mask, "folder_id"].values
    dir_labels, code = C.directory_codes(test_f)
    print(f"test: {test_mask.sum()} files in {len(dir_labels)} directories; "
          f"B={args.B} seed={args.seed}", flush=True)

    # ---- 1-2: reproduction ------------------------------------------------ #
    configs: Dict[Config, Dict] = {}
    for c in cells:
        configs.setdefault(c.config, {"tasks": set()})["tasks"].add(c.task)

    units: Dict[Config, Units] = {}
    repro_rows = []
    for cfg, info in configs.items():
        t0 = time.time()
        row, ref = paper_reproduce(src, cfg, results, ft_layers, lex)
        D = None if cfg.kind == "lex" else int(row["D"])
        u = build_units(src, cfg, D, None, need_pairs=True, need_files=True)
        fast = {
            "tau": u.tau,
            "D": D,
            "aucroc": float(C.weighted_auroc(u.ri_pairs, np.ones(len(u.pair_scores)))[0]),
            "gap": float(u.pair_scores[u.ri_pairs.pos_idx].mean()
                         - np.delete(u.pair_scores, u.ri_pairs.pos_idx).mean()),
            "overall_assignment_acc": float(u.tb.assign_correct.mean()),
            "dir_acc_at_1": float(u.tb.dir1_correct.mean()),
        }
        rec = {"config": cfg.tag(), "seconds": round(time.time() - t0, 1)}
        ok = True
        for k in CHECK_KEYS:
            p, r = row.get(k), ref.get(k)
            rec[f"paper_{k}"] = p
            rec[f"csv_{k}"] = r
            if r is not None and not (isinstance(r, float) and np.isnan(r)) and p is not None:
                tol = TOL[k]
                if not _close(p, r, tol):
                    ok = False
            if k in fast:
                rec[f"fast_{k}"] = fast[k]
                if fast[k] is not None and p is not None:
                    tol = TOL[k]
                    if not _close(fast[k], p, tol):
                        ok = False
        rec["all_match"] = ok
        repro_rows.append(rec)
        print(f"  {cfg.tag():<70} match={ok} ({rec['seconds']}s)", flush=True)
        units[cfg] = u

    repro = pd.DataFrame(repro_rows)
    repro.to_csv(out_dir / "reproduction_configs.csv", index=False)

    cell_checks = []
    for c in cells:
        u = units[c.config]
        fast = {"auroc": float(C.weighted_auroc(u.ri_pairs, np.ones(len(u.pair_scores)))[0]),
                "gap": float(u.pair_scores[u.ri_pairs.pos_idx].mean()
                             - np.delete(u.pair_scores, u.ri_pairs.pos_idx).mean()),
                "assign": float(u.tb.assign_correct.mean()),
                "dir1": float(u.tb.dir1_correct.mean())}
        for m in TASK_METRICS[c.task]:
            _, fmt, scale = METRICS[m]
            pr = printed_value(printed[c.task], c, m)
            got = format(fast[m] * scale, fmt)
            want = format(c.reference[m] * scale, fmt)
            cell_checks.append({
                "task": c.task, "row": c.row, "setting": c.setting, "metric": m,
                "layer": c.config.layer, "config": c.config.tag(),
                "csv_value": c.reference[m], "recomputed": fast[m],
                "csv_printed": want, "recomputed_printed": got,
                "table_printed": pr if pr is not None else "",
                "match": got == want and (pr is None or pr == got),
            })
    checks = pd.DataFrame(cell_checks)
    checks.to_csv(out_dir / "reproduction_cells.csv", index=False)
    n_bad = int((~checks["match"]).sum()) + int((~repro["all_match"]).sum())
    print(f"reproduction: {len(checks)} printed cells, {len(repro)} configs, "
          f"{n_bad} mismatches", flush=True)
    if n_bad and not args.allow_mismatch:
        raise SystemExit("reproduction failed; see reproduction_*.csv. Not resampling.")

    # ---- extra configs for PQ check 4 (centering) ------------------------- #
    zs_b = {(c.config.source, c.setting): c for c in cells if c.task == "B" and c.kind == "zs"}
    zs_a = {(c.config.source, c.setting): c for c in cells if c.task == "A" and c.kind == "zs"}
    extra: Dict[str, Config] = {}
    for model_id, _ in MODELS:
        if (model_id, "ABTT") not in zs_b:
            continue
        for task, table in (("B", zs_b), ("A", zs_a)):
            L_abtt = table[(model_id, "ABTT")].config.layer
            L_base = table[(model_id, "Base")].config.layer
            extra[f"{task}|{model_id}|center@abtt"] = Config("zs", model_id, L_abtt, "center")
            extra[f"{task}|{model_id}|base@abtt"] = Config("zs", model_id, L_abtt, "baseline")
            extra[f"{task}|{model_id}|center@base"] = Config("zs", model_id, L_base, "center")
    for key, cfg in extra.items():
        if cfg not in units:
            units[cfg] = build_units(src, cfg, 0, None, need_pairs=True, need_files=True)

    # ---- 3: bootstrap ----------------------------------------------------- #
    counts = C.bootstrap_counts(len(dir_labels), args.B, args.seed)
    pdirs = C.PairDirs.from_codes(code)
    pos_mask = pdirs.is_pos
    cfg_list = list(units)
    # Pair-level (Task A) replicates cost a pass over 367k pairs each, so they
    # are drawn only where a Task A number needs them.
    need_pairs = {c.config for c in cells if c.task == "A"}
    need_pairs |= {cfg for k, cfg in extra.items() if k.startswith("A|")}
    need_files = {c.config for c in cells if c.task == "B"}
    need_files |= {cfg for k, cfg in extra.items() if k.startswith("B|")}
    PAIR_M = ("auroc", "gap")
    FILE_M = ("assign", "dir1", "ev_auroc", "oracle_assign", "oracle_dir1")
    keys = []
    for cfg in cfg_list:
        for m in (PAIR_M if cfg in need_pairs else ()) + (FILE_M if cfg in need_files else ()):
            keys.append((cfg.tag(), m))
    _BOOT.update(counts=counts, pdirs=pdirs, code=code, chunk=args.chunk,
                 jobs=[(cfg.tag(), units[cfg], cfg in need_pairs, cfg in need_files)
                       for cfg in cfg_list])
    reps: Dict[Tuple[str, str], np.ndarray] = {k: np.empty(args.B) for k in keys}
    n_proc = max(1, args.procs)
    step = max(args.chunk, -(-args.B // (4 * n_proc)))
    ranges = [(a, min(a + step, args.B)) for a in range(0, args.B, step)]
    t0 = time.time()
    if n_proc == 1:
        parts = map(_boot_range, ranges)
    else:
        import multiprocessing as mp

        pool = mp.get_context("fork").Pool(n_proc)
        parts = pool.imap_unordered(_boot_range, ranges)
    for k, (a, b, out) in enumerate(parts, 1):
        for key, v in out.items():
            reps[key][a:b] = v
        print(f"  bootstrap range {k}/{len(ranges)} done ({time.time() - t0:.0f}s)", flush=True)
    if n_proc > 1:
        pool.close()
        pool.join()

    # point estimates (unit weights)
    point: Dict[Tuple[str, str], float] = {}
    for cfg in cfg_list:
        u = units[cfg]
        tag = cfg.tag()
        s = u.pair_scores.astype(np.float64)
        point[(tag, "auroc")] = float(C.weighted_auroc(u.ri_pairs, np.ones(len(s)))[0])
        point[(tag, "gap")] = float(s[pos_mask].mean() - s[~pos_mask].mean())
        point[(tag, "assign")] = float(u.tb.assign_correct.mean())
        point[(tag, "dir1")] = float(u.tb.dir1_correct.mean())
        point[(tag, "ev_auroc")] = float(C.weighted_auroc(u.ri_files, np.ones(len(u.tb.max_cos)))[0])
        new = (~u.tb.existing).astype(float)
        acc, cut = C.oracle_accuracy(u.tb.max_cos, u.tb.existing.astype(float), new)
        point[(tag, "oracle_assign")] = float(acc[0])
        point[(tag, "oracle_assign_tau")] = float(cut[0])
        acc, cut = C.oracle_accuracy(u.tb.max_cos, (u.tb.existing & u.tb.top_dir_correct)
                                     .astype(float), new)
        point[(tag, "oracle_dir1")] = float(acc[0])
        point[(tag, "oracle_dir1_tau")] = float(cut[0])

    def ci_row(tag: str, metric: str) -> Dict:
        v = reps[(tag, metric)]
        lo, hi = C.percentile_ci(v)
        return {"estimate": point[(tag, metric)], "ci_lo": lo, "ci_hi": hi,
                "se": float(v.std(ddof=1))}

    # ---- headline CI table ----------------------------------------------- #
    rows = []
    for c in cells:
        for m in TASK_METRICS[c.task]:
            _, fmt, scale = METRICS[m]
            r = {"task": c.task, "row": c.row, "setting": c.setting, "kind": c.kind,
                 "metric": m, "layer": c.config.layer if c.kind != "lex" else "",
                 "D": units[c.config].D if units[c.config].D is not None else "",
                 "config": c.config.tag(), "scale": scale, "fmt": fmt}
            r.update(ci_row(c.config.tag(), m))
            rows.append(r)
    ci = pd.DataFrame(rows)
    ci["B"] = args.B
    ci["seed"] = args.seed
    ci["interval"] = "percentile95"
    ci.to_csv(out_dir / "headline_ci.csv", index=False)

    # ---- 4: paired differences ------------------------------------------- #
    def cell_tag(task, row, setting):
        for c in cells:
            if c.task == task and c.row == row and c.setting == setting:
                return c.config.tag()
        return None

    diffs = []

    def add_diff(group, label, task, metric, tag_a, tag_b, note=""):
        if tag_a is None or tag_b is None:
            return
        if (tag_a, metric) not in reps or (tag_b, metric) not in reps:
            return
        d = reps[(tag_a, metric)] - reps[(tag_b, metric)]
        lo, hi = C.percentile_ci(d)
        diffs.append({"group": group, "label": label, "task": task, "metric": metric,
                      "a": tag_a, "b": tag_b,
                      "estimate": point[(tag_a, metric)] - point[(tag_b, metric)],
                      "ci_lo": lo, "ci_hi": hi, "se": float(d.std(ddof=1)),
                      "share_le_0": float((d <= 0).mean()),
                      "share_ge_0": float((d >= 0).mean()), "note": note})

    for _, disp in MODELS:
        for task in ("A", "B"):
            for m in TASK_METRICS[task]:
                add_diff("abtt_minus_base", disp, task, m,
                         cell_tag(task, disp, "ABTT"), cell_tag(task, disp, "Base"))
    for disp, _, _, _ in FINETUNED:
        ft = f"{disp} (fine-tuned)"
        for task in ("A", "B"):
            for m in TASK_METRICS[task]:
                add_diff("ft_abtt_minus_ft_base", disp, task, m,
                         cell_tag(task, ft, "ABTT"), cell_tag(task, ft, "Base"))
                add_diff("ft_abtt_minus_zs_abtt", disp, task, m,
                         cell_tag(task, ft, "ABTT"), cell_tag(task, disp, "ABTT"))
                add_diff("ft_base_minus_zs_base", disp, task, m,
                         cell_tag(task, ft, "Base"), cell_tag(task, disp, "Base"))
    for task in ("A", "B"):
        for m in TASK_METRICS[task]:
            tf = cell_tag(task, "TF-IDF char 3-5", "ref")
            for _, disp in MODELS:
                add_diff("tfidf_minus_abtt", disp, task, m, tf, cell_tag(task, disp, "ABTT"))
    for key, cfg in extra.items():
        task, model_id, which = key.split("|")
        disp = DISPLAY[model_id]
        mets = TASK_METRICS[task][:1] if task == "A" else TASK_METRICS[task]
        for m in mets:
            if which == "center@abtt":
                add_diff("center_minus_base_at_abtt_layer", disp, task, m, cfg.tag(),
                         extra[f"{task}|{model_id}|base@abtt"].tag())
                add_diff("abtt_minus_center_at_abtt_layer", disp, task, m,
                         cell_tag(task, disp, "ABTT"), cfg.tag())
            elif which == "center@base":
                add_diff("center_minus_base_at_base_layer", disp, task, m, cfg.tag(),
                         cell_tag(task, disp, "Base"))
    # derived statistics over the six zero-shot models: spreads and maxima
    zs_disp = [d for _, d in MODELS]
    for task in ("A", "B"):
        for m in TASK_METRICS[task][:1] + (("dir1",) if task == "B" else ()):
            spreads = {}
            for setting in ("Base", "ABTT", "SIF+ABTT"):
                tags = [cell_tag(task, d, setting) for d in zs_disp]
                if any(t is None for t in tags):
                    continue
                mat = np.vstack([reps[(t, m)] for t in tags])
                pt = np.array([point[(t, m)] for t in tags])
                rng_rep = mat.max(axis=0) - mat.min(axis=0)
                spreads[setting] = (pt.max() - pt.min(), rng_rep)
                lo, hi = C.percentile_ci(rng_rep)
                diffs.append({"group": "spread_across_models", "label": setting,
                              "task": task, "metric": m, "a": "max", "b": "min",
                              "estimate": pt.max() - pt.min(), "ci_lo": lo, "ci_hi": hi,
                              "se": float(rng_rep.std(ddof=1)), "share_le_0": np.nan,
                              "share_ge_0": np.nan, "note": "max minus min over six models"})
                if setting == "ABTT":
                    tf = cell_tag(task, "TF-IDF char 3-5", "ref")
                    if tf is not None:
                        d = reps[(tf, m)] - mat.max(axis=0)
                        lo, hi = C.percentile_ci(d)
                        diffs.append({"group": "tfidf_minus_best_abtt", "label": "six models",
                                      "task": task, "metric": m, "a": tf, "b": "max ABTT",
                                      "estimate": point[(tf, m)] - pt.max(), "ci_lo": lo,
                                      "ci_hi": hi, "se": float(d.std(ddof=1)),
                                      "share_le_0": float((d <= 0).mean()),
                                      "share_ge_0": float((d >= 0).mean()),
                                      "note": "best ABTT cell re-chosen in each replicate"})
            if "Base" in spreads and "ABTT" in spreads:
                d = spreads["Base"][1] - spreads["ABTT"][1]
                lo, hi = C.percentile_ci(d)
                diffs.append({"group": "spread_base_minus_abtt", "label": "six models",
                              "task": task, "metric": m, "a": "Base spread",
                              "b": "ABTT spread",
                              "estimate": spreads["Base"][0] - spreads["ABTT"][0],
                              "ci_lo": lo, "ci_hi": hi, "se": float(d.std(ddof=1)),
                              "share_le_0": float((d <= 0).mean()),
                              "share_ge_0": float((d >= 0).mean()), "note": ""})
    diff_df = pd.DataFrame(diffs)
    diff_df["B"] = args.B
    diff_df["seed"] = args.seed
    diff_df.to_csv(out_dir / "headline_ci_diffs.csv", index=False)

    # ---- 5: PQ checks per Task B configuration ---------------------------- #
    pq_rows = []
    b_cells = {c.config: c for c in cells if c.task == "B"}
    pq_cfgs = list(b_cells) + [cfg for k, cfg in extra.items() if k.startswith("B|")]
    for cfg in dict.fromkeys(pq_cfgs):
        u = units[cfg]
        tag = cfg.tag()
        c = b_cells.get(cfg)
        test_hp = u.tb.existing
        r = {"config": tag, "row": c.row if c else DISPLAY.get(cfg.source, cfg.source),
             "setting": c.setting if c else cfg.method, "kind": cfg.kind,
             "layer": cfg.layer if cfg.kind != "lex" else "", "method": cfg.method,
             "D": u.D if u.D is not None else "", "headline": c is not None}
        s = u.pair_scores.astype(np.float64)
        r.update(tau_paper=u.tau, pair_cos_mean=float(s.mean()), pair_cos_sd=float(s.std()),
                 maxcos_sd=float(u.tb.max_cos.std()),
                 maxcos_sd_existing=float(u.tb.max_cos[test_hp].std()),
                 maxcos_sd_new=float(u.tb.max_cos[~test_hp].std()))
        for m in ("assign", "dir1", "ev_auroc", "oracle_assign", "oracle_dir1"):
            cr = ci_row(tag, m)
            r[m] = cr["estimate"]
            r[f"{m}_lo"], r[f"{m}_hi"] = cr["ci_lo"], cr["ci_hi"]
        r["oracle_assign_tau"] = point[(tag, "oracle_assign_tau")]
        r["oracle_dir1_tau"] = point[(tag, "oracle_dir1_tau")]
        for m in ("assign", "dir1"):
            d = reps[(tag, f"oracle_{m}")] - reps[(tag, m)]
            lo, hi = C.percentile_ci(d)
            r[f"oracle_gap_{m}"] = point[(tag, f"oracle_{m}")] - point[(tag, m)]
            r[f"oracle_gap_{m}_lo"], r[f"oracle_gap_{m}_hi"] = lo, hi
        r.update(C.hubness(u.test_sim, k=10))
        train_f_sims, train_lab = C.pair_arrays(u.train_sim, train_f)
        for grid in ("fine", "exact"):
            tau_g = C.fit_tau(train_f_sims, train_lab, grid=grid)
            tb = C.taskb_units(u.test_sim, test_f, test_hp, tau_g)
            r[f"tau_{grid}"] = tau_g
            r[f"assign_{grid}"] = float(tb.assign_correct.mean())
            r[f"dir1_{grid}"] = float(tb.dir1_correct.mean())
        for m in ("assign", "dir1"):
            p0 = format(100 * r[m], ".1f")
            r[f"{m}_printed"] = p0
            r[f"{m}_moves_fine"] = format(100 * r[f"{m}_fine"], ".1f") != p0
            r[f"{m}_moves_exact"] = format(100 * r[f"{m}_exact"], ".1f") != p0
        pq_rows.append(r)
    pq = pd.DataFrame(pq_rows)
    pq["B"] = args.B
    pq["seed"] = args.seed
    pq.to_csv(out_dir / "pq_cells.csv", index=False)

    np.savez_compressed(out_dir / "bootstrap_replicates.npz",
                        **{f"{t}|{m}": v for (t, m), v in reps.items()},
                        counts=counts.astype(np.int16), directories=dir_labels)
    info = run_info(args, t_start, cpu_start)
    info.update(n_configs=len(units), n_cells=len(cells), n_test_dirs=int(len(dir_labels)),
                reproduction_mismatches=n_bad)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


def run_info(args, t_start, cpu_start) -> Dict:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ""
    return {
        "command": " ".join(sys.argv),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "host": socket.gethostname(),
        "git_head": head,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "wall_seconds": round(time.time() - t_start, 1),
        "cpu_seconds_main_process": round(time.process_time() - cpu_start, 1),
        "B": getattr(args, "B", None),
        "seed": getattr(args, "seed", None),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #


def discover_layers(path_dir: Path, suffix: str) -> List[int]:
    layers = []
    for f in path_dir.glob(f"hidden_layer*_embeddings{suffix}.npy"):
        m = re.fullmatch(rf"hidden_layer(\d+)_embeddings{suffix}", f.stem)
        if m and int(m.group(1)) >= 1:
            layers.append(int(m.group(1)))
    return sorted(layers)


_SWEEP: Dict = {}


def _sweep_layer(task: Tuple[str, str, int, List[str]]) -> List[Dict]:
    """Every method and grid at one layer of one source (runs in a worker)."""
    kind, source, layer, methods = task
    src, split = _SWEEP["src"], _SWEEP["split"]
    tr_mask = split["split"].values == "train"
    train_f = split.loc[tr_mask, "folder_id"].values
    test_f = split.loc[~tr_mask, "folder_id"].values
    test_hp = split.loc[~tr_mask, "has_test_partner"].values.astype(bool)
    rows = []
    t0 = time.time()
    for method in methods:
        cfg = Config(kind, source, layer, method)
        emb = src.embeddings(cfg)
        tr_raw = emb[tr_mask]
        for grid in GRIDS:
            D = 0
            if method.endswith("abtt_optimal"):
                D, _ = select_D(tr_raw, train_f, grid)
            tr, te = process(emb, split, method, D)
            trs, tes = tr @ tr.T, te @ te.T
            tsims, tlab = C.pair_arrays(trs, train_f)
            tau = C.fit_tau(tsims, tlab, grid=grid)
            tb = C.taskb_units(tes, test_f, test_hp, tau)
            s, lab = C.pair_arrays(tes, test_f)
            rows.append({
                "kind": kind, "source": source, "layer": layer, "method": method,
                "grid": grid, "D": D, "tau": tau,
                "train_aucroc": float(C.weighted_auroc(
                    C.rank_index(tsims, tlab), np.ones(len(tsims)))[0]),
                "train_dir_acc_at_1": C.train_dir_acc_at_1(trs, train_f, tau),
                "aucroc": float(C.weighted_auroc(C.rank_index(s, lab), np.ones(len(s)))[0]),
                "gap": float(s[lab].mean() - s[~lab].mean()),
                "overall_assignment_acc": float(tb.assign_correct.mean()),
                "dir_acc_at_1": float(tb.dir1_correct.mean()),
            })
    print(f"  {kind}:{source} L{layer} ({time.time() - t0:.1f}s)", flush=True)
    return rows


def cmd_sweep(args: argparse.Namespace) -> None:
    t_start = time.time()
    cpu_start = time.process_time()
    data_root = Path(args.data_root).resolve()
    out_dir = data_root / args.out_dir
    out_csv = out_dir / "sweep_per_layer.csv"
    if out_csv.exists():
        raise SystemExit(f"{out_csv} exists; results are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(data_root / SPLIT_CSV)
    src = Sources(data_root, split)

    jobs: List[Tuple[str, str, List[str]]] = []  # (kind, source, methods)
    for model_id, _ in MODELS:
        jobs.append(("zs", model_id, ["baseline", "center", "abtt_optimal",
                                      "sif_only", "sif_abtt_optimal"]))
    for disp, _, _, _ in FINETUNED:
        jobs.append(("ft", disp, ["baseline", "center", "abtt_optimal"]))
    if args.only_models:
        keep = set(args.only_models.split("|"))
        jobs = [j for j in jobs if DISPLAY.get(j[1], j[1]) in keep]

    tasks = []
    for kind, source, methods in jobs:
        probe = src.emb_path(Config(kind, source, 1, "baseline")).parent
        layers = discover_layers(probe, "")
        print(f"{kind}:{source} layers {layers[0]}..{layers[-1]}", flush=True)
        tasks += [(kind, source, layer, methods) for layer in layers]
    _SWEEP.update(src=src, split=split)
    t0 = time.time()
    if args.procs > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(args.procs) as pool:
            parts = list(pool.imap_unordered(_sweep_layer, tasks))
    else:
        parts = [_sweep_layer(t) for t in tasks]
    rows = [r for part in parts for r in part]
    print(f"swept {len(tasks)} layers in {time.time() - t0:.0f}s", flush=True)
    if not rows:
        raise SystemExit("nothing to sweep")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_csv, index=False)

    # ---- apply the selection rule and compare with the published cells ---- #
    results = pd.read_csv(data_root / RESULTS_CSV)
    cells = headline_cells(data_root, results)
    published = {(c.task, c.kind, c.config.source, c.config.method): c for c in cells
                 if c.kind != "lex"}
    sel_rows = []
    keycols = ["kind", "source", "method", "grid"]
    for (kind, source, method, grid), sub in sweep.groupby(keycols, sort=False):
        sub = sub.sort_values("layer").reset_index(drop=True)
        for task, crit in (("A", "train_aucroc"), ("B", "train_dir_acc_at_1")):
            best = sub.loc[sub[crit].idxmax()]
            pub = published.get((task, kind, source, method))
            r = {"task": task, "kind": kind, "source": source,
                 "row": DISPLAY.get(source, source) + (" (fine-tuned)" if kind == "ft" else ""),
                 "method": method, "setting": SETTING_OF.get(method, method), "grid": grid,
                 "layer": int(best["layer"]), "D": int(best["D"]), "tau": float(best["tau"])}
            for m in TASK_METRICS[task]:
                col, fmt, scale = METRICS[m]
                r[m] = float(best[col])
                r[f"{m}_printed"] = format(scale * float(best[col]), fmt)
                if pub is not None:
                    r[f"{m}_published"] = format(scale * pub.reference[m], fmt)
            if pub is not None:
                r["published_layer"] = pub.config.layer
                r["published_D"] = pub.reference.get("D")
                r["same_layer"] = r["layer"] == pub.config.layer
                r["same_D"] = (not method.endswith("abtt_optimal")
                               or r["D"] == int(pub.reference.get("D")))
                r["moves"] = any(r[f"{m}_printed"] != r[f"{m}_published"]
                                 for m in TASK_METRICS[task])
            sel_rows.append(r)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(out_dir / "sweep_selected.csv", index=False)
    paper = sel[(sel["grid"] == "paper") & sel["published_layer"].notna()]
    bad = paper[paper["moves"].astype(bool) | ~paper["same_layer"].astype(bool)
                | ~paper["same_D"].astype(bool)]
    print(f"paper-grid re-selection: {len(paper)} published cells, {len(bad)} differ")
    if len(bad):
        print(bad.to_string())
    info = run_info(args, t_start, cpu_start)
    info.update(paper_grid_mismatches=int(len(bad)), n_rows=len(sweep))
    (out_dir / "sweep_run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


# --------------------------------------------------------------------------- #
# publish + render
# --------------------------------------------------------------------------- #

PUBLISHED = ["headline_ci.csv", "headline_ci_diffs.csv", "pq_cells.csv",
             "reproduction_cells.csv", "reproduction_configs.csv", "run_info.json",
             "sweep_selected.csv", "sweep_run_info.json"]


def cmd_publish(args: argparse.Namespace) -> None:
    """Copy the small result files from ``runs/`` into the repo."""
    data_root = Path(args.data_root).resolve()
    src_dir = data_root / args.out_dir
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED:
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, PUBLISH_DIR / name)
            print(f"published {name}")
    if args.sweep_dir:
        for name in ("sweep_selected.csv", "sweep_run_info.json"):
            p = data_root / args.sweep_dir / name
            if p.exists():
                shutil.copy2(p, PUBLISH_DIR / name)
                print(f"published {name} from {args.sweep_dir}")


def cmd_render(args: argparse.Namespace) -> None:
    import ci_pq_render as R

    R.render_all(Path(args.in_dir), Path(args.table_dir))


# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("compute")
    p.add_argument("--data_root", default=str(REPO))
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--seed", type=int, default=233)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--procs", type=int, default=1,
                   help="Forked worker processes for the bootstrap.")
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--only_models", default="",
                   help="'|'-separated display names, e.g. 'LaBSE|TF-IDF char 3-5' (pilots).")
    p.add_argument("--allow_mismatch", action="store_true")
    p.set_defaults(func=cmd_compute)

    p = sub.add_parser("sweep")
    p.add_argument("--data_root", default=str(REPO))
    p.add_argument("--out_dir", default=DEFAULT_OUT + "_sweep")
    p.add_argument("--only_models", default="")
    p.add_argument("--procs", type=int, default=1)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("publish")
    p.add_argument("--data_root", default=str(REPO))
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--sweep_dir", default=DEFAULT_OUT + "_sweep")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("render")
    p.add_argument("--in_dir", default=str(PUBLISH_DIR))
    p.add_argument("--table_dir", default=str(TABLE_DIR))
    p.set_defaults(func=cmd_render)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
