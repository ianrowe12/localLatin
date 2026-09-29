"""Issue #233 (CI + PQ): the vectorized evaluator, the bootstrap and the tables.

The fast path in ``scripts/paper/reframe/ci_pq_core.py`` must reproduce the
paper's evaluator (``scripts/resubmit/run_resubmit_evaluate.py``) exactly, and
the directory bootstrap's pair weights must equal what an explicit resample
with duplicated directories gives. Both are checked here on small random
fixtures. The tables are checked by regenerating them from the committed CSVs
and comparing byte for byte. Tests that need ``runs/`` skip when it is absent.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "resubmit", REPO / "scripts" / "paper" / "reframe"):
    sys.path.insert(0, str(p))

import ci_pq_core as C  # noqa: E402
import ci_pq_render as R  # noqa: E402

sklearn_metrics = pytest.importorskip("sklearn.metrics")
E = pytest.importorskip("run_resubmit_evaluate")

PUBLISHED = REPO / "docs/research/data/reframe_ci_pq"
TABLES = REPO / "overleaf_drafts/tables"


def _fixture(seed: int, n: int = 90, n_dirs: int = 50, dim: int = 12):
    rng = np.random.default_rng(seed)
    fids = np.array([f"d{rng.integers(0, n_dirs):03d}" for _ in range(n)])
    X = rng.normal(size=(n, dim)).astype(np.float32)
    for d in np.unique(fids):
        X[fids == d] += 1.5 * rng.normal(size=dim).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T, fids


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fit_tau_paper_grid_matches_evaluator(seed):
    sim, fids = _fixture(seed)
    assert C.fit_tau(*C.pair_arrays(sim, fids), grid="paper") == E.learn_tau_from_similarity(sim, fids)


@pytest.mark.parametrize("seed", [0, 1])
def test_exact_cut_is_at_least_as_good_as_any_grid(seed):
    sim, fids = _fixture(seed)
    s, lab = C.pair_arrays(sim, fids)

    def f1(t):
        pred = s >= np.float32(t)
        tp = (pred & lab).sum()
        return 2 * tp / (pred.sum() + lab.sum())

    best = f1(C.fit_tau(s, lab, "exact"))
    assert best >= f1(C.fit_tau(s, lab, "fine")) - 1e-12
    assert best >= f1(C.fit_tau(s, lab, "paper")) - 1e-12


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_taskb_units_match_evaluator(seed):
    sim, fids = _fixture(seed)
    hp = E.has_partner_flags(fids)
    for tau in (E.learn_tau_from_similarity(sim, fids), 0.2, 0.95):
        u = C.taskb_units(sim, fids, hp, tau)
        assert u.dir1_correct.mean() == E.directory_assignment_accuracy_at_k(sim, fids, hp, tau, 1)
        assert u.assign_correct.mean() == E.compute_assignment_acc(sim, hp, tau)[2]
        a, d = C.at_threshold(u, tau)
        assert (a == u.assign_correct).all() and (d == u.dir1_correct).all()


def test_weighted_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(3)
    s = np.round(rng.normal(size=400), 1)  # heavy ties
    y = rng.random(400) < 0.3
    w = rng.integers(0, 4, size=400).astype(float)
    ri = C.rank_index(s, y)
    assert C.weighted_auroc(ri, np.ones(400))[0] == pytest.approx(
        sklearn_metrics.roc_auc_score(y, s), abs=1e-12)
    assert C.weighted_auroc(ri, w)[0] == pytest.approx(
        sklearn_metrics.roc_auc_score(y, s, sample_weight=w), abs=1e-12)


def test_directory_bootstrap_equals_explicit_resample():
    """Pair weights w_d (within) and w_d * w_e (across) = duplicating directories."""
    sim, fids = _fixture(4, n=40, n_dirs=18)
    labels, code = C.directory_codes(fids)
    counts = C.bootstrap_counts(len(labels), 4, seed=7)
    pdirs = C.PairDirs.from_codes(code)
    s, lab = C.pair_arrays(sim, fids)
    got = C.weighted_auroc(C.rank_index(s, lab), pdirs.weights(counts))
    for r in range(4):
        rows, copy_of = [], []
        for d, c in enumerate(counts[r].astype(int)):
            for k in range(c):
                for i in np.flatnonzero(code == d):
                    rows.append(i)
                    copy_of.append((d, k))
        ss, ll = [], []
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                if copy_of[a][0] == copy_of[b][0] and copy_of[a][1] != copy_of[b][1]:
                    continue  # a file is never paired with a copy of its directory
                ss.append(sim[rows[a], rows[b]])
                ll.append(code[rows[a]] == code[rows[b]])
        assert got[r] == pytest.approx(sklearn_metrics.roc_auc_score(ll, ss), abs=1e-12)


def test_bootstrap_counts_are_seeded_and_sum_to_n():
    a = C.bootstrap_counts(514, 5, seed=233)
    b = C.bootstrap_counts(514, 5, seed=233)
    assert (a == b).all() and (a.sum(axis=1) == 514).all()


def test_oracle_accuracy_matches_brute_force():
    sim, fids = _fixture(5)
    hp = E.has_partner_flags(fids)
    u = C.taskb_units(sim, fids, hp, 0.5)
    new = (~u.existing).astype(float)
    acc, cut = C.oracle_accuracy(u.max_cos, u.existing.astype(float), new)
    brute = max(np.mean(np.where(hp, u.max_cos >= t, u.max_cos < t))
                for t in np.r_[np.unique(u.max_cos), np.inf])
    assert acc[0] == pytest.approx(brute)
    assert np.mean(np.where(hp, u.max_cos >= cut[0], u.max_cos < cut[0])) == pytest.approx(brute)


def test_k_occurrence_counts_every_neighbour_once():
    sim, _ = _fixture(6)
    nk = C.k_occurrence(sim, k=10)
    assert nk.sum() == 10 * sim.shape[0]
    assert C.skewness(np.array([1.0, 1.0, 1.0, 10.0])) > 0


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

needs_published = pytest.mark.skipif(
    not (PUBLISHED / "headline_ci.csv").exists(), reason="published CI CSVs absent")


@needs_published
def test_tables_regenerate_byte_identical(tmp_path):
    out = R.render_all(PUBLISHED, tmp_path, head_dir=TABLES)
    for name, text in out.items():
        committed = TABLES / name
        assert committed.exists(), name
        assert committed.read_text() == text, f"{name} is stale; run ci_pq.py render"


@needs_published
def test_compact_table_refuses_a_stale_csv(tmp_path):
    ci = pd.read_csv(PUBLISHED / "headline_ci.csv")
    ci.loc[(ci["row"] == "LaTa") & (ci["setting"] == "Base") & (ci["metric"] == "auroc"),
           "estimate"] += 0.01
    for name in ("headline_ci_diffs.csv", "pq_cells.csv", "run_info.json",
                 "sweep_selected.csv"):
        if (PUBLISHED / name).exists():
            shutil.copy(PUBLISHED / name, tmp_path / name)
    ci.to_csv(tmp_path / "headline_ci.csv", index=False)
    with pytest.raises(SystemExit, match="stale"):
        R.render_all(tmp_path, tmp_path / "out", head_dir=TABLES)


@needs_published
def test_compact_table_handles_lexical_rows():
    """``build_headline_tables.py --lexical_csv`` adds rows spanning each block."""
    ci = pd.read_csv(PUBLISHED / "headline_ci.csv")
    info = {"B": 10000, "seed": 233, "n_test_dirs": 514}
    tf = ci[(ci["row"] == "TF-IDF char 3-5") & (ci["task"] == "B")].set_index("metric")
    a = format(100 * tf.loc["assign", "estimate"], ".1f")
    d = format(100 * tf.loc["dir1", "estimate"], ".1f")
    row = f"TF-IDF char 3--5 & \\multicolumn{{4}}{{c}}{{{a}}} & \\multicolumn{{4}}{{c}}{{{d}}} \\\\"
    head = (TABLES / "taskB_headline.tex").read_text().replace(
        "\\bottomrule", row + "\n\\bottomrule", 1)
    out = R.render_compact(head, ci, "B", info).splitlines()
    k = out.index(row)
    assert out[k + 1].count("\\multicolumn{4}{c}{{\\scriptsize [") == 2


@needs_published
def test_published_labels():
    """D only where fitted; the replicate-max rows are labelled as unused."""
    import ci_pq

    ci = pd.read_csv(PUBLISHED / "headline_ci.csv")
    assert ci.loc[ci["setting"].isin(["Base", "SIF"]), "D"].isna().all()
    assert ci.loc[ci["setting"].isin(["ABTT", "SIF+ABTT"]), "D"].notna().all()
    diffs = pd.read_csv(PUBLISHED / "headline_ci_diffs.csv")
    assert "tfidf_minus_best_abtt" not in set(diffs["group"])
    rm = diffs[diffs["group"] == ci_pq.MAX_OVER_MODELS_GROUP]
    assert len(rm) and (rm["note"] == ci_pq.MAX_OVER_MODELS_NOTE).all()
    raw = pd.read_csv(PUBLISHED / "headline_ci.csv", dtype=str, keep_default_na=False)
    again = ci_pq.normalize_published("headline_ci.csv", raw)
    # idempotent: labels only, text preserved
    assert again.to_csv(index=False) == raw.to_csv(index=False)


@needs_published
def test_every_printed_cell_was_reproduced():
    checks = pd.read_csv(PUBLISHED / "reproduction_cells.csv")
    configs = pd.read_csv(PUBLISHED / "reproduction_configs.csv")
    assert checks["match"].all()
    assert configs["all_match"].all()
    # 30 zero-shot + 6 fine-tuned + 3 lexical cells per task, two metrics each
    assert len(checks) == 2 * 2 * (24 + 6 + 3)


RUNS_INPUTS = [
    "runs/active/resubmit/results/phase_resubmit_results.csv",
    "runs/active/resubmit/results/lexical_baselines.csv",
    "runs/active/resubmit/results/finetune/finetune_lata_ceiling_comparison.csv",
    "runs/active/resubmit/results/finetune/finetune_qwen3_0.6b_ceiling_comparison.csv",
    "runs/active/resubmit/results/finetune/finetune_kalm_mini_ceiling_comparison.csv",
]


@pytest.mark.skipif(not all((REPO / p).exists() for p in RUNS_INPUTS),
                    reason="runs/ inputs absent")
def test_headline_cells_use_the_published_layers():
    import ci_pq

    results = pd.read_csv(REPO / "runs/active/resubmit/results/phase_resubmit_results.csv")
    cells = ci_pq.headline_cells(REPO, results)
    text = (TABLES / "selected_layers.tex").read_text()
    for c in cells:
        if c.kind != "zs":
            continue
        line = next(ln for ln in text.splitlines() if ln.startswith(c.row + " &"))
        layers = [x.strip() for x in line.rstrip("\\ ").split("&")[1:]]
        k = [s for _, s in ci_pq.SETTINGS].index(c.setting) + (0 if c.task == "A" else 4)
        assert int(layers[k]) == c.config.layer
