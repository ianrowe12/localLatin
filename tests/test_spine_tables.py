"""The spine tables (scripts/paper/reframe/spine_tables.py): T1 tab:models, T2 tab:predictions,
T3 tab:headline and the appendix tab:all_models.

The tables are regenerated from the committed CSVs and compared byte for byte with the
committed .tex files, and the numbers the captions and the prose rely on are pinned against
the CSVs. Tests that need ``runs/`` skip when it is absent.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "paper" / "reframe"))

import spine_tables as S  # noqa: E402

TABLES = REPO / "overleaf_drafts" / "tables"
NEEDED = [S.P2X2_CSV, S.D2_CSV, S.RES_CSV, S.GEO_CSV, S.KSWEEP_CSV, S.DABL_CSV, S.SPLIT3_CSV,
          S.TOKABL_CSV, S.AUDIT_CSV, S.CI_DIR / "headline_ci.csv", S.CI_DIR / "run_info.json"]
needs_data = pytest.mark.skipif(not all((REPO / p).exists() for p in NEEDED),
                                reason="committed result CSVs not present")


def test_ranges():
    assert S._ranges([]) == "none"
    assert S._ranges([5]) == "5"
    assert S._ranges([2, 3, 4, 7, 9, 10]) == "2--4, 7, 9--10"


def test_interval_format():
    assert S._int(0.90958, 0.96277, "auroc") == "[.910,.963]"
    assert S._int(0.6838, 0.7582, "dir1") == "[68.4,75.8]"


@needs_data
@pytest.mark.parametrize("name", ["models_main.tex", "all_models.tex", "predictions.tex",
                                  "headline_main.tex"])
def test_tables_match_committed(name):
    out = S.render_all(REPO)
    assert (TABLES / name).read_text() == out[name]


@needs_data
def test_model_summary():
    summ = S.summarize(S.per_layer(REPO))
    assert list(summ.index) == list(S.ALL_MODELS)
    collapsing = {m for m in S.ALL_MODELS if summ.loc[m, "n_collapsed"] > 0}
    assert collapsing == {"LaTa", "PhilTa", "mT5-base", "T5-v1.1-base"}
    assert collapsing == {m for m in S.ALL_MODELS if S.MODELS[m][3] == "v11"}
    assert int(summ["n_collapsed"].sum()) == 36
    assert summ.loc[["LaTa", "PhilTa", "mT5-base"], "n_collapsed"].sum() == 26
    assert round(float(summ["pc1_min_collapsed"].min()), 3) == 0.764
    exp = {"LaTa": (0.496, 6), "PhilTa": (0.538, 10), "mT5-base": (0.654, 5),
           "T5-v1.1-base": (0.489, 2), "T5-base": (0.816, 11), "LaBERTa": (0.826, 1),
           "PhilBERTa": (0.883, 6), "LaBSE": (0.806, 1), "T5-efficient-base": (0.736, 11)}
    for m, (auc, layer) in exp.items():
        assert f"{summ.loc[m, 'auroc_min']:.3f}" == f"{auc:.3f}"
        assert summ.loc[m, "auroc_min_layer"] == layer
    assert summ.loc["Qwen3-0.6B", "n_layers"] == 28 and summ.loc["KaLM-mini", "n_layers"] == 24
    # D2: T5-efficient-base (original layout, C4 only) does not collapse and stays below 0.76
    assert len(S.ALL_MODELS) == 13
    assert summ.loc["T5-efficient-base", "n_collapsed"] == 0
    assert summ.loc["T5-efficient-base", "n_high_pc1"] == 0
    assert f"{summ.loc['T5-efficient-base', 'pc1_max']:.3f}" == "0.627"


@needs_data
def test_prediction_numbers():
    x = S.prediction_numbers(REPO)
    assert x["n_collapsed"] == 26
    assert x["zero_repaired"] == 0 and f"{x['zero_best']:.3f}" == "0.794"
    assert x["rand_repaired"] == 0 and x["rand_k10_shift"] < 0.001
    assert x["center_repaired"] == 0 and f"{x['center_median']:.3f}" == "0.503"
    assert round(100 * x["d1_share_median"]) == 45 and x["d1_share_80"] == 1
    assert x["d3_n_091"] == 26 and x["d3_min"] >= 0.91
    assert f"{x['removed_median']:.3f}" == "0.489" and f"{x['retained_median']:.3f}" == "0.977"
    assert x["removed_below"] == 26
    assert (x["lata_restored"], x["lata_n"]) == (10, 10)
    assert f"{x['lata_median']:.3f}" == "0.919" and f"{x['lata_ctrl_median']:.3f}" == "0.500"
    assert x["mt5_restored"] == 0 and x["mt5_max_m"] == 100
    assert x["len_n05"] == 0


@needs_data
def test_headline_spread_and_reference():
    ci = pd.read_csv(REPO / S.CI_DIR / "headline_ci.csv")
    x = S.headline_numbers(ci)
    assert x["dir1_Base_spread"] == 39.3 and x["dir1_ABTT_spread"] == 3.3
    assert (x["dir1_Base_min"], x["dir1_Base_max"]) == (46.6, 85.9)
    assert (x["dir1_ABTT_min"], x["dir1_ABTT_max"]) == (86.1, 89.4)
    assert (x["lex_auroc"], x["lex_dir1"]) == (0.987, 89.9)


@needs_data
def test_sources_must_agree(tmp_path):
    for p in (S.P2X2_CSV, S.D2_CSV, S.RES_CSV, S.GEO_CSV):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, tmp_path / p)
    S.per_layer(tmp_path)  # unchanged copies agree
    p2 = pd.read_csv(tmp_path / S.P2X2_CSV)
    p2.loc[(p2["model"] == "LaTa") & (p2["layer"] == 6), "aucroc"] += 1e-4
    p2.to_csv(tmp_path / S.P2X2_CSV, index=False)
    with pytest.raises(SystemExit, match="disagree"):
        S.per_layer(tmp_path)


@needs_data
def test_high_share_column_uses_the_collapse_floor():
    """One top-PC threshold in the paper (0.76): every collapsed layer reaches it, and the
    appendix column counts the layers at or above it, so the non-collapsed high-share layers
    (mT5-base layer 4, Sentence-T5) show up as the gap between the two columns."""
    assert S.HIGH_PC1 == 0.76
    summ = S.summarize(S.per_layer(REPO))
    assert float(summ["pc1_min_collapsed"].min()) >= S.HIGH_PC1
    assert summ.loc["mT5-base", "high_pc1_layers"] == "4--11"
    assert summ.loc["Sentence-T5", "high_pc1_layers"] == "2--9"
    assert summ.loc["Sentence-T5", "n_collapsed"] == 0
    tex = (TABLES / "all_models.tex").read_text()
    assert r"PC1$\ge$0.76" in tex and "0.6 " not in tex and "0.6." not in tex
    for name in ("models_main.tex", "all_models.tex", "predictions.tex", "headline_main.tex"):
        text = (TABLES / name).read_text()
        assert "centred" not in text and "Centre" not in text, name


@needs_data
def test_main_text_layout_of_t2_and_t3():
    """Layout review (2 October): T2 is a one-column table that keeps all its numbers; T3
    prints intervals for the six panel encoders and the n-gram reference only (the fine-tuned
    intervals are in tab:headline_ci)."""
    out = S.render_all(REPO)
    t2 = out["predictions.tex"]
    assert r"\begin{table}[t]" in t2 and "table*" not in t2 and r"\textwidth" not in t2
    for num in ("0.794", "0.0004", "400", "0.503", "0.541", "45\\%", "1/26", "0.914",
                "0.934", "0.978", "0.489", "0.977", "11\\%", "10/10", "0.919", "0.500",
                "0/7", "0.748", "0.44"):
        assert num in t2, num
    assert t2.count(r"\textbf{failed}") == 4
    t3 = out["headline_main.tex"]
    assert t3.count(r"& {\scriptsize [") == 6 * 4  # six panel rows, four cells each
    assert t3.count(r"\scriptsize [") == 6 * 4 + 2  # plus the n-gram row
    assert "Task" not in t3 and "$n$-gram" not in t3
