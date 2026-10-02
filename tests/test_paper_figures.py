"""Guards for scripts/paper/reframe/paper_figures.py (main-text figures F1-F3).

Synthetic frames for the data functions, plus the committed result CSVs when they are
checked out: the numbers the paper quotes from each figure must be what the figure plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "paper" / "reframe"))

import paper_figures as pf  # noqa: E402


def _res(rows):
    return pd.DataFrame(rows, columns=["model", "layer", "method", "aucroc"])


def test_collapsed_layers_uses_baseline_below_cutoff():
    res = _res([("bowphs/LaTa", 1, "baseline", 0.93), ("bowphs/LaTa", 2, "baseline", 0.55),
                ("bowphs/LaTa", 2, "abtt_optimal", 0.97), ("bowphs/LaTa", 3, "baseline", 0.70)])
    c = pf.collapsed_layers(res)
    assert c.to_dict("records") == [{"model": "bowphs/LaTa", "layer": 2}]


def test_depth_data_pairs_baseline_and_abtt():
    res = _res([("bowphs/LaTa", 1, "baseline", 0.9), ("bowphs/LaTa", 1, "abtt_optimal", 0.97),
                ("x/other", 1, "baseline", 0.5), ("x/other", 1, "abtt_optimal", 0.6)])
    d = pf.depth_data(res)
    assert len(d) == 1 and d.m.iloc[0] == "LaTa"
    assert d.aucroc_base.iloc[0] == 0.9 and d.aucroc_abtt.iloc[0] == 0.97


def test_zeroing_bands_seed_mean_then_layer_band():
    coll = pd.DataFrame({"model": ["m", "m"], "layer": [1, 2]})
    rows = []
    for layer, base in ((1, 0.5), (2, 0.6)):
        rows.append(("m", layer, "mean_abs", -1, 10, base + 0.1))
        rows.append(("m", layer, "random", 0, 10, base))
        rows.append(("m", layer, "random", 1, 10, base + 0.02))
        rows.append(("m", layer, "mean_abs", -1, 500, 0.99))  # beyond K_MAX, dropped
    sweep = pd.DataFrame(rows, columns=["model", "layer", "ranking", "seed", "k", "aucroc"])
    zb = pf.zeroing_bands(sweep, coll)
    r = zb["ranked"]
    assert list(r.k) == [10]
    assert r["min"].iloc[0] == pytest.approx(0.6) and r["max"].iloc[0] == pytest.approx(0.7)
    q = zb["random"]
    assert q["min"].iloc[0] == pytest.approx(0.51) and q["max"].iloc[0] == pytest.approx(0.61)


def test_components_band_keeps_center_and_grid():
    coll = pd.DataFrame({"model": ["m"], "layer": [1]})
    h1 = pd.DataFrame({"model": ["m"] * 4, "layer": [1] * 4,
                       "variant": ["raw", "center", "abtt", "abtt"], "D": [-1, 0, 3, 15],
                       "aucroc": [0.5, 0.49, 0.95, 0.97]})
    b = pf.components_band(h1, coll)
    assert list(b.D) == [0, 3]


# ---- the committed CSVs: numbers quoted in the paper ----------------------------------

def _need(*paths):
    for p in paths:
        if not (REPO_ROOT / p).exists():
            pytest.skip(f"{p} not checked out")


@pytest.fixture
def res(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    _need(pf.RES_CSV)
    return pd.read_csv(pf.RES_CSV)


def test_real_depth_band(res):
    d = pf.depth_data(res)
    assert len(d) == 100
    assert round(d.aucroc_abtt.min(), 3) == 0.962 and round(d.aucroc_abtt.max(), 3) == 0.987


def test_real_collapsed_count(res):
    c = pf.collapsed_layers(res)
    assert len(c) == 26
    assert c.groupby("model").size().to_dict() == {
        "bowphs/LaTa": 10, "bowphs/PhilTa": 9, "google/mt5-base": 7}


def test_real_localize_numbers(res):
    _need(pf.KSWEEP_CSV, pf.H1_CSV)
    coll = pf.collapsed_layers(res)
    zb = pf.zeroing_bands(pd.read_csv(pf.KSWEEP_CSV), coll)
    r = zb["ranked"]
    assert round(r[r.k <= 10]["max"].max(), 2) == 0.79  # best AUROC with ten zeroed
    assert r[r.k <= 10]["max"].max() < pf.REPAIR_AUROC  # none repaired
    cb = pf.components_band(pd.read_csv(pf.H1_CSV), coll)
    assert cb[cb.D == 3]["min"].min() >= 0.91  # three components: every layer >= 0.91


def test_real_diagnostics_ranges(res):
    _need(pf.GEOM_CSV, pf.P2X2_CSV)
    d = pf.diagnostics_data(res, pd.read_csv(pf.GEOM_CSV), pd.read_csv(pf.P2X2_CSV))
    assert len(d) == 124  # 100 panel layers + 12 + 12
    panel_coll = d[d.m.isin(pf.T5_COLLAPSING) & (d.aucroc < pf.COLLAPSE_AUROC)]
    assert round(panel_coll.pc1.min(), 2) == 0.76
    assert (round(panel_coll.mean_cos.min(), 2), round(panel_coll.mean_cos.max(), 2)) == (0.22, 0.58)
    emb = d[d.m.isin(["LaBSE", "Qwen3-0.6B", "KaLM-mini"])]
    # 0.5855 to 0.9737: "0.59-0.97" at two decimals (the spine draft said 0.58)
    assert (round(emb.mean_cos.min(), 3), round(emb.mean_cos.max(), 3)) == (0.585, 0.974)
    v11 = d[(d.m == "T5-v1.1-base") & (d.aucroc < pf.COLLAPSE_AUROC)]
    assert len(v11) == 10
    assert (round(v11.mean_cos.min(), 2), round(v11.mean_cos.max(), 2)) == (0.87, 0.96)
    assert v11.pc1.min() >= 0.76
    assert d[d.m == "T5-base"].aucroc.min() >= 0.80
