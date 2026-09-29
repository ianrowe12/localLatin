"""tab:layer_diagnostics_main is generated, not hand-written (issue #235 item 17)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "paper" / "reframe"))

import layer_diagnostics_table as ldt  # noqa: E402

GEOM = REPO_ROOT / "runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv"
RULES = REPO_ROOT / "runs/active/resubmit/layer_diagnostics/layer_rule_candidates.csv"
TABLE = REPO_ROOT / "overleaf_drafts/tables/layer_diagnostics_main.tex"


def _geom_rows():
    rows = []
    for model, _ in ldt.MODELS:
        for split in ("train", "test"):
            for layer in range(1, 13):
                # Peak at layer 5 on test, layer 4 on train; LaTa's band 3--11 spans 0.5--0.9.
                pc1 = 0.9 if layer == (5 if split == "test" else 4) else 0.5 + layer / 100
                for view, scale in (("raw", 1.0), ("abtt_d10", 0.05)):
                    rows.append({"model": model, "split": split, "view": view,
                                 "repr": "hidden", "pooling": "mean", "layer": layer,
                                 "pc1_variance_ratio": pc1 * scale,
                                 "effective_rank_entropy": 2.0 if view == "raw" else 150.0})
    return pd.DataFrame(rows)


def test_peak_is_the_test_split_argmax_and_rows_follow_it():
    rules = pd.DataFrame({"model": [m for m, _ in ldt.MODELS],
                          "recommended_operational_layer": [7, 1, 1]})
    tex = ldt.render(_geom_rows(), rules)
    assert r"LaTa & 7 & 5 & 0.900 & 0.045 & 2.00$\rightarrow$150.00 \\" in tex
    assert "where LaTa peaks at layer 4 instead of 5" in tex
    assert tex.splitlines()[0] == "% generated table"
    assert r"\label{tab:layer_diagnostics_main}" in tex


def test_committed_table_is_the_generator_output():
    for path in (GEOM, RULES, TABLE):
        if not path.exists():
            pytest.skip(f"{path} not checked out")
    tex = ldt.render(pd.read_csv(GEOM), pd.read_csv(RULES))
    assert tex == TABLE.read_text()
