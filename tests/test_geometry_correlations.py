"""Guards for the geometry-correlation table of Appendix C (appendix trim B8).

The table correlates each per-layer statistic with baseline test ranking AUROC over all
100 panel layers and within the 26 collapsed and 74 healthy layers. The prose quotes the
within-collapsed and healthy top-PC values, so the test pins those cells and that the
committed table is exactly what the generator writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "paper" / "reframe"))

# The generator draws two figures with matplotlib, which the CI requirement
# files deliberately leave out; skip the module there, as the figure tests do.
pytest.importorskip("matplotlib")

import geometry_vs_retrieval as gvr  # noqa: E402

needs_inputs = pytest.mark.skipif(
    not ((REPO / gvr.GEOM_CSV).exists() and (REPO / gvr.RES_CSV).exists()),
    reason="geometry_per_layer.csv / phase_resubmit_results.csv not present")


@needs_inputs
def test_correlation_table_rerenders_and_carries_the_quoted_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    gvr.main(["--tab_dir", str(tmp_path / "tab"), "--fig_dir", str(tmp_path / "fig"),
              "--facts_md", str(tmp_path / "facts.md")])
    tex = (tmp_path / "tab" / "geometry_correlations.tex").read_text()
    committed = REPO / "overleaf_drafts" / "tables" / "geometry_correlations.tex"
    assert tex == committed.read_text()
    assert r"\textbf{Collapsed} (26)" in tex and r"\textbf{Healthy} (74)" in tex
    assert r"\textbf{T5}" not in tex and "Non-T5" not in tex  # no T5 / non-T5 split
    assert "ABTT AUROC gain" not in tex
    row = next(ln for ln in tex.splitlines() if ln.startswith("Top-PC share & "))
    cells = [c.strip().strip("$").replace(r"\phantom{-}", "+") for c in
             row.rstrip(" \\").split(" & ")[1:]]
    # all r, all rho, collapsed r, collapsed rho, healthy r, healthy rho, LOMO min, max
    assert cells[1] == "-0.77" and cells[3] == "+0.24" and cells[5] == "-0.49"
    gap = next(ln for ln in tex.splitlines() if ln.startswith("Raw cosine gap & "))
    assert gap.split(" & ")[2].strip("$").replace(r"\phantom{-}", "+") == "+0.22"
    assert r"$\rho=+0.24$ ($p=0.23$)" in tex
    for gone in ("tab:geometry_regimes", "tab:layer_diagnostics_main", "fig:gap",
                 "fig:standardized_gap"):
        assert gone not in tex


@needs_inputs
def test_collapsed_subset_is_the_26_raw_t5_family_layers(monkeypatch):
    monkeypatch.chdir(REPO)
    df, _, _ = gvr.load()
    col = df[df.aucroc < gvr.COLLAPSE_AUROC]
    assert len(col) == 26 and set(col.m) == {"LaTa", "PhilTa", "mT5-base"}
