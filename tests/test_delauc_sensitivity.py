"""Guards for the issue #195 deletion-gap sensitivity sweep.

The sweep's whole claim is that it recomputes the *same* metric the published
run computed, with one knob moved at a time. That claim rests on three things,
and each has a test here.

1. At the predeclared setting the sweep's metric must be numerically identical
   to ``attribution_metrics.deletion_auc`` on the hidden backend, including the
   random-order reference: the reference is what makes the metric
   chance-corrected, and it is drawn from a seeded stream that the sweep must
   not perturb.
2. The knobs must actually be the knobs they are named after. The step
   schedule has to cover the endpoints whatever fraction it is given, and the
   ``zero`` erasure has to be a pure rescaling of the ``drop`` erasure, which
   is invisible to a cosine under the baseline variant and visible under ABTT.
   That asymmetry is the memo's main finding about the erasure knob, so it is
   pinned here rather than left to the run.
3. The table generator must render what the sweep wrote, including the
   win/tie/loss counts, and must not silently drop a configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

from attribution_metrics import PairContext, deletion_auc  # noqa: E402
from run_attribution_metrics import HiddenPairEvaluator  # noqa: E402
import build_delauc_sensitivity_table as tablegen  # noqa: E402
from run_delauc_sensitivity import (  # noqa: E402
    PREDECLARED,
    DelAucConfig,
    PairEvaluator,
    build_configs,
    del_auc_gap,
    deletion_grid,
    paired_cells,
    summarise_configs,
    verdict,
)

DIM = 16


def synthetic_pair(seed: int, n_q: int = 23, n_c: int = 17):
    """A pair with enough structure that the deletion curve is not flat."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n_q, DIM))
    # A shared direction, so the full-query cosine clears FULL_COS_FLOOR.
    shared = rng.normal(size=DIM)
    q += 0.8 * shared
    c = rng.normal(size=(n_c, DIM)) + 0.8 * shared
    pcs = np.linalg.qr(rng.normal(size=(DIM, 3)))[0].T
    mean_vec = rng.normal(size=DIM) * 0.1
    scores = rng.normal(size=n_q)
    return q, c, pcs, mean_vec, scores


@pytest.mark.parametrize("variant", ["baseline", "abtt"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_predeclared_matches_production_metric(variant, seed):
    """The predeclared configuration is the published metric, to float noise."""
    q, c, pcs, mean_vec, scores = synthetic_pair(seed)

    reference = HiddenPairEvaluator(q, c, pcs, mean_vec, variant)
    ctx: PairContext = reference.context()
    expected = deletion_auc(ctx, scores)

    got = del_auc_gap(
        PairEvaluator(q, c, pcs, mean_vec, variant, PREDECLARED.erasure, PREDECLARED.side),
        scores, None, PREDECLARED,
    )

    assert got["del_auc"] == pytest.approx(expected["del_auc"], abs=1e-12)
    assert got["del_auc_random"] == pytest.approx(expected["del_auc_random"], abs=1e-12)
    assert got["del_auc_gap"] == pytest.approx(expected["del_auc_gap"], abs=1e-12)


def test_random_reference_is_the_same_stream_at_more_draws():
    """Raising the draw count extends the reference, it does not replace it.

    A sweep in which ``draws=20`` used different first five orderings than
    ``draws=5`` would confound the Monte-Carlo size with the draw identity, and
    the knob would no longer be the knob.
    """
    q, c, pcs, mean_vec, scores = synthetic_pair(3)
    ev = PairEvaluator(q, c, pcs, mean_vec, "baseline", "drop", "query")
    five = del_auc_gap(ev, scores, None, PREDECLARED)
    twenty = del_auc_gap(ev, scores, None,
                         DelAucConfig(name="d20", knob="draws", draws=20))
    assert five["del_auc"] == pytest.approx(twenty["del_auc"], abs=1e-12)
    # Same attribution curve, different reference size, so the gap moves only
    # through the reference mean.
    assert five["del_auc_random"] != pytest.approx(twenty["del_auc_random"], abs=1e-9)


@pytest.mark.parametrize("n", [3, 10, 23, 200])
@pytest.mark.parametrize("schedule", ["every_token", "frac_0.05", "frac_0.10", "frac_0.20"])
def test_deletion_grid_spans_the_whole_query(n, schedule):
    ks = deletion_grid(n, schedule)
    assert ks[0] == 0
    assert ks[-1] == n
    assert np.all(np.diff(ks) > 0)
    assert np.all((ks >= 0) & (ks <= n))
    if schedule == "every_token":
        assert len(ks) == n + 1


def test_deletion_grid_rejects_unknown_schedule():
    with pytest.raises(ValueError):
        deletion_grid(10, "sqrt")


def test_centroid_erasure_is_invisible_to_the_abtt_cosine():
    """The mirror image of the ``zero`` identity, and the other half of the
    memo's erasure finding. After centring, replacing a deleted token with the
    corpus mean scales the cleaned vector by (n - k)/n, so ABTT cannot see it
    while the uncleaned baseline can."""
    q, c, pcs, mean_vec, scores = synthetic_pair(7)
    cfg_centroid = DelAucConfig(name="ct", knob="erasure", erasure="centroid")

    abtt_drop = del_auc_gap(PairEvaluator(q, c, pcs, mean_vec, "abtt", "drop", "query"),
                            scores, None, PREDECLARED)
    abtt_centroid = del_auc_gap(
        PairEvaluator(q, c, pcs, mean_vec, "abtt", "centroid", "query"),
        scores, None, cfg_centroid)
    assert abtt_centroid["del_auc_gap"] == pytest.approx(abtt_drop["del_auc_gap"], abs=1e-12)

    base_drop = del_auc_gap(PairEvaluator(q, c, pcs, mean_vec, "baseline", "drop", "query"),
                            scores, None, PREDECLARED)
    base_centroid = del_auc_gap(
        PairEvaluator(q, c, pcs, mean_vec, "baseline", "centroid", "query"),
        scores, None, cfg_centroid)
    assert base_centroid["del_auc_gap"] != pytest.approx(base_drop["del_auc_gap"], abs=1e-6)


def test_zero_erasure_is_invisible_to_the_baseline_cosine():
    """``zero`` rescales each pooled vector by (n - k)/n, and a cosine does not
    see a positive rescaling. Under ABTT the mean subtraction happens before the
    normalisation, so the same rescaling does move the number."""
    q, c, pcs, mean_vec, scores = synthetic_pair(4)
    cfg_drop = PREDECLARED
    cfg_zero = DelAucConfig(name="z", knob="erasure", erasure="zero")

    base_drop = del_auc_gap(PairEvaluator(q, c, pcs, mean_vec, "baseline", "drop", "query"),
                            scores, None, cfg_drop)
    base_zero = del_auc_gap(PairEvaluator(q, c, pcs, mean_vec, "baseline", "zero", "query"),
                            scores, None, cfg_zero)
    assert base_zero["del_auc_gap"] == pytest.approx(base_drop["del_auc_gap"], abs=1e-12)

    abtt_drop = del_auc_gap(PairEvaluator(q, c, pcs, mean_vec, "abtt", "drop", "query"),
                            scores, None, cfg_drop)
    abtt_zero = del_auc_gap(PairEvaluator(q, c, pcs, mean_vec, "abtt", "zero", "query"),
                            scores, None, cfg_zero)
    assert abtt_zero["del_auc_gap"] != pytest.approx(abtt_drop["del_auc_gap"], abs=1e-6)


@pytest.mark.parametrize("erasure", ["drop", "zero"])
@pytest.mark.parametrize("variant", ["baseline", "abtt"])
def test_empty_query_endpoint_is_pinned_to_zero(erasure, variant):
    """Deleting every token leaves no vector, so the curve ends at 0 under both
    variants. Without the pre-cleaning check, ABTT would report a cosine for an
    input with no tokens in it."""
    q, c, pcs, mean_vec, _ = synthetic_pair(5)
    ev = PairEvaluator(q, c, pcs, mean_vec, variant, erasure, "query")
    curve = ev.drop_curve(np.arange(q.shape[0]), np.arange(c.shape[0]),
                          deletion_grid(q.shape[0], "every_token"))
    assert curve[0] == pytest.approx(ev.full_cos, abs=1e-12)
    assert curve[-1] == 0.0


def test_both_sides_erases_the_candidate_too():
    q, c, pcs, mean_vec, scores = synthetic_pair(6)
    c_scores = np.linspace(1.0, 0.0, c.shape[0])
    cfg = DelAucConfig(name="both", knob="side", side="both")
    one = del_auc_gap(PairEvaluator(q, c, pcs, mean_vec, "baseline", "drop", "query"),
                      scores, None, PREDECLARED)
    two = del_auc_gap(PairEvaluator(q, c, pcs, mean_vec, "baseline", "drop", "both"),
                      scores, c_scores, cfg)
    assert two["del_auc"] != pytest.approx(one["del_auc"], abs=1e-6)


def test_config_set_is_bounded_and_unique():
    cfgs = build_configs()
    names = [c.name for c in cfgs]
    assert len(names) == len(set(names))
    assert len(cfgs) <= 40, "the sweep must stay small enough to be a sensitivity analysis"
    assert cfgs[0] == PREDECLARED
    # Every non-predeclared configuration moves at least one knob.
    for cfg in cfgs[1:]:
        assert cfg != PREDECLARED
    # The two token-filter arms are the only ones flagged as not matching the
    # pooling the artifacts were generated with.
    mismatched = {c.name for c in cfgs if not c.pooling_matches_generator}
    assert mismatched == {"filter_all", "filter_no_empty"}


@pytest.mark.parametrize("mean,se,expected", [
    (0.4, 0.1, "win"),
    (-0.4, 0.1, "loss"),
    (0.1, 0.1, "tie"),
    (0.2, 0.1, "win"),
    (float("nan"), 0.1, "tie"),
    (0.5, 0.0, "tie"),
])
def test_verdict_uses_the_two_se_rule(mean, se, expected):
    assert verdict(mean, se) == expected


def _fake_per_pair() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(11)
    for config, shift in (("predeclared", 0.3), ("erase_zero", -0.3)):
        for model in ("bowphs/LaTa", "bowphs/PhilTa"):
            for view in ("ig", "retrieval_mark"):
                for i in range(30):
                    base = float(rng.normal(0.4, 0.05))
                    rows.append({"config": config, "model": model, "view": view,
                                 "example_tag": f"example{i:03d}", "variant": "baseline",
                                 "del_auc_gap": base})
                    rows.append({"config": config, "model": model, "view": view,
                                 "example_tag": f"example{i:03d}", "variant": "abtt",
                                 "del_auc_gap": base + shift})
    # One pair that is undefined under ABTT: it must drop out of both variants.
    rows.append({"config": "predeclared", "model": "bowphs/LaTa", "view": "ig",
                 "example_tag": "example999", "variant": "baseline", "del_auc_gap": 0.4})
    rows.append({"config": "predeclared", "model": "bowphs/LaTa", "view": "ig",
                 "example_tag": "example999", "variant": "abtt",
                 "del_auc_gap": float("nan")})
    return pd.DataFrame(rows)


def test_paired_cells_drops_pairs_undefined_in_either_variant():
    cells = paired_cells(_fake_per_pair())
    row = cells[(cells["config"] == "predeclared") & (cells["view"] == "ig")
                & (cells["model"] == "bowphs/LaTa")].iloc[0]
    assert row["n_pairs"] == 30
    assert row["paired_mean"] == pytest.approx(0.3, abs=1e-9)
    assert row["verdict"] == "win"


def test_summarise_counts_every_cell_once():
    per_pair = _fake_per_pair()
    cells = paired_cells(per_pair)
    cfgs = [DelAucConfig(name="predeclared", knob="-"),
            DelAucConfig(name="erase_zero", knob="erasure", erasure="zero")]
    summary = summarise_configs(cells, cfgs)
    assert list(summary["config"]) == ["predeclared", "erase_zero"]
    for row in summary.itertuples():
        assert row.wins + row.ties + row.losses == row.n_cells == 4
    assert summary.iloc[0]["wins"] == 4
    assert summary.iloc[1]["losses"] == 4


def test_table_generator_renders_every_configuration(tmp_path):
    per_pair = _fake_per_pair()
    cells = paired_cells(per_pair)
    cfgs = [DelAucConfig(name="predeclared", knob="-"),
            DelAucConfig(name="erase_zero", knob="erasure", erasure="zero")]
    summary = summarise_configs(cells, cfgs)

    tex = tablegen.render(summary, cells)
    assert "\\begin{table*}" in tex and "\\end{table*}" in tex
    assert tex.count("\\\\") >= len(summary)
    assert "\\textbf{4/0/0}" in tex, "the predeclared row is the one in boldface"
    assert "0/0/4" in tex
    assert "tab:attribution_delauc_sensitivity" in tex
    # Self-contained caption: it has to say what a tie is and what the ranges are.
    assert "two\nstandard errors" in tex or "two standard errors" in tex

    out = tmp_path / "t.tex"
    summary.to_csv(tmp_path / "configs.csv", index=False)
    cells.to_csv(tmp_path / "cells.csv", index=False)
    tablegen.main(["--configs_csv", str(tmp_path / "configs.csv"),
                   "--cells_csv", str(tmp_path / "cells.csv"),
                   "--out", str(out)])
    assert out.read_text() == tex


def test_table_labels_cover_the_shipped_configuration_set():
    """A new configuration without a label would print its raw name."""
    for cfg in build_configs():
        assert cfg.name in tablegen.SETTING_LABELS
        assert cfg.knob in tablegen.KNOB_LABELS


PAPER_TEX = REPO_ROOT / "overleaf_drafts" / "acl_latex.tex"


@pytest.mark.skipif(
    not PAPER_TEX.exists(),
    reason="ci.yml sparse-checkouts without overleaf_drafts/",
)
def test_paper_does_not_input_the_table_yet():
    """Issue #195 is an analysis. Whether the appendix carries this table is a
    separate decision, so the generated file must stay unwired until it is
    taken."""
    assert "attribution_delauc_sensitivity" not in PAPER_TEX.read_text()


@pytest.mark.parametrize("value,expected", [
    (-0.0001, "0.00"),
    (0.0, "0.00"),
    (-0.44, "-0.44"),
    (float("nan"), "--"),
])
def test_fmt_does_not_print_a_negative_zero(value, expected):
    assert tablegen.fmt(value, 2) == expected
