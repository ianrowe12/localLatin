"""Guards for scripts/paper/reframe/e1_coordinate_ablation.py (issue #246: E1).

Two layers of checks:

* synthetic arrays (always run, no embedding cache needed): the two rankings find
  planted coordinates; zeroing and standardization are fit on train only; the closed-form
  cosine shares equal a brute-force pair loop and sum to 1; r matches its definition; the
  reproduction gates pass on matching references and fail on a perturbed one; the
  concentration measure counts the coordinates of a planted direction; the table and the
  facts file render from a tiny fixture and omit absent models.
* committed result CSVs (skipped when runs/active/reframe/e1 is not checked out): the
  gates hold on the committed numbers and the generated table regenerates byte for byte.

Nothing here reads an embedding cache or probes a path outside the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "paper" / "reframe"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))
sys.path.insert(0, str(REPO_ROOT / "src"))

pytest.importorskip("sklearn")
import abtt_subspace_whiten as asw  # noqa: E402
import e1_coordinate_ablation as e1  # noqa: E402
from canon_retrieval import upper_triangle_labels  # noqa: E402

LATA, LABSE = "bowphs/LaTa", "sentence-transformers/LaBSE"


def _synthetic(n_tr=60, n_te=50, d=24, n_dirs=12, seed=0, nuisance=40.0):
    """Directory-clustered vectors; coordinate 0 varies hugely across passages, coordinate 1
    is a large shared offset that barely varies."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_dirs, d))

    def draw(n):
        ids = rng.integers(0, n_dirs, size=n)
        x = centers[ids] + 0.3 * rng.normal(size=(n, d))
        x[:, 0] += nuisance * rng.normal(size=n)
        x[:, 1] += 200.0 + 0.1 * rng.normal(size=n)
        return x.astype(np.float32), ids.astype(str)

    tr, tr_ids = draw(n_tr)
    te, te_ids = draw(n_te)
    return tr, te, tr_ids, te_ids


def _auroc_fn(tr_ids, te_ids):
    lab_tr, lab_te = upper_triangle_labels(tr_ids), upper_triangle_labels(te_ids)
    return lambda a, b: {"aucroc": asw.pair_auroc(b, lab_te),
                         "train_aucroc": asw.pair_auroc(a, lab_tr)}


def _fixture_frames():
    """Two fake models (one T5 name, one not), two layers each, through layer_rows."""
    abl, coords, shares = [], [], []
    for mid, nuisance in ((LATA, 40.0), (LABSE, 0.0)):
        for layer in (1, 2):
            tr, te, tr_ids, te_ids = _synthetic(seed=layer, nuisance=nuisance * layer)
            a, c, s = e1.layer_rows(mid, layer, tr, te, _auroc_fn(tr_ids, te_ids))
            abl += a
            coords += c
            shares += s
    return pd.DataFrame(abl), pd.DataFrame(coords), pd.DataFrame(shares)


def _fixture_conc():
    """The concentration rows of the same four fake model-layers."""
    rows = []
    for mid, nuisance in ((LATA, 40.0), (LABSE, 0.0)):
        for layer in (1, 2):
            tr, *_ = _synthetic(seed=layer, nuisance=nuisance * layer)
            rows.append({"model": mid, "layer": layer, **e1.concentration(tr)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# rankings, r
# --------------------------------------------------------------------------- #

def test_rankings_find_the_planted_coordinates():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 30))
    x[:, 7] = 500.0 + 0.5 * rng.normal(size=200)  # massive, nearly constant
    x[:, 12] = 60.0 * rng.normal(size=200)  # mean near 0, varies across passages
    x[:, 20] = 90.0 + 20.0 * rng.normal(size=200)  # second on both counts
    by_mag = e1.rank_coords(x, "mean_abs")
    by_var = e1.rank_coords(x, "variance")
    assert list(by_mag[:3]) == [7, 20, 12]
    assert list(by_var[:2]) == [12, 20]
    assert 7 not in by_var[:2]
    assert sorted(by_mag) == list(range(30)) and sorted(by_var) == list(range(30))
    with pytest.raises(ValueError):
        e1.rank_coords(x, "cosine")


def test_ranking_is_on_raw_values_and_breaks_ties_by_index():
    x = np.zeros((4, 5))
    x[:, 3] = [-5.0, 5.0, -5.0, 5.0]  # mean 0, mean |x| 5: mean_abs must not center first
    x[:, 1] = [2.0, 2.0, 2.0, 2.0]
    assert list(e1.rank_coords(x, "mean_abs")) == [3, 1, 0, 2, 4]
    assert list(e1.rank_coords(x, "variance")) == [3, 0, 1, 2, 4]


def test_r_is_sd_over_abs_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(loc=-3.0, scale=2.0, size=(500, 6)).astype(np.float32)
    x[:, 5] = np.tile([-1.0, 1.0], 250)  # mean exactly 0
    s = e1.coord_stats(x)
    x64 = x.astype(np.float64)
    for k in range(5):
        col = x64[:, k]
        sd = np.sqrt(np.mean((col - col.mean()) ** 2))
        assert s["r"][k] == pytest.approx(sd / abs(col.mean()), rel=1e-12)
        assert s["variance"][k] == pytest.approx(sd ** 2, rel=1e-12)
        assert s["mean_abs"][k] == pytest.approx(np.mean(np.abs(col)), rel=1e-12)
    assert np.isinf(s["r"][5])


# --------------------------------------------------------------------------- #
# interventions are fit on train only
# --------------------------------------------------------------------------- #

def test_zeroing_sets_the_columns_in_both_splits_and_copies():
    tr, te, *_ = _synthetic()
    tr0, te0 = tr.copy(), te.copy()
    a, b = e1.zero_coords(tr, te, [0, 5])
    assert not a[:, [0, 5]].any() and not b[:, [0, 5]].any()
    keep = [i for i in range(tr.shape[1]) if i not in (0, 5)]
    np.testing.assert_array_equal(a[:, keep], tr[:, keep])
    np.testing.assert_array_equal(b[:, keep], te[:, keep])
    np.testing.assert_array_equal(tr, tr0)  # inputs untouched
    np.testing.assert_array_equal(te, te0)
    assert a.dtype == tr.dtype and b.dtype == te.dtype


def test_standardize_uses_train_statistics_only():
    tr, te, *_ = _synthetic()
    other = te * 3.0 - 7.0
    a1, b1, flat1 = e1.standardize(tr, te)
    a2, b2, flat2 = e1.standardize(tr, other)
    np.testing.assert_array_equal(a1, a2)  # the test rows do not enter the fit
    assert flat1 == flat2 == 0
    mu, sd = tr.astype(np.float64).mean(axis=0), tr.astype(np.float64).std(axis=0)
    np.testing.assert_allclose(a1.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(a1.std(axis=0), 1.0, atol=1e-9)
    np.testing.assert_allclose(b1, (te - mu) / sd, rtol=0, atol=1e-9)
    np.testing.assert_allclose(b2, (other - mu) / sd, rtol=0, atol=1e-9)


def test_standardize_guards_zero_sd_coordinates():
    tr, te, *_ = _synthetic()
    tr[:, 4] = 2.5
    a, b, flat = e1.standardize(tr, te)
    assert flat == 1
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert not a[:, 4].any()
    np.testing.assert_allclose(b[:, 4], te[:, 4].astype(np.float64) - 2.5, atol=1e-9)


def test_layer_rows_fit_nothing_on_test():
    tr, te, tr_ids, te_ids = _synthetic()
    rng = np.random.default_rng(9)
    te_other = (te[rng.permutation(len(te))] * -2.0 + 11.0).astype(np.float32)
    fn = _auroc_fn(tr_ids, te_ids)
    a1, c1, s1 = (pd.DataFrame(x) for x in e1.layer_rows(LATA, 3, tr, te, fn))
    a2, c2, s2 = (pd.DataFrame(x) for x in e1.layer_rows(LATA, 3, tr, te_other, fn))
    # chosen coordinates, train metrics and train geometry do not depend on the test rows
    train_cols = ["tag", "coords", "n_zero_sd", "train_aucroc", "pc1_share_train",
                  "pc10_share_train", "eff_rank_train"]
    pd.testing.assert_frame_equal(a1[train_cols], a2[train_cols])
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(s1[s1.split == "train"], s2[s2.split == "train"])
    assert (s1[s1.split == "test"].coords.values == s2[s2.split == "test"].coords.values).all()
    # and the test metrics do move, so the comparison above is not vacuous
    assert not np.allclose(a1["aucroc"], a2["aucroc"])


def test_layer_rows_cover_every_intervention_and_zero_before_normalizing():
    tr, te, tr_ids, te_ids = _synthetic()
    abl, coords, shares = e1.layer_rows(LATA, 1, tr, te, _auroc_fn(tr_ids, te_ids))
    a = pd.DataFrame(abl).set_index("tag")
    assert list(a.index) == e1.ALL_TAGS
    # variance ranks the passage-varying coordinate first, mean |x| the shared offset
    assert a.loc["zero_variance_k1", "coords"] == "0"
    assert a.loc["zero_mean_abs_k1", "coords"] == "1"
    assert a.loc["zero_variance_k3", "coords"].split(";")[0] == "0"
    # the planted nuisance collapses retrieval, and zeroing that one coordinate repairs it
    assert a.loc["base", "aucroc"] < 0.75
    assert a.loc["zero_variance_k1", "aucroc"] > 0.9
    assert a.loc["zero_mean_abs_k1", "aucroc"] < 0.75
    assert a.loc["zero_variance_k1", "pc1_share_train"] < 0.5 < a.loc["base", "pc1_share_train"]
    # centering cannot change a share computed on re-centered vectors
    assert a.loc["center", "pc1_share_train"] == pytest.approx(a.loc["base", "pc1_share_train"],
                                                               abs=1e-6)
    # the zero rows are exactly zero_coords on the raw vectors, scored by the metric
    za, zb = e1.zero_coords(tr, te, [0])
    assert a.loc["zero_variance_k1", "aucroc"] == _auroc_fn(tr_ids, te_ids)(za, zb)["aucroc"]
    # same functions as the H1 script for the reference cells
    ca, cb = asw.abtt(tr, te, 3)
    assert a.loc["abtt_D3", "aucroc"] == _auroc_fn(tr_ids, te_ids)(ca, cb)["aucroc"]
    c = pd.DataFrame(coords)
    assert len(c) == 2 * e1.TOP_N and set(c.ranking) == set(e1.RANKINGS)
    top_var = c[(c.ranking == "variance") & (c["rank"] == 1)].iloc[0]
    assert top_var.coord == 0 and top_var.r > 1  # varies more than it shifts
    top_mag = c[(c.ranking == "mean_abs") & (c["rank"] == 1)].iloc[0]
    assert top_mag.coord == 1 and top_mag.r < 0.01 and top_mag.rank_other > 1
    assert top_mag.median_mean_abs == pytest.approx(
        np.median(np.abs(tr.astype(np.float64)).mean(axis=0)))
    s = pd.DataFrame(shares)
    assert len(s) == 2 * len(e1.RANKINGS) * len(e1.KS)


def test_layer_rows_run_through_the_paper_metric_block():
    tr, te, tr_ids, te_ids = _synthetic()
    counts = pd.Series(te_ids).value_counts()
    saved = dict(asw._CTX)
    asw._CTX.update(tr_ids=tr_ids, te_ids=te_ids,
                    te_partner=np.array([counts[i] > 1 for i in te_ids]))
    try:
        abl, _, _ = e1.layer_rows(LATA, 1, tr, te, asw._metrics)
    finally:
        asw._CTX.clear()
        asw._CTX.update(saved)
    a = pd.DataFrame(abl).set_index("tag")
    assert set(asw.KEEP) <= set(a.columns)
    ref = _auroc_fn(tr_ids, te_ids)(tr, te)
    assert a.loc["base", "aucroc"] == pytest.approx(ref["aucroc"], abs=1e-12)
    assert a.loc["base", "train_aucroc"] == pytest.approx(ref["train_aucroc"], abs=1e-12)


# --------------------------------------------------------------------------- #
# cosine shares
# --------------------------------------------------------------------------- #

def _brute_contributions(x):
    u = x.astype(np.float64)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    c = np.zeros(u.shape[1])
    total = 0.0
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            c += u[i] * u[j]
            total += float(u[i] @ u[j])
    return c, total


def test_cosine_contributions_equal_the_pair_loop_and_sum_to_one():
    tr, *_ = _synthetic(n_tr=25, d=9)
    c, n_pairs = e1.cosine_contributions(tr)
    brute, total = _brute_contributions(tr)
    assert n_pairs == 25 * 24 // 2
    np.testing.assert_allclose(c, brute, rtol=1e-9, atol=1e-9)
    assert c.sum() == pytest.approx(total, rel=1e-9)
    assert e1.cosine_share(c, range(9)) == pytest.approx(1.0, abs=1e-12)
    assert (e1.cosine_share(c, [0, 1]) + e1.cosine_share(c, range(2, 9))
            == pytest.approx(1.0, abs=1e-12))
    assert e1.cosine_share(c, [1]) == pytest.approx(brute[1] / total, rel=1e-9)


def test_cosine_share_can_exceed_one_when_other_coordinates_are_negative():
    x = np.array([[10.0, 3.0], [10.0, -3.0], [10.0, 3.0], [10.0, -3.0]])
    c, _ = e1.cosine_contributions(x)
    assert c[1] < 0 < c[0]
    assert e1.cosine_share(c, [0]) > 1.0
    assert e1.cosine_share(np.zeros(3), [0]) != e1.cosine_share(np.zeros(3), [0])  # nan


def test_share_rows_match_the_closed_form_per_split():
    tr, te, tr_ids, te_ids = _synthetic(n_tr=30, n_te=20, d=10)
    _, _, shares = e1.layer_rows(LATA, 1, tr, te, _auroc_fn(tr_ids, te_ids))
    s = pd.DataFrame(shares)
    order = e1.rank_coords(tr, "mean_abs")  # chosen on train for both splits
    for split, x in (("train", tr), ("test", te)):
        brute, total = _brute_contributions(x)
        row = s[(s.ranking == "mean_abs") & (s.k == 3) & (s.split == split)].iloc[0]
        assert row.coords == ";".join(str(i) for i in order[:3])
        assert row.share == pytest.approx(brute[order[:3]].sum() / total, rel=1e-9)
        n_pairs = len(x) * (len(x) - 1) / 2
        assert row.mean_cosine == pytest.approx(total / n_pairs, rel=1e-9)
        assert row.n_negative_all == int((brute < 0).sum())


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def _references(abl):
    a = abl.set_index(["model", "layer", "tag"])
    res, h1, geom = [], [], []
    for (mid, layer), _ in abl.groupby(["model", "layer"]):
        res.append({"model": mid, "repr": "hidden", "pooling": "mean", "layer": layer,
                    "method": "baseline", "aucroc": a.loc[(mid, layer, "base"), "aucroc"]})
        res.append({"model": mid, "repr": "hidden", "pooling": "mean", "layer": layer,
                    "method": "abtt_fixed", "aucroc": 0.123})
        for tag, (variant, D) in {"center": ("center", 0), "abtt_D1": ("abtt", 1),
                                  "abtt_D3": ("abtt", 3), "abtt_D10": ("abtt", 10)}.items():
            h1.append({"model": mid, "layer": layer, "variant": variant, "D": D,
                       "aucroc": a.loc[(mid, layer, tag), "aucroc"]})
        for split, view, off in (("train", "raw", 0.0), ("test", "raw", 0.3),
                                 ("train", "abtt_d10", 0.4)):
            geom.append({"model": mid, "pooling": "mean", "layer": layer, "split": split,
                         "view": view, "pc1_variance_ratio":
                             a.loc[(mid, layer, "base"), "pc1_share_train"] + off})
    return pd.DataFrame(res), pd.DataFrame(h1), pd.DataFrame(geom)


def test_gates_pass_on_matching_references_and_fail_on_drift():
    abl, _, _ = _fixture_frames()
    res, h1, geom = _references(abl)
    g = e1.gate_table(abl, res, h1, geom)
    assert len(g) == 4 * 2 and g.ok.all()
    assert (g.max_abs_diff == 0).all()
    assert list(g[g.gate.str.startswith("2a")].n_cells) == [2, 2]  # D=0, two layers
    assert list(g[g.gate.str.startswith("2b")].n_cells) == [6, 6]  # ABTT D=1,3,10
    assert (g[g.gate.str.startswith("2a")].tolerance == e1.GATE_TOL_CENTER).all()
    assert (g[~g.gate.str.startswith(("2a", "3"))].tolerance == e1.GATE_TOL_AUROC).all()
    assert (g.n_over_strict == 0).all()

    drift = h1.copy()
    drift.loc[(drift.model == LATA) & (drift.D == 3) & (drift.layer == 2), "aucroc"] += 5e-6
    g = e1.gate_table(abl, res, drift, geom)
    bad = g[~g.ok]
    assert len(bad) == 1 and bad.iloc[0].model == LATA and bad.iloc[0].gate.startswith("2b")
    # the failing cell is named, and a looser tolerance is an explicit argument
    assert bad.iloc[0].n_over_tolerance == 1
    assert bad.iloc[0].cells_over_tolerance.startswith("L2 ABTT D=3 5.00e-06")
    assert (g[g.ok].n_over_tolerance == 0).all()
    assert e1.gate_table(abl, res, drift, geom, tol_auroc=1e-5).ok.all()

    # D=0 cells alone have the looser tolerance: 5e-6 passes but is listed, 5e-5 fails
    center = h1.copy()
    sel = (center.model == LATA) & (center.variant == "center") & (center.layer == 1)
    center.loc[sel, "aucroc"] += 5e-6
    g = e1.gate_table(abl, res, center, geom)
    assert g.ok.all()
    listed = g[g.n_over_strict > 0]
    assert len(listed) == 1 and listed.iloc[0].gate.startswith("2a")
    assert listed.iloc[0].cells_over_strict.startswith("L1 D=0 5.00e-06")
    assert listed.iloc[0].strict_tolerance == e1.GATE_TOL_AUROC
    center.loc[sel, "aucroc"] += 5e-5
    g = e1.gate_table(abl, res, center, geom)
    bad = g[~g.ok]
    assert len(bad) == 1 and bad.iloc[0].gate.startswith("2a") and bad.iloc[0].model == LATA
    # a looser --tol_auroc never tightens the D=0 tolerance
    loose = e1.gate_table(abl, res, h1, geom, tol_auroc=1e-4)
    assert (loose[loose.gate.str.startswith("2a")].tolerance == 1e-4).all()

    # a published layer that was not computed fails the gate unless a layer subset was asked
    extra = pd.concat([res, res[res.layer == 2].assign(layer=3)], ignore_index=True)
    assert not e1.gate_table(abl, extra, h1, geom).ok.any()
    assert e1.gate_table(abl, extra, h1, geom, complete=False).ok.all()

    # a missing reference cell is a failure, never a silent skip
    g = e1.gate_table(abl, res, h1, geom[geom.layer != 1])
    assert not g[g.gate.str.startswith("3")].ok.any()
    assert g[~g.gate.str.startswith("3")].ok.all()


# --------------------------------------------------------------------------- #
# concentration (descriptive)
# --------------------------------------------------------------------------- #

def _rank_one(direction, n=80, seed=3, offset=7.0):
    """Vectors that vary along one direction only, plus a shared offset."""
    rng = np.random.default_rng(seed)
    u = np.asarray(direction, dtype=np.float64)
    u = u / np.linalg.norm(u)
    return rng.normal(size=(n, 1)) * 50.0 * u[None, :] + offset


def test_concentration_of_a_one_coordinate_direction():
    u = np.zeros(40)
    u[17] = 1.0
    c = e1.concentration(_rank_one(u))
    assert c["pc1_n50"] == 1 and c["pc1_n90"] == 1
    assert c["pc1_participation"] == pytest.approx(1.0, abs=1e-9)
    assert c["pc1_mass_top10var"] == pytest.approx(1.0, abs=1e-9)
    assert c["var_share_top1"] == pytest.approx(1.0, abs=1e-12)
    assert c["dim"] == 40
    # a little isotropic noise on every coordinate does not change the counts
    rng = np.random.default_rng(4)
    noisy = e1.concentration(_rank_one(u) + 0.1 * rng.normal(size=(80, 40)))
    assert noisy["pc1_n50"] == 1 and noisy["pc1_n90"] == 1
    assert noisy["pc1_participation"] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("m", [4, 10, 25])
def test_concentration_of_a_direction_spread_evenly_over_m_coordinates(m):
    d = 60
    u = np.zeros(d)
    u[5:5 + m] = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)  # signs must not matter
    c = e1.concentration(_rank_one(u))
    assert c["pc1_participation"] == pytest.approx(m, rel=1e-9)
    assert c["pc1_n50"] == int(np.ceil(0.5 * m))
    assert c["pc1_n90"] == int(np.ceil(0.9 * m))
    # the top-10 coordinates by variance hold 10/m of the direction (all of it if m <= 10)
    assert c["pc1_mass_top10var"] == pytest.approx(min(1.0, 10 / m), abs=1e-9)
    for k in e1.CONC_KS:
        assert c[f"var_share_top{k}"] == pytest.approx(min(1.0, k / m), abs=1e-9)


def test_variance_shares_follow_the_coordinate_variances_and_use_train_only():
    rng = np.random.default_rng(5)
    sd = np.array([3.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    x = rng.normal(size=(4000, 6)) * sd + 100.0  # a shared offset is not variance
    c = e1.concentration(x)
    var = x.var(axis=0)
    assert c["var_share_top1"] == pytest.approx(var[0] / var.sum(), rel=1e-12)
    assert c["var_share_top3"] == pytest.approx(np.sort(var)[::-1][:3].sum() / var.sum(),
                                                rel=1e-12)
    assert c["var_share_top1"] == pytest.approx(9 / 17, abs=0.03)
    # k above the width counts every coordinate
    assert c["var_share_top50"] == pytest.approx(1.0) and c["var_share_top100"] == pytest.approx(1.0)
    assert c["total_variance"] == pytest.approx(var.sum(), rel=1e-12)
    # the function takes the training matrix only, so nothing of the test split can enter
    assert e1.concentration(x.astype(np.float32))["pc1_n50"] == c["pc1_n50"]


# --------------------------------------------------------------------------- #
# compute plumbing
# --------------------------------------------------------------------------- #

def test_missing_cache_is_an_error_unless_allowed(tmp_path, capsys):
    run = tmp_path / "phase9_bases" / asw.slug(LATA) / asw.SUBDIR
    run.mkdir(parents=True)
    for layer in (0, 1, 2):
        np.save(run / f"hidden_layer{layer}_embeddings.npy", np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(SystemExit) as err:
        e1.build_tasks(tmp_path, [LATA, LABSE], None, allow_missing=False)
    assert "LaBSE" in str(err.value) and "--allow_missing" in str(err.value)
    tasks = e1.build_tasks(tmp_path, [LATA, LABSE], None, allow_missing=True)
    assert tasks == [(str(tmp_path), LATA, 1), (str(tmp_path), LATA, 2)]  # layer 0 is not scored
    assert "WARNING: skipping LaBSE" in capsys.readouterr().out
    assert e1.build_tasks(tmp_path, [LATA], [2], allow_missing=False) == [(str(tmp_path), LATA, 2)]


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

def test_table_renders_from_a_fixture_and_omits_absent_models(tmp_path):
    abl, _, _ = _fixture_frames()
    w = e1.wide(abl)
    out = tmp_path / "t.tex"
    omitted = e1.write_table(w, out)
    assert omitted == ["PhilTa", "mT5-base", "Qwen3-0.6B", "KaLM-mini"]
    tex = out.read_text()
    assert tex.startswith(asw.HEADER + "\n")
    assert r"\label{tab:e1_coordinate_ablation}" in tex
    assert r"\begin{table*}" in tex and r"\end{table*}" in tex
    assert chr(0x2014) not in tex and "---" not in tex
    rows = [ln for ln in tex.splitlines() if ln.startswith(("LaTa &", "LaBSE &"))]
    assert len(rows) == 4  # two models, AUROC block and top-PC block
    for name in omitted:
        assert name not in tex
    # 15 columns in every body row, in the placeholder's order
    assert all(r.count("&") == 14 for r in rows)
    # each row is read at the model's lowest-AUROC layer, and the cells are the CSV's
    x = e1.worst_layer(w, "LaTa")
    raw = abl[(abl.model == LATA)].set_index(["layer", "tag"])
    base = raw.xs("base", level="tag")["aucroc"]
    assert int(x["layer"]) == int(base.idxmin())
    cells = [c.strip() for c in rows[0].rstrip("\\ ").split("&")]
    assert cells[1] == str(int(base.idxmin()))
    expect = [f"{raw.loc[(int(base.idxmin()), t), 'aucroc']:.3f}" for t in e1.TABLE_TAGS]
    assert cells[2:] == expect
    share = [c.strip() for c in rows[2].rstrip("\\ ").split("&")][2:]
    assert share == [f"{raw.loc[(int(base.idxmin()), t), 'pc1_share_train']:.3f}"
                     for t in e1.TABLE_TAGS]


def test_render_cli_writes_table_and_facts_and_names_omitted_models(tmp_path, capsys):
    abl, coords, shares = _fixture_frames()
    out_dir, tab_dir = tmp_path / "e1", tmp_path / "tables"
    out_dir.mkdir()
    abl.to_csv(out_dir / e1.ABL_NAME, index=False, float_format="%.10g")
    coords.to_csv(out_dir / e1.COORD_NAME, index=False, float_format="%.10g")
    shares.to_csv(out_dir / e1.SHARE_NAME, index=False, float_format="%.10g")
    argv = ["render", "--out_dir", str(out_dir), "--tab_dir", str(tab_dir),
            "--results_csv", str(tmp_path / "none.csv"),
            "--selected_tex", str(tmp_path / "none.tex")]
    # without the concentration CSV the section says so and nothing else breaks
    assert e1.main(argv) == 0
    early = (out_dir / e1.FACTS_NAME).read_text()
    assert f"`{e1.CONC_NAME}` not found" in early
    capsys.readouterr()
    _fixture_conc().to_csv(out_dir / e1.CONC_NAME, index=False, float_format="%.10g")
    rc = e1.main(argv)
    assert rc == 0
    said = capsys.readouterr().out
    for name in ("PhilTa", "mT5-base", "Qwen3-0.6B", "KaLM-mini"):
        assert f"omitting {name}" in said
    assert (tab_dir / e1.TABLE_NAME).exists()
    facts = (out_dir / e1.FACTS_NAME).read_text()
    assert chr(0x2014) not in facts
    assert "ABSENT from the CSV" in facts and "KaLM-mini" in facts
    assert "## 2. Collapsed T5 layers" in facts and "### LaBSE" in facts
    # the concentration section is labelled descriptive and post hoc, and adds no verdict
    sec = facts.split("## 9. Concentration")[1].split("## 10. All layers")[0]
    assert "descriptive and was added after the ablation results were seen" in sec
    assert "No prediction is attached" in sec
    assert "PASS" not in sec and "FAIL" not in sec
    assert "- LaTa (2 layers):" in sec and "LaBSE L1 (worst baseline layer)" in sec
    verdicts = facts.split("## 8. Verdicts in one place")[1].split("## 9.")[0]
    assert verdicts == early.split("## 8. Verdicts in one place")[1].split("## 9.")[0]
    # the CSV round trip keeps the empty ranking / coords cells as empty strings
    back = e1.read_abl(out_dir / e1.ABL_NAME)
    assert (back.loc[back.tag == "base", "coords"] == "").all()
    pd.testing.assert_frame_equal(e1.wide(back)[["m", "layer", "collapsed"]],
                                  e1.wide(abl)[["m", "layer", "collapsed"]])


def test_selected_layers_parser_reads_the_base_task_a_column(tmp_path):
    tex = tmp_path / "s.tex"
    tex.write_text("\\textbf{Model} & Base & SIF \\\\\nLaTa & 12 & 1 & 12 \\\\\n"
                   "Qwen3-0.6B & 26 & 22 & 2 \\\\\nLaTa (fine-tuned) & 11 & -- \\\\\n")
    assert e1.parse_selected_layers(tex) == {"LaTa": 12, "Qwen3-0.6B": 26}
    assert e1.parse_selected_layers(tmp_path / "absent.tex") == {}


# --------------------------------------------------------------------------- #
# committed results (skipped when the result CSVs are not checked out)
# --------------------------------------------------------------------------- #

E1_DIR = REPO_ROOT / "runs" / "active" / "reframe" / "e1"
REFS = [REPO_ROOT / e1.RES_CSV, REPO_ROOT / e1.H1_CSV, REPO_ROOT / e1.GEOM_CSV]
needs_results = pytest.mark.skipif(not (E1_DIR / e1.ABL_NAME).exists(),
                                   reason="runs/active/reframe/e1 not checked out")


@needs_results
def test_committed_results_pass_the_reproduction_gates():
    """Every gate holds at its default tolerance.

    The D=0 cells of gate 2 have their own tolerance of 1e-5 (GATE_TOL_CENTER in the
    script explains why). Outside that exception nothing may be over 1e-6: the only cells
    between the two tolerances are D=0 cells of mT5-base.
    """
    if not all(p.exists() for p in REFS):
        pytest.skip("reference CSVs not checked out")
    abl = e1.read_abl(E1_DIR / e1.ABL_NAME)
    g = e1.gate_table(abl, *(pd.read_csv(p) for p in REFS))
    assert len(g) and g.ok.all(), g[~g.ok].to_string()
    assert (g[g.gate.str.startswith("2a")].tolerance == 1e-5).all()
    assert (g[g.gate.str.startswith(("1", "2b"))].tolerance == 1e-6).all()
    for row in g[g.n_over_strict > 0].itertuples():
        assert row.model == "google/mt5-base" and row.gate.startswith("2a"), row
        assert all(" D=0 " in f"{cell} " for cell in row.cells_over_strict.split("; ")), row
    # the gate file the script wrote says the same
    written = pd.read_csv(E1_DIR / e1.GATE_NAME)
    assert written.ok.all() and list(written.gate) == list(g.gate)


@needs_results
def test_committed_concentration_covers_every_model_layer():
    path = E1_DIR / e1.CONC_NAME
    if not path.exists():
        pytest.skip("concentration CSV not committed")
    conc = pd.read_csv(path)
    abl = e1.read_abl(E1_DIR / e1.ABL_NAME)
    assert (set(zip(conc.model, conc.layer))
            == set(zip(abl.model, abl.layer))) and not conc.duplicated(["model", "layer"]).any()
    shares = conc[[f"var_share_top{k}" for k in e1.CONC_KS]].to_numpy()
    assert (np.diff(shares, axis=1) >= 0).all() and (shares > 0).all() and (shares <= 1).all()
    assert (conc.pc1_n50 <= conc.pc1_n90).all() and (conc.pc1_n90 <= conc.dim).all()
    assert ((conc.pc1_participation >= 1) & (conc.pc1_participation <= conc.dim)).all()


@needs_results
def test_committed_table_regenerates_byte_identically(tmp_path):
    committed = REPO_ROOT / e1.TAB_DIR / e1.TABLE_NAME
    if not committed.exists():
        pytest.skip("generated table not committed")
    e1.write_table(e1.wide(e1.read_abl(E1_DIR / e1.ABL_NAME)), tmp_path / "t.tex")
    assert (tmp_path / "t.tex").read_text() == committed.read_text()
