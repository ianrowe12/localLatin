"""Guards for scripts/paper/reframe/e2_zeroing_followup.py (issue #252: E2, zeroing follow-up).

Two layers of checks:

* synthetic arrays (always run, no embedding cache needed): a planted direction lying on
  the zeroed coordinates has loading 1 and no variance left; one lying off them is
  untouched (angle 0, correlation 1); one lying half on them follows the closed form;
  components that survive zeroing move up in rank and keep the subspace; principal angles
  ignore a rotation within a span; the sign of a component does not matter; the removal
  of chosen components is ABTT when they are the first D; nothing is fit on test; the
  zeroed sets and the gated cells are E1's; the reproduction gates pass on matching
  references and fail on a changed coordinate set, a drifted cell, a NaN or a missing
  row; the facts file renders from a tiny fixture.
* committed result CSVs (skipped when runs/active/reframe/e2 holds no follow-up CSV): the
  gates hold on the committed numbers and the facts file regenerates byte for byte.

Nothing here reads an embedding cache or probes a path outside the repository.
"""

from __future__ import annotations

import re
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
import e2_zeroing_followup as e2  # noqa: E402
from canon_retrieval import upper_triangle_labels  # noqa: E402

LATA, LABSE = "bowphs/LaTa", "sentence-transformers/LaBSE"


def _synthetic(n_tr=60, n_te=50, d=24, n_dirs=12, seed=0, nuisance=40.0):
    """Directory-clustered vectors; coordinate 0 varies hugely across passages, coordinate 1
    is a large shared offset that barely varies (the fixture of the E1 tests)."""
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


def _planted(directions, scales, n_tr=80, n_te=40, seed=3, offset=7.0):
    """Vectors varying only along the given orthonormal directions, plus a shared offset.

    The training scores are exactly uncorrelated (QR of centered draws), so the principal
    components of the training vectors are the directions, in decreasing ``scales``.
    """
    rng = np.random.default_rng(seed)
    u = np.asarray(directions, dtype=np.float64)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    t = rng.normal(size=(n_tr, len(u)))
    q, _ = np.linalg.qr(t - t.mean(axis=0))
    tr = (q * np.sqrt(n_tr)) @ (np.asarray(scales, dtype=np.float64)[:, None] * u) + offset
    te = (rng.normal(size=(n_te, len(u))) * np.asarray(scales)) @ u + offset
    return tr, te


def _unit(d, coords):
    u = np.zeros(d)
    u[list(coords)] = 1.0
    return u


def _fixture():
    """Two fake models (one T5 name, one not), two layers each: the follow-up rows and the
    E1 ablation rows of the same vectors."""
    rows, abl = [], []
    for mid, nuisance in ((LATA, 40.0), (LABSE, 0.0)):
        for layer in (1, 2):
            tr, te, tr_ids, te_ids = _synthetic(seed=layer, nuisance=nuisance * layer)
            fn = _auroc_fn(tr_ids, te_ids)
            rows += e2.layer_rows(mid, layer, tr, te, fn)
            abl += e1.layer_rows(mid, layer, tr, te, fn)[0]
    return pd.DataFrame(rows), pd.DataFrame(abl)


def _h1(rows):
    """An H1 frame holding the reference cells of the follow-up rows."""
    out = []
    for (mid, layer), g in rows.groupby(["model", "layer"], sort=False):
        x = g.iloc[0]
        for col, (variant, D) in e2.H1_KEYS.items():
            out.append({"model": mid, "layer": layer, "variant": variant, "D": D,
                        "aucroc": x[col]})
        out.append({"model": mid, "layer": layer, "variant": "center", "D": 0, "aucroc": 0.5})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# measures 1 to 4 on planted directions
# --------------------------------------------------------------------------- #

def test_a_direction_lying_on_the_zeroed_coordinates_is_removed():
    tr, te = _planted([_unit(8, (0, 1))], [50.0])
    g = e2.zeroing_geometry(tr, te, [0, 1])
    assert g["load_pc1"] == pytest.approx(1.0, abs=1e-12)
    assert g["var_left_pc1"] == pytest.approx(0.0, abs=1e-12)
    assert g["total_var_left"] == pytest.approx(0.0, abs=1e-12)
    # nothing of PC1 is left outside S: the remainder cosine is undefined, not 0 or 1
    assert np.isnan(g["remainder_cos_pc1"]) and np.isnan(g["remainder_share_pc1"])
    # and the zeroed vectors are constant, so their scores correlate with nothing
    assert np.isnan(g["pearson_train_pc1"]) and np.isnan(g["spearman_test_pc1"])


def test_a_direction_off_the_zeroed_coordinates_is_untouched():
    tr, te = _planted([_unit(8, (4, 5, 6))], [50.0])
    g = e2.zeroing_geometry(tr, te, [0, 1])
    assert g["load_pc1"] == pytest.approx(0.0, abs=1e-12)
    assert g["var_left_pc1"] == pytest.approx(1.0, abs=1e-12)
    assert g["var_share_zeroed_pc1"] == pytest.approx(1.0, abs=1e-12)
    assert g["total_var_left"] == pytest.approx(1.0, abs=1e-12)
    assert g["remainder_share_pc1"] == pytest.approx(1.0, abs=1e-12)
    assert g["angle_pc1"] == pytest.approx(0.0, abs=1e-4)
    assert g["remainder_cos_pc1"] == pytest.approx(1.0, abs=1e-12)
    assert g["abscos_newpc1_oldpc1"] == pytest.approx(1.0, abs=1e-12)
    for split in ("train", "test"):
        assert g[f"pearson_{split}_pc1"] == pytest.approx(1.0, abs=1e-9)
        assert g[f"spearman_{split}_pc1"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("m,n_zeroed", [(4, 2), (4, 1), (10, 9)])
def test_a_direction_partly_on_the_zeroed_coordinates_follows_the_closed_form(m, n_zeroed):
    """u spread evenly over m coordinates, n_zeroed of them zeroed: loading q = n_zeroed/m,
    variance left (1-q)^2, share of the zeroed variance 1-q, angle arccos sqrt(1-q), and
    the new top direction is exactly the remainder of the old one."""
    u = np.zeros(16)
    u[2:2 + m] = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)  # signs must not matter
    tr, te = _planted([u], [50.0])
    g = e2.zeroing_geometry(tr, te, list(range(2, 2 + n_zeroed)))
    q = n_zeroed / m
    assert g["load_pc1"] == pytest.approx(q, abs=1e-12)
    assert g["var_left_pc1"] == pytest.approx((1 - q) ** 2, abs=1e-12)
    assert g["var_share_zeroed_pc1"] == pytest.approx(1 - q, abs=1e-12)
    # which understates the surviving direction: along the renormalized remainder it is all
    assert g["remainder_share_pc1"] == pytest.approx(1.0, abs=1e-9)
    assert g["total_var_left"] == pytest.approx(1 - q, abs=1e-12)
    assert g["angle_pc1"] == pytest.approx(np.degrees(np.arccos(np.sqrt(1 - q))), abs=1e-6)
    assert g["abscos_newpc1_oldpc1"] == pytest.approx(np.sqrt(1 - q), abs=1e-12)
    assert g["remainder_cos_pc1"] == pytest.approx(1.0, abs=1e-12)
    assert g["pearson_train_pc1"] == pytest.approx(1.0, abs=1e-9)
    assert g["pearson_test_pc1"] == pytest.approx(1.0, abs=1e-9)


def test_components_that_survive_zeroing_move_up_and_keep_their_subspace():
    """PC1 on S, PCs 2 and 3 off S: after zeroing, the old PC2 is the new PC1."""
    d = 12
    tr, te = _planted([_unit(d, (0, 1)), _unit(d, (4, 5)), _unit(d, (8, 9))],
                      [30.0, 20.0, 10.0])
    mu, w, ev = e2.pc_basis(tr)
    np.testing.assert_allclose(np.abs(w[:3]),
                               [_unit(d, c) / np.sqrt(2) for c in ((0, 1), (4, 5), (8, 9))],
                               atol=1e-9)
    np.testing.assert_allclose(ev[:3] / len(tr), [900.0, 400.0, 100.0], rtol=1e-9)
    g = e2.zeroing_geometry(tr, te, [0, 1], (mu, w, ev))
    assert [g[f"load_pc{j}"] for j in (1, 2, 3)] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert g["var_left_pc1"] == pytest.approx(0.0, abs=1e-12)
    assert g["var_left_pc2"] == pytest.approx(1.0, abs=1e-12)
    assert g["var_left_pc3"] == pytest.approx(1.0, abs=1e-12)
    assert g["var_share_zeroed_pc2"] == pytest.approx(0.8, abs=1e-12)
    assert g["var_share_zeroed_pc3"] == pytest.approx(0.2, abs=1e-12)
    assert np.isnan(g["remainder_share_pc1"])
    assert g["remainder_share_pc2"] == pytest.approx(0.8, abs=1e-12)
    assert g["remainder_share_pc3"] == pytest.approx(0.2, abs=1e-12)
    assert g["total_var_left"] == pytest.approx(500.0 / 1400.0, abs=1e-12)
    # one to one the components no longer match, but the new PC1 is the old PC2
    assert g["angle_pc1"] == pytest.approx(90.0, abs=1e-4)
    assert g["abscos_newpc1_oldpc2"] == pytest.approx(1.0, abs=1e-12)
    assert g["abscos_newpc1_oldpc3"] == pytest.approx(0.0, abs=1e-9)
    # two of the three old directions are still inside the span of the new top three
    assert g["pangle_1"] == pytest.approx(0.0, abs=1e-4)
    assert g["pangle_2"] == pytest.approx(0.0, abs=1e-4)
    # zeroing off-S coordinates instead leaves every component where it was
    g = e2.zeroing_geometry(tr, te, [2, 3], (mu, w, ev))
    assert [g[f"angle_pc{j}"] for j in (1, 2, 3)] == pytest.approx([0.0] * 3, abs=1e-4)
    assert [g[f"pangle_{j}"] for j in (1, 2, 3)] == pytest.approx([0.0] * 3, abs=1e-4)
    assert [g[f"var_left_pc{j}"] for j in (1, 2, 3)] == pytest.approx([1.0] * 3, abs=1e-12)
    for j in (1, 2, 3):
        assert g[f"pearson_train_pc{j}"] == pytest.approx(1.0, abs=1e-9)
        assert g[f"spearman_test_pc{j}"] == pytest.approx(1.0, abs=1e-9)


def test_zeroing_geometry_is_the_geometry_of_the_zeroed_raw_vectors():
    """Zeroing the centered columns equals zeroing the raw vectors and centering them."""
    tr, te, *_ = _synthetic()
    idx = e1.rank_coords(tr, "variance")[:3]
    g = e2.zeroing_geometry(tr, te, idx)
    za, _ = e1.zero_coords(tr, te, idx)
    z = za.astype(np.float64)
    _, _, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    _, w, _ = e2.pc_basis(tr)
    for j in range(3):
        assert g[f"angle_pc{j + 1}"] == pytest.approx(
            np.degrees(np.arccos(min(1.0, abs(float(w[j] @ vt[j]))))), abs=1e-6)
    assert not vt[:3][:, idx].any() or np.abs(vt[:3][:, idx]).max() < 1e-12  # v_j is 0 on S
    # the remainder cosine is bounded by 1, and the variance left by the unzeroed variance
    assert 0.0 <= g["remainder_cos_pc1"] <= 1.0
    assert 0.0 <= g["total_var_left"] <= 1.0
    scores = (z - z.mean(axis=0)) @ w.T
    x = tr.astype(np.float64)
    before = ((x - x.mean(axis=0)) @ w.T).var(axis=0)
    for j in range(e2.N_PC):
        assert g[f"var_left_pc{j + 1}"] == pytest.approx(scores[:, j].var() / before[j],
                                                         rel=1e-9)
        # PC j's own scores contribute (1 - loading)^2; the rest comes from the other
        # components, so the ratio has this floor and no ceiling at 1
        assert g[f"var_left_pc{j + 1}"] >= (1 - g[f"load_pc{j + 1}"]) ** 2 - 1e-12
    # no direction of the zeroed vectors holds more than their top principal component
    top = e1._geometry(za)["pc1_share_train"]
    for j in (1, 2, 3):
        assert g[f"remainder_share_pc{j}"] == pytest.approx(
            g[f"var_share_zeroed_pc{j}"] / (1 - g[f"load_pc{j}"]), rel=1e-12)
        assert g[f"remainder_share_pc{j}"] <= top + 1e-6
    # the loading of PC1 on the ten top-variance coordinates is E1's concentration cell
    g10 = e2.zeroing_geometry(tr, te, e1.rank_coords(tr, "variance")[:10])
    assert g10["load_pc1"] == pytest.approx(e1.concentration(tr)["pc1_mass_top10var"],
                                            abs=1e-12)


def test_geometry_columns_are_nan_for_components_that_do_not_exist():
    rng = np.random.default_rng(0)
    tr, te = rng.normal(size=(5, 30)), rng.normal(size=(4, 30))  # at most 5 components
    g = e2.zeroing_geometry(tr, te, [0])
    assert np.isfinite(g["load_pc5"]) and np.isnan(g["load_pc6"]) and np.isnan(g["load_pc10"])
    assert np.isnan(g["var_left_pc10"]) and np.isfinite(g["angle_pc3"])


# --------------------------------------------------------------------------- #
# angles, signs, correlations
# --------------------------------------------------------------------------- #

def test_principal_angles_of_identical_orthogonal_and_tilted_subspaces():
    eye = np.eye(6)
    a = eye[:3]
    np.testing.assert_allclose(e2.principal_angles(a, a), 0.0, atol=1e-6)
    # a rotation of the basis within the span changes the vectors, not the subspace
    rot, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    np.testing.assert_allclose(e2.principal_angles(a, rot @ a), 0.0, atol=1e-5)
    assert e2.pc_angles(a, rot @ a).max() > 1.0
    np.testing.assert_allclose(e2.principal_angles(a, eye[3:]), 90.0, atol=1e-9)
    theta = np.radians(25.0)
    b = np.array([eye[0], eye[1], np.cos(theta) * eye[2] + np.sin(theta) * eye[3]])
    np.testing.assert_allclose(e2.principal_angles(a, b), [0.0, 0.0, 25.0], atol=1e-5)
    np.testing.assert_allclose(e2.pc_angles(a, b), [0.0, 0.0, 25.0], atol=1e-5)
    np.testing.assert_allclose(e2.pc_angles(a, -b), [0.0, 0.0, 25.0], atol=1e-5)


def test_the_sign_of_a_component_does_not_change_the_measures():
    w = np.eye(5)[:3]
    v = np.array([-w[0], w[1], -w[2]])
    np.testing.assert_array_equal(e2.align_signs(w, v), w)
    np.testing.assert_array_equal(e2.align_signs(w, w), w)
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    assert e2.score_corr(a, -a) == pytest.approx((-1.0, -1.0))
    assert e2.remainder_cosine(w[0], -w[0], 0.0) == 1.0
    assert e2.remainder_cosine(w[0], 0.6 * w[0] + 0.8 * w[1], 0.0) == pytest.approx(0.6)
    assert e2.remainder_cosine(np.array([0.6, 0.8]), np.array([0.0, -1.0]), 0.36) == 1.0
    assert np.isnan(e2.remainder_cosine(w[0], w[1], 1.0))


def test_score_corr_is_pearson_and_rank_pearson():
    from scipy.stats import pearsonr, spearmanr

    rng = np.random.default_rng(3)
    a = rng.normal(size=300)
    b = np.exp(a) + 0.5 * rng.normal(size=300)
    b[:40] = np.round(b[:40])  # ties
    p, s = e2.score_corr(a, b)
    assert p == pytest.approx(pearsonr(a, b)[0], abs=1e-12)
    assert s == pytest.approx(spearmanr(a, b)[0], abs=1e-12)
    # a monotone map keeps Spearman at 1 and lowers Pearson
    p, s = e2.score_corr(a, np.exp(3 * a))
    assert s == pytest.approx(1.0, abs=1e-12) and p < 0.9
    assert all(np.isnan(v) for v in e2.score_corr(a, np.zeros(300)))


# --------------------------------------------------------------------------- #
# removing chosen components
# --------------------------------------------------------------------------- #

def test_remove_pcs_is_abtt_for_the_first_components_and_keeps_the_others():
    tr, te, *_ = _synthetic()
    a, b = e2.remove_pcs(tr, te, (1, 2, 3))
    ra, rb = asw.abtt(tr, te, 3)
    np.testing.assert_allclose(a, ra, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b, rb, rtol=0, atol=1e-5)
    # PCs 2 and 3 removed, PC1 kept: the scores on PC1 are those of the centered vectors
    s_tr, s_te, _ = asw.pc_scores(tr, te, 3)
    cleaner = asw.EmbeddingCleaner(num_components=3, center=True).fit(tr)
    a, b = e2.remove_pcs(tr, te, (2, 3))
    np.testing.assert_allclose(a @ cleaner.pcs[0], s_tr[:, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(b @ cleaner.pcs[0], s_te[:, 0], rtol=1e-4, atol=1e-3)
    assert np.abs(a @ cleaner.pcs[1:].T).max() < 1e-3 < np.abs(s_tr[:, 1:]).max()
    assert np.abs(b @ cleaner.pcs[1:].T).max() < 1e-3
    # fit on train only
    a2, _ = e2.remove_pcs(tr, te * 3.0 - 7.0, (2, 3))
    np.testing.assert_array_equal(a, a2)


# --------------------------------------------------------------------------- #
# layer rows
# --------------------------------------------------------------------------- #

def test_layer_rows_zero_the_sets_of_e1_and_reproduce_its_cells():
    tr, te, tr_ids, te_ids = _synthetic()
    fn = _auroc_fn(tr_ids, te_ids)
    rows = pd.DataFrame(e2.layer_rows(LATA, 1, tr, te, fn))
    assert len(rows) == len(e2.RANKINGS) * len(e2.KS)
    assert list(zip(rows.ranking, rows.k)) == [(r, k) for r in e1.RANKINGS for k in e1.KS]
    abl = pd.DataFrame(e1.layer_rows(LATA, 1, tr, te, fn)[0]).set_index("tag")
    for _, x in rows.iterrows():
        ref = abl.loc[e1.zero_tag(x.ranking, x.k)]
        assert x.coords == ref.coords
        assert x.auc_zero == ref.aucroc
        assert x.pc1_share_train == ref.pc1_share_train
        assert x.eff_rank_train == ref.eff_rank_train
        assert x.ref_auc_base == abl.loc["base", "aucroc"]
        for D in (1, 3, 10):
            assert x[f"ref_auc_abtt_D{D}"] == abl.loc[f"abtt_D{D}", "aucroc"]
    # the per-layer reference cells repeat on every row
    for col in [c for c in rows.columns if c.startswith("ref_")]:
        assert rows[col].nunique() == 1
    # the intervention cells are the named functions, scored by the metric
    x = rows[(rows.ranking == "variance") & (rows.k == 3)].iloc[0]
    za, zb = e1.zero_coords(tr, te, e1.rank_coords(tr, "variance")[:3])
    for D in e2.ZERO_ABTT_D:
        assert x[f"auc_zero_abtt_D{D}"] == fn(*asw.abtt(za, zb, D))["aucroc"]
    assert x.ref_auc_abtt_D2 == fn(*asw.abtt(tr, te, 2))["aucroc"]
    assert x.ref_auc_rm_pc23 == fn(*e2.remove_pcs(tr, te, (2, 3)))["aucroc"]
    g = e2.zeroing_geometry(tr, te, e1.rank_coords(tr, "variance")[:3])
    for col, v in g.items():
        assert x[col] == pytest.approx(v, abs=1e-12)
    # the planted nuisance coordinate is PC1 and the first coordinate by variance: zeroing
    # it removes PC1 and repairs retrieval, zeroing the shared offset does neither
    k1 = rows[rows.k == 1].set_index("ranking")
    assert k1.loc["variance", "coords"] == "0" and k1.loc["mean_abs", "coords"] == "1"
    assert k1.loc["variance", "load_pc1"] > 0.99 and k1.loc["variance", "var_left_pc1"] < 1e-3
    assert k1.loc["mean_abs", "load_pc1"] < 0.01 and k1.loc["mean_abs", "var_left_pc1"] > 0.98
    assert k1.loc["mean_abs", "angle_pc1"] < 1.0 < 80.0 < k1.loc["variance", "angle_pc1"]
    assert k1.loc["mean_abs", "pearson_train_pc1"] > 0.999
    assert x.ref_auc_base < 0.75 < 0.9 < k1.loc["variance", "auc_zero"]
    assert x.ref_var_share_pc1 > 0.9 and x.dim == 24 and x.n_train == 60 and x.n_test == 50
    shares = rows[e2.REF_SHARE_COLS].iloc[0].to_numpy()
    assert (np.diff(shares) <= 0).all() and shares.sum() <= 1.0


def test_layer_rows_fit_nothing_on_test():
    tr, te, tr_ids, te_ids = _synthetic()
    rng = np.random.default_rng(9)
    # not an affine map of the test rows: a correlation would not see one
    te_other = (te[rng.permutation(len(te))] * -2.0 + 11.0
                + 5.0 * rng.normal(size=te.shape)).astype(np.float32)
    fn = _auroc_fn(tr_ids, te_ids)
    r1 = pd.DataFrame(e2.layer_rows(LATA, 3, tr, te, fn))
    r2 = pd.DataFrame(e2.layer_rows(LATA, 3, tr, te_other, fn))
    test_cols = [c for c in r1.columns
                 if c.startswith(("auc_", "ref_auc_", "pearson_test", "spearman_test"))]
    train_cols = [c for c in r1.columns if c not in test_cols]
    assert {"coords", "load_pc1", "var_left_pc10", "angle_pc3", "pangle_3",
            "remainder_cos_pc1", "remainder_share_pc3", "pearson_train_pc1",
            "pc1_share_train", "ref_var_share_pc1"} <= set(train_cols)
    pd.testing.assert_frame_equal(r1[train_cols], r2[train_cols])
    # and the test readouts do move, so the comparison above is not vacuous
    assert not np.allclose(r1["auc_zero"], r2["auc_zero"])
    assert not np.allclose(r1["pearson_test_pc1"], r2["pearson_test_pc1"])


def test_layer_rows_run_through_the_paper_metric_block():
    tr, te, tr_ids, te_ids = _synthetic()
    counts = pd.Series(te_ids).value_counts()
    saved = dict(asw._CTX)
    asw._CTX.update(tr_ids=tr_ids, te_ids=te_ids,
                    te_partner=np.array([counts[i] > 1 for i in te_ids]))
    try:
        rows = pd.DataFrame(e2.layer_rows(LATA, 1, tr, te, asw._metrics))
    finally:
        asw._CTX.clear()
        asw._CTX.update(saved)
    ref = pd.DataFrame(e2.layer_rows(LATA, 1, tr, te, _auroc_fn(tr_ids, te_ids)))
    auc = [c for c in rows.columns if c.startswith(("auc_", "ref_auc_"))]
    assert len(auc) == 4 + 6
    np.testing.assert_allclose(rows[auc].to_numpy(), ref[auc].to_numpy(), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def _gt(*args, **kwargs):
    """gate_table for the two-model fixture (its default expects all six panel models)."""
    kwargs.setdefault("expected", [LATA, LABSE])
    return e2.gate_table(*args, **kwargs)


def _failing(g):
    return sorted((row.model, row.gate[:1]) for row in g[~g.ok].itertuples())


def _edit(frame, mask, col, value=None, by=None):
    out = frame.copy()
    assert mask.sum() == 1
    out.loc[mask, col] = value if by is None else out.loc[mask, col] + by
    return out


def test_gates_pass_on_matching_references_and_fail_on_drift():
    rows, abl = _fixture()
    h1 = _h1(rows)
    g = _gt(rows, abl, h1)
    assert len(g) == 4 * 2 and g.ok.all()
    assert list(g.gate.str[:1]) == ["1", "2", "3", "4"] * 2
    assert list(g[g.gate.str.startswith("1")].n_cells) == [16, 16]  # 2 layers x 8 sets
    assert list(g[g.gate.str.startswith("4")].n_cells) == [10, 10]  # base and D=1,2,3,10
    numeric = g[~g.gate.str.startswith("1")]
    assert (numeric.max_abs_diff == 0).all() and (numeric.tolerance == e2.GATE_TOL).all()
    assert (g.n_over_tolerance == 0).all() and (g.n_missing_reference == 0).all()

    zero = (abl.model == LATA) & (abl.layer == 2) & (abl.tag == "zero_variance_k3")
    # a different coordinate set: gate 1, named with both sets
    g = _gt(rows, _edit(abl, zero, "coords", "0;1;2"), h1)
    assert _failing(g) == [(LATA, "1")]
    bad = g[~g.ok].iloc[0]
    ours = rows[(rows.model == LATA) & (rows.layer == 2) & (rows.ranking == "variance")
                & (rows.k == 3)].coords.iloc[0]
    assert bad.n_over_tolerance == 1
    assert bad.cells_over_tolerance == f"L2 variance k=3 {ours} (E1 0;1;2)"
    assert "1 differ: FAIL; sets that differ: L2 variance k=3" in e2.gate_line(bad)
    # the same coordinates in another order are not the same cell of E1
    swapped = ";".join(reversed(ours.split(";")))
    assert _failing(_gt(rows, _edit(abl, zero, "coords", swapped), h1)) == [(LATA, "1")]
    # a drifted zero-only AUROC: gate 2; a looser tolerance is an explicit argument
    drift = _edit(abl, zero, "aucroc", by=5e-6)
    g = _gt(rows, drift, h1)
    assert _failing(g) == [(LATA, "2")]
    assert g[~g.ok].iloc[0].cells_over_tolerance == "L2 variance k=3 5.00e-06"
    assert _gt(rows, drift, h1, tol=1e-5).ok.all()
    # a drifted top-PC share: gate 3
    g = _gt(rows, _edit(abl, zero, "pc1_share_train", by=-5e-6), h1)
    assert _failing(g) == [(LATA, "3")]
    # a drifted H1 cell: gate 4, for D=2 as for the others
    for variant, D, lab in (("abtt", 2, "ABTT D=2"), ("abtt", 10, "ABTT D=10"),
                            ("raw", -1, "base")):
        cell = (h1.model == LABSE) & (h1.layer == 1) & (h1.variant == variant) & (h1.D == D)
        g = _gt(rows, abl, _edit(h1, cell, "aucroc", by=5e-6))
        assert _failing(g) == [(LABSE, "4")]
        assert g[~g.ok].iloc[0].cells_over_tolerance == f"L1 {lab} 5.00e-06"
    # the centering cell of H1 is not gated
    cell = (h1.model == LABSE) & (h1.layer == 1) & (h1.variant == "center")
    assert _gt(rows, abl, _edit(h1, cell, "aucroc", by=0.1)).ok.all()


def test_missing_rows_and_nan_cells_fail_their_gates():
    rows, abl = _fixture()
    h1 = _h1(rows)
    zero = (abl.model == LATA) & (abl.layer == 1) & (abl.tag == "zero_mean_abs_k10")
    # an E1 row that is not there is a failure of gates 1 to 3, never a silent skip
    g = _gt(rows, abl[~zero], h1)
    assert _failing(g) == [(LATA, "1"), (LATA, "2"), (LATA, "3")]
    assert (g[~g.ok].n_missing_reference == 1).all()
    assert "missing reference cells 1" in e2.gate_line(g[~g.ok].iloc[0])
    # an H1 cell that is not there fails gate 4
    cell = (h1.model == LATA) & (h1.layer == 2) & (h1.variant == "abtt") & (h1.D == 3)
    assert _failing(_gt(rows, abl, h1[~cell])) == [(LATA, "4")]
    # a follow-up row that is not there fails unless a layer subset was asked for
    mine = rows[~((rows.model == LABSE) & (rows.layer == 2))]
    g = _gt(mine, abl, h1)
    assert _failing(g) == [(LABSE, "1"), (LABSE, "2"), (LABSE, "3"), (LABSE, "4")]
    assert list(g[~g.ok].n_reference_rows_absent) == [8, 8, 8, 1]
    assert _gt(mine, abl, h1, complete=False).ok.all()
    # NaN on our side or on the reference side
    ours = (rows.model == LATA) & (rows.layer == 1) & (rows.ranking == "mean_abs") & (rows.k == 10)
    g = _gt(_edit(rows, ours, "auc_zero", np.nan), abl, h1)
    assert _failing(g) == [(LATA, "2")]
    bad = g[~g.ok].iloc[0]
    assert np.isnan(bad.max_abs_diff) and bad.cells_over_tolerance == "L1 mean_abs k=10 nan"
    assert _failing(_gt(rows, _edit(abl, zero, "pc1_share_train", np.nan), h1)) == [(LATA, "3")]
    assert _failing(_gt(rows, abl, _edit(h1, cell, "aucroc", np.nan))) == [(LATA, "4")]
    # duplicated rows are an error, not a comparison
    with pytest.raises(ValueError):
        _gt(pd.concat([rows, rows.iloc[:1]], ignore_index=True), abl, h1)


def test_an_expected_model_without_rows_fails_the_gates():
    rows, abl = _fixture()
    h1 = _h1(rows)
    only = rows[rows.model == LATA]
    g = e2.gate_table(only, abl, h1, expected=[LATA, LABSE])
    missing = g[~g.ok]
    assert len(missing) == 1 and missing.iloc[0].model == LABSE
    assert missing.iloc[0].gate.startswith("0") and missing.iloc[0].n_cells == 0
    assert "no rows: FAIL" in e2.gate_line(missing.iloc[0])
    assert e2.gate_table(only, abl, h1, expected=[LATA]).ok.all()
    assert e2.gate_table(only, abl, h1, expected=None).ok.all()
    g = e2.gate_table(rows, abl, h1)  # the default expects the whole panel
    assert sorted(g[g.gate.str.startswith("0")].model) == sorted(
        m for m in e2.ALL_MODEL_IDS if m not in (LATA, LABSE))
    assert g[~g.gate.str.startswith("0")].ok.all()


def _write_fixture(tmp_path):
    """The fixture as CSV files: (out_dir, reference arguments)."""
    rows, abl = _fixture()
    out = tmp_path / "e2"
    out.mkdir()
    rows.to_csv(out / e2.CSV_NAME, index=False, float_format="%.10g")
    abl.to_csv(tmp_path / "e1.csv", index=False, float_format="%.10g")
    _h1(e2.read_rows(out / e2.CSV_NAME)).to_csv(tmp_path / "h1.csv", index=False)
    return out, ["--e1_csv", str(tmp_path / "e1.csv"), "--h1_csv", str(tmp_path / "h1.csv")]


def test_check_cli_requires_every_model_and_is_idempotent(tmp_path, capsys):
    out, refs = _write_fixture(tmp_path)
    base = ["check", "--out_dir", str(out), *refs]
    # four of the six panel models have no rows
    assert e2.main(base) == e2.GATE_EXIT
    said = capsys.readouterr().out
    assert "KaLM-mini: no rows: FAIL" in said and "REPRODUCTION GATES FAILED" in said
    assert not pd.read_csv(out / e2.GATE_NAME).ok.all()
    assert e2.main(base + ["--allow_missing"]) == 0
    assert e2.main(base + ["--models", "LaTa,LaBSE"]) == 0
    first = (out / e2.GATE_NAME).read_bytes()
    assert e2.main(base + ["--models", "LaTa,LaBSE"]) == 0
    assert (out / e2.GATE_NAME).read_bytes() == first
    assert e2.main(base + ["--models", "LaTa,LaBSE", "--no_write"]) == 0
    written = pd.read_csv(out / e2.GATE_NAME)
    assert written.ok.all() and (written.max_abs_diff.dropna() == 0).all()
    # the CSV round trip keeps a single zeroed coordinate as a string
    back = e2.read_rows(out / e2.CSV_NAME)
    assert back.coords.map(type).eq(str).all() and "0" in set(back.coords)


# --------------------------------------------------------------------------- #
# compute plumbing
# --------------------------------------------------------------------------- #

def _fake_cache(tmp_path, layers=(1, 2)):
    """A tiny split CSV and a LaTa cache whose row order differs from the split's."""
    tr, te, tr_ids, te_ids = _synthetic()
    n = len(tr) + len(te)
    names = [f"f{i:03d}.txt" for i in range(n)]
    counts = pd.Series(te_ids).value_counts()
    split = pd.DataFrame({
        "filename": names, "folder_id": list(tr_ids) + list(te_ids),
        "split": ["train"] * len(tr) + ["test"] * len(te),
        "has_test_partner": [False] * len(tr) + [bool(counts[i] > 1) for i in te_ids]})
    split.to_csv(tmp_path / "split.csv", index=False)
    run = tmp_path / "bases" / "phase9_bases" / asw.slug(LATA) / asw.SUBDIR
    run.mkdir(parents=True)
    order = np.random.default_rng(11).permutation(n)
    pd.DataFrame({"path": [f"data/x/{names[i]}" for i in order]}).to_csv(run / "meta.csv",
                                                                       index=False)
    for layer in layers:
        x = np.vstack([tr, te]).copy()
        x[:, 0] *= layer  # a different nuisance scale per layer
        np.save(run / f"hidden_layer{layer}_embeddings.npy", x[order])
    return tmp_path / "split.csv", tmp_path / "bases"


def test_compute_check_gates_the_written_csv_against_an_e1_run_of_the_same_cache(tmp_path,
                                                                                capsys):
    """compute --check, check and a second compute agree byte for byte, and the zeroed sets
    and gated cells equal those of an E1 compute on the same cache."""
    split_csv, bases = _fake_cache(tmp_path)
    out, e1_out = tmp_path / "e2", tmp_path / "e1"
    common = ["--models", "LaTa", "--workers", "1", "--bases_root", str(bases),
              "--split_csv", str(split_csv)]
    run = ["compute", "--out_dir", str(out), *common]
    saved = dict(asw._CTX)
    try:
        assert e1.main(["compute", "--out_dir", str(e1_out), *common]) == 0
        assert e2.main(run) == 0  # no --check: the H1 reference does not exist yet
        first = (out / e2.CSV_NAME).read_bytes()
        _h1(e2.read_rows(out / e2.CSV_NAME)).to_csv(tmp_path / "h1.csv", index=False)
        refs = ["--e1_csv", str(e1_out / e1.ABL_NAME), "--h1_csv", str(tmp_path / "h1.csv")]
        assert e2.main(run + ["--check", *refs]) == 0
    finally:
        asw._CTX.clear()
        asw._CTX.update(saved)
    assert (out / e2.CSV_NAME).read_bytes() == first  # compute is deterministic
    rows = e2.read_rows(out / e2.CSV_NAME)
    assert sorted(rows.layer.unique()) == [1, 2] and set(rows.model) == {LATA}
    assert len(rows) == 2 * len(e2.RANKINGS) * len(e2.KS)
    gate_file = (out / e2.GATE_NAME).read_bytes()
    written = pd.read_csv(out / e2.GATE_NAME)
    assert written.ok.all() and (written.max_abs_diff.dropna() == 0).all()
    capsys.readouterr()
    check = ["check", "--out_dir", str(out), "--models", "LaTa", *refs]
    assert e2.main(check) == 0 and (out / e2.GATE_NAME).read_bytes() == gate_file
    # a layer subset is gated without requiring the other layers
    assert e2.main(run + ["--layers", "2", "--check", *refs]) == 0
    assert sorted(e2.read_rows(out / e2.CSV_NAME).layer.unique()) == [2]
    assert e2.main(check) == e2.GATE_EXIT  # check expects every layer of E1 and H1


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

def test_render_cli_writes_the_facts_file_from_a_fixture(tmp_path, capsys):
    out, refs = _write_fixture(tmp_path)
    conc = []
    for mid, nuisance in ((LATA, 40.0), (LABSE, 0.0)):
        for layer in (1, 2):
            tr, *_ = _synthetic(seed=layer, nuisance=nuisance * layer)
            conc.append({"model": mid, "layer": layer, **e1.concentration(tr)})
    pd.DataFrame(conc).to_csv(tmp_path / "conc.csv", index=False, float_format="%.10g")
    argv = ["render", "--out_dir", str(out), "--models", "LaTa,LaBSE"]
    # without the reference CSVs the gates are not evaluated and nothing else breaks
    assert e2.main(argv + ["--e1_csv", str(tmp_path / "none.csv"), "--h1_csv",
                           str(tmp_path / "none.csv"), "--e1_conc_csv",
                           str(tmp_path / "none.csv")]) == 0
    early = (out / e2.FACTS_NAME).read_text()
    assert "- gates: reference CSVs not found, not evaluated" in early
    assert "consistency with E1" not in early
    capsys.readouterr()
    assert e2.main(argv + refs + ["--e1_conc_csv", str(tmp_path / "conc.csv")]) == 0
    said = capsys.readouterr().out
    for name in ("PhilTa", "mT5-base", "Qwen3-0.6B", "KaLM-mini"):
        assert f"omitting {name}" in said
    facts = (out / e2.FACTS_NAME).read_text()
    assert chr(0x2014) not in facts and chr(0x2013) not in facts
    assert early.split("## 1.")[1] == facts.split("## 1.")[1]  # only section 0 differs
    assert "ABSENT from the CSV" in facts and "KaLM-mini" in facts
    # the header says what is post hoc and that no verdict is printed outside the gates
    head = facts.split("## Definitions")[0]
    assert "The whole follow-up was designed after the E1 results were read." in head
    assert "were added post hoc: no prediction is attached to them" in head
    body = facts.split("## 1.")[1]
    assert "PASS" not in body and "FAIL" not in body
    sec3 = facts.split("## 3. Intervention cells")[1].split("## 4.")[0]
    assert "(post hoc, no prediction)" in facts and "No prediction is attached" in sec3
    # gates and the consistency line
    gates = e2.gate_table(e2.read_rows(out / e2.CSV_NAME), e1.read_abl(tmp_path / "e1.csv"),
                          pd.read_csv(tmp_path / "h1.csv"), expected=[LATA, LABSE])
    assert len(gates) == 8
    for row in gates.itertuples():
        assert f"- {e2.gate_line(row)}" in facts
    m = re.search(r"consistency with E1 \(not a gate\).*: 4 model-layers, max \|diff\| (\S+)",
                  facts)
    assert m and float(m.group(1)) < 1e-9
    # every section is there, in order
    marks = ["## Definitions", "## 0. Coverage and reproduction gates",
             "## 1. Collapsed T5 layers, k=10, ranking by variance",
             "### 1.1 Loading", "### 1.2 Variance left", "### 1.3 Angles",
             "### 1.4 Correlation", "### 1.5 Top-PC share", "### 1.6 The counts in one place",
             "## 2. Collapsed T5 layers, k=10, ranking by mean |x| (brief)",
             "## 3. Intervention cells", "## 4. Smaller k", "## 5. Single layers",
             "## 6. Layers that are not collapsed", "## 7. All layers", "### LaTa", "### LaBSE"]
    pos = [facts.index(s) for s in marks]
    assert pos == sorted(pos)
    assert "- model-layers: LaTa 2, LaBSE 2 (total 4); rows: 32" in facts
    assert "- 2 layers: LaTa 2 (1, 2)" in facts
    assert "- LaTa L6: no rows" in facts and "- mT5-base L5: no rows" in facts
    # the counts and medians are those of the CSV
    d = e2.prepare(e2.read_rows(out / e2.CSV_NAME))
    coll = e2.at(d)[lambda t: t.collapsed]
    assert list(coll.m) == ["LaTa", "LaTa"]
    assert not e2.at(d)[lambda t: t.m == "LaBSE"].collapsed.any()
    assert (f"- PC1 loading on the zeroed coordinates below 0.5 at "
            f"{int((coll.load_pc1 < 0.5).sum())}/2; PC2 at "
            f"{int((coll.load_pc2 < 0.5).sum())}/2; PC3 at "
            f"{int((coll.load_pc3 < 0.5).sum())}/2") in facts
    assert (f"- angle between old and new PC1 below 30 degrees at "
            f"{int((coll.angle_pc1 < 30).sum())}/2") in facts
    assert (f"- |Pearson| of old and new PC1 scores above 0.9 at "
            f"{int((coll.pearson_train_pc1.abs() > 0.9).sum())}/2 (train passages)") in facts
    s = coll.load_pc1
    assert (f"  - PC1: {s.median():.3f} ({s.min():.3f} to {s.max():.3f}); LaTa "
            f"{s.median():.3f}") in facts
    for label, col in (("zero + ABTT D=1", "auc_zero_abtt_D1"),
                       ("PCs 2 and 3 removed, PC1 kept", "ref_auc_rm_pc23"),
                       ("ABTT D=3", "ref_auc_abtt_D3")):
        s = coll[col]
        assert (f"- {label}: AUROC >= 0.90 at {int((s >= 0.9).sum())}/2; {s.median():.3f} "
                f"({s.min():.3f} to {s.max():.3f}); LaTa {int((s >= 0.9).sum())}/2") in sec3
    # the per-layer table marks the collapsed layers and prints the cells of the CSV
    x = e2.at(d)[lambda t: (t.m == "LaTa") & (t.layer == 2)].iloc[0]
    row = next(ln for ln in facts.split("### LaTa")[1].splitlines() if ln.startswith("| 2*"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert len(cells) == 1 + len(e2.TABLE_COLS)
    assert cells[1:] == [format(x[col], fmt) for _, col, fmt in e2.TABLE_COLS]
    assert any(ln.startswith("| 1 |") for ln in facts.split("### LaBSE")[1].splitlines())
    # rendering twice gives the same bytes
    again = tmp_path / "again.md"
    assert e2.main(argv + refs + ["--e1_conc_csv", str(tmp_path / "conc.csv"), "--facts_md",
                                  str(again)]) == 0
    assert again.read_text() == facts


def test_rng_and_counts_leave_nan_cells_out_and_say_so():
    s = pd.Series([0.2, np.nan, 0.6])
    assert e2._rng(s) == "0.400 (0.200 to 0.600) [1 NaN left out]"
    assert e2._rng(pd.Series([np.nan, np.nan])) == "nan"
    assert e2._rng(pd.Series([10.0, 30.0]), ".1f") == "20.0 (10.0 to 30.0)"
    assert e2._cnt(s > 0.5) == "1/3"


# --------------------------------------------------------------------------- #
# committed results (skipped when the result CSV is not checked out)
# --------------------------------------------------------------------------- #

E2_DIR = REPO_ROOT / "runs" / "active" / "reframe" / "e2"
REFS = [REPO_ROOT / e2.E1_CSV, REPO_ROOT / e2.H1_CSV]
needs_results = pytest.mark.skipif(not (E2_DIR / e2.CSV_NAME).exists(),
                                   reason="runs/active/reframe/e2 follow-up CSV not checked out")


@needs_results
def test_committed_results_pass_the_reproduction_gates(tmp_path):
    if not all(p.exists() for p in REFS):
        pytest.skip("reference CSVs not checked out")
    rows = e2.read_rows(E2_DIR / e2.CSV_NAME)
    g = e2.gate_table(rows, e1.read_abl(REFS[0]), pd.read_csv(REFS[1]))
    assert len(g) == 4 * len(e2.ALL_MODEL_IDS) and g.ok.all(), g[~g.ok].to_string()
    assert (g[~g.gate.str.startswith("1")].tolerance == 1e-6).all()
    # check must leave the committed gate file untouched: regenerate it and compare bytes
    e2.write_gates(g, tmp_path / "gates.csv")
    assert (tmp_path / "gates.csv").read_bytes() == (E2_DIR / e2.GATE_NAME).read_bytes()
    # one row per (model-layer, ranking, k), and bounded measures stay in their bounds
    assert not rows.duplicated(["model", "layer", "ranking", "k"]).any()
    assert len(rows) == len(rows[["model", "layer"]].drop_duplicates()) * 8
    for cols in (e2.LOAD_COLS, e2.ZSHARE_COLS, ["total_var_left"]):
        v = rows[cols].to_numpy()
        assert (v >= -1e-9).all() and (v <= 1 + 1e-9).all()
    # the variance left has the floor (1 - loading)^2 and no ceiling at 1
    floor = (1 - rows[e2.LOAD_COLS].to_numpy()) ** 2
    assert (rows[e2.LEFT_COLS].to_numpy() >= floor - 1e-6).all()
    rem = rows["remainder_cos_pc1"].dropna()
    assert ((rem >= 0) & (rem <= 1)).all()
    for j in (1, 2, 3):
        ok = rows[f"remainder_share_pc{j}"].notna()
        assert (rows.loc[ok, f"remainder_share_pc{j}"]
                <= rows.loc[ok, "pc1_share_train"] + 1e-4).all()
    ang = rows[[f"pangle_{j}" for j in (1, 2, 3)]].to_numpy()
    assert (np.diff(ang, axis=1) >= -1e-9).all() and (ang >= 0).all() and (ang <= 90).all()


@needs_results
def test_committed_facts_regenerate_byte_identically(tmp_path, capsys):
    committed = E2_DIR / e2.FACTS_NAME
    conc = REPO_ROOT / e2.E1_CONC_CSV
    if not (committed.exists() and conc.exists() and all(p.exists() for p in REFS)):
        pytest.skip("facts file or its inputs not checked out")
    rc = e2.main(["render", "--out_dir", str(E2_DIR), "--facts_md", str(tmp_path / "facts.md"),
                  "--e1_csv", str(REFS[0]), "--h1_csv", str(REFS[1]),
                  "--e1_conc_csv", str(conc)])
    capsys.readouterr()
    assert rc == 0
    assert (tmp_path / "facts.md").read_text() == committed.read_text()
