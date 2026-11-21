"""Data-free checks for the external fixes scoring and transforms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "paper" / "reframe"))
import external_fixes as ef  # noqa: E402


def test_loo_keeps_singleton_candidates_and_excludes_self():
    labels = ef.ScoreLabels.from_ids(np.array(["a", "a", "b", "b", "singleton"]))
    sim = np.array([
        [1., .5, .1, .05, .8],
        [.5, 1., .2, .3, .4],
        [.1, .2, 1., .1, .2],
        [.05, .3, .1, 1., .4],
        [.8, .4, .2, .4, 1.],
    ])
    # Partner ranks are 2, 1, 4, 3. In row 2 the earlier tied candidate wins.
    acc1, mrr = ef.loo_metrics(sim, labels)
    assert acc1 == .25
    assert mrr == pytest.approx((1 / 2 + 1 + 1 / 4 + 1 / 3) / 4)


def test_component_sd_matches_sklearn_whitening():
    rng = np.random.default_rng(23)
    fit = rng.normal(size=(20, 6)).astype(np.float32)
    test = rng.normal(size=(9, 6))
    centered = fit.astype(np.float64) - fit.mean(axis=0, dtype=np.float64)
    pcs, var = ef.pca(centered)
    rank = 4
    scores = (test - fit.mean(axis=0, dtype=np.float64)) @ pcs.T
    actual = ef.component_standardize(scores, var, rank, by_variance=False)
    pca = PCA(n_components=rank, whiten=True, svd_solver="full").fit(fit.astype(np.float64))
    signs = np.sign((pcs[:rank] * pca.components_).sum(axis=1))
    np.testing.assert_allclose(actual * signs, pca.transform(test), atol=1e-12)


def test_component_variance_divides_by_eigenvalues():
    centered = np.array([[-4., 0., 0.], [4., 0., 0.], [0., -2., 0.], [0., 2., 0.], [0., 0., 0.]])
    pcs, var = ef.pca(centered)
    actual = ef.component_standardize(np.array([[8., 6., 1.]]) @ pcs.T, var, 2, by_variance=True)
    np.testing.assert_allclose(np.abs(actual), [[1., 3.]])


def test_coordinate_sd_floor_counts_constant_and_near_constant_columns():
    fit = np.array([[0, -1e-10, -2, -1], [0, 1e-10, 2, 1]], dtype=np.float32)
    sd, n = ef.coordinate_scale(fit)
    np.testing.assert_allclose(sd, [2e-8, 2e-8, 2., 1.], atol=0)
    assert n == 2


def test_canon_loader_rejects_unverified_row_order(tmp_path, monkeypatch):
    resolver = ef.asw.AlignmentResolver(pd.DataFrame({"filename": ["a.txt", "b.txt"]}))
    monkeypatch.setattr(ef, "_resolver", resolver, raising=False)
    directory = ef.cache_dir(tmp_path, "m")
    directory.mkdir(parents=True)
    np.save(directory / "hidden_layer1_embeddings.npy", np.eye(2))
    with pytest.raises(ValueError, match="row-order manifest is required"):
        ef.load_canon(tmp_path, "m", 1)


def test_settings_use_only_decretal_auroc_and_break_ties_by_layer_then_rank():
    rows = []
    for layer in (2, 1):
        for group in ef.GROUPS.values():
            for correction in reversed(group):
                for corpus in ("canon", "decretals"):
                    auroc = .8
                    if corpus == "canon" and layer == 2:
                        auroc = .99
                    if corpus == "decretals" and correction == "std_coord" and layer == 2:
                        auroc = .9
                    rows.append(dict(model="m", layer=layer, correction=correction,
                                     scored_on=corpus, auroc=auroc, acc1=.5, mrr=.6))
    selected = ef.select_settings(pd.DataFrame(rows)).set_index("method")
    assert selected.loc["abtt", "layer"] == 1
    assert selected.loc["abtt", "correction"] == "abtt_D1"
    assert selected.loc["std_comp", "correction"] == "std_comp_r64"
    assert selected.loc["std_coord", "layer"] == 2
    assert selected.loc["abtt", "canon_auroc"] == .8
