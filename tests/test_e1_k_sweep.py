"""Guards for scripts/paper/reframe/e1_k_sweep.py (issue #246, post-hoc E1 k sweep).

Synthetic arrays only, plus the committed sweep CSV when it is checked out. Nothing here
reads an embedding cache.
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
import e1_k_sweep as ks  # noqa: E402
from canon_retrieval import upper_triangle_labels  # noqa: E402

OUT = REPO_ROOT / "runs" / "active" / "reframe" / "e1"


def _synthetic(n_tr=60, n_te=50, d=24, n_dirs=12, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_dirs, d))

    def draw(n):
        ids = rng.integers(0, n_dirs, size=n)
        x = centers[ids] + 0.3 * rng.normal(size=(n, d))
        x[:, 0] += 40.0 * rng.normal(size=n)
        x[:, 1] += 200.0 + 0.1 * rng.normal(size=n)
        return x.astype(np.float32), ids.astype(str)

    tr, tr_ids = draw(n_tr)
    te, te_ids = draw(n_te)
    return tr, te, tr_ids, te_ids


def _auroc_fn(tr_ids, te_ids):
    lab_tr, lab_te = upper_triangle_labels(tr_ids), upper_triangle_labels(te_ids)
    return lambda a, b: {"aucroc": asw.pair_auroc(b, lab_te),
                         "train_aucroc": asw.pair_auroc(a, lab_tr)}


def test_ks_leave_at_least_one_coordinate():
    assert ks.ks_for(768) == [k for k in ks.K_SWEEP if k < 768]
    assert ks.ks_for(24, (1, 10, 24, 50)) == [1, 10]


def test_random_orders_are_seeded_permutations_independent_of_the_data():
    a, b = ks.random_orders(24), ks.random_orders(24)
    assert len(a) == ks.N_SEEDS
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
        assert sorted(x.tolist()) == list(range(24))
    assert not np.array_equal(a[0], a[1])


def test_sweep_matches_e1_zero_rows_at_the_shared_k():
    tr, te, tr_ids, te_ids = _synthetic()
    fn = _auroc_fn(tr_ids, te_ids)
    rows = pd.DataFrame(ks.sweep_rows("m", 3, tr, te, fn, ks=(1, 3, 5, 10, 20)))
    abl, _, _ = e1.layer_rows("m", 3, tr, te, fn)
    ref = pd.DataFrame(abl)
    ref = ref[ref["intervention"] == "zero"]
    t = ks.gate_table(rows, ref)
    assert t["pass"].all() and (t["n_cells"] == len(e1.KS)).all()


def test_sweep_rows_cover_every_ranking_seed_and_k():
    tr, te, tr_ids, te_ids = _synthetic()
    rows = pd.DataFrame(ks.sweep_rows("m", 1, tr, te, _auroc_fn(tr_ids, te_ids),
                                      ks=(1, 5, 23, 24), n_seeds=3))
    assert set(rows["k"]) == {1, 5, 23}  # k = width is dropped
    assert len(rows) == (2 + 3) * 3
    top1 = rows[(rows["ranking"] == "variance") & (rows["k"] == 1)].iloc[0]
    # coordinate 0 carries almost all the variance, so zeroing it removes most of it
    assert top1["var_removed"] > 0.9


def test_gate_fails_on_drift_and_on_a_missing_cell():
    tr, te, tr_ids, te_ids = _synthetic()
    fn = _auroc_fn(tr_ids, te_ids)
    rows = pd.DataFrame(ks.sweep_rows("m", 2, tr, te, fn, ks=(1, 3, 5, 10)))
    ref = pd.DataFrame(e1.layer_rows("m", 2, tr, te, fn)[0])
    ref = ref[ref["intervention"] == "zero"].copy()
    bad = ref.copy()
    bad.loc[bad.index[0], "aucroc"] += 1e-4
    assert not ks.gate_table(rows, bad)["pass"].all()
    assert not ks.gate_table(rows, ref.iloc[1:])["pass"].all()


@pytest.mark.skipif(not (OUT / ks.SWEEP_NAME).exists(), reason="sweep CSV not checked out")
def test_committed_sweep_passes_its_gate():
    sweep = pd.read_csv(OUT / ks.SWEEP_NAME)
    abl = e1.read_abl(OUT / e1.ABL_NAME)
    t = ks.gate_table(sweep, abl[abl["model"].isin(sweep["model"].unique())])
    assert t["pass"].all(), t


def test_gram_fallback_equals_the_svd_geometry(monkeypatch):
    tr, _, _, _ = _synthetic()
    ref = e1._geometry(tr)

    def boom(_):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(e1, "_geometry", boom)
    got = ks.geometry(tr)
    for c in ("pc1_share_train", "pc10_share_train", "eff_rank_train"):
        assert got[c] == pytest.approx(ref[c], rel=1e-8)
