"""Guards for scripts/paper/reframe/abtt_subspace_whiten.py (issue #232: E3, H1, WHITEN).

Three layers of checks:

* synthetic data (always run, CI included): the transforms mean what the paper says
  they mean. D=0 is centering (EmbeddingCleaner(0) silently is not), the removed and
  retained subspaces add back up to the centered vectors, reduced whitening is
  deterministic and white on train, and the Task A AUROC helper is the evaluator's.
* committed result CSVs (skipped when runs/ is not checked out, as in CI): the
  generated tables regenerate byte-identically, and so does the figure when
  matplotlib is installed.
* cached embeddings (skipped unless the gitignored caches are present): one E3
  model-layer recomputed from the vectors matches the committed CSV.
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
from sif_abtt import EmbeddingCleaner  # noqa: E402


def _synthetic(n_tr=60, n_te=50, d=24, n_dirs=12, seed=0):
    """Directory-clustered vectors plus one huge passage-varying nuisance direction."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_dirs, d))
    nuisance = np.zeros(d)
    nuisance[0] = 1.0

    def draw(n):
        ids = rng.integers(0, n_dirs, size=n)
        x = centers[ids] + 0.3 * rng.normal(size=(n, d))
        x += 40.0 * rng.normal(size=(n, 1)) * nuisance + 5.0  # nuisance + shared offset
        return x.astype(np.float32), ids.astype(str)

    tr, tr_ids = draw(n_tr)
    te, te_ids = draw(n_te)
    return tr, te, tr_ids, te_ids


def test_d0_is_centering_not_the_cleaner_short_circuit():
    tr, te, *_ = _synthetic()
    a, b = asw.abtt(tr, te, 0)
    mu = tr.mean(axis=0)
    np.testing.assert_allclose(a, tr - mu, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b, te - mu, rtol=0, atol=1e-5)
    # the reason abtt() special-cases D=0: the cleaner returns the input uncentered
    raw = EmbeddingCleaner(num_components=0, center=True).fit(tr)
    assert raw.mean_vec is None


def test_abtt_matches_embedding_cleaner_for_positive_d():
    tr, te, *_ = _synthetic()
    a, b = asw.abtt(tr, te, 3)
    c = EmbeddingCleaner(num_components=3, center=True).fit(tr)
    np.testing.assert_array_equal(a, c.transform(tr))
    np.testing.assert_array_equal(b, c.transform(te))


def test_removed_plus_retained_is_the_centered_vector():
    tr, te, *_ = _synthetic()
    D = 4
    s_tr, s_te, shares = asw.pc_scores(tr, te, D)
    pcs = EmbeddingCleaner(num_components=D, center=True).fit(tr).pcs
    ret_tr, ret_te = asw.abtt(tr, te, D)
    c_tr, c_te = asw.center(tr, te)
    np.testing.assert_allclose(s_tr @ pcs + ret_tr, c_tr, atol=1e-3)
    np.testing.assert_allclose(s_te @ pcs + ret_te, c_te, atol=1e-3)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[0] > 0.9  # the planted nuisance dominates the centered variance


def test_reduced_whitening_is_white_on_train_and_deterministic():
    tr, te, *_ = _synthetic(n_tr=80, d=30)
    a1, b1, info = asw.whiten(tr, te, 8)
    a2, b2, _ = asw.whiten(tr, te, 8)
    assert a1.shape == (80, 8) and b1.shape == (te.shape[0], 8)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_allclose(np.cov(a1.astype(np.float64), rowvar=False), np.eye(8), atol=1e-3)
    assert info["n_components"] == 8 and info["cond"] >= 1.0


def test_full_rank_whitening_keeps_min_n_d_components():
    tr, te, *_ = _synthetic(n_tr=20, d=30)  # fewer passages than dimensions, as for Qwen3
    _, _, info = asw.whiten(tr, te, None)
    assert info["n_components"] == 20
    assert info["eig_last"] < 1e-6 * info["eig_first"]  # rank <= n-1 after centering


def test_pair_auroc_is_the_evaluator_auroc():
    tr, te, tr_ids, te_ids = _synthetic()
    import run_resubmit_evaluate as ev
    from canon_retrieval import l2_normalize, similarity_matrix, upper_triangle_labels

    m = ev.evaluate_from_similarity(
        train_sim=similarity_matrix(l2_normalize(tr)), test_sim=similarity_matrix(l2_normalize(te)),
        train_folder_ids=tr_ids, test_folder_ids=te_ids,
        test_has_partner=np.ones(len(te_ids), dtype=bool))
    assert asw.pair_auroc(te, upper_triangle_labels(te_ids)) == m["aucroc"]
    assert asw.pair_auroc(tr, upper_triangle_labels(tr_ids)) == m["train_aucroc"]


def test_planted_nuisance_is_rank_one():
    """Sanity of the H1 logic on data whose nuisance is exactly one direction."""
    tr, te, tr_ids, te_ids = _synthetic()
    from canon_retrieval import upper_triangle_labels

    lab = upper_triangle_labels(te_ids)
    raw = asw.pair_auroc(te, lab)
    d0 = asw.pair_auroc(asw.abtt(tr, te, 0)[1], lab)
    d1 = asw.pair_auroc(asw.abtt(tr, te, 1)[1], lab)
    d10 = asw.pair_auroc(asw.abtt(tr, te, 10)[1], lab)
    f1, _ = asw.rank1_fractions(raw, d0, d1, d10)
    assert d1 > raw + 0.1
    assert f1 >= asw.RANK1_BAR


def test_random_basis_is_orthonormal_and_seeded():
    q = asw.random_basis(24, 5, 3)
    np.testing.assert_allclose(q.T @ q, np.eye(5), atol=1e-10)
    np.testing.assert_array_equal(q, asw.random_basis(24, 5, 3))
    assert not np.allclose(q, asw.random_basis(24, 5, 4))


def test_next_d_components_lie_in_the_retained_subspace():
    """PCs D+1..2D (the E3 control) are orthogonal to the removed PCs 1..D."""
    tr, te, *_ = _synthetic()
    D = 3
    pcs = EmbeddingCleaner(num_components=2 * D, center=True).fit(tr).pcs
    s_tr, _, _ = asw.pc_scores(tr, te, 2 * D)
    ret_tr, _ = asw.abtt(tr, te, D)
    np.testing.assert_allclose(ret_tr @ pcs[D:].T, s_tr[:, D:], atol=1e-3)


def test_rank1_fractions_and_select_d():
    f1, f0 = asw.rank1_fractions(0.5, 0.55, 0.9, 1.0)
    assert f1 == pytest.approx(0.8) and f0 == pytest.approx(0.1)
    assert all(np.isnan(v) for v in asw.rank1_fractions(0.9, 0.9, 0.9, 0.9))
    rows = pd.DataFrame({"D": asw.SEL_GRID, "train_dir_acc_at_1": [0.1, 0.5, 0.5, 0.4, 0.5, 0.5]})
    assert asw.select_D(rows) == 2  # first maximum wins, as find_optimal_D_phase11


# --------------------------------------------------------------------------- #
# Committed outputs
# --------------------------------------------------------------------------- #

INPUTS = [REPO_ROOT / p for p in (asw.H1_CSV, asw.E3_CSV, asw.WH_CSV, asw.RES_CSV)]
GEOM = REPO_ROOT / "runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv"
FT = [REPO_ROOT / "runs/active/resubmit/results/finetune" / f"finetune_{n}_layer_results.csv"
      for n in ("lata", "qwen3_0.6b", "kalm_mini")]
needs_csvs = pytest.mark.skipif(
    not all(p.exists() for p in INPUTS + [GEOM]),
    reason="committed reframe CSVs (runs/) not checked out")
TABLES = ["d_ablation.tex", "e3_subspace_split.tex", "whiten_reduced.tex"]


def _render(tmp_path, figure: bool):
    argv = ["render", "--h1_csv", str(INPUTS[0]), "--e3_csv", str(INPUTS[1]),
            "--whiten_csv", str(INPUTS[2]), "--results_csv", str(INPUTS[3]),
            "--geom_csv", str(GEOM), "--tab_dir", str(tmp_path), "--fig_dir", str(tmp_path),
            "--no_facts"]
    for f in FT:
        argv += ["--ft_csv", str(f)]
    if not figure:
        argv.append("--no_figure")
    asw.main(argv)


@needs_csvs
def test_tables_regenerate_byte_identically(tmp_path):
    _render(tmp_path, figure=False)
    for name in TABLES:
        committed = REPO_ROOT / "overleaf_drafts" / "tables" / name
        assert (tmp_path / name).read_bytes() == committed.read_bytes(), name


@needs_csvs
def test_figure_regenerates_byte_identically(tmp_path):
    pytest.importorskip("matplotlib")
    committed = REPO_ROOT / "overleaf_drafts" / "figures" / "fig_d_ablation.pdf"
    if not committed.exists():
        pytest.skip("figure not checked out")
    _render(tmp_path, figure=True)
    assert (tmp_path / "fig_d_ablation.pdf").read_bytes() == committed.read_bytes()


def _bases_root():
    for root in (REPO_ROOT / "runs/active/resubmit_bases",
                 Path("/u/irowerojas/localLatin/runs/active/resubmit_bases")):
        try:
            if (root / "phase9_bases/bowphs_LaTa" / asw.SUBDIR / "hidden_layer12_embeddings.npy").exists():
                return root
        except OSError:  # an unreadable path (PermissionError on another user's home) is absent
            continue
    return None


@pytest.mark.skipif(_bases_root() is None or not INPUTS[1].exists(),
                    reason="embedding caches or committed E3 CSV absent")
def test_e3_row_recomputes_from_cached_vectors(tmp_path):
    out = tmp_path / "e3.csv"
    asw.main(["e3", "--models", "LaTa", "--layers", "12", "--workers", "1",
              "--split_csv", str(REPO_ROOT / asw.SPLIT_CSV),
              "--bases_root", str(_bases_root()), "--out", str(out)])
    mine = pd.read_csv(out)
    ref = pd.read_csv(INPUTS[1])
    ref = ref[(ref.model == "bowphs/LaTa") & (ref.layer == 12)].reset_index(drop=True)
    assert list(mine.columns) == list(ref.columns)
    num = mine.select_dtypes("number").columns
    np.testing.assert_allclose(mine[num].to_numpy(float), ref[num].to_numpy(float), atol=1e-6)
