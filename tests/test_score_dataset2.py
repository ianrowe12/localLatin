"""Unit tests for the dataset 2 scorer (issue #289) and the one-pass parity check."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

import score_dataset2 as sd  # noqa: E402
import check_one_pass_parity as parity  # noqa: E402


def test_match_form_is_the_webapp_comparison():
    assert sd.match_form("  CARL.501?.18 ") == sd.match_form("carl.501?.18")
    assert sd.match_form("A  B\tC") == "a b c"
    assert sd.match_form(None) == "" and sd.match_form(float("nan")) == ""


def test_key_index_maps_keys_to_directory_columns():
    dirs = ["CONC.300.1", "Can.apost.4", "(ed. X, 1; trans. Y)"]
    idx = sd.key_to_dir_index(dirs)
    assert idx[sd.match_form("conc.300.1")] == 0
    assert idx[sd.match_form("(ed. X, 1; trans.  Y)")] == 2


def test_label_dirs_agreed_conflict_and_no_key():
    dirs = ["K.1", "K.2", "K.3"]
    q = pd.DataFrame({
        "status": ["agreed", "conflict", "no_key", "agreed"],
        "ccl_key": ["k.2", None, None, "NOT.A.DIR"],
        "conflicting_keys": [None, "K.1 | K.3 | (none_of_top_k)", None, None],
    })
    cols, unmapped = sd.label_dirs(q, sd.key_to_dir_index(dirs))
    assert cols == [[1], [0, 2], [], []]
    assert unmapped == [0, 0, 0, 1]


def test_directory_scores_take_the_max_and_drop_excluded_refs():
    lab = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    folder = np.array(["a", "b", "b"])
    q = np.array([[1.0, 0.0]])
    s = sd.directory_scores(lab, q, folder, ["a", "b"], np.zeros(3, bool))
    np.testing.assert_allclose(s, [[1.0, 0.6]])
    s = sd.directory_scores(lab, q, folder, ["a", "b"], np.array([True, False, False]))
    assert s[0, 0] == -np.inf


def test_rank_order_breaks_ties_by_directory_order_like_the_served_lists():
    scores = np.array([[0.5, 0.9, 0.9, 0.1]])
    assert sd.rank_order(scores)[0].tolist() == [1, 2, 0, 3]


def test_ranks_of_uses_the_best_ranked_label_directory():
    order = np.array([[2, 0, 1], [0, 1, 2]])
    ranks = sd.ranks_of(order, [[0], [1, 2], []], np.array([0, 1, 0]))
    assert ranks[0] == 2 and ranks[1] == 2 and np.isinf(ranks[2])


def test_metric_block_and_bootstrap_bounds():
    r = np.array([1, 2, 6, 11, 1.0])
    m = sd.metric_block(r)
    assert m["acc@1"] == pytest.approx(0.4)
    assert m["acc@5"] == pytest.approx(0.6)
    assert m["acc@10"] == pytest.approx(0.8)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 1 / 6 + 1 / 11 + 1) / 5)
    idx = np.random.default_rng(0).integers(0, 5, size=(500, 5))
    ci = sd.bootstrap(r, idx)
    for k, (lo, hi) in ci.items():
        assert lo <= m[k] <= hi


def test_post_process_fits_on_train_rows_only():
    rng = np.random.default_rng(0)
    lab = rng.normal(size=(20, 5))
    unlab = rng.normal(size=(4, 5))
    train = np.zeros(20, bool)
    train[:10] = True
    fids = np.repeat(np.arange(5), 2)
    a_lab, a_unlab, D = sd.post_process("abtt_fixed", lab, unlab, train, fids)
    assert D == sd.FIXED_D
    # Changing a test row must not move the fitted transform.
    lab2 = lab.copy()
    lab2[15] += 100
    b_lab, b_unlab, _ = sd.post_process("abtt_fixed", lab2, unlab, train, fids)
    np.testing.assert_allclose(a_unlab, b_unlab)


def test_select_layers_checks_the_paper_table(tmp_path):
    rows = []
    for model, layers in sd.PAPER_ROUTING_LAYERS.items():
        for method, best in zip(sd.PAPER_TABLE_METHODS, layers):
            pooling = sd.METHODS[method][0]
            for layer in (1, best, 12, 28):
                rows.append({"model": model, "repr": "hidden", "pooling": pooling,
                             "layer": layer, "method": method, "D": 3,
                             sd.TRAIN_METRIC: 0.9 if layer == best else 0.1})
    csv = tmp_path / "r.csv"
    pd.DataFrame(rows).drop_duplicates(["model", "method", "layer"]).to_csv(csv, index=False)
    sel = sd.select_layers(csv, tmp_path / "missing.csv")
    lata = sel[(sel["model"] == "bowphs/LaTa") & (sel["method"] == "sif_abtt_optimal")]
    assert int(lata["layer"].iloc[0]) == 12 and int(lata["D"].iloc[0]) == 3

    bad = pd.read_csv(csv)
    bad.loc[(bad["model"] == "bowphs/LaTa") & (bad["method"] == "baseline"), sd.TRAIN_METRIC] = (
        bad["layer"] / 100.0
    )
    bad.to_csv(csv, index=False)
    with pytest.raises(SystemExit, match="selected_layers.tex"):
        sd.select_layers(csv, tmp_path / "missing.csv")


def _write_cache(d: Path, names, emb, suffix=""):
    d.mkdir(parents=True)
    pd.DataFrame({"file_id": range(len(names)), "filename": names,
                  "path": [f"x/{n}" for n in names]}).to_csv(d / "meta.csv", index=False)
    np.save(d / f"hidden_layer1_embeddings{suffix}.npy", emb)


def test_parity_compare_matches_by_filename(tmp_path):
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(4, 3)).astype(np.float32)
    _write_cache(tmp_path / "cache", ["a", "b", "c", "d"], emb, "_lasttok")
    _write_cache(tmp_path / "fresh", ["b", "a"], emb[[1, 0]], "_lasttok")
    assert parity.compare(tmp_path / "cache", tmp_path / "fresh", "_lasttok", 0.9999) == 0
    np.save(tmp_path / "fresh" / "hidden_layer1_embeddings_lasttok.npy", emb[[0, 1]])
    assert parity.compare(tmp_path / "cache", tmp_path / "fresh", "_lasttok", 0.9999) == 1
