"""Unit tests for the dataset 2 scorer (issue #289) and the one-pass parity check."""

from __future__ import annotations

import argparse
import json
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


# --- whitening (issue #289 round 1: rank-deficient fits) -----------------------

MODEL = "toy/model"


def _toy_ctx(tmp_path: Path, n_train: int, dim: int, n_query: int = 6) -> dict:
    """Labelled and unlabelled caches for one model at layer 1, plus a score ctx."""
    rng = np.random.default_rng(7)
    n_lab = 2 * n_train
    lab_names = [f"l{i}.txt" for i in range(n_lab)]
    q_names = [f"q{i}.txt" for i in range(n_query)]
    split = pd.DataFrame({
        "filename": lab_names,
        "folder_id": [f"D{i % (n_train // 2)}" for i in range(n_lab)],
        "split": ["train"] * n_train + ["test"] * n_train,
    })
    qmeta = pd.DataFrame({"filename": q_names})
    for bases, names, n in (("lab", lab_names, n_lab), ("unlab", q_names, n_query)):
        _write_cache(tmp_path / bases / sd.slug(MODEL) / "hidden_mean_tokempty", names,
                     rng.normal(size=(n, dim)).astype(np.float32))
    train = split["split"].values == "train"
    return {
        "labelled_bases": tmp_path / "lab", "unlabelled_bases": tmp_path / "unlab",
        "lab_resolver": sd.AlignmentResolver(split),
        "unlab_resolver": sd.AlignmentResolver(qmeta),
        "dir_names": list(dict.fromkeys(split["folder_id"])),
        "folder_ids": split["folder_id"].values, "train_mask": train,
        "train_fids": split.loc[train, "folder_id"].values,
        "lab_blank": np.zeros(n_lab, bool), "unlab_blank": np.zeros(n_query, bool),
    }


def test_rank_deficient_whitening_is_marked_not_scored(tmp_path):
    ctx = _toy_ctx(tmp_path, n_train=12, dim=16)  # n_train <= d, like the 1024-d models
    res = sd.score_one(MODEL, "whitening", 1, None, ctx)
    assert res.get("status") == "not_applicable_rank_deficient"
    assert "scores" not in res
    assert "n_train=12 <= d=16" in res["reason"]
    with pytest.raises(sd.RankDeficientWhitening):
        sd.paper_whitening(np.zeros((16, 16), np.float32))  # n_train == d is deficient too


def test_full_rank_whitening_is_scored(tmp_path):
    ctx = _toy_ctx(tmp_path, n_train=40, dim=8)
    res = sd.score_one(MODEL, "whitening", 1, None, ctx)
    assert res["status"] == sd.SCORED
    assert np.isfinite(res["scores"]).all()
    # Whitened at the cache's float32, as the paper does, not on a float64 copy.
    lab = ctx["lab_resolver"].load(sd.emb_path(ctx["labelled_bases"], MODEL, "mean", 1))
    unlab = ctx["unlab_resolver"].load(sd.emb_path(ctx["unlabelled_bases"], MODEL, "mean", 1))
    lab_w, unlab_w = sd.paper_whitening(lab[ctx["train_mask"]], lab, unlab)
    want = sd.directory_scores(sd.l2_normalize(lab_w), sd.l2_normalize(unlab_w),
                               ctx["folder_ids"], ctx["dir_names"], np.zeros(len(lab), bool))
    np.testing.assert_array_equal(res["scores"], want)


def test_whitening_is_the_paper_evaluators_whitening():
    """Same transform and precision as run_resubmit_evaluate.evaluate_single."""
    import run_resubmit_evaluate as paper

    rng = np.random.default_rng(3)
    n, dim = 60, 12
    emb = (rng.normal(size=(n, dim)) * np.geomspace(1, 1e-3, dim)).astype(np.float32)
    split = pd.DataFrame({"split": ["train", "test"] * (n // 2),
                          "folder_id": [f"D{i // 4}" for i in range(n)]})
    test = split["split"] == "test"
    split["has_test_partner"] = False
    split.loc[test, "has_test_partner"] = sd.has_partner_flags(
        split.loc[test, "folder_id"].values)
    want = paper.evaluate_single(emb, split, "whitening", 0, 0.0, "m", "hidden", "mean", 1, [1])

    train_mask = (split["split"] == "train").values
    tr, te = sd.paper_whitening(emb[train_mask], emb[train_mask], emb[~train_mask])
    assert tr.dtype == np.float32
    got = paper.evaluate_from_similarity(
        train_sim=sd.similarity_matrix(sd.l2_normalize(tr)),
        test_sim=sd.similarity_matrix(sd.l2_normalize(te)),
        train_folder_ids=split.loc[train_mask, "folder_id"].values,
        test_folder_ids=split.loc[~train_mask, "folder_id"].values,
        test_has_partner=split.loc[~train_mask, "has_test_partner"].values.astype(bool),
    )
    assert got == {k: want[k] for k in got}


def test_not_applicable_rows_carry_the_marker_and_no_number():
    q = pd.DataFrame({"status": ["agreed", "conflict", "agreed"],
                      "seen_group": ["a", "a", "b"]})
    sets = {"agreed": q["status"] == "agreed"}
    rows = pd.DataFrame(sd.not_applicable_rows(q, sets, "M", "whitening", 5, "why"))
    assert set(rows["seen_group"]) == {"all", "a", "b"}
    assert (rows["status"] == sd.NOT_APPLICABLE).all() and rows["value"].isna().all()
    head = sd.headline_table(rows)
    assert len(head) == 1 and head["status"].iloc[0] == sd.NOT_APPLICABLE
    assert head[["acc@1", "mrr"]].isna().all(axis=None)


@pytest.fixture
def snapshot_scoring(tmp_path, monkeypatch):
    """Run the real scoring/output path with tiny deterministic model rankings."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    names = [f"q{i}.txt" for i in range(10)]
    keys = pd.DataFrame({
        "xml_id": [f"q{i}" for i in range(10)],
        "status": ["agreed"] * 4 + ["conflict"] * 2 + ["no_key"] * 2 + ["needs_check", "agreed"],
        "ccl_key": ["K.1", "K.2", "K.3", "K.1", None, None, None, None, None, "UNKNOWN"],
        "conflicting_keys": [None] * 4 + [
            "K.1 | K.3", "K.2 | (none_of_top_k)", None, None, "K.1 | (unresolved)", None,
        ],
    })
    pd.DataFrame({
        "xml_id": keys["xml_id"],
        "outcome": ["matched_rank"] * 6 + ["none_of_top_k"] * 2 + ["matched_rank"] * 2,
        "model_slug": ["m1", "m1", "m2", "m2", "m1", "m2", "m1", "m2", "m1", "m1"],
        "correct_rank": [1] * 10,
    }).to_csv(snapshot / "reviews.csv", index=False)
    split = pd.DataFrame({
        "filename": [f"ref{i}.txt" for i in range(6)],
        "path": [f"x/ref{i}.txt" for i in range(6)],
        "folder_id": ["K.1", "K.2", "K.3"] * 2,
        "split": ["train"] * 3 + ["test"] * 3,
    })
    split.to_csv(tmp_path / "split.csv", index=False)
    pd.DataFrame({"filename": names, "path": [f"x/{name}" for name in names]}).to_csv(
        tmp_path / "qmeta.csv", index=False,
    )
    args = argparse.Namespace(**{name: tmp_path / name for name in sd.DEFAULTS})
    args.split_csv = tmp_path / "split.csv"
    args.unlabelled_meta = tmp_path / "qmeta.csv"
    args.snapshot = snapshot
    args.n_boot, args.seed = 37, 289
    args.min_served_agreement, args.force = 0.98, False
    scores = np.array([
        [3, 2, 1], [3, 2, 1], [3, 2, 1], [1, 3, 2], [1, 2, 3],
        [3, 2, 1], [2, 3, 1], [3, 1, 2], [1, 3, 2], [3, 2, 1],
    ], dtype=float)
    model_scores = {"baseline": scores, "sif_only": scores[:, ::-1]}
    monkeypatch.setattr(sd, "sanity_served", lambda *a: {"top10_exact_agree_frac": 1.0})
    monkeypatch.setattr(sd, "read_blank_source_mask", lambda paths, root: np.zeros(len(paths), bool))
    monkeypatch.setattr(sd, "load_served", lambda *a: [["K.1", "K.2", "K.3"]] * len(names))
    monkeypatch.setattr(sd, "select_layers", lambda *a: pd.DataFrame([
        {"model": sd.MT5, "method": method, "layer": 1, "D": None}
        for method in model_scores
    ]))
    monkeypatch.setattr(sd, "score_one", lambda model, method, *a: {
        "status": sd.SCORED, "reason": "", "D": None, "scores": model_scores[method],
        "excluded_queries": np.zeros(len(names), bool), "excluded_refs": np.zeros(len(split), bool),
    })
    return args, keys, model_scores


@pytest.mark.parametrize("legacy", [True, False])
def test_snapshot_without_decisions_preserves_existing_scoring_outputs(snapshot_scoring, legacy):
    args, keys, scores = snapshot_scoring
    if not legacy:
        keys["decided"] = 0
        keys["reviewer_status"] = keys["status"]
    keys.to_csv(args.snapshot / "ccl_key_assignments.csv", index=False)
    sd.cmd_score(args)
    results = args.out_dir / "results"
    counts = json.loads((results / "counts.json").read_text())
    assert counts == {
        "queries_with_snapshot_row": 10,
        "status": {"agreed": 5, "conflict": 2, "no_key": 2, "needs_check": 1},
        "seen_group": {"m1": 6, "m2": 4},
        "keyed_queries_with_unmapped_key": 1, "keyed_queries_with_no_mapped_dir": 2,
        "agreed_by_final_answer": 0,
    }
    per_query = pd.read_csv(results / "per_query_ranks.csv")
    assert per_query.columns.tolist() == [
        "xml_id", "status", "seen_group", "unmapped_keys",
        "rank:mT5-base|baseline", "rank:mT5-base|sif_only",
    ]
    assert per_query["status"].tolist() == keys["status"].tolist()
    assert per_query["unmapped_keys"].tolist() == [0] * 9 + [1]
    acc = pd.read_csv(results / "accuracy_long.csv")
    old_sets = {
        "agreed": keys["status"] == "agreed",
        "agreed_plus_conflict": keys["status"].isin(["agreed", "conflict"]),
        "conflict": keys["status"] == "conflict",
    }
    # Reproduce the pre-decisions bootstrap stream across both model cells.
    # Consuming draws for the new set must not change any original CI.
    rng = np.random.default_rng(args.seed)
    boot_idx = {}
    seen = per_query["seen_group"]
    labels = [[0], [1], [2], [0], [0, 2], [1], [], [], [], []]
    for method, matrix in scores.items():
        ranks = sd.ranks_of(sd.rank_order(matrix), labels, np.arange(len(keys)))
        np.testing.assert_array_equal(per_query[f"rank:mT5-base|{method}"], ranks)
        with np.load(args.out_dir / "rankings" / f"{sd.slug(sd.MT5)}__{method}.npz") as ranking:
            np.testing.assert_array_equal(ranking["scores"], matrix)
        for set_name, mask in old_sets.items():
            groups = {"all": mask}
            groups.update({g: mask & (seen == g) for g in sorted(seen[mask].unique())})
            for group, group_mask in groups.items():
                r = ranks[group_mask]
                slot = (set_name, group)
                if slot not in boot_idx:
                    boot_idx[slot] = rng.integers(0, len(r), size=(args.n_boot, len(r)))
                expected_ci = sd.bootstrap(r, boot_idx[slot])
                got = acc[(acc["method"] == method) & (acc["label_set"] == set_name)
                          & (acc["seen_group"] == group)].set_index("metric")
                assert (got["n"] == len(r)).all()
                for metric, value in sd.metric_block(r).items():
                    assert got.loc[metric, "value"] == pytest.approx(value)
                    assert got.loc[metric, ["ci_lo", "ci_hi"]].tolist() == pytest.approx(
                        expected_ci[metric],
                    )
    new_set = acc[acc["label_set"] == "reviewers_agreed"]
    agreed = acc[acc["label_set"] == "agreed"]
    assert new_set["n"].tolist() == agreed["n"].tolist()
    np.testing.assert_array_equal(new_set["value"], agreed["value"])
    nokey = pd.read_csv(results / "nokey_top1_vs_served.csv")
    assert nokey["n"].tolist() == [2, 2]
    assert nokey["top1_in_mt5_served_top10"].tolist() == [2, 2]
    summary = (results / "summary.md").read_text()
    assert "Agreed queries settled by a final answer: 0." in summary
    assert "## reviewers_agreed (n=5)" in summary


def test_final_answers_drive_labels_counts_and_reviewer_agreement_set(snapshot_scoring):
    args, keys, _ = snapshot_scoring
    keys["reviewer_status"] = keys["status"]
    keys["decided"] = 0
    # A conflict becomes agreed, an agreed key changes, another becomes explicit
    # none, and an unresolved item with no counted review gets a final key.
    keys.loc[[1, 2, 4, 8], "decided"] = 1
    keys.loc[[1, 4, 8], ["status", "ccl_key"]] = ["agreed", "K.3"]
    keys.loc[2, ["status", "ccl_key"]] = ["no_key", None]
    keys.loc[[1, 2, 4, 8], "conflicting_keys"] = None
    keys.loc[8, "reviewer_status"] = None
    keys.to_csv(args.snapshot / "ccl_key_assignments.csv", index=False)
    reviews = pd.read_csv(args.snapshot / "reviews.csv")
    reviews[reviews["xml_id"] != "q8"].to_csv(args.snapshot / "reviews.csv", index=False)
    sd.cmd_score(args)
    results = args.out_dir / "results"
    counts = json.loads((results / "counts.json").read_text())
    assert counts["agreed_by_final_answer"] == 3
    assert counts["status"] == {"agreed": 6, "no_key": 3, "conflict": 1}
    ranks = pd.read_csv(results / "per_query_ranks.csv").set_index("xml_id")
    assert ranks.loc["q1", "rank:mT5-base|baseline"] == 3
    assert ranks.loc["q4", "rank:mT5-base|baseline"] == 1
    assert ranks.loc["q8", "rank:mT5-base|baseline"] == 2
    assert ranks.loc["q8", "seen_group"] == "none"
    assert np.isinf(ranks.loc["q2", "rank:mT5-base|baseline"])
    acc = pd.read_csv(results / "accuracy_long.csv")
    all_baseline = acc[(acc["method"] == "baseline") & (acc["seen_group"] == "all")]
    assert all_baseline.groupby("label_set")["n"].first().to_dict() == {
        "agreed": 6, "agreed_plus_conflict": 7, "conflict": 1, "reviewers_agreed": 5,
    }
    headline = pd.read_csv(results / "headline.csv")
    assert "reviewers_agreed" in set(headline["label_set"])
    assert pd.read_csv(results / "nokey_top1_vs_served.csv")["n"].tolist() == [3, 3]
    summary = (results / "summary.md").read_text()
    assert "Agreed queries settled by a final answer: 3." in summary
    assert "## reviewers_agreed (n=5)" in summary
