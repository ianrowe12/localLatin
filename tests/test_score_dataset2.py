"""Unit tests for the dataset 2 scorer (issue #289) and the one-pass parity check."""

from __future__ import annotations

import argparse
import json
import subprocess
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
import dataset2_parity as dp  # noqa: E402


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
        "excluded_queries": np.array([method == "sif_only"] + [False] * (len(names) - 1)),
        "excluded_refs": np.zeros(len(split), bool),
    })
    return args, keys, model_scores


@pytest.mark.parametrize("legacy", [True, False])
def test_snapshot_without_decisions_preserves_existing_scoring_outputs(snapshot_scoring, legacy):
    args, keys, scores = snapshot_scoring
    if not legacy:
        keys["decided"] = 0
        keys["reviewer_status"] = keys["status"]
        pd.DataFrame(columns=["xml_id", "ccl_key"]).to_csv(
            args.snapshot / "deciding_answers.csv", index=False,
        )
    keys.to_csv(args.snapshot / "ccl_key_assignments.csv", index=False)
    sd.cmd_score(args)
    results = args.out_dir / "results"
    counts = json.loads((results / "counts.json").read_text())
    assert counts == {
        "queries_with_snapshot_row": 10,
        "status": {"agreed": 5, "conflict": 2, "no_key": 2, "needs_check": 1},
        "seen_group": {"m1": 6, "m2": 4},
        "keyed_queries_with_unmapped_key": 1, "keyed_queries_with_no_mapped_dir": 2,
        "agreed_settled_by_final_answer": 0, "agreed_confirmed_by_final_answer": 0,
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
        if method == "sif_only":
            ranks[0] = np.nan
        np.testing.assert_array_equal(per_query[f"rank:mT5-base|{method}"], ranks)
        with np.load(args.out_dir / "rankings" / f"{sd.slug(sd.MT5)}__{method}.npz") as ranking:
            np.testing.assert_array_equal(ranking["scores"], matrix)
        for set_name, mask in old_sets.items():
            groups = {"all": mask}
            groups.update({g: mask & (seen == g) for g in sorted(seen[mask].unique())})
            for group, group_mask in groups.items():
                r = ranks[group_mask]
                r = r[~np.isnan(r)]
                if not len(r):
                    continue
                slot = (set_name, group)
                if slot not in boot_idx or boot_idx[slot].shape[1] != len(r):
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
    assert "Agreed queries confirmed by a final answer: 0." in summary
    assert "## reviewers_agreed (n=5)" in summary


def test_reviewers_set_does_not_change_existing_output_bytes(snapshot_scoring, monkeypatch):
    args, keys, _ = snapshot_scoring
    monkeypatch.setattr(sd, "select_layers", lambda *a: pd.DataFrame([
        {"model": sd.MT5, "method": "baseline", "layer": 1, "D": None},
        {"model": "bowphs/LaTa", "method": "sif_only", "layer": 1, "D": None},
    ]))
    keys.to_csv(args.snapshot / "ccl_key_assignments.csv", index=False)
    args.out_dir = args.out_dir.parent / "with_reviewers"
    sd.cmd_score(args)
    with_reviewers = args.out_dir / "results"
    accuracy = pd.read_csv(with_reviewers / "accuracy_long.csv")
    agreed = accuracy[(accuracy["label_set"] == "agreed")
                      & (accuracy["seen_group"] == "all")].drop_duplicates("model")
    assert agreed.set_index("model")[["n", "n_excluded"]].to_dict("index") == {
        "mT5-base": {"n": 5, "n_excluded": 0}, "LaTa": {"n": 4, "n_excluded": 1},
    }

    load_snapshot = sd.load_snapshot

    def without_reviewers_set(*a):
        q = load_snapshot(*a)
        q["reviewer_status"] = ""
        return q

    monkeypatch.setattr(sd, "load_snapshot", without_reviewers_set)
    args.out_dir = args.out_dir.parent / "without_reviewers"
    sd.cmd_score(args)
    without_reviewers = args.out_dir / "results"

    for name in ("accuracy_long.csv", "headline.csv"):
        existing_rows = b"".join(
            line for line in (with_reviewers / name).read_bytes().splitlines(keepends=True)
            if not line.startswith(b"reviewers_agreed,")
        )
        assert existing_rows == (without_reviewers / name).read_bytes(), name
    for name in ("per_query_ranks.csv", "nokey_top1_vs_served.csv", "counts.json"):
        assert (with_reviewers / name).read_bytes() == (without_reviewers / name).read_bytes()


def test_final_answers_drive_labels_counts_and_reviewer_agreement_set(snapshot_scoring):
    args, keys, _ = snapshot_scoring
    keys.to_csv(args.snapshot / "ccl_key_assignments_reviewers_only.csv", index=False)
    keys["reviewer_status"] = keys["status"]
    keys["decided"] = 0
    # Confirm one key, override another, settle a conflict, record final none,
    # and decide an item with no counted review.
    keys.loc[[0, 1, 2, 4, 8], "decided"] = 1
    keys.loc[0, "ccl_key"] = " k.1 "
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
    assert counts["agreed_settled_by_final_answer"] == 2
    assert counts["agreed_confirmed_by_final_answer"] == 1
    assert "agreed_by_final_answer" not in counts
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
    reviewers = all_baseline[all_baseline["label_set"] == "reviewers_agreed"].set_index("metric")
    assert reviewers.loc["acc@5", "value"] == pytest.approx(0.8)
    assert reviewers.loc["mrr", "value"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 3) / 5)
    headline = pd.read_csv(results / "headline.csv")
    assert "reviewers_agreed" in set(headline["label_set"])
    assert pd.read_csv(results / "nokey_top1_vs_served.csv")["n"].tolist() == [3, 3]
    summary = (results / "summary.md").read_text()
    assert "Agreed queries settled by a final answer: 2." in summary
    assert "Agreed queries confirmed by a final answer: 1." in summary
    assert "## reviewers_agreed (n=5)" in summary


@pytest.mark.parametrize("final_key", [None, "K.3"])
def test_reviewers_agreed_uses_its_own_status_and_key(snapshot_scoring, final_key):
    args, keys, _ = snapshot_scoring
    keys.to_csv(args.snapshot / "ccl_key_assignments_reviewers_only.csv", index=False)
    keys["decided"] = 0
    keys["reviewer_status"] = "conflict"  # Membership must come from the separate file.
    keys.loc[1, ["decided", "status", "ccl_key"]] = [
        1, "no_key" if final_key is None else "agreed", final_key,
    ]
    keys.to_csv(args.snapshot / "ccl_key_assignments.csv", index=False)
    sd.cmd_score(args)
    acc = pd.read_csv(args.out_dir / "results" / "accuracy_long.csv")
    reviewers = acc[(acc["label_set"] == "reviewers_agreed")
                    & (acc["method"] == "baseline") & (acc["seen_group"] == "all")]
    assert reviewers["n"].tolist() == [5] * 4
    assert reviewers.set_index("metric").loc["mrr", "value"] == pytest.approx(
        (1 + 1 / 2 + 1 / 3 + 1 / 3) / 5,
    )


@pytest.mark.parametrize("decision_source", ["assignments", "log"])
def test_snapshot_with_decisions_requires_reviewers_only_file(snapshot_scoring, decision_source):
    args, keys, _ = snapshot_scoring
    if decision_source == "assignments":
        keys["decided"] = 1
    else:
        pd.DataFrame({"xml_id": ["q0"], "ccl_key": [None]}).to_csv(
            args.snapshot / "deciding_answers.csv", index=False,
        )
    keys.to_csv(args.snapshot / "ccl_key_assignments.csv", index=False)
    with pytest.raises(SystemExit, match="ccl_key_assignments_reviewers_only.csv"):
        sd.load_snapshot(args.snapshot, [f"q{i}.txt" for i in range(len(keys))])


@pytest.fixture
def parity_scoring(snapshot_scoring, monkeypatch):
    args, keys, scores = snapshot_scoring
    args.data_root = args.out_dir.parent / "data"
    (args.data_root / "x").mkdir(parents=True)
    for frame in (pd.read_csv(args.split_csv), pd.read_csv(args.unlabelled_meta)):
        for i, path in enumerate(frame["path"]):
            (args.data_root / path).write_text(["alpha sancta", "beta canon", "gamma lex"][i % 3])
    keys.to_csv(args.snapshot / "ccl_key_assignments_reviewers_only.csv", index=False)
    keys["decided"] = 0
    keys.loc[1, ["decided", "status", "ccl_key"]] = [1, "agreed", "K.3"]
    keys.to_csv(args.snapshot / "ccl_key_assignments.csv", index=False)
    selection = pd.DataFrame([
        {"model": sd.MT5, "display": sd.DISPLAY[sd.MT5], "method": method,
         "pooling": sd.METHODS[method][0], "layer": 1, "D": None,
         sd.TRAIN_METRIC: 0.8 if method == "baseline" else 0.9,
         "status": sd.SCORED, "reason": ""}
        for method in scores
    ])
    monkeypatch.setattr(sd, "select_layers", lambda *a: selection.copy())
    sd.cmd_score(args)
    return args


def test_reference_reuses_paper_vectors_train_fit_and_max_over_files(parity_scoring, monkeypatch):
    from lexical_baselines import char_tfidf_score_matrix

    args = parity_scoring
    ctx = dp.context(args)
    real_vectors = dp.char_tfidf_vectors
    observed = []

    def vectors(texts, train):
        observed.append(train.copy())
        return real_vectors(texts, train)

    monkeypatch.setattr(dp, "char_tfidf_vectors", vectors)
    result = dp.reference_scores(ctx, args.data_root)
    texts = [(args.data_root / path).read_text()
             for frame in (ctx["split"], ctx["qmeta"]) for path in frame["path"]]
    n_lab = len(ctx["split"])
    matrix = char_tfidf_score_matrix(texts, np.flatnonzero(ctx["train_mask"]))
    expected = np.column_stack([
        matrix[n_lab:, np.flatnonzero(ctx["folder_ids"] == name)].max(axis=1)
        for name in ctx["dir_names"]
    ])
    np.testing.assert_array_equal(observed, [np.arange(3)])
    np.testing.assert_array_equal(result["scores"], expected)
    assert not result["excluded_queries"].any()
    # OOV queries have no arbitrary all-zero ranking win.
    (args.data_root / "x/q0.txt").write_text("zzzzzz")
    result = dp.reference_scores(ctx, args.data_root)
    assert result["excluded_queries"][0] and np.isnan(result["scores"][0]).all()
    np.testing.assert_array_equal(result["scores"][1:], expected[1:])


def test_compare_preserves_every_legacy_output_byte_and_legacy_ci_draws(parity_scoring):
    args = parity_scoring
    before = {path.relative_to(args.out_dir): path.read_bytes()
              for path in args.out_dir.rglob("*") if path.is_file()}
    subprocess.run([
        sys.executable, str(REPO_ROOT / "scripts/resubmit/score_dataset2.py"), "compare",
        "--snapshot", str(args.snapshot), "--split_csv", str(args.split_csv),
        "--unlabelled_meta", str(args.unlabelled_meta), "--out_dir", str(args.out_dir),
        "--data_root", str(args.data_root), "--n_boot", str(args.n_boot), "--seed", str(args.seed),
    ], check=True, capture_output=True, text=True)
    for path, content in before.items():
        assert (args.out_dir / path).read_bytes() == content, path
    # Running the additions cannot advance score's RNG streams, either.
    sd.cmd_score(args)
    for path, content in before.items():
        assert (args.out_dir / path).read_bytes() == content, path

    ctx = dp.context(args)
    q = dp.labels(args, ctx)
    ref = dp.reference_scores(ctx, args.data_root)
    ranks = dp.rank_vectors(sd.rank_order(ref["scores"]), ref["excluded_queries"], q)
    selection = pd.read_csv(args.out_dir / "results/selected_layers.csv")
    cells = dp.load_rankings(args.out_dir, selection, ctx, q)
    baseline = cells[(sd.MT5, "baseline")]
    # Replay independently: the first all/agreed legacy draw precedes every other draw.
    mask = (q["status"] == "agreed").to_numpy()
    idx = np.random.default_rng(args.seed).integers(
        0, int(mask.sum()), size=(args.n_boot, int(mask.sum())),
    )
    want = sd.bootstrap(ranks["final"][mask], idx)
    result = pd.read_csv(args.out_dir / "results/reference_accuracy_long.csv")
    got = result[(result["label_set"] == "agreed") & (result["seen_group"] == "all")]
    assert set(got["bootstrap_stream"]) == {"legacy_score"}
    for metric, row in got.set_index("metric").iterrows():
        np.testing.assert_allclose([row["ci_lo"], row["ci_hi"]], want[metric], atol=1e-14)
    reviewers = result[(result["label_set"] == "reviewers_agreed")
                       & (result["seen_group"] == "all")].set_index("metric")
    reviewer_mask = (q["reviewer_status"] == "agreed").to_numpy()
    assert reviewers.loc["acc@1", "value"] == pytest.approx(
        sd.metric_block(ranks["reviewers"][reviewer_mask])["acc@1"],
    )
    assert not np.array_equal(baseline["final"], baseline["reviewers"])
    info = json.loads((args.out_dir / "results/comparison_info.json").read_text())
    assert info["status_counts"]["conflict"] == 2
    assert set(result["label_set"]) == {
        "agreed", "agreed_plus_conflict", "conflict", "reviewers_agreed",
    }


def test_reference_different_exclusions_get_an_explicit_independent_stream(parity_scoring):
    args = parity_scoring
    (args.data_root / "x/q2.txt").write_text("zzzzzz")
    dp.cmd_compare(args)
    ref = pd.read_csv(args.out_dir / "results/reference_accuracy_long.csv")
    agreed = ref[(ref["label_set"] == "agreed") & (ref["seen_group"] == "all")]
    assert agreed["n"].tolist() == [4] * 4
    assert agreed["n_excluded"].tolist() == [1] * 4
    assert set(agreed["bootstrap_stream"]) == {"reference_named"}


def test_compare_refuses_different_legacy_bootstrap_configuration(parity_scoring):
    args = parity_scoring
    args.seed += 1
    with pytest.raises(ValueError, match="snapshot, seed and n_boot"):
        dp.cmd_compare(args)


def test_paired_ci_uses_common_queries_same_resamples_and_keeps_misses(monkeypatch):
    q = pd.DataFrame({"status": ["agreed"] * 4, "reviewer_status": ["agreed"] * 4,
                      "seen_group": ["model"] * 4})
    a = {"final": np.array([1., 2., np.nan, np.inf])}
    b = {"final": np.array([2., 1., 1., np.inf])}
    a["reviewers"], b["reviewers"] = a["final"], b["final"]
    idx = np.array([[0, 0, 1], [1, 1, 2], [2, 2, 0]])
    monkeypatch.setattr(dp, "draws", lambda args, name, group, n, domain: idx)
    rows = pd.DataFrame(dp.paired_rows(None, q, a, b, {"contrast": "test"}))
    got = rows[(rows["label_set"] == "agreed") & (rows["seen_group"] == "all")]
    assert got["n"].tolist() == [3] * 4  # inf is a miss; NaN alone is excluded.
    assert got["n_excluded"].tolist() == [1] * 4
    for metric, delta in (("acc@1", np.array([1., -1., 0.])),
                          ("mrr", np.array([.5, -.5, 0.]))):
        row = got[got["metric"] == metric].iloc[0]
        assert row["estimate"] == 0
        np.testing.assert_allclose([row["ci_lo"], row["ci_hi"]],
                                   np.percentile(delta[idx].mean(axis=1), [2.5, 97.5]))
    same = pd.DataFrame(dp.paired_rows(None, q, a, a, {}))
    assert (same[["estimate", "ci_lo", "ci_hi"]] == 0).all(axis=None)
    undefined = pd.DataFrame(dp.paired_rows(
        None, q, None, b, {}, sd.NOT_APPLICABLE, "rank deficient",
    ))
    assert set(undefined["status"]) == {sd.NOT_APPLICABLE}
    assert undefined[["estimate", "ci_lo", "ci_hi"]].isna().all(axis=None)


def test_train_best_does_not_use_dataset2_values_or_model_order():
    selection = pd.DataFrame({
        "model": ["m"] * 4, "method": ["baseline", "whitening", "sif_only", "abtt_optimal"],
        "status": [sd.SCORED, sd.NOT_APPLICABLE, sd.SCORED, sd.SCORED],
        sd.TRAIN_METRIC: [.99, 1., .8, .8],
        "dataset2_acc": [.1, 1., .2, 1.],
    })
    assert dp.best_methods(selection)["method"].tolist() == ["sif_only"]
    assert dp.best_methods(selection.iloc[::-1])["method"].tolist() == ["sif_only"]
    args = argparse.Namespace(seed=289, n_boot=37)
    original = dp.draws(args, "agreed", "all", 5, 2)
    dp.draws(args, "conflict", "all", 7, 2)
    np.testing.assert_array_equal(original, dp.draws(args, "agreed", "all", 5, 2))
    assert not np.array_equal(original, dp.draws(args, "agreed", "all", 5, 1))


def test_layer_sweep_covers_cache_layers_uses_each_train_d_and_marks_whitening(tmp_path, monkeypatch):
    ctx = _toy_ctx(tmp_path, n_train=12, dim=16)
    ctx["query_names"] = [f"q{i}.txt" for i in range(6)]
    for bases in ("labelled_bases", "unlabelled_bases"):
        path = sd.emb_path(ctx[bases], MODEL, "mean", 1)
        emb = np.load(path)
        for layer in (2, 3):
            np.save(path.with_name(f"hidden_layer{layer}_embeddings.npy"), emb + layer / 10)
        names = pd.read_csv(path.parent / "meta.csv")["filename"]
        for pooling in ("sif", "lasttok"):
            target = sd.emb_path(ctx[bases], MODEL, pooling, 1)
            _write_cache(target.parent, names, emb, sd.SUFFIX[pooling])
    # Every unlabelled layer must be reported, including one with no labelled counterpart.
    np.save(sd.emb_path(ctx["unlabelled_bases"], MODEL, "mean", 9),
            np.ones((6, 16), dtype=np.float32))
    q = pd.DataFrame({
        "query_row": np.arange(6), "status": ["agreed"] * 6,
        "label_cols": [[i % 6] for i in range(6)],
        "reviewer_label_cols": [[i % 6] for i in range(6)],
    })
    training = pd.DataFrame([
        {"model": MODEL, "repr": "hidden", "pooling": "mean", "layer": layer,
         "method": "abtt_optimal", "D": layer, sd.TRAIN_METRIC: .8}
        for layer in (1, 2)
    ] + [
        {"model": MODEL, "repr": "hidden", "pooling": pooling, "layer": 1,
         "method": method, "D": d, sd.TRAIN_METRIC: .7}
        for pooling, method, d in (("sif", "sif_abtt_optimal", 5),
                                    ("lasttok", "abtt_optimal", 7))
    ])
    args = argparse.Namespace(out_dir=tmp_path / "out")
    monkeypatch.setattr(dp, "context", lambda args: ctx)
    monkeypatch.setattr(dp, "labels", lambda args, ctx: q)
    monkeypatch.setattr(dp, "train_rows", lambda args: training)
    seen = []

    def train_d(train, fids, grid):
        seen.append(train.copy())
        return 3, .7

    monkeypatch.setattr(sd, "find_optimal_D_phase11", train_d)
    dp.cmd_layers(args)
    result = pd.read_csv(args.out_dir / "results/per_layer.csv")
    assert set(result["layer"]) == {1, 2, 3, 9}
    assert len(result) == 16  # Twelve mean cells, three SIF cells, one last-token cell.
    assert set(result["method"]) == set(sd.METHODS)
    assert result.loc[result["method"] == "sif_abtt_optimal", "D"].tolist() == [5]
    assert result.loc[result["method"] == "lasttok_abtt_optimal", "D"].tolist() == [7]
    abtt = result[result["method"] == "abtt_optimal"].set_index("layer")
    assert abtt.loc[[1, 2, 3], "D"].tolist() == [1, 2, 3]
    assert abtt.loc[3, "D_source"] == "retuned_benchmark_v1_train"
    expected = ctx["lab_resolver"].load(
        sd.emb_path(ctx["labelled_bases"], MODEL, "mean", 3),
    )[ctx["train_mask"]].astype(np.float64)
    np.testing.assert_array_equal(seen, [expected])
    white = result[(result["method"] == "whitening") & (result["layer"] != 9)]
    assert set(white["status"]) == {sd.NOT_APPLICABLE}
    assert white[["acc@1", "mrr"]].isna().all(axis=None)
    assert set(result[result["layer"] == 9]["status"]) == {dp.MISSING}
    baseline = result[(result["method"] == "baseline") & (result["layer"] == 1)].iloc[0]
    direct = sd.score_one(MODEL, "baseline", 1, None, ctx)
    ranks = dp.rank_vectors(sd.rank_order(direct["scores"]), direct["excluded_queries"], q)["final"]
    assert baseline["acc@1"] == pytest.approx(sd.metric_block(ranks)["acc@1"])
    assert baseline["mrr"] == pytest.approx(sd.metric_block(ranks)["mrr"])
    # A radically different dataset-2 answer set cannot change D.
    q["status"] = "no_key"
    dp.cmd_layers(args)
    again = pd.read_csv(args.out_dir / "results/per_layer.csv")
    np.testing.assert_array_equal(result["D"], again["D"])


def test_render_is_csv_driven_and_does_not_touch_legacy_outputs(parity_scoring, monkeypatch):
    import dataset2_render as dr

    args = parity_scoring
    before = {path.relative_to(args.out_dir): path.read_bytes()
              for path in args.out_dir.rglob("*") if path.is_file()}
    dp.cmd_compare(args)
    layers = pd.DataFrame([
        {"model": sd.MT5, "display": "mT5-base", "method": method, "layer": layer,
         "acc@1": .2 * layer, "mrr": .2 * layer + .1,
         "status": sd.SCORED if method == "baseline" else sd.NOT_APPLICABLE}
        for method in ("baseline", "whitening") for layer in (1, 2)
    ])
    layers.to_csv(args.out_dir / "results/per_layer.csv", index=False)
    # Rendering must not read sources, snapshots or embeddings.
    monkeypatch.setattr(dp, "context", lambda *a: pytest.fail("render read raw data"))
    dr.cmd_render(args)
    paper = args.out_dir / "paper"
    table = (paper / "headline_main.tex").read_text()
    assert r"\begin{table*}" in table and "mT5-base & SIF" in table
    assert r"Char.\ n-grams" in table and "train DirAcc@1" in table
    accuracy = pd.read_csv(args.out_dir / "results/accuracy_long.csv")
    cell = accuracy[(accuracy["label_set"] == "agreed") & (accuracy["seen_group"] == "all")
                    & (accuracy["method"] == "baseline") & (accuracy["metric"] == "acc@1")].iloc[0]
    assert f"{100 * cell['value']:.1f}" in table
    assert f"{100 * cell['ci_lo']:.1f},{100 * cell['ci_hi']:.1f}" in table
    assert (paper / "fig_depth.pdf").read_bytes().startswith(b"%PDF")
    assert (paper / "fig_depth.png").read_bytes().startswith(b"\x89PNG")
    assert "gaps, not zeros" in (paper / "fig_depth.tex").read_text()
    assert "Best $-$ n-grams" in (paper / "headline_ci_diffs.tex").read_text()
    for path, content in before.items():
        assert (args.out_dir / path).read_bytes() == content, path
