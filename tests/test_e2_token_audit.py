"""Guards for scripts/paper/reframe/e2_token_audit.py (issue #252: E2).

Three layers of checks, none of which loads a model, a tokenizer or an embedding cache:

* synthetic arrays (always run): the score identity; the token-share algebra (shares sum
  to 1 and equal Cov(s_g, s) / Var(s)); the token-mix explained variance on a direction
  carried by one token type and on a passage-level shift; the subspace readouts under a
  rotation of the basis; the pooling-arm weight lookups against sif_weights_from_ids; the
  matched random sampler; the frozen decision rules; the gates; the merge-by-model CSV
  writer; the table and the facts file from a tiny fabricated fixture.
* toy tensors (skipped without torch): every pooling arm against the extraction CLIs'
  own pooling functions, and the whole audit of a fake encoder against a fake cache.
* the module imports without torch, as CI needs.
"""

from __future__ import annotations

import subprocess
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
import e2_token_audit as e2  # noqa: E402
from canon_retrieval import upper_triangle_labels  # noqa: E402
from sif_abtt import EmbeddingCleaner, sif_weights_from_ids, weighted_mean_pool  # noqa: E402

LATA, LABSE = "bowphs/LaTa", "sentence-transformers/LaBSE"
EOS = 1  # the one special token of the synthetic passages


# --------------------------------------------------------------------------- #
# synthetic token tables
# --------------------------------------------------------------------------- #

def _tokens(n_tr=40, n_te=30, vocab=50, seed=0, lo=4, hi=15, frequent=(3, 4, 5)):
    """Random passages: token ids 3..vocab-1, an EOS at the end of each."""
    rng = np.random.default_rng(seed)
    n = n_tr + n_te
    lengths = rng.integers(lo, hi, size=n)
    pid = np.repeat(np.arange(n), lengths)
    pos = np.concatenate([np.arange(k) for k in lengths])
    tid = rng.integers(3, vocab, size=len(pid))
    tid[pos == np.repeat(lengths, lengths) - 1] = EOS
    tok = e2.build_token_table(pid, tid, pos == 0, n, [0, EOS], frequent)
    train = np.arange(n) < n_tr
    return tok, train, ~train, rng


def _brute_share(c, tok, rows, members):
    """Cov(s_G, s) / Var(s) over the passages in ``rows`` for the tokens in ``members``."""
    n = len(tok.n)
    s = np.array([c[tok.pid == p].sum() / tok.n[p] for p in range(n)])
    s_g = np.array([c[(tok.pid == p) & members].sum() / tok.n[p] for p in range(n)])
    cov = np.cov(s_g[rows], s[rows], bias=True)
    return cov[0, 1] / cov[1, 1]


# --------------------------------------------------------------------------- #
# score identity, shares, token mix
# --------------------------------------------------------------------------- #

def test_passage_score_is_the_mean_token_contribution():
    tok, train, _, rng = _tokens()
    d = 12
    h = rng.normal(size=(len(tok.pid), d)) * 5 + 40
    pooled = np.stack([h[tok.pid == p].mean(axis=0) for p in range(len(tok.n))])
    mu = pooled[train].mean(axis=0)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    c = (h - mu) @ w
    s = e2.passage_scores(c, tok)[:, 0]
    np.testing.assert_allclose(s, (pooled - mu) @ w, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(tok.n, np.bincount(tok.pid))


def test_token_table_groups_and_first_tokens():
    tok, _, _, _ = _tokens()
    assert set(np.unique(tok.group)) == {0, 1, 2}
    assert (tok.group[tok.tid == EOS] == e2.GROUPS.index("special")).all()
    assert (tok.group[np.isin(tok.tid, [3, 4, 5])] == e2.GROUPS.index("frequent")).all()
    assert tok.first.sum() == len(tok.n)
    np.testing.assert_array_equal(tok.types[tok.type_idx], tok.tid)
    masses = e2.group_masses(tok)
    np.testing.assert_allclose(masses.sum(axis=1), 1.0)
    p = 3
    assert masses[p, 0] == pytest.approx(1 / tok.n[p])  # one EOS per passage


def test_shares_sum_to_one_and_equal_cov_over_var():
    tok, train, test, rng = _tokens()
    c = rng.normal(size=len(tok.pid)) + 2.0 * (tok.group == 1) - 3.0 * (tok.group == 0)
    r = e2.audit_scores(c, tok, train, test)
    for split, rows in (("train", train), ("test", test)):
        share = r[split]["token_share"]
        assert share.sum() == pytest.approx(1.0, abs=1e-12)
        cols = e2.share_columns(share, tok)
        assert abs(cols["share_sum_err"]) < 1e-12
        for i, name in enumerate(e2.GROUPS):
            assert cols[f"share_{name}"] == pytest.approx(
                _brute_share(c, tok, rows, tok.group == i), rel=1e-9, abs=1e-12)
        assert cols["share_first"] == pytest.approx(_brute_share(c, tok, rows, tok.first),
                                                    rel=1e-9, abs=1e-12)
        s = r["s"][rows, 0]
        assert r[split]["var_s"] == pytest.approx(s.var(), rel=1e-12)
        assert r[split]["n_passages"] == int(rows.sum())
    # tokens of the other split carry no share of this split's variance
    assert not r["train"]["token_share"][test[tok.pid]].any()


def test_direction_carried_by_one_token_type():
    tok, train, test, _ = _tokens(n_tr=80, n_te=60)
    carrier = 7
    c = np.where(tok.tid == carrier, 4.0, 0.0)
    r = e2.audit_scores(c, tok, train, test)
    assert r["test"]["ev"] == pytest.approx(1.0, abs=1e-12)
    assert r["test"]["r2"] == pytest.approx(1.0, abs=1e-12)
    type_share = np.bincount(tok.type_idx, weights=r["train"]["token_share"],
                             minlength=len(tok.types))
    order = e2.rank_types(type_share, r["type_count"])
    assert tok.types[order[0]] == carrier
    assert type_share[order[0]] == pytest.approx(1.0, abs=1e-12)
    assert np.abs(np.delete(type_share, order[0])).max() < 1e-12
    assert r["type_mean"][order[0], 0] == pytest.approx(4.0)


def test_passage_level_shift_is_not_explained_by_the_token_mix():
    tok, train, test, rng = _tokens(n_tr=300, n_te=300, vocab=40, lo=25, hi=35)
    shift = rng.normal(size=len(tok.n)) * 10.0
    r = e2.audit_scores(shift[tok.pid], tok, train, test)  # the same for every token
    assert abs(r["test"]["ev"]) < 0.1
    assert r["test"]["r2"] < 0.1
    np.testing.assert_allclose(r["s"][:, 0], shift)


def test_unseen_types_get_the_global_train_mean():
    tok, train, test, rng = _tokens(n_tr=10, n_te=10, vocab=400)
    c = rng.normal(size=len(tok.pid)) + 5.0
    count, mean = e2.type_means(c, tok, train)
    tr = train[tok.pid]
    assert (count == 0).any()
    np.testing.assert_allclose(mean[count == 0, 0], c[tr].mean())
    seen = int(np.flatnonzero(count > 0)[0])
    assert mean[seen, 0] == pytest.approx(c[tr & (tok.type_idx == seen)].mean())
    assert count.sum() == tr.sum()
    info = e2.type_table(tok, train)
    np.testing.assert_array_equal(info["train_count"], count)
    assert info["train_passages"][np.flatnonzero(tok.types == EOS)[0]] == train.sum()
    assert (info["train_passages"] <= info["train_count"]).all()


# --------------------------------------------------------------------------- #
# subspaces
# --------------------------------------------------------------------------- #

def _rotation(j, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(j, j)))
    return q


@pytest.mark.parametrize("j", [2, 3])
def test_subspace_readouts_do_not_depend_on_the_basis(j):
    tok, train, test, rng = _tokens(n_tr=120, n_te=100)
    type_effect = rng.normal(size=(len(tok.types), j)) * np.array([30.0, 3.0, 1.0])[:j]
    c = type_effect[tok.type_idx] + rng.normal(size=(len(tok.pid), j))
    rot = _rotation(j, 5)
    for equal_weight in (False, True):
        a = e2.audit_scores(c, tok, train, test, equal_weight=equal_weight)
        b = e2.audit_scores(c @ rot.T, tok, train, test, equal_weight=equal_weight)
        for split in ("train", "test"):
            assert a[split]["ev"] == pytest.approx(b[split]["ev"], abs=1e-10)
            assert a[split]["var_s"] == pytest.approx(b[split]["var_s"], rel=1e-10)
            np.testing.assert_allclose(a[split]["token_share"], b[split]["token_share"],
                                       rtol=0, atol=1e-10)
            cols_a = e2.share_columns(a[split]["token_share"], tok)
            cols_b = e2.share_columns(b[split]["token_share"], tok)
            for key in cols_a:
                assert cols_a[key] == pytest.approx(cols_b[key], abs=1e-10)
            assert abs(cols_a["share_sum_err"]) < 1e-10
            assert np.isnan(a[split]["r2"])  # defined for a single direction only


def test_subspace_readout_is_the_variance_weighted_mean_of_its_directions():
    tok, train, test, rng = _tokens(n_tr=120, n_te=100)
    c = rng.normal(size=(len(tok.pid), 3)) * np.array([20.0, 4.0, 1.0]) + (tok.group == 1)[:, None]
    joint = e2.audit_scores(c, tok, train, test)
    single = [e2.audit_scores(c[:, k], tok, train, test) for k in range(3)]
    for split in ("train", "test"):
        var = np.array([x[split]["var_s"] for x in single])
        ev = np.array([x[split]["ev"] for x in single])
        assert joint[split]["var_s"] == pytest.approx(var.sum(), rel=1e-12)
        assert joint[split]["ev"] == pytest.approx((ev * var).sum() / var.sum(), abs=1e-12)
        share = sum(x[split]["token_share"] * v for x, v in zip(single, var)) / var.sum()
        np.testing.assert_allclose(joint[split]["token_share"], share, rtol=0, atol=1e-12)


def test_equal_weight_divides_each_principal_component_by_its_train_sd():
    tok, train, test, rng = _tokens(n_tr=150, n_te=100)
    c = rng.normal(size=(len(tok.pid), 3)) * np.array([50.0, 5.0, 1.0])
    c += (tok.type_idx % 7)[:, None] * np.array([3.0, 0.5, 0.2])
    # rotate to the basis in which the TRAIN passage scores are uncorrelated (the PC basis)
    s = e2.passage_scores(c, tok)[train]
    vals, vecs = np.linalg.eigh(np.cov(s.T, bias=True))
    pc = c @ vecs
    eq = e2.audit_scores(pc, tok, train, test, equal_weight=True)
    by_hand = e2.audit_scores(pc / np.sqrt(vals), tok, train, test)
    plain = e2.audit_scores(pc, tok, train, test)
    for split in ("train", "test"):
        assert eq[split]["ev"] == pytest.approx(by_hand[split]["ev"], abs=1e-9)
        np.testing.assert_allclose(eq[split]["token_share"], by_hand[split]["token_share"],
                                   rtol=0, atol=1e-9)
    # on train every whitened component has variance 1, so the EV is the plain mean
    single = [e2.audit_scores(pc[:, k], tok, train, test)["train"]["ev"] for k in range(3)]
    assert eq["train"]["ev"] == pytest.approx(np.mean(single), abs=1e-9)
    assert eq["train"]["var_s"] == pytest.approx(3.0, rel=1e-9)
    assert abs(eq["train"]["ev"] - plain["train"]["ev"]) > 1e-4  # the two weightings differ
    # for one direction the equal-weight variant is the plain one
    one = e2.audit_scores(c[:, 0], tok, train, test, equal_weight=True)
    assert one["test"]["ev"] == pytest.approx(
        e2.audit_scores(c[:, 0], tok, train, test)["test"]["ev"], abs=1e-10)


def test_subspace_list_covers_the_two_pc_spans_and_disjoint_random_controls():
    names = [s[0] for s in e2.SUBSPACES]
    assert names[:2] == ["pcs2_3", "pcs1_3"]
    assert dict((s[0], s[2]) for s in e2.SUBSPACES)["pcs2_3"] == ("pc2", "pc3")
    assert dict((s[0], s[2]) for s in e2.SUBSPACES)["pcs1_3"] == ("pc1", "pc2", "pc3")
    pairs = [s[2] for s in e2.SUBSPACES if s[0].startswith("randpair")]
    triples = [s[2] for s in e2.SUBSPACES if s[0].startswith("randtriple")]
    assert len(pairs) == 10 and len(triples) == 6
    assert len({m for p in pairs for m in p}) == 20 and len({m for t in triples for m in t}) == 18
    assert all(m in e2.RAND_NAMES for p in pairs + triples for m in p)


# --------------------------------------------------------------------------- #
# directions
# --------------------------------------------------------------------------- #

def test_directions_are_abtt_components_e1_coordinates_and_orthonormal_randoms():
    rng = np.random.default_rng(3)
    train = (rng.normal(size=(60, 30)) * np.linspace(9, 1, 30)).astype(np.float32) + 4.0
    d = e2.fit_directions(train, seed=(e2.RANDOM_SEED, 0, 5))
    assert d["names"] == list(e2.DIRECTIONS) and d["W"].shape == (26, 30)
    cleaner = EmbeddingCleaner(num_components=3, center=True).fit(train)
    for k in range(3):
        w = d["W"][k]
        assert abs(abs(w @ cleaner.pcs[k]) - 1.0) < 1e-5  # ABTT's component, up to sign
        assert w[np.argmax(np.abs(w))] > 0  # sign convention
    np.testing.assert_allclose(d["mu"], train.mean(axis=0), rtol=0, atol=1e-5)
    coords = list(e1.rank_coords(train, "variance")[:3])
    assert d["coords"][3:6] == coords and d["coords"][:3] == [-1] * 3
    for k, i in enumerate(coords):
        assert d["W"][3 + k, i] == 1.0 and np.abs(d["W"][3 + k]).sum() == 1.0
    rand = d["W"][6:]
    np.testing.assert_allclose(rand @ rand.T, np.eye(20), atol=1e-10)
    top10 = EmbeddingCleaner(num_components=10, center=True).fit(train).pcs.astype(np.float64)
    assert np.abs(rand @ top10.T).max() < 1e-6
    # seeded per model-layer: same seed, same directions; another layer, others
    again = e2.fit_directions(train, seed=(e2.RANDOM_SEED, 0, 5))
    np.testing.assert_array_equal(again["W"], d["W"])
    other = e2.fit_directions(train, seed=(e2.RANDOM_SEED, 0, 6))
    assert not np.allclose(other["W"][6:], rand)
    np.testing.assert_array_equal(other["W"][:6], d["W"][:6])


def test_fix_sign_makes_the_largest_loading_positive():
    np.testing.assert_array_equal(e2.fix_sign(np.array([0.1, -0.9, 0.3])), [-0.1, 0.9, -0.3])
    np.testing.assert_array_equal(e2.fix_sign(np.array([0.1, 0.9, -0.3])), [0.1, 0.9, -0.3])


# --------------------------------------------------------------------------- #
# pooling arms
# --------------------------------------------------------------------------- #

def _vocab(seed=0, vocab=40):
    rng = np.random.default_rng(seed)
    keep = np.ones(vocab, dtype=np.float32)
    keep[[2, 9]] = 0.0  # dropped by the token filter
    probs = {int(t): float(p) for t, p in zip(range(3, 30), rng.dirichlet(np.ones(27)))}
    return keep, probs, [0, EOS]


def test_arm_lookups_are_the_cli_weights_per_token_id():
    keep, probs, special = _vocab()
    look = e2.arm_weight_lookups(len(keep), probs, special, keep)
    assert set(look) == set(e2.ARMS) and all(v.dtype == np.float32 for v in look.values())
    ids = np.random.default_rng(1).integers(0, len(keep), size=(6, 11))
    np.testing.assert_array_equal(
        look["sif"][ids], sif_weights_from_ids(ids, probs, a=e2.SIF_A, special_ids=special,
                                               token_keep_lookup=keep))
    np.testing.assert_array_equal(
        look["sif_keepspecial"][ids],
        sif_weights_from_ids(ids, probs, a=e2.SIF_A, special_ids=None, token_keep_lookup=keep))
    # the two SIF arms differ on the special tokens only, where one is 0 and the other 1
    differ = np.flatnonzero(look["sif"] != look["sif_keepspecial"])
    assert list(differ) == special
    assert (look["sif"][special] == 0).all() and (look["sif_keepspecial"][special] == 1).all()
    np.testing.assert_array_equal(look["mean"], keep)
    assert list(np.flatnonzero(look["mean"] != look["mean_nospecial"])) == special
    top = e2.frequent_ids(probs, 100)
    assert len(top) == len(probs)  # fewer than 100 types: all of them
    assert not look["mean_nofreq100"][top].any() and look["mean_nofreq100"][special].all()
    assert look["mean_nofreq100"][35] == 1.0 and look["mean_nofreq100"][9] == 0.0
    # an unseen, kept token has SIF weight 1; a filtered one 0 under every arm
    assert look["sif"][35] == 1.0 and all(v[9] == 0.0 for v in look.values())
    p = probs[3]
    assert look["sif"][3] == np.float32(e2.SIF_A / (e2.SIF_A + p))


def test_frequent_ids_are_the_most_probable_with_ties_by_id():
    probs = {5: 0.2, 9: 0.3, 2: 0.2, 7: 0.1}
    assert list(e2.frequent_ids(probs, 3)) == [9, 2, 5]
    assert list(e2.frequent_ids(probs, 100)) == [9, 2, 5, 7]


def _toy_batch(seed=0, b=5, t=9, d=6, vocab=40):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    ids = rng.integers(2, vocab, size=(b, t))
    att = np.ones((b, t), dtype=np.int64)
    for i, n in enumerate(rng.integers(3, t + 1, size=b)):
        ids[i, n - 1] = EOS
        ids[i, n:] = 0
        att[i, n:] = 0
    ids[0, :] = [9, 2, EOS] + [0] * (t - 3)  # a passage whose only kept token is special
    att[0, :] = [1, 1, 1] + [0] * (t - 3)
    hidden = torch.tensor(rng.normal(size=(b, t, d)) * 20, dtype=torch.float32)
    return torch, torch.tensor(ids), torch.tensor(att), hidden


@pytest.mark.parametrize("cli_name,fn_name", [("extract_hidden_cli", "pool_hidden"),
                                              ("extract_encoder_cli", "pool_embeddings")])
def test_every_arm_equals_the_cli_pooling_function(cli_name, fn_name):
    torch, ids, att, hidden = _toy_batch()
    pytest.importorskip("transformers")
    cli_pool = getattr(__import__(cli_name), fn_name)
    keep, probs, special = _vocab()
    look = e2.arm_weight_lookups(len(keep), probs, special, keep)

    def ours(arm):
        return e2.pool_weighted(hidden, att.float() * torch.as_tensor(look[arm])[ids])

    def cli(pooling, lookup, special_ids):
        return cli_pool(hidden, att, pooling, input_ids=ids, token_probs=probs,
                        sif_a=e2.SIF_A, special_ids=special_ids, token_keep_lookup=lookup)

    nospecial = keep.copy()
    nospecial[special] = 0.0
    nofreq = keep.copy()
    nofreq[e2.frequent_ids(probs)] = 0.0
    expect = {"mean": cli("mean", keep, set(special)),
              "mean_nospecial": cli("mean", nospecial, set(special)),
              "sif_keepspecial": cli("sif", keep, set()),
              "sif": cli("sif", keep, set(special)),
              "mean_nofreq100": cli("mean", nofreq, set(special))}
    for arm in e2.ARMS:
        assert torch.equal(ours(arm), expect[arm]), arm
    # the arms are not all the same thing
    assert not torch.equal(expect["mean"], expect["mean_nospecial"])
    assert not torch.equal(expect["sif"], expect["sif_keepspecial"])
    # a passage with no token under an arm is a zero vector there (the audit falls back)
    assert not ours("mean_nospecial")[0].any() and ours("mean")[0].any()
    # and the numpy pooling of src agrees
    w = (att.numpy() * look["sif"][ids.numpy()]).astype(np.float32)
    np.testing.assert_allclose(ours("sif").numpy(),
                               weighted_mean_pool(hidden.numpy(), att.numpy(), w / 1.0),
                               rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# matched random control and ablation arms
# --------------------------------------------------------------------------- #

def test_matched_random_draws_the_nearest_counts_without_replacement():
    count = np.array([0, 1000, 990, 980, 500, 495, 490, 10, 9, 8, 7, 0])
    carriers = [1, 4]
    seen = set()
    for seed in range(40):
        picks = e2.matched_random(count, carriers, excluded=carriers,
                                  rng=np.random.default_rng(seed), window=2)
        assert len(picks) == 2 and len(set(picks)) == 2
        assert picks[0] in (2, 3)  # the two nearest to 1000
        assert picks[1] in (5, 6)  # the two nearest to 500
        seen.add(tuple(picks))
    assert len(seen) > 1  # it is a draw, not a fixed choice
    # never a carrier, an excluded type or a type unseen in train
    picks = e2.matched_random(count, [1, 4, 7], excluded=[1, 4, 7, 2, 3],
                              rng=np.random.default_rng(0), window=3)
    assert not set(picks) & {0, 11, 1, 4, 7, 2, 3}
    # without replacement: with one candidate left per step the draw is forced
    picks = e2.matched_random(np.array([5, 5, 5]), [0, 0], excluded=[0],
                              rng=np.random.default_rng(0), window=10)
    assert sorted(picks) == [1, 2]
    # the eligible types can run out
    assert len(e2.matched_random(np.array([5, 5]), [0, 0, 0], excluded=[0],
                                 rng=np.random.default_rng(0))) == 1
    # same generator state, same draw
    a = e2.matched_random(count, carriers, carriers, np.random.default_rng(7))
    b = e2.matched_random(count, carriers, carriers, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_ablation_arms_are_nested_seeded_and_exclude_every_carrier():
    rng = np.random.default_rng(0)
    n_types = 600
    share = rng.normal(size=(3, n_types))
    count = rng.integers(0, 500, size=n_types)
    arms = e2.ablation_arms(share, count, seed=(e2.RANDOM_SEED, 2, 7))
    assert len(arms) == len(e2.RANKINGS) * (len(e2.ABL_MS) * (1 + e2.CONTROL_DRAWS))
    carriers = {r: next(a["types"] for a in arms if a["ranking"] == r and a["kind"] == "carrier"
                        and a["m"] == 100) for r in e2.RANKINGS}
    seen = np.flatnonzero(count > 0)
    np.testing.assert_array_equal(
        carriers["pc1"], seen[np.argsort(-share[0][seen], kind="stable")][:100])
    np.testing.assert_array_equal(
        carriers["pc123"], seen[np.argsort(-share.mean(axis=0)[seen], kind="stable")][:100])
    every_carrier = set(carriers["pc1"]) | set(carriers["pc123"])
    for a in arms:
        assert len(a["types"]) == a["m"] and (count[a["types"]] > 0).all()
        full = next(b["types"] for b in arms if (b["ranking"], b["kind"], b["draw"], b["m"])
                    == (a["ranking"], a["kind"], a["draw"], 100))
        np.testing.assert_array_equal(a["types"], full[:a["m"]])  # nested in m
        if a["kind"] == "control":
            assert not set(a["types"]) & every_carrier
            assert len(set(a["types"])) == a["m"]
    draws = [tuple(a["types"]) for a in arms if a["kind"] == "control" and a["m"] == 100
             and a["ranking"] == "pc1"]
    assert len(set(draws)) == e2.CONTROL_DRAWS
    # matched in train frequency: the control's counts track the carriers'
    ctl = next(a["types"] for a in arms if a["kind"] == "control" and a["m"] == 100)
    assert np.abs(count[ctl] - count[carriers["pc1"]]).mean() < 25
    again = e2.ablation_arms(share, count, seed=(e2.RANDOM_SEED, 2, 7))
    for a, b in zip(arms, again):
        np.testing.assert_array_equal(a["types"], b["types"])
    other = e2.ablation_arms(share, count, seed=(e2.RANDOM_SEED, 2, 8))
    assert any(not np.array_equal(a["types"], b["types"]) for a, b in zip(arms, other))


# --------------------------------------------------------------------------- #
# length check
# --------------------------------------------------------------------------- #

def test_length_stats_match_a_pair_loop():
    rng = np.random.default_rng(4)
    n = rng.integers(1, 200, size=30)
    ids = rng.integers(0, 6, size=30).astype(str)
    stats = e2.length_stats(n, upper_triangle_labels(ids))
    same, diff = [], []
    for i in range(30):
        for j in range(i + 1, 30):
            (same if ids[i] == ids[j] else diff).append(abs(np.log(n[i]) - np.log(n[j])))
    assert stats["n_same_pairs"] == len(same) and stats["n_diff_pairs"] == len(diff)
    assert stats["mean_dlog_same"] == pytest.approx(np.mean(same))
    assert stats["mean_dlog_diff"] == pytest.approx(np.mean(diff))
    assert stats["median_dlog_same"] == pytest.approx(np.median(same))
    assert stats["median_dlog_diff"] == pytest.approx(np.median(diff))
    # same-directory passages of equal length: -|delta log n| separates the pairs perfectly
    n2 = np.array([10, 10, 50, 50, 300, 300])
    perfect = e2.length_stats(n2, upper_triangle_labels(np.array(list("aabbcc"))))
    assert perfect["auroc_neg_dlog"] == pytest.approx(1.0)
    assert perfect["mean_dlog_same"] == 0.0 < perfect["mean_dlog_diff"]
    # a passage with no token counts as length 1, not as log 0
    assert np.isfinite(e2.length_stats(np.array([0, 3, 4]),
                                       upper_triangle_labels(np.array(list("aab"))))
                       ["mean_dlog_same"])


# --------------------------------------------------------------------------- #
# decision rules
# --------------------------------------------------------------------------- #

def test_frozen_constants():
    assert (e2.R1_RESCUE_FRAC, e2.R1_MIN_SIF_GAIN) == (0.80, 0.05)
    assert e2.R2_EV == 0.5 and (e2.R3_AUROC, e2.R3_MAX_M) == (0.90, 100)
    assert e2.R4_RHO == 0.5 and e2.COLLAPSE_AUROC == 0.70
    assert e2.ABL_MS == (1, 3, 10, 30, 100) and e2.CONTROL_DRAWS == 5
    assert e2.N_FREQUENT == 100 and e2.N_RANDOM == 20 and e2.RANDOM_SEED == 233
    assert e2.ARMS == ("mean", "mean_nospecial", "sif_keepspecial", "sif", "mean_nofreq100")
    assert (e2.MAX_LENGTH, e2.BATCH_SIZE, e2.SIF_A, e2.TOKEN_FILTER) == (
        512, 8, 1e-3, "tokenizer_empty")
    assert set(e2.EXTRACT) == set(e2.ALL_MODEL_IDS)


def test_r1_rescue():
    assert e2.r1_rescue(0.86, 0.50, 0.90)[0] == "rescue"
    status, frac, change = e2.r1_rescue(0.821, 0.50, 0.90)
    assert status == "rescue" and frac == pytest.approx(0.8025) and change == pytest.approx(0.321)
    assert e2.r1_rescue(0.75, 0.50, 0.75)[1] == 1.0  # SIF itself recovers all of its gain
    assert e2.r1_rescue(0.875, 0.375, 1.0)[:2] == ("rescue", 0.8)  # the bar is inclusive
    status, frac, _ = e2.r1_rescue(0.81, 0.50, 0.90)
    assert status == "no_rescue" and frac == pytest.approx(0.775)
    assert e2.r1_rescue(0.40, 0.50, 0.90)[0] == "no_rescue"  # a drop is not a rescue
    # no SIF gain to recover: the share is not evaluated, the raw change is reported
    status, frac, change = e2.r1_rescue(0.70, 0.654, 0.672)
    assert status == "no_sif_gain" and np.isnan(frac) and change == pytest.approx(0.046)
    assert e2.r1_rescue(0.60, 0.50, 0.5501)[0] == "rescue"  # just above the floor
    assert e2.r1_rescue(0.60, 0.50, 0.5499)[0] == "no_sif_gain"
    assert e2.r1_rescue(0.60, 0.70, 0.60)[0] == "no_sif_gain"  # SIF hurts
    assert e2.r1_rescue(float("nan"), 0.5, 0.9)[0] == "undefined"
    assert e2.r1_rescue(0.9, 0.5, float("nan"))[0] == "undefined"


def test_r2_r3_r4():
    assert e2.r2_token_mix(0.5) and not e2.r2_token_mix(0.4999)
    assert not e2.r2_token_mix(float("nan")) and not e2.r2_token_mix(-2.0)
    assert e2.r3_restoring_m({1: 0.5, 3: 0.6, 10: 0.91, 30: 0.95, 100: 0.99}) == 10
    assert e2.r3_restoring_m({1: 0.5, 100: 0.90}) == 100
    assert e2.r3_restoring_m({1: 0.5, 100: 0.8999}) is None
    assert e2.r3_restoring_m({300: 0.99, 100: float("nan")}) is None  # m above 100 is no repair
    assert e2.r4_length(-0.6, 0.9, 0.7) and e2.r4_length(0.5, 0.9, 0.7)
    assert not e2.r4_length(0.49, 0.9, 0.7)  # weak correlation
    assert not e2.r4_length(0.8, 0.7, 0.9)  # same-directory pairs differ less in length
    assert not e2.r4_length(0.8, 0.7, 0.7) and not e2.r4_length(float("nan"), 0.9, 0.7)


# --------------------------------------------------------------------------- #
# CSV plumbing
# --------------------------------------------------------------------------- #

def test_merge_write_replaces_only_the_rerun_models(tmp_path):
    path = tmp_path / "x.csv"
    first = pd.DataFrame({"model": [LABSE, LABSE, LATA, LATA], "layer": [1, 2, 1, 2],
                          "aucroc": [0.81234567891234, 0.82, 0.5, np.nan],
                          "token": ["nan", "", "NA", " et"]})
    e2.merge_write(path, first)
    back = e2.read_csv(path)
    assert list(back.model) == [LATA, LATA, LABSE, LABSE]  # panel order
    assert list(back.token) == ["NA", " et", "nan", ""]  # text stays text
    assert np.isnan(back.aucroc[1]) and back.aucroc[2] == pytest.approx(0.8123456789)
    before = path.read_bytes()
    labse_lines = [ln for ln in path.read_text().splitlines() if ln.startswith(LABSE)]
    # a rerun of LaTa alone: its rows change, LaBSE's stay byte for byte
    e2.merge_write(path, pd.DataFrame({"model": [LATA], "layer": [7], "aucroc": [0.9],
                                       "token": ["x,y"]}))
    back = e2.read_csv(path)
    assert list(back.model) == [LATA, LABSE, LABSE] and list(back.layer) == [7, 1, 2]
    assert back.token[0] == "x,y"
    assert [ln for ln in path.read_text().splitlines() if ln.startswith(LABSE)] == labse_lines
    # writing the same rows again changes nothing
    e2.merge_write(path, first[first.model == LATA])
    e2.merge_write(path, first[first.model == LABSE])
    assert path.read_bytes() == before
    e2.merge_write(path, first)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_visible_escapes_what_would_break_a_cell():
    assert e2.visible(" et") == " et" and e2.visible("▁et") == "▁et"
    assert e2.visible("a\nb\tc\rd") == "a\\nb\\tc\\rd"
    assert e2.visible("a\\b") == "a\\\\b"
    assert "\n" not in e2.visible("\x00\x85 ")


def test_limit_rows_takes_the_first_of_each_split():
    split = pd.DataFrame({"split": ["train", "test", "test", "train", "train", "test"]})
    np.testing.assert_array_equal(e2.limit_rows(split, 0), np.arange(6))
    np.testing.assert_array_equal(e2.limit_rows(split, 2), [0, 1, 2, 3])
    np.testing.assert_array_equal(e2.limit_rows(split, 1), [0, 1])
    np.testing.assert_array_equal(e2.limit_rows(split, 50), np.arange(6))


# --------------------------------------------------------------------------- #
# fabricated result CSVs (numpy only): gates and render
# --------------------------------------------------------------------------- #

MT5 = "google/mt5-base"
# Task A test AUROC per arm, in the order of e2.ARMS. LaTa 2 and 3 and mT5-base 5 are
# collapsed; at LaTa 3 and mT5-base 5 SIF gains less than 0.05.
FIX_AUC = {(LATA, 1): (0.93, 0.93, 0.94, 0.95, 0.94),
           (LATA, 2): (0.50, 0.51, 0.86, 0.88, 0.80),
           (LATA, 3): (0.66, 0.67, 0.67, 0.68, 0.69),
           (MT5, 5): (0.654, 0.655, 0.66, 0.672, 0.70),
           (LABSE, 1): (0.81, 0.80, 0.82, 0.82, 0.83)}
# carrier-drop AUROC by m for the pc1 ranking; pc123 is 0.02 lower; controls stay near mean
FIX_ABL = {(LATA, 2): (0.52, 0.60, 0.85, 0.93, 0.95), (LATA, 3): (0.66, 0.67, 0.70, 0.75, 0.80)}


def _np_layer(mid, layer, seed, nuisance):
    """One fake model-layer through layer_audit: random token states in which the
    frequent tokens carry a planted direction of size ``nuisance``."""
    tok, train, test, rng = _tokens(n_tr=40, n_te=30, vocab=60, seed=seed)
    d = 40
    h = rng.normal(size=(60, d))[tok.tid] + 0.5 * rng.normal(size=(len(tok.pid), d))
    h[:, 0] += nuisance * (tok.group == 1)
    h = h.astype(np.float32).astype(np.float64)
    pooled = e2.passage_scores(h, tok)
    cached = pooled.astype(np.float32)
    dirs = e2.fit_directions(cached[train], seed=(e2.RANDOM_SEED, 0, layer))
    audit, carriers, share = e2.layer_audit(
        {"model": mid, "layer": layer}, (h - dirs["mu"]) @ dirs["W"].T, tok, dirs, train, test,
        e2.group_masses(tok), (pooled - dirs["mu"]) @ dirs["W"].T, cached,
        lambda i: (f"▁t{i}", f" t{i}" if i != EOS else "</s>"), e2.type_table(tok, train))
    return audit, carriers, share, tok, train, test


def _fixture(limit=0):
    """The five result frames of three fake models, and the matching published cells."""
    arms, audit, carriers, abl, length, res = [], [], [], [], [], []
    block = list(asw.KEEP) + ["pc1_share_train", "pc10_share_train", "eff_rank_train"]
    for i, ((mid, layer), aucs) in enumerate(FIX_AUC.items()):
        a, c, _, tok, train, test = _np_layer(mid, layer, seed=i,
                                              nuisance=30.0 if aucs[0] < 0.7 else 0.0)
        audit += a
        carriers += c
        for arm, auc in zip(e2.ARMS, aucs):
            ref = arm in ("mean", "sif")
            arms.append({"model": mid, "layer": layer, "arm": arm,
                         **{k: 0.5 for k in block}, "aucroc": auc, "train_aucroc": auc + 0.01,
                         "pc1_share_train": 0.9 if arm == "mean" and auc < 0.7 else 0.2,
                         "n_no_token": 0, "n_fallback": 0, "special_mass": 0.05,
                         "frequent_mass": 0.4,
                         "vec_max_abs_diff": 1e-5 if ref else np.nan,
                         "vec_max_rel_diff": 1e-7 if ref else np.nan,
                         "cli_pool_max_abs_diff": 0.0 if ref else np.nan,
                         "n_train": 40, "n_test": 30, "limit": limit})
        res.append({"model": mid, "repr": "hidden", "pooling": "mean", "layer": layer,
                    "method": "baseline", "aucroc": aucs[0]})
        res.append({"model": mid, "repr": "hidden", "pooling": "sif", "layer": layer,
                    "method": "sif_only", "aucroc": aucs[3]})
        res.append({"model": mid, "repr": "hidden", "pooling": "mean", "layer": layer,
                    "method": "abtt_fixed", "aucroc": 0.123})
        by_m = FIX_ABL.get((mid, layer), (aucs[0],) * 5)
        for ri, ranking in enumerate(e2.RANKINGS):
            for kind, draws in (("carrier", [-1]), ("control", range(e2.CONTROL_DRAWS))):
                for draw in draws:
                    for m, auc in zip(e2.ABL_MS, by_m):
                        val = auc - 0.02 * ri if kind == "carrier" else aucs[0] + 0.001 * draw
                        abl.append({"model": mid, "layer": layer, "ranking": ranking,
                                    "kind": kind, "draw": draw, "m": m, "n_types": m,
                                    "types": ";".join(str(3 + j) for j in range(m)),
                                    **{k: np.nan for k in block}, "aucroc": val,
                                    "train_aucroc": val, "n_fallback": 0,
                                    "dropped_mass_train": 0.01 * m ** 0.5,
                                    "dropped_mass_test": 0.01 * m ** 0.5,
                                    "train_count_dropped": 10 * m})
        if layer == min(x for m_, x in FIX_AUC if m_ == mid):
            ids = np.random.default_rng(i).integers(0, 9, size=len(tok.n)).astype(str)
            for split, rows in (("train", train), ("test", test)):
                n = tok.n[rows]
                length.append({"model": mid, "split": split, "n_passages": int(rows.sum()),
                               "n_zero_token": 0, "n_truncated": 0, "mean_n": float(n.mean()),
                               "median_n": float(np.median(n)),
                               **e2.length_stats(n, upper_triangle_labels(ids[rows]))})
    frames = {e2.ARMS_NAME: pd.DataFrame(arms), e2.AUDIT_NAME: pd.DataFrame(audit),
              e2.CARRIER_NAME: pd.DataFrame(carriers), e2.ABL_NAME: pd.DataFrame(abl),
              e2.LENGTH_NAME: pd.DataFrame(length)}
    return frames, pd.DataFrame(res)


FIX_MODELS = [LATA, MT5, LABSE]


def _gt(frames, res, **kwargs):
    kwargs.setdefault("expected", FIX_MODELS)
    return e2.gate_table(frames[e2.ARMS_NAME], frames[e2.AUDIT_NAME], res, **kwargs)


def _failing(g):
    return sorted((row.model, row.gate.split(" ")[0]) for row in g.itertuples()
                  if row.gated and not row.ok)


def _set(df, where, column, value):
    out = df.copy()
    assert where(out).sum() >= 1
    out.loc[where(out), column] = value
    return out


def test_layer_audit_rows_identity_shares_and_carriers():
    audit, carriers, share, tok, train, test = _np_layer(LATA, 4, seed=3, nuisance=30.0)
    a = pd.DataFrame(audit)
    n_dir = len(e2.DIRECTIONS) + len(e2.SUBSPACES)
    assert len(a) == 2 * n_dir and list(a.direction[:2]) == ["pc1", "pc1"]
    assert list(a.split[:2]) == ["train", "test"]
    single, joint = a[a.dim == 1], a[a.dim > 1]
    assert set(joint.direction) == {s[0] for s in e2.SUBSPACES}
    assert set(joint.kind) == {"pc_subspace", "random_subspace"}
    # gates 3 and 4 hold on exact arithmetic
    assert single.identity_max_rel_diff.max() < 1e-12
    assert single.cache_score_max_abs_diff.max() < 1e-4  # float32 cache against float64
    assert a.share_sum_err.abs().max() < 1e-10 and joint.share_sum_err_eqw.abs().max() < 1e-10
    assert joint.identity_max_rel_diff.isna().all() and joint.rho_logn.isna().all()
    assert single.ev_tokenmix_eqw.isna().all() and joint.ev_tokenmix_eqw.notna().all()
    pc1 = a[(a.direction == "pc1") & (a.split == "test")].iloc[0]
    # the planted direction is carried by the frequent tokens, and the token mix explains it
    assert pc1.share_frequent > 0.8 and pc1.ev_tokenmix > 0.8
    assert pc1.rho_freqmass > 0.8
    assert pc1.mass_special + pc1.mass_frequent + pc1.mass_other == pytest.approx(1.0)
    assert pc1.var_share > 0.25
    rand = a[(a.kind == "random") & (a.split == "test")]
    assert len(rand) == e2.N_RANDOM and rand.var_share.max() < 0.1
    # PC variances decrease; the joint variance is the sum of its members'
    var = {d: a[(a.direction == d) & (a.split == "train")].var_s.iloc[0] for d in e2.PC_NAMES}
    assert var["pc1"] > var["pc2"] > var["pc3"]
    j = a[(a.direction == "pcs2_3") & (a.split == "train")].iloc[0]
    assert j.var_s == pytest.approx(var["pc2"] + var["pc3"], rel=1e-12) and j.dim == 2
    # the train score variance on PC1 is the top eigenvalue of the cached train vectors
    c = pd.DataFrame(carriers)
    assert len(c) == e2.N_PC * e2.TOP_CARRIERS and list(c["rank"][:3]) == [1, 2, 3]
    top = c[(c.direction == "pc1")]
    assert (top.share.diff().dropna() <= 1e-15).all()  # sorted by share
    assert set(top.group[:3]) == {"frequent"} and set(top.token_id[:3]) == {3, 4, 5}
    assert top.token.iloc[0].startswith("▁t") and top.decoded.iloc[0].startswith(" t")
    assert (top.train_passages <= top.train_count).all()
    assert share.shape == (e2.N_PC, len(tok.types))
    np.testing.assert_allclose(share.sum(axis=1), 1.0, atol=1e-10)


def test_gates_pass_on_matching_references_and_fail_on_drift():
    frames, res = _fixture()
    g = _gt(frames, res)
    assert len(g) == 6 * len(FIX_MODELS) and e2.gates_passed(g) and g.ok.all()
    assert list(g.gate.str.split(" ").str[0][:6]) == ["1a", "1b", "2a", "2b", "3", "4"]
    assert list(g[g.gate.str.startswith("2b")].gated) == [0, 0, 0]
    assert (g[g.gate.str.startswith(("1a", "2a"))].max_diff == 0).all()
    assert list(g[g.gate.str.startswith("1a")].n_cells) == [3, 1, 1]

    drift = _set(res, lambda r: (r.model == LATA) & (r.layer == 2) & (r.method == "baseline"),
                 "aucroc", 0.50 + 5e-6)
    g = _gt(frames, drift)
    assert _failing(g) == [(LATA, "1a")] and not e2.gates_passed(g)
    assert g[~g.ok].iloc[0].cells_over_tolerance.startswith("L2 5.00e-06")
    assert e2.gates_passed(_gt(frames, drift, tol_auroc=1e-5))
    drift = _set(res, lambda r: (r.model == MT5) & (r.method == "sif_only"), "aucroc", 0.7)
    assert _failing(_gt(frames, drift)) == [(MT5, "2a")]

    # a published layer that was not computed, and a missing published cell
    extra = pd.concat([res, res[(res.model == LATA) & (res.layer == 2)].assign(layer=9)])
    assert _failing(_gt(frames, extra)) == [(LATA, "1a"), (LATA, "2a")]
    assert e2.gates_passed(_gt(frames, extra, complete=False))
    fewer = res[~((res.model == LABSE) & (res.method == "sif_only"))]
    g = _gt(frames, fewer)
    assert _failing(g) == [(LABSE, "2a")] and g[~g.ok].iloc[0].n_missing_reference == 1
    # no published cells at all: the AUROC gates fail, the others still pass
    assert _failing(_gt(frames, None)) == sorted((m, t) for m in FIX_MODELS for t in ("1a", "2a"))
    # a limit run has no AUROC gates
    g = _gt(frames, None, auroc_gates=False)
    assert len(g) == 4 * len(FIX_MODELS) and e2.gates_passed(g)


def test_vector_identity_and_share_gates():
    frames, res = _fixture()
    arms, audit = frames[e2.ARMS_NAME], frames[e2.AUDIT_NAME]

    def table(arms_=arms, audit_=audit):
        return e2.gate_table(arms_, audit_, res, expected=FIX_MODELS)

    # 1b: the re-derived mean vectors must be the cache
    bad = _set(arms, lambda a: (a.model == LATA) & (a.layer == 3) & (a.arm == "mean"),
               "vec_max_rel_diff", 0.02)
    g = table(bad)
    assert _failing(g) == [(LATA, "1b")] and "max abs diff" in g[~g.ok].iloc[0].note
    # 2b: the sif vectors are reported and never fail; an absent cache is said so
    bad = _set(arms, lambda a: (a.model == LATA) & (a.layer == 3) & (a.arm == "sif"),
               "vec_max_rel_diff", 0.02)
    g = table(bad)
    row = g[(g.model == LATA) & g.gate.str.startswith("2b")].iloc[0]
    assert e2.gates_passed(g) and row.n_over_tolerance == 1 and row.gated == 0
    assert "reference 1e-03; 1 over" in e2.gate_line(row) and "PASS" not in e2.gate_line(row)
    none = _set(arms, lambda a: (a.model == LABSE) & (a.arm == "sif"),
                ["vec_max_rel_diff", "vec_max_abs_diff"], np.nan)
    g = table(none)
    row = g[(g.model == LABSE) & g.gate.str.startswith("2b")].iloc[0]
    assert e2.gates_passed(g) and row.n_cells == 0 and "no hidden_sif_tokempty cache" in row.note
    # 3: the score identity, per direction and split; a NaN fails
    for value in (1e-6, np.nan):
        bad = _set(audit, lambda a: (a.model == MT5) & (a.direction == "rand07")
                   & (a.split == "test"), "identity_max_rel_diff", value)
        g = table(audit_=bad)
        assert _failing(g) == [(MT5, "3")]
        assert g[~g.ok].iloc[0].cells_over_tolerance.startswith("L5 rand07 test")
    # 4: shares sum to 1, also the equal-weight shares of the subspaces
    bad = _set(audit, lambda a: (a.model == LATA) & (a.layer == 1) & (a.direction == "pc2")
               & (a.split == "train"), "share_sum_err", -1e-5)
    assert _failing(table(audit_=bad)) == [(LATA, "4")]
    bad = _set(audit, lambda a: (a.model == LATA) & (a.layer == 1) & (a.direction == "pcs1_3")
               & (a.split == "train"), "share_sum_err_eqw", 1e-5)
    g = table(audit_=bad)
    assert _failing(g) == [(LATA, "4")] and "equal-weight" in g[~g.ok].iloc[0].cells_over_tolerance
    # a NaN AUROC of ours fails its gate
    bad = _set(arms, lambda a: (a.model == LABSE) & (a.arm == "sif"), "aucroc", np.nan)
    g = table(bad)
    assert _failing(g) == [(LABSE, "2a")] and np.isnan(g[~g.ok].iloc[0].max_diff)


def test_an_expected_model_without_rows_fails_the_gates():
    frames, res = _fixture()
    g = e2.gate_table(frames[e2.ARMS_NAME], frames[e2.AUDIT_NAME], res)
    missing = g[g.gate.str.startswith("0")]
    assert sorted(missing.model) == sorted(m for m in e2.ALL_MODEL_IDS if m not in FIX_MODELS)
    assert not e2.gates_passed(g) and "no rows: FAIL" in e2.gate_line(missing.iloc[0])
    assert e2.gates_passed(e2.gate_table(frames[e2.ARMS_NAME], frames[e2.AUDIT_NAME], res,
                                         expected=None))


def _write(frames, res, tmp_path, name="e2"):
    out = tmp_path / name
    out.mkdir()
    for fname, frame in frames.items():
        e2.merge_write(out / fname, frame)
    res.to_csv(tmp_path / f"{name}_res.csv", index=False)
    return out, ["--out_dir", str(out), "--results_csv", str(tmp_path / f"{name}_res.csv")]


def test_check_cli_requires_every_model_and_is_idempotent(tmp_path, capsys):
    out, base = _write(*_fixture(), tmp_path)
    assert e2.main(["check", *base]) == e2.GATE_EXIT  # three panel models have no rows
    said = capsys.readouterr().out
    assert "KaLM-mini: no rows: FAIL" in said and "GATES FAILED" in said
    assert e2.main(["check", *base, "--allow_missing"]) == 0
    assert e2.main(["check", *base, "--models", "LaTa,mT5-base,LaBSE"]) == 0
    first = (out / e2.GATE_NAME).read_bytes()
    assert e2.main(["check", *base, "--models", "LaTa,mT5-base,LaBSE"]) == 0
    assert (out / e2.GATE_NAME).read_bytes() == first
    assert pd.read_csv(out / e2.GATE_NAME).ok.all()
    capsys.readouterr()
    # a limit run reports its gates and never fails on them
    frames, res = _fixture(limit=10)
    out, base = _write(frames, res.assign(aucroc=0.0), tmp_path, "smoke")
    assert e2.main(["check", *base]) == 0
    said = capsys.readouterr().out
    assert "limit run" in said and "1a" not in said and "gate 3" in said


def test_summarize_applies_the_frozen_rules_to_the_cells():
    frames, res = _fixture()
    args = [frames[k] for k in (e2.ARMS_NAME, e2.AUDIT_NAME, e2.ABL_NAME, e2.LENGTH_NAME)]
    w = e2.summarize(*args, res).set_index(["m", "layer"])
    assert list(w.index[w.collapsed]) == [("LaTa", 2), ("LaTa", 3), ("mT5-base", 5)]
    x = w.loc[("LaTa", 2)]
    assert x.r1_sif_keepspecial == "rescue" and x.r1_mean_nospecial == "no_rescue"
    assert x.r1_mean_nofreq100 == "no_rescue"
    assert x.r1frac_mean_nofreq100 == pytest.approx(0.30 / 0.38)
    assert x.r3m_pc1 == 30 and x.r3m_pc123 == 30 and x.r3  # 0.93 and 0.91 at m = 30
    assert (x.best_abl, x.best_abl_ranking, x.best_abl_m) == (0.95, "pc1", 100)
    assert x.best_ctl == pytest.approx(0.502) and not x.ctl_r3
    assert x.ctl_pc1_m100_min == pytest.approx(0.50) and x.ctl_pc1_m100_max == pytest.approx(0.504)
    y = w.loc[("LaTa", 3)]
    assert y.r1_sif_keepspecial == "no_sif_gain" and np.isnan(y.r1frac_sif_keepspecial)
    assert y.r1chg_mean_nofreq100 == pytest.approx(0.03) and y.r3m_pc1 == -1 and not y.r3
    assert w.loc[("mT5-base", 5)].r1_mean_nofreq100 == "no_sif_gain"
    # the audit cells and the rules on them
    audit = frames[e2.AUDIT_NAME]
    cell = audit[(audit.model == LATA) & (audit.layer == 2) & (audit.direction == "pc1")
                 & (audit.split == "test")].iloc[0]
    assert x.ev_pc1 == cell.ev_tokenmix and x.r2 == e2.r2_token_mix(cell.ev_tokenmix) and x.r2
    assert x.sh_frequent_pc1 == cell.share_frequent and x.top_group == "frequent"
    rand = audit[(audit.model == LATA) & (audit.layer == 2) & (audit.kind == "random")
                 & (audit.split == "test")]
    assert x.rand_ev_mean == pytest.approx(rand.ev_tokenmix.mean())
    assert x.rand_ev_max == rand.ev_tokenmix.max()
    pairs = audit[(audit.model == LATA) & (audit.layer == 2) & (audit.dim == 2)
                  & (audit.kind == "random_subspace") & (audit.split == "test")]
    assert len(pairs) == 10 and x.rand2_ev_mean == pytest.approx(pairs.ev_tokenmix.mean())
    joint = audit[(audit.model == LATA) & (audit.layer == 2) & (audit.direction == "pcs2_3")
                  & (audit.split == "test")].iloc[0]
    assert x.ev_pcs2_3 == joint.ev_tokenmix and x.eveq_pcs2_3 == joint.ev_tokenmix_eqw
    assert x.sheq_other_pcs2_3 == joint.share_other_eqw
    var = {d: audit[(audit.model == LATA) & (audit.layer == 2) & (audit.direction == d)
                    & (audit.split == "train")].var_s.iloc[0] for d in ("pc2", "pc3")}
    assert x.var_ratio_pc2_pc3 == pytest.approx(var["pc2"] / var["pc3"]) and x.var_ratio_pc2_pc3 >= 1
    ln = frames[e2.LENGTH_NAME]
    t = ln[(ln.model == LATA) & (ln.split == "test")].iloc[0]
    assert x.len_same == t.mean_dlog_same
    assert x.r4 == e2.r4_length(cell.rho_logn, t.mean_dlog_same, t.mean_dlog_diff)
    # collapsed is read from the published baseline, not from the re-derived mean arm
    pub = _set(res, lambda r: (r.model == LATA) & (r.layer == 1) & (r.method == "baseline"),
               "aucroc", 0.69)
    assert e2.summarize(*args, pub).set_index(["m", "layer"]).loc[("LaTa", 1)].collapsed
    own = e2.summarize(*args, None).set_index(["m", "layer"])
    assert not own.loc[("LaTa", 1)].collapsed and own.loc[("LaTa", 2)].collapsed
    assert not own.base_published.any() and e2.worst_layer(w.reset_index(), "LaTa").layer == 2


def test_table_renders_from_the_fixture_and_omits_absent_models(tmp_path):
    frames, res = _fixture()
    w = e2.summarize(*[frames[k] for k in (e2.ARMS_NAME, e2.AUDIT_NAME, e2.ABL_NAME,
                                           e2.LENGTH_NAME)], res)
    out = tmp_path / "t.tex"
    gates = _gt(frames, res)
    assert e2.write_table(w, out, gates) == ["PhilTa", "Qwen3-0.6B", "KaLM-mini"]
    tex = out.read_text()
    assert tex.startswith(asw.HEADER + "\n") and r"\label{tab:e2_token_audit}" in tex
    assert r"\begin{table*}" in tex and r"\end{table*}" in tex
    assert chr(0x2014) not in tex and "---" not in tex and "PhilTa" not in tex
    rows = [ln for ln in tex.splitlines() if ln.startswith(("LaTa &", "mT5-base &", "LaBSE &"))]
    assert len(rows) == 3 and all(r.count("&") == 14 for r in rows)
    header = next(ln for ln in tex.splitlines() if ln.startswith("Model &"))
    assert header.count("&") == 14
    cells = [c.strip() for c in rows[0].rstrip("\\ ").split("&")]
    x = w[(w.m == "LaTa") & (w.layer == 2)].iloc[0]
    assert cells[1] == "2"  # the worst baseline layer
    assert cells[2:7] == ["0.500", "0.510", "0.860", "0.880", "0.800"]
    assert cells[7:9] == ["0.950", "0.502"]
    assert cells[9:] == [f"{x.ev_pc1:.2f}", f"{x.rand_ev_mean:.2f}", f"{x.sh_special_pc1:.2f}",
                         f"{x.sh_frequent_pc1:.2f}", f"{x.sh_other_pc1:.2f}", f"{x.rho_pc1:.2f}"]
    cap = next(ln for ln in tex.splitlines() if ln.startswith(r"\caption{"))
    assert cap.endswith("}") and "--" not in cap and chr(0x2014) not in cap
    for phrase in ("worst baseline layer", "special tokens", "100 most frequent",
                   "five random sets", "20 random directions", "training passages only",
                   "recomputed in this experiment's forward pass with training-only token "
                   "frequencies (SIF; it reproduces the published SIF cells)"):
        assert phrase in cap
    assert "as published" not in cap
    # a negative cell has a math minus sign; the LaBSE shares of -0.01 are the fixture's
    labse = [c.strip() for c in rows[2].rstrip("\\ ").split("&")]
    y = w[w.m == "LaBSE"].iloc[0]
    assert y.sh_special_pc1 < 0 and labse[11] == f"$-${abs(y.sh_special_pc1):.2f}"
    assert "& -" not in tex
    # the caption's statement about the published cells is read from gate 2a
    worse = _set(res, lambda r: (r.model == LATA) & (r.layer == 3) & (r.method == "sif_only"),
                 "aucroc", 0.68 - 0.0150072)
    worse = _set(worse, lambda r: (r.model == MT5) & (r.method == "sif_only"), "aucroc", 0.670)
    e2.write_table(w, out, _gt(frames, worse))
    cap = next(ln for ln in out.read_text().splitlines() if ln.startswith(r"\caption{"))
    assert ("(SIF; its AUROC differs from the published SIF cells by up to 0.015 over all "
            "models and layers)") in cap and "reproduces" not in cap
    assert e2.sif_reference_gap(_gt(frames, worse)) == pytest.approx(0.0150072, abs=1e-9)
    # without gates (no published cells) the caption makes no claim about them
    e2.write_table(w, out)
    assert "published SIF cells" not in out.read_text()
    assert np.isnan(e2.sif_reference_gap(None))


def test_table_cells_use_a_math_minus_and_no_negative_zero():
    assert e2.tex_num(-0.38, 2) == "$-$0.38" and e2.tex_num(0.95, 2) == "0.95"
    assert e2.tex_num(-0.004, 2) == "0.00" and e2.tex_num(-0.0004, 3) == "0.000"
    assert e2.tex_num(-0.005001, 2) == "$-$0.01" and e2.tex_num(0.0, 3) == "0.000"
    assert e2.tex_num(-1.5, 3) == "$-$1.500" and e2.tex_num(0.4996, 3) == "0.500"
    assert e2.tex_num(float("nan"), 2) == "--" and e2.tex_num(None, 3) == "--"


def test_r1_under_the_published_sif_reference():
    frames, res = _fixture()
    args = [frames[k] for k in (e2.ARMS_NAME, e2.AUDIT_NAME, e2.ABL_NAME, e2.LENGTH_NAME)]

    def compare(published):
        w = e2.summarize(*args, published)
        return w, e2.r1_reference_comparison(w[w.collapsed])

    # the published cells equal the sif arm: nothing differs
    w, cmp = compare(res)
    assert (w.sif_minus_pub == 0).all() and cmp["n"] == cmp["n_compared"] == 9
    assert len(cmp["differ"]) == 0 and cmp["n_both"] == 3 and cmp["max_dfrac"] == 0
    assert list(w.r1p_sif_keepspecial) == list(w.r1_sif_keepspecial)
    # LaTa 2: published 0.89 against the arm's 0.88. Shares move, verdicts do not
    near = _set(res, lambda r: (r.model == LATA) & (r.layer == 2) & (r.method == "sif_only"),
                "aucroc", 0.89)
    w, cmp = compare(near)
    x = w[(w.m == "LaTa") & (w.layer == 2)].iloc[0]
    assert x.pub_sif == 0.89 and x.sif_minus_pub == pytest.approx(-0.01)
    assert x.sif_gain_pub == pytest.approx(0.39) and x.sif_gain == pytest.approx(0.38)
    assert (x.r1p_mean_nospecial, x.r1p_sif_keepspecial, x.r1p_mean_nofreq100) == (
        "no_rescue", "rescue", "no_rescue")
    assert x.r1pfrac_sif_keepspecial == pytest.approx(0.36 / 0.39)
    assert x.r1_sif_keepspecial == "rescue" and x.r1frac_sif_keepspecial == pytest.approx(0.36 / 0.38)
    assert len(cmp["differ"]) == 0 and cmp["n_both"] == 3
    assert cmp["max_dfrac"] == pytest.approx(0.36 / 0.38 - 0.36 / 0.39)
    assert cmp["max_cell"] == "LaTa L2 `sif_keepspecial`"
    # LaTa 2: published 0.70. mean_nofreq100 recovers 0.30 of a gain of 0.20: now a rescue
    far = _set(res, lambda r: (r.model == LATA) & (r.layer == 2) & (r.method == "sif_only"),
               "aucroc", 0.70)
    w, cmp = compare(far)
    d = cmp["differ"]
    assert len(d) == 1 and (d.iloc[0].m, d.iloc[0].layer, d.iloc[0].arm) == (
        "LaTa", 2, "mean_nofreq100")
    assert (d.iloc[0].own, d.iloc[0].pub) == ("no_rescue", "rescue")
    assert cmp["max_dfrac"] == pytest.approx(0.36 / 0.20 - 0.36 / 0.38)
    # mT5-base 5: published 0.71 lifts the SIF gain over the floor, so R1 becomes evaluable
    lift = _set(res, lambda r: (r.model == MT5) & (r.method == "sif_only"), "aucroc", 0.71)
    w, cmp = compare(lift)
    assert sorted(cmp["differ"].arm) == sorted(e2.R1_ARMS)
    assert set(cmp["differ"].own) == {"no_sif_gain"} and cmp["n_both"] == 3
    assert w[w.m == "mT5-base"].iloc[0].r1p_mean_nofreq100 == "rescue"  # 0.046 / 0.056
    # a missing published cell is not compared, and no published CSV compares nothing
    w, cmp = compare(res[~((res.model == MT5) & (res.method == "sif_only"))])
    assert cmp["n"] == 9 and cmp["n_compared"] == 6 and len(cmp["differ"]) == 0
    assert set(cmp["cells"][cmp["cells"].m == "mT5-base"].pub) == {"undefined"}
    w, cmp = compare(None)
    assert cmp["n_compared"] == 0 and np.isnan(cmp["max_dfrac"]) and cmp["max_cell"] == ""
    assert e2.r1_reference_comparison(w[w.layer < 0])["n"] == 0


def test_render_cli_writes_table_and_facts_from_the_csvs_alone(tmp_path, capsys):
    out, base = _write(*_fixture(), tmp_path)
    tab = tmp_path / "tables"
    argv = ["render", *base, "--tab_dir", str(tab), "--models", "LaTa,mT5-base,LaBSE"]
    assert e2.main(argv) == 0
    said = capsys.readouterr().out
    assert "omitting PhilTa" in said and "worst baseline layer LaTa: 2" in said
    facts = (out / e2.FACTS_NAME).read_text()
    table = (tab / e2.TABLE_NAME).read_text()
    assert chr(0x2014) not in facts
    for section in ("## Frozen decision rules", "## Definitions", "## 0. Coverage and gates",
                    "## 3. R1: pooling control", "## 4. R2:", "## 5. Which token group",
                    "## 6. R3: token ablation", "## 7. R4: passage length",
                    "## 8. Expectations", "## 9. Carrier tokens", "## 10. Contrast",
                    "## 11. Verdicts in one place", "## 12. All layers"):
        assert section in facts, section
    assert "collapsed layers (baseline AUROC < 0.70): 3: LaTa 2 (2, 3), mT5-base 1 (5)" in facts
    assert "ABSENT from the CSVs" in facts and "SMOKE RUN" not in facts
    assert "gate 1a mean arm AUROC vs published baseline, LaTa: 3 cells" in facts
    assert "GATE 2 FAILED" not in facts and "A gate FAILED" not in facts
    # R1 per arm over the three collapsed layers, and LaTa alone
    r1 = facts.split("## 3. R1: pooling control")[1].split("## 4.")[0]
    assert ("- `sif_keepspecial`: rescues 1/3; does not rescue 0/3; no SIF gain to recover 2/3"
            in r1)
    assert "- `mean_nospecial`: rescues 0/2; does not rescue 1/2; no SIF gain to recover 1/2" in r1
    assert "below it at LaTa 3, mT5-base 5" in r1
    # verdict lines: counts over collapsed layers, the expectations, the vacuous mT5 case
    verdicts = facts.split("## 11. Verdicts in one place")[1].split("## 12.")[0]
    assert "R3 token ablation restores collapsed layers (AUROC >= 0.90, m <= 100): 1/3" in verdicts
    assert "LaTa: sif_keepspecial rescues collapsed layers (R1): NOT MET (1/2)" in verdicts
    assert ("LaTa: mean_nospecial does not rescue collapsed layers (R1): NOT MET (1/2 not "
            "rescued; 1 with no SIF gain to recover)") in verdicts
    assert ("mT5-base: no arm rescues collapsed layers (R1): NOT EVALUABLE BY R1 (SIF gain "
            "below 0.05 at all 1 layers") in verdicts
    assert "R2 token mix carries PC1 (test EV >= 0.5) at collapsed layers: 3/3" in verdicts
    assert "LaTa: frequent tokens have the largest PC1 share at collapsed layers: MET (2/2)" in verdicts
    # the subspace readouts sit next to the per-PC numbers and carry no verdict
    r2 = facts.split("## 4. R2:")[1].split("## 5.")[0]
    assert "span(PC2, PC3): test EV variance-weighted" in r2 and "equal-weight" in r2
    assert "train variance ratio var(PC2) / var(PC3)" in r2 and "random 3-D subspaces" in r2
    assert "span" not in verdicts
    # carriers of the worst T5 layers, with decoded strings
    car = facts.split("## 9. Carrier tokens")[1].split("## 10.")[0]
    assert "- LaTa L2:" in car and "- mT5-base L5:" in car and "LaBSE" not in car
    assert '`▁t3` " t3" [frequent]' in car
    assert "### LaBSE: token audit of PC1 (test)" in facts and "| 2* |" in facts
    # deterministic, and rendered from the CSVs alone
    assert e2.main(argv) == 0
    assert (out / e2.FACTS_NAME).read_text() == facts
    assert (tab / e2.TABLE_NAME).read_text() == table
    capsys.readouterr()


def _render(frames, res, tmp_path, name):
    out, base = _write(frames, res, tmp_path, name)
    assert e2.main(["render", *base, "--tab_dir", str(tmp_path / f"{name}_tab"),
                    "--allow_missing"]) == 0
    return (out / e2.FACTS_NAME).read_text(), (tmp_path / f"{name}_tab" / e2.TABLE_NAME).read_text()


def test_facts_report_r1_under_both_references_when_gate_2a_fails(tmp_path, capsys):
    frames, res = _fixture()
    near = _set(res, lambda r: (r.model == LATA) & (r.layer == 2) & (r.method == "sif_only"),
                "aucroc", 0.89)
    facts, table = _render(frames, near, tmp_path, "near")
    assert "BLOCKED" not in facts and "not to be quoted" not in facts
    assert "DO NOT QUOTE" not in facts and "GATE 2 FAILED" not in facts
    head = facts.split("## Frozen decision rules")[0]
    assert "Changed after the results were read: one consequence, in `render` only." in head
    assert "Gate 2a failed in this run (section 0)." in head
    statement = (
        "**Deviation (gate 2a).** Gate 2a failed: the `sif` arm does not reproduce the "
        "published `sif_only` cells. Largest AUROC difference per model: LaTa 1.00e-02. The "
        "`sif` arm is the repo CLI's SIF pooling on the tracked split with train-only token "
        "probabilities, and it differs from the local re-extraction (`hidden_sif_tokempty`) "
        "by at most 1.00e-07 in relative L2 at the 3 models that have one (gate 2b). R1 is "
        "therefore reported against the job's own `sif` arm and, under \"R1 under the "
        "published SIF reference\" in section 3, against the published cells. Changed after "
        "the results were read: the pre-registered consequence of a gate 2 failure (block "
        "the pooling-control conclusions) was replaced by reporting under both references. "
        "The R1 verdicts are the same under both references at all 9 (collapsed layer, arm) "
        "cells.")
    # in section 0, at the head of section 3 and in section 11
    assert facts.count(statement) == 3
    assert "- " + statement in facts.split("## 0. Coverage")[1].split("## 1.")[0]
    sec3 = facts.split("## 3. R1: pooling control")[1].split("## 4.")[0]
    assert sec3.lstrip().startswith(statement)
    assert "- " + statement in facts.split("## 11. Verdicts")[1].split("## 12.")[0]
    # gate 2a itself still fails in the gate lines
    assert ("gate 2a sif arm AUROC vs published sif_only, LaTa: 3 cells, max diff 1.00e-02 "
            "(tolerance 1e-06): FAIL") in facts
    sub = sec3.split("### R1 under the published SIF reference")[1]
    assert ("  - LaTa: collapsed layers (n = 2) -5.00e-03 (-1.00e-02 to +0.00e+00); all layers "
            "(n = 3) +0.00e+00 (-1.00e-02 to +0.00e+00)") in sub
    assert "  - LaBSE: all layers (n = 1) +0.00e+00 (+0.00e+00 to +0.00e+00)" in sub
    assert ("at least 0.05 at 1/3 collapsed layers (job's `sif` arm: 1/3)") in sub
    assert ("  - all collapsed layers (n = 3): `mean_nospecial` 0 / 1 / 2, then 0 / 1 / 2; "
            "`sif_keepspecial` 1 / 0 / 2, then 1 / 0 / 2; `mean_nofreq100` 0 / 1 / 2, then "
            "0 / 1 / 2") in sub
    assert "  - mT5-base (n = 1): `mean_nospecial` 0 / 0 / 1, then 0 / 0 / 1" in sub
    assert "- verdicts that differ between the two references: 0 of 9 (collapsed layer, arm) cells\n" in sub
    assert ("- largest absolute change of the recovered share of the SIF gain: 0.024 (LaTa L2 "
            "`sif_keepspecial`), over the 3 cells where the share is evaluated under both "
            "references") in sub
    sec8 = facts.split("## 8. Expectations")[1].split("## 9.")[0]
    assert ("- LaTa: `sif_keepspecial` rescues: **NOT MET (1/2)**; with the published "
            "`sif_only` cell as AUROC_sif: NOT MET (1/2)") in sec8
    assert ("- mT5-base: no arm rescues: **NOT EVALUABLE BY R1 (SIF gain below 0.05 at all 1 "
            "layers, so there is no SIF gain to recover)**; with the published `sif_only` cell "
            "as AUROC_sif: NOT EVALUABLE BY R1") in sec8
    verdicts = facts.split("## 11. Verdicts in one place")[1].split("## 12.")[0]
    assert ("R1 verdict is the same with the published sif_only cell as AUROC_sif "
            "((collapsed layer, arm) cells): 9/9") in verdicts
    assert "by up to 0.010 over all models and layers" in table
    capsys.readouterr()


def test_facts_are_loud_when_the_two_references_disagree(tmp_path, capsys):
    frames, res = _fixture()
    far = _set(res, lambda r: (r.model == LATA) & (r.layer == 2) & (r.method == "sif_only"),
               "aucroc", 0.70)
    far = far[~((far.model == MT5) & (far.method == "sif_only"))]
    facts, _ = _render(frames, far, tmp_path, "far")
    loud = ("**THE R1 VERDICTS DIFFER BETWEEN THE TWO REFERENCES AT 1 OF 6 (COLLAPSED LAYER, "
            "ARM) CELLS. QUOTE NO R1 COUNT WITHOUT NAMING ITS REFERENCE.**")
    assert facts.count(loud) == 3 and "are the same under both references" not in facts
    assert "LaTa 1.80e-01, mT5-base no published cell" in facts
    assert ("The published cell is missing at 3 of the 9 (collapsed layer, arm) cells, which "
            "are not compared.") in facts
    sub = facts.split("### R1 under the published SIF reference")[1].split("## 4.")[0]
    assert ("- verdicts that differ between the two references: 1 of 6 (collapsed layer, arm) "
            "cells: LaTa L2 `mean_nofreq100` no_rescue -> rescue") in sub
    assert "published cell missing at 3 of the 9 (collapsed layer, arm) cells" in sub
    assert "mT5-base (n = 1)" not in sub and "  - mT5-base: all layers (n = 0) n/a" in sub
    sec8 = facts.split("## 8. Expectations")[1].split("## 9.")[0]
    assert ("- LaTa: `sif_keepspecial` rescues: **NOT MET (1/2)**; with the published "
            "`sif_only` cell as AUROC_sif: NOT MET (1/2)") in sec8
    assert "- mT5-base: no arm rescues: **NOT EVALUABLE BY R1" in sec8
    assert "mT5-base: no arm rescues: **NOT EVALUABLE BY R1 (SIF gain below 0.05 at all 1 layers, so there is no SIF gain to recover)**. Raw" in sec8
    assert "(collapsed layer, arm) cells): 5/6" in facts.split("## 11. Verdicts")[1]
    capsys.readouterr()


def test_facts_without_a_gate_2_failure_and_smoke_runs(tmp_path, capsys):
    facts, table = _render(*_fixture(), tmp_path, "ok")
    assert "Deviation (gate 2a)" not in facts and "Gate 2a did not fail in this run." in facts
    sub = facts.split("### R1 under the published SIF reference")[1].split("## 4.")[0]
    assert "- verdicts that differ between the two references: 0 of 9" in sub
    assert "largest absolute change of the recovered share of the SIF gain: 0.000" in sub
    assert "it reproduces the published SIF cells" in table
    # a smoke run says so at the top and skips the AUROC gates
    facts, table = _render(*_fixture(limit=10), tmp_path, "smoke")
    assert "**SMOKE RUN (--limit 10)" in facts.split("## Frozen")[0]
    assert "gate 1a" not in facts and "Deviation (gate 2a)" not in facts
    assert "published SIF cells" not in table
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# the whole audit on a fake encoder and a fake cache (torch, tiny tensors)
# --------------------------------------------------------------------------- #

def _fake_world(tmp_path, n_tr=36, n_te=32, vocab=300, d=40, n_layers=2, seed=0):
    """A split CSV, a fake LaTa cache in a different row order, and an Encoder whose hidden
    states are fixed toy tensors. At layer 2 frequent tokens and the EOS carry two large
    directions, so that layer is "collapsed"."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    n = n_tr + n_te
    folders = rng.integers(0, 14, size=n)
    zipf = 1.0 / np.arange(1, vocab - 2)
    topic = rng.dirichlet(np.full(vocab - 3, 0.05), size=14)
    emb = rng.normal(size=(n_layers + 1, vocab, d))
    ids, states = [], []
    for p in range(n):
        size = int(rng.integers(4, 26))
        t = rng.choice(np.arange(3, vocab), size=size,
                       p=0.5 * zipf / zipf.sum() + 0.5 * topic[folders[p]])
        t[rng.random(size) < 0.05] = 2  # a token the filter drops
        if p == 0:
            t[:] = 2  # nothing but the EOS is kept in this passage
        t[-1] = EOS
        h = emb[:, t, :] + 0.3 * rng.normal(size=(n_layers + 1, size, d))
        h[2, :, 0] += 25.0 * (t < 12)
        h[2, :, 1] += 40.0 * (t == EOS)
        ids.append(t)
        states.append(h.astype(np.float32))
    keep = np.ones(vocab, dtype=np.float32)
    keep[2] = 0.0
    special = [0, EOS]
    split = np.array(["train"] * n_tr + ["test"] * n_te)[rng.permutation(n)]
    names = [f"f{i:03d}.txt" for i in range(n)]
    test_counts = pd.Series(folders[split == "test"]).value_counts()
    frame = pd.DataFrame({
        "filename": names, "folder_id": [f"d{f}" for f in folders], "split": split,
        "has_test_partner": [bool(s == "test" and test_counts[f] > 1)
                             for s, f in zip(split, folders)],
        "path": [f"data/x/{nm}" for nm in names]})
    frame.to_csv(tmp_path / "split.csv", index=False)

    def probs(rows):
        t = np.concatenate([ids[p] for p in rows])
        t = t[(keep[t] > 0) & ~np.isin(t, special)]
        types, counts = np.unique(t, return_counts=True)
        return {int(a): float(b) / len(t) for a, b in zip(types, counts)}

    token_probs = probs(np.flatnonzero(split == "train"))
    look = e2.arm_weight_lookups(vocab, token_probs, special, keep)
    order = rng.permutation(n)
    for sub, arm, suffix in ((e2.MEAN_SUBDIR, "mean", ""), (e2.SIF_SUBDIR, "sif", "_sif")):
        run = tmp_path / "bases" / "phase9_bases" / asw.slug(LATA) / sub
        run.mkdir(parents=True)
        pd.DataFrame({"path": [f"data/x/{names[i]}" for i in order]}).to_csv(run / "meta.csv",
                                                                           index=False)
        for layer in range(1, n_layers + 1):
            vec = []
            for p in order:
                w = look[arm][ids[p]].astype(np.float32)
                vec.append((states[p][layer] * w[:, None]).sum(axis=0) / max(w.sum(), 1.0))
            np.save(run / f"hidden_layer{layer}_embeddings{suffix}.npy",
                    np.array(vec, dtype=np.float32))

    def batches(rows_in_order=order):
        for start in range(0, len(rows_in_order), e2.BATCH_SIZE):
            rows = rows_in_order[start:start + e2.BATCH_SIZE]
            width = max(len(ids[p]) for p in rows)
            tok = np.zeros((len(rows), width), dtype=np.int64)
            att = np.zeros((len(rows), width), dtype=np.int64)
            hid = np.full((n_layers + 1, len(rows), width, d), 99.0, dtype=np.float32)
            for b, p in enumerate(rows):
                tok[b, :len(ids[p])] = ids[p]
                att[b, :len(ids[p])] = 1
                hid[:, b, :len(ids[p])] = states[p]
            yield rows, torch.tensor(tok), torch.tensor(att), tuple(torch.tensor(x) for x in hid)

    def encoder(ctx=None, reference_pool=None):
        """The encoder for a run; a --limit run sees its passages only, as open_encoder."""
        if ctx is None or not ctx.limit:
            return e2.Encoder(batches=batches, keep_lookup=keep, special_ids=special,
                              token_probs=token_probs,
                              describe=lambda i: (f"▁t{i}", f" t{i}"),
                              reference_pool=reference_pool)
        local = {int(g): i for i, g in enumerate(ctx.sel)}
        sub_order = [p for p in order if p in local]

        def limited():
            for rows, tok, att, hid in batches(np.array(sub_order)):
                yield np.array([local[int(p)] for p in rows]), tok, att, hid

        return e2.Encoder(batches=limited, keep_lookup=keep, special_ids=special,
                          token_probs=probs(ctx.sel[ctx.tr]),
                          describe=lambda i: (f"▁t{i}", f" t{i}"))

    return {"split_csv": tmp_path / "split.csv", "bases": tmp_path / "bases", "ids": ids,
            "states": states, "split": split, "folders": folders, "keep": keep,
            "encoder": encoder, "frame": frame}


def _numpy_pool(world, layer, drop=()):
    """Mean pooling of the fake states over kept tokens, minus dropped types, in numpy."""
    out = []
    for t, h in zip(world["ids"], world["states"]):
        kept = world["keep"][t] > 0
        use = kept & ~np.isin(t, list(drop))
        if not use.any():
            use = kept
        out.append(h[layer][use].astype(np.float64).mean(axis=0))
    return np.array(out)


def _auroc(world, vectors, split="test"):
    """Task A AUROC of numpy-pooled vectors, to compare with the audit's cells. The audit
    pools in float32 on torch, so a near-tied pair of cosines may rank the other way."""
    rows = world["split"] == split
    return asw.pair_auroc(vectors[rows].astype(np.float32),
                          upper_triangle_labels(world["folders"][rows]))


def test_audit_of_a_fake_encoder_reproduces_numpy_pooling(tmp_path):
    world = _fake_world(tmp_path)
    pytest.importorskip("transformers")
    import extract_hidden_cli

    ctx = e2.build_context(world["split_csv"], tmp_path / "out", 0)
    saved = dict(asw._CTX)
    try:
        asw._init(str(ctx.split_path))
        frames = e2.audit_model(LATA, world["encoder"](reference_pool=extract_hidden_cli.pool_hidden),
                                ctx, world["bases"])
    finally:
        asw._CTX.clear()
        asw._CTX.update(saved)
    arms, audit = frames[e2.ARMS_NAME], frames[e2.AUDIT_NAME]
    carriers, abl, length = frames[e2.CARRIER_NAME], frames[e2.ABL_NAME], frames[e2.LENGTH_NAME]
    assert len(arms) == 2 * len(e2.ARMS) and list(arms.arm[:5]) == list(e2.ARMS)
    assert len(audit) == 2 * 2 * (len(e2.DIRECTIONS) + len(e2.SUBSPACES))
    assert len(carriers) == 2 * e2.N_PC * e2.TOP_CARRIERS
    assert len(abl) == 2 * len(e2.RANKINGS) * len(e2.ABL_MS) * (1 + e2.CONTROL_DRAWS)
    assert list(length.split) == ["train", "test"] and (arms.limit == 0).all()
    # the two reference arms are the CLI's pooling and the cache
    ref = arms[arms.arm.isin(["mean", "sif"])]
    assert (ref.cli_pool_max_abs_diff == 0).all()
    assert ref.vec_max_rel_diff.max() < 1e-5 and arms[arms.arm == "mean_nospecial"].vec_max_rel_diff.isna().all()
    for layer in (1, 2):
        a = arms[arms.layer == layer].set_index("arm")
        assert a.loc["mean", "aucroc"] == pytest.approx(
            _auroc(world, _numpy_pool(world, layer)), abs=5e-4)
        assert a.loc["mean", "train_aucroc"] == pytest.approx(
            _auroc(world, _numpy_pool(world, layer), "train"), abs=5e-4)
        assert a.loc["mean_nospecial", "aucroc"] == pytest.approx(
            _auroc(world, _numpy_pool(world, layer, drop=[EOS])), abs=5e-4)
    a = arms[arms.layer == 2].set_index("arm")
    # passage 0 keeps only its EOS: empty without special tokens, so it falls back to mean
    assert a.loc["mean_nospecial", "n_no_token"] == 1 and a.loc["mean_nospecial", "n_fallback"] == 1
    assert a.loc["sif", "n_no_token"] == 1 and a.loc["sif", "n_fallback"] == 0
    assert a.loc["mean", "n_no_token"] == 0 and a.loc["sif_keepspecial", "n_no_token"] == 0
    assert a.loc["sif", "special_mass"] == 0 and a.loc["mean_nospecial", "special_mass"] == 0
    assert a.loc["sif_keepspecial", "special_mass"] > a.loc["mean", "special_mass"] > 0
    assert a.loc["mean_nofreq100", "frequent_mass"] == 0 < a.loc["mean", "frequent_mass"]
    assert set(asw.KEEP) <= set(arms.columns) and arms.eff_rank_train.notna().all()
    # gates 3 and 4 on the real forward path
    single = audit[audit.dim == 1]
    assert single.identity_max_rel_diff.max() < e2.GATE_TOL_IDENTITY
    assert audit.share_sum_err.abs().max() < e2.GATE_TOL_SHARE_SUM
    assert single.cache_score_max_abs_diff.max() < 1e-3
    # layer 2: the planted frequent-token direction is PC1 or PC2, and its carriers are
    # the frequent token ids (3..11); the token mix explains it
    top = carriers[(carriers.layer == 2) & (carriers["rank"] <= 3)
                   & carriers.direction.isin(["pc1", "pc2"])]
    assert (top.token_id < 12).mean() > 0.6
    pc = audit[(audit.layer == 2) & audit.direction.isin(["pc1", "pc2"]) & (audit.split == "test")]
    assert pc.ev_tokenmix.max() > 0.6
    n = np.array([int((world["keep"][t] > 0).sum()) for t in world["ids"]])
    assert length.set_index("split").loc["test", "mean_n"] == pytest.approx(
        n[world["split"] == "test"].mean())
    # token ablation: the m = 1 arm of the pc1 ranking drops the top PC1 carrier
    for layer in (1, 2):
        first = carriers[(carriers.layer == layer) & (carriers.direction == "pc1")
                         & (carriers["rank"] == 1)].token_id.iloc[0]
        t = abl[(abl.layer == layer) & (abl.ranking == "pc1") & (abl.kind == "carrier")]
        assert list(t.m) == list(e2.ABL_MS) and t.types.iloc[0] == str(first)
        assert t.aucroc.iloc[0] == pytest.approx(
            _auroc(world, _numpy_pool(world, layer, drop=[first])), abs=5e-4)
        ten = [int(v) for v in t.types.iloc[2].split(";")]
        assert len(ten) == 10 and ten[0] == first
        assert t.aucroc.iloc[2] == pytest.approx(
            _auroc(world, _numpy_pool(world, layer, drop=ten)), abs=5e-4)
        assert t.dropped_mass_test.is_monotonic_increasing and t.eff_rank_train.notna().all()
        ctl = abl[(abl.layer == layer) & (abl.ranking == "pc1") & (abl.kind == "control")]
        assert sorted(ctl.draw.unique()) == list(range(e2.CONTROL_DRAWS))
        assert ctl.aucroc.notna().all() and ctl.eff_rank_train.isna().all()  # AUROC only
        used = {int(v) for s in ctl.types for v in s.split(";") if v}
        both = abl[(abl.layer == layer) & (abl.kind == "carrier") & (abl.m == 100)]
        assert not used & {int(v) for s in both.types for v in s.split(";")}
        c1 = ctl[(ctl.draw == 0) & (ctl.m == 1)].iloc[0]
        assert c1.aucroc == pytest.approx(
            _auroc(world, _numpy_pool(world, layer, drop=[int(c1.types)])), abs=5e-4)


def test_audit_cli_is_deterministic_merges_by_model_and_gates(tmp_path, monkeypatch, capsys):
    world = _fake_world(tmp_path)
    monkeypatch.setattr(e2, "open_encoder", lambda mid, ctx, bases: world["encoder"](ctx))
    out = tmp_path / "e2"
    run = ["audit", "--models", "LaTa", "--bases_root", str(world["bases"]), "--split_csv",
           str(world["split_csv"]), "--out_dir", str(out), "--workers", "1",
           "--e1_coords_csv", str(tmp_path / "none.csv")]
    names = (e2.ARMS_NAME, e2.AUDIT_NAME, e2.CARRIER_NAME, e2.ABL_NAME, e2.LENGTH_NAME)
    saved = dict(asw._CTX)
    try:
        assert e2.main(run + ["--results_csv", str(tmp_path / "none.csv")]) == 0
        first = {n: (out / n).read_bytes() for n in names}
        # another model's rows, and published cells that match what was just written
        other = {n: e2.read_csv(out / n).assign(model=LABSE) for n in names}
        for n in names:
            e2.merge_write(out / n, other[n])
        mixed = {n: (out / n).read_bytes() for n in names}
        arms = e2.read_csv(out / e2.ARMS_NAME)
        res = pd.concat([pd.DataFrame({
            "model": t.model, "repr": "hidden", "pooling": pooling, "layer": t.layer,
            "method": method, "aucroc": t.aucroc})
            for arm, pooling, method in (("mean", "mean", "baseline"), ("sif", "sif", "sif_only"))
            for t in [arms[arms.arm == arm]]])
        res.to_csv(tmp_path / "res.csv", index=False)
        refs = ["--results_csv", str(tmp_path / "res.csv")]
        # the rerun replaces LaTa's rows with identical ones and keeps LaBSE's
        assert e2.main(run + refs + ["--check"]) == 0
        assert {n: (out / n).read_bytes() for n in names} == mixed
        assert mixed[e2.ARMS_NAME].startswith(first[e2.ARMS_NAME])
        gates = (out / e2.GATE_NAME).read_bytes()
        assert e2.main(["check", "--out_dir", str(out), "--models", "LaTa", *refs]) == 0
        assert (out / e2.GATE_NAME).read_bytes() == gates
        g = pd.read_csv(out / e2.GATE_NAME)
        assert len(g) == 12 and g.ok.all() and (g[g.gate.str[1] == "a"].max_diff == 0).all()
        # a published cell that disagrees: exit 3, CSVs still written
        res.assign(aucroc=res.aucroc + 1e-4).to_csv(tmp_path / "bad.csv", index=False)
        assert e2.main(run + ["--results_csv", str(tmp_path / "bad.csv"), "--check"]) == e2.GATE_EXIT
        assert "GATES FAILED" in capsys.readouterr().out
        assert (out / e2.ARMS_NAME).read_bytes() == mixed[e2.ARMS_NAME]
        assert e2.main(["render", "--out_dir", str(out), "--tab_dir", str(tmp_path / "tab"),
                        "--models", "LaTa,LaBSE", *refs]) == 0
        assert "## 12. All layers" in (out / e2.FACTS_NAME).read_text()
        # a smoke run goes to its own directory and never fails on gates
        smoke = tmp_path / "smoke"
        limit = ["audit", "--models", "LaTa", "--bases_root", str(world["bases"]),
                 "--split_csv", str(world["split_csv"]), "--out_dir", str(smoke), "--workers",
                 "1", "--limit", "14", "--results_csv", str(tmp_path / "bad.csv"), "--check"]
        assert e2.main(limit) == 0
        said = capsys.readouterr().out
        assert "14 train, 14 test" in said and "limit run" in said
        small = e2.read_csv(smoke / e2.ARMS_NAME)
        assert (small.limit == 14).all() and (small.n_train == 14).all()
        assert small[small.arm == "mean"].vec_max_rel_diff.max() < 1e-5
        assert (smoke / "split_limit14.csv").exists()
        assert (out / e2.ARMS_NAME).read_bytes() == mixed[e2.ARMS_NAME]  # untouched
    finally:
        asw._CTX.clear()
        asw._CTX.update(saved)


def test_default_output_directory_separates_smoke_runs(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(e2, "cmd_audit", lambda args: seen.setdefault("out", args.out_dir) and 0)
    e2.main(["audit", "--limit", "32"])
    assert seen.pop("out") == e2.OUT_DIR / "smoke_limit32"
    e2.main(["audit"])
    assert seen.pop("out") == e2.OUT_DIR
    e2.main(["audit", "--limit", "32", "--out_dir", str(tmp_path)])
    assert seen.pop("out") == tmp_path


def test_missing_cache_is_an_error_before_any_model_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(e2, "open_encoder", lambda *a: pytest.fail("a model was loaded"))
    with pytest.raises(SystemExit) as err:
        e2.main(["audit", "--models", "LaTa", "--bases_root", str(tmp_path), "--out_dir",
                 str(tmp_path / "o")])
    assert "no cached vectors for LaTa" in str(err.value)
    with pytest.raises(SystemExit):
        e2.main(["audit", "--models", "nope", "--out_dir", str(tmp_path / "o")])


def test_forward_order_follows_the_cache_manifest(tmp_path, capsys):
    sp = pd.DataFrame({"filename": ["a.txt", "b.txt", "c.txt"]})
    np.testing.assert_array_equal(e2.forward_order(tmp_path, sp), [0, 1, 2])  # no manifest
    assert "WARNING" in capsys.readouterr().out
    pd.DataFrame({"path": ["x/c.txt", "x/z.txt", "x/a.txt", "x/b.txt"]}).to_csv(
        tmp_path / "meta.csv", index=False)
    np.testing.assert_array_equal(e2.forward_order(tmp_path, sp), [2, 0, 1])
    # a limit run: the cache order, restricted to the selected passages
    np.testing.assert_array_equal(e2.forward_order(tmp_path, sp.iloc[[0, 2]].reset_index()), [1, 0])
    with pytest.raises(SystemExit):
        e2.forward_order(tmp_path, pd.DataFrame({"filename": ["a.txt", "q.txt"]}))


def test_module_imports_without_torch():
    code = ("import sys; sys.path.insert(0, r'%s'); import e2_token_audit; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
            % (REPO_ROOT / "scripts" / "paper" / "reframe"))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
