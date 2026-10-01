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


def test_module_imports_without_torch():
    code = ("import sys; sys.path.insert(0, r'%s'); import e2_token_audit; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
            % (REPO_ROOT / "scripts" / "paper" / "reframe"))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
