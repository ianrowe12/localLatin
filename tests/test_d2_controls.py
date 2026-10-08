"""Spine D2: t5-efficient-base through the P2x2 panel and the projection on the four controls
(scripts/paper/reframe/d2_controls.py).

The synthetic tests pin the read-out rules (collapse criterion, band check at three
decimals), the architecture check, the comparability check, the published-cell gate and the
opt-in model list of p2x2_panel.py; they run in CI. The last group reads the committed outputs
under runs/active/reframe/d2/ and skips when they are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "paper" / "reframe"))

pytest.importorskip("sklearn")
import d2_controls as d2  # noqa: E402
import p2x2_panel as p2  # noqa: E402

OUT = REPO / "runs" / "active" / "reframe" / "d2"


# ----------------------------------------------------------------------------- p2x2 opt-in
def _toy_split() -> pd.DataFrame:
    rows = []
    for d in range(4):
        for s in ("train", "test"):
            for k in range(3):
                rows.append({"folder_id": f"dir{d}", "filename": f"f{d}_{s}_{k}.txt",
                             "split": s, "has_test_partner": s == "test"})
    return pd.DataFrame(rows).sort_values(["folder_id", "filename"]).reset_index(drop=True)


def _cache(run_dir: Path, split: pd.DataFrame, seed: int) -> None:
    run_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(split))
    pd.DataFrame({"filename": split["filename"].values[order]}).to_csv(run_dir / "meta.csv",
                                                                       index=False)
    for layer in (1, 2):
        np.save(run_dir / f"hidden_layer{layer}_embeddings.npy",
                rng.standard_normal((len(split), 8))[order])


def test_extra_models_run_only_when_named(tmp_path, monkeypatch):
    split = _toy_split()
    monkeypatch.setattr(p2, "MODELS", [("LaBERTa", "bowphs/LaBerta", "Enc., raw", "none", "p2x2")])
    monkeypatch.setattr(p2, "EXTRA_MODELS", [("Extra", "o/extra", "T5, raw", "none", "p2x2")])
    monkeypatch.setattr(p2, "LAYERS", (1, 2))
    for i, mid in enumerate(("bowphs/LaBerta", "o/extra")):
        _cache(p2.run_dir("p2x2", mid, tmp_path / "b", tmp_path / "p"), split, i)
    default = p2.score_models(split, tmp_path / "b", tmp_path / "p")
    assert set(default["model"]) == {"LaBERTa"}
    named = p2.score_models(split, tmp_path / "b", tmp_path / "p", names=["Extra", "LaBERTa"])
    assert list(dict.fromkeys(named["model"])) == ["LaBERTa", "Extra"]
    assert list(p2.summarize(named)["model"]) == ["LaBERTa", "Extra"]


def test_t5_efficient_is_an_extra_model_only():
    """The P2x2 tables and its default run must not change."""
    assert "T5-efficient-base" in [m[0] for m in p2.EXTRA_MODELS]
    assert "T5-efficient-base" not in [m[0] for m in p2.MODELS]
    assert len(p2.MODELS) == 10
    ids = {m[0]: m[1] for m in p2.MODELS + p2.EXTRA_MODELS}
    for name, model_id, _, rev in d2.D2_MODELS:   # the D2 caches sit where p2x2 looks
        assert ids[name] == model_id and len(rev) == 40


# ----------------------------------------------------------------------------- config
GIN = """MIXTURE_NAME = 'c4_v220_unsupervised'
dropout_rate = 0.0
num_layers = 12
d_ff = 3072
Bitransformer.shared_embedding = True
encoder/DenseReluDense.activation = 'relu'
run.train_steps = 524288
"""


def _good_row():
    g = d2.parse_gin(GIN)
    return {"feed_forward_proj": "relu", "tie_word_embeddings": True, "lm_head_separate": False,
            "num_layers": 12, "gin_mixture": g["MIXTURE_NAME"],
            "gin_shared_embedding": g["shared_embedding"],
            "gin_encoder_activation": g["encoder_activation"], "card_datasets": "c4"}


def test_parse_gin():
    g = d2.parse_gin(GIN)
    assert g["MIXTURE_NAME"] == "c4_v220_unsupervised"
    assert (g["dropout_rate"], g["num_layers"], g["train_steps"]) == ("0.0", "12", "524288")
    assert (g["shared_embedding"], g["encoder_activation"]) == ("True", "relu")
    assert d2.parse_gin("")["MIXTURE_NAME"] == ""


def test_layout_check_accepts_original_c4_only_and_rejects_the_rest():
    assert d2.layout_problems(_good_row()) == []
    v11 = dict(_good_row(), feed_forward_proj="gated-gelu", tie_word_embeddings=False,
               lm_head_separate=True)
    assert len(d2.layout_problems(v11)) == 3
    mixed = dict(_good_row(), gin_mixture="en_mix", card_datasets="c4,glue")
    assert len(d2.layout_problems(mixed)) == 2
    assert d2.layout_problems(dict(_good_row(), num_layers=24))


# ----------------------------------------------------------------------------- read-outs
def _layers(name, auc, pc1):
    return pd.DataFrame({"model": name, "layer": range(1, 13), "aucroc": auc, "pc1": pc1})


def test_collapse_readout_needs_low_auroc_and_high_share_at_mid_depth():
    auc = [0.9] + [0.55] * 10 + [0.9]
    pc1 = [0.2] + [0.9] * 10 + [0.2]
    c = d2.collapse_readout(_layers("A", auc, pc1), "A")
    assert c["collapses"] and c["n_collapsed_high_pc1"] == 10 and c["collapsed_layers"] == list(range(2, 12))
    # Low AUROC with a low share is not the collapse the spine names.
    c = d2.collapse_readout(_layers("B", auc, [0.5] * 12), "B")
    assert not c["collapses"] and c["n_low_auroc"] == 10
    # The first and last layers are not mid-depth.
    c = d2.collapse_readout(_layers("C", [0.5] + [0.9] * 10 + [0.5], [0.9] * 12), "C")
    assert not c["collapses"] and c["n_low_auroc"] == 0
    # The threshold is strict on AUROC and inclusive on the share.
    c = d2.collapse_readout(_layers("D", [0.9] * 5 + [0.6999] + [0.70] + [0.9] * 5,
                                    [0.76] * 12), "D")
    assert c["collapsed_layers"] == [6]


def test_band_is_read_at_three_decimals():
    assert d2.in_band(0.9616) and d2.in_band(0.9874) and d2.in_band(0.962)
    assert not d2.in_band(0.9614) and not d2.in_band(0.9876)
    proj = pd.DataFrame({"model": ["A"] * 3 + ["B"] * 3, "layer": [1, 2, 3] * 2,
                         "method": "abtt_optimal", "D": [1, 2, 10] * 2,
                         "aucroc": [0.97, 0.965, 0.98, 0.97, 0.99, 0.95]})
    r = d2.band_readout(proj, "abtt_optimal").set_index("model")
    assert bool(r.loc["A", "all_in_band"]) and r.loc["A", "D_values"] == "1,2,10"
    assert not r.loc["B", "all_in_band"] and r.loc["B", "n_in_band"] == 1
    assert (r.loc["B", "auroc_min_layer"], r.loc["B", "auroc_max_layer"]) == (3, 2)


def test_comparability_check_flags_drift_and_missing_rows():
    ref = pd.DataFrame({"model": ["T5-base"] * 2, "layer": [1, 2], "aucroc": [0.8, 0.85],
                        "pc1": [0.3, 0.4], "erank": [20.0, 10.0], "mean_cos": [0.5, 0.6]})
    same = ref.copy()
    same["aucroc"] += 5e-5
    same["erank"] *= 1 + 5e-5
    rep = d2.compare_to_reference(same, ref, ["T5-base"])
    assert len(rep) == 8 and rep["ok"].all()
    drift = ref.copy()
    drift.loc[1, "pc1"] += 2e-4
    rep = d2.compare_to_reference(drift, ref, ["T5-base"])
    bad = rep[~rep["ok"]]
    assert list(zip(bad["layer"], bad["metric"])) == [(2, "pc1")]
    rep = d2.compare_to_reference(ref.iloc[:1], ref, ["T5-base"])
    assert (~rep["ok"]).sum() == 4


def test_published_gate(tmp_path):
    pub = pd.DataFrame({"model": "bowphs/LaTa", "repr": "hidden", "pooling": "mean",
                        "layer": [6, 6, 6], "method": ["baseline", "abtt_fixed", "abtt_optimal"],
                        "D": [10, 10, 3], "aucroc": [0.4957, 0.9618, 0.9620],
                        "dir_acc_at_1": [0.3, 0.86, 0.87], "train_dir_acc_at_1": [0.3, 0.8, 0.81]})
    path = tmp_path / "res.csv"
    pub.to_csv(path, index=False)
    mine = pub.drop(columns=["repr", "pooling"]).assign(model="LaTa", model_id="bowphs/LaTa")
    mine.loc[0, "D"] = 0
    assert d2.published_gate(mine, path) == []
    off = mine.copy()
    off.loc[2, "D"] = 5
    off.loc[1, "aucroc"] += 1e-5
    assert len(d2.published_gate(off, path)) == 2
    # Rows of models outside the gate are not checked.
    assert d2.published_gate(mine.assign(model="T5-base"), path) == []


# ----------------------------------------------------------------------------- committed outputs
needs_outputs = pytest.mark.skipif(not (OUT / "d2_projection.csv").exists(),
                                   reason="D2 outputs not present (runs/active/reframe/d2)")


@needs_outputs
def test_committed_config_confirms_the_layout():
    cfg = pd.read_csv(OUT / "d2_config_check.csv", keep_default_na=False).set_index("model")
    x = cfg.loc["T5-efficient-base"]
    assert x["layout_problems"] == ""
    assert x["feed_forward_proj"] == "relu" and str(x["tie_word_embeddings"]) == "True"
    assert x["gin_mixture"] == "c4_v220_unsupervised"
    assert cfg.loc["T5-v1.1-base", "feed_forward_proj"] == "gated-gelu"
    assert str(cfg.loc["T5-v1.1-base", "lm_head_separate"]) == "True"


@needs_outputs
def test_committed_comparability_and_gate():
    rep = pd.read_csv(OUT / "d2_repro.csv")
    assert set(rep["model"]) == set(d2.CONTROLS) and rep["ok"].all()
    assert len(rep) == 4 * 12 * len(d2.REPRO_TOL)
    proj = pd.read_csv(OUT / "d2_projection.csv")
    assert d2.published_gate(proj, REPO / d2.RES_CSV) == []
    n_models = len(d2.GATE_MODELS) + len(d2.D2_MODELS)
    assert len(proj) == n_models * 12 * len(d2.METHODS)
    # The projection's baseline equals the panel script's AUROC for every D2 model.
    lay = pd.read_csv(OUT / "p2x2_layers.csv").set_index(["model", "layer"])["aucroc"]
    base = proj[(proj["method"] == "baseline") & proj["model"].isin(lay.index.levels[0])]
    for r in base.itertuples():
        assert r.aucroc == pytest.approx(lay.loc[(r.model, r.layer)], abs=1e-9)


@needs_outputs
def test_committed_facts_and_tables_rerender(tmp_path):
    import shutil
    out, tab = tmp_path / "out", tmp_path / "tab"
    out.mkdir()
    for name in ("p2x2_layers.csv", "d2_projection.csv", "d2_repro.csv", "d2_config_check.csv"):
        shutil.copy(OUT / name, out / name)
    status = d2.main(["--stage", "render", "--out_dir", str(out), "--tab_dir", str(tab),
                      "--res_csv", str(REPO / d2.RES_CSV)])
    assert status == 0
    assert (out / "d2_facts.md").read_text() == (OUT / "d2_facts.md").read_text()
    for name in ("d2_t5_efficient.tex", "d2_controls_abtt.tex"):
        committed = REPO / "overleaf_drafts" / "tables" / name
        assert (tab / name).read_text() == committed.read_text()
