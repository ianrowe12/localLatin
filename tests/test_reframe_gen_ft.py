"""Issue #234 (reframe GEN + FT): sampler logic and the table/figure renderers.

The sampler tests use a toy token counter (one token per word plus EOS), so they pin the
matching logic without a tokenizer download. The renderer tests build a small synthetic
geometry CSV and check that rendering is deterministic and byte-identical across runs.
The last test regenerates the committed tables from the real run outputs and compares
them byte for byte; it skips when runs/ is absent (CI, fresh clones).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "paper" / "reframe"))

import gen_english_sample as ges  # noqa: E402
import gen_ft_geometry as gfg  # noqa: E402


def toy_count(text: str) -> int:
    return len(text.split()) + 1  # words + EOS


# ----------------------------------------------------------------------------- sampler
def test_prose_paragraphs_drops_caption_and_citation_lines():
    text = ("\n    SMITH v. JONES.\n    No. 12-345.\n"
            "The court below erred in holding that the statute applies to the contract here.\n"
            "See 12 F.2d 34, 35 (1970); 45 U.S. 67, 89 (1901); 23 N.E.2d 1, 2 (1950) § 4.\n"
            "Short line.\n")
    paras = ges.prose_paragraphs(text, min_words=8)
    assert paras == ["The court below erred in holding that the statute applies to the contract here."]


def test_sentence_starts():
    words = "The court held. It was wrong. see also Smith. Then (done).".split()
    assert ges.sentence_starts(words) == [0, 3, 9]


def test_fit_span_hits_target_exactly_with_additive_counter():
    words = [f"w{i}" for i in range(50)]
    k, n = ges.fit_span(words, 5, 21, toy_count)
    assert (k, n) == (20, 21)
    assert ges.fit_span(words, 45, 21, toy_count) == (0, 6)  # remainder too short


def test_tolerance():
    assert ges.tolerance(10) == 2 and ges.tolerance(100) == 3 and ges.tolerance(1000) == 30


def _toy_docs(n_docs=40, n_words=300, seed=0):
    rng = np.random.default_rng(seed)
    docs = []
    for d in range(n_docs):
        words = []
        while len(words) < n_words:
            sent = [f"x{rng.integers(1000)}" for _ in range(rng.integers(4, 15))]
            sent[0] = sent[0].capitalize()
            sent[-1] += "."
            words += sent
        docs.append(("shard", f"doc{d}", words))
    return docs


def test_match_sample_is_deterministic_one_doc_per_passage_and_within_tolerance():
    targets = [1, 5, 17, 60, 120, 240, 33, 2]
    docs = _toy_docs()
    a = ges.match_sample(targets, docs, toy_count, seed=42)
    b = ges.match_sample(targets, docs, toy_count, seed=42)
    assert a == b
    assert a[0]["text"] == "" and a[0]["len_mt5"] == 1  # empty Latin file -> empty partner
    used = [r["doc_id"] for r in a if r["doc_id"]]
    assert len(used) == len(set(used))
    for t, r in zip(targets, a):
        assert abs(r["len_mt5"] - t) <= ges.tolerance(t)
        assert toy_count(r["text"]) == r["len_mt5"]
        if r["doc_id"]:  # a passage opens at a sentence start
            assert r["text"][0].isupper()
    assert ges.match_sample(targets, docs, toy_count, seed=7) != a


# ----------------------------------------------------------------------------- render
def _geo_fixture() -> pd.DataFrame:
    rows = []
    for _, disp, _, _ in gfg.GEN_MODELS:
        for ti, (text, _) in enumerate(gfg.TEXTS):
            for layer in range(1, 13):
                mid = 3 <= layer <= 10
                pc1 = (0.9 - 0.02 * layer if mid else 0.3) - 0.1 * ti
                for subset, n in [("train", 847), ("all", 1705)]:
                    rows.append(dict(model=disp, text=text, layer=layer, subset=subset, n=n,
                                     pc1=round(pc1, 4), erank=round(1.5 / pc1, 4), pc10=0.99,
                                     mean_cos=0.5))
    return pd.DataFrame(rows)


def _ft_fixture() -> pd.DataFrame:
    rows = []
    for tag, top in [("pretrained", 0.938), ("finetuned", 0.984)]:
        for layer in range(1, 13):
            auc = top if layer == 12 else (0.93 if layer == 1 else 0.51)
            rows.append(dict(model=tag, layer=layer, auroc=auc, gap=0.1, auroc_abtt=0.97, D_abtt=10,
                             pc1=0.9 if 2 <= layer <= 11 else 0.2, erank=1.5, mean_cos_train=0.5))
    return pd.DataFrame(rows)


def test_ranges():
    assert gfg._ranges([]) == "none"
    assert gfg._ranges([3]) == "3"
    assert gfg._ranges([2, 3, 4, 7, 9, 10]) == "2--4, 7, 9--10"


def test_summarize_and_gen_table(tmp_path):
    summ = gfg.summarize(_geo_fixture())
    assert len(summ) == 6
    x = summ.iloc[0]
    assert (x.model, x.text, x.pc1_max_layer, x.high_layers) == ("mT5-base", "Latin", 3, "3--10")
    p1, p2 = tmp_path / "a.tex", tmp_path / "b.tex"
    gfg.write_gen_table(summ, p1)
    gfg.write_gen_table(gfg.summarize(_geo_fixture()), p2)
    assert p1.read_bytes() == p2.read_bytes()
    tex = p1.read_text()
    assert r"\label{tab:gen_geometry}" in tex and "T5-v1.1-base" in tex and "847" in tex
    assert "—" not in tex  # no em-dashes in paper text


def test_ft_table(tmp_path):
    p = tmp_path / "ft.tex"
    gfg.write_ft_table(_ft_fixture(), p)
    tex = p.read_text()
    assert r"12 & 0.938 & 0.984" in tex and r"\label{tab:ft_lata_layerwise}" in tex


def test_render_end_to_end_is_byte_identical(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "runs"
    out.mkdir()
    _geo_fixture().to_csv(out / "gen_geometry.csv", index=False)
    _ft_fixture().to_csv(out / "ft_layerwise.csv", index=False)
    outputs = []
    for run in ("r1", "r2"):
        args = type("A", (), dict(out_dir=str(out), tab_dir=str(tmp_path / run / "t"),
                                  fig_dir=str(tmp_path / run / "f"), no_figure=False))
        gfg.render(args)
        outputs.append({p.relative_to(tmp_path / run): p.read_bytes()
                        for p in sorted((tmp_path / run).rglob("*")) if p.is_file()})
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        assert outputs[0][key] == outputs[1][key], key


def test_committed_tables_regenerate_from_run_outputs(tmp_path):
    out = REPO / "runs/active/reframe/gen"
    if not (out / "gen_geometry.csv").exists() or not (out / "ft_layerwise.csv").exists():
        pytest.skip("runs/active/reframe/gen is absent (gitignored run outputs)")
    args = type("A", (), dict(out_dir=str(tmp_path), tab_dir=str(tmp_path / "t"),
                              fig_dir=str(tmp_path / "f"), no_figure=True))
    for name in ("gen_geometry.csv", "ft_layerwise.csv", "gen_repro.csv"):
        if (out / name).exists():
            (tmp_path / name).write_bytes((out / name).read_bytes())
    gfg.render(args)
    for name in ("gen_geometry.tex", "ft_lata_layerwise.tex"):
        assert (tmp_path / "t" / name).read_bytes() == (REPO / "overleaf_drafts/tables" / name).read_bytes(), name
