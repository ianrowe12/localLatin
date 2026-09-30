"""Issue #248: the architecture-by-objective panel (p2x2_panel.py).

The synthetic tests build a toy split and toy caches whose rows are stored in a different
order from the split, so they pin filename alignment, the minimum / maximum / range rules,
the reproduction gate and the renderers without any real embeddings; they run in CI. The
last test recomputes two published cells from the real caches. It is opt-in: set
P2X2_DATA_ROOT to a checkout that holds ``runs/active/resubmit_bases`` (nothing outside the
repo is probed otherwise).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "paper" / "reframe"))

pytest.importorskip("sklearn")
import p2x2_panel as p2  # noqa: E402
import raw_auroc_layers as ral  # noqa: E402

FT_TEX = REPO / "overleaf_drafts" / "tables" / "ft_lata_layerwise.tex"
TOY_MODELS = [
    ("LaTa", "bowphs/LaTa", "T5, raw", "none", "panel"),
    ("LaBERTa", "bowphs/LaBerta", "Enc., raw", "none", "p2x2"),
]


# ----------------------------------------------------------------------------- fixtures
def toy_split(n_dirs: int = 4, per_split: int = 3) -> pd.DataFrame:
    rows = []
    for d in range(n_dirs):
        for s in ("train", "test"):
            for k in range(per_split):
                rows.append({"folder_id": f"dir{d}", "filename": f"f{d}_{s}_{k}.txt",
                             "split": s, "has_test_partner": s == "test"})
    return pd.DataFrame(rows).sort_values(["folder_id", "filename"]).reset_index(drop=True)


def write_cache(run_dir: Path, split: pd.DataFrame, layers: dict, seed: int = 0) -> None:
    """Store each layer's split-ordered matrix in a shuffled cache order plus meta.csv."""
    run_dir.mkdir(parents=True)
    order = np.random.default_rng(seed).permutation(len(split))
    pd.DataFrame({"filename": split["filename"].values[order]}).to_csv(run_dir / "meta.csv", index=False)
    for layer, emb in layers.items():
        np.save(run_dir / f"hidden_layer{layer}_embeddings.npy", emb[order])


def clustered(split: pd.DataFrame, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = sorted(split["folder_id"].unique())
    centers = rng.standard_normal((len(dirs), 16)) * 5
    idx = split["folder_id"].map({d: i for i, d in enumerate(dirs)}).values
    return centers[idx] + noise * rng.standard_normal((len(split), 16))


def collapsed_train(split: pd.DataFrame, seed: int) -> np.ndarray:
    """Train rows on one line, test rows isotropic: the train geometry is only right when
    the train rows are picked by filename."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((len(split), 16))
    tr = split["split"].values == "train"
    emb[tr] = rng.standard_normal((tr.sum(), 1)) * rng.standard_normal((1, 16)) * 10 + 3.0
    return emb


def toy_caches(tmp_path, monkeypatch, split):
    monkeypatch.setattr(p2, "MODELS", TOY_MODELS)
    monkeypatch.setattr(p2, "LAYERS", (1, 2, 3))
    good, line = clustered(split, 0.01, 0), collapsed_train(split, 1)
    write_cache(p2.run_dir("panel", "bowphs/LaTa", tmp_path / "bases", tmp_path / "p2x2"),
                split, {1: good, 2: line, 3: good}, seed=5)
    write_cache(p2.run_dir("p2x2", "bowphs/LaBerta", tmp_path / "bases", tmp_path / "p2x2"),
                split, {1: good, 2: good, 3: line}, seed=6)
    return good, line


def layers_fixture(models=None) -> pd.DataFrame:
    """A synthetic p2x2_layers.csv: raw T5 rows collapse at mid-depth, the rest do not."""
    rows = []
    for i, (name, model_id, cell, objective, source) in enumerate(p2.MODELS):
        if models is not None and name not in models:
            continue
        for layer in range(1, 13):
            mid = 2 <= layer <= 11
            t5_raw = cell == "T5, raw"
            auc = (0.50 + 0.01 * i + 0.001 * layer) if (t5_raw and mid) else 0.85 + 0.005 * i
            pc1 = (0.90 + 0.005 * i + 0.001 * layer) if (t5_raw and mid) else 0.10 + 0.02 * i
            rows.append({"model": name, "model_id": model_id, "cell": cell,
                         "emb_objective": objective, "source": source, "layer": layer,
                         "aucroc": round(auc, 6), "n_train": 847, "pc1": round(pc1, 6),
                         "erank": round(1.2 / pc1, 6), "pc10": 0.99, "mean_cos": 0.5})
    return pd.DataFrame(rows)


def gate_refs(tmp_path: Path, df: pd.DataFrame):
    """Reference CSVs that agree with ``df`` exactly (the gate's own input formats)."""
    panel = df[df.source == "panel"]
    res = pd.DataFrame({"model": panel.model_id, "repr": "hidden", "pooling": "mean",
                        "method": "baseline", "layer": panel.layer, "aucroc": panel.aucroc})
    geo = pd.DataFrame({"model": panel.model_id, "repr": "hidden", "pooling": "mean",
                        "layer": panel.layer, "split": "train", "view": "raw",
                        "pc1_variance_ratio": panel.pc1, "effective_rank_entropy": panel.erank})
    decoy = geo.assign(split="test", pc1_variance_ratio=0.0, effective_rank_entropy=1.0)
    t5 = df[df.model == p2.T5V11]
    old = pd.DataFrame({"model": t5.model, "model_id": t5.model_id, "source": "gen",
                        "layer": t5.layer, "aucroc": t5.aucroc})
    paths = [tmp_path / n for n in ("res.csv", "geo.csv", "t5v11.csv")]
    res.to_csv(paths[0], index=False)
    pd.concat([decoy, geo]).to_csv(paths[1], index=False)
    old.to_csv(paths[2], index=False)
    return paths


def printed_df() -> pd.DataFrame:
    """A full synthetic table whose summary equals every cell the paper prints."""
    df = layers_fixture()

    def put(model, col, base, layer, value):
        """``base`` at every layer of the model, ``value`` at one layer."""
        rows = df.model == model
        df.loc[rows, col] = base
        df.loc[rows & (df.layer == layer), col] = value

    put("LaTa", "aucroc", 0.51, 6, 0.4957)
    put("LaTa", "pc1", 0.93, 4, 0.9518)
    put("PhilTa", "aucroc", 0.55, 10, 0.5380)
    put("PhilTa", "pc1", 0.80, 6, 0.8584)
    put("mT5-base", "aucroc", 0.66, 5, 0.6537)
    put("mT5-base", "pc1", 0.95, 5, 0.99995)
    put("LaBSE", "aucroc", 0.90, 1, 0.8057)
    put("LaBSE", "pc1", 0.30, 12, 0.5241)
    put(p2.T5V11, "aucroc", 0.52, 2, 0.4886)
    put(p2.T5V11, "pc1", 0.96, 9, 0.9752)
    df.loc[(df.model == p2.T5V11) & df.layer.isin([1, 12]), "pc1"] = 0.2
    df["erank"] = (1.2 / df["pc1"]).round(6)
    return df


# ----------------------------------------------------------------------------- compute
def test_score_models_aligns_by_filename(tmp_path, monkeypatch):
    split = toy_split()
    good, line = toy_caches(tmp_path, monkeypatch, split)
    df = p2.score_models(split, tmp_path / "bases", tmp_path / "p2x2")
    assert list(df["model"]) == ["LaTa"] * 3 + ["LaBERTa"] * 3
    assert list(df["layer"]) == [1, 2, 3, 1, 2, 3]
    assert set(df["n_train"]) == {12}
    by = df.set_index(["model", "layer"])
    # Shuffled caches still score 1.0 at the separable layers: rows went by filename.
    assert by.loc[("LaTa", 1), "aucroc"] == pytest.approx(1.0)
    assert by.loc[("LaBERTa", 2), "aucroc"] == pytest.approx(1.0)
    # Only the true train rows lie on one line; a positional read would mix in test rows.
    assert by.loc[("LaTa", 2), "pc1"] > 0.999 and by.loc[("LaTa", 2), "erank"] < 1.01
    assert by.loc[("LaBERTa", 3), "pc1"] > 0.999
    train = split["split"].values == "train"
    want = p2.layer_stats(good[train])
    assert by.loc[("LaTa", 1), "pc1"] == pytest.approx(want["pc1"])
    assert by.loc[("LaTa", 1), "erank"] == pytest.approx(want["erank"])
    assert by.loc[("LaTa", 1), "mean_cos"] == pytest.approx(want["mean_cos"])
    assert by.loc[("LaTa", 1), "aucroc"] == pytest.approx(ral.task_a_auroc(good, split))
    assert list(by.loc["LaTa", "cell"]) == ["T5, raw"] * 3


def test_score_models_restricts_and_rejects_unknown_and_missing(tmp_path, monkeypatch):
    split = toy_split()
    toy_caches(tmp_path, monkeypatch, split)
    df = p2.score_models(split, tmp_path / "bases", tmp_path / "p2x2", names=["LaBERTa"])
    assert set(df["model"]) == {"LaBERTa"}
    with pytest.raises(SystemExit, match="unknown model"):
        p2.score_models(split, tmp_path / "bases", tmp_path / "p2x2", names=["Nope"])
    monkeypatch.setattr(p2, "LAYERS", (1, 2, 3, 4))  # layer 4 was never extracted
    with pytest.raises(SystemExit, match="missing"):
        p2.score_models(split, tmp_path / "bases", tmp_path / "p2x2")


def test_summarize_minimum_maximum_and_ties():
    df = pd.DataFrame({
        "model": ["A"] * 4, "model_id": ["o/a"] * 4, "cell": ["T5, raw"] * 4,
        "emb_objective": ["none"] * 4, "source": ["p2x2"] * 4, "layer": [1, 2, 3, 4],
        "aucroc": [0.9, 0.5, 0.5, 0.95], "n_train": [10] * 4, "pc1": [0.2, 0.7, 0.7, 0.6],
        "erank": [50.0, 1.5, 1.2, 1.2], "pc10": [0.9] * 4, "mean_cos": [0.5] * 4})
    x = p2.summarize(df).iloc[0]
    assert (x.auroc_min, x.auroc_min_layer) == (0.5, 2)       # first layer on ties
    assert (x.auroc_max, x.auroc_max_layer) == (0.95, 4)
    assert (x.auroc_first, x.auroc_last) == (0.9, 0.95)
    assert (x.pc1_max, x.pc1_max_layer, x.erank_at_pc1_max) == (0.7, 2, 1.5)
    assert (x.erank_min, x.erank_min_layer) == (1.2, 3)
    assert (x.n_high_pc1, x.high_pc1_layers) == (3, "2--4")   # threshold is inclusive
    assert (x.n_low_auroc, x.low_auroc_layers) == (2, "2--3")


def test_summarize_keeps_panel_order():
    df = layers_fixture().sample(frac=1.0, random_state=0)
    assert list(p2.summarize(df)["model"]) == [m[0] for m in p2.MODELS]


def test_printed_minima_agree_with_raw_auroc_layers():
    for name, (layer, value) in ral.PRINTED_MIN.items():
        assert p2.PRINTED[name]["auroc_min"] == value
        assert p2.PRINTED[name]["auroc_min_layer"] == layer


# ----------------------------------------------------------------------------- gate
def test_gate_passes_on_matching_references(tmp_path):
    df = printed_df()
    res, geo, t5 = gate_refs(tmp_path, df)
    problems, records, lines = p2.reproduction_gate(df, res, geo, t5)
    assert problems == []
    assert len(records) == 48 + 48 + 48 + 12     # a, b (pc1 and erank), c
    assert sum("PASS" in line for line in lines) == 5 and not any("SKIPPED" in x for x in lines)


def test_gate_flags_each_kind_of_mismatch(tmp_path):
    df = printed_df()
    res, geo, t5 = gate_refs(tmp_path, df)

    bad = df.copy()
    bad.loc[(bad.model == "PhilTa") & (bad.layer == 3), "aucroc"] += 5e-6
    problems, _, lines = p2.reproduction_gate(bad, res, geo, t5)
    assert len(problems) == 1 and "gate a: PhilTa L3 aucroc" in problems[0]
    assert any(line.startswith("gate a") and "FAIL" in line for line in lines)

    bad = df.copy()
    bad.loc[(bad.model == "LaBSE") & (bad.layer == 7), "erank"] *= 1 + 1e-4
    bad.loc[(bad.model == "LaTa") & (bad.layer == 2), "pc1"] -= 1e-4
    problems, _, _ = p2.reproduction_gate(bad, res, geo, t5)
    assert sorted(p[:22] for p in problems) == ["gate b: LaBSE L7 erank", "gate b: LaTa L2 pc1 0."]

    bad = df.copy()
    bad.loc[(bad.model == p2.T5V11) & (bad.layer == 12), "aucroc"] += 5e-5   # inside 1e-4
    assert p2.reproduction_gate(bad, res, geo, t5)[0] == []
    bad.loc[(bad.model == p2.T5V11) & (bad.layer == 12), "aucroc"] += 5e-4
    problems, _, _ = p2.reproduction_gate(bad, res, geo, t5)
    assert len(problems) == 1 and problems[0].startswith(f"gate c: {p2.T5V11} L12")


def test_gate_checks_printed_cells_and_ranges(tmp_path):
    df = printed_df()
    df.loc[(df.model == "LaTa") & (df.layer == 6), "aucroc"] = 0.60       # minimum moves
    df.loc[(df.model == "LaBSE") & (df.layer == 12), "pc1"] = 0.530
    df.loc[df.model == "PhilTa", "pc1"] = 0.84                           # range 0.84--1.00
    res, geo, t5 = gate_refs(tmp_path, df)                               # a, b, c still agree
    problems, _, _ = p2.reproduction_gate(df, res, geo, t5)
    text = "\n".join(problems)
    assert "gate d: LaTa auroc_min printed 0.496" in text
    assert "gate d: LaTa auroc_min_layer printed 6" in text
    assert "gate d: LaBSE pc1_max printed 0.524, got 0.5300" in text
    assert "pc1_max printed 0.86--1.00, got 0.84--1.00" in text
    assert all(p.startswith("gate d") for p in problems)


def test_gate_skips_models_that_were_not_computed(tmp_path):
    full = printed_df()
    res, geo, t5 = gate_refs(tmp_path, full)
    four = full[full.source == "panel"]
    problems, records, lines = p2.reproduction_gate(four, res, geo, t5)
    assert problems == [] and len(records) == 48 * 3
    assert any(line.startswith("gate c") and "SKIPPED" in line for line in lines)
    assert any(line.startswith("gate d") and "SKIPPED" in line and "PASS" in line for line in lines)
    new = full[full.source == "p2x2"]
    problems, _, lines = p2.reproduction_gate(new, res, geo, t5)
    assert problems == [] and sum("SKIPPED" in line for line in lines) == 3


def test_gate_fails_when_a_reference_is_missing(tmp_path):
    df = printed_df()
    res, geo, t5 = gate_refs(tmp_path, df)
    problems, _, _ = p2.reproduction_gate(df, tmp_path / "absent.csv", geo, None)
    assert any("gate a: published results CSV not found" in p for p in problems)
    assert any("gate c" in p and "not found" in p for p in problems)
    pd.read_csv(res).query("layer != 4").to_csv(res, index=False)   # a published cell is gone
    problems, _, _ = p2.reproduction_gate(df, res, geo, t5)
    assert len(problems) == 4 and all("L4 aucroc has no reference cell" in p for p in problems)


def test_compute_exits_nonzero_and_withholds_the_csv_on_gate_failure(tmp_path, monkeypatch, capsys):
    split = toy_split()
    toy_caches(tmp_path, monkeypatch, split)
    monkeypatch.setattr(p2, "PRINTED", {})
    monkeypatch.setattr(p2, "PRINTED_RANGES", {})
    split_csv = tmp_path / "split.csv"
    split.to_csv(split_csv, index=False)
    base = ["--stage", "compute", "--split_csv", str(split_csv),
            "--bases_root", str(tmp_path / "bases"), "--p2x2_bases", str(tmp_path / "p2x2"),
            "--t5v11_csv", str(tmp_path / "unused.csv")]

    # References written from a first, ungated look at the numbers, then gated for real.
    df = p2.score_models(split, tmp_path / "bases", tmp_path / "p2x2")
    res, geo, _ = gate_refs(tmp_path, df)
    out = tmp_path / "ok"
    assert p2.main(base + ["--res_csv", str(res), "--geo_csv", str(geo), "--out_dir", str(out)]) == 0
    assert "reproduction gate passed" in capsys.readouterr().out
    got = pd.read_csv(out / "p2x2_layers.csv")
    assert list(got.columns) == ["model", "model_id", "cell", "emb_objective", "source", "layer",
                                 "aucroc", "n_train", "pc1", "erank", "pc10", "mean_cos"]
    assert len(got) == 6 and len(pd.read_csv(out / "p2x2_repro.csv")) == 9

    wrong = pd.read_csv(res)
    wrong.loc[wrong.layer == 2, "aucroc"] += 0.01
    wrong.to_csv(res, index=False)
    out = tmp_path / "bad"
    assert p2.main(base + ["--res_csv", str(res), "--geo_csv", str(geo), "--out_dir", str(out)]) == 1
    printed = capsys.readouterr().out
    assert "REPRODUCTION MISMATCH: gate a: LaTa L2 aucroc" in printed
    assert "reproduction gate FAILED" in printed
    assert not (out / "p2x2_layers.csv").exists()
    assert (out / "p2x2_layers.rejected.csv").exists()


# ----------------------------------------------------------------------------- formatting
def test_fmt_cell_single_range_and_collapsing_range():
    assert p2.fmt_cell([0.48859]) == "0.489"
    assert p2.fmt_cell([0.4957, 0.5380, 0.6537]) == "0.50--0.65"
    assert p2.fmt_cell([0.8584, 0.9518, 0.99995]) == "0.86--1.00"
    assert p2.fmt_cell([0.9012, 0.9038]) == "0.901--0.904"    # two decimals would collapse
    assert p2.fmt_cell([0.9012, 0.9012]) == "0.901"           # nothing to range over
    assert p2.fmt_cell([16.19], 2, 2) == "16.19"
    assert p2.fmt_cell([1.0007, 1.8054, 1.3619], 2, 2) == "1.00--1.81"


def test_read_ft_cells_from_the_tracked_table():
    ft = p2.read_ft_cells(FT_TEX)
    assert ft["auroc"] == "0.50--0.57" and (ft["auroc_lo"], ft["auroc_hi"]) == (0.499, 0.569)
    assert (ft["pc1_max"], ft["pc1_max_layer"]) == (0.945, 4)
    assert (ft["erank_min"], ft["erank_min_layer"]) == (1.42, 8)


def test_read_ft_cells_rejects_missing_short_or_changed_tables(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        p2.read_ft_cells(tmp_path / "absent.tex")
    text = FT_TEX.read_text()
    short = tmp_path / "short.tex"
    short.write_text("\n".join(x for x in text.splitlines() if not x.startswith("7 &")))
    with pytest.raises(SystemExit, match=r"layer\(s\) \[7\]"):
        p2.read_ft_cells(short)
    changed = tmp_path / "changed.tex"
    changed.write_text(text.replace("4 & 0.500 & 0.499 & 0.952 & 0.945", "4 & 0.500 & 0.499 & 0.952 & 0.960"))
    with pytest.raises(SystemExit, match="differ from the cells the paper prints"):
        p2.read_ft_cells(changed)


# ----------------------------------------------------------------------------- render
def _render(tmp_path: Path, df: pd.DataFrame, tag: str = "r"):
    out, tab = tmp_path / tag / "out", tmp_path / tag / "tab"
    out.mkdir(parents=True)
    df.to_csv(out / "p2x2_layers.csv", index=False)
    assert p2.main(["--stage", "render", "--out_dir", str(out), "--tab_dir", str(tab),
                    "--ft_tex", str(FT_TEX)]) == 0
    return out, tab


def test_render_panel_table_rows(tmp_path):
    out, tab = _render(tmp_path, printed_df())
    tex = (tab / "panel_2x2.tex").read_text()
    lines = tex.splitlines()
    assert lines[0] == "% generated table" and "p2x2_panel.py --stage render" in lines[1]
    assert lines[2] == r"\begin{table}[t]" and lines[-1] == r"\end{table}"
    assert r"\label{tab:panel_2x2}" in tex
    body = lines[lines.index(r"\midrule") + 1:lines.index(r"\bottomrule")]
    assert [x.split(" & ")[0] for x in body] == [
        "T5, raw", "T5, raw", "T5, raw", "T5, raw+FT", "T5, emb.", r"\midrule", "Enc., raw",
        "Enc., emb."]
    assert body[0] == r"T5, raw & \makecell[l]{LaTa, PhilTa,\\mT5-base} & 0.50--0.65 & 0.86--1.00 \\"
    assert body[1] == r"T5, raw & T5-v1.1-base & 0.489 & 0.975 \\"
    assert body[2].startswith("T5, raw & T5-base & ")
    assert body[3] == r"T5, raw+FT & LaTa (fine-tuned) & 0.50--0.57 & 0.945 \\"
    summ = p2.summarize(printed_df()).set_index("model")
    st5 = summ.loc["Sentence-T5"]
    assert body[4] == rf"T5, emb. & Sentence-T5 & {st5.auroc_min:.3f} & {st5.pc1_max:.3f} \\"
    assert body[6].startswith("Enc., raw & LaBERTa, PhilBERTa & ")
    lo, hi = sorted([summ.loc["LaBSE", "auroc_min"], summ.loc["SPhilBERTa", "auroc_min"]])
    assert body[7].startswith(f"Enc., emb. & LaBSE, SPhilBERTa & {lo:.2f}--{hi:.2f} & ")
    assert all(x.count("&") == 3 for x in body if x != r"\midrule")
    caption = [x for x in lines if x.startswith(r"\caption{")][0]
    for phrase in ("lowest Task~A test AUROC", "unmodified mean-pooled vectors",
                   "peak top-PC", "847 training passages", "two decimals",
                   "T5-base is the raw partner of Sentence-T5", "original T5 checkpoint",
                   "layers 2--11", "range of the per-model values"):
        assert phrase in caption, phrase
    assert "—" not in tex and "pendingnum" not in tex   # no em-dashes, nothing pending


def test_render_layerwise_table(tmp_path):
    out, tab = _render(tmp_path, printed_df())
    lines = (tab / "p2x2_layerwise.tex").read_text().splitlines()
    assert lines[0] == "% generated table" and lines[2] == r"\begin{table*}[t]"
    assert r"\label{tab:p2x2_layerwise}" in lines and lines[-1] == r"\end{table*}"
    header = [x for x in lines if x.startswith("& \\multicolumn")][0]
    assert [header.index(n) for n in p2.LAYERWISE_MODELS] == sorted(header.index(n) for n in p2.LAYERWISE_MODELS)
    assert "LaBSE" not in header and "LaTa" not in header
    body = lines[lines.index(r"\midrule") + 1:lines.index(r"\bottomrule")]
    assert [int(x.split(" & ")[0]) for x in body] == list(range(1, 13))
    assert all(x.count("&") == 18 for x in body)
    assert body[1].startswith("2 & 0.489 & 0.960 & ") and body[8].startswith("9 & ")
    assert " & 0.975 & " in body[8]
    assert "—" not in "\n".join(lines)


def test_render_facts(tmp_path):
    df = printed_df()
    out, _ = _render(tmp_path, df)
    text = (out / "p2x2_facts.md").read_text()
    assert "Partial run" not in text and "Reproduction gate" not in text   # no p2x2_repro.csv
    row = [x for x in text.splitlines() if x.startswith("| T5-v1.1-base |")][0]
    assert "| 0.489 (2) |" in row and "| 0.975 (9) |" in row and "| 10 | 2-11 |" in row
    assert "| T5, raw | LaTa, PhilTa, mT5-base | 0.50 to 0.65 | 0.86 to 1.00 |" in text
    assert "| T5, raw+FT | LaTa (fine-tuned) | 0.50 to 0.57 | 0.945 | 1.42 |" in text
    per_layer = [x for x in text.splitlines() if x.startswith("- LaBSE: ")][0]
    assert per_layer.count(";") == 11 and per_layer.startswith("- LaBSE: 1: 0.806 / ")
    res, geo, t5 = gate_refs(tmp_path, df)
    _, records, _ = p2.reproduction_gate(df, res, geo, t5)
    pd.DataFrame(records).to_csv(out / "p2x2_repro.csv", index=False)
    assert p2.main(["--stage", "render", "--out_dir", str(out), "--tab_dir", str(tmp_path / "t2"),
                    "--ft_tex", str(FT_TEX)]) == 0
    text = (out / "p2x2_facts.md").read_text()
    assert "| a | aucroc | LaTa, PhilTa, mT5-base, LaBSE | 48 | 0.00e+00 |" in text
    assert "| c | aucroc | T5-v1.1-base | 12 |" in text


def test_render_is_byte_identical_across_runs(tmp_path):
    a = _render(tmp_path, printed_df(), "a")
    b = _render(tmp_path, printed_df(), "b")
    for name in ("panel_2x2.tex", "p2x2_layerwise.tex"):
        assert (a[1] / name).read_bytes() == (b[1] / name).read_bytes()
    assert (a[0] / "p2x2_facts.md").read_bytes() == (b[0] / "p2x2_facts.md").read_bytes()


def test_render_partial_csv_marks_missing_cells(tmp_path, capsys):
    four = printed_df()
    four = four[four.source == "panel"]
    out, tab = _render(tmp_path, four)
    assert "WARNING: partial CSV" in capsys.readouterr().out
    lines = (tab / "panel_2x2.tex").read_text().splitlines()
    body = lines[lines.index(r"\midrule") + 1:lines.index(r"\bottomrule")]
    assert body[0].endswith(r" & 0.50--0.65 & 0.86--1.00 \\")
    assert body[1] == "T5, raw & T5-v1.1-base & " + " & ".join([p2.PENDING] * 2) + r" \\"
    assert p2.PENDING in body[7] and "0.806" not in body[7]   # LaBSE alone is not the cell
    assert not (tab / "p2x2_layerwise.tex").exists()
    assert "**Partial run: no rows for T5-v1.1-base" in (out / "p2x2_facts.md").read_text()


# ----------------------------------------------------------------- real caches (opt-in)
DATA_ROOT = os.environ.get("P2X2_DATA_ROOT")


@pytest.mark.skipif(not DATA_ROOT, reason="set P2X2_DATA_ROOT to a checkout with embedding caches")
def test_published_cells_recompute_from_caches(monkeypatch):
    bases = Path(DATA_ROOT) / "runs/active/resubmit_bases"
    probe = p2.run_dir("panel", "bowphs/LaTa", bases, bases) / "hidden_layer6_embeddings.npy"
    if not probe.exists() or not (REPO / p2.SPLIT_CSV).exists():
        pytest.skip("LaTa cache or split CSV absent")
    monkeypatch.setattr(p2, "LAYERS", (6,))
    split = pd.read_csv(REPO / p2.SPLIT_CSV)
    df = p2.score_models(split, bases, bases, names=["LaTa"])
    problems, records, _ = p2.reproduction_gate(df, REPO / p2.RES_CSV, REPO / p2.GEO_CSV, None)
    assert [p for p in problems if not p.startswith("gate d")] == [] and len(records) == 3
    assert round(df["aucroc"].item(), 3) == 0.496
