"""Issue #244: per-layer raw Task A AUROC for tab:panel_2x2 (raw_auroc_layers.py).

The synthetic tests build a toy split and a toy cache whose rows are stored in a
different order from the split, so they check filename alignment, the minimum-layer
rule and the reproduction check without any real embeddings; they run in CI. The last
test recomputes two committed cells from the real caches and skips when those are
absent (CI, fresh clones).
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
import raw_auroc_layers as ral  # noqa: E402


def toy_split(n_dirs: int = 4, per_split: int = 2) -> pd.DataFrame:
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


def test_task_a_auroc_separable_and_random():
    split = toy_split()
    assert ral.task_a_auroc(clustered(split, 0.01, 0), split) == pytest.approx(1.0)
    rand = np.random.default_rng(1).standard_normal((len(split), 16))
    assert 0.0 <= ral.task_a_auroc(rand, split) <= 1.0


def test_score_models_aligns_by_filename_and_finds_minimum(tmp_path, monkeypatch):
    split = toy_split()
    good = clustered(split, 0.01, 0)
    bad = np.random.default_rng(3).standard_normal((len(split), 16))
    monkeypatch.setattr(ral, "MODELS", [("Toy", "org/toy", "gen")])
    write_cache(tmp_path / "gen" / "org_toy" / "latin", split, {1: good, 2: bad, 3: good})
    df = ral.score_models(split, tmp_path / "panel", tmp_path / "gen")
    assert list(df["layer"]) == [1, 2, 3]
    # A shuffled cache still scores 1.0 at the separable layers: rows went by filename.
    assert df.loc[df.layer == 1, "aucroc"].item() == pytest.approx(1.0)
    m = ral.minima(df)
    assert m.loc[0, "layer"] == 2


def test_minima_takes_first_layer_on_ties():
    df = pd.DataFrame({"model": ["A"] * 3, "layer": [1, 2, 3], "aucroc": [0.9, 0.5, 0.5]})
    assert ral.minima(df).loc[0, "layer"] == 2


def test_check_reproduction_flags_mismatch(tmp_path):
    df = pd.DataFrame({"model": ["LaTa", "LaTa"], "model_id": ["bowphs/LaTa"] * 2,
                       "source": ["panel"] * 2, "layer": [5, 6], "aucroc": [0.51, 0.4957]})
    assert ral.check_reproduction(df, None) == []
    res = pd.DataFrame({"model": ["bowphs/LaTa"] * 2, "repr": ["hidden"] * 2,
                        "pooling": ["mean"] * 2, "method": ["baseline"] * 2,
                        "layer": [5, 6], "aucroc": [0.51, 0.4957]})
    res.to_csv(tmp_path / "res.csv", index=False)
    assert ral.check_reproduction(df, tmp_path / "res.csv") == []
    df.loc[1, "aucroc"] = 0.60  # the minimum moves to layer 5 and the cell disagrees
    problems = ral.check_reproduction(df, tmp_path / "res.csv")
    assert any("printed 0.496" in p for p in problems)
    assert any("LaTa L6" in p for p in problems)


# ----------------------------------------------------------------- real caches (skip)
def _root(rel: str, probe: str):
    for base in (REPO, Path("/u/irowerojas/localLatin")):
        try:
            if (base / rel / probe).exists():
                return base / rel
        except OSError:  # an unreadable path (PermissionError on another user's home) is absent
            continue
    return None


BASES = _root("runs/active/resubmit_bases", "phase9_bases/bowphs_LaTa/hidden_mean_tokempty/hidden_layer6_embeddings.npy")
GEN = _root("runs/active/reframe/gen/bases", "google_t5-v1_1-base/latin/hidden_layer2_embeddings.npy")
COMMITTED = REPO / ral.OUT_CSV


@pytest.mark.skipif(BASES is None or GEN is None or not COMMITTED.exists()
                    or not (REPO / ral.SPLIT_CSV).exists(),
                    reason="embedding caches or committed raw_auroc_layers.csv absent")
def test_committed_cells_recompute_from_caches():
    split = pd.read_csv(REPO / ral.SPLIT_CSV)
    ref = pd.read_csv(COMMITTED).set_index(["model", "layer"])["aucroc"]
    for name, layer in (("LaTa", 6), ("T5-v1.1-base", 2)):
        df = ral.score_models(split, BASES, GEN, names=[name], layers=[layer])
        assert df["aucroc"].item() == pytest.approx(float(ref.loc[(name, layer)]), abs=1e-6)
    assert round(float(ref.loc[("LaTa", 6)]), 3) == 0.496
