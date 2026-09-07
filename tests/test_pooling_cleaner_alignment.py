"""The ABTT cleaners must be fitted on the true train rows (issue #113, #87).

``fit_cleaner`` selects its fit set with ``emb[split["split"] == "train"]``. That
is only the train set if the cached matrix is in split order, and it is not: the
cache is frozen in corpus-walk order while the split is re-sorted by
``(folder_id, filename)`` whenever a label changes. The old guard compared row
*counts*, which a permutation leaves untouched, so a relabelling silently pulled
test documents into the fit that the deployed cleaner is built from.

Every test here uses a cache whose manifest order is a real permutation of the
split order, with the counts equal on both sides.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

from embedding_alignment import AlignmentResolver  # noqa: E402

from pooling_cleaners import (  # noqa: E402
    embeddings_path,
    fit_cleaner,
    load_pooled_embeddings,
)

SLUG = "bowphs_LaTa"
LAYER = 1
DIM = 6

# Six files; the relabelling moves e and f to the front of the split, which is
# where two train files used to sit.
CACHE_ORDER = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt", "f.txt"]
SPLIT_ORDER = ["e.txt", "f.txt", "a.txt", "b.txt", "c.txt", "d.txt"]
# Keyed by filename, so the split's own labels travel with the file, not the row.
SPLIT_OF = {
    "a.txt": "train", "b.txt": "train", "c.txt": "train",
    "d.txt": "test", "e.txt": "test", "f.txt": "test",
}


def vector_for(filename: str) -> np.ndarray:
    """A distinct, non-collinear row per file, deterministic across runs."""
    rng = np.random.default_rng(CACHE_ORDER.index(filename))
    return rng.standard_normal(DIM).astype(np.float32)


def matrix_in(order: Sequence[str]) -> np.ndarray:
    return np.stack([vector_for(name) for name in order])


def split_frame(order: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": range(len(order)),
            "folder_id": [f"Dir{n[0]}" for n in order],
            "filename": list(order),
            "path": [f"data/canon_labelled/Dir{n[0]}/{n}" for n in order],
            "split": [SPLIT_OF[n] for n in order],
        }
    )


def write_cache(tmp_path: Path, pooling: str = "mean", manifest: bool = True) -> Path:
    """A cache in CACHE_ORDER, with the manifest the extractor writes beside it."""
    bases_root = tmp_path / "phase9_bases"
    path = embeddings_path(bases_root, SLUG, pooling, LAYER)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix_in(CACHE_ORDER))
    if manifest:
        pd.DataFrame(
            {
                "file_id": range(len(CACHE_ORDER)),
                "path": [f"canon_labelled/Dir{n[0]}/{n}" for n in CACHE_ORDER],
            }
        ).to_csv(path.parent / "meta.csv", index=False)
    return bases_root


def true_train_matrix() -> np.ndarray:
    return np.stack([vector_for(n) for n in SPLIT_ORDER if SPLIT_OF[n] == "train"])


def positional_train_matrix() -> np.ndarray:
    """What the old code fitted on: the cache rows under the split's mask."""
    mask = np.array([SPLIT_OF[n] == "train" for n in SPLIT_ORDER])
    return matrix_in(CACHE_ORDER)[mask]


# --- load_pooled_embeddings ------------------------------------------------


def test_load_reorders_into_split_order_when_given_the_split(tmp_path):
    bases_root = write_cache(tmp_path)
    emb = load_pooled_embeddings(
        bases_root, SLUG, "mean", LAYER, n_expected=len(SPLIT_ORDER),
        split=split_frame(SPLIT_ORDER),
    )
    for row, name in enumerate(SPLIT_ORDER):
        assert emb[row].tobytes() == vector_for(name).tobytes()


def test_load_without_a_split_keeps_cache_order(tmp_path):
    """The unlabelled cache is not described by the split and must not move."""
    bases_root = write_cache(tmp_path)
    emb = load_pooled_embeddings(bases_root, SLUG, "mean", LAYER)
    assert np.array_equal(emb, matrix_in(CACHE_ORDER))


def test_load_reuses_a_prebuilt_resolver(tmp_path):
    bases_root = write_cache(tmp_path)
    resolver = AlignmentResolver(split_frame(SPLIT_ORDER))
    emb = load_pooled_embeddings(
        bases_root, SLUG, "mean", LAYER, n_expected=len(SPLIT_ORDER), resolver=resolver
    )
    assert emb[0].tobytes() == vector_for("e.txt").tobytes()
    assert "verified-permuted" in resolver.summary()


def test_load_keeps_the_row_count_error_for_a_stale_cache(tmp_path):
    bases_root = write_cache(tmp_path)
    with pytest.raises(SystemExit, match="Wrong --labelled_bases"):
        load_pooled_embeddings(bases_root, SLUG, "mean", LAYER, n_expected=99)


def test_load_still_reports_a_missing_cache(tmp_path):
    with pytest.raises(SystemExit, match="Embedding cache missing"):
        load_pooled_embeddings(tmp_path / "nothing", SLUG, "mean", LAYER)


# --- fit_cleaner -----------------------------------------------------------


def test_cleaner_is_fitted_on_the_true_train_rows(tmp_path):
    """The fit set is chosen by filename, so no test document leaks into it."""
    bases_root = write_cache(tmp_path)
    cleaner = fit_cleaner(
        bases_root, SLUG, "mean", LAYER, split_frame(SPLIT_ORDER),
        fixed_d=1, verbose=False,
    )

    expected = true_train_matrix()
    assert np.allclose(cleaner.mean_vec, expected.mean(axis=0), atol=1e-6)
    # ...and the positional fit is a genuinely different one, so this is a real
    # discrimination rather than two means that happen to coincide.
    assert not np.allclose(
        cleaner.mean_vec, positional_train_matrix().mean(axis=0), atol=1e-6
    )


def test_cleaner_removes_the_train_subspace_not_the_positional_one(tmp_path):
    bases_root = write_cache(tmp_path, pooling="sif")
    cleaner = fit_cleaner(
        bases_root, SLUG, "sif", LAYER, split_frame(SPLIT_ORDER),
        fixed_d=1, verbose=False,
    )

    centered = true_train_matrix() - true_train_matrix().mean(axis=0)
    expected_pc = np.linalg.svd(centered, full_matrices=False)[2][0]
    cos = abs(float(cleaner.pcs[0] @ expected_pc))
    assert cos == pytest.approx(1.0, abs=1e-5)


def test_cleaner_is_unchanged_when_the_orders_already_agree(tmp_path):
    """The alignment is a no-op on a cache that was never relabelled."""
    bases_root = write_cache(tmp_path)
    split = split_frame(CACHE_ORDER)
    cleaner = fit_cleaner(
        bases_root, SLUG, "mean", LAYER, split, fixed_d=1, verbose=False
    )
    mask = np.array([SPLIT_OF[n] == "train" for n in CACHE_ORDER])
    assert np.allclose(
        cleaner.mean_vec, matrix_in(CACHE_ORDER)[mask].mean(axis=0), atol=1e-6
    )


def test_cleaner_falls_back_to_position_without_a_manifest(tmp_path, capsys):
    """Old caches still run, loudly: the warning is the only thing standing in."""
    bases_root = write_cache(tmp_path, manifest=False)
    cleaner = fit_cleaner(
        bases_root, SLUG, "mean", LAYER, split_frame(SPLIT_ORDER),
        fixed_d=1, verbose=False,
    )
    assert "WARNING" in capsys.readouterr().err
    assert np.allclose(
        cleaner.mean_vec, positional_train_matrix().mean(axis=0), atol=1e-6
    )
