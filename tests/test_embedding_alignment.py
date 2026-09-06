"""Filename alignment between a cached embedding matrix and a split CSV (#113).

The bug this guards against is silent: after a label correction the split rows
are permuted while the cached ``.npy`` keeps its extraction order, and a
row-count check sees nothing wrong. Every test here is a case where counts match
but the pairing does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embedding_alignment import (  # noqa: E402
    STATUS_IDENTITY,
    STATUS_PERMUTED,
    STATUS_UNVERIFIED,
    AlignmentError,
    AlignmentResolver,
    EmbeddingAligner,
    build_permutation,
    find_manifest,
    load_row_order,
)

CACHE_ORDER = ["a.txt", "b.txt", "c.txt", "d.txt"]
# What the split looks like after c is relabelled into a directory that sorts
# last: two rows swap, the other two stay put.
SPLIT_ORDER = ["a.txt", "b.txt", "d.txt", "c.txt"]


def write_manifest(run_dir: Path, names, column="path") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if column == "path":
        rows = {"file_id": range(len(names)), "path": [f"canon/Dir/{n}" for n in names]}
    else:
        rows = {"file_id": range(len(names)), "filename": list(names)}
    path = run_dir / "meta.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def split_frame(names) -> pd.DataFrame:
    return pd.DataFrame({"filename": list(names), "file_id": range(len(names))})


def marked_embeddings(n_rows: int, dim: int = 3) -> np.ndarray:
    """Row i is all-i, so a misalignment is visible in the values themselves."""
    return np.arange(n_rows, dtype=np.float32).repeat(dim).reshape(n_rows, dim)


# --------------------------------------------------------------------------- #
# build_permutation
# --------------------------------------------------------------------------- #


def test_identical_orders_return_no_permutation():
    assert build_permutation(CACHE_ORDER, CACHE_ORDER) is None


def test_permutation_maps_split_rows_back_to_cache_rows():
    perm = build_permutation(CACHE_ORDER, SPLIT_ORDER)
    assert list(perm) == [0, 1, 3, 2]


def test_permutation_rejects_a_different_file_set():
    with pytest.raises(AlignmentError, match="different files"):
        build_permutation(CACHE_ORDER, ["a.txt", "b.txt", "c.txt", "e.txt"])


def test_permutation_rejects_a_stale_cache_by_length():
    with pytest.raises(AlignmentError, match="stale"):
        build_permutation(CACHE_ORDER, SPLIT_ORDER[:3])


def test_permutation_rejects_duplicate_split_filenames():
    with pytest.raises(AlignmentError, match="duplicate"):
        build_permutation(CACHE_ORDER, ["a.txt", "b.txt", "c.txt", "c.txt"])


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #


def test_load_row_order_takes_the_basename_of_a_path_column(tmp_path):
    path = write_manifest(tmp_path / "run", CACHE_ORDER)
    assert load_row_order(path) == CACHE_ORDER


def test_load_row_order_accepts_a_filename_column(tmp_path):
    path = write_manifest(tmp_path / "run", CACHE_ORDER, column="filename")
    assert load_row_order(path) == CACHE_ORDER


def test_load_row_order_rejects_a_manifest_without_usable_columns(tmp_path):
    path = tmp_path / "meta.csv"
    pd.DataFrame({"file_id": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(AlignmentError, match="neither"):
        load_row_order(path)


def test_load_row_order_rejects_duplicate_names(tmp_path):
    path = write_manifest(tmp_path / "run", ["a.txt", "a.txt"])
    with pytest.raises(AlignmentError, match="duplicate"):
        load_row_order(path)


def test_find_manifest_prefers_the_run_directory(tmp_path):
    root = tmp_path / "bases"
    run_dir = root / "model" / "hidden_mean"
    write_manifest(run_dir, CACHE_ORDER)
    pd.DataFrame({"filename": CACHE_ORDER}).to_csv(root / "row_order.csv", index=False)
    assert find_manifest(run_dir) == run_dir / "meta.csv"


def test_find_manifest_falls_back_to_a_bases_root(tmp_path):
    root = tmp_path / "bases"
    run_dir = root / "model" / "hidden_mean"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"filename": CACHE_ORDER}).to_csv(root / "row_order.csv", index=False)
    assert find_manifest(run_dir) == root / "row_order.csv"


def test_find_manifest_returns_none_when_there_is_nothing(tmp_path):
    run_dir = tmp_path / "bases" / "model" / "hidden_mean"
    run_dir.mkdir(parents=True)
    assert find_manifest(run_dir) is None


# --------------------------------------------------------------------------- #
# EmbeddingAligner
# --------------------------------------------------------------------------- #


def test_aligner_reorders_rows_to_split_order(tmp_path):
    run_dir = tmp_path / "run"
    write_manifest(run_dir, CACHE_ORDER)
    aligner = EmbeddingAligner.for_run_dir(run_dir, split_frame(SPLIT_ORDER))

    assert aligner.status == STATUS_PERMUTED
    assert aligner.n_moved == 2

    aligned = aligner.apply(marked_embeddings(4))
    # Split row 2 is d.txt, which sits at cache row 3.
    assert list(aligned[:, 0]) == [0.0, 1.0, 3.0, 2.0]


def test_aligner_is_a_no_op_when_orders_agree(tmp_path):
    run_dir = tmp_path / "run"
    write_manifest(run_dir, CACHE_ORDER)
    aligner = EmbeddingAligner.for_run_dir(run_dir, split_frame(CACHE_ORDER))

    assert aligner.status == STATUS_IDENTITY
    assert aligner.n_moved == 0
    emb = marked_embeddings(4)
    assert aligner.apply(emb) is emb


def test_aligner_preserves_vectors_bit_for_bit(tmp_path):
    """The relabelling must not change any file's vector, only where it sits."""
    run_dir = tmp_path / "run"
    write_manifest(run_dir, CACHE_ORDER)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((4, 8)).astype(np.float32)

    before = EmbeddingAligner.for_run_dir(run_dir, split_frame(CACHE_ORDER)).apply(emb)
    after = EmbeddingAligner.for_run_dir(run_dir, split_frame(SPLIT_ORDER)).apply(emb)

    for name in CACHE_ORDER:
        b = before[CACHE_ORDER.index(name)]
        a = after[SPLIT_ORDER.index(name)]
        assert a.tobytes() == b.tobytes()


def test_aligner_rejects_a_matrix_with_the_wrong_row_count(tmp_path):
    run_dir = tmp_path / "run"
    write_manifest(run_dir, CACHE_ORDER)
    aligner = EmbeddingAligner.for_run_dir(run_dir, split_frame(SPLIT_ORDER))
    with pytest.raises(AlignmentError, match="rows"):
        aligner.apply(marked_embeddings(3))


def test_aligner_without_a_manifest_falls_back_and_says_so(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    aligner = EmbeddingAligner.for_run_dir(run_dir, split_frame(SPLIT_ORDER))

    assert aligner.status == STATUS_UNVERIFIED
    assert "WARNING" in capsys.readouterr().err
    emb = marked_embeddings(4)
    assert aligner.apply(emb) is emb


def test_cache_row_for_locates_a_named_file(tmp_path):
    run_dir = tmp_path / "run"
    write_manifest(run_dir, CACHE_ORDER)
    split = split_frame(SPLIT_ORDER)
    aligner = EmbeddingAligner.for_run_dir(run_dir, split)
    assert aligner.cache_row_for(split, "c.txt") == 2
    assert aligner.cache_row_for(split, "d.txt") == 3


# --------------------------------------------------------------------------- #
# AlignmentResolver
# --------------------------------------------------------------------------- #


def test_resolver_loads_and_reorders(tmp_path):
    run_dir = tmp_path / "run"
    write_manifest(run_dir, CACHE_ORDER)
    emb_path = run_dir / "hidden_layer1_embeddings.npy"
    np.save(emb_path, marked_embeddings(4))

    resolver = AlignmentResolver(split_frame(SPLIT_ORDER))
    assert list(resolver.load(emb_path)[:, 0]) == [0.0, 1.0, 3.0, 2.0]
    assert "verified-permuted" in resolver.summary()


def test_resolver_reuses_one_aligner_per_directory(tmp_path):
    run_dir = tmp_path / "run"
    write_manifest(run_dir, CACHE_ORDER)
    for layer in (1, 2):
        np.save(run_dir / f"hidden_layer{layer}_embeddings.npy", marked_embeddings(4))

    resolver = AlignmentResolver(split_frame(SPLIT_ORDER))
    first = resolver.aligner_for(run_dir / "hidden_layer1_embeddings.npy")
    second = resolver.aligner_for(run_dir / "hidden_layer2_embeddings.npy")
    assert first is second


def test_resolver_summary_before_any_load(tmp_path):
    resolver = AlignmentResolver(split_frame(SPLIT_ORDER))
    assert "no embedding caches" in resolver.summary()
