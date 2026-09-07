"""Align cached embedding matrices to a split CSV by filename.

Why this exists
---------------

A cached ``*_embeddings.npy`` holds one row per corpus file, in the order the
extractor walked the corpus. That order is frozen the moment the matrix is
written. The split CSV, on the other hand, is rebuilt from the corpus whenever
the labels change, and its rows are sorted by ``(folder_id, filename)``. So a
label correction that moves a file from one directory to another permutes the
split rows while leaving the cached matrix untouched.

Consumers used to pair the two by row position, guarded only by a row-count
check. A count check cannot see a permutation: after the benchmark v1 label
corrections (issue #112) seventeen rows changed position and every count stayed
the same, so a naive re-run would have scored seventeen vectors against the
wrong labels without a word of complaint.

Everything here aligns by filename instead. Filenames are unique across the
corpus, so the mapping is a bijection, and the bijection is checked rather than
assumed. The embeddings on disk are never rewritten, which keeps them
bit-identical across relabellings and makes the whole thing idempotent.

Row-order manifest
------------------

The extractor already writes a ``meta.csv`` into each run directory recording
the corpus order it used. That file is the manifest. A ``row_order.csv`` at a
bases root serves as a fallback for caches whose run directories predate it.
Either file needs a ``path`` column (basenames are taken) or a ``filename``
column, one row per matrix row, in cache order.

Usage
-----

    from embedding_alignment import AlignmentResolver

    resolver = AlignmentResolver(split_meta)
    ...
    emb = resolver.load(emb_path)          # loads and reorders
    print(resolver.summary())              # once, after the loop
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

RUN_DIR_MANIFEST = "meta.csv"
ROOT_MANIFEST = "row_order.csv"

STATUS_IDENTITY = "verified-identity"
STATUS_PERMUTED = "verified-permuted"
STATUS_UNVERIFIED = "unverified"


class AlignmentError(RuntimeError):
    """The cache manifest and the split describe different file sets."""


def find_manifest(run_dir, search_roots: Sequence[Path] = ()) -> Optional[Path]:
    """Return the row-order manifest for a run directory, or None if there is none.

    Looks for the extractor's own ``meta.csv`` first, then for a ``row_order.csv``
    in each ancestor of the run directory (nearest first), which is where a bases
    root keeps the frozen order for caches written before per-run manifests.
    """
    run_dir = Path(run_dir)
    own = run_dir / RUN_DIR_MANIFEST
    if own.exists():
        return own
    for parent in list(run_dir.parents) + [Path(r) for r in search_roots]:
        candidate = parent / ROOT_MANIFEST
        if candidate.exists():
            return candidate
    return None


def load_row_order(path) -> List[str]:
    """Read a row-order manifest and return the basenames in cache row order."""
    df = pd.read_csv(path)
    if "path" in df.columns:
        names = [Path(str(p)).name for p in df["path"].tolist()]
    elif "filename" in df.columns:
        names = [str(f) for f in df["filename"].tolist()]
    else:
        raise AlignmentError(
            f"{path} has neither a 'path' nor a 'filename' column; the cached row "
            "order cannot be recovered from it."
        )
    seen: Dict[str, int] = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    duplicates = sorted(n for n, c in seen.items() if c > 1)
    if duplicates:
        raise AlignmentError(
            f"{path} lists {len(duplicates)} duplicate filename(s), e.g. "
            f"{duplicates[:5]}. Row order can only be recovered from unique names."
        )
    return names


def build_permutation(
    cache_order: Sequence[str],
    split_filenames: Sequence[str],
) -> Optional[np.ndarray]:
    """Return ``perm`` such that ``emb[perm][i]`` is the vector for split row ``i``.

    Returns None when the two orders already agree, so callers can skip the copy.
    Raises AlignmentError when the two describe different file sets, which means
    the cache is stale rather than merely permuted.
    """
    cache_order = [str(n) for n in cache_order]
    split_filenames = [str(f) for f in split_filenames]

    if len(cache_order) != len(split_filenames):
        raise AlignmentError(
            f"cache manifest has {len(cache_order)} rows but the split has "
            f"{len(split_filenames)}; the cache is stale, not permuted."
        )

    position = {name: i for i, name in enumerate(cache_order)}
    if len(position) != len(cache_order):
        raise AlignmentError("cache manifest has duplicate filenames")

    split_set = set(split_filenames)
    if len(split_set) != len(split_filenames):
        counts: Dict[str, int] = {}
        for f in split_filenames:
            counts[f] = counts.get(f, 0) + 1
        dupes = sorted(f for f, c in counts.items() if c > 1)
        raise AlignmentError(
            f"the split has {len(dupes)} duplicate filename(s), e.g. {dupes[:5]}; "
            "filename alignment needs them to be unique."
        )

    cache_set = set(position)
    if cache_set != split_set:
        only_cache = sorted(cache_set - split_set)
        only_split = sorted(split_set - cache_set)
        raise AlignmentError(
            "cache manifest and split describe different files: "
            f"{len(only_cache)} only in the cache (e.g. {only_cache[:5]}), "
            f"{len(only_split)} only in the split (e.g. {only_split[:5]}). "
            "Re-extract the embeddings."
        )

    if cache_order == split_filenames:
        return None
    return np.array([position[name] for name in split_filenames], dtype=np.int64)


@dataclass
class EmbeddingAligner:
    """Reorders cached embedding rows into split-CSV row order."""

    perm: Optional[np.ndarray]
    n_rows: int
    status: str
    manifest_path: Optional[Path] = None

    @classmethod
    def for_run_dir(
        cls,
        run_dir,
        split_meta: pd.DataFrame,
        search_roots: Sequence[Path] = (),
    ) -> "EmbeddingAligner":
        """Build an aligner for one embedding run directory.

        When no manifest is found the aligner falls back to positional alignment
        and says so loudly: that is the old, unverifiable behaviour, kept only so
        caches predating the manifest still run.
        """
        path = find_manifest(run_dir, search_roots)
        if path is None:
            print(
                f"WARNING: no row-order manifest for {run_dir}. Falling back to "
                "positional alignment, which cannot detect a relabelled corpus.",
                file=sys.stderr,
            )
            return cls(perm=None, n_rows=len(split_meta), status=STATUS_UNVERIFIED)

        perm = build_permutation(
            load_row_order(path), split_meta["filename"].tolist()
        )
        status = STATUS_IDENTITY if perm is None else STATUS_PERMUTED
        return cls(perm=perm, n_rows=len(split_meta), status=status, manifest_path=path)

    @property
    def n_moved(self) -> int:
        """How many rows the permutation actually moves."""
        if self.perm is None:
            return 0
        return int((self.perm != np.arange(len(self.perm))).sum())

    def describe(self) -> str:
        if self.status == STATUS_UNVERIFIED:
            return f"{self.status} (positional, {self.n_rows} rows)"
        return (
            f"{self.status} ({self.n_rows} rows, {self.n_moved} moved, "
            f"manifest {self.manifest_path})"
        )

    def apply(self, emb: np.ndarray) -> np.ndarray:
        """Return ``emb`` with rows in split order, validating the row count."""
        if emb.shape[0] != self.n_rows:
            raise AlignmentError(
                f"embedding matrix has {emb.shape[0]} rows but the split has "
                f"{self.n_rows}."
            )
        if self.perm is None:
            return emb
        return emb[self.perm]

    def cache_row_for(self, split_meta: pd.DataFrame, filename: str) -> int:
        """Cache row index holding the vector for ``filename``. For spot checks."""
        matches = np.flatnonzero(split_meta["filename"].to_numpy() == filename)
        if len(matches) != 1:
            raise AlignmentError(f"{filename!r} matches {len(matches)} split rows")
        split_row = int(matches[0])
        return split_row if self.perm is None else int(self.perm[split_row])


class AlignmentResolver:
    """Per-run-directory aligners, built once and reused across layers.

    Scripts hold one resolver for the whole run and call :meth:`load` in place of
    ``np.load``. It keeps one aligner per run directory, so a cache re-extracted
    on its own is still handled correctly.
    """

    def __init__(self, split_meta: pd.DataFrame, search_roots: Sequence[Path] = ()):
        self.split_meta = split_meta
        self.search_roots = tuple(Path(r) for r in search_roots)
        self._by_dir: Dict[Path, EmbeddingAligner] = {}

    def aligner_for(self, emb_path) -> EmbeddingAligner:
        run_dir = Path(emb_path).parent
        if run_dir not in self._by_dir:
            aligner = EmbeddingAligner.for_run_dir(
                run_dir, self.split_meta, self.search_roots
            )
            print(f"  [alignment] {run_dir}: {aligner.describe()}", flush=True)
            self._by_dir[run_dir] = aligner
        return self._by_dir[run_dir]

    def load(self, emb_path) -> np.ndarray:
        """``np.load`` the matrix and reorder its rows into split order."""
        return self.aligner_for(emb_path).apply(np.load(emb_path))

    def summary(self) -> str:
        if not self._by_dir:
            return "row alignment: no embedding caches loaded"
        statuses: Dict[str, int] = {}
        moved = set()
        for aligner in self._by_dir.values():
            statuses[aligner.status] = statuses.get(aligner.status, 0) + 1
            moved.add(aligner.n_moved)
        parts = ", ".join(f"{n} {s}" for s, n in sorted(statuses.items()))
        return (
            f"row alignment: {len(self._by_dir)} cache dir(s) [{parts}]; "
            f"rows moved per cache: {sorted(moved)}"
        )
