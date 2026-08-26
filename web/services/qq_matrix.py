"""Query-query cosine matrices, and the scoring built on top of them.

A reviewer-created directory's members are unlabelled *queries*. The prediction
CSVs only hold query -> labelled-directory scores, so scoring such a directory
for a new query needs the other half of the similarity space: query -> query.

``scripts/resubmit/build_qq_matrices.py`` writes one ``qq_sim_<slug>.npz`` per
model at exactly the configuration the deployed ``sif_abtt`` predictions were
computed with (same bases, same layer, same train-fit cleaner -- the script
verifies that against the deployed CSV before writing). This module is the read
side: it loads a matrix and answers "how similar is query q to this set of
member queries", on the same similarity scale the prediction cards already show.

Loading is lazy and per model, mirroring how the store defers non-default
prediction variants: a matrix is ~10 MB and most sessions touch one model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QQMatrix:
    """One model's 2,238 x 2,238 query-query cosine matrix.

    ``sim`` is float16 as stored. Rounding costs ~1e-3 of absolute precision on
    a [-1, 1] cosine, orders of magnitude below the 0.5/0.7 confidence bands, so
    scores are read straight out and cast to float rather than kept in a second
    float32 copy of the same 10 MB.
    """

    sim: np.ndarray
    file_ids: np.ndarray
    excluded: np.ndarray
    meta: dict = field(default_factory=dict)
    _index: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._index:
            self._index = {int(fid): i for i, fid in enumerate(self.file_ids)}

    def row_of(self, file_id: int) -> int | None:
        """Matrix row for a query file_id, or None if it has no usable row.

        Degenerate queries (empty or whitespace-only source files) are excluded
        by the build script and their row and column are zeroed. They are
        reported as having no row at all so they can neither seed a reviewer
        directory nor be scored against one -- an empty file matching an empty
        file at cosine 1.0 is exactly the artefact issue #66 removed from the
        labelled path.
        """
        idx = self._index.get(int(file_id))
        if idx is None or bool(self.excluded[idx]):
            return None
        return idx

    def score(self, query_file_id: int, member_file_ids: list[int]) -> float | None:
        """Max cosine between one query and a set of member queries.

        ``max`` over members mirrors how a labelled directory is scored in
        ``run_resubmit_unlabelled_retrieval.py`` (max cosine over the files it
        contains), so a reviewer directory's number is on the same scale as the
        model's own candidates and the two can share a ranked list.

        The query is never compared with itself: a directory seeded by query q
        would otherwise score a perfect 1.0 when q is the one being reviewed.
        Returns None when no member is usable.
        """
        row = self.row_of(query_file_id)
        if row is None:
            return None
        cols = [
            idx
            for member in member_file_ids
            if int(member) != int(query_file_id)
            and (idx := self.row_of(int(member))) is not None
        ]
        if not cols:
            return None
        return float(np.max(self.sim[row, cols].astype(np.float32)))

    def best_external_score(self, member_file_ids: list[int]) -> float:
        """Best score any *non-member* query achieves against these members.

        This is what decides whether a reviewer directory is still awaiting a
        match: the directory is matched as soon as some other query in the
        corpus reaches the band. Computed live rather than stored, so the answer
        is always consistent with the matrix currently deployed and nothing has
        to be mutated when it changes.
        """
        rows = [
            idx
            for member in member_file_ids
            if (idx := self.row_of(int(member))) is not None
        ]
        if not rows:
            return 0.0
        block = self.sim[rows, :].astype(np.float32)
        per_query = block.max(axis=0)
        # Members themselves, and guard-excluded queries, are not candidates.
        per_query[rows] = -np.inf
        per_query[self.excluded] = -np.inf
        best = float(per_query.max())
        return best if np.isfinite(best) else 0.0


def load_qq_matrix(path: Path) -> QQMatrix:
    """Read one ``qq_sim_<slug>.npz``."""
    with np.load(path, allow_pickle=False) as data:
        sim = data["sim"]
        file_ids = data["file_ids"]
        excluded = (
            data["excluded"]
            if "excluded" in data.files
            else np.zeros(len(file_ids), dtype=bool)
        )
        meta: dict = {}
        if "meta" in data.files:
            try:
                meta = json.loads(str(data["meta"]))
            except (TypeError, ValueError):
                logger.warning("Unreadable meta in %s; continuing without it", path)
    if sim.shape[0] != sim.shape[1] or sim.shape[0] != len(file_ids):
        raise ValueError(
            f"{path}: sim {sim.shape} does not match {len(file_ids)} file_ids"
        )
    return QQMatrix(sim=sim, file_ids=file_ids, excluded=excluded.astype(bool), meta=meta)
