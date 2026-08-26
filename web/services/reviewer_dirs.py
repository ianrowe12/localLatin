"""Scoring and shaping of reviewer-created directories.

Kept out of the routers because three of them need the same answers: the
predictions endpoint merges these directories into a candidate list, the
reviewer_dirs endpoint reports them on their own, and the feedback endpoint has
to recognise one by name.

The model of the world is small:

* a reviewer directory is a set of unlabelled queries (its *members*), seeded
  by the query that had no labelled match;
* its score for some other query is the max cosine between that query and any
  member -- the same "max over the files it contains" rule a labelled directory
  is scored by, so the two numbers are comparable and can share a ranked list;
* its status is ``matched`` once *any* non-member query reaches
  ``REVIEWER_DIR_MATCH_BAND`` against it, and ``awaiting_match`` until then.

Status is derived, never stored. The alternative -- writing a status column and
flipping it -- would need a scan of every directory on every feedback write to
stay honest, and would go stale the moment a matrix is rebuilt. Deriving it
costs one row-max over a 2,238-wide float16 block.
"""

from __future__ import annotations

from web.bands import REVIEWER_DIR_MATCH_BAND
from web.models import (
    CandidateFile,
    CandidateSource,
    Prediction,
    ReviewerDir,
    ReviewerDirStatus,
)
from web.services.data_store import DataStore
from web.services.qq_matrix import QQMatrix

#: Prefix of every reviewer directory id. Disjoint from every labelled
#: directory name in the corpus, so `correct_dir` and the candidate-files route
#: can tell the two apart by name alone.
REVIEWER_DIR_PREFIX = "reviewer-dir-"


def is_reviewer_dir_id(dir_name: str) -> bool:
    return dir_name.startswith(REVIEWER_DIR_PREFIX)


def default_label(filename: str) -> str:
    """Label for a directory the reviewer did not name.

    The seed document's filename is the only thing known about a brand-new
    directory that is meaningful to a human scanning a candidate list.
    """
    stem = filename[:-4] if filename.endswith(".txt") else filename
    return f"New directory from {stem}"


def status_of(
    record: dict, qq: QQMatrix | None
) -> tuple[ReviewerDirStatus, float | None]:
    """(status, best non-member score) for one directory under one model.

    Without a matrix the status cannot be computed, and reporting ``matched``
    on no evidence would be worse than reporting the conservative answer: a
    model with no q-q matrix always sees ``awaiting_match`` and a null score.
    """
    if qq is None:
        return ReviewerDirStatus.AWAITING_MATCH, None
    best = qq.best_external_score(record.get("member_query_ids", []))
    status = (
        ReviewerDirStatus.MATCHED
        if best >= REVIEWER_DIR_MATCH_BAND
        else ReviewerDirStatus.AWAITING_MATCH
    )
    return status, best


def to_api(record: dict, qq: QQMatrix | None, model_slug: str) -> ReviewerDir:
    status, best = status_of(record, qq)
    return ReviewerDir(
        dir_id=record["dir_id"],
        label=record["label"],
        status=status,
        seed_query_id=int(record["seed_query_id"]),
        member_query_ids=list(record.get("member_query_ids", [])),
        created_at=str(record["created_at"]),
        created_by=str(record["created_by"] or ""),
        model_slug=model_slug or str(record.get("model_slug") or ""),
        variant=record.get("variant") or None,
        best_match_score=best,
    )


def member_files(store: DataStore, record: dict) -> list[CandidateFile]:
    """The member queries' texts, presented like a labelled dir's files."""
    files: list[CandidateFile] = []
    for query_id in record.get("member_query_ids", []):
        filename = store.file_id_to_filename.get(int(query_id))
        if filename is None:
            continue
        files.append(
            CandidateFile(
                filename=filename, text=store.unlabelled_texts.get(int(query_id), "")
            )
        )
    return files


def candidates_for_query(
    *,
    store: DataStore,
    records: list[dict],
    qq: QQMatrix | None,
    query_id: int,
    first_rank: int,
) -> list[Prediction]:
    """Reviewer directories scorable for `query_id`, ranked among themselves.

    Ranks start at `first_rank`, i.e. *after* the model's own candidates, and
    the model's ranks are left exactly as the retrieval CSV produced them. That
    is what keeps every feedback row ever written -- which records a rank -- a
    valid reference to the candidate the reviewer actually chose.

    A directory is skipped when the query is already one of its members (a
    directory does not propose itself to its own seed) or when nothing about
    the pair is scorable.
    """
    if qq is None:
        return []

    scored: list[tuple[float, dict]] = []
    for record in records:
        members = [int(m) for m in record.get("member_query_ids", [])]
        if int(query_id) in members:
            continue
        score = qq.score(int(query_id), members)
        if score is None:
            continue
        scored.append((score, record))

    scored.sort(key=lambda item: (-item[0], item[1]["dir_id"]))

    predictions: list[Prediction] = []
    for offset, (score, record) in enumerate(scored):
        files = member_files(store, record)
        predictions.append(
            Prediction(
                rank=first_rank + offset,
                dir_name=record["dir_id"],
                score=score,
                dir_files=[f.filename for f in files],
                preview_text=files[0].text[:200] if files else "",
                candidate_files=files,
                source=CandidateSource.REVIEWER,
                label=record["label"],
                created_by=str(record["created_by"] or ""),
                seed_query_id=int(record["seed_query_id"]),
            )
        )
    return predictions
