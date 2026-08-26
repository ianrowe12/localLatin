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
* its status is ``matched`` once **a human has actually filed a second document
  into it**, and ``awaiting_match`` until then.

WHY STATUS IS NOT SIMILARITY. The first version called a directory matched as
soon as any non-member query scored above the band. The independent review
measured what that means on the real corpus: 57-70% of directories, depending
on model, would show the green "matched" pill at the moment of creation, before
any human confirmed anything. That guts the badge -- "Awaiting future match"
would be the rare state, and the pill would be reporting a property of the
embedding space rather than a scholarly finding. The PI's intent behind the
badge is a discovery event, so the trigger is now a discovery event: a feedback
row in which some later reviewer chose this directory for another query.

Membership is exactly that record. The seed is member one, and the only other
way in is `add_reviewer_dir_member`, which runs solely when a reviewer submits
`matched_rank` against a server-resolved reviewer-directory candidate. So
``len(members) > 1`` *is* "a human confirmed a second document", and no extra
bookkeeping is needed.

The similarity number has not gone away: `best_match_score` still reports the
best non-member score, and `has_potential_match` flags when it crosses the band.
That is the honest framing -- the model thinks something is related, a human has
not yet agreed -- and it is what lets the UI say so without claiming a match.
Above-band queries still get the directory as a candidate card, which is the
loop working; confirming from that card is what flips the badge.

Both are derived, never stored. A status column would need a scan of every
directory on every feedback write to stay honest, and would go stale the moment
a matrix is rebuilt.
"""

from __future__ import annotations

from web.bands import REVIEWER_DIR_MATCH_BAND
from web.models import (
    MAX_MODEL_RANK,
    MAX_REVIEWER_CANDIDATES,
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

#: How many directories one reviewer account may create. A guard against a
#: runaway client, not a scholarly limit; every one of these is permanent.
MAX_REVIEWER_DIRS_PER_ACCOUNT = 50


def is_reviewer_dir_id(dir_name: str) -> bool:
    return dir_name.startswith(REVIEWER_DIR_PREFIX)


def default_label(filename: str) -> str:
    """Label for a directory the reviewer did not name.

    The seed document's filename is the only thing known about a brand-new
    directory that is meaningful to a human scanning a candidate list.
    """
    stem = filename[:-4] if filename.endswith(".txt") else filename
    return f"New directory from {stem}"


def status_of(record: dict) -> ReviewerDirStatus:
    """``matched`` once a human filed a second document in; else awaiting.

    Deliberately independent of the q-q matrix and of the model being viewed:
    a confirmation is a scholarly fact, so a directory must not appear matched
    under one model and awaiting under another.
    """
    members = record.get("member_query_ids", [])
    return (
        ReviewerDirStatus.MATCHED
        if len(members) > 1
        else ReviewerDirStatus.AWAITING_MATCH
    )


def to_api(record: dict, qq: QQMatrix | None, model_slug: str) -> ReviewerDir:
    # The similarity number is reported but does not decide the status. Without
    # a matrix it is simply unknown, which is not the same as zero.
    best = (
        qq.best_external_score(record.get("member_query_ids", []))
        if qq is not None
        else None
    )
    return ReviewerDir(
        dir_id=record["dir_id"],
        label=record["label"],
        status=status_of(record),
        seed_query_id=int(record["seed_query_id"]),
        member_query_ids=list(record.get("member_query_ids", [])),
        created_at=str(record["created_at"]),
        created_by=str(record["created_by"] or ""),
        model_slug=model_slug or str(record.get("model_slug") or ""),
        variant=record.get("variant") or None,
        best_match_score=best,
        # "The model sees something related, nobody has confirmed it yet."
        has_potential_match=best is not None and best >= REVIEWER_DIR_MATCH_BAND,
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


def packet_dirs_for_query(
    *,
    store: DataStore,
    records: list[dict],
    qq: QQMatrix | None,
    query_id: int,
) -> list[dict]:
    """Reviewer directories a packet should document for this query. NO RANKS.

    Two groups, both unranked:

    * ``filed_into`` -- the query is a member, so it is no longer offered them;
    * ``offered``    -- currently scorable candidates for it.

    RANKS ARE DELIBERATELY ABSENT, and that is the whole fix. A reviewer
    directory's rank is computed live, and confirming one *changes* that
    ranking: the query joins the directory, the directory stops being offered to
    it, and everything below shifts up. A packet built after the confirmation
    therefore cannot reproduce the ranks the reviewer saw.

    Measured on a real packet rather than reasoned about: a reviewer confirmed
    rank 12, and the regenerated list put a different directory at 12, on the
    same page as a feedback row reading "rank 12". Renumbering a concatenated
    list was one way to get that wrong; recomputing the live list is another,
    and the second survived the first fix.

    Printing no rank removes the class of error. `dir_id` is the join key
    instead, and it is the same string the Reviewer Outcomes section prints
    beside the recorded rank -- both sides read the stored, server-resolved
    `correct_dir`, so they cannot disagree.

    `score` is still reported for an offered directory: it is a property of the
    pair, not a position in a list, so it does not drift the same way.
    """
    out: list[dict] = []
    for record in records:
        members = [int(m) for m in record.get("member_query_ids", [])]
        is_member = int(query_id) in members
        score = (
            None
            if is_member
            else (qq.score(int(query_id), members) if qq is not None else None)
        )
        if not is_member and score is None:
            continue  # neither offered to this query nor filed into by it
        files = member_files(store, record)
        out.append(
            {
                "dir_id": record["dir_id"],
                "label": record["label"],
                "created_by": str(record["created_by"] or ""),
                "seed_query_id": int(record["seed_query_id"]),
                "group": "filed_into" if is_member else "offered",
                "is_seed": int(record["seed_query_id"]) == int(query_id),
                "score": score,
                "member_query_ids": members,
                "candidate_files": [
                    {"filename": f.filename, "text": f.text} for f in files
                ],
            }
        )
    # Filed-into first (that is the answer of record), then best-scoring offers.
    out.sort(key=lambda e: (e["group"] != "filed_into", -(e["score"] or 0.0)))
    return out


def candidates_for_query(
    *,
    store: DataStore,
    records: list[dict],
    qq: QQMatrix | None,
    query_id: int,
) -> list[Prediction]:
    """Reviewer directories scorable for `query_id`, ranked among themselves.

    Ranks are anchored at ``MAX_MODEL_RANK + 1`` -- a fixed 11, not "one past
    however many model candidates were returned". The retrieval CSVs always
    rank ten labelled directories, so 11 is where reviewer candidates begin
    regardless of the caller's `top_k`. Anchoring rather than offsetting is
    what makes a recorded rank mean the same thing in every response: the
    earlier offset-based version disagreed with itself between
    `get_predictions` (which counted the *sliced* list) and `get_candidates`
    (which counted the unsliced row), so at `top_k=5` a reviewer card served at
    rank 6 resolved to the labelled directory at model rank 6. Feedback records
    that rank, so the ambiguity was a data-integrity problem, not a display one.

    Model ranks themselves are untouched, which is what keeps every feedback row
    ever written a valid reference to the candidate its reviewer chose.

    A directory is skipped when the query is already one of its members (a
    directory does not propose itself to its own seed) or when nothing about
    the pair is scorable. At most `MAX_REVIEWER_CANDIDATES` are returned, best
    first: the list is reviewer-authored and permanent, so an unbounded tail of
    weak directories would accumulate on every query with no way to prune it.
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
    scored = scored[:MAX_REVIEWER_CANDIDATES]

    first_rank = MAX_MODEL_RANK + 1
    return [
        _to_prediction(store, record, score, first_rank + offset)
        for offset, (score, record) in enumerate(scored)
    ]


def _to_prediction(
    store: DataStore, record: dict, score: float, rank: int
) -> Prediction:
    files = member_files(store, record)
    return Prediction(
        rank=rank,
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
