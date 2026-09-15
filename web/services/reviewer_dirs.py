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
  is scored by, so the two numbers are comparable. They no longer share a ranked
  list: since issue #196 a reviewer directory is served unranked, in its own
  block, because a comparable number is not the same thing as a comparable kind
  of evidence and reviewers read a numbered slot as the model's answer;
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

Membership is exactly that record. The seed is member one, and the only ways in
are `add_reviewer_dir_member` (a `matched_rank` against a server-resolved
reviewer-directory candidate -- the pre-#196 route, still the meaning of every
row it wrote) and `FeedbackDB.insert_none_of_top_k`, where an evaluator typed
the directory's CCL key beside "None of the top N". Both are a human naming a
second document, so ``len(members) > 1`` still *is* "a human confirmed a second
document", and no extra bookkeeping is needed.

The similarity number has not gone away: `best_match_score` still reports the
best non-member score, and `has_potential_match` flags when it crosses the band.
That is the honest framing -- the model thinks something is related, a human has
not yet agreed -- and it is what lets the UI say so without claiming a match.
Above-band queries still get the directory as an unranked candidate, which is
the loop working; naming its key is what flips the badge.

Both are derived, never stored. A status column would need a scan of every
directory on every feedback write to stay honest, and would go stale the moment
a matrix is rebuilt.
"""

from __future__ import annotations

from web.bands import REVIEWER_DIR_MATCH_BAND
from web.models import (
    MAX_REVIEWER_CANDIDATES,
    CandidateFile,
    ReviewerDir,
    ReviewerDirCandidate,
    ReviewerDirStatus,
    SupportingMember,
)
from web.services.data_store import DataStore
from web.services.qq_matrix import QQMatrix, QQScore

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


def unranked_candidates_for_query(
    *,
    store: DataStore,
    records: list[dict],
    qq: QQMatrix | None,
    query_id: int,
) -> list[ReviewerDirCandidate]:
    """Reviewer directories scorable for `query_id`, best first and WITHOUT ranks.

    Issue #196 point 3, and the reason the old `candidates_for_query` is gone.
    These used to be returned as `Prediction`s at ranks anchored on
    ``MAX_MODEL_RANK + 1``, appended to the model's ten. Two things were wrong
    with that, and only one of them was cosmetic:

    * Reviewers read #11 and #12 as the model's answers. They are not. A
      labelled directory is ranked by the retrieval run; a reviewer directory is
      scored live from the query-query matrix against documents a colleague
      grouped by hand. Abigail's verdict on the pair sharing one numbered list:
      "the red button causes more confusion than the clarity we hoped it would
      provide".
    * A reviewer directory's rank was computed per request, and confirming one
      CHANGED it -- the query joined the directory, the directory stopped being
      offered to it, and everything below shifted up. The PDF packets already
      refused to print these ranks for exactly that reason (see
      `packet_dirs_for_query`), and the web list now agrees with them.

    What survives is the score, because a score is a property of the pair rather
    than a position in a list, and `dir_id`, which is the stable join key. The
    cap and the best-first order survive too: the list is reviewer-authored and
    permanent, so an unbounded tail of weak directories would accumulate on
    every query with no way to prune it.

    A directory is skipped when the query is already one of its members (a
    directory does not propose itself to its own seed) or when nothing about the
    pair is scorable.
    """
    if qq is None:
        return []

    scored: list[tuple[QQScore, dict]] = []
    for record in records:
        members = [int(m) for m in record.get("member_query_ids", [])]
        if int(query_id) in members:
            continue
        score = qq.score_with_support(int(query_id), members)
        if score is None:
            continue
        scored.append((score, record))

    scored.sort(key=lambda item: (-item[0].score, item[1]["dir_id"]))
    scored = scored[:MAX_REVIEWER_CANDIDATES]
    return [_to_candidate(store, record, score) for score, record in scored]


def _to_candidate(
    store: DataStore, record: dict, score: QQScore
) -> ReviewerDirCandidate:
    files = member_files(store, record)
    support = (
        SupportingMember(
            query_id=score.supporting_query_id,
            filename=store.file_id_to_filename.get(score.supporting_query_id),
            score=score.score,
        )
        if score.supporting_query_id is not None
        else None
    )
    preview = (
        store.unlabelled_texts.get(support.query_id, "")[:200]
        if support is not None and support.filename is not None
        else ""
    )
    return ReviewerDirCandidate(
        dir_id=record["dir_id"],
        label=record["label"],
        ccl_key=str(record.get("ccl_key") or ""),
        score=score.score,
        created_by=str(record["created_by"] or ""),
        seed_query_id=int(record["seed_query_id"]),
        member_query_ids=[int(m) for m in record.get("member_query_ids", [])],
        dir_files=[f.filename for f in files],
        candidate_files=files,
        preview_text=preview,
        supporting_member=support,
    )
