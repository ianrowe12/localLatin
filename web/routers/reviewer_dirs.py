"""Reviewer-created directories (issue #95).

The API contract fixed with issue #94:

    POST /api/reviewer_dirs {query_file_id, label?}
      -> 201 {dir_id, label, status: 'awaiting_match', ...}

The 201 always reports ``awaiting_match``, and that is now true by construction
rather than by assertion: status is "a human has filed a second document into
this directory", a new directory has exactly one member, so there is no path by
which creation can report anything else. (An earlier version derived status from
q-q similarity and would have reported ``matched`` at creation for 57-70% of
directories, depending on model -- see services/reviewer_dirs.py.)

AUTH: any signed-in, approved reviewer. ``get_current_user`` already rejects
pending, rejected, inactive and unauthenticated callers, so no extra role gate
is needed -- and adding one would wrongly exclude reviewers, who are the users
this feature is for.

REFUSED CREATIONS, all because nothing can ever remove a directory:
  409  the query already seeds one (double-click, stale form)
  422  the seed is a guard-excluded document, so it could never be matched
  429  this account is at MAX_REVIEWER_DIRS_PER_ACCOUNT
  400  the named model does not exist

APPEND-ONLY: creation writes to ``reviewer_dirs`` and ``reviewer_dir_members``
and to nothing else. No feedback row is read, written, updated or deleted on
this path.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from web.dependencies import get_current_user, get_db, get_store
from web.exceptions import InvalidModelError, QueryNotFoundError
from web.models import (
    CandidateFile,
    ReviewerDir,
    ReviewerDirCreate,
    UserPublic,
)
from web.routers.predictions import resolve_variant
from web.services import reviewer_dirs as svc
from web.services.data_store import DataStore, normalize_slug
from web.services.feedback_db import FeedbackDB

router = APIRouter(prefix="/api", tags=["reviewer_dirs"])


def _resolve_model(store: DataStore, model: str | None) -> str:
    """The model a directory is reported under.

    Omitting `model` falls back to the first served model: it only selects which
    q-q matrix supplies the informational score, and a client that does not care
    should not have to name one. A model that *is* named must exist, exactly as
    the predictions and feedback routes require -- an unknown slug used to
    return 201 with a null score, which looks like success.
    """
    if not model:
        return store.model_slugs[0] if store.model_slugs else ""
    slug = normalize_slug(model)
    if slug not in store.model_slugs:
        raise InvalidModelError(slug, store.model_slugs)
    return slug


@router.post("/reviewer_dirs", response_model=ReviewerDir, status_code=201)
async def create_reviewer_dir(
    body: ReviewerDirCreate,
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> ReviewerDir:
    query_id = body.query_file_id
    if query_id not in store.file_id_to_filename:
        raise QueryNotFoundError(query_id)

    slug = _resolve_model(store, body.model_slug)
    variant = resolve_variant(store, body.variant)
    qq = await store.ensure_qq_async(slug)

    # One directory per seed document. A second create on the same query is a
    # double-click or a stale form, not a second discovery -- and since nothing
    # can remove a directory, the duplicate would be a permanent extra card on
    # every query in the corpus. The reviewer gets the existing one back.
    existing = await db.get_reviewer_dir_by_seed(query_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Query {query_id} already seeds reviewer directory "
                f"'{existing['dir_id']}' ({existing['label']})."
            ),
        )

    # A query the degenerate-file guard excluded is unscorable in both
    # directions forever, so a directory seeded there could never appear as a
    # candidate and never leave 'awaiting_match': a permanent dead end with a
    # permanent badge. Refuse rather than create it.
    if qq is not None and qq.row_of(query_id) is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Query {query_id} ({store.file_id_to_filename[query_id]}) has no "
                "usable embedding (empty or whitespace-only source), so a directory "
                "seeded here could never be matched."
            ),
        )

    created_count = await db.count_reviewer_dirs_by_account(current_user.id)
    if created_count >= svc.MAX_REVIEWER_DIRS_PER_ACCOUNT:
        raise HTTPException(
            status_code=429,
            detail=(
                f"You have created {created_count} reviewer directories, the "
                f"maximum is {svc.MAX_REVIEWER_DIRS_PER_ACCOUNT}. Ask a PI/admin "
                "if you genuinely need more."
            ),
        )

    label = (body.label or "").strip()
    if not label:
        label = svc.default_label(store.file_id_to_filename[query_id])

    record = await db.create_reviewer_dir(
        label=label,
        seed_query_id=query_id,
        model_slug=slug,
        variant=variant,
        created_by=current_user.display_name,
        created_by_account_id=current_user.id,
    )
    full = await db.get_reviewer_dir(record["dir_id"])
    assert full is not None
    # Always 'awaiting_match' here by construction: the directory has exactly
    # one member, and only a human confirmation adds a second.
    return svc.to_api(full, qq, slug)


@router.get("/reviewer_dirs", response_model=list[ReviewerDir])
async def list_reviewer_dirs(
    model: str | None = Query(None, description="Model slug to score status under"),
    seed_query_id: int | None = Query(
        None, description="Only directories seeded by this query"
    ),
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> list[ReviewerDir]:
    del current_user
    slug = _resolve_model(store, model)
    qq = await store.ensure_qq_async(slug)
    records = await db.list_reviewer_dirs()
    if seed_query_id is not None:
        records = [r for r in records if int(r["seed_query_id"]) == seed_query_id]
    return [svc.to_api(record, qq, slug) for record in records]


@router.get("/reviewer_dirs/{dir_id}", response_model=ReviewerDir)
async def get_reviewer_dir(
    dir_id: str,
    model: str | None = Query(None),
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> ReviewerDir:
    del current_user
    record = await db.get_reviewer_dir(dir_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Reviewer directory not found")
    slug = _resolve_model(store, model)
    qq = await store.ensure_qq_async(slug)
    return svc.to_api(record, qq, slug)


@router.get("/reviewer_dirs/{dir_id}/files", response_model=list[CandidateFile])
async def get_reviewer_dir_files(
    dir_id: str,
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> list[CandidateFile]:
    del current_user
    record = await db.get_reviewer_dir(dir_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Reviewer directory not found")
    return svc.member_files(store, record)
