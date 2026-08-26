"""Reviewer-created directories (issue #95).

The API contract fixed with issue #94:

    POST /api/reviewer_dirs {query_file_id, label?}
      -> 201 {dir_id, label, status: 'awaiting_match', ...}

One deliberate generalisation of that contract: the 201 reports the *computed*
status rather than the literal string ``awaiting_match``. It is
``awaiting_match`` in the ordinary case, and the only way it comes back
``matched`` is that some other unlabelled query already scores at or above the
band against the seed -- which is precisely the discovery this feature exists
to surface, so hiding it behind a hardcoded initial status would be a bug.
Everything downstream reads ``status``, never assumes it.

AUTH: any signed-in, approved reviewer. ``get_current_user`` already rejects
pending, rejected, inactive and unauthenticated callers, so no extra role gate
is needed -- and adding one would wrongly exclude reviewers, who are the users
this feature is for.

APPEND-ONLY: creation writes to ``reviewer_dirs`` and ``reviewer_dir_members``
and to nothing else. No feedback row is read, written, updated or deleted on
this path.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from web.dependencies import get_current_user, get_db, get_store
from web.exceptions import QueryNotFoundError
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

    Defaults to the first served model rather than erroring: `model` only
    selects which q-q matrix scores the status, and a client that does not care
    should not have to name one.
    """
    if model:
        return normalize_slug(model)
    return store.model_slugs[0] if store.model_slugs else ""


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

    label = (body.label or "").strip()
    if not label:
        label = svc.default_label(store.file_id_to_filename[query_id])

    slug = _resolve_model(store, body.model_slug)
    variant = resolve_variant(store, body.variant)

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
    qq = await store.ensure_qq_async(slug)
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
