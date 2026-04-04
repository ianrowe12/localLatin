from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from web.dependencies import get_db, get_store
from web.exceptions import InvalidModelError, QueryNotFoundError
from web.models import FeedbackCreate, FeedbackEntry
from web.services.data_store import DataStore, normalize_slug
from web.services.feedback_db import FeedbackDB

router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackEntry, status_code=201)
async def create_feedback(
    body: FeedbackCreate,
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
) -> FeedbackEntry:
    if body.query_id not in store.file_id_to_filename:
        raise QueryNotFoundError(body.query_id)

    slug = normalize_slug(body.model_slug)
    if slug not in store.predictions:
        raise InvalidModelError(slug, store.model_slugs)

    row = await db.insert(
        query_id=body.query_id,
        model_slug=slug,
        correct_rank=body.correct_rank,
        correct_dir=body.correct_dir,
        notes=body.notes,
        reviewer=body.reviewer,
    )
    return FeedbackEntry(**row)


@router.get("/feedback/export")
async def export_feedback(
    model: str | None = None,
    reviewer: str | None = None,
    db: FeedbackDB = Depends(get_db),
) -> PlainTextResponse:
    slug = normalize_slug(model) if model else None
    csv_data = await db.export_csv(model=slug, reviewer=reviewer)
    return PlainTextResponse(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=feedback_export.csv"},
    )
