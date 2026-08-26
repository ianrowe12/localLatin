from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from web.dependencies import get_current_user, get_db, get_store, require_pi_admin
from web.exceptions import InvalidModelError, QueryNotFoundError, VariantUnavailableError
from web.models import FeedbackCreate, FeedbackEntry, PredictionVariant, UserPublic
from web.routers.predictions import resolve_variant
from web.services.data_store import DataStore, normalize_slug
from web.services.feedback_db import FeedbackDB

router = APIRouter(prefix="/api", tags=["feedback"])


async def _check_model_variant(store: DataStore, slug: str, variant: str) -> None:
    if not await store.ensure_variant_async(variant):
        raise VariantUnavailableError(variant, store.variants)
    if (slug, variant) not in store.predictions:
        raise InvalidModelError(slug, store.model_slugs)


@router.post("/feedback", response_model=FeedbackEntry, status_code=201)
async def create_feedback(
    body: FeedbackCreate,
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> FeedbackEntry:
    if body.query_id not in store.file_id_to_filename:
        raise QueryNotFoundError(body.query_id)

    slug = normalize_slug(body.model_slug)
    variant = resolve_variant(store, body.variant)
    await _check_model_variant(store, slug, variant)

    correct_dir = body.correct_dir
    if body.selected_ranks:
        # Legacy consumers still read correct_rank/correct_dir as a single choice.
        # For multi-select submissions, the first selected rank is the canonical
        # legacy choice while selected_ranks carries the full reviewer answer.
        correct_dir = _dir_for_rank(
            store, slug, variant, body.query_id, body.selected_ranks[0]
        )

    row = await db.insert(
        query_id=body.query_id,
        model_slug=slug,
        variant=variant,
        outcome=body.outcome.value,
        correct_rank=body.correct_rank,
        correct_dir=correct_dir,
        notes=body.notes,
        reviewer=current_user.display_name,
        reviewer_account_id=current_user.id,
        selected_ranks=body.selected_ranks,
    )
    return FeedbackEntry(**row)


@router.get("/feedback/latest", response_model=FeedbackEntry | None)
async def latest_feedback(
    query_id: int,
    model: str,
    variant: PredictionVariant | None = Query(
        None,
        description=(
            "Only prefill from feedback saved for this variant. "
            "Defaults to the deployment's configured default variant."
        ),
    ),
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    _current_user: UserPublic = Depends(get_current_user),
) -> FeedbackEntry | None:
    """Most recent assessment recorded for this query, from any reviewer.

    Shared across the team by design (issue #96). Sign-in is still required --
    the dependency is what enforces that -- but the caller's identity no longer
    filters the result, so the returned row carries `reviewer` and
    `reviewer_username` for the UI to attribute it to its author. Saving never
    edits this row: POST /api/feedback appends a new one under whoever is
    signed in.
    """
    if query_id not in store.file_id_to_filename:
        raise QueryNotFoundError(query_id)

    slug = normalize_slug(model)
    resolved = resolve_variant(store, variant)
    await _check_model_variant(store, slug, resolved)

    row = await db.get_latest_feedback(
        query_id=query_id,
        model_slug=slug,
        variant=resolved,
    )
    return FeedbackEntry(**row) if row is not None else None


@router.get("/feedback/export")
async def export_feedback(
    model: str | None = None,
    variant: PredictionVariant | None = Query(
        None, description="Restrict the export to one prediction variant"
    ),
    reviewer: str | None = None,
    outcome: str | None = Query(
        None,
        pattern="^(matched_rank|none_of_top_k|skipped|legacy_unresolved)$",
    ),
    status: str | None = Query(
        None,
        pattern="^(reviewed|skipped|needs_attention)$",
    ),
    date_from: str | None = None,
    date_to: str | None = None,
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(require_pi_admin),
) -> PlainTextResponse:
    slug = normalize_slug(model) if model else None
    csv_data = await db.export_csv(
        model=slug,
        variant=variant.value if variant else None,
        reviewer=reviewer,
        outcome=outcome,
        status=status,
        date_from=date_from,
        date_to=date_to,
        filename_by_query=store.file_id_to_filename,
    )
    return PlainTextResponse(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=feedback_export.csv"},
    )


def _dir_for_rank(
    store: DataStore,
    model_slug: str,
    variant: str,
    query_id: int,
    rank: int,
) -> str | None:
    for row in store.predictions.get((model_slug, variant), []):
        if row["file_id"] != query_id:
            continue
        for prediction in row["predictions"]:
            if prediction["rank"] == rank:
                return prediction["dir_name"]
    return None
