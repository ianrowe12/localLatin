from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from web.dependencies import get_current_user, get_db, get_store, require_pi_admin
from web.exceptions import InvalidModelError, QueryNotFoundError, VariantUnavailableError
from web.models import (
    FeedbackCreate,
    FeedbackEntry,
    FeedbackOutcome,
    PredictionVariant,
    UserPublic,
)
from web.routers.predictions import resolve_variant
from web.services import reviewer_dirs as reviewer_dirs_svc
from web.services.data_store import DataStore, normalize_slug
from web.services.feedback_db import FeedbackDB

logger = logging.getLogger(__name__)
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

    # `correct_dir` is ALWAYS resolved server-side from the rank, never taken
    # from the request body. Trusting the body let a client file a query into an
    # arbitrary directory: posting {correct_rank: 1, correct_dir:
    # "reviewer-dir-X"} appended query 900 to a reviewer directory it had never
    # been offered, permanently changing the score that directory shows to all
    # 2,238 queries, and a nonexistent id wrote an unreachable orphan row. A
    # stale `correct_dir` left over from a previously rendered card produces the
    # same corruption without any ill intent, and both tables are append-only
    # with no removal route, so there is no way back.
    #
    # Resolving from the rank also *is* the rank validation: `_dir_for_rank`
    # walks the candidate list actually served for this query, so a rank with no
    # candidate behind it -- 47, or 11 when no reviewer directory is scorable
    # here -- is rejected rather than persisted into the pilot's primary
    # research output.
    correct_dir = None
    if body.outcome == FeedbackOutcome.MATCHED_RANK:
        # For multi-select submissions the first selected rank is the canonical
        # legacy choice, while selected_ranks carries the full reviewer answer.
        ranks = body.selected_ranks or [body.correct_rank]
        resolved = [
            await _dir_for_rank(store, db, slug, variant, body.query_id, rank)
            for rank in ranks
        ]
        unknown = [rank for rank, dir_name in zip(ranks, resolved) if dir_name is None]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No candidate at rank {unknown[0]} for query {body.query_id} "
                    f"under model '{slug}' (variant '{variant}')."
                ),
            )
        correct_dir = resolved[0]

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

    # Confirming a reviewer-created directory is what makes it grow: the query
    # joins the directory's members, so the next query is scored against both
    # documents rather than only the seed, and the directory's badge flips from
    # "Awaiting future match" to matched. That flip is precisely why this write
    # must follow a server-resolved `correct_dir` and nothing else -- it is the
    # record of a human confirmation.
    #
    # Append-only and idempotent: the feedback row above is untouched, and
    # re-submitting the same answer adds nothing. Recording feedback is
    # unconditional; membership is the extra consequence of confirming a
    # reviewer directory.
    if correct_dir and reviewer_dirs_svc.is_reviewer_dir_id(correct_dir):
        try:
            await db.add_reviewer_dir_member(
                dir_id=correct_dir,
                query_id=body.query_id,
                added_by=current_user.display_name,
                added_by_account_id=current_user.id,
            )
        except KeyError:
            # Unreachable via _dir_for_rank, which only ever returns directories
            # read out of this same table. Logged rather than raised so a race
            # cannot lose an otherwise valid feedback row, which is already
            # committed above and is the more valuable record.
            logger.warning(
                "Reviewer directory %s vanished between resolution and membership write",
                correct_dir,
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
    current_user: UserPublic = Depends(get_current_user),
) -> FeedbackEntry | None:
    if query_id not in store.file_id_to_filename:
        raise QueryNotFoundError(query_id)

    slug = normalize_slug(model)
    resolved = resolve_variant(store, variant)
    await _check_model_variant(store, slug, resolved)

    row = await db.get_latest_feedback(
        query_id=query_id,
        model_slug=slug,
        reviewer_account_id=current_user.id,
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


async def _dir_for_rank(
    store: DataStore,
    db: FeedbackDB,
    model_slug: str,
    variant: str,
    query_id: int,
    rank: int,
) -> str | None:
    """Directory name at `rank` in the list this query was shown, or None.

    Model candidates exactly as the retrieval CSV ranked them, then the
    reviewer-created directories anchored at MAX_MODEL_RANK + 1 -- the same
    ordering `get_predictions` builds, so a rank the client sends back always
    resolves to the card the reviewer clicked, independently of `top_k`.

    None means "no candidate at that rank", which the caller turns into a 422.
    This is the only rank validation that can be trusted, since the candidate
    count depends on the query, the model and how many reviewer directories are
    currently scorable.
    """
    for row in store.predictions.get((model_slug, variant), []):
        if row["file_id"] != query_id:
            continue
        for prediction in row["predictions"]:
            if prediction["rank"] == rank:
                return prediction["dir_name"]
        break

    records = await db.list_reviewer_dirs()
    if not records:
        return None
    qq = await store.ensure_qq_async(model_slug)
    for candidate in reviewer_dirs_svc.candidates_for_query(
        store=store, records=records, qq=qq, query_id=query_id
    ):
        if candidate.rank == rank:
            return candidate.dir_name
    return None
