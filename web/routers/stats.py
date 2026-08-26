from __future__ import annotations

from fastapi import APIRouter, Depends

from web.dependencies import get_db, get_store, require_pi_admin
from web.models import (
    ConfidenceBands,
    ModelInfo,
    NeedsAttentionItem,
    RecentReview,
    StatsResponse,
    UserPublic,
)
from web.services.data_store import DataStore
from web.services.feedback_db import FeedbackDB

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(require_pi_admin),
) -> StatsResponse:
    db_stats = await db.get_stats()
    total_queries = len(store.file_ids)
    reviewed = db_stats["reviewed_count"]
    skipped = db_stats["skipped_count"]

    raw_recent = await db.get_recent_reviews(10)
    recent_reviews = [
        RecentReview(
            file_id=r["file_id"],
            filename=store.file_id_to_filename.get(r["file_id"], f"unknown-{r['file_id']}"),
            timestamp=r["timestamp"],
            model_slug=r["model_slug"],
            outcome=r["outcome"],
            reviewer=r["reviewer"],
            correct_rank=r["correct_rank"],
        )
        for r in raw_recent
    ]

    raw_needs_attention = await db.get_needs_attention(10)
    needs_attention = [
        NeedsAttentionItem(
            file_id=r["file_id"],
            filename=store.file_id_to_filename.get(r["file_id"], f"unknown-{r['file_id']}"),
            timestamp=r["timestamp"],
            model_slug=r["model_slug"],
            outcome=r["outcome"],
            notes=r["notes"],
            reviewer=r["reviewer"],
        )
        for r in raw_needs_attention
    ]

    next_unreviewed_ids = await db.get_next_unreviewed(store.file_ids, 5)

    return StatsResponse(
        total_queries=total_queries,
        reviewed_count=reviewed,
        skipped_count=skipped,
        unreviewed_count=total_queries - reviewed - skipped,
        unresolved_count=db_stats["unresolved_count"],
        feedback_count=db_stats["feedback_count"],
        reviews_by_model=db_stats["reviews_by_model"],
        reviews_by_reviewer=db_stats["reviews_by_reviewer"],
        rank_distribution=db_stats["rank_distribution"],
        outcome_distribution=db_stats["outcome_distribution"],
        recent_reviews=recent_reviews,
        needs_attention=needs_attention,
        next_unreviewed_ids=next_unreviewed_ids,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models(
    store: DataStore = Depends(get_store),
) -> list[ModelInfo]:
    # One entry per model, described by its default-variant metadata. The
    # variant CSVs cover the same models, so available_variants is the set of
    # variants this deployment found on disk.
    metas = [
        meta
        for (_slug, variant), meta in store.model_meta.items()
        if variant == store.default_variant
    ]
    return [
        ModelInfo(
            slug=meta.slug,
            display_name=meta.display_name,
            layer=meta.layer,
            pooling=meta.pooling,
            prediction_count=meta.prediction_count,
            available_variants=list(store.variants),
            default_variant=store.default_variant,
            # Bands come from the backend so the UI and the reviewer-directory
            # status logic can never disagree about what counts as a match.
            confidence_bands=ConfidenceBands(),
            supports_reviewer_dirs=meta.slug in store.qq_paths,
        )
        for meta in sorted(metas, key=lambda m: m.display_name)
    ]
