from __future__ import annotations

from fastapi import APIRouter, Depends, Query
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
    current_user: UserPublic = Depends(get_current_user),
) -> FeedbackEntry | None:
    """What to prefill for this query: the team's newest note, the caller's own decision.

    Issue #96 splits the two halves of a review deliberately (the meeting asked
    for shared *notes*, not shared answers):

    - **Notes are shared.** Two reviewers on the same query should read each
      other's reasoning rather than silently duplicate it, so the note, its
      timestamp and its attribution come from the newest row by ANY reviewer.
    - **Decisions are not.** A rank pressed by somebody else is an answer the
      caller never gave, and one they could submit as their own by reflex, so
      the selection comes from the caller's OWN newest row and is left unset
      when they have none.

    The response is therefore a merged view rather than a verbatim DB row --
    see `_merge_shared_note_with_own_decision`. Saving never edits either
    source row: POST /api/feedback appends a new one under whoever is signed in.
    """
    if query_id not in store.file_id_to_filename:
        raise QueryNotFoundError(query_id)

    slug = normalize_slug(model)
    resolved = resolve_variant(store, variant)
    await _check_model_variant(store, slug, resolved)

    shared = await db.get_latest_feedback(
        query_id=query_id,
        model_slug=slug,
        variant=resolved,
    )
    if shared is None:
        return None
    own = await db.get_latest_feedback(
        query_id=query_id,
        model_slug=slug,
        variant=resolved,
        reviewer_account_id=current_user.id,
    )
    return FeedbackEntry(**_merge_shared_note_with_own_decision(shared, own))


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


#: The reviewer's answer, as opposed to their prose. Never inherited from
#: another reviewer's row -- see `latest_feedback`.
_DECISION_FIELDS = ("outcome", "correct_rank", "correct_dir", "selected_ranks")


def _merge_shared_note_with_own_decision(shared: dict, own: dict | None) -> dict:
    """Combine the team's newest note with the caller's own newest decision.

    The result keeps `shared`'s identity fields -- `id`, `timestamp`, `notes`,
    `reviewer`, `reviewer_username` -- because those describe the note being
    displayed and the attribution line rendered above it. Only the decision
    fields are replaced. When the caller has never reviewed this query, the
    decision is cleared to `legacy_unresolved`, which is the outcome the panel
    already reads as "no selection to restore".

    Callers should treat the result as a prefill view, not as a stored row: it
    can pair one reviewer's note with another's (absent) answer, which is the
    whole point.
    """
    merged = dict(shared)
    if own is not None:
        merged.update({field: own[field] for field in _DECISION_FIELDS})
    else:
        merged.update(
            {
                "outcome": FeedbackOutcome.LEGACY_UNRESOLVED.value,
                "correct_rank": None,
                "correct_dir": None,
                "selected_ranks": None,
            }
        )
    return merged


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
