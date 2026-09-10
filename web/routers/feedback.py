from __future__ import annotations

import logging
import math

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from web.dependencies import get_current_user, get_db, get_store, require_pi_admin
from web.exceptions import InvalidModelError, QueryNotFoundError, VariantUnavailableError
from web.models import (
    CandidateSource,
    ErrorDetail,
    ErrorResponse,
    FeedbackCreate,
    FeedbackEntry,
    FeedbackOutcome,
    MAX_MODEL_RANK,
    Prediction,
    PredictionVariant,
    UserPublic,
)
from web.routers.predictions import get_predictions, resolve_variant
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
) -> FeedbackEntry | JSONResponse:
    if body.query_id not in store.file_id_to_filename:
        raise QueryNotFoundError(body.query_id)

    slug = normalize_slug(body.model_slug)
    variant = resolve_variant(store, body.variant)
    await _check_model_variant(store, slug, variant)

    candidates: dict[int, Prediction] = {}
    if body.outcome in (FeedbackOutcome.MATCHED_RANK, FeedbackOutcome.NONE_OF_TOP_K):
        # Use one snapshot of the same candidates the predictions route serves.
        # Reviewer ranks can move as memberships change, even during a save.
        snapshot = await get_predictions(
            file_id=body.query_id,
            model=slug,
            variant=PredictionVariant(variant),
            top_k=MAX_MODEL_RANK,
            store=store,
            db=db,
            current_user=current_user,
        )
        excluded = (snapshot.status or "").strip().startswith("excluded")
        usable_model_ranking = any(
            candidate.source == CandidateSource.MODEL and _candidate_is_usable(candidate)
            for candidate in snapshot.predictions
        )
        if excluded or not usable_model_ranking:
            return _feedback_error(
                422,
                "RANKING_NOT_EVALUABLE",
                "Evaluation requires a usable model ranking. "
                "Keep a draft or deliberately skip with a note.",
            )
        candidates = {candidate.rank: candidate for candidate in snapshot.predictions}

    # Client correct_dir is never an assignment authority.
    correct_dir = None
    if body.outcome == FeedbackOutcome.MATCHED_RANK:
        assert body.correct_rank is not None
        ranks = body.selected_ranks or [body.correct_rank]
        for rank in ranks:
            candidate = candidates.get(rank)
            if body.expected_candidate_dirs is not None and (
                candidate is None
                or candidate.dir_name != body.expected_candidate_dirs[rank]
            ):
                return _feedback_error(
                    409,
                    "CANDIDATE_IDENTITY_CHANGED",
                    f"Candidate at rank {rank} has changed. "
                    "Refresh the ranking and review your selections before saving.",
                )
            if candidate is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"No candidate at rank {rank} for query {body.query_id} "
                        f"under model '{slug}' (variant '{variant}')."
                    ),
                )
            if not _candidate_is_usable(candidate):
                return _feedback_error(
                    422,
                    "CANDIDATE_NOT_EVALUABLE",
                    f"Candidate at rank {rank} has no usable evidence. "
                    "Choose another candidate or keep a draft.",
                )
        # Preserve selection order: only the first choice assigns membership.
        correct_dir = candidates[ranks[0]].dir_name

    assert body.outcome is not None
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
    # Membership insertion is idempotent for this resolved directory/query.
    # Feedback is append-only, not idempotent: each accepted save adds a row.
    if correct_dir and reviewer_dirs_svc.is_reviewer_dir_id(correct_dir):
        try:
            await db.add_reviewer_dir_member(
                dir_id=correct_dir,
                query_id=body.query_id,
                added_by=current_user.display_name,
                added_by_account_id=current_user.id,
            )
        except KeyError:
            # Unreachable via the snapshot, which only ever returns directories
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
    """What to prefill for this query: the team's newest note, the caller's own decision.

    Issue #96 splits the two halves of a review deliberately (the meeting asked
    for shared *notes*, not shared answers):

    - **Notes are shared.** Two reviewers on the same query should read each
      other's reasoning rather than silently duplicate it, so the note, its
      timestamp and its attribution come from the newest row by ANY reviewer
      that actually carries a note. Rows saved without prose are skipped for
      this half: the box exists to surface notes, so an answer recorded in
      silence must not blank out a colleague's substantive note.
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

    shared_note = await db.get_latest_feedback(
        query_id=query_id,
        model_slug=slug,
        variant=resolved,
        require_note=True,
    )
    own = await db.get_latest_feedback(
        query_id=query_id,
        model_slug=slug,
        variant=resolved,
        reviewer_account_id=current_user.id,
    )
    merged = _merge_shared_note_with_own_decision(shared_note, own)
    return FeedbackEntry(**merged) if merged is not None else None


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


def _merge_shared_note_with_own_decision(
    shared: dict | None, own: dict | None
) -> dict | None:
    """Combine the team's newest note with the caller's own newest decision.

    The result keeps the note row's identity fields -- `id`, `timestamp`,
    `notes`, `reviewer`, `reviewer_username` -- because those describe the note
    being displayed and the attribution line rendered above it. Only the
    decision fields are replaced. When the caller has never reviewed this query,
    the decision is cleared to `legacy_unresolved`, which is the outcome the
    panel already reads as "no selection to restore".

    Either half can be missing:

    - No note anywhere, but the caller has an answer: their own row becomes the
      base, so their selection is still restored. Nothing is lost by the
      note filter.
    - A note but no answer from the caller: the note shows, unselected.
    - Neither: None, and the panel stays empty.

    Callers should treat the result as a prefill view, not as a stored row: it
    can pair one reviewer's note with another's (absent) answer, which is the
    whole point.
    """
    base = shared if shared is not None else own
    if base is None:
        return None
    merged = dict(base)
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


def _candidate_is_usable(candidate: Prediction) -> bool:
    return (
        bool(candidate.dir_name.strip())
        and math.isfinite(candidate.score)
        and any(file.text.strip() for file in candidate.candidate_files or [])
    )


def _feedback_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(),
    )
