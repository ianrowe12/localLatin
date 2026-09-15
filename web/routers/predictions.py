from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from web.dependencies import get_current_user, get_db, get_store
from web.exceptions import InvalidModelError, QueryNotFoundError, VariantUnavailableError
from web.models import (
    CandidateFile,
    Prediction,
    PredictionResponse,
    PredictionVariant,
    UserPublic,
)
from web.services import reviewer_dirs as reviewer_dirs_svc
from web.services.data_store import DataStore, normalize_slug
from web.services.feedback_db import FeedbackDB

router = APIRouter(prefix="/api", tags=["predictions"])

# Deliberately defaults to None rather than a hardcoded variant: the served
# default is `PathsConfig.default_variant`, which deployments can change.
VariantParam = Query(
    None,
    description=(
        "Post-processing variant of the predictions to serve. "
        "Defaults to the deployment's configured default variant."
    ),
)


def resolve_variant(store: DataStore, variant: PredictionVariant | None) -> str:
    """The requested variant, or the deployment's configured default."""
    return variant.value if variant is not None else store.default_variant


async def resolve_variant_rows(
    store: DataStore, slug: str, variant: str, file_id: int
) -> list[dict]:
    """Prediction rows for (slug, variant), raising the right API error."""
    if not await store.ensure_variant_async(variant):
        raise VariantUnavailableError(variant, store.variants)
    rows = store.predictions.get((slug, variant))
    if rows is None:
        raise InvalidModelError(slug, store.model_slugs)
    if file_id not in store.file_id_to_filename:
        raise QueryNotFoundError(file_id)
    return rows


def _get_prediction_row(rows: list[dict], file_id: int) -> dict | None:
    if 0 <= file_id < len(rows):
        row = rows[file_id]
        if row["file_id"] == file_id:
            return row
    # Fallback linear search if not aligned
    for row in rows:
        if row["file_id"] == file_id:
            return row
    return None


@router.get("/query/{file_id}/predictions", response_model=PredictionResponse)
async def get_predictions(
    file_id: int,
    model: str = Query(..., description="Model slug"),
    variant: PredictionVariant | None = VariantParam,
    top_k: int = Query(10, ge=1, le=10),
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> PredictionResponse:
    slug = normalize_slug(model)
    resolved = resolve_variant(store, variant)
    rows = await resolve_variant_rows(store, slug, resolved, file_id)

    row = _get_prediction_row(rows, file_id)
    if row is None:
        raise QueryNotFoundError(file_id)

    predictions = []
    for pred in row["predictions"][:top_k]:
        dir_name = pred["dir_name"]
        dir_files = store.labelled_dir_files.get(dir_name, [])
        texts = store.labelled_texts.get(dir_name, {})
        preview = ""
        if dir_files and dir_files[0] in texts:
            preview = texts[dir_files[0]][:200]

        candidate_files = [
            CandidateFile(filename=fname, text=texts.get(fname, ""))
            for fname in dir_files
        ]

        predictions.append(Prediction(
            rank=pred["rank"],
            dir_name=dir_name,
            score=pred["score"],
            dir_files=dir_files,
            preview_text=preview,
            candidate_files=candidate_files,
        ))

    # --- reviewer-created directories (issue #95, reshaped by #196) ---------
    # NEVER merged into `predictions`. Every entry there is the retrieval run's
    # own answer at the rank the CSV gave it, and nothing else is ever numbered
    # alongside them: a reviewer directory arrives in its own field, unranked,
    # the way the PDF packets have always printed it.
    reviewer_records = await db.list_reviewer_dirs()
    seeded_dirs = []
    reviewer_dir_candidates = []
    if reviewer_records:
        qq = await store.ensure_qq_async(slug)
        reviewer_dir_candidates = reviewer_dirs_svc.unranked_candidates_for_query(
            store=store, records=reviewer_records, qq=qq, query_id=file_id
        )
        seeded_dirs = [
            reviewer_dirs_svc.to_api(record, qq, slug)
            for record in reviewer_records
            if int(record["seed_query_id"]) == file_id
        ]

    return PredictionResponse(
        file_id=file_id,
        filename=store.file_id_to_filename[file_id],
        model=slug,
        variant=resolved,
        predictions=predictions,
        # Straight from the CSV row, never derived from how many candidates came
        # back: an excluded query and a query whose ranking is missing for an
        # unknown reason both serialise `predictions: []`, and only this tells
        # the reviewer which one they are looking at.
        status=row.get("status"),
        seeded_dirs=seeded_dirs,
        reviewer_dir_candidates=reviewer_dir_candidates,
    )


@router.get("/query/{file_id}/predictions/{rank}/candidates", response_model=list[CandidateFile])
async def get_candidates(
    file_id: int,
    rank: int,
    model: str = Query(..., description="Model slug"),
    variant: PredictionVariant | None = VariantParam,
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> list[CandidateFile]:
    slug = normalize_slug(model)
    resolved = resolve_variant(store, variant)
    rows = await resolve_variant_rows(store, slug, resolved, file_id)

    row = _get_prediction_row(rows, file_id)
    if row is None:
        raise QueryNotFoundError(file_id)

    preds = row["predictions"]
    pred = next((p for p in preds if p["rank"] == rank), None)
    if pred is None:
        # Nothing but the model's own candidates has a rank any more (issue
        # #196), so a rank this row does not carry names nothing. Reviewer
        # directories are addressed by `dir_id` through
        # /api/candidate_dir/{dir}/files, which is what their unranked block
        # links to; historical feedback rows recording ranks 11-15 stay readable
        # through that same route and through the packets.
        return []

    dir_name = pred["dir_name"]
    texts = store.labelled_texts.get(dir_name, {})
    return [
        CandidateFile(filename=fname, text=text)
        for fname, text in sorted(texts.items())
    ]


@router.get("/candidate_dir/{candidate_dir}/files", response_model=list[CandidateFile])
async def get_candidate_dir_files(
    candidate_dir: str,
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> list[CandidateFile]:
    """Return all files in a candidate directory, labelled or reviewer-created.

    Used by the example gallery when navigating to an off-top-10 candidate
    that's not in the predictions list, and by the reviewer-directory card,
    which addresses its directory by id rather than by rank.
    """
    if reviewer_dirs_svc.is_reviewer_dir_id(candidate_dir):
        record = await db.get_reviewer_dir(candidate_dir)
        return reviewer_dirs_svc.member_files(store, record) if record else []
    texts = store.labelled_texts.get(candidate_dir)
    if texts is None:
        return []
    return [
        CandidateFile(filename=fname, text=text)
        for fname, text in sorted(texts.items())
    ]
