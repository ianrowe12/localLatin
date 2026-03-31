from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from web.dependencies import get_store
from web.exceptions import InvalidModelError, QueryNotFoundError
from web.models import CandidateFile, Prediction, PredictionResponse
from web.services.data_store import DataStore, normalize_slug

router = APIRouter(prefix="/api", tags=["predictions"])


def _get_prediction_row(store: DataStore, slug: str, file_id: int) -> dict | None:
    rows = store.predictions.get(slug)
    if rows is None:
        return None
    if file_id < 0 or file_id >= len(rows):
        return None
    row = rows[file_id]
    if row["file_id"] != file_id:
        # Fallback linear search if not aligned
        for r in rows:
            if r["file_id"] == file_id:
                return r
        return None
    return row


@router.get("/query/{file_id}/predictions", response_model=PredictionResponse)
async def get_predictions(
    file_id: int,
    model: str = Query(..., description="Model slug"),
    top_k: int = Query(10, ge=1, le=10),
    store: DataStore = Depends(get_store),
) -> PredictionResponse:
    slug = normalize_slug(model)
    if slug not in store.predictions:
        raise InvalidModelError(slug, store.model_slugs)
    if file_id not in store.file_id_to_filename:
        raise QueryNotFoundError(file_id)

    row = _get_prediction_row(store, slug, file_id)
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

    return PredictionResponse(
        file_id=file_id,
        filename=store.file_id_to_filename[file_id],
        model=slug,
        predictions=predictions,
    )


@router.get("/query/{file_id}/predictions/{rank}/candidates", response_model=list[CandidateFile])
async def get_candidates(
    file_id: int,
    rank: int,
    model: str = Query(..., description="Model slug"),
    store: DataStore = Depends(get_store),
) -> list[CandidateFile]:
    slug = normalize_slug(model)
    if slug not in store.predictions:
        raise InvalidModelError(slug, store.model_slugs)
    if file_id not in store.file_id_to_filename:
        raise QueryNotFoundError(file_id)

    row = _get_prediction_row(store, slug, file_id)
    if row is None:
        raise QueryNotFoundError(file_id)

    preds = row["predictions"]
    pred = next((p for p in preds if p["rank"] == rank), None)
    if pred is None:
        return []

    dir_name = pred["dir_name"]
    texts = store.labelled_texts.get(dir_name, {})
    return [
        CandidateFile(filename=fname, text=text)
        for fname, text in sorted(texts.items())
    ]
