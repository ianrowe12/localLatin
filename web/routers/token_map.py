from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from web.dependencies import get_current_user, get_store
from web.exceptions import ExampleNotFoundError
from web.models import (
    TokenMapExampleSummary,
    TokenMapExamplesGroupedResponse,
    TokenMapResponse,
    UserPublic,
)
from web.services.data_store import DataStore
from web.services import token_map_svc

router = APIRouter(prefix="/api", tags=["token_map"])

# Kept in sync with token_map_svc.ATTRIBUTION_METHODS / ATTRIBUTION_VARIANTS.
# Spelled as Literals so FastAPI rejects typos with a 422 instead of silently
# returning an empty payload.
AttributionMethodName = Literal[
    "ig",
    "bertscore",
    "ot",
    "attention_weighted",
    "dla",
    "attention_standalone",
    "retrieval_mark",
]
# NOTE: the attribution artifacts call the no-post-processing variant
# "baseline" while the prediction CSVs call it "raw". This endpoint speaks the
# artifact vocabulary; the single raw<->baseline translation point lives in the
# frontend (`toAttributionVariant` in src/api/variants.ts).
AttributionVariantName = Literal["baseline", "abtt", "sif", "sif_abtt"]

MethodParam = Query(
    None,
    description=(
        "Serialise only this attribution method's matrices. "
        "Omit to receive every method present in the artifact."
    ),
)
VariantParam = Query(
    None,
    description=(
        "Serialise only this attribution variant's matrices. "
        "Omit to receive every variant present in the artifact."
    ),
)


@router.get("/token_map_examples", response_model=list[TokenMapExampleSummary])
async def list_token_map_examples(
    model: str | None = None,
    bucket: str | None = None,
    store: DataStore = Depends(get_store),
    current_user: UserPublic = Depends(get_current_user),
) -> list[TokenMapExampleSummary]:
    return token_map_svc.list_examples(store, model=model, bucket=bucket)


@router.get("/token_map_examples_grouped", response_model=TokenMapExamplesGroupedResponse)
async def list_token_map_examples_grouped(
    store: DataStore = Depends(get_store),
    current_user: UserPublic = Depends(get_current_user),
) -> TokenMapExamplesGroupedResponse:
    result = token_map_svc.list_examples_grouped(store)
    return TokenMapExamplesGroupedResponse(**result)


@router.get("/token_map/{example_id}", response_model=TokenMapResponse)
async def get_token_map(
    example_id: int,
    method: AttributionMethodName | None = MethodParam,
    variant: AttributionVariantName | None = VariantParam,
    store: DataStore = Depends(get_store),
    current_user: UserPublic = Depends(get_current_user),
) -> TokenMapResponse:
    result = token_map_svc.load_token_map(store, example_id, method=method, variant=variant)
    if result is None:
        raise ExampleNotFoundError(example_id)
    return result


@router.get("/query/{file_id}/token_map", response_model=TokenMapResponse)
async def get_token_map_by_query(
    file_id: int,
    candidate_dir: str = Query(..., description="Candidate directory name"),
    model: str = Query("", description="Model slug (optional, narrows lookup)"),
    method: AttributionMethodName | None = MethodParam,
    variant: AttributionVariantName | None = VariantParam,
    store: DataStore = Depends(get_store),
    current_user: UserPublic = Depends(get_current_user),
) -> TokenMapResponse:
    """Look up a token map by query file_id + candidate directory.

    ``file_id`` here always indexes the unlabelled review queue, so prefer an
    unlabelled example when the CSV records the query's corpus. Labelled rows
    stay reachable as a fallback for CSVs written before that column existed.
    """
    example_id = token_map_svc.resolve_example_id(
        store, file_id, candidate_dir, model or None, query_source="unlabelled",
    )
    if example_id is None:
        raise ExampleNotFoundError(f"{file_id}/{candidate_dir}")
    result = token_map_svc.load_token_map(store, example_id, method=method, variant=variant)
    if result is None:
        raise ExampleNotFoundError(example_id)
    return result
