from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from web.dependencies import get_store
from web.exceptions import ExampleNotFoundError
from web.models import TokenMapExampleSummary, TokenMapResponse
from web.services.data_store import DataStore
from web.services import token_map_svc

router = APIRouter(prefix="/api", tags=["token_map"])


@router.get("/token_map_examples", response_model=list[TokenMapExampleSummary])
async def list_token_map_examples(
    model: str | None = None,
    bucket: str | None = None,
    store: DataStore = Depends(get_store),
) -> list[TokenMapExampleSummary]:
    return token_map_svc.list_examples(store, model=model, bucket=bucket)


@router.get("/token_map/{example_id}", response_model=TokenMapResponse)
async def get_token_map(
    example_id: int,
    store: DataStore = Depends(get_store),
) -> TokenMapResponse:
    result = token_map_svc.load_token_map(store, example_id)
    if result is None:
        raise ExampleNotFoundError(example_id)
    return result
