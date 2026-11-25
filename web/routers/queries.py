from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal

from fastapi import APIRouter, Depends, Query

from web.dependencies import get_current_user, get_db, get_store
from web.exceptions import InvalidModelError, QueryNotFoundError
from web.models import (
    LabelledKeyItem,
    LabelledKeyListResponse,
    NextQueryResponse,
    PredictionVariant,
    QueryDetail,
    QueryKeyMatch,
    QueryListItem,
    QueryListResponse,
    UserPublic,
)
from web.routers.predictions import VariantParam, require_prediction_variant, resolve_variant
from web.services.ccl_keys import match_form
from web.services.data_store import DataStore, PredictionKeyHit, normalize_slug
from web.services.feedback_db import FeedbackDB
from web.services.text_tokenizer import latin_tokenize

router = APIRouter(prefix="/api", tags=["queries"])

_DIGIT_RUN = re.compile(r"(\d+)")


def _natural_sort_key(name: str) -> list[str | int]:
    return [
        int(part) if i % 2 else part
        for i, part in enumerate(_DIGIT_RUN.split(name))
    ]


@router.get("/labelled_keys", response_model=LabelledKeyListResponse)
async def list_labelled_keys(
    search: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    store: DataStore = Depends(get_store),
    current_user: UserPublic = Depends(get_current_user),
) -> LabelledKeyListResponse:
    target = match_form(search)
    hits = []
    for name in store.labelled_dir_files:
        normalized = match_form(name)
        if target in normalized:
            hits.append((not normalized.startswith(target), _natural_sort_key(normalized), name))
    hits.sort()

    start = (page - 1) * page_size
    end = start + page_size
    return LabelledKeyListResponse(
        items=[
            LabelledKeyItem(key=name, unit_count=len(store.labelled_dir_files[name]))
            for _, _, name in hits[start:end]
        ],
        total=len(hits),
        page=page,
        page_size=page_size,
        has_more=end < len(hits),
    )


@router.get("/queries", response_model=QueryListResponse)
async def list_queries(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str | None = None,
    status: str | None = Query(None, pattern="^(reviewed|unreviewed|skipped|all)$"),
    sort: str = Query("file_id", pattern="^(file_id|filename)$"),
    key: str | None = None,
    key_match: Literal["prefix", "exact"] = "prefix",
    key_model: str | None = None,
    variant: PredictionVariant | None = VariantParam,
    key_top1: bool = False,
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> QueryListResponse:
    matches: dict[int, list[PredictionKeyHit]] = defaultdict(list)
    if key is not None:
        resolved = resolve_variant(store, variant)
        await require_prediction_variant(store, resolved)
        slug = normalize_slug(key_model) if key_model is not None else None
        if slug is not None and (slug, resolved) not in store.predictions:
            raise InvalidModelError(slug, store.model_slugs)

        target = match_form(key)
        index = store.prediction_key_index[resolved]
        buckets = (
            (index.get(target, []),)
            if key_match == "exact"
            else (hits for name, hits in index.items() if name.startswith(target))
        )
        for hits in buckets:
            for hit in hits:
                if (slug is None or hit.model == slug) and (not key_top1 or hit.rank == 1):
                    matches[hit.file_id].append(hit)

    query_statuses = await db.get_query_statuses()

    # Build filtered list
    items: list[tuple[int, str]] = []
    for fid in store.file_ids:
        if key is not None and fid not in matches:
            continue
        fname = store.file_id_to_filename[fid]
        if search and search.lower() not in fname.lower():
            continue
        review_status = query_statuses.get(fid, {}).get("review_status", "unreviewed")
        if status and status != "all" and status != review_status:
            continue
        items.append((fid, fname))

    # Sort
    if sort == "filename":
        items.sort(key=lambda x: x[1])
    else:
        items.sort(key=lambda x: x[0])

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    result_items = []
    for fid, fname in page_items:
        text = store.unlabelled_texts.get(fid, "")
        status_info = query_statuses.get(
            fid, {"review_status": "unreviewed", "review_count": 0}
        )
        result_items.append(QueryListItem(
            file_id=fid,
            filename=fname,
            text_preview=text[:150],
            review_status=status_info["review_status"],
            review_count=status_info["review_count"],
            key_matches=[
                QueryKeyMatch(model=hit.model, rank=hit.rank, dir_name=hit.dir_name)
                for hit in sorted(matches.get(fid, []), key=lambda hit: (hit.rank, hit.model))
            ],
        ))

    return QueryListResponse(
        items=result_items,
        total=total,
        page=page,
        page_size=page_size,
        has_more=end < total,
    )


@router.get("/queries/next", response_model=NextQueryResponse)
async def get_next_query(
    after: int | None = None,
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> NextQueryResponse:
    query_statuses = await db.get_query_statuses()
    actionable_ids = [
        fid
        for fid in store.file_ids
        if query_statuses.get(fid, {}).get("review_status", "unreviewed")
        == "unreviewed"
    ]

    if after is not None:
        for fid in actionable_ids:
            if fid > after:
                return NextQueryResponse(file_id=fid)

    return NextQueryResponse(file_id=actionable_ids[0] if actionable_ids else None)


@router.get("/query/{file_id}", response_model=QueryDetail)
async def get_query(
    file_id: int,
    store: DataStore = Depends(get_store),
    current_user: UserPublic = Depends(get_current_user),
) -> QueryDetail:
    if file_id not in store.unlabelled_texts:
        raise QueryNotFoundError(file_id)

    text = store.unlabelled_texts[file_id]
    tokens = latin_tokenize(text)

    return QueryDetail(
        file_id=file_id,
        filename=store.file_id_to_filename[file_id],
        text=text,
        tokens=tokens,
        char_count=len(text),
        token_count=len(tokens),
    )
