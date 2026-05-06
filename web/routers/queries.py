from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from web.dependencies import get_current_user, get_db, get_store
from web.exceptions import QueryNotFoundError
from web.models import QueryDetail, QueryListItem, QueryListResponse, UserPublic
from web.services.data_store import DataStore
from web.services.feedback_db import FeedbackDB
from web.services.text_tokenizer import latin_tokenize

router = APIRouter(prefix="/api", tags=["queries"])


@router.get("/queries", response_model=QueryListResponse)
async def list_queries(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str | None = None,
    status: str | None = Query(None, pattern="^(reviewed|unreviewed|skipped|all)$"),
    sort: str = Query("file_id", pattern="^(file_id|filename)$"),
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(get_current_user),
) -> QueryListResponse:
    query_statuses = await db.get_query_statuses()

    # Build filtered list
    items: list[tuple[int, str]] = []
    for fid in store.file_ids:
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
        ))

    return QueryListResponse(
        items=result_items,
        total=total,
        page=page,
        page_size=page_size,
        has_more=end < total,
    )


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
