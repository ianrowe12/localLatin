from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from web.dependencies import get_db, get_store, require_pi_admin
from web.exceptions import QueryNotFoundError
from web.models import PredictionVariant, UserPublic
from web.routers.predictions import resolve_variant_rows
from web.services.data_store import DataStore, normalize_slug
from web.services.feedback_db import FeedbackDB
from web.services.pdf_packets import build_review_packet_pdf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["packets"])


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "packet"


def _get_prediction_row(rows: list[dict], file_id: int) -> dict | None:
    for row in rows:
        if row["file_id"] == file_id:
            return row
    return None


@router.get("/packets/review/{file_id}")
async def get_review_packet(
    file_id: int,
    model: str = Query(..., description="Model slug"),
    variant: PredictionVariant = Query(
        PredictionVariant.SIF_ABTT,
        description="Post-processing variant the packet documents",
    ),
    top_k: int = Query(10, ge=1, le=10),
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(require_pi_admin),
) -> Response:
    slug = normalize_slug(model)
    rows = resolve_variant_rows(store, slug, variant.value, file_id)

    row = _get_prediction_row(rows, file_id)
    if row is None:
        raise QueryNotFoundError(file_id)

    feedback_rows = await db.get_feedback_for_query(
        file_id, model=slug, limit=10, variant=variant.value
    )
    pdf_data = build_review_packet_pdf(
        store=store,
        query_id=file_id,
        model_slug=slug,
        predictions=row["predictions"],
        feedback_rows=feedback_rows,
        actor=current_user.display_name,
        top_k=top_k,
    )
    packet_name = _safe_filename(
        f"review_packet_{store.file_id_to_filename[file_id]}_{slug}.pdf"
    )
    logger.info(
        "Generated review packet actor=%s query_id=%s model=%s variant=%s top_k=%s bytes=%s",
        current_user.username,
        file_id,
        slug,
        variant.value,
        top_k,
        len(pdf_data),
    )
    return Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{packet_name}"'},
    )
