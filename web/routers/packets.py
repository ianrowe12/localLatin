from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from web.dependencies import get_db, get_store, require_pi_admin
from web.exceptions import QueryNotFoundError
from web.models import MAX_MODEL_RANK, PredictionVariant, UserPublic
from web.routers.predictions import resolve_variant, resolve_variant_rows
from web.services import reviewer_dirs as reviewer_dirs_svc
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
    variant: PredictionVariant | None = Query(
        None,
        description=(
            "Post-processing variant whose predictions the packet documents. "
            "Defaults to the deployment's configured default variant."
        ),
    ),
    feedback_variant: PredictionVariant | None = Query(
        None,
        description=(
            "Restrict the reviewer-feedback section to one variant. By default "
            "every review for this query and model is included, each labelled "
            "with the variant it was recorded against."
        ),
    ),
    top_k: int = Query(MAX_MODEL_RANK, ge=1, le=MAX_MODEL_RANK),
    store: DataStore = Depends(get_store),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(require_pi_admin),
) -> Response:
    slug = normalize_slug(model)
    resolved = resolve_variant(store, variant)
    rows = await resolve_variant_rows(store, slug, resolved, file_id)

    row = _get_prediction_row(rows, file_id)
    if row is None:
        raise QueryNotFoundError(file_id)

    # `top_k` bounds the MODEL's candidates and nothing else -- its original
    # meaning, and what "Prediction scope: top N" says on the page. It used to
    # bound the whole list, which is why the only real caller (Header.tsx, then
    # hardcoding top_k=10) produced packets with every reviewer directory
    # truncated away, while a test that omitted the parameter and so got the new
    # default passed. Bounding only the model half means no caller, old or new,
    # can produce a packet that drops them.
    predictions = list(row["predictions"])[:top_k]

    # Reviewer directories are documented WITHOUT ranks; see
    # packet_dirs_for_query for why a live rank cannot be trusted in a packet.
    reviewer_dirs: list[dict] = []
    reviewer_records = await db.list_reviewer_dirs()
    if reviewer_records:
        qq = await store.ensure_qq_async(slug)
        reviewer_dirs = reviewer_dirs_svc.packet_dirs_for_query(
            store=store, records=reviewer_records, qq=qq, query_id=file_id
        )

    # The feedback section is variant-agnostic by default. Filtering it by the
    # requested variant would silently drop every pre-variant review (those rows
    # have variant NULL by design), quietly emptying the PI's record of the
    # pilot. Each row is labelled with its variant instead, and a caller who
    # genuinely wants one variant asks for it explicitly.
    feedback_rows = await db.get_feedback_for_query(
        file_id,
        model=slug,
        limit=10,
        variant=feedback_variant.value if feedback_variant else None,
    )
    pdf_data = build_review_packet_pdf(
        store=store,
        query_id=file_id,
        model_slug=slug,
        variant=resolved,
        feedback_variant=feedback_variant.value if feedback_variant else None,
        predictions=predictions,
        reviewer_dirs=reviewer_dirs,
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
        resolved,
        top_k,
        len(pdf_data),
    )
    return Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{packet_name}"'},
    )
