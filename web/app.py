from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from web.config import Settings, load_settings
from web.dependencies import set_db, set_store
from web.exceptions import ExampleNotFoundError, InvalidModelError, QueryNotFoundError
from web.models import ErrorResponse, ErrorDetail
from web.routers import feedback, predictions, queries, stats, token_map
from web.services.data_store import build_store
from web.services.feedback_db import FeedbackDB

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings

    # Build in-memory data store
    logger.info("Building data store...")
    store = build_store(settings)
    set_store(store)

    # Initialize feedback database
    db_path = settings.paths.resolve(settings.paths.feedback_db)
    db = FeedbackDB(db_path)
    await db.connect()
    set_db(db)

    logger.info("Startup complete. Serving %d queries, %d models.",
                len(store.file_ids), len(store.model_slugs))
    yield

    await db.close()
    logger.info("Shutdown complete.")


def create_app(config_path: str | None = None) -> FastAPI:
    settings = load_settings(config_path)

    logging.basicConfig(
        level=logging.DEBUG if settings.app.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = FastAPI(
        title=settings.app.title,
        version=settings.app.version,
        lifespan=lifespan,
    )
    app.state.settings = settings

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_credentials=True,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )

    # Exception handlers
    @app.exception_handler(QueryNotFoundError)
    async def query_not_found_handler(request: Request, exc: QueryNotFoundError):
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(code="NOT_FOUND", message=exc.message)
            ).model_dump(),
        )

    @app.exception_handler(InvalidModelError)
    async def invalid_model_handler(request: Request, exc: InvalidModelError):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(code="INVALID_MODEL", message=exc.message)
            ).model_dump(),
        )

    @app.exception_handler(ExampleNotFoundError)
    async def example_not_found_handler(request: Request, exc: ExampleNotFoundError):
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(code="NOT_FOUND", message=exc.message)
            ).model_dump(),
        )

    # Routers
    app.include_router(queries.router)
    app.include_router(predictions.router)
    app.include_router(token_map.router)
    app.include_router(feedback.router)
    app.include_router(stats.router)

    # Serve frontend static files if they exist
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
