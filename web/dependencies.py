from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from web.config import Settings
from web.models import UserPublic
from web.services.data_store import DataStore
from web.services.feedback_db import FeedbackDB

# Singletons set during app lifespan
_store: DataStore | None = None
_db: FeedbackDB | None = None


def set_store(store: DataStore) -> None:
    global _store
    _store = store


def set_db(db: FeedbackDB) -> None:
    global _db
    _db = db


def get_store() -> DataStore:
    assert _store is not None, "DataStore not initialized"
    return _store


def get_db() -> FeedbackDB:
    assert _db is not None, "FeedbackDB not initialized"
    return _db


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_current_user(
    request: Request,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    token = request.cookies.get(settings.auth.session_cookie)
    account = await db.get_account_by_session(token)
    if account is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return UserPublic(**account)


async def require_pi_admin(
    current_user: UserPublic = Depends(get_current_user),
) -> UserPublic:
    if current_user.role != "pi_admin":
        raise HTTPException(status_code=403, detail="PI/admin access required")
    return current_user
