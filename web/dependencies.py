from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from web.config import Settings
from web.models import UserPublic
from web.services.data_store import DataStore
from web.services.feedback_db import FeedbackDB
from web.services.rate_limit import RateLimiter

# Singletons set during app lifespan
_store: DataStore | None = None
_db: FeedbackDB | None = None

# Routes an account with must_change_password set may still reach: enough to
# learn who it is, change the password, or sign out. Everything else is 403
# until the password is changed, so the gate is backend-enforced and not just
# a UI convention.
PASSWORD_CHANGE_EXEMPT_PATHS = frozenset(
    {
        "/api/auth/me",
        "/api/auth/change_password",
        "/api/auth/signout",
        "/api/auth/signin",
        "/api/auth/register",
    }
)


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


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


async def get_current_user(
    request: Request,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    token = request.cookies.get(settings.auth.session_cookie)
    account = await db.get_account_by_session(token)
    if account is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = UserPublic(**account)
    if user.must_change_password and _request_path(request) not in PASSWORD_CHANGE_EXEMPT_PATHS:
        raise HTTPException(
            status_code=403,
            detail="Password change required before continuing",
        )
    return user


def _request_path(request: Request) -> str:
    path = request.url.path
    root_path = request.scope.get("root_path") or ""
    if root_path and path.startswith(root_path):
        path = path[len(root_path) :] or "/"
    return path.rstrip("/") or "/"


async def require_pi_admin(
    current_user: UserPublic = Depends(get_current_user),
) -> UserPublic:
    if current_user.role != "pi_admin":
        raise HTTPException(status_code=403, detail="PI/admin access required")
    return current_user
