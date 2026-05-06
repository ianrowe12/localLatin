from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from web.config import Settings
from web.dependencies import get_current_user, get_db, get_settings
from web.models import RegisterRequest, SignInRequest, UserPublic
from web.services.feedback_db import FeedbackDB

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserPublic, status_code=201)
async def register(
    body: RegisterRequest,
    response: Response,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    role = await _registration_role(body, db, settings)
    try:
        account = await db.create_account(
            username=body.username,
            display_name=body.display_name,
            password=body.password,
            role=role,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already exists") from exc
    await _set_session_cookie(response, db, settings, account["id"])
    return UserPublic(**account)


@router.post("/signin", response_model=UserPublic)
async def signin(
    body: SignInRequest,
    response: Response,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    account = await db.verify_account(body.username, body.password)
    if account is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    await _set_session_cookie(response, db, settings, account["id"])
    return UserPublic(**account)


@router.post("/signout", status_code=204)
async def signout(
    request: Request,
    response: Response,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    await db.revoke_session(request.cookies.get(settings.auth.session_cookie))
    response.delete_cookie(
        settings.auth.session_cookie,
        httponly=True,
        secure=settings.auth.secure_cookies,
        samesite="lax",
    )
    return response


@router.get("/me", response_model=UserPublic)
async def me(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return current_user


async def _registration_role(
    body: RegisterRequest,
    db: FeedbackDB,
    settings: Settings,
) -> str:
    if await db.account_count() == 0:
        return "pi_admin"
    if body.admin_code:
        if not settings.auth.admin_registration_code:
            raise HTTPException(status_code=403, detail="Admin registration is disabled")
        if body.admin_code != settings.auth.admin_registration_code:
            raise HTTPException(status_code=403, detail="Invalid admin registration code")
        return "pi_admin"
    return "reviewer"


async def _set_session_cookie(
    response: Response,
    db: FeedbackDB,
    settings: Settings,
    account_id: int,
) -> None:
    token = await db.create_session(account_id, settings.auth.session_days)
    response.set_cookie(
        settings.auth.session_cookie,
        token,
        httponly=True,
        secure=settings.auth.secure_cookies,
        samesite="lax",
        max_age=settings.auth.session_days * 24 * 60 * 60,
    )
