from __future__ import annotations

import sqlite3
import secrets
import string

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from web.config import Settings
from web.dependencies import get_current_user, get_db, get_settings, require_pi_admin
from web.models import (
    AccountCreateRequest,
    AccountCreateResponse,
    AccountPublic,
    ApprovalDecisionRequest,
    RegisterRequest,
    RegistrationResponse,
    SignInRequest,
    UserPublic,
)
from web.services.feedback_db import FeedbackDB

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=RegistrationResponse, status_code=201)
async def register(
    body: RegisterRequest,
    response: Response,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RegistrationResponse:
    role = await _registration_role(body, db, settings)
    approval_status = "approved" if role == "pi_admin" else "pending"
    try:
        account = await db.create_account(
            username=body.username,
            display_name=body.display_name,
            password=body.password,
            role=role,
            approval_status=approval_status,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already exists") from exc
    if approval_status == "approved":
        await _set_session_cookie(response, db, settings, account["id"])
        return RegistrationResponse(
            status="approved",
            message="Account created.",
            account=UserPublic(**account),
        )
    return RegistrationResponse(
        status="pending_approval",
        message="Account request submitted. A PI/admin must approve it before sign-in.",
        account=UserPublic(**account),
    )


@router.post("/signin", response_model=UserPublic)
async def signin(
    body: SignInRequest,
    response: Response,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    account, reason = await db.verify_account(body.username, body.password)
    if account is None:
        if reason == "pending":
            raise HTTPException(status_code=403, detail="Account pending approval")
        if reason == "rejected":
            raise HTTPException(status_code=403, detail="Account registration was rejected")
        if reason == "inactive":
            raise HTTPException(status_code=403, detail="Account is inactive")
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
    response.status_code = 204
    return response


@router.get("/me", response_model=UserPublic)
async def me(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return current_user


@router.get("/accounts", response_model=list[AccountPublic])
async def list_accounts(
    status: str | None = Query(None, pattern="^(pending|approved|rejected|all)$"),
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(require_pi_admin),
) -> list[AccountPublic]:
    del current_user
    approval_status = None if status in (None, "all") else status
    accounts = await db.list_accounts(approval_status=approval_status)
    return [AccountPublic(**account) for account in accounts]


@router.post("/accounts", response_model=AccountCreateResponse, status_code=201)
async def create_account_as_admin(
    body: AccountCreateRequest,
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(require_pi_admin),
) -> AccountCreateResponse:
    password = body.password or _generate_temporary_password()
    try:
        account = await db.create_account(
            username=body.username,
            display_name=body.display_name,
            password=password,
            role=body.role,
            approval_status="approved",
            approved_by_account_id=current_user.id,
            approval_note=body.approval_note,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already exists") from exc
    full_account = (await db.list_accounts(approval_status="approved"))
    created = next(item for item in full_account if item["id"] == account["id"])
    return AccountCreateResponse(
        account=AccountPublic(**created),
        temporary_password=password if body.password is None else None,
    )


@router.post("/accounts/{account_id}/approve", response_model=AccountPublic)
async def approve_account(
    account_id: int,
    body: ApprovalDecisionRequest | None = None,
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(require_pi_admin),
) -> AccountPublic:
    account = await db.set_account_approval(
        account_id=account_id,
        approval_status="approved",
        approver_account_id=current_user.id,
        note=body.note if body else "",
    )
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return AccountPublic(**account)


@router.post("/accounts/{account_id}/reject", response_model=AccountPublic)
async def reject_account(
    account_id: int,
    body: ApprovalDecisionRequest | None = None,
    db: FeedbackDB = Depends(get_db),
    current_user: UserPublic = Depends(require_pi_admin),
) -> AccountPublic:
    account = await db.set_account_approval(
        account_id=account_id,
        approval_status="rejected",
        approver_account_id=current_user.id,
        note=body.note if body else "",
    )
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return AccountPublic(**account)


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


def _generate_temporary_password() -> str:
    alphabet = string.ascii_letters + string.digits + "-_!@#%"
    return "".join(secrets.choice(alphabet) for _ in range(24))
