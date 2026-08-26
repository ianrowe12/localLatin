from __future__ import annotations

import sqlite3
import secrets
import string

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from web.config import Settings
from web.dependencies import (
    get_current_user,
    get_db,
    get_rate_limiter,
    get_settings,
    require_pi_admin,
)
from web.models import (
    AccountCreateRequest,
    AccountCreateResponse,
    AccountPublic,
    ApprovalDecisionRequest,
    PasswordChangeRequest,
    PasswordResetResponse,
    RegisterRequest,
    RegistrationResponse,
    SignInRequest,
    UserPublic,
)
from web.services.feedback_db import FeedbackDB
from web.services.rate_limit import RateLimiter

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


@router.post("/change_password", response_model=UserPublic)
async def change_password(
    body: PasswordChangeRequest,
    request: Request,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
    limiter: RateLimiter = Depends(get_rate_limiter),
    current_user: UserPublic = Depends(get_current_user),
) -> UserPublic:
    _enforce_rate_limit(limiter, settings, f"change:{current_user.id}")
    if not await db.verify_account_password(current_user.id, body.current_password):
        raise HTTPException(status_code=403, detail="Current password is incorrect")
    if body.new_password == body.current_password:
        raise HTTPException(
            status_code=400, detail="New password must differ from the current one"
        )
    account = await db.set_account_password(
        account_id=current_user.id,
        new_password=body.new_password,
        must_change_password=False,
        # The tab doing the change keeps working; every other session for this
        # account is revoked.
        keep_session_token=request.cookies.get(settings.auth.session_cookie),
    )
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return UserPublic(**{key: account[key] for key in UserPublic.model_fields})


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


@router.post("/accounts/{account_id}/reset_password", response_model=PasswordResetResponse)
async def reset_account_password(
    account_id: int,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
    limiter: RateLimiter = Depends(get_rate_limiter),
    current_user: UserPublic = Depends(require_pi_admin),
) -> PasswordResetResponse:
    _enforce_rate_limit(limiter, settings, f"reset:{current_user.id}")
    temporary_password = _generate_temporary_password()
    account = await db.set_account_password(
        account_id=account_id,
        new_password=temporary_password,
        must_change_password=True,
        # Admin reset: every session of the affected account dies.
        keep_session_token=None,
    )
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return PasswordResetResponse(
        account=AccountPublic(**account),
        temporary_password=temporary_password,
    )


def _enforce_rate_limit(limiter: RateLimiter, settings: Settings, key: str) -> None:
    allowed = limiter.check(
        key,
        settings.auth.password_rate_limit_max_attempts,
        settings.auth.password_rate_limit_window_seconds,
    )
    if not allowed:
        raise HTTPException(
            status_code=429, detail="Too many password attempts. Try again later."
        )


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
