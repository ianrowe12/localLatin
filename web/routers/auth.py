from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

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
    request: Request,
    response: Response,
    db: FeedbackDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> UserPublic:
    limit_key = _signin_limit_key(request, body.username)
    account, reason = await db.verify_account(body.username, body.password)
    if account is None:
        if reason == "pending":
            raise HTTPException(status_code=403, detail="Account pending approval")
        if reason == "rejected":
            raise HTTPException(status_code=403, detail="Account registration was rejected")
        if reason == "inactive":
            raise HTTPException(status_code=403, detail="Account is inactive")
        # Only a bad username/password counts as a guess.
        _record_failure(limiter, settings, limit_key)
        raise HTTPException(status_code=401, detail="Invalid username or password")
    limiter.reset(limit_key)
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
    limit_key = f"change:{current_user.id}"
    # Verification runs BEFORE the window is consulted, and only a wrong password
    # is ever recorded. This route is the sole escape hatch out of a forced-change
    # gate and there is no email recovery, so a correct password must always get
    # through: throttling a success would brick the account for the whole window.
    if not await db.verify_account_password(current_user.id, body.current_password):
        _record_failure(limiter, settings, limit_key)
        raise HTTPException(status_code=403, detail="Current password is incorrect")
    if body.new_password == body.current_password:
        # A validation error, not a failed guess: nothing recorded.
        raise HTTPException(
            status_code=400, detail="New password must differ from the current one"
        )
    limiter.reset(limit_key)
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
    logger.info(
        "Password changed account_id=%s forced=%s",
        current_user.id,
        current_user.must_change_password,
    )
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
    if account_id == current_user.id:
        # Self-reset is a lockout trap: it would kill the caller's own session and
        # leave the only copy of the temporary password in a browser tab, with no
        # email recovery behind it. Admins change their own password instead, and
        # a second PI/admin resets an admin who is locked out.
        raise HTTPException(
            status_code=400,
            detail=(
                "Use Change password for your own account. "
                "Another PI/admin must reset it for you."
            ),
        )
    limit_key = f"reset:{current_user.id}"
    _enforce_rate_limit(limiter, settings, limit_key)
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
    # Audit trail: ids only, never the generated password.
    logger.info(
        "Password reset actor_account_id=%s target_account_id=%s",
        current_user.id,
        account_id,
    )
    limiter.record(limit_key)
    return PasswordResetResponse(
        account=AccountPublic(**account),
        temporary_password=temporary_password,
    )


def _enforce_rate_limit(limiter: RateLimiter, settings: Settings, key: str) -> None:
    """Refuse the request when ``key`` has filled its window."""
    if limiter.allows(
        key,
        settings.auth.password_rate_limit_max_attempts,
        settings.auth.password_rate_limit_window_seconds,
    ):
        return
    raise _too_many_attempts(limiter, settings, key)


def _record_failure(limiter: RateLimiter, settings: Settings, key: str) -> None:
    """Count one failed attempt, or refuse when the window is already full.

    A refused attempt is not itself recorded, so a client hammering the endpoint
    cannot keep pushing its own window forward indefinitely.
    """
    if not limiter.allows(
        key,
        settings.auth.password_rate_limit_max_attempts,
        settings.auth.password_rate_limit_window_seconds,
    ):
        raise _too_many_attempts(limiter, settings, key)
    limiter.record(key)


def _too_many_attempts(
    limiter: RateLimiter, settings: Settings, key: str
) -> HTTPException:
    retry_after = limiter.retry_after(
        key, settings.auth.password_rate_limit_window_seconds
    )
    return HTTPException(
        status_code=429,
        detail=f"Too many failed attempts. Try again in {retry_after} seconds.",
        headers={"Retry-After": str(retry_after)},
    )


def _signin_limit_key(request: Request, username: str) -> str:
    """Throttle key for sign-in: the client address plus the account.

    Keyed per (client, account) rather than per account so a stranger cannot lock
    a reviewer out of their own account by guessing at it.

    Only proxy-written values are trusted. `deploy/nginx.conf` sets
    `X-Real-IP $remote_addr`, which nginx *overwrites* on every request, so it is
    the one address a client cannot choose. `X-Forwarded-For` there is
    `$proxy_add_x_forwarded_for`, which *appends* the peer to whatever the client
    sent: its leftmost hops are attacker-chosen, and only the rightmost one was
    written by the proxy. Taking the left hop would let one attacker rotate a
    fake address per request and never fill a window (while growing the key space
    unboundedly), so it is deliberately not used.

    This holds because the app binds 127.0.0.1 and is only reachable through that
    proxy; a deployment that exposes uvicorn directly must not trust either
    header.
    """
    real_ip = request.headers.get("x-real-ip", "").strip()
    if not real_ip:
        forwarded = request.headers.get("x-forwarded-for", "")
        hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
        real_ip = hops[-1] if hops else ""
    if not real_ip:
        real_ip = request.client.host if request.client else "unknown"
    return f"signin:{real_ip}:{username.strip().lower()}"


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
