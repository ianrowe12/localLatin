"""Changing an account's role (issue #265).

`POST /api/auth/accounts/{id}/role` and `FeedbackDB.set_account_role`. Pins the
guards (self-change 400, last PI/admin 409, unknown 404, unchanged role is a
write-free 200, reviewer callers 403), the `account_role_changes` audit row,
and that the new role applies to the account's next request on the session it
already holds.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import create_app
from web.dependencies import require_pi_admin
from web.exceptions import LastPiAdminError
from web.models import UserPublic
from web.services.feedback_db import FeedbackDB
from web.tests.test_auth_passwords import (
    ADMIN_PASSWORD,
    REVIEWER_PASSWORD,
    _client,
    _create_reviewer,
    _register_admin,
    _sign_in,
    _write_fixture_data,
)

SESSION_COOKIE = "locallatin_session"


def _db_path(root: Path) -> Path:
    return root / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"


def _audit_rows(root: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(_db_path(root))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM account_role_changes ORDER BY id").fetchall()
    finally:
        conn.close()


def _as(client: TestClient, cookie: str) -> TestClient:
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE, cookie)
    return client


def test_promotion_applies_to_the_accounts_next_request_and_is_audited(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as client:
        admin_cookie = _register_admin(client)
        admin_id = client.get("/api/auth/me").json()["id"]
        reviewer_id = _create_reviewer(client)
        reviewer_cookie = _sign_in(client, "reviewer", REVIEWER_PASSWORD)

        # Before: the reviewer's session is refused on a PI/admin route.
        assert _as(client, reviewer_cookie).get("/api/stats").status_code == 403

        promoted = _as(client, admin_cookie).post(
            f"/api/auth/accounts/{reviewer_id}/role",
            json={"role": "pi_admin", "note": "co-PI on the project"},
        )
        assert promoted.status_code == 200, promoted.text
        body = promoted.json()
        assert body["id"] == reviewer_id
        assert body["role"] == "pi_admin"
        # The full AccountPublic shape comes back, not just UserPublic.
        assert body["approval_status"] == "approved"
        assert "created_at" in body and "is_active" in body

        # After: the SAME session (no re-sign-in) now carries the new role,
        # because the role is re-read from the DB on every request.
        _as(client, reviewer_cookie)
        assert client.get("/api/auth/me").json()["role"] == "pi_admin"
        assert client.get("/api/stats").status_code == 200
        assert client.get("/api/feedback/export").status_code == 200

    rows = _audit_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["account_id"] == reviewer_id
    assert row["old_role"] == "reviewer"
    assert row["new_role"] == "pi_admin"
    assert row["changed_by_account_id"] == admin_id
    assert row["note"] == "co-PI on the project"
    assert row["changed_at"]


def test_demotion_removes_admin_access_on_the_next_request(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as client:
        admin_cookie = _register_admin(client)
        reviewer_id = _create_reviewer(client)
        assert (
            client.post(
                f"/api/auth/accounts/{reviewer_id}/role", json={"role": "pi_admin"}
            ).status_code
            == 200
        )
        promoted_cookie = _sign_in(client, "reviewer", REVIEWER_PASSWORD)
        assert client.get("/api/stats").status_code == 200

        demoted = _as(client, admin_cookie).post(
            f"/api/auth/accounts/{reviewer_id}/role", json={"role": "reviewer"}
        )
        assert demoted.status_code == 200, demoted.text
        assert demoted.json()["role"] == "reviewer"

        _as(client, promoted_cookie)
        # Still signed in (no session was revoked) but no longer an admin.
        assert client.get("/api/auth/me").status_code == 200
        assert client.get("/api/auth/me").json()["role"] == "reviewer"
        assert client.get("/api/stats").status_code == 403
        assert client.get("/api/feedback/export").status_code == 403
        assert client.get("/api/queries").status_code == 200

    rows = _audit_rows(tmp_path)
    assert [(r["old_role"], r["new_role"]) for r in rows] == [
        ("reviewer", "pi_admin"),
        ("pi_admin", "reviewer"),
    ]


def test_changing_your_own_role_is_refused(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as client:
        _register_admin(client)
        me = client.get("/api/auth/me").json()
        # A second admin exists, so the 400 is the self rule, not the last-admin rule.
        reviewer_id = _create_reviewer(client)
        assert (
            client.post(
                f"/api/auth/accounts/{reviewer_id}/role", json={"role": "pi_admin"}
            ).status_code
            == 200
        )

        response = client.post(
            f"/api/auth/accounts/{me['id']}/role", json={"role": "reviewer"}
        )
        assert response.status_code == 400
        assert "Another PI/admin" in response.json()["detail"]
        assert client.get("/api/auth/me").json()["role"] == "pi_admin"
    assert len(_audit_rows(tmp_path)) == 1  # only the promotion above


def test_unknown_account_is_404(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as client:
        _register_admin(client)
        response = client.post("/api/auth/accounts/9999/role", json={"role": "pi_admin"})
        assert response.status_code == 404
    assert _audit_rows(tmp_path) == []


def test_unchanged_role_is_a_200_that_writes_nothing(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as client:
        _register_admin(client)
        reviewer_id = _create_reviewer(client)
        conn = sqlite3.connect(_db_path(tmp_path))
        before = conn.execute(
            "SELECT role, updated_at FROM accounts WHERE id = ?", (reviewer_id,)
        ).fetchone()
        conn.close()

        response = client.post(
            f"/api/auth/accounts/{reviewer_id}/role", json={"role": "reviewer"}
        )
        assert response.status_code == 200
        assert response.json()["role"] == "reviewer"

        conn = sqlite3.connect(_db_path(tmp_path))
        after = conn.execute(
            "SELECT role, updated_at FROM accounts WHERE id = ?", (reviewer_id,)
        ).fetchone()
        conn.close()
        assert after == before
    assert _audit_rows(tmp_path) == []


def test_reviewers_cannot_change_roles(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as client:
        _register_admin(client)
        admin_id = client.get("/api/auth/me").json()["id"]
        reviewer_id = _create_reviewer(client)
        _sign_in(client, "reviewer", REVIEWER_PASSWORD)
        # Neither self-promotion nor demoting somebody else.
        assert (
            client.post(
                f"/api/auth/accounts/{reviewer_id}/role", json={"role": "pi_admin"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                f"/api/auth/accounts/{admin_id}/role", json={"role": "reviewer"}
            ).status_code
            == 403
        )
    assert _audit_rows(tmp_path) == []


def test_an_unknown_role_is_a_422(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as client:
        _register_admin(client)
        reviewer_id = _create_reviewer(client)
        response = client.post(
            f"/api/auth/accounts/{reviewer_id}/role", json={"role": "superuser"}
        )
        assert response.status_code == 422
    assert _audit_rows(tmp_path) == []


def test_demoting_the_last_approved_pi_admin_is_409(tmp_path: Path) -> None:
    """Reached through the endpoint with an out-of-DB principal.

    A signed-in caller is itself an approved PI/admin and cannot target its own
    row, so through ordinary sessions the guard only fires under a race. The
    override stands in for that caller so the HTTP mapping is still pinned.
    """
    config_path = _write_fixture_data(tmp_path)
    app = create_app(str(config_path))
    with TestClient(app) as client:
        _register_admin(client)
        admin_id = client.get("/api/auth/me").json()["id"]
        app.dependency_overrides[require_pi_admin] = lambda: UserPublic(
            id=10_000,
            username="ops",
            display_name="Ops",
            role="pi_admin",
        )
        try:
            response = client.post(
                f"/api/auth/accounts/{admin_id}/role", json={"role": "reviewer"}
            )
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 409
        assert "last approved PI/admin" in response.json()["detail"]
        assert client.get("/api/auth/me").json()["role"] == "pi_admin"
    assert _audit_rows(tmp_path) == []


def test_last_admin_guard_ignores_pending_rejected_and_inactive_admins(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        db = FeedbackDB(tmp_path / "feedback.db")
        await db.connect()
        try:
            only = await db.create_account("pi", "PI", ADMIN_PASSWORD, "pi_admin")
            pending = await db.create_account(
                "pending-admin", "Pending", ADMIN_PASSWORD, "pi_admin",
                approval_status="pending",
            )
            rejected = await db.create_account(
                "rejected-admin", "Rejected", ADMIN_PASSWORD, "pi_admin"
            )
            await db.set_account_approval(rejected["id"], "rejected", only["id"])
            assert pending["approval_status"] == "pending"

            # None of the other three "admins" can sign in, so they do not count.
            with pytest.raises(LastPiAdminError):
                await db.set_account_role(only["id"], "reviewer", None)

            # Once a second approved admin exists the demotion goes through.
            second = await db.create_account("pi2", "PI 2", ADMIN_PASSWORD, "reviewer")
            account, changed = await db.set_account_role(
                second["id"], "pi_admin", only["id"], note="second admin"
            )
            assert changed and account["role"] == "pi_admin"
            account, changed = await db.set_account_role(only["id"], "reviewer", second["id"])
            assert changed and account["role"] == "reviewer"

            # Now pi2 is the last one.
            with pytest.raises(LastPiAdminError):
                await db.set_account_role(second["id"], "reviewer", None)

            # Demoting a pending admin is not a lockout risk, so it is allowed.
            account, changed = await db.set_account_role(pending["id"], "reviewer", None)
            assert changed and account["role"] == "reviewer"

            trail = await db.list_account_role_changes()
            assert [(r["account_id"], r["old_role"], r["new_role"]) for r in trail] == [
                (second["id"], "reviewer", "pi_admin"),
                (only["id"], "pi_admin", "reviewer"),
                (pending["id"], "pi_admin", "reviewer"),
            ]
            assert trail[0]["note"] == "second admin"
            assert trail[2]["changed_by_account_id"] is None
            assert await db.list_account_role_changes(only["id"]) == [trail[1]]

            assert await db.set_account_role(99_999, "reviewer", None) == (None, False)
            with pytest.raises(ValueError):
                await db.set_account_role(second["id"], "owner", None)
        finally:
            await db.close()

    asyncio.run(scenario())


def test_migration_adds_the_audit_table_without_touching_accounts(
    tmp_path: Path,
) -> None:
    """A database from before #265 gains the table; every account row survives."""
    path = tmp_path / "pre265.db"

    async def first_boot() -> None:
        db = FeedbackDB(path)
        await db.connect()
        await db.create_account("pi", "PI", ADMIN_PASSWORD, "pi_admin")
        await db.create_account("rev", "Rev", REVIEWER_PASSWORD, "reviewer")
        await db.close()

    asyncio.run(first_boot())
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE account_role_changes")
    conn.commit()
    accounts_before = conn.execute("SELECT * FROM accounts ORDER BY id").fetchall()
    conn.close()

    async def second_boot() -> None:
        db = FeedbackDB(path)
        await db.connect()
        assert await db.list_account_role_changes() == []
        await db.close()

    asyncio.run(second_boot())
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT * FROM accounts ORDER BY id").fetchall() == accounts_before
        columns = [r[1] for r in conn.execute("PRAGMA table_info(account_role_changes)")]
        assert columns == [
            "id",
            "account_id",
            "old_role",
            "new_role",
            "changed_by_account_id",
            "note",
            "changed_at",
        ]
    finally:
        conn.close()
