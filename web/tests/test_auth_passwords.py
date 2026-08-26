from __future__ import annotations

import asyncio
import csv
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import create_app
from web.services.feedback_db import FeedbackDB

ADMIN_PASSWORD = "correct horse battery staple"
REVIEWER_PASSWORD = "reviewer horse battery staple"


def _write_fixture_data(root: Path, rate_limit_attempts: int = 10) -> Path:
    unlabelled = root / "data" / "canon_unlabelled"
    labelled = root / "data" / "canon_labelled" / "candidate-a"
    predictions = root / "runs" / "active" / "resubmit" / "unlabelled"
    feedback = root / "runs" / "active" / "resubmit" / "webapp"

    unlabelled.mkdir(parents=True)
    labelled.mkdir(parents=True)
    predictions.mkdir(parents=True)
    feedback.mkdir(parents=True)

    (unlabelled / "query-1.txt").write_text("query text 1", encoding="utf-8")
    (labelled / "a.txt").write_text("candidate text a", encoding="utf-8")

    with (predictions / "unlabelled_predictions_sif_abtt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_id",
                "filename",
                "model",
                "layer",
                "pooling",
                "rank1_dir",
                "rank1_score",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_id": 1,
                "filename": "query-1.txt",
                "model": "bowphs/LaTa",
                "layer": 12,
                "pooling": "mean",
                "rank1_dir": "candidate-a",
                "rank1_score": 0.91,
            }
        )

    config_path = root / "config.yaml"
    config_path.write_text(
        f"""
paths:
  data_root: "{root}"
  canon_unlabelled: "data/canon_unlabelled"
  canon_labelled: "data/canon_labelled"
  predictions_variant_pattern: "runs/active/resubmit/unlabelled/unlabelled_predictions_{{variant}}.csv"
  variants: ["sif_abtt"]
  default_variant: "sif_abtt"
  feedback_db: "runs/active/resubmit/webapp/feedback.db"
  ig_examples_csv: "missing/phase12f_examples.csv"
  ig_artifacts_dir: "missing/artifacts"
auth:
  secure_cookies: false
  session_days: 14
  password_rate_limit_max_attempts: {rate_limit_attempts}
  password_rate_limit_window_seconds: 900
""",
        encoding="utf-8",
    )
    return config_path


def _client(config_path: Path) -> TestClient:
    return TestClient(create_app(str(config_path)))


def _register_admin(client: TestClient) -> str:
    response = client.post(
        "/api/auth/register",
        json={
            "username": "pi",
            "display_name": "PI Scholar",
            "password": ADMIN_PASSWORD,
        },
    )
    assert response.status_code == 201
    assert response.json()["account"]["must_change_password"] is False
    cookie = client.cookies.get("locallatin_session")
    assert cookie
    return cookie


def _create_reviewer(client: TestClient) -> int:
    created = client.post(
        "/api/auth/accounts",
        json={
            "username": "reviewer",
            "display_name": "External Reviewer",
            "role": "reviewer",
            "password": REVIEWER_PASSWORD,
        },
    )
    assert created.status_code == 201
    return created.json()["account"]["id"]


def _sign_in(client: TestClient, username: str, password: str) -> str:
    client.cookies.clear()
    response = client.post(
        "/api/auth/signin", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    cookie = client.cookies.get("locallatin_session")
    assert cookie
    return cookie


def test_change_password_keeps_current_session_and_revokes_the_others(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        first_session = _register_admin(client)
        second_session = _sign_in(client, "pi", ADMIN_PASSWORD)
        assert first_session != second_session

        changed = client.post(
            "/api/auth/change_password",
            json={
                "current_password": ADMIN_PASSWORD,
                "new_password": "a brand new passphrase",
            },
        )
        assert changed.status_code == 200, changed.text
        assert changed.json()["username"] == "pi"
        assert changed.json()["must_change_password"] is False

        # The tab that made the change stays signed in.
        client.cookies.clear()
        client.cookies.set("locallatin_session", second_session)
        assert client.get("/api/auth/me").status_code == 200

        # Every other session of that account is revoked.
        client.cookies.clear()
        client.cookies.set("locallatin_session", first_session)
        assert client.get("/api/auth/me").status_code == 401

        client.cookies.clear()
        stale = client.post(
            "/api/auth/signin", json={"username": "pi", "password": ADMIN_PASSWORD}
        )
        assert stale.status_code == 401
        _sign_in(client, "pi", "a brand new passphrase")


def test_change_password_rejects_a_wrong_current_password(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        _register_admin(client)
        response = client.post(
            "/api/auth/change_password",
            json={
                "current_password": "not the right password",
                "new_password": "a brand new passphrase",
            },
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "Current password is incorrect"

        # The old password still works, so nothing was rehashed.
        _sign_in(client, "pi", ADMIN_PASSWORD)


def test_change_password_rejects_a_too_short_new_password(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        _register_admin(client)
        response = client.post(
            "/api/auth/change_password",
            json={"current_password": ADMIN_PASSWORD, "new_password": "short123"},
        )
        assert response.status_code == 422

        reused = client.post(
            "/api/auth/change_password",
            json={"current_password": ADMIN_PASSWORD, "new_password": ADMIN_PASSWORD},
        )
        assert reused.status_code == 400

        _sign_in(client, "pi", ADMIN_PASSWORD)


def test_change_password_throttles_wrong_guesses_only(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path, rate_limit_attempts=2)

    with _client(config_path) as client:
        _register_admin(client)

        # Wrong guesses fill the window and are then refused outright.
        for _ in range(2):
            attempt = client.post(
                "/api/auth/change_password",
                json={"current_password": "wrong", "new_password": "long enough pass"},
            )
            assert attempt.status_code == 403
        throttled = client.post(
            "/api/auth/change_password",
            json={"current_password": "wrong", "new_password": "long enough pass"},
        )
        assert throttled.status_code == 429
        assert int(throttled.headers["retry-after"]) > 0

        # The correct password still gets through: the window counts failed
        # attempts only, so an account can always clear its own gate.
        changed = client.post(
            "/api/auth/change_password",
            json={
                "current_password": ADMIN_PASSWORD,
                "new_password": "long enough pass",
            },
        )
        assert changed.status_code == 200

        # ...and the success cleared the counter.
        retry = client.post(
            "/api/auth/change_password",
            json={"current_password": "wrong", "new_password": "another long pass"},
        )
        assert retry.status_code == 403


def test_forced_change_survives_ten_wrong_temp_password_attempts(
    tmp_path: Path,
) -> None:
    """The reviewer's brick repro: ten wrong guesses, then the right one."""
    config_path = _write_fixture_data(tmp_path, rate_limit_attempts=10)

    with _client(config_path) as client:
        _register_admin(client)
        reviewer_id = _create_reviewer(client)
        reset = client.post(f"/api/auth/accounts/{reviewer_id}/reset_password")
        temporary_password = reset.json()["temporary_password"]

        _sign_in(client, "reviewer", temporary_password)
        for _ in range(10):
            wrong = client.post(
                "/api/auth/change_password",
                json={
                    "current_password": "not the temp password",
                    "new_password": "reviewer chosen passphrase",
                },
            )
            assert wrong.status_code in (403, 429)

        recovered = client.post(
            "/api/auth/change_password",
            json={
                "current_password": temporary_password,
                "new_password": "reviewer chosen passphrase",
            },
        )
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["must_change_password"] is False
        assert client.get("/api/queries").status_code == 200


def test_admin_cannot_reset_their_own_password(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        _register_admin(client)
        me = client.get("/api/auth/me").json()

        refused = client.post(f"/api/auth/accounts/{me['id']}/reset_password")
        assert refused.status_code == 400
        assert "Change password" in refused.json()["detail"]

        # The admin is still signed in and their password is untouched.
        assert client.get("/api/auth/me").status_code == 200
        assert client.get("/api/stats").status_code == 200
        _sign_in(client, "pi", ADMIN_PASSWORD)


def test_signin_throttles_failures_without_locking_the_owner_out(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_data(tmp_path, rate_limit_attempts=3)

    with _client(config_path) as client:
        _register_admin(client)
        client.cookies.clear()

        for _ in range(3):
            wrong = client.post(
                "/api/auth/signin",
                json={"username": "pi", "password": "wrong password"},
            )
            assert wrong.status_code == 401
        throttled = client.post(
            "/api/auth/signin",
            json={"username": "pi", "password": "wrong password"},
        )
        assert throttled.status_code == 429
        assert int(throttled.headers["retry-after"]) > 0

        # A guesser cannot lock the owner out: the correct password is verified
        # before the window is consulted.
        _sign_in(client, "pi", ADMIN_PASSWORD)

        # A different client address carries its own window.
        client.cookies.clear()
        elsewhere = client.post(
            "/api/auth/signin",
            json={"username": "pi", "password": "wrong password"},
            headers={"X-Forwarded-For": "203.0.113.9"},
        )
        assert elsewhere.status_code == 401


def test_change_password_floor_matches_registration(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        _register_admin(client)
        assert len("elevenchars") == 11
        eleven = client.post(
            "/api/auth/change_password",
            json={"current_password": ADMIN_PASSWORD, "new_password": "elevenchars"},
        )
        assert eleven.status_code == 422

        assert len("twelvecharsx") == 12
        twelve = client.post(
            "/api/auth/change_password",
            json={"current_password": ADMIN_PASSWORD, "new_password": "twelvecharsx"},
        )
        assert twelve.status_code == 200


def test_admin_reset_forces_a_change_before_any_other_route(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        admin_session = _register_admin(client)
        reviewer_id = _create_reviewer(client)

        reviewer_session = _sign_in(client, "reviewer", REVIEWER_PASSWORD)
        assert client.get("/api/queries").status_code == 200

        client.cookies.clear()
        client.cookies.set("locallatin_session", admin_session)
        reset = client.post(f"/api/auth/accounts/{reviewer_id}/reset_password")
        assert reset.status_code == 200, reset.text
        temporary_password = reset.json()["temporary_password"]
        assert isinstance(temporary_password, str) and len(temporary_password) >= 12
        assert reset.json()["account"]["must_change_password"] is True

        # The reviewer's live session is gone and the old password is dead.
        client.cookies.clear()
        client.cookies.set("locallatin_session", reviewer_session)
        assert client.get("/api/queries").status_code == 401
        client.cookies.clear()
        assert (
            client.post(
                "/api/auth/signin",
                json={"username": "reviewer", "password": REVIEWER_PASSWORD},
            ).status_code
            == 401
        )

        # Signing in with the temp password works but gates every other route.
        _sign_in(client, "reviewer", temporary_password)
        me = client.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["must_change_password"] is True

        gated = client.get("/api/queries")
        assert gated.status_code == 403
        assert gated.json()["detail"] == "Password change required before continuing"
        assert (
            client.get(
                "/api/query/1/predictions", params={"model": "bowphs/LaTa"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/feedback",
                json={
                    "query_id": 1,
                    "model_slug": "bowphs/LaTa",
                    "correct_rank": 1,
                    "correct_dir": "candidate-a",
                },
            ).status_code
            == 403
        )

        changed = client.post(
            "/api/auth/change_password",
            json={
                "current_password": temporary_password,
                "new_password": "reviewer chosen passphrase",
            },
        )
        assert changed.status_code == 200
        assert changed.json()["must_change_password"] is False

        assert client.get("/api/queries").status_code == 200
        assert client.get("/api/auth/me").json()["must_change_password"] is False

        _sign_in(client, "reviewer", "reviewer chosen passphrase")
        assert client.get("/api/queries").status_code == 200


def test_reset_password_is_pi_admin_only(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        admin_session = _register_admin(client)
        reviewer_id = _create_reviewer(client)

        _sign_in(client, "reviewer", REVIEWER_PASSWORD)
        forbidden = client.post(f"/api/auth/accounts/{reviewer_id}/reset_password")
        assert forbidden.status_code == 403
        assert client.post("/api/auth/accounts/1/reset_password").status_code == 403

        client.cookies.clear()
        assert (
            client.post(f"/api/auth/accounts/{reviewer_id}/reset_password").status_code
            == 401
        )

        client.cookies.set("locallatin_session", admin_session)
        assert client.post("/api/auth/accounts/999/reset_password").status_code == 404

        # The reviewer password is untouched by the refused attempts.
        _sign_in(client, "reviewer", REVIEWER_PASSWORD)


def test_legacy_accounts_migrate_with_must_change_password_off(tmp_path: Path) -> None:
    """A pre-flag database gains the column, defaulted to 0, never forced."""
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'reviewer',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_login_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO accounts (username, display_name, password_hash, role)
            VALUES ('legacy_pi', 'Legacy PI', 'not-a-real-hash', 'pi_admin')
            """
        )

    async def exercise() -> list[dict]:
        db = FeedbackDB(db_path)
        await db.connect()
        try:
            return await db.list_accounts()
        finally:
            await db.close()

    accounts = asyncio.run(exercise())
    assert accounts[0]["username"] == "legacy_pi"
    assert accounts[0]["must_change_password"] is False

    with sqlite3.connect(db_path) as conn:
        columns = {row[1]: row for row in conn.execute("PRAGMA table_info(accounts)")}
        stored = conn.execute("SELECT must_change_password FROM accounts").fetchone()
    assert "must_change_password" in columns
    # NOT NULL DEFAULT 0, so the flag can never read back as NULL.
    assert columns["must_change_password"][3] == 1
    assert stored[0] == 0


def test_connect_fails_closed_when_the_flag_column_is_missing(tmp_path: Path) -> None:
    """If the migration ever stopped adding the gate column, startup must die."""
    db_path = tmp_path / "unmigrated.db"

    async def exercise() -> None:
        db = FeedbackDB(db_path)

        async def skip_account_columns() -> None:
            assert db._db is not None
            await db._db.execute("DROP TABLE IF EXISTS accounts")
            await db._db.execute(
                """
                CREATE TABLE accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'reviewer',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    approval_status TEXT NOT NULL DEFAULT 'approved',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    last_login_at TEXT
                )
                """
            )

        db._migrate = skip_account_columns  # type: ignore[method-assign]
        try:
            await db.connect()
        finally:
            await db.close()

    with pytest.raises(RuntimeError, match="must_change_password"):
        asyncio.run(exercise())
