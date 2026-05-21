from __future__ import annotations

import asyncio
import csv
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from web.app import create_app
from web.services.feedback_db import FeedbackDB


def _write_fixture_data(root: Path) -> Path:
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

    with (predictions / "unlabelled_predictions.csv").open("w", newline="") as f:
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
  predictions_combined: "runs/active/resubmit/unlabelled/unlabelled_predictions.csv"
  feedback_db: "runs/active/resubmit/webapp/feedback.db"
  ig_examples_csv: "missing/phase12f_examples.csv"
  ig_artifacts_dir: "missing/artifacts"
auth:
  secure_cookies: false
  session_days: 14
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
            "password": "correct horse battery staple",
        },
    )
    assert response.status_code == 201
    assert response.json()["status"] == "approved"
    assert response.json()["account"]["role"] == "pi_admin"
    assert response.json()["account"]["approval_status"] == "approved"
    session_cookie = client.cookies.get("locallatin_session")
    assert session_cookie
    return session_cookie


def test_auth_session_persists_across_app_restart(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        unauthenticated = client.get("/api/auth/me")
        assert unauthenticated.status_code == 401

        session_cookie = _register_admin(client)

    with _client(config_path) as restarted:
        restarted.cookies.set("locallatin_session", session_cookie)
        me = restarted.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["username"] == "pi"
        assert me.json()["display_name"] == "PI Scholar"
        assert me.json()["role"] == "pi_admin"
        assert me.json()["approval_status"] == "approved"


def test_reviewer_requires_pi_approval_before_access(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        admin_cookie = _register_admin(client)
        client.cookies.clear()

        registered = client.post(
            "/api/auth/register",
            json={
                "username": "reviewer",
                "display_name": "External Reviewer",
                "password": "correct horse battery staple",
            },
        )
        assert registered.status_code == 201
        assert registered.json()["status"] == "pending_approval"
        assert registered.json()["account"]["role"] == "reviewer"
        assert registered.json()["account"]["approval_status"] == "pending"
        reviewer_id = registered.json()["account"]["id"]
        assert client.cookies.get("locallatin_session") is None

        assert client.get("/api/queries").status_code == 401
        pending_signin = client.post(
            "/api/auth/signin",
            json={"username": "reviewer", "password": "correct horse battery staple"},
        )
        assert pending_signin.status_code == 403
        assert pending_signin.json()["detail"] == "Account pending approval"

        client.cookies.set("locallatin_session", admin_cookie)
        pending_accounts = client.get("/api/auth/accounts", params={"status": "pending"})
        assert pending_accounts.status_code == 200
        assert [account["username"] for account in pending_accounts.json()] == ["reviewer"]

        approved = client.post(f"/api/auth/accounts/{reviewer_id}/approve")
        assert approved.status_code == 200
        assert approved.json()["approval_status"] == "approved"
        assert approved.json()["approved_by_account_id"] == 1

        client.cookies.clear()
        signed_in = client.post(
            "/api/auth/signin",
            json={"username": "reviewer", "password": "correct horse battery staple"},
        )
        assert signed_in.status_code == 200
        assert signed_in.json()["role"] == "reviewer"
        assert signed_in.json()["approval_status"] == "approved"

        queries = client.get("/api/queries")
        assert queries.status_code == 200

        feedback = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 1,
                "correct_dir": "candidate-a",
                "notes": "body reviewer should not be trusted",
                "reviewer": "Spoofed PI",
            },
        )
        assert feedback.status_code == 201
        assert feedback.json()["reviewer"] == "External Reviewer"

        assert client.get("/api/stats").status_code == 403
        assert client.get("/api/feedback/export").status_code == 403


def test_rejected_account_loses_existing_session(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        admin_cookie = _register_admin(client)
        client.cookies.clear()
        registered = client.post(
            "/api/auth/register",
            json={
                "username": "reviewer",
                "display_name": "External Reviewer",
                "password": "correct horse battery staple",
            },
        )
        reviewer_id = registered.json()["account"]["id"]

        client.cookies.set("locallatin_session", admin_cookie)
        assert client.post(f"/api/auth/accounts/{reviewer_id}/approve").status_code == 200

        client.cookies.clear()
        signed_in = client.post(
            "/api/auth/signin",
            json={"username": "reviewer", "password": "correct horse battery staple"},
        )
        assert signed_in.status_code == 200
        reviewer_cookie = client.cookies.get("locallatin_session")
        assert reviewer_cookie
        assert client.get("/api/queries").status_code == 200

        client.cookies.set("locallatin_session", admin_cookie)
        rejected = client.post(f"/api/auth/accounts/{reviewer_id}/reject")
        assert rejected.status_code == 200
        assert rejected.json()["approval_status"] == "rejected"

        client.cookies.clear()
        client.cookies.set("locallatin_session", reviewer_cookie)
        assert client.get("/api/queries").status_code == 401

        client.cookies.clear()
        rejected_signin = client.post(
            "/api/auth/signin",
            json={"username": "reviewer", "password": "correct horse battery staple"},
        )
        assert rejected_signin.status_code == 403
        assert rejected_signin.json()["detail"] == "Account registration was rejected"


def test_pi_admin_can_access_admin_data_after_sign_in(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as setup:
        _register_admin(setup)
        signed_out = setup.post(
            "/api/auth/signout",
        )
        assert signed_out.status_code == 204
        assert setup.get("/api/auth/me").status_code == 401

    with _client(config_path) as client:
        signed_in = client.post(
            "/api/auth/signin",
            json={"username": "pi", "password": "correct horse battery staple"},
        )
        assert signed_in.status_code == 200
        assert signed_in.json()["role"] == "pi_admin"
        assert signed_in.json()["approval_status"] == "approved"

        stats = client.get("/api/stats")
        assert stats.status_code == 200
        assert "reviews_by_reviewer" in stats.json()

        export = client.get("/api/feedback/export")
        assert export.status_code == 200
        assert "reviewer" in export.text.splitlines()[0]


def test_legacy_accounts_migrate_to_approved(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'reviewer' CHECK (role IN ('reviewer', 'pi_admin')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_login_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO accounts
                (username, display_name, password_hash, role)
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
    assert accounts[0]["approval_status"] == "approved"
    assert accounts[0]["is_active"] is True
