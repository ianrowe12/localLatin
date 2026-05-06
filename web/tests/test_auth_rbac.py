from __future__ import annotations

import csv
from pathlib import Path

from fastapi.testclient import TestClient

from web.app import create_app


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


def test_auth_session_persists_across_app_restart(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as client:
        unauthenticated = client.get("/api/auth/me")
        assert unauthenticated.status_code == 401

        registered = client.post(
            "/api/auth/register",
            json={
                "username": "pi",
                "display_name": "PI Scholar",
                "password": "correct horse battery staple",
            },
        )
        assert registered.status_code == 201
        assert registered.json()["role"] == "pi_admin"
        session_cookie = client.cookies.get("locallatin_session")
        assert session_cookie

    with _client(config_path) as restarted:
        restarted.cookies.set("locallatin_session", session_cookie)
        me = restarted.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["username"] == "pi"
        assert me.json()["display_name"] == "PI Scholar"
        assert me.json()["role"] == "pi_admin"


def test_reviewer_can_submit_but_cannot_access_admin_data(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as admin:
        assert admin.post(
            "/api/auth/register",
            json={
                "username": "pi",
                "display_name": "PI Scholar",
                "password": "correct horse battery staple",
            },
        ).status_code == 201

    with _client(config_path) as reviewer:
        registered = reviewer.post(
            "/api/auth/register",
            json={
                "username": "reviewer",
                "display_name": "External Reviewer",
                "password": "correct horse battery staple",
            },
        )
        assert registered.status_code == 201
        assert registered.json()["role"] == "reviewer"

        queries = reviewer.get("/api/queries")
        assert queries.status_code == 200

        feedback = reviewer.post(
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

        assert reviewer.get("/api/stats").status_code == 403
        assert reviewer.get("/api/feedback/export").status_code == 403


def test_pi_admin_can_access_admin_data_after_sign_in(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)

    with _client(config_path) as setup:
        setup.post(
            "/api/auth/register",
            json={
                "username": "pi",
                "display_name": "PI Scholar",
                "password": "correct horse battery staple",
            },
        )
        setup.post(
            "/api/auth/signout",
        )

    with _client(config_path) as client:
        signed_in = client.post(
            "/api/auth/signin",
            json={"username": "pi", "password": "correct horse battery staple"},
        )
        assert signed_in.status_code == 200
        assert signed_in.json()["role"] == "pi_admin"

        stats = client.get("/api/stats")
        assert stats.status_code == 200
        assert "reviews_by_reviewer" in stats.json()

        export = client.get("/api/feedback/export")
        assert export.status_code == 200
        assert "reviewer" in export.text.splitlines()[0]
