from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from web.app import create_app


def _write_fixture_data(root: Path) -> Path:
    unlabelled = root / "data" / "canon_unlabelled"
    labelled = root / "data" / "canon_labelled"
    predictions = root / "runs" / "active" / "resubmit" / "unlabelled"
    feedback = root / "runs" / "active" / "resubmit" / "webapp"

    unlabelled.mkdir(parents=True)
    predictions.mkdir(parents=True)
    feedback.mkdir(parents=True)
    (labelled / "candidate-a").mkdir(parents=True)
    (labelled / "candidate-b").mkdir(parents=True)

    (unlabelled / "query-1.txt").write_text(
        "query text from the unlabelled manuscript", encoding="utf-8"
    )
    (labelled / "candidate-a" / "a.txt").write_text(
        "candidate text a from the selected source", encoding="utf-8"
    )
    (labelled / "candidate-b" / "b.txt").write_text(
        "candidate text b should be outside top one packets", encoding="utf-8"
    )

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
                "rank2_dir",
                "rank2_score",
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
                "rank2_dir": "candidate-b",
                "rank2_score": 0.72,
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
""",
        encoding="utf-8",
    )
    return config_path


def _client(config_path: Path) -> TestClient:
    return TestClient(create_app(str(config_path)))


def _register_admin(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "username": "pi",
            "display_name": "PI Scholar",
            "password": "correct horse battery staple",
        },
    )
    assert response.status_code == 201


def test_pi_admin_can_generate_bounded_review_packet_pdf(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as client:
        _register_admin(client)
        feedback = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "matched_rank",
                "correct_rank": 1,
                "correct_dir": "candidate-a",
                "notes": "rank one note",
            },
        )
        assert feedback.status_code == 201

        response = client.get(
            "/api/packets/review/1",
            params={"model": "bowphs/LaTa", "top_k": 1},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "review_packet_query-1.txt_bowphs_LaTa.pdf" in response.headers[
            "content-disposition"
        ]

        pdf = fitz.open(stream=response.content, filetype="pdf")
        assert pdf.page_count >= 1
        assert pdf[0].get_text().strip()
        text = "\n".join(page.get_text() for page in pdf)
        assert "LocalLatin Review Packet" in text
        assert "query-1.txt" in text
        assert "query text from the unlabelled manuscript" in text
        assert "candidate-a" in text
        assert "candidate text a from the selected source" in text
        assert "rank one note" in text
        assert "Attribution/token-map data" in text
        assert "candidate-b" not in text
        # Rows are attributed by display name plus login, and the packet says
        # out loud that it spans every reviewer (issue #96).
        assert "PI Scholar (@pi)" in text
        assert "Reviewer Outcomes" in text
        assert "Every reviewer's rows are listed" in text


def test_packet_attributes_a_row_whose_account_is_gone(tmp_path: Path) -> None:
    """Pre-account rows and deleted accounts keep their display name.

    `reviewer_username` comes from a LEFT JOIN, so it is NULL whenever no
    account backs the row. The packet must still name somebody rather than
    printing an empty '(@)'.
    """
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as client:
        _register_admin(client)
        assert (
            client.post(
                "/api/feedback",
                json={
                    "query_id": 1,
                    "model_slug": "bowphs/LaTa",
                    "outcome": "matched_rank",
                    "correct_rank": 1,
                    "correct_dir": "candidate-a",
                    "notes": "note from a reviewer who has since left",
                },
            ).status_code
            == 201
        )

        # Sever the row from its account the way a pre-account row looks.
        db_path = tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("UPDATE feedback SET reviewer_account_id = NULL")
            connection.commit()
        finally:
            connection.close()

        response = client.get(
            "/api/packets/review/1",
            params={"model": "bowphs/LaTa", "top_k": 1},
        )
        assert response.status_code == 200
        pdf = fitz.open(stream=response.content, filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
        assert "PI Scholar" in text
        assert "(@" not in text
        assert "note from a reviewer who has since left" in text


def test_reviewers_cannot_generate_review_packet_pdf(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    with _client(config_path) as reviewer:
        _register_admin(reviewer)
        admin_cookie = reviewer.cookies.get("locallatin_session")
        assert admin_cookie
        reviewer.cookies.clear()
        registered = reviewer.post(
            "/api/auth/register",
            json={
                "username": "reviewer",
                "display_name": "External Reviewer",
                "password": "correct horse battery staple",
            },
        )
        assert registered.status_code == 201
        reviewer_id = registered.json()["account"]["id"]
        reviewer.cookies.set("locallatin_session", admin_cookie)
        assert reviewer.post(f"/api/auth/accounts/{reviewer_id}/approve").status_code == 200
        reviewer.cookies.clear()
        signed_in = reviewer.post(
            "/api/auth/signin",
            json={"username": "reviewer", "password": "correct horse battery staple"},
        )
        assert signed_in.status_code == 200
        response = reviewer.get(
            "/api/packets/review/1",
            params={"model": "bowphs/LaTa"},
        )
        assert response.status_code == 403
