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
    labelled = root / "data" / "canon_labelled"
    predictions = root / "runs" / "active" / "resubmit" / "unlabelled"
    feedback = root / "runs" / "active" / "resubmit" / "webapp"

    unlabelled.mkdir(parents=True)
    (labelled / "candidate-a").mkdir(parents=True)
    (labelled / "candidate-b").mkdir(parents=True)
    predictions.mkdir(parents=True)
    feedback.mkdir(parents=True)

    for file_id in range(1, 5):
        (unlabelled / f"query-{file_id}.txt").write_text(
            f"query text {file_id}", encoding="utf-8"
        )
    (labelled / "candidate-a" / "a.txt").write_text("candidate text a", encoding="utf-8")
    (labelled / "candidate-b" / "b.txt").write_text("candidate text b", encoding="utf-8")

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
                "rank2_dir",
                "rank2_score",
            ],
        )
        writer.writeheader()
        for file_id in range(1, 5):
            writer.writerow(
                {
                    "file_id": file_id,
                    "filename": f"query-{file_id}.txt",
                    "model": "bowphs/LaTa",
                    "layer": 12,
                    "pooling": "mean",
                    "rank1_dir": "candidate-a",
                    "rank1_score": 0.91,
                    "rank2_dir": "candidate-b",
                    "rank2_score": 0.82,
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
""",
        encoding="utf-8",
    )
    return config_path


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(str(_write_fixture_data(tmp_path))))


def _sign_in(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "username": "pi",
            "display_name": "PI",
            "password": "correct horse battery staple",
        },
    )
    assert response.status_code == 201


def test_feedback_outcomes_drive_status_stats_and_export(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in(client)

        matched = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "matched_rank",
                "correct_rank": 2,
                "correct_dir": "candidate-b",
                "notes": "same Greek text, different translations",
                "reviewer": "PI",
            },
        )
        assert matched.status_code == 201
        assert matched.json()["outcome"] == "matched_rank"
        assert matched.json()["correct_rank"] == 2
        assert matched.json()["correct_dir"] == "candidate-b"

        none_of_top_k = client.post(
            "/api/feedback",
            json={
                "query_id": 2,
                "model_slug": "bowphs/LaTa",
                "outcome": "none_of_top_k",
                "correct_rank": 0,
                "notes": "no candidate is acceptable",
                "reviewer": "PI",
            },
        )
        assert none_of_top_k.status_code == 201
        assert none_of_top_k.json()["outcome"] == "none_of_top_k"
        assert none_of_top_k.json()["correct_rank"] == 0
        assert none_of_top_k.json()["correct_dir"] is None

        skipped = client.post(
            "/api/feedback",
            json={
                "query_id": 3,
                "model_slug": "bowphs/LaTa",
                "outcome": "skipped",
                "notes": "needs reference work",
                "reviewer": "PI",
            },
        )
        assert skipped.status_code == 201
        assert skipped.json()["outcome"] == "skipped"
        assert skipped.json()["correct_rank"] is None
        assert skipped.json()["correct_dir"] is None

        first_actionable = client.get("/api/queries/next")
        assert first_actionable.status_code == 200
        assert first_actionable.json() == {"file_id": 4}

        reviewed = client.get("/api/queries", params={"status": "reviewed"}).json()
        assert [item["file_id"] for item in reviewed["items"]] == [1, 2]
        assert {item["review_status"] for item in reviewed["items"]} == {"reviewed"}

        skipped_list = client.get("/api/queries", params={"status": "skipped"}).json()
        assert [item["file_id"] for item in skipped_list["items"]] == [3]
        assert skipped_list["items"][0]["review_status"] == "skipped"

        unreviewed = client.get("/api/queries", params={"status": "unreviewed"}).json()
        assert [item["file_id"] for item in unreviewed["items"]] == [4]

        next_after_skipped = client.get("/api/queries/next", params={"after": 3})
        assert next_after_skipped.status_code == 200
        assert next_after_skipped.json() == {"file_id": 4}

        next_after_reviewed = client.get("/api/queries/next", params={"after": 1})
        assert next_after_reviewed.status_code == 200
        assert next_after_reviewed.json() == {"file_id": 4}

        final_review = client.post(
            "/api/feedback",
            json={
                "query_id": 4,
                "model_slug": "bowphs/LaTa",
                "outcome": "matched_rank",
                "correct_rank": 1,
                "correct_dir": "candidate-a",
                "notes": "last unreviewed item",
                "reviewer": "PI",
            },
        )
        assert final_review.status_code == 201
        assert client.get("/api/queries/next").json() == {"file_id": None}

        stats = client.get("/api/stats").json()
        assert stats["reviewed_count"] == 3
        assert stats["skipped_count"] == 1
        assert stats["unreviewed_count"] == 0
        assert stats["feedback_count"] == 4
        assert stats["outcome_distribution"] == {
            "matched_rank": 2,
            "none_of_top_k": 1,
            "skipped": 1,
        }
        assert stats["rank_distribution"]["2"] == 1
        assert stats["rank_distribution"]["1"] == 1
        assert stats["rank_distribution"]["none_of_top_k"] == 1
        assert stats["rank_distribution"]["skipped"] == 1
        assert stats["next_unreviewed_ids"] == []
        assert [item["file_id"] for item in stats["needs_attention"]] == [3, 2]
        assert {
            item["outcome"] for item in stats["needs_attention"]
        } == {"skipped", "none_of_top_k"}
        assert stats["recent_reviews"][0]["reviewer"] == "PI"
        assert stats["recent_reviews"][0]["outcome"] == "matched_rank"

        exported = client.get("/api/feedback/export").text
        rows = list(csv.DictReader(exported.splitlines()))
        assert rows[0]["filename"] == "query-1.txt"
        assert [row["outcome"] for row in rows] == [
            "matched_rank",
            "none_of_top_k",
            "skipped",
            "matched_rank",
        ]
        assert rows[0]["correct_dir"] == "candidate-b"

        skipped_export = client.get(
            "/api/feedback/export", params={"status": "skipped"}
        )
        skipped_rows = list(csv.DictReader(skipped_export.text.splitlines()))
        assert [row["query_id"] for row in skipped_rows] == ["3"]
        assert skipped_rows[0]["filename"] == "query-3.txt"

        none_export = client.get(
            "/api/feedback/export", params={"outcome": "none_of_top_k"}
        )
        none_rows = list(csv.DictReader(none_export.text.splitlines()))
        assert [row["query_id"] for row in none_rows] == ["2"]

        empty_export = client.get(
            "/api/feedback/export",
            params={"reviewer": "Missing Reviewer", "date_from": "2999-01-01"},
        )
        empty_lines = empty_export.text.splitlines()
        assert empty_lines == [",".join(rows[0].keys())]


def test_feedback_outcome_validation_rejects_ambiguous_payloads(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in(client)

        missing_rank = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "matched_rank",
                "reviewer": "PI",
            },
        )
        assert missing_rank.status_code == 422

        skipped_with_rank = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "skipped",
                "correct_rank": 1,
                "reviewer": "PI",
            },
        )
        assert skipped_with_rank.status_code == 422

        legacy_unresolved = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "legacy_unresolved",
                "reviewer": "PI",
            },
        )
        assert legacy_unresolved.status_code == 422


def test_old_feedback_database_is_migrated_idempotently(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                model_slug TEXT NOT NULL,
                correct_rank INTEGER,
                correct_dir TEXT,
                notes TEXT NOT NULL DEFAULT '',
                reviewer TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO feedback
                (query_id, timestamp, model_slug, correct_rank, correct_dir, notes, reviewer)
            VALUES (?, datetime('now'), ?, ?, ?, ?, ?)
            """,
            [
                (1, "bowphs_LaTa", 1, "candidate-a", "accepted", "PI"),
                (2, "bowphs_LaTa", 0, None, "none", "PI"),
                (3, "bowphs_LaTa", None, None, "old ambiguous row", "PI"),
            ],
        )
        conn.commit()

    async def exercise() -> tuple[str, dict]:
        db = FeedbackDB(db_path)
        await db.connect()
        await db.connect()
        csv_data = await db.export_csv()
        stats = await db.get_stats()
        await db.close()
        return csv_data, stats

    exported, stats = asyncio.run(exercise())
    rows = list(csv.DictReader(exported.splitlines()))
    assert [row["outcome"] for row in rows] == [
        "matched_rank",
        "none_of_top_k",
        "legacy_unresolved",
    ]
    assert rows[2]["correct_rank"] == ""
    assert stats["outcome_distribution"] == {
        "legacy_unresolved": 1,
        "matched_rank": 1,
        "none_of_top_k": 1,
    }
    assert stats["reviewed_count"] == 2
    assert stats["unresolved_count"] == 1
