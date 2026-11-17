"""Admin queue, append-only deciding answers and reviewer privacy (issue #294)."""

from __future__ import annotations

import asyncio
import csv
import io
import sqlite3
from pathlib import Path

import pytest

from web.dependencies import get_db
from web.services.feedback_db import (
    FeedbackDB,
    MissingWitnessThresholdError,
    corpus_digest,
)
from web.tests.test_ccl_key_export import _pick, _put, _typed
from web.tests.test_reviewer_dirs import _client, _register


async def _seed() -> None:
    db = get_db()
    await _pick(db, "candidate-a", query_id=0, notes="Check the ending")
    await _typed(db, None, query_id=0, account=2, reviewer="Fake Two")
    await _pick(db, "candidate-a", query_id=1)
    await _put(
        db, query_id=2, outcome="matched_rank", correct_rank=11,
        correct_dir="reviewer-dir-old", notes="Unresolved pick",
    )
    await _put(db, query_id=3, outcome="skipped", notes="Only a skip")


def test_queue_reuses_export_resolution_filters_and_pagination(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "Fake Admin")
        asyncio.run(_seed())
        queue = client.get("/api/feedback/review_queue", params={"filter": "all", "page_size": 2})
        assert queue.status_code == 200
        body = queue.json()
        assert body["counts"] == {
            "all": 3, "conflict": 1, "needs_check": 1,
            "with_notes": 2, "reviewed_once": 1, "decided": 0,
        }
        assert body["total"] == 3 and body["has_more"]
        second = client.get(
            "/api/feedback/review_queue", params={"filter": "all", "page": 2, "page_size": 2}
        ).json()
        assert [row["xml_id"] for row in second["items"]] == ["query-2"]
        assert not second["has_more"]
        exported = list(csv.DictReader(io.StringIO(client.get("/api/feedback/export/ccl_keys").text)))
        assert {row["xml_id"]: row["status"] for row in exported} == {
            row["xml_id"]: row["status"] for row in body["items"] + second["items"]
        }
        conflict = client.get("/api/feedback/review_queue").json()["items"][0]
        assert conflict["query_id"] == 0
        assert conflict["answers"][0] == {
            "reviewer": "Alice Example", "model_slug": "model_one",
            "variant": "sif_abtt", "keys": ["candidate-a"], "none_of_these": False,
            "unresolved": False, "notes": "Check the ending",
            "timestamp": "2026-11-01T10:00:00Z",
        }
        assert conflict["answers"][1]["none_of_these"]
        for filter_name, xml_id in (
            ("conflict", "query-0"), ("needs_check", "query-2"), ("reviewed_once", "query-1"),
        ):
            result = client.get("/api/feedback/review_queue", params={"filter": filter_name}).json()
            assert [row["xml_id"] for row in result["items"]] == [xml_id]
        assert client.get("/api/feedback/review_queue", params={"page": 0}).status_code == 422
        assert client.get("/api/feedback/review_queue", params={"page_size": 101}).status_code == 422
    finally:
        client.__exit__(None, None, None)


def test_queue_shows_only_current_counted_answers_across_models(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "Fake Admin")

        async def seed() -> None:
            db = get_db()
            await _pick(db, "candidate-b", query_id=0, notes="Old note")
            await _pick(db, "candidate-a", query_id=0, ts="2026-11-02 10:00:00")
            await _pick(db, "candidate-b", query_id=0, model="model_two")
            await _put(db, query_id=0, outcome="skipped", notes="Later skip", ts="2026-11-03 10:00:00")

        asyncio.run(seed())
        item = client.get("/api/feedback/review_queue/0").json()
        assert item["status"] == "conflict"
        assert item["reviewer_count"] == 1
        assert len(item["answers"]) == 2
        assert all(not answer["notes"] for answer in item["answers"])
        assert client.get("/api/feedback/review_queue", params={"filter": "with_notes"}).json()["total"] == 0
    finally:
        client.__exit__(None, None, None)


def test_deciding_answers_append_and_newest_overrides_export(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "Fake Admin")
        asyncio.run(_seed())
        db_path = get_db().db_path
        with sqlite3.connect(db_path) as connection:
            before = connection.execute("SELECT * FROM feedback ORDER BY id").fetchall()
            connection.executescript(
                """CREATE TRIGGER no_decision_update BEFORE UPDATE ON deciding_answers
                       BEGIN SELECT RAISE(ABORT, 'append only'); END;
                   CREATE TRIGGER no_decision_delete BEFORE DELETE ON deciding_answers
                       BEGIN SELECT RAISE(ABORT, 'append only'); END;"""
            )
        first = client.post(
            "/api/feedback/decisions/0", json={"ccl_key": "  CTOU.567.16  ", "notes": " Checked "}
        )
        assert first.status_code == 201
        assert first.json()["xml_id"] == "query-0"
        assert first.json()["ccl_key"] == "CTOU.567.16"
        assert first.json()["notes"] == "Checked"
        item = client.get("/api/feedback/review_queue/0").json()
        assert item["status"] == "agreed" and item["reviewer_status"] == "conflict"
        assert item["deciding_answer"] == first.json()
        assert client.get("/api/feedback/review_queue").json()["total"] == 0
        assert client.get("/api/feedback/review_queue", params={"filter": "decided"}).json()["total"] == 1
        exported = list(csv.DictReader(io.StringIO(client.get("/api/feedback/export/ccl_keys").text)))[0]
        assert (exported["ccl_key"], exported["status"], exported["decided"], exported["reviewer_status"]) == (
            "CTOU.567.16", "agreed", "1", "conflict",
        )
        second = client.post("/api/feedback/decisions/0", json={"none_of_these": True})
        assert second.status_code == 201
        assert second.json()["id"] > first.json()["id"]
        item = client.get("/api/feedback/review_queue/0").json()
        assert item["status"] == "no_key"
        assert item["deciding_answer"]["id"] == second.json()["id"]
        assert len(item["answers"]) == 2
        with sqlite3.connect(db_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM deciding_answers").fetchone()[0] == 2
            assert connection.execute("SELECT * FROM feedback ORDER BY id").fetchall() == before
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        # A final answer can also identify an item that has no counted reviews.
        assert client.post("/api/feedback/decisions/3", json={"ccl_key": "CTOU.567.16"}).status_code == 201
        item = client.get("/api/feedback/review_queue/3").json()
        assert item["reviewer_status"] == "" and item["answers"] == []
        assert client.get("/api/feedback/review_queue/999").status_code == 404
        assert client.post("/api/feedback/decisions/999", json={"none_of_these": True}).status_code == 404
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize("body", [
    {}, {"ccl_key": " "}, {"ccl_key": "CTOU.567.16", "none_of_these": True},
    {"ccl_key": "\t=1+1"}, {"ccl_key": "CTOU|567"}, {"none_of_these": True, "notes": "+1"},
    {"none_of_these": True, "notes": "Note | other"}, {"ccl_key": "x" * 201},
    {"none_of_these": True, "notes": "x" * 10001},
])
def test_invalid_deciding_answers_write_nothing(tmp_path: Path, body: dict) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "Fake Admin")
        response = client.post("/api/feedback/decisions/0", json=body)
        assert response.status_code == 422
        assert client.get("/api/feedback/review_queue", params={"filter": "decided"}).json()["total"] == 0
    finally:
        client.__exit__(None, None, None)


def test_reviewer_is_forbidden_and_existing_responses_gain_no_answers(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "Fake Admin")
        admin_cookie = client.cookies.get("locallatin_session")
        asyncio.run(_seed())

        async def seed_visible_model() -> None:
            await _pick(
                get_db(), "candidate-b", query_id=0, model="bowphs_LaTa",
                notes="Existing shared note", reviewer="Other Fake Reviewer", account=1,
            )

        asyncio.run(seed_visible_model())
        client.cookies.clear()
        reviewer = _register(client, "reviewer", "Fake Reviewer")["account"]["id"]
        client.cookies.set("locallatin_session", admin_cookie)
        assert client.post(f"/api/auth/accounts/{reviewer}/approve").status_code == 200
        client.cookies.clear()
        assert client.post("/api/auth/signin", json={
            "username": "reviewer", "password": "correct horse battery staple",
        }).status_code == 200
        reviewer_cookie = client.cookies.get("locallatin_session")
        paths = [
            "/api/feedback/latest?query_id=0&model=bowphs_LaTa",
            "/api/query/0", "/api/query/0/predictions?model=bowphs_LaTa", "/api/queries",
        ]
        before = {path: client.get(path).json() for path in paths}
        prefill = before[paths[0]]
        assert prefill["notes"] == "Existing shared note"
        assert prefill["outcome"] == "legacy_unresolved"
        assert prefill["correct_rank"] is None and prefill["correct_dir"] is None
        assert prefill["selected_ranks"] is None and prefill["ccl_key"] is None
        client.cookies.set("locallatin_session", admin_cookie)
        assert client.post("/api/feedback/decisions/0", json={
            "ccl_key": "SECRET.123.1", "notes": "Admin-only reasoning",
        }).status_code == 201
        client.cookies.set("locallatin_session", reviewer_cookie)
        for path in ("/api/feedback/review_queue", "/api/feedback/review_queue/0"):
            assert client.get(path).status_code == 403
        assert client.post("/api/feedback/decisions/0", json={"none_of_these": True}).status_code == 403
        assert client.get("/api/feedback/export/ccl_keys").status_code == 403
        after = {path: client.get(path).json() for path in paths}
        assert after == before
        assert "SECRET.123.1" not in str(after)
        assert "Admin-only reasoning" not in str(after)
        client.cookies.clear()
        assert client.get("/api/feedback/review_queue").status_code == 401
        assert client.post("/api/feedback/decisions/0", json={"none_of_these": True}).status_code == 401
    finally:
        client.__exit__(None, None, None)


def test_decision_only_witness_rekeys_and_missing_threshold_protects_it(tmp_path: Path) -> None:
    path = tmp_path / "feedback.db"
    original = {0: "w-a.txt", 1: "w-b.txt"}
    moved = {0: "w-0.txt", 1: "w-a.txt", 2: "w-b.txt"}
    removed = {0: "w-0.txt", 1: "w-a.txt"}

    async def run() -> None:
        db = FeedbackDB(path)
        await db.connect()
        try:
            await db.adopt_corpus(original)
            saved = await db.insert_deciding_answer(
                query_id=1, reviewer_account_id=1, reviewer="Fake Admin",
                ccl_key="CTOU.567.16", notes="Only a final answer",
            )
            result = await db.adopt_corpus(moved)
            assert result["changed"] and result["change"]["moved"] == 1
            assert await db.get_deciding_answers() == [saved]
            assert db._served_query_id(saved["query_id"], saved["xml_id"]) == 2
            with pytest.raises(MissingWitnessThresholdError):
                await db.adopt_corpus(removed)
            await db.adopt_corpus(removed, accept_missing_reviewed_sha256=corpus_digest(removed))
            assert db._served_query_id(saved["query_id"], saved["xml_id"]) is None
            stats = await db.get_stats()
            assert stats["unserved_witness_count"] == stats["unserved_reviewed_count"] == 1
            rows = await db.ccl_key_assignments(filename_by_query=removed, labelled_dir_names=[])
            assert [(row["xml_id"], row["ccl_key"]) for row in rows] == [("w-b", "CTOU.567.16")]
            await db.adopt_corpus(moved)
            assert db._served_query_id(saved["query_id"], saved["xml_id"]) == 2
        finally:
            await db.close()
        db = FeedbackDB(path)
        await db.connect()
        try:
            assert not (await db.adopt_corpus(moved))["changed"]
            assert await db.get_deciding_answers() == [saved]
        finally:
            await db.close()

    asyncio.run(run())
    backups = list(tmp_path.glob("feedback.db.pre-corpus-change-*"))
    assert backups
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("SELECT xml_id FROM deciding_answers").fetchone() == ("w-b",)


def test_read_only_pre_decision_database_exports_without_migration(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"

    async def run() -> None:
        db = FeedbackDB(path)
        await db.connect()
        await _pick(db, "CTOU.567.16", query_id=0)
        await db.close()
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TABLE deciding_answers")
            connection.execute("PRAGMA user_version = 2")
        before = path.read_bytes()
        await db.connect(read_only=True)
        try:
            rows = await db.ccl_key_assignments(
                filename_by_query={0: "w-a.txt"}, labelled_dir_names=["CTOU.567.16"],
            )
            assert rows[0]["ccl_key"] == "CTOU.567.16"
            assert rows[0]["decided"] == 0 and rows[0]["reviewer_status"] == "agreed"
        finally:
            await db.close()
        assert path.read_bytes() == before

    asyncio.run(run())


def test_missing_decisions_table_in_current_schema_is_not_silently_ignored(tmp_path: Path) -> None:
    async def run() -> None:
        db = FeedbackDB(tmp_path / "feedback.db")
        await db.connect()
        try:
            assert db._db is not None
            await db._db.execute("DROP TABLE deciding_answers")
            with pytest.raises(sqlite3.OperationalError, match="no such table"):
                await db.get_deciding_answers()
        finally:
            await db.close()

    asyncio.run(run())
