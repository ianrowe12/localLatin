"""Creation invariants on disposable SQLite files, including competing writers."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from pathlib import Path

import aiosqlite
import httpx
import pytest
from fastapi import FastAPI

from web.app import create_app
from web.dependencies import get_db
from web.services.feedback_db import FeedbackDB
from web.tests.test_reviewer_dirs import MODEL_SLUG, _write_fixture_data


@asynccontextmanager
async def _running_app(config: Path) -> AsyncIterator[FastAPI]:
    app = create_app(str(config))
    async with app.router.lifespan_context(app):
        try:
            yield app
        finally:
            # The app lifespan skips close if an assertion exits through yield.
            await get_db().close()


def _rows(path: Path, table: str) -> list[tuple]:
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()


async def _create(db: FeedbackDB, seed: int, account: int = 1, cap: int = 50) -> dict:
    return await db.create_reviewer_dir(
        label=f"Group {seed}",
        seed_query_id=seed,
        model_slug=MODEL_SLUG,
        variant="sif_abtt",
        created_by=f"Reviewer {account}",
        created_by_account_id=account,
        max_per_account=cap,
    )


@pytest.mark.parametrize("same_seed", [True, False], ids=["global-seed", "account-cap"])
def test_concurrent_requests_from_approved_reviewers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_seed: bool
) -> None:
    from web.services import reviewer_dirs as svc

    monkeypatch.setattr(svc, "MAX_REVIEWER_DIRS_PER_ACCOUNT", 3)
    config = _write_fixture_data(tmp_path, n_queries=30)

    async def run() -> None:
        async with _running_app(config) as app:
            transport = httpx.ASGITransport(app=app)
            async with (
                httpx.AsyncClient(transport=transport, base_url="http://test") as admin,
                httpx.AsyncClient(transport=transport, base_url="http://test") as first,
                httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as second,
            ):
                credentials = {"password": "correct horse battery staple"}
                registration = await admin.post(
                    "/api/auth/register",
                    json={**credentials, "username": "pi", "display_name": "PI"},
                )
                assert registration.status_code == 201
                accounts = []
                for i, client in enumerate((first, second)):
                    body = {
                        **credentials,
                        "username": f"scholar{i}",
                        "display_name": f"Scholar {i}",
                    }
                    registration = await client.post("/api/auth/register", json=body)
                    assert registration.status_code == 201
                    account_id = registration.json()["account"]["id"]
                    approved = await admin.post(
                        f"/api/auth/accounts/{account_id}/approve"
                    )
                    assert approved.status_code == 200
                    signin = await client.post("/api/auth/signin", json=body)
                    assert signin.status_code == 200
                    assert signin.json()["role"] == "reviewer"
                    accounts.append(account_id)

                # Both reviewers start one below the cap for the distinct-seed race.
                if not same_seed:
                    for client, seeds in ((first, (0, 1)), (second, (2, 4))):
                        for seed in seeds:
                            response = await client.post(
                                "/api/reviewer_dirs", json={"query_file_id": seed}
                            )
                            assert response.status_code == 201
                responses = await asyncio.gather(
                    *(
                        (first if i % 2 == 0 else second).post(
                            "/api/reviewer_dirs",
                            json={"query_file_id": 0 if same_seed else i + 5},
                        )
                        for i in range(20)
                    )
                )
                codes = [response.status_code for response in responses]
                assert codes.count(201) == (1 if same_seed else 2), codes
                assert codes.count(409 if same_seed else 429) == (
                    19 if same_seed else 18
                ), codes
                listed = await first.get("/api/reviewer_dirs")
                assert listed.status_code == 200
                directories = listed.json()
                assert len(directories) == (1 if same_seed else 6)
                for directory in directories:
                    assert directory["member_query_ids"] == [directory["seed_query_id"]]
                if not same_seed:
                    for account in accounts:
                        assert (
                            await get_db().count_reviewer_dirs_by_account(account) == 3
                        )
                else:
                    recovery = await second.get(
                        "/api/reviewer_dirs", params={"seed_query_id": 0}
                    )
                    assert recovery.json() == directories

    asyncio.run(run())


@pytest.mark.parametrize("same_seed", [True, False], ids=["global-seed", "account-cap"])
def test_independent_connections_serialize_creation(
    tmp_path: Path, same_seed: bool
) -> None:
    from web.exceptions import ReviewerDirLimitError, ReviewerDirSeedExistsError

    path = tmp_path / "independent.db"

    async def run() -> None:
        first, second = FeedbackDB(path), FeedbackDB(path)
        await first.connect()
        await second.connect()
        try:
            results = await asyncio.gather(
                *(
                    _create(
                        first if i % 2 == 0 else second,
                        0 if same_seed else i,
                        account=i % 2 + 1 if same_seed else 1,
                        cap=3,
                    )
                    for i in range(12)
                ),
                return_exceptions=True,
            )
            saved = [r for r in results if isinstance(r, dict)]
            rejected = [r for r in results if isinstance(r, Exception)]
            assert len(saved) == (1 if same_seed else 3), results
            error = ReviewerDirSeedExistsError if same_seed else ReviewerDirLimitError
            assert len(rejected) == 12 - len(saved)
            assert all(isinstance(r, error) for r in rejected), rejected
            assert len(_rows(path, "reviewer_dirs")) == len(saved)
            assert len(_rows(path, "reviewer_dir_members")) == len(saved)
            # Duplicate recovery has precedence even when this account is over quota.
            with pytest.raises(ReviewerDirSeedExistsError):
                await _create(second, saved[0]["seed_query_id"], cap=0)
        finally:
            await second.close()
            await first.close()

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["success", "insert-failure", "cancel"])
def test_creation_isolated_from_auth_and_feedback_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    path = tmp_path / "interleaving.db"

    async def run() -> None:
        db = FeedbackDB(path)
        await db.connect()
        try:
            account = await db.create_account(
                "scholar", "Scholar", "secret", "reviewer"
            )
            token = await db.create_session(account["id"], 1)
            if outcome == "insert-failure":
                await db._db.execute(
                    """CREATE TRIGGER fail_seed BEFORE INSERT ON reviewer_dir_members
                       WHEN NEW.query_id = 0
                       BEGIN SELECT RAISE(ABORT, 'injected seed failure'); END"""
                )
                await db._db.commit()

            inserted, release = asyncio.Event(), asyncio.Event()
            auth_started, feedback_started = asyncio.Event(), asyncio.Event()
            original = aiosqlite.Connection._execute

            async def interleave(connection, fn, *args, **kwargs):
                sql = (
                    " ".join(args[0].split())
                    if args and isinstance(args[0], str)
                    else ""
                )
                if sql.startswith("UPDATE account_sessions SET last_seen_at"):
                    auth_started.set()
                if sql.startswith("INSERT INTO feedback"):
                    feedback_started.set()
                result = await original(connection, fn, *args, **kwargs)
                if sql.startswith("INSERT INTO reviewer_dirs") and args[1][2] == 0:
                    inserted.set()
                    await release.wait()
                return result

            monkeypatch.setattr(aiosqlite.Connection, "_execute", interleave)
            creation = asyncio.create_task(_create(db, 0, account["id"]))
            auth = feedback = None
            try:
                await asyncio.wait_for(inserted.wait(), 5)
                # A shared-connection commit must not publish the directory-only write.
                await db._db.commit()
                await db._db.rollback()
                assert _rows(path, "reviewer_dirs") == []
                assert _rows(path, "reviewer_dir_members") == []
                auth = asyncio.create_task(db.get_account_by_session(token))
                await asyncio.wait_for(auth_started.wait(), 5)
                feedback = asyncio.create_task(
                    db.insert(
                        1,
                        MODEL_SLUG,
                        "matched_rank",
                        1,
                        "candidate-a",
                        "Keep this note",
                        "Scholar",
                    )
                )
                await asyncio.wait_for(feedback_started.wait(), 5)
                assert not auth.done()
                assert not feedback.done()
                if outcome == "cancel":
                    creation.cancel()
                release.set()
                if outcome == "cancel":
                    with pytest.raises(asyncio.CancelledError):
                        await creation
                elif outcome == "insert-failure":
                    with pytest.raises(
                        sqlite3.IntegrityError, match="injected seed failure"
                    ):
                        await creation
                else:
                    await creation
                assert (await asyncio.wait_for(auth, 5))["id"] == account["id"]
                assert (await asyncio.wait_for(feedback, 5))[
                    "notes"
                ] == "Keep this note"
                expected = 1 if outcome == "success" else 0
                assert len(_rows(path, "reviewer_dirs")) == expected
                assert len(_rows(path, "reviewer_dir_members")) == expected
                assert len(_rows(path, "feedback")) == 1
                with closing(sqlite3.connect(path)) as observer:
                    assert observer.execute(
                        "SELECT last_seen_at FROM account_sessions"
                    ).fetchone()[0]
                # Failed/cancelled creation released its lock and did not poison auth's connection.
                await _create(db, 2, account["id"])
                assert len(_rows(path, "reviewer_dirs")) == expected + 1
                assert len(_rows(path, "reviewer_dir_members")) == expected + 1
            finally:
                release.set()
                for task in (creation, auth, feedback):
                    if task is not None and not task.done():
                        task.cancel()
                await asyncio.gather(
                    *(task for task in (creation, auth, feedback) if task is not None),
                    return_exceptions=True,
                )
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize("with_matrix", [True, False], ids=["matrix", "no-matrix"])
def test_legacy_duplicates_survive_reopen_and_recover_oldest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_matrix: bool
) -> None:
    from web.exceptions import ReviewerDirSeedExistsError
    from web.services import reviewer_dirs as svc

    config = _write_fixture_data(tmp_path, with_matrix=with_matrix, n_queries=8)
    path = tmp_path / "runs/active/resubmit/webapp/feedback.db"
    monkeypatch.setattr(svc, "MAX_REVIEWER_DIRS_PER_ACCOUNT", 1)
    tables = ("reviewer_dirs", "reviewer_dir_members", "feedback")

    async def run() -> None:
        db = FeedbackDB(path)
        await db.connect()
        try:
            first = await db.create_account("pi", "PI", "secret", "pi_admin")
            second = await db.create_account("scholar", "Scholar", "secret", "reviewer")
            # Fixture-only raw inserts reproduce the pre-fix duplicate-bearing schema.
            for dir_id, label, account, timestamp, member in (
                ("reviewer-dir-z-old", "First group", first, "2026-01-02 00:00:00", 2),
                (
                    "reviewer-dir-a-new",
                    "Second group",
                    second,
                    "2026-01-01 00:00:00",
                    4,
                ),
            ):
                await db._db.execute(
                    """INSERT INTO reviewer_dirs
                           (dir_id, label, seed_query_id, model_slug, variant,
                            created_at, created_by, created_by_account_id)
                       VALUES (?, ?, 0, ?, 'sif_abtt', ?, ?, ?)""",
                    (
                        dir_id,
                        label,
                        MODEL_SLUG,
                        timestamp,
                        account["display_name"],
                        account["id"],
                    ),
                )
                for query_id, role in ((0, "seed"), (member, "member")):
                    await db._db.execute(
                        """INSERT INTO reviewer_dir_members
                               (dir_id, query_id, role, added_at, added_by, added_by_account_id)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            dir_id,
                            query_id,
                            role,
                            timestamp,
                            account["display_name"],
                            account["id"],
                        ),
                    )
                await db.insert(
                    1,
                    MODEL_SLUG,
                    "matched_rank",
                    12 if member == 2 else 11,
                    dir_id,
                    f"Note for {label}",
                    account["display_name"],
                    account["id"],
                )
            await db._db.commit()
        finally:
            await db.close()

        before = {table: _rows(path, table) for table in tables}
        with closing(sqlite3.connect(path)) as fixture, fixture:
            for table in tables:
                for action in ("UPDATE", "DELETE"):
                    fixture.execute(
                        f"""CREATE TRIGGER preserve_{table}_{action}
                            BEFORE {action} ON {table}
                            BEGIN SELECT RAISE(ABORT, 'historical row changed'); END"""
                    )
            index = next(
                row
                for row in fixture.execute("PRAGMA index_list(reviewer_dirs)")
                if row[1] == "idx_reviewer_dirs_seed"
            )
            assert index[2] == 0  # A unique migration must not discard this history.

        previous_predictions = None
        for _ in range(2):
            async with _running_app(config) as app:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    signin = await client.post(
                        "/api/auth/signin",
                        json={"username": "pi", "password": "secret"},
                    )
                    assert signin.status_code == 200
                    for model_params in ({}, {"model": MODEL_SLUG}):
                        response = await client.get(
                            "/api/reviewer_dirs",
                            params={"seed_query_id": 0, **model_params},
                        )
                        assert response.status_code == 200
                        recovered = response.json()
                        assert [r["dir_id"] for r in recovered] == [
                            "reviewer-dir-z-old",
                            "reviewer-dir-a-new",
                        ]
                        assert [r["label"] for r in recovered] == [
                            "First group",
                            "Second group",
                        ]
                        assert [r["member_query_ids"] for r in recovered] == [
                            [0, 2],
                            [0, 4],
                        ]
                        assert [r["created_by"] for r in recovered] == ["PI", "Scholar"]
                        assert all(r["status"] == "matched" for r in recovered)
                    assert (await get_db().get_reviewer_dir_by_seed(0))[
                        "dir_id"
                    ] == "reviewer-dir-z-old"
                    duplicate = await client.post(
                        "/api/reviewer_dirs",
                        json={"query_file_id": 0, "label": "Not saved"},
                    )
                    assert duplicate.status_code == 409
                    assert duplicate.json()["detail"] == (
                        "Query 0 already seeds reviewer directory 'reviewer-dir-z-old' (First group)."
                    )
                    over_cap = await client.post(
                        "/api/reviewer_dirs", json={"query_file_id": 5}
                    )
                    assert over_cap.status_code == 429
                    assert "created 1 reviewer directories" in over_cap.json()["detail"]
                    predictions = await client.get(
                        "/api/query/1/predictions", params={"model": MODEL_SLUG}
                    )
                    assert predictions.status_code == 200
                    cards = [
                        p
                        for p in predictions.json()["predictions"]
                        if p["source"] == "reviewer"
                    ]
                    assert [p["dir_name"] for p in cards] == (
                        ["reviewer-dir-a-new", "reviewer-dir-z-old"]
                        if with_matrix
                        else []
                    )
                    if with_matrix:
                        assert [p["rank"] for p in cards] == [11, 12]
                        assert [p["score"] for p in cards] == [0.7998046875] * 2
                    if previous_predictions is not None:
                        assert predictions.json() == previous_predictions
                    previous_predictions = predictions.json()
                    assert {table: _rows(path, table) for table in tables} == before
                    with pytest.raises(ReviewerDirSeedExistsError) as error:
                        await _create(get_db(), 0, cap=0)
                    assert error.value.dir_id == "reviewer-dir-z-old"

        # Legacy duplicates must not disable protection for unused seeds either.
        await db.connect()
        try:
            results = await asyncio.gather(
                *(_create(db, 5, account=i % 2 + 1) for i in range(12)),
                return_exceptions=True,
            )
            assert sum(isinstance(r, dict) for r in results) == 1
            assert sum(isinstance(r, ReviewerDirSeedExistsError) for r in results) == 11
            for table in tables:
                assert _rows(path, table)[: len(before[table])] == before[table]
        finally:
            await db.close()

    asyncio.run(run())


def test_cancel_while_waiting_for_write_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "waiting.db"

    async def run() -> None:
        db = FeedbackDB(path)
        await db.connect()
        waiting = asyncio.Event()
        original = aiosqlite.Connection._execute

        async def notice_begin(connection, fn, *args, **kwargs):
            if args and args[0] == "BEGIN IMMEDIATE":
                waiting.set()
            return await original(connection, fn, *args, **kwargs)

        monkeypatch.setattr(aiosqlite.Connection, "_execute", notice_begin)
        creation = None
        try:
            # Hold the write lock on a different connection with an unrelated write.
            await db._db.execute(
                """INSERT INTO feedback (query_id, model_slug, notes, reviewer)
                   VALUES (1, 'bowphs_LaTa', 'Keep pending feedback', 'Scholar')"""
            )
            creation = asyncio.create_task(_create(db, 0))
            await asyncio.wait_for(waiting.wait(), 5)
            assert not creation.done()
            creation.cancel()
            await db._db.commit()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(creation, 5)
            assert _rows(path, "reviewer_dirs") == []
            assert _rows(path, "reviewer_dir_members") == []
            assert len(_rows(path, "feedback")) == 1
            await _create(db, 0)
            assert len(_rows(path, "reviewer_dirs")) == 1
            assert len(_rows(path, "reviewer_dir_members")) == 1
        finally:
            await db._db.rollback()
            if creation is not None:
                if not creation.done():
                    creation.cancel()
                await asyncio.gather(creation, return_exceptions=True)
            await db.close()

    asyncio.run(run())


def test_guard_validation_and_unrelated_storage_errors_keep_http_semantics(
    tmp_path: Path,
) -> None:
    config = _write_fixture_data(tmp_path)

    async def run() -> None:
        async with _running_app(config) as app:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                registration = await client.post(
                    "/api/auth/register",
                    json={
                        "username": "pi",
                        "display_name": "PI",
                        "password": "long enough password",
                    },
                )
                assert registration.status_code == 201
                excluded = await client.post(
                    "/api/reviewer_dirs", json={"query_file_id": 3}
                )
                assert excluded.status_code == 422
                assert await get_db().list_reviewer_dirs() == []
                # Emulate an older saved group whose seed is now guard-excluded.
                historical = await _create(get_db(), 3)
                duplicate = await client.post(
                    "/api/reviewer_dirs", json={"query_file_id": 3}
                )
                assert duplicate.status_code == 409
                assert historical["dir_id"] in duplicate.json()["detail"]

                await get_db()._db.execute(
                    """CREATE TRIGGER fail_seed BEFORE INSERT ON reviewer_dir_members
                       WHEN NEW.query_id = 0
                       BEGIN SELECT RAISE(ABORT, 'unrelated integrity error'); END"""
                )
                await get_db()._db.commit()
                failed = await client.post(
                    "/api/reviewer_dirs", json={"query_file_id": 0}
                )
                assert failed.status_code == 500
                recovery = await client.get(
                    "/api/reviewer_dirs", params={"seed_query_id": 0}
                )
                assert recovery.status_code == 200
                assert recovery.json() == []
                assert len(await get_db().list_reviewer_dirs()) == 1

    asyncio.run(run())


def test_cancellation_after_commit_keeps_both_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "committed.db"

    async def run() -> None:
        db = FeedbackDB(path)
        await db.connect()
        committed = asyncio.Event()
        original = aiosqlite.Connection.commit

        async def commit(connection):
            await original(connection)
            committed.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(aiosqlite.Connection, "commit", commit)
        creation = asyncio.create_task(_create(db, 0))
        try:
            await asyncio.wait_for(committed.wait(), 5)
            creation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await creation
            assert len(_rows(path, "reviewer_dirs")) == 1
            assert len(_rows(path, "reviewer_dir_members")) == 1
        finally:
            if not creation.done():
                creation.cancel()
            await asyncio.gather(creation, return_exceptions=True)
            await db.close()

    asyncio.run(run())
