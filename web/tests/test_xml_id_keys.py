"""Feedback and reviewer directories keyed by witness xml_id (issue #273).

`query_id` is a position in the sorted unlabelled corpus, so a data release
that adds or removes one query file would shift every later review onto
another witness. These tests pin the fix:

* the migration is additive and idempotent on a pre-#273 database, and takes
  one integrity-checked backup before its first write;
* the backfill writes only the new xml_id columns, from the served corpus,
  and refuses to guess on a deployment whose corpus is not the one the rows
  were saved against;
* the startup guard refuses a corpus that disagrees with stored ids, and does
  not fire on a correct one, including the real repo corpus;
* new rows carry their xml_id and the directory behind every selected rank;
* the CCL key export reads stored names and stored witnesses first.

Reviewer names here are fake.
"""

from __future__ import annotations

import asyncio
import csv
import os
import sqlite3
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import create_app
from web.services.feedback_db import (
    PRE_XML_ID_CORPUS_SHA256,
    CorpusMismatchError,
    FeedbackDB,
    corpus_digest,
)
from web.tests.test_reviewer_dirs import _register, _write_fixture_data

REPO_CORPUS = Path(__file__).resolve().parents[2] / "data" / "canon_unlabelled"
TRACKED_META = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "active"
    / "resubmit"
    / "unlabelled"
    / "meta_unlabelled.csv"
)

CORPUS = {0: "BAV1341.13v.1.txt", 1: "BAV1341.13v.2.txt", 2: "BN2123.89r.5.txt"}
# One witness added at the front: every later id now names another text.
SHIFTED = {0: "AAA0001.1r.1.txt", 1: "BAV1341.13v.1.txt", 2: "BAV1341.13v.2.txt",
           3: "BN2123.89r.5.txt"}

FEEDBACK_PRE_273 = """
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    model_slug TEXT NOT NULL,
    variant TEXT,
    outcome TEXT NOT NULL DEFAULT 'legacy_unresolved',
    correct_rank INTEGER,
    correct_dir TEXT,
    selected_ranks_json TEXT,
    notes TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL,
    reviewer_account_id INTEGER,
    schema_version INTEGER NOT NULL DEFAULT 2,
    ccl_key TEXT,
    ccl_key_action TEXT,
    ccl_key_dir TEXT,
    ccl_key_rank INTEGER
);
CREATE TABLE reviewer_dirs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dir_id TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    seed_query_id INTEGER NOT NULL,
    model_slug TEXT NOT NULL DEFAULT '',
    variant TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    created_by TEXT NOT NULL DEFAULT '',
    created_by_account_id INTEGER,
    ccl_key TEXT NOT NULL DEFAULT ''
);
CREATE TABLE reviewer_dir_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dir_id TEXT NOT NULL,
    query_id INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    added_by TEXT NOT NULL DEFAULT '',
    added_by_account_id INTEGER,
    UNIQUE (dir_id, query_id)
);
INSERT INTO feedback
    (query_id, timestamp, model_slug, variant, outcome, correct_rank, correct_dir,
     selected_ranks_json, notes, reviewer, reviewer_account_id)
VALUES
    (1, '2026-10-01 09:00:00', 'bowphs_LaTa', 'sif_abtt', 'matched_rank', 1,
     'CTOU.567.16', '[1, 2]', 'first note', 'Reviewer One', 1),
    (2, '2026-10-02 09:00:00', 'bowphs_LaTa', 'sif_abtt', 'none_of_top_k', 0,
     NULL, NULL, '', 'Reviewer Two', 2),
    (9, '2026-10-03 09:00:00', 'bowphs_LaTa', 'sif_abtt', 'skipped', NULL,
     NULL, NULL, 'out of range', 'Reviewer Two', 2);
INSERT INTO reviewer_dirs (dir_id, label, seed_query_id, created_by)
VALUES ('reviewer-dir-aaaaaaaaaaaa', 'New directory from BN2123.89r.5', 2, 'Reviewer Two');
INSERT INTO reviewer_dir_members (dir_id, query_id, role)
VALUES ('reviewer-dir-aaaaaaaaaaaa', 2, 'seed'),
       ('reviewer-dir-aaaaaaaaaaaa', 0, 'member');
"""

ORIGINAL_COLUMNS = {
    "feedback": [
        "id", "query_id", "timestamp", "model_slug", "variant", "outcome",
        "correct_rank", "correct_dir", "selected_ranks_json", "notes", "reviewer",
        "reviewer_account_id", "schema_version", "ccl_key", "ccl_key_action",
        "ccl_key_dir", "ccl_key_rank",
    ],
    "reviewer_dirs": [
        "id", "dir_id", "label", "seed_query_id", "model_slug", "variant",
        "created_at", "created_by", "created_by_account_id", "ccl_key",
    ],
    "reviewer_dir_members": [
        "id", "dir_id", "query_id", "role", "added_at", "added_by",
        "added_by_account_id",
    ],
}


def _pre_273_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(FEEDBACK_PRE_273)
    connection.commit()
    connection.close()


def _select(path: Path, table: str, columns: list[str]) -> list[tuple]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            f"SELECT {', '.join(columns)} FROM {table} ORDER BY id"
        ).fetchall()
    finally:
        connection.close()


def _originals(path: Path) -> dict[str, list[tuple]]:
    return {table: _select(path, table, cols) for table, cols in ORIGINAL_COLUMNS.items()}


def _xml_ids(path: Path) -> dict[str, list]:
    return {
        "feedback": [r[0] for r in _select(path, "feedback", ["xml_id"])],
        "reviewer_dirs": [r[0] for r in _select(path, "reviewer_dirs", ["seed_xml_id"])],
        "reviewer_dir_members": [
            r[0] for r in _select(path, "reviewer_dir_members", ["xml_id"])
        ],
    }


def _backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.pre-xml-id-*"))


async def _adopt(path: Path, corpus: dict[int, str], **kwargs: object) -> dict:
    db = FeedbackDB(path)
    await db.connect()
    try:
        return await db.adopt_corpus(corpus, **kwargs)
    finally:
        await db.close()


# --- migration ---------------------------------------------------------------


def test_migration_is_additive_idempotent_and_backs_up_once(tmp_path: Path) -> None:
    path = tmp_path / "feedback.db"
    _pre_273_db(path)
    before = _originals(path)

    async def open_close() -> None:
        db = FeedbackDB(path)
        await db.connect()
        await db.close()

    asyncio.run(open_close())

    # One backup, taken before the first write: it still has the old schema
    # and every original row. 0600, because it holds every password hash.
    backups = _backups(path)
    assert len(backups) == 1
    assert stat.S_IMODE(os.stat(backups[0]).st_mode) == 0o600
    assert _originals(backups[0]) == before
    backup_columns = {
        r[1] for r in sqlite3.connect(backups[0]).execute("PRAGMA table_info(feedback)")
    }
    assert "xml_id" not in backup_columns

    # The live database gained the nullable columns and nothing else changed.
    assert _originals(path) == before
    assert _xml_ids(path) == {
        "feedback": [None, None, None],
        "reviewer_dirs": [None],
        "reviewer_dir_members": [None, None],
    }
    assert _select(path, "feedback", ["selected_dirs_json"]) == [(None,)] * 3

    # A second boot neither migrates, backs up nor writes again.
    asyncio.run(open_close())
    assert len(_backups(path)) == 1
    assert _originals(path) == before


def test_a_fresh_or_empty_database_takes_no_backup(tmp_path: Path) -> None:
    path = tmp_path / "feedback.db"

    async def open_close() -> None:
        db = FeedbackDB(path)
        await db.connect()
        await db.close()

    asyncio.run(open_close())
    asyncio.run(open_close())
    assert _backups(path) == []


def test_an_old_schema_database_without_feedback_rows_takes_no_backup(
    tmp_path: Path,
) -> None:
    """A pre-#273 schema with zero feedback rows has nothing to protect."""
    path = tmp_path / "feedback.db"
    _pre_273_db(path)
    connection = sqlite3.connect(path)
    connection.execute("DELETE FROM feedback")
    connection.commit()
    connection.close()

    async def open_close() -> None:
        db = FeedbackDB(path)
        await db.connect()
        await db.close()

    asyncio.run(open_close())
    assert _backups(path) == []
    columns = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(feedback)")}
    assert "xml_id" in columns


_CRASH_MID_UPDATE = """
import os, sqlite3, sys
connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute("PRAGMA cache_size=5")
connection.execute("BEGIN")
connection.execute("UPDATE feedback SET notes = notes || 'x'")
os._exit(0)  # die mid-transaction: the rollback journal stays behind
"""


@pytest.mark.parametrize("schema", ["migrated", "pre_273"])
def test_startup_rolls_back_a_hot_journal_left_by_a_crash(
    tmp_path: Path, schema: str
) -> None:
    """A process killed mid-write leaves a hot journal; the next boot must start.

    Only a read-write connection can roll a hot journal back. A read-only probe
    fails with "attempt to write a readonly database", which would keep the
    app down until someone opened the file by hand.
    """
    import subprocess
    import sys

    path = tmp_path / "feedback.db"
    if schema == "pre_273":
        _pre_273_db(path)
        connection = sqlite3.connect(path)
    else:
        asyncio.run(_adopt(path, CORPUS))
        connection = sqlite3.connect(path)
    connection.executemany(
        "INSERT INTO feedback (query_id, model_slug, notes, reviewer) VALUES (?, ?, ?, ?)",
        [(0, "bowphs_LaTa", "n" * 200, "Fake Reviewer")] * 5000,
    )
    connection.commit()
    connection.close()
    notes_before = _select(path, "feedback", ["notes"])

    subprocess.run([sys.executable, "-c", _CRASH_MID_UPDATE, str(path)], check=True)
    journal = path.with_name(path.name + "-journal")
    assert journal.exists() and journal.stat().st_size > 0

    asyncio.run(_adopt(path, CORPUS))
    # The crashed transaction was rolled back: no note gained its 'x'.
    assert _select(path, "feedback", ["notes"]) == notes_before


# --- backfill ----------------------------------------------------------------


def test_backfill_writes_only_the_new_columns_from_the_served_corpus(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feedback.db"
    _pre_273_db(path)
    before = _originals(path)

    result = asyncio.run(_adopt(path, CORPUS))

    assert _xml_ids(path) == {
        # query 9 is not in the corpus: nothing to fill it from, so it stays NULL.
        "feedback": ["BAV1341.13v.2", "BN2123.89r.5", None],
        "reviewer_dirs": ["BN2123.89r.5"],
        "reviewer_dir_members": ["BN2123.89r.5", "BAV1341.13v.1"],
    }
    assert _originals(path) == before
    assert result["backfilled"] == {
        "feedback": 2,
        "reviewer_dirs": 1,
        "reviewer_dir_members": 2,
    }
    audit = _select(
        path,
        "xml_id_backfills",
        ["corpus_sha256", "corpus_size", "feedback_rows", "reviewer_dir_rows",
         "reviewer_dir_member_rows"],
    )
    assert audit == [(corpus_digest(CORPUS), 3, 2, 1, 2)]

    # Idempotent: a second boot fills nothing and writes no audit row.
    again = asyncio.run(_adopt(path, CORPUS))
    assert sum(again["backfilled"].values()) == 0
    assert len(_select(path, "xml_id_backfills", ["id"])) == 1
    assert _originals(path) == before


def test_a_deployment_refuses_to_backfill_from_an_unknown_corpus(tmp_path: Path) -> None:
    """With a data release set, old rows are named only from the pre-#273 corpus."""
    path = tmp_path / "feedback.db"
    _pre_273_db(path)

    with pytest.raises(CorpusMismatchError, match="predate witness ids"):
        asyncio.run(_adopt(path, CORPUS, data_release="data-20991231"))
    assert _xml_ids(path)["feedback"] == [None, None, None]
    assert _select(path, "xml_id_backfills", ["id"]) == []

    # Once every row carries its witness, the release no longer matters.
    asyncio.run(_adopt(path, CORPUS))
    asyncio.run(_adopt(path, CORPUS, data_release="data-20991231"))


# --- startup guard -------------------------------------------------------------


def test_guard_refuses_a_shifted_corpus_and_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "feedback.db"
    _pre_273_db(path)
    asyncio.run(_adopt(path, CORPUS))
    snapshot = {**_originals(path), **_xml_ids(path)}

    with pytest.raises(CorpusMismatchError) as raised:
        asyncio.run(_adopt(path, SHIFTED))
    message = str(raised.value)
    assert "Refusing to start" in message
    assert "query_id 1 stores 'BAV1341.13v.2' but the corpus serves 'BAV1341.13v.1'" in message
    assert {**_originals(path), **_xml_ids(path)} == snapshot


def test_guard_passes_the_same_corpus_and_tolerates_a_removed_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feedback.db"
    _pre_273_db(path)
    asyncio.run(_adopt(path, CORPUS))
    asyncio.run(_adopt(path, dict(CORPUS)))

    # The last witness leaves the corpus: no id names another text, so the
    # app starts; that witness's reviews are kept but not shown.
    async def statuses() -> dict:
        db = FeedbackDB(path)
        await db.connect()
        try:
            result = await db.adopt_corpus({0: CORPUS[0], 1: CORPUS[1]})
            assert result["unserved"] == 3  # one pair in each of the three tables
            return await db.get_query_statuses()
        finally:
            await db.close()

    shown = asyncio.run(statuses())
    assert set(shown) == {1, 9}
    assert shown[1]["review_status"] == "reviewed"


def _repo_corpus() -> dict[int, str]:
    if not REPO_CORPUS.is_dir():
        pytest.skip("the research repo corpus is not present (standalone webapp checkout)")
    # Exactly how scripts/resubmit/index_unlabelled.py numbers the corpus.
    names = sorted(p.name for p in REPO_CORPUS.glob("*.txt"))
    return dict(enumerate(names))


def test_pinned_digest_is_the_repo_corpus_and_its_tracked_index() -> None:
    corpus = _repo_corpus()
    assert len(corpus) == 2238
    assert corpus_digest(corpus) == PRE_XML_ID_CORPUS_SHA256
    if TRACKED_META.is_file():
        with TRACKED_META.open(newline="") as handle:
            tracked = {int(r["file_id"]): r["filename"] for r in csv.DictReader(handle)}
        assert tracked == corpus


def test_guard_does_not_fire_on_the_real_corpus_with_a_release(tmp_path: Path) -> None:
    """The production path: a populated pre-#273 DB booting on the real corpus."""
    corpus = _repo_corpus()
    path = tmp_path / "feedback.db"
    _pre_273_db(path)

    first = asyncio.run(_adopt(path, corpus, data_release="data-20260906-v1"))
    assert first["corpus_sha256"] == PRE_XML_ID_CORPUS_SHA256
    assert _xml_ids(path)["feedback"] == [
        corpus[1][:-4],
        corpus[2][:-4],
        corpus[9][:-4],
    ]
    second = asyncio.run(_adopt(path, corpus, data_release="data-20260906-v1"))
    assert sum(second["backfilled"].values()) == 0


# --- through the app -------------------------------------------------------------


def _post(client: TestClient, **body: object) -> dict:
    payload = {"model_slug": "bowphs/LaTa", "variant": "sif_abtt", "notes": "", **body}
    response = client.post("/api/feedback", json=payload)
    assert response.status_code in (200, 201), response.text
    return response.json()


def _rewrite_predictions(root: Path, names: list[str]) -> None:
    unlabelled = root / "data" / "canon_unlabelled"
    for name in names:
        (unlabelled / name).write_text(f"text of {name}", encoding="utf-8")
    path = root / "runs/active/resubmit/unlabelled/unlabelled_predictions_sif_abtt.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        template = next(reader)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for file_id, name in enumerate(sorted(names)):
            writer.writerow({**template, "file_id": file_id, "filename": name})


def test_new_rows_carry_xml_id_and_selected_dirs(tmp_path: Path) -> None:
    config = _write_fixture_data(tmp_path, with_matrix=False)
    db_path = tmp_path / "runs/active/resubmit/webapp/feedback.db"
    with TestClient(create_app(str(config))) as client:
        _register(client, "pi", "Fake Reviewer")
        picked = _post(
            client, query_id=1, outcome="matched_rank", correct_rank=2,
            selected_ranks=[2, 1],
        )
        assert picked["xml_id"] == "query-1"
        assert picked["selected_dirs"] == ["candidate-b", "candidate-a"]
        assert picked["correct_dir"] == "candidate-b"

        none = _post(client, query_id=2, outcome="none_of_top_k", ccl_key="CTOU.567.16")
        assert none["xml_id"] == "query-2"
        assert none["selected_dirs"] is None
        assert none["ccl_key_action"] == "created_reviewer_dir"

        created = client.post("/api/reviewer_dirs", json={"query_file_id": 0})
        assert created.status_code == 201, created.text

        latest = client.get(
            "/api/feedback/latest", params={"query_id": 1, "model": "bowphs/LaTa"}
        ).json()
        assert latest["selected_dirs"] == ["candidate-b", "candidate-a"]

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute(
            "SELECT query_id, xml_id, selected_dirs_json FROM feedback ORDER BY id"
        ).fetchall() == [
            (1, "query-1", '["candidate-b", "candidate-a"]'),
            (2, "query-2", None),
        ]
        assert connection.execute(
            "SELECT seed_query_id, seed_xml_id FROM reviewer_dirs ORDER BY id"
        ).fetchall() == [(2, "query-2"), (0, "query-0")]
        assert connection.execute(
            "SELECT query_id, xml_id FROM reviewer_dir_members ORDER BY id"
        ).fetchall() == [(2, "query-2"), (0, "query-0")]
    finally:
        connection.close()


def test_app_refuses_to_start_on_a_shifted_corpus(tmp_path: Path) -> None:
    config = _write_fixture_data(tmp_path, with_matrix=False)
    db_path = tmp_path / "runs/active/resubmit/webapp/feedback.db"
    with TestClient(create_app(str(config))) as client:
        _register(client, "pi", "Fake Reviewer")
        _post(client, query_id=2, outcome="matched_rank", correct_rank=1)

    # A release inserts one witness that sorts before query-1: ids 1..3 shift.
    _rewrite_predictions(
        tmp_path, ["query-0.txt", "query-0a.txt", "query-1.txt", "query-2.txt", "query-3.txt"]
    )
    app = create_app(str(config))
    with (
        pytest.raises(CorpusMismatchError, match="query_id 2 stores 'query-2'"),
        TestClient(app),
    ):
        pass

    # The same corpus as before starts normally and still shows the review.
    _rewrite_predictions(tmp_path, ["query-0.txt", "query-1.txt", "query-2.txt", "query-3.txt"])
    (tmp_path / "data/canon_unlabelled/query-0a.txt").unlink()
    with TestClient(create_app(str(config))) as client:
        client.post(
            "/api/auth/signin",
            json={"username": "pi", "password": "correct horse battery staple"},
        )
        statuses = {q["file_id"]: q for q in client.get("/api/queries").json()["items"]}
        assert statuses[2]["review_status"] == "reviewed"
    assert sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM feedback").fetchone() == (1,)


# --- CCL key export -------------------------------------------------------------


def test_export_reads_stored_names_and_stored_witnesses(tmp_path: Path) -> None:
    """A multi-rank pick is read from what the row stored, never today's ranking."""
    path = tmp_path / "feedback.db"

    async def run() -> list[dict]:
        db = FeedbackDB(path)
        await db.connect()
        try:
            await db.adopt_corpus({1: "BAV1341.13v.2.txt", 2: "BN2123.89r.5.txt"})
            await db.insert(
                1, "model_one", "matched_rank", 1, "CTOU.567.16", "", "Fake One", 1,
                selected_ranks=[1, 2], variant="sif_abtt",
                selected_dirs=["CTOU.567.16", "CTOU.567.16"],
            )
            await db.insert(
                2, "model_one", "matched_rank", 1, "CARL.501?.18", "", "Fake One", 1,
                selected_ranks=[1, 2], variant="sif_abtt",
                selected_dirs=["CARL.501?.18", "Capit.Martini.7"],
            )

            calls: list[tuple] = []

            async def moved_ranking(query_id: int, slug: str, variant, rank: int):
                calls.append((query_id, rank))
                return "candidate-a"  # the ranking has moved since the save

            rows = await db.ccl_key_assignments(
                # The corpus no longer names query 1; its stored witness does.
                filename_by_query={2: "BN2123.89r.5.txt"},
                labelled_dir_names=["CTOU.567.16", "CARL.501?.18", "Capit.Martini.7"],
                rank_dir=moved_ranking,
                exported_at="2026-11-12T00:00:00Z",
            )
            assert calls == []
            return rows
        finally:
            await db.close()

    rows = {row["xml_id"]: row for row in asyncio.run(run())}
    assert rows["BAV1341.13v.2"]["status"] == "agreed"
    assert rows["BAV1341.13v.2"]["ccl_key"] == "CTOU.567.16"
    assert rows["BN2123.89r.5"]["status"] == "conflict"
    assert rows["BN2123.89r.5"]["conflicting_keys"] == "Capit.Martini.7 | CARL.501?.18"
