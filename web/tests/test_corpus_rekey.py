"""Re-keying stored reviews by xml_id when the query corpus changes (issue #278).

A data release that adds, removes or reorders unlabelled query files moves
witnesses to other positional ids. These tests pin what startup does then:

* every read and write path follows the WITNESS, not the stored id, so a
  shift at the start, middle or end, a reorder and a removal all show each
  stored review, directory and membership against the same text as before.
  `_witness_view` reads every path and names what it returns by xml_id, so a
  path that still pairs a stored id with the served corpus by position makes
  the before/after comparison fail (the gap the #276 review named);
* a change is backed up and audited, a NULL xml_id refuses, too many missing
  reviewed witnesses refuse unless overridden for that exact corpus;
* reviewer-directory membership is unique by witness across a shift;
* a release whose prediction CSVs or q-q matrices disagree with the served
  listing refuses to start.

Reviewer names here are fake.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import sqlite3
import stat
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from web.app import create_app
from web.exceptions import ReviewerDirSeedExistsError
from web.services.data_store import ReleaseCorpusMismatchError
from web.services.feedback_db import (
    UNSERVED_QUERY_ID,
    CorpusMismatchError,
    FeedbackDB,
    MissingWitnessThresholdError,
    UnkeyedRowsError,
    corpus_digest,
)

MODEL = "bowphs_LaTa"
VARIANT = "sif_abtt"
KEY = "CTOU.567.16"
LABELLED = ["candidate-a", "candidate-b", KEY]

BASE_NAMES = ["w-b.txt", "w-d.txt", "w-f.txt", "w-h.txt"]


def _corpus(names: list[str]) -> dict[int, str]:
    """Numbered the way scripts/resubmit/index_unlabelled.py numbers a listing."""
    return dict(enumerate(sorted(names)))


BASE = _corpus(BASE_NAMES)
AT_START = _corpus(["w-a.txt", *BASE_NAMES])
IN_MIDDLE = _corpus([*BASE_NAMES, "w-e.txt"])
AT_END = _corpus([*BASE_NAMES, "w-z.txt"])
# The same witnesses, numbered in another order.
REORDERED = dict(enumerate(sorted(BASE_NAMES, reverse=True)))
# w-h carries only a skip, so losing it loses no reviewed witness.
WITHOUT_SKIPPED = _corpus(["w-b.txt", "w-d.txt", "w-f.txt"])
# w-d carries a matched_rank decision.
WITHOUT_REVIEWED = _corpus(["w-b.txt", "w-f.txt", "w-h.txt"])


def _qid(corpus: dict[int, str], stem: str) -> int:
    return {name[:-4]: qid for qid, name in corpus.items()}[stem]


def _select(path: Path, sql: str) -> list[tuple]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _dump(path: Path) -> dict[str, list[tuple]]:
    """Every stored row of the three query-referencing tables, all columns."""
    return {
        table: _select(path, f"SELECT * FROM {table} ORDER BY id")
        for table in ("feedback", "reviewer_dirs", "reviewer_dir_members")
    }


def _backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.pre-corpus-change-*"))


async def _open(path: Path, corpus: dict[int, str], **kwargs: object) -> FeedbackDB:
    db = FeedbackDB(path)
    await db.connect()
    try:
        await db.adopt_corpus(corpus, **kwargs)
    except BaseException:
        await db.close()
        raise
    return db


async def _adopt(path: Path, corpus: dict[int, str], **kwargs: object) -> dict:
    db = FeedbackDB(path)
    await db.connect()
    try:
        return await db.adopt_corpus(corpus, **kwargs)
    finally:
        await db.close()


def _seed(path: Path) -> str:
    """A production-like database bound to BASE. Returns the directory id.

    One legacy row (no xml_id) is backfilled by the first boot, which records
    BASE's digest exactly as production's xml_id_backfills row does. Then:
    w-d has a matched_rank pick, w-f a None with a key that creates a
    directory it seeds, w-b joins that directory and files into it, w-h has
    only skips.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    async def run() -> str:
        db = FeedbackDB(path)
        await db.connect()
        await db.close()
        connection = sqlite3.connect(path)
        connection.execute(
            "INSERT INTO feedback (query_id, timestamp, model_slug, variant, outcome, "
            "notes, reviewer, reviewer_account_id) VALUES (?, '2026-10-01 09:00:00', "
            "?, ?, 'skipped', 'legacy skip', 'Fake Two', 2)",
            (_qid(BASE, "w-h"), MODEL, VARIANT),
        )
        connection.commit()
        connection.close()

        db = await _open(path, BASE)
        try:
            await db.insert(
                _qid(BASE, "w-d"), MODEL, "matched_rank", 1, "candidate-a",
                "picked a", "Fake One", 1, selected_ranks=[1], variant=VARIANT,
                selected_dirs=["candidate-a"],
            )
            row, appended = await db.insert_none_of_top_k(
                query_id=_qid(BASE, "w-f"), model_slug=MODEL, variant=VARIANT,
                notes="none here", reviewer="Fake Two", reviewer_account_id=2,
                ccl_key=KEY, labelled_dir=None, labelled_rank=None,
                seed_scorable=True, max_per_account=50,
            )
            assert appended and row["ccl_key_action"] == "created_reviewer_dir"
            dir_id = row["ccl_key_dir"]
            assert await db.add_reviewer_dir_member(
                dir_id=dir_id, query_id=_qid(BASE, "w-b"), added_by="Fake One",
                added_by_account_id=1,
            )
            await db.insert(
                _qid(BASE, "w-b"), MODEL, "matched_rank", 1, dir_id, "filed",
                "Fake One", 1, selected_ranks=[1], variant=VARIANT,
                selected_dirs=[dir_id],
            )
            await db.insert(
                _qid(BASE, "w-h"), MODEL, "skipped", None, None, "later skip",
                "Fake One", 1, variant=VARIANT,
            )
            return dir_id
        finally:
            await db.close()

    return asyncio.run(run())


async def _witness_view(db: FeedbackDB, corpus: dict[int, str]) -> dict:
    """What every read path says, named by witness rather than by id.

    Each path is asked about a query by the id the corpus serves it at now,
    and each answer is translated back to an xml_id through that same corpus.
    A path that matched a stored id positionally would name another witness
    (or none) after a shift, and the view would differ.
    """
    xml_of = {qid: name[:-4] for qid, name in corpus.items()}
    view: dict = {}
    statuses = await db.get_query_statuses()
    view["statuses"] = {xml_of[q]: s for q, s in statuses.items() if q in xml_of}
    view["reviewed"] = sorted(xml_of[q] for q in await db.get_reviewed_query_ids())
    view["next_unreviewed"] = [
        xml_of[q] for q in await db.get_next_unreviewed(sorted(corpus), 10)
    ]
    latest, team_note, history, by_seed = {}, {}, {}, {}
    for qid, xml_id in xml_of.items():
        own = await db.get_latest_feedback(qid, MODEL, VARIANT, reviewer_account_id=1)
        if own is not None:
            assert own["query_id"] == qid
            latest[xml_id] = (own["xml_id"], own["outcome"], own["notes"])
        note = await db.get_latest_feedback(qid, MODEL, VARIANT, require_note=True)
        if note is not None:
            team_note[xml_id] = (note["xml_id"], note["notes"])
        rows = await db.get_feedback_for_query(qid)
        if rows:
            assert {r["query_id"] for r in rows} == {qid}
            history[xml_id] = [(r["xml_id"], r["notes"]) for r in rows]
        seeded = await db.get_reviewer_dir_by_seed(qid)
        if seeded is not None:
            by_seed[xml_id] = seeded["dir_id"]
    view.update(latest=latest, team_note=team_note, history=history, by_seed=by_seed)

    def dir_view(record: dict) -> tuple:
        seed = record["seed_query_id"]
        return (
            record["dir_id"],
            xml_of.get(seed) if seed != UNSERVED_QUERY_ID else None,
            sorted(xml_of[m] for m in record["member_query_ids"]),
            record["member_witness_count"],
        )

    view["dirs"] = [dir_view(r) for r in await db.list_reviewer_dirs()]
    view["dir"] = [
        dir_view(await db.get_reviewer_dir(r["dir_id"]))
        for r in await db.list_reviewer_dirs()
    ]
    view["recent"] = [
        (xml_of[r["file_id"]], r["outcome"]) for r in await db.get_recent_reviews(10)
    ]
    view["attention"] = [
        (xml_of[r["file_id"]], r["notes"]) for r in await db.get_needs_attention(10)
    ]
    stats = await db.get_stats()
    view["stats"] = {
        k: stats[k]
        for k in (
            "feedback_count",
            "reviewed_count",
            "skipped_count",
            "unserved_witness_count",
            "unserved_reviewed_count",
        )
    }
    exported = list(csv.DictReader(io.StringIO(await db.export_csv(filename_by_query=corpus))))
    view["export"] = [
        (
            row["filename"],
            xml_of[int(row["served_query_id"])] if row["served_query_id"] else None,
            row["notes"],
        )
        for row in exported
    ]

    async def no_ranking(*_args: object) -> None:
        raise AssertionError("every pick here is stored; no ranking is read")

    ccl = await db.ccl_key_assignments(
        filename_by_query=corpus,
        labelled_dir_names=LABELLED,
        rank_dir=no_ranking,
        exported_at="2026-11-12T00:00:00Z",
    )
    view["ccl"] = {
        r["xml_id"]: (r["status"], r["ccl_key"], r["reviewer_count"]) for r in ccl
    }
    return view


def _view(path: Path, corpus: dict[int, str], **kwargs: object) -> dict:
    async def run() -> dict:
        db = await _open(path, corpus, **kwargs)
        try:
            return await _witness_view(db, corpus)
        finally:
            await db.close()

    return asyncio.run(run())


# --- every read path follows the witness ---------------------------------------


@pytest.mark.parametrize(
    "shifted",
    [AT_START, IN_MIDDLE, AT_END, REORDERED],
    ids=["insert_at_start", "insert_in_middle", "insert_at_end", "reorder"],
)
def test_every_read_path_returns_the_same_witness_after_a_shift(
    tmp_path: Path, shifted: dict[int, str]
) -> None:
    path = tmp_path / "feedback.db"
    _seed(path)
    before = _view(path, BASE)
    stored = _dump(path)

    after = _view(path, shifted)

    # The added witness has no review, so it shows up only as unreviewed.
    added = {n[:-4] for n in shifted.values()} - {n[:-4] for n in BASE.values()}
    assert [x for x in after["next_unreviewed"] if x not in added] == before[
        "next_unreviewed"
    ]
    after["next_unreviewed"] = before["next_unreviewed"]
    assert after == before
    # Sanity: the seeded data really is spread over every read path.
    assert before["statuses"]["w-d"]["review_status"] == "reviewed"
    assert before["dirs"][0][1:] == ("w-f", ["w-b", "w-f"], 2)
    assert before["ccl"]["w-f"] == ("agreed", KEY, 1)
    # Nothing stored was rewritten: the log keeps the ids rows were saved at.
    assert _dump(path) == stored


def test_a_shift_is_backed_up_and_audited_once(tmp_path: Path) -> None:
    path = tmp_path / "feedback.db"
    _seed(path)
    stored = _dump(path)

    result = asyncio.run(_adopt(path, AT_START, data_release="data-20991231"))

    change = result["change"]
    assert result["changed"] is True
    assert (change["moved"], change["unchanged"], change["missing"]) == (4, 0, 0)
    backups = _backups(path)
    assert len(backups) == 1 and backups[0] == change["backup"]
    assert stat.S_IMODE(os.stat(backups[0]).st_mode) == 0o600
    # The backup is the database as it was just before the change.
    assert _dump(backups[0]) == stored
    assert _select(backups[0], "SELECT COUNT(*) FROM corpus_changes") == [(0,)]
    assert _select(backups[0], "PRAGMA integrity_check") == [("ok",)]

    audit = _select(
        path,
        "SELECT old_corpus_sha256, old_corpus_size, new_corpus_sha256, "
        "new_corpus_size, data_release, backup_name, witnesses_unchanged, "
        "witnesses_moved, witnesses_missing, witnesses_newly_missing, "
        "reviewed_witnesses, reviewed_witnesses_newly_missing, "
        "missing_override_used, members_table_rebuilt FROM corpus_changes",
    )
    assert audit == [
        (
            corpus_digest(BASE), 4, corpus_digest(AT_START), 5, "data-20991231",
            backups[0].name, 0, 4, 0, 0, 3, 0, 0, 1,
        )
    ]
    assert _select(
        path,
        "SELECT xml_id, old_query_id, new_query_id FROM corpus_change_witnesses "
        "ORDER BY xml_id",
    ) == [("w-b", 0, 1), ("w-d", 1, 2), ("w-f", 2, 3), ("w-h", 3, 4)]

    # A second boot on the same corpus is not a change: no backup, no row.
    again = asyncio.run(_adopt(path, AT_START, data_release="data-20991231"))
    assert again["changed"] is False
    assert len(_backups(path)) == 1
    assert _select(path, "SELECT COUNT(*) FROM corpus_changes") == [(1,)]


def test_a_second_change_reads_the_first_change_s_id_map(tmp_path: Path) -> None:
    """Rows keep the ids they were saved at, so the map says where each was."""
    path = tmp_path / "feedback.db"
    _seed(path)
    asyncio.run(_adopt(path, AT_START))
    # Saved under AT_START: w-d's new row stores id 2.
    async def save() -> None:
        db = await _open(path, AT_START)
        try:
            await db.insert(
                _qid(AT_START, "w-d"), MODEL, "skipped", None, None, "second era",
                "Fake One", 1, variant=VARIANT,
            )
        finally:
            await db.close()

    asyncio.run(save())
    before = _view(path, AT_START)
    twice = _corpus(["w-a.txt", "w-aa.txt", *BASE_NAMES])
    result = asyncio.run(_adopt(path, twice))
    assert result["change"]["moved"] == 4
    assert _select(
        path, "SELECT old_corpus_sha256 FROM corpus_changes ORDER BY id"
    ) == [(corpus_digest(BASE),), (corpus_digest(AT_START),)]
    assert _select(
        path,
        "SELECT xml_id, old_query_id, new_query_id FROM corpus_change_witnesses "
        "WHERE change_id = 2 ORDER BY xml_id",
    ) == [("w-b", 1, 2), ("w-d", 2, 3), ("w-f", 3, 4), ("w-h", 4, 5)]
    after = _view(path, twice)
    after["next_unreviewed"] = before["next_unreviewed"] = []
    assert after == before


def test_an_unrecorded_database_detects_a_change_from_its_stored_ids(
    tmp_path: Path,
) -> None:
    """No digest was ever recorded: the stored positions decide."""
    path = tmp_path / "feedback.db"

    async def seed() -> None:
        db = await _open(path, BASE)
        try:
            await db.insert(
                _qid(BASE, "w-f"), MODEL, "skipped", None, None, "", "Fake One", 1,
                variant=VARIANT,
            )
        finally:
            await db.close()

    asyncio.run(seed())
    assert _select(path, "SELECT COUNT(*) FROM xml_id_backfills") == [(0,)]
    # An appended file moves no stored witness: nothing to re-key, no write.
    assert asyncio.run(_adopt(path, AT_END))["changed"] is False
    assert _backups(path) == []
    # One inserted before w-f does.
    result = asyncio.run(_adopt(path, AT_START))
    assert result["changed"] is True and result["change"]["moved"] == 1
    assert _select(path, "SELECT old_corpus_sha256 FROM corpus_changes") == [("",)]


# --- write paths follow the witness ---------------------------------------------


def test_writes_after_a_shift_are_checked_by_witness(tmp_path: Path) -> None:
    path = tmp_path / "feedback.db"
    dir_id = _seed(path)

    async def run() -> None:
        db = await _open(path, AT_START)
        try:
            w_f, w_b, w_a = (_qid(AT_START, s) for s in ("w-f", "w-b", "w-a"))
            # Identical repeat of w-f's None + key: found by witness, not
            # appended. Positionally, id 3 was w-h, which has no such row.
            row, appended = await db.insert_none_of_top_k(
                query_id=w_f, model_slug=MODEL, variant=VARIANT, notes="none here",
                reviewer="Fake Two", reviewer_account_id=2, ccl_key=KEY,
                labelled_dir=None, labelled_rank=None, seed_scorable=True,
                max_per_account=50,
            )
            assert appended is False
            assert (row["query_id"], row["xml_id"]) == (w_f, "w-f")

            # w-f still seeds its directory: a second one is refused, and a
            # key with no directory is recorded as seed_taken.
            with pytest.raises(ReviewerDirSeedExistsError):
                await db.create_reviewer_dir(
                    label="again", seed_query_id=w_f, model_slug=MODEL,
                    variant=VARIANT, created_by="Fake One", created_by_account_id=1,
                    max_per_account=50,
                )
            row, _ = await db.insert_none_of_top_k(
                query_id=w_f, model_slug=MODEL, variant=VARIANT, notes="other",
                reviewer="Fake One", reviewer_account_id=1, ccl_key="CTOU.999.1",
                labelled_dir=None, labelled_rank=None, seed_scorable=True,
                max_per_account=50,
            )
            assert row["ccl_key_action"] == "seed_taken"

            # w-b is already a member under its old id 0: no second row.
            assert not await db.add_reviewer_dir_member(
                dir_id=dir_id, query_id=w_b, added_by="Fake One", added_by_account_id=1
            )
            row, _ = await db.insert_none_of_top_k(
                query_id=w_b, model_slug=MODEL, variant=VARIANT, notes="",
                reviewer="Fake One", reviewer_account_id=1, ccl_key=KEY,
                labelled_dir=None, labelled_rank=None, seed_scorable=True,
                max_per_account=50,
            )
            assert row["ccl_key_action"] == "already_joined"

            # w-a now sits at id 0, w-b's old id in the same directory. The
            # positional UNIQUE (dir_id, query_id) would have refused it.
            row, _ = await db.insert_none_of_top_k(
                query_id=w_a, model_slug=MODEL, variant=VARIANT, notes="",
                reviewer="Fake One", reviewer_account_id=1, ccl_key=KEY,
                labelled_dir=None, labelled_rank=None, seed_scorable=True,
                max_per_account=50,
            )
            assert row["ccl_key_action"] == "joined_reviewer_dir"
            # And w-a may seed a directory of its own: w-f's old id is 2,
            # which now names w-d, not w-a; w-d may not be blocked by it either.
            created = await db.create_reviewer_dir(
                label="w-d group", seed_query_id=_qid(AT_START, "w-d"),
                model_slug=MODEL, variant=VARIANT, created_by="Fake One",
                created_by_account_id=1, max_per_account=50,
            )
            assert created["seed_xml_id"] == "w-d"

            record = await db.get_reviewer_dir(dir_id)
            assert sorted(record["member_query_ids"]) == sorted([w_f, w_b, w_a])
            assert record["member_witness_count"] == 3
        finally:
            await db.close()

    asyncio.run(run())
    members = _select(
        path,
        f"SELECT query_id, xml_id, role FROM reviewer_dir_members "
        f"WHERE dir_id = '{dir_id}' ORDER BY id",
    )
    assert members == [(2, "w-f", "seed"), (0, "w-b", "member"), (0, "w-a", "member")]


def test_the_members_rebuild_keeps_every_row_and_drops_only_the_constraint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feedback.db"
    _seed(path)
    members_before = _select(path, "SELECT * FROM reviewer_dir_members ORDER BY id")
    highest = max(r[0] for r in members_before)

    asyncio.run(_adopt(path, AT_START))

    assert _select(path, "SELECT * FROM reviewer_dir_members ORDER BY id") == members_before
    columns = [r[1] for r in _select(path, "PRAGMA table_info(reviewer_dir_members)")]
    assert columns == [
        "id", "dir_id", "query_id", "role", "added_at", "added_by",
        "added_by_account_id", "xml_id",
    ]
    indexes = {r[1]: (r[2], r[3]) for r in _select(path, "PRAGMA index_list(reviewer_dir_members)")}
    assert not any(origin == "u" for _unique, origin in indexes.values())
    assert indexes["idx_reviewer_dir_members_witness"] == (1, "c")
    assert {"idx_reviewer_dir_members_dir", "idx_reviewer_dir_members_query",
            "idx_reviewer_dir_members_xml_id"} <= set(indexes)
    assert _select(
        path, "SELECT name FROM sqlite_master WHERE name LIKE '%rekey%'"
    ) == []
    assert _select(path, "PRAGMA integrity_check") == [("ok",)]

    # AUTOINCREMENT carries on past every old id.
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO reviewer_dir_members (dir_id, query_id, xml_id) VALUES ('d', 9, 'w-z')"
    )
    new_id = connection.execute("SELECT MAX(id) FROM reviewer_dir_members").fetchone()[0]
    connection.rollback()
    connection.close()
    assert new_id == highest + 1

    # Only once: a later change finds no positional constraint to remove.
    result = asyncio.run(_adopt(path, _corpus(["w-0.txt", "w-a.txt", *BASE_NAMES])))
    assert result["change"]["members_table_rebuilt"] is False


def test_a_failed_rebuild_rolls_the_whole_change_back(tmp_path: Path) -> None:
    """A trigger on the table would be dropped by a rebuild: refuse, write nothing."""
    path = tmp_path / "feedback.db"
    _seed(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TRIGGER members_audit AFTER INSERT ON reviewer_dir_members "
        "BEGIN SELECT 1; END"
    )
    connection.commit()
    connection.close()
    stored = _dump(path)

    with pytest.raises(CorpusMismatchError, match="trigger"):
        asyncio.run(_adopt(path, AT_START))
    assert _dump(path) == stored
    assert _select(path, "SELECT COUNT(*) FROM corpus_changes") == [(0,)]
    assert _select(path, "SELECT COUNT(*) FROM corpus_change_witnesses") == [(0,)]
    # The backup was taken before the attempt and is left for the operator.
    assert len(_backups(path)) == 1


def test_a_rebuild_that_fails_midway_restores_the_original_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail after the copy, drop and rename: the transaction undoes all three."""
    path = tmp_path / "feedback.db"
    _seed(path)
    stored = _dump(path)
    schema = _select(path, "SELECT sql FROM sqlite_master ORDER BY name")

    async def run() -> None:
        db = FeedbackDB(path)
        await db.connect()

        async def fail() -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr(db, "_ensure_member_witness_index", fail)
        try:
            with pytest.raises(RuntimeError, match="disk full"):
                await db.adopt_corpus(AT_START)
        finally:
            await db.close()

    asyncio.run(run())
    assert _dump(path) == stored
    assert _select(path, "SELECT sql FROM sqlite_master ORDER BY name") == schema
    assert _select(path, "SELECT COUNT(*) FROM corpus_changes") == [(0,)]
    # The next boot simply tries again, and succeeds.
    assert asyncio.run(_adopt(path, AT_START))["change"]["members_table_rebuilt"] is True


# --- witnesses that leave the corpus --------------------------------------------


def test_a_removed_unreviewed_witness_is_kept_hidden_and_counted(tmp_path: Path) -> None:
    path = tmp_path / "feedback.db"
    _seed(path)
    stored = _dump(path)
    before = _view(path, BASE)

    after = _view(path, WITHOUT_SKIPPED)

    assert _dump(path) == stored
    assert "w-h" not in after["statuses"]
    assert all(x != "w-h" for x, _ in after["recent"] + after["attention"])
    assert after["stats"]["skipped_count"] == before["stats"]["skipped_count"] - 1
    assert after["stats"]["unserved_witness_count"] == 1
    assert after["stats"]["unserved_reviewed_count"] == 0
    assert after["stats"]["feedback_count"] == before["stats"]["feedback_count"]
    # The raw export keeps the row, named by its stored witness, served nowhere.
    assert ("w-h.txt", None, "later skip") in after["export"]
    for key in ("latest", "history"):
        assert after[key] == {x: v for x, v in before[key].items() if x != "w-h"}
    for key in ("dirs", "ccl", "by_seed"):
        assert after[key] == before[key]
    change = _select(
        path, "SELECT witnesses_missing, witnesses_newly_missing FROM corpus_changes"
    )
    assert change == [(1, 1)]


def test_too_many_missing_reviewed_witnesses_refuse_and_write_nothing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feedback.db"
    _seed(path)
    stored = _dump(path)

    with pytest.raises(MissingWitnessThresholdError) as raised:
        asyncio.run(_adopt(path, WITHOUT_REVIEWED))
    message = str(raised.value)
    assert "1 of the 3 reviewed witnesses" in message
    assert "First missing: w-d" in message
    assert corpus_digest(WITHOUT_REVIEWED) in message
    assert _dump(path) == stored
    assert _backups(path) == []
    assert _select(path, "SELECT COUNT(*) FROM corpus_changes") == [(0,)]

    # An override for some other corpus does not apply.
    with pytest.raises(MissingWitnessThresholdError):
        asyncio.run(
            _adopt(path, WITHOUT_REVIEWED, accept_missing_reviewed_sha256=corpus_digest(BASE))
        )
    assert _backups(path) == []


def test_a_missing_reviewed_witness_is_accepted_by_override_or_a_higher_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feedback.db"
    _seed(path)
    before = _view(path, BASE)

    after = _view(
        path,
        WITHOUT_REVIEWED,
        accept_missing_reviewed_sha256=corpus_digest(WITHOUT_REVIEWED),
    )
    assert _select(
        path,
        "SELECT reviewed_witnesses, reviewed_witnesses_newly_missing, "
        "missing_override_used FROM corpus_changes",
    ) == [(3, 1, 1)]
    # Hidden from the queue and every list, counted for the PI, and still in
    # the CCL key export: its key assignment is valid for CCL.
    assert "w-d" not in after["statuses"] and "w-d" not in after["reviewed"]
    assert after["stats"]["reviewed_count"] == before["stats"]["reviewed_count"] - 1
    assert after["stats"]["unserved_reviewed_count"] == 1
    assert after["ccl"]["w-d"] == before["ccl"]["w-d"]
    assert "w-d" not in after["latest"] and "w-d" not in after["history"]

    # A higher configured limit also lets the change through.
    other = tmp_path / "other" / "feedback.db"
    _seed(other)
    result = asyncio.run(_adopt(other, WITHOUT_REVIEWED, max_missing_reviewed_share=0.5))
    assert result["change"]["missing_override_used"] is False


def test_a_removed_member_keeps_its_directory_matched_and_never_names_another(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feedback.db"
    dir_id = _seed(path)
    # w-b (member, id 0) and w-f (seed, id 2) leave; a new witness takes id 0.
    corpus = _corpus(["w-a.txt", "w-d.txt", "w-h.txt"])
    view = _view(path, corpus, max_missing_reviewed_share=1.0)
    assert view["dirs"] == [(dir_id, None, [], 2)]
    assert view["by_seed"] == {}

    async def record() -> dict:
        db = await _open(path, corpus)
        try:
            return await db.get_reviewer_dir(dir_id)
        finally:
            await db.close()

    shown = asyncio.run(record())
    assert shown["seed_query_id"] == UNSERVED_QUERY_ID and shown["seed_served"] is False
    from web.services import reviewer_dirs as svc

    assert svc.status_of(shown).value == "matched"


# --- refusals before any re-key --------------------------------------------------


def test_a_null_xml_id_row_refuses_a_changed_corpus(tmp_path: Path) -> None:
    path = tmp_path / "feedback.db"
    _seed(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO reviewer_dir_members (dir_id, query_id, role) VALUES ('x', 1, 'member')"
    )
    connection.commit()
    connection.close()
    stored = _dump(path)

    with pytest.raises(UnkeyedRowsError, match=r"1 in reviewer_dir_members"):
        asyncio.run(_adopt(path, AT_START))
    assert _dump(path) == stored
    assert _backups(path) == []


def test_a_corpus_naming_one_witness_twice_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CorpusMismatchError, match="same witness under two file ids"):
        asyncio.run(_adopt(tmp_path / "feedback.db", {0: "w-a.txt", 1: "w-a.txt"}))


# --- through the app: q-q scoring and the release consistency check --------------


def _sim(a: str, b: str) -> float:
    """A similarity that belongs to the pair of witnesses, not to positions."""
    if a == b:
        return 1.0
    return round(0.2 + (sum(map(ord, a + b)) % 50) / 100, 2)


def _write_release(
    root: Path,
    names: list[str],
    *,
    variants: tuple[str, ...] = ("sif_abtt",),
    csv_names: dict[str, list[str]] | None = None,
    matrix_names: list[str] | None = None,
) -> Path:
    """A data root whose CSVs and q-q matrix are numbered from `names`."""
    unlabelled = root / "data" / "canon_unlabelled"
    predictions = root / "runs" / "active" / "resubmit" / "unlabelled"
    unlabelled.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    (root / "runs" / "active" / "resubmit" / "webapp").mkdir(parents=True, exist_ok=True)
    for old in unlabelled.glob("*.txt"):
        old.unlink()
    for name in names:
        (unlabelled / name).write_text(f"text of {name}", encoding="utf-8")
    for directory, filename in (("candidate-a", "a.txt"), ("candidate-b", "b.txt")):
        (root / "data" / "canon_labelled" / directory).mkdir(parents=True, exist_ok=True)
        (root / "data" / "canon_labelled" / directory / filename).write_text(
            directory, encoding="utf-8"
        )

    fields = ["file_id", "filename", "model", "variant", "layer", "pooling",
              "rank1_dir", "rank1_score", "rank2_dir", "rank2_score"]
    for variant in variants:
        listed = sorted((csv_names or {}).get(variant, names))
        with (predictions / f"unlabelled_predictions_{variant}.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for file_id, name in enumerate(listed):
                writer.writerow({
                    "file_id": file_id, "filename": name, "model": "bowphs/LaTa",
                    "variant": variant, "layer": 12, "pooling": "sif",
                    "rank1_dir": "candidate-a", "rank1_score": 0.41,
                    "rank2_dir": "candidate-b", "rank2_score": 0.22,
                })
    ordered = sorted(matrix_names if matrix_names is not None else names)
    sim = np.array([[_sim(a, b) for b in ordered] for a in ordered], dtype=np.float16)
    np.savez(
        predictions / f"qq_sim_{MODEL}.npz",
        sim=sim,
        file_ids=np.arange(len(ordered), dtype=np.int32),
        excluded=np.zeros(len(ordered), dtype=bool),
        meta=np.array(json.dumps({"model": "bowphs/LaTa"})),
    )
    config = root / "config.yaml"
    config.write_text(
        f"""
paths:
  data_root: "{root}"
  canon_unlabelled: "data/canon_unlabelled"
  canon_labelled: "data/canon_labelled"
  predictions_variant_pattern: "runs/active/resubmit/unlabelled/unlabelled_predictions_{{variant}}.csv"
  qq_matrix_pattern: "runs/active/resubmit/unlabelled/qq_sim_{{model_slug}}.npz"
  variants: {json.dumps(list(variants))}
  default_variant: "sif_abtt"
  feedback_db: "runs/active/resubmit/webapp/feedback.db"
  ig_examples_csv: "missing/phase12f_examples.csv"
  ig_artifacts_dir: "missing/artifacts"
auth:
  secure_cookies: false
""",
        encoding="utf-8",
    )
    return config


def _signin(client: TestClient) -> None:
    response = client.post(
        "/api/auth/signin",
        json={"username": "pi", "password": "correct horse battery staple"},
    )
    assert response.status_code == 200, response.text


def _reviewer_dir_evidence(client: TestClient, file_id: int) -> dict:
    body = client.get(
        f"/api/query/{file_id}/predictions", params={"model": "bowphs/LaTa"}
    ).json()
    return {
        "filename": body["filename"],
        "candidates": [
            (c["dir_id"], round(c["score"], 3), c["supporting_member"]["filename"],
             c["dir_files"])
            for c in body["reviewer_dir_candidates"]
        ],
        "seeded": [d["dir_id"] for d in body["seeded_dirs"]],
    }


def test_reviewer_dir_scores_follow_the_witness_through_the_app(tmp_path: Path) -> None:
    config = _write_release(tmp_path, BASE_NAMES)
    with TestClient(create_app(str(config))) as client:
        response = client.post(
            "/api/auth/register",
            json={"username": "pi", "display_name": "Fake PI",
                  "password": "correct horse battery staple"},
        )
        assert response.status_code == 201
        saved = client.post("/api/feedback", json={
            "query_id": _qid(BASE, "w-f"), "model_slug": "bowphs/LaTa",
            "variant": VARIANT, "outcome": "none_of_top_k", "notes": "",
            "ccl_key": KEY,
        })
        assert saved.status_code == 201, saved.text
        before = {
            stem: _reviewer_dir_evidence(client, _qid(BASE, stem))
            for stem in ("w-b", "w-d", "w-f", "w-h")
        }
    assert before["w-d"]["candidates"][0][2] == "w-f.txt"
    assert before["w-f"]["seeded"] and not before["w-d"]["seeded"]

    # The release gains w-a at the front; the matrix is rebuilt for it.
    _write_release(tmp_path, ["w-a.txt", *BASE_NAMES])
    with TestClient(create_app(str(config))) as client:
        _signin(client)
        after = {
            stem: _reviewer_dir_evidence(client, _qid(AT_START, stem))
            for stem in ("w-b", "w-d", "w-f", "w-h")
        }
        assert after == before
        stats = client.get("/api/stats").json()
        assert stats["reviewed_count"] == 1 and stats["unserved_witness_count"] == 0
        exported = list(csv.DictReader(io.StringIO(
            client.get("/api/feedback/export").text
        )))
        assert [(r["query_id"], r["served_query_id"], r["filename"]) for r in exported] == [
            (str(_qid(BASE, "w-f")), str(_qid(AT_START, "w-f")), "w-f.txt")
        ]


def test_a_matching_release_starts(tmp_path: Path) -> None:
    config = _write_release(tmp_path, BASE_NAMES, variants=("raw", "sif_abtt"))
    with TestClient(create_app(str(config))) as client:
        assert client.get("/api/models").status_code == 200


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"csv_names": {"sif_abtt": [*BASE_NAMES, "w-z.txt"]}},
            "served queries have no file",
        ),
        (
            {"csv_names": {"sif_abtt": BASE_NAMES[:-1]}},
            "are in no prediction row",
        ),
        (
            {"variants": ("raw", "sif_abtt"),
             "csv_names": {"raw": ["w-a.txt", *BASE_NAMES[:-1]]}},
            "raw/bowphs_LaTa has file_id 0 = 'w-a.txt'",
        ),
        ({"matrix_names": [*BASE_NAMES, "w-z.txt"]}, "q-q matrix"),
        ({"matrix_names": BASE_NAMES[:-1]}, "q-q matrix"),
    ],
    ids=[
        "csv_names_a_missing_file",
        "file_missing_from_csv",
        "other_variant_disagrees",
        "matrix_too_big",
        "matrix_too_small",
    ],
)
def test_a_release_that_disagrees_with_the_corpus_refuses_to_start(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    config = _write_release(tmp_path, BASE_NAMES, **kwargs)
    with pytest.raises(ReleaseCorpusMismatchError, match=message), TestClient(
        create_app(str(config))
    ):
        pass
    # Refused before the feedback database was even opened.
    assert not (tmp_path / "runs/active/resubmit/webapp/feedback.db").exists()


def test_a_matrix_whose_shape_disagrees_with_its_ids_refuses_to_start(
    tmp_path: Path,
) -> None:
    config = _write_release(tmp_path, BASE_NAMES)
    n = len(BASE_NAMES)
    np.savez(
        tmp_path / f"runs/active/resubmit/unlabelled/qq_sim_{MODEL}.npz",
        sim=np.eye(n + 1, dtype=np.float16),
        file_ids=np.arange(n, dtype=np.int32),
        excluded=np.zeros(n, dtype=bool),
    )
    with pytest.raises(ReleaseCorpusMismatchError, match=r"has sim \(5, 5\)"), TestClient(
        create_app(str(config))
    ):
        pass


def test_a_second_model_that_disagrees_refuses_to_start(tmp_path: Path) -> None:
    config = _write_release(tmp_path, BASE_NAMES)
    path = tmp_path / "runs/active/resubmit/unlabelled/unlabelled_predictions_sif_abtt.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    other = [
        {**row, "model": "bowphs/PhilTa", "filename": rows[-1 - i]["filename"]}
        for i, row in enumerate(rows)
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + other)
    with pytest.raises(ReleaseCorpusMismatchError, match="bowphs_PhilTa"), TestClient(
        create_app(str(config))
    ):
        pass
