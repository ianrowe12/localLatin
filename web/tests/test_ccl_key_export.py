"""The CCL key assignments export (issue #270).

One resolved row per reviewed query, for CCL's import. Most tests drive
`FeedbackDB.ccl_key_assignments` directly over rows written with explicit
timestamps, because "newest wins" and the decided-at columns are about time and
the save path stamps every row with the same second. The route tests at the end
go through the real API: the role guard, the file name, the data release, and a
multi-rank pick whose second directory is resolved from the served ranking.
"""

from __future__ import annotations

import asyncio
import csv
import io
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from web.services import ccl_key_export
from web.services.data_store import read_data_release
from web.services.feedback_db import FeedbackDB
from web.tests.test_reviewer_dirs import _client, _register

LABELLED = ["CTOU.567.16", "CARL.501?.18", "Capit.Martini.7", "candidate-a"]
FILENAMES = {
    1: "BAV1341.13v.1.txt",
    2: "BAV1341.13v.2.txt",
    3: "BAV1341.13v.3.txt",
    4: "BAV1341.14r.1.txt",
}
EXPORTED_AT = "2026-11-11T12:00:00Z"
RELEASE = "data-20261101"


async def _open(tmp_path: Path) -> FeedbackDB:
    db = FeedbackDB(tmp_path / "feedback.db")
    await db.connect()
    return db


async def _put(
    db: FeedbackDB,
    *,
    query_id: int = 1,
    outcome: str,
    reviewer: str = "Alice Example",
    account: int | None = 1,
    model: str = "model_one",
    variant: str | None = "sif_abtt",
    ts: str = "2026-11-01 10:00:00",
    correct_rank: int | None = None,
    correct_dir: str | None = None,
    selected: str | None = None,
    ccl_key: str | None = None,
    notes: str = "",
) -> None:
    assert db._db is not None
    await db._db.execute(
        """INSERT INTO feedback
               (query_id, timestamp, model_slug, variant, outcome, correct_rank,
                correct_dir, selected_ranks_json, notes, reviewer,
                reviewer_account_id, ccl_key, schema_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 2)""",
        (
            query_id,
            ts,
            model,
            variant,
            outcome,
            correct_rank,
            correct_dir,
            selected,
            notes,
            reviewer,
            account,
            ccl_key,
        ),
    )
    await db._db.commit()


async def _pick(db: FeedbackDB, directory: str, rank: int = 1, **extra: object) -> None:
    await _put(db, outcome="matched_rank", correct_rank=rank, correct_dir=directory, **extra)


async def _typed(db: FeedbackDB, key: str | None, **extra: object) -> None:
    await _put(db, outcome="none_of_top_k", correct_rank=0, ccl_key=key, **extra)


async def _reviewer_dir(db: FeedbackDB, dir_id: str, *, key: str, label: str) -> None:
    assert db._db is not None
    await db._db.execute(
        "INSERT INTO reviewer_dirs (dir_id, label, ccl_key, seed_query_id) VALUES (?, ?, ?, ?)",
        (dir_id, label, key, 4),
    )
    await db._db.commit()


async def _export(
    db: FeedbackDB,
    *,
    ranking: dict[tuple, str] | None = None,
    status: str | None = None,
) -> list[dict]:
    async def rank_dir(query_id: int, slug: str, variant: str | None, rank: int) -> str | None:
        return (ranking or {}).get((query_id, slug, variant, rank))

    return await db.ccl_key_assignments(
        filename_by_query=FILENAMES,
        labelled_dir_names=LABELLED,
        rank_dir=rank_dir,
        data_release=RELEASE,
        exported_at=EXPORTED_AT,
        status=status,
    )


def _only(rows: list[dict]) -> dict:
    assert len(rows) == 1, rows
    return rows[0]


def _scenario(tmp_path: Path, build, **export_args) -> list[dict]:  # type: ignore[no-untyped-def]
    async def go() -> list[dict]:
        db = await _open(tmp_path)
        try:
            await build(db)
            return await _export(db, **export_args)
        finally:
            await db.close()

    return asyncio.run(go())


# --- statuses ----------------------------------------------------------------


def test_two_reviewers_picking_the_same_directory_agree(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _pick(db, "CTOU.567.16", rank=3)
        await _pick(
            db, "CTOU.567.16", rank=3, reviewer="Bruno Example", account=2,
            ts="2026-11-02 09:00:00",
        )

    row = _only(_scenario(tmp_path, build))
    assert row["xml_id"] == "BAV1341.13v.1"
    assert row["status"] == "agreed"
    assert row["ccl_key"] == "CTOU.567.16"
    assert row["conflicting_keys"] == ""
    assert row["key_source"] == "model_suggestion"
    assert row["suggested_by"] == "model_one:sif_abtt:3"
    assert row["key_shaped"] == "true"
    assert row["reviewer_count"] == 2
    assert row["reviewers"] == "Alice Example | Bruno Example"
    assert row["first_decided_at"] == "2026-11-01T10:00:00Z"
    assert row["last_decided_at"] == "2026-11-02T09:00:00Z"
    assert row["export_schema_version"] == ccl_key_export.EXPORT_SCHEMA_VERSION == 1
    assert row["exported_at"] == EXPORTED_AT
    assert row["data_release"] == RELEASE


def test_reviewers_naming_different_keys_conflict(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _pick(db, "CTOU.567.16", rank=1)
        await _typed(db, "CARL.501?.18", reviewer="Bruno Example", account=2)

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "conflict"
    # No key is asserted when the reviewers disagree.
    assert row["ccl_key"] == ""
    assert row["conflicting_keys"] == "CARL.501?.18 | CTOU.567.16"
    assert row["key_source"] == "both"
    # In a conflict each suggestion says which key it was for.
    assert row["suggested_by"] == "CTOU.567.16=model_one:sif_abtt:1"
    assert row["key_shaped"] == ""
    assert row["reviewer_count"] == 2


def test_none_without_a_key_is_no_key(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, None)

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "no_key"
    assert row["ccl_key"] == ""
    assert row["key_source"] == ""
    assert row["suggested_by"] == ""
    assert row["key_shaped"] == ""
    assert row["reviewer_count"] == 1
    assert row["reviewers"] == "Alice Example"


def test_a_typed_key_is_reviewer_typed(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "DSIR.384.255 cap. 11")

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "agreed"
    assert row["ccl_key"] == "DSIR.384.255 cap. 11"
    assert row["key_source"] == "reviewer_typed"
    assert row["suggested_by"] == ""
    assert row["key_shaped"] == "true"


def test_a_key_reached_both_ways_is_both(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _pick(db, "CTOU.567.16", rank=2)
        await _typed(db, "CTOU.567.16", reviewer="Bruno Example", account=2)

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "agreed"
    assert row["ccl_key"] == "CTOU.567.16"
    assert row["key_source"] == "both"
    assert row["suggested_by"] == "model_one:sif_abtt:2"


def test_a_no_key_none_beside_a_typed_key_is_a_conflict(tmp_path: Path) -> None:
    """None without a key is an answer of its own, never an abstention."""

    async def build(db: FeedbackDB) -> None:
        await _typed(db, None)
        await _typed(db, "CTOU.567.16", reviewer="Bruno Example", account=2)

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "conflict"
    assert row["ccl_key"] == ""
    assert row["conflicting_keys"] == "CTOU.567.16 | (none_of_top_k)"
    assert row["reviewer_count"] == 2


def test_every_decision_a_no_key_none_stays_no_key(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, None)
        await _typed(db, None, reviewer="Bruno Example", account=2)

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "no_key"
    assert row["conflicting_keys"] == ""
    assert row["reviewer_count"] == 2


# Review round 1, finding 1: the two reproduced scenarios.


def test_a_none_reviewer_who_rejected_the_key_is_not_reported_as_agreeing(
    tmp_path: Path,
) -> None:
    async def build(db: FeedbackDB) -> None:
        await _pick(db, "CTOU.567.16", rank=2, reviewer="Alice", account=1)
        await _typed(db, None, reviewer="Bob", account=2, ts="2026-11-02 10:00:00")

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "conflict"
    # Nobody is credited with a key in a row that carries none.
    assert row["ccl_key"] == ""
    assert row["conflicting_keys"] == "CTOU.567.16 | (none_of_top_k)"
    assert row["suggested_by"] == "CTOU.567.16=model_one:sif_abtt:2"
    assert row["reviewers"] == "Alice | Bob"


def test_one_reviewer_picking_a_key_then_none_under_another_model_conflicts(
    tmp_path: Path,
) -> None:
    async def build(db: FeedbackDB) -> None:
        await _pick(db, "CTOU.567.16", rank=2, model="model_one", ts="2026-11-01 08:00:00")
        await _typed(db, None, model="model_two", ts="2026-11-02 08:00:00")

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "conflict"
    assert row["ccl_key"] == ""
    assert row["conflicting_keys"] == "CTOU.567.16 | (none_of_top_k)"
    assert row["reviewer_count"] == 1


# --- key forms ---------------------------------------------------------------


def test_case_only_differences_agree_and_print_the_key_shaped_form(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "ctou.999.1", ts="2026-11-01 08:00:00")
        await _typed(
            db, "CTOU.999.1", reviewer="Bruno Example", account=2, ts="2026-11-01 09:00:00"
        )
        await _typed(
            db, "Ctou.999.1", reviewer="Carla Example", account=3, ts="2026-11-01 10:00:00"
        )

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "agreed"
    assert row["conflicting_keys"] == ""
    assert row["ccl_key"] == "CTOU.999.1"
    assert row["key_shaped"] == "true"


def test_case_only_differences_otherwise_print_the_earliest_form(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(
            db, "CAN.APOST.88", reviewer="Bruno Example", account=2, ts="2026-11-01 09:00:00"
        )
        await _typed(db, "Can.apost.88", ts="2026-11-01 08:00:00")

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "agreed"
    assert row["ccl_key"] == "Can.apost.88"


def test_case_only_differences_print_the_labelled_directory_s_name(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "ctou.567.16", ts="2026-11-01 08:00:00")
        await _typed(
            db, "Ctou.567.16", reviewer="Bruno Example", account=2, ts="2026-11-01 09:00:00"
        )

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "agreed"
    assert row["ccl_key"] == "CTOU.567.16"


def test_a_key_that_is_not_key_shaped_is_flagged_not_filtered(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "Can.apost.49")
        await _pick(db, "Capit.Martini.7", query_id=2)

    rows = _scenario(tmp_path, build)
    assert [(r["xml_id"], r["ccl_key"], r["key_shaped"], r["status"]) for r in rows] == [
        ("BAV1341.13v.1", "Can.apost.49", "false", "agreed"),
        ("BAV1341.13v.2", "Capit.Martini.7", "false", "agreed"),
    ]


def test_an_edition_citation_with_a_semicolon_splits_back_cleanly(tmp_path: Path) -> None:
    citation = (
        "(ed. MOMMSEN, as Gildas, Epist.deperd.fragm. 1, in Chronica minora, 86, "
        "lines 1-10; trans. WINTERBOTTOM, Gildas, 80)"
    )

    async def build(db: FeedbackDB) -> None:
        await _pick(db, citation)
        await _typed(db, "CTOU.567.16", reviewer="Bruno Example", account=2)

    row = _only(_scenario(tmp_path, build))
    assert row["conflicting_keys"].split(ccl_key_export.LIST_SEPARATOR) == [
        citation,
        "CTOU.567.16",
    ]


def test_a_legacy_reviewer_directory_pick_needs_a_check(tmp_path: Path) -> None:
    """Ranks 11-15 were pre-#196 reviewer directories, scored live: a human looks."""

    async def build(db: FeedbackDB) -> None:
        await _reviewer_dir(db, "reviewer-dir-keyed", key="CTOU.700.2", label="CTOU.700.2")
        await _reviewer_dir(
            db, "reviewer-dir-siglum", key="", label="New directory from BAV1341.14r.1"
        )
        await _pick(db, "reviewer-dir-keyed", rank=11)
        await _pick(db, "reviewer-dir-siglum", rank=11, query_id=2)

    rows = {r["xml_id"]: r for r in _scenario(tmp_path, build)}
    for xml_id in ("BAV1341.13v.1", "BAV1341.13v.2"):
        assert rows[xml_id]["status"] == "needs_check"
        assert rows[xml_id]["ccl_key"] == ""
        assert rows[xml_id]["conflicting_keys"] == "(unresolved)"
        assert rows[xml_id]["suggested_by"] == "(unresolved)=model_one:sif_abtt:11"


def test_a_trusted_pick_of_a_keyless_reviewer_directory_needs_a_check(
    tmp_path: Path,
) -> None:
    """A directory without a key names nothing; the other reviewer is not agreed with."""

    async def build(db: FeedbackDB) -> None:
        await _reviewer_dir(db, "reviewer-dir-siglum", key="", label="BAV1341.14r.1")
        await _pick(db, "reviewer-dir-siglum", rank=1)
        await _typed(db, "CTOU.567.16", reviewer="Bruno Example", account=2)

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "needs_check"
    assert row["ccl_key"] == ""
    assert row["conflicting_keys"] == "CTOU.567.16 | (unresolved)"


# --- multi-rank picks --------------------------------------------------------


def test_a_multi_rank_pick_naming_two_keys_is_a_conflict(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        # correct_dir stores the first choice only; rank 4 comes from the ranking,
        # which still holds the stored first choice at rank 2.
        await _pick(db, "CTOU.567.16", rank=2, selected="[2, 4]")

    ranking = {
        (1, "model_one", "sif_abtt", 2): "CTOU.567.16",
        (1, "model_one", "sif_abtt", 4): "CARL.501?.18",
    }
    row = _only(_scenario(tmp_path, build, ranking=ranking))
    assert row["status"] == "conflict"
    assert row["conflicting_keys"] == "CARL.501?.18 | CTOU.567.16"
    assert row["suggested_by"] == (
        "CARL.501?.18=model_one:sif_abtt:4 | CTOU.567.16=model_one:sif_abtt:2"
    )
    assert row["reviewer_count"] == 1


def test_a_multi_rank_pick_of_one_key_agrees(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _reviewer_dir(db, "reviewer-dir-same", key="ctou.567.16", label="ctou.567.16")
        await _pick(db, "CTOU.567.16", rank=1, selected="[1, 3]")

    ranking = {
        (1, "model_one", "sif_abtt", 1): "CTOU.567.16",
        (1, "model_one", "sif_abtt", 3): "reviewer-dir-same",
    }
    row = _only(_scenario(tmp_path, build, ranking=ranking))
    assert row["status"] == "agreed"
    assert row["ccl_key"] == "CTOU.567.16"
    assert row["suggested_by"] == "model_one:sif_abtt:1 | model_one:sif_abtt:3"


# Review round 1, finding 2: the four reproduced scenarios.


def test_a_multi_rank_pick_is_not_read_from_a_ranking_that_moved(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _put(
            db, outcome="matched_rank", correct_rank=1, correct_dir="CTOU.567.16",
            selected="[1, 3]",
        )

    # A release moved the stored first pick from rank 1 to rank 3, and put
    # another directory at rank 1.
    ranking = {
        (1, "model_one", "sif_abtt", 1): "CARL.501?.18",
        (1, "model_one", "sif_abtt", 3): "CTOU.567.16",
    }
    row = _only(_scenario(tmp_path, build, ranking=ranking))
    assert row["status"] == "needs_check"
    assert row["ccl_key"] == ""
    assert row["conflicting_keys"] == "CTOU.567.16 | (unresolved)"
    # Rank 3 is never listed as support for the first pick's key.
    assert row["suggested_by"] == (
        "CTOU.567.16=model_one:sif_abtt:1 | (unresolved)=model_one:sif_abtt:3"
    )


def test_a_multi_rank_pick_no_longer_served_needs_a_check(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _put(
            db, outcome="matched_rank", correct_rank=1, correct_dir="CTOU.567.16",
            selected="[1, 3]",
        )
        # Pre-variant row: there is no ranking to read at all.
        await _put(
            db, query_id=2, outcome="matched_rank", correct_rank=1,
            correct_dir="CTOU.567.16", selected="[1, 3]", variant=None,
        )

    rows = _scenario(tmp_path, build, ranking={})
    assert [(r["xml_id"], r["status"], r["ccl_key"]) for r in rows] == [
        ("BAV1341.13v.1", "needs_check", ""),
        ("BAV1341.13v.2", "needs_check", ""),
    ]


def test_a_later_legacy_reviewer_directory_pick_is_not_dropped(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _reviewer_dir(db, "reviewer-dir-aaaaaaaaaaaa", key="QXAN.512.7", label="QXAN.512.7")
        await _put(
            db, outcome="matched_rank", correct_rank=1, correct_dir="CTOU.567.16",
            selected="[1, 12]",
        )

    # Even a resolver that answered for rank 12 is not believed: no served
    # ranking has had a rank above ten since #196.
    ranking = {
        (1, "model_one", "sif_abtt", 1): "CTOU.567.16",
        (1, "model_one", "sif_abtt", 12): "reviewer-dir-aaaaaaaaaaaa",
    }
    row = _only(_scenario(tmp_path, build, ranking=ranking))
    assert row["status"] == "needs_check"
    assert row["suggested_by"] == (
        "CTOU.567.16=model_one:sif_abtt:1 | (unresolved)=model_one:sif_abtt:12"
    )


def test_no_key_is_taken_from_a_pick_nobody_can_be_shown_to_have_seen(
    tmp_path: Path,
) -> None:
    async def build(db: FeedbackDB) -> None:
        await _reviewer_dir(db, "reviewer-dir-bbbbbbbbbbbb", key="", label="BAV1341.13v.1")
        await _put(
            db, outcome="matched_rank", correct_rank=11,
            correct_dir="reviewer-dir-bbbbbbbbbbbb", selected="[11, 2]",
        )

    ranking = {(1, "model_one", "sif_abtt", 2): "CARL.501?.18"}
    row = _only(_scenario(tmp_path, build, ranking=ranking))
    assert row["status"] == "needs_check"
    assert row["ccl_key"] == ""
    assert "CARL.501?.18" not in row["conflicting_keys"]
    assert "CARL.501?.18" not in row["suggested_by"]


def test_a_legacy_row_with_no_stored_directory_is_not_read_from_today(
    tmp_path: Path,
) -> None:
    async def build(db: FeedbackDB) -> None:
        await _put(db, outcome="matched_rank", correct_rank=1, correct_dir=None)

    ranking = {(1, "model_one", "sif_abtt", 1): "CTOU.567.16"}
    row = _only(_scenario(tmp_path, build, ranking=ranking))
    assert row["status"] == "needs_check"
    assert row["ccl_key"] == ""


def test_the_first_pick_uses_the_stored_directory_not_the_ranking(tmp_path: Path) -> None:
    """correct_dir was resolved server-side at save time; it wins over a re-read."""

    async def build(db: FeedbackDB) -> None:
        await _pick(db, "CTOU.567.16", rank=1)

    ranking = {(1, "model_one", "sif_abtt", 1): "CARL.501?.18"}
    row = _only(_scenario(tmp_path, build, ranking=ranking))
    assert row["ccl_key"] == "CTOU.567.16"


# --- what counts -------------------------------------------------------------


def test_skipped_and_legacy_rows_are_excluded(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _put(db, query_id=1, outcome="skipped", notes="illegible")
        await _put(db, query_id=2, outcome="legacy_unresolved")
        await _typed(db, "CTOU.567.16", query_id=3)

    rows = _scenario(tmp_path, build)
    assert [r["xml_id"] for r in rows] == ["BAV1341.13v.3"]


def test_a_later_skip_does_not_retract_a_decision(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "CTOU.567.16", ts="2026-11-01 08:00:00")
        await _put(db, outcome="skipped", notes="second look", ts="2026-11-02 08:00:00")

    row = _only(_scenario(tmp_path, build))
    assert row["ccl_key"] == "CTOU.567.16"
    assert row["last_decided_at"] == "2026-11-01T08:00:00Z"


def test_a_reviewer_counts_once_and_their_newest_decision_wins(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "CARL.501?.18", ts="2026-11-01 08:00:00")
        await _pick(db, "CTOU.567.16", ts="2026-11-03 08:00:00")

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "agreed"
    assert row["ccl_key"] == "CTOU.567.16"
    assert row["reviewer_count"] == 1
    assert row["reviewers"] == "Alice Example"
    # The superseded decision does not count, so it does not date the row.
    assert row["first_decided_at"] == "2026-11-03T08:00:00Z"


def test_newest_breaks_a_same_second_tie_by_row_id(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "CARL.501?.18", ts="2026-11-01 08:00:00")
        await _typed(db, "CTOU.567.16", ts="2026-11-01 08:00:00")

    row = _only(_scenario(tmp_path, build))
    assert row["ccl_key"] == "CTOU.567.16"


def test_one_reviewer_naming_different_keys_under_two_models_is_a_conflict(
    tmp_path: Path,
) -> None:
    """The app still shows them both answers, so neither is silently dropped."""

    async def build(db: FeedbackDB) -> None:
        await _pick(db, "CTOU.567.16", model="model_one", ts="2026-11-01 08:00:00")
        await _typed(db, "CARL.501?.18", model="model_two", ts="2026-11-02 08:00:00")

    row = _only(_scenario(tmp_path, build))
    assert row["status"] == "conflict"
    assert row["conflicting_keys"] == "CARL.501?.18 | CTOU.567.16"
    assert row["reviewer_count"] == 1


def test_a_reviewer_s_name_comes_from_their_newest_decision(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "CTOU.567.16", reviewer="A. Example", model="model_one",
                     ts="2026-11-01 08:00:00")
        await _typed(db, "CTOU.567.16", reviewer="Alice Example", model="model_two",
                     ts="2026-11-02 08:00:00")

    row = _only(_scenario(tmp_path, build))
    assert row["reviewer_count"] == 1
    assert row["reviewers"] == "Alice Example"


def test_pre_account_rows_are_told_apart_by_name(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "CTOU.567.16", reviewer="Old Reviewer", account=None)
        await _typed(db, "CTOU.567.16", reviewer="Older Reviewer", account=None)

    row = _only(_scenario(tmp_path, build))
    assert row["reviewer_count"] == 2
    assert row["reviewers"] == "Old Reviewer | Older Reviewer"


def test_a_query_the_corpus_does_not_know_is_left_out(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "CTOU.567.16", query_id=99)
        await _typed(db, "CTOU.567.16", query_id=2)

    rows = _scenario(tmp_path, build)
    assert [r["xml_id"] for r in rows] == ["BAV1341.13v.2"]


def test_the_status_filter_keeps_only_that_status(tmp_path: Path) -> None:
    async def build(db: FeedbackDB) -> None:
        await _typed(db, "CTOU.567.16", query_id=1)
        await _typed(db, None, query_id=2)

    rows = _scenario(tmp_path, build, status="no_key")
    assert [r["xml_id"] for r in rows] == ["BAV1341.13v.2"]


def test_the_export_covers_exactly_the_queries_the_app_calls_reviewed(
    tmp_path: Path,
) -> None:
    async def go() -> tuple[set[str], set[int]]:
        db = await _open(tmp_path)
        try:
            await _typed(db, None, query_id=1)
            await _pick(db, "CTOU.567.16", query_id=2)
            await _put(db, query_id=3, outcome="skipped", notes="x")
            await _put(db, query_id=4, outcome="legacy_unresolved")
            rows = await _export(db)
            return {r["xml_id"] for r in rows}, await db.get_reviewed_query_ids()
        finally:
            await db.close()

    exported, reviewed = asyncio.run(go())
    assert exported == {ccl_key_export.xml_id_of(FILENAMES[q]) for q in reviewed}
    assert reviewed == {1, 2}


def test_xml_id_strips_only_the_txt_suffix() -> None:
    assert ccl_key_export.xml_id_of("BAV1341.13v.1.txt") == "BAV1341.13v.1"
    assert ccl_key_export.xml_id_of("BN2123.89r.5") == "BN2123.89r.5"


def test_the_csv_has_the_issue_s_columns_in_order() -> None:
    text = ccl_key_export.to_csv([])
    assert text.splitlines()[0].split(",") == [
        "xml_id",
        "ccl_key",
        "status",
        "conflicting_keys",
        "key_source",
        "suggested_by",
        "key_shaped",
        "reviewer_count",
        "reviewers",
        "first_decided_at",
        "last_decided_at",
        "export_schema_version",
        "exported_at",
        "data_release",
    ]


# --- data release ------------------------------------------------------------


def test_the_data_release_is_read_from_deploy_s_state_file(tmp_path: Path) -> None:
    state = tmp_path / "installed.state"
    state.write_text("data-20261101 " + "a" * 64 + "\n", encoding="utf-8")
    assert read_data_release(state) == "data-20261101"
    assert read_data_release(tmp_path / "missing.state") == ""
    state.write_text("not,a tag\n", encoding="utf-8")
    assert read_data_release(state) == ""


# --- the route ---------------------------------------------------------------


def _approve_reviewer(client: TestClient) -> None:
    registration = client.post(
        "/api/auth/register",
        json={
            "username": "scholar",
            "display_name": "Scholar",
            "password": "correct horse battery staple",
        },
    )
    account_id = registration.json()["account"]["id"]
    assert client.post(f"/api/auth/accounts/{account_id}/approve").status_code == 200


def _sign_in(client: TestClient, username: str) -> None:
    client.post("/api/auth/signout")
    response = client.post(
        "/api/auth/signin",
        json={"username": username, "password": "correct horse battery staple"},
    )
    assert response.status_code == 200, response.text


def test_the_route_requires_sign_in(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        assert client.get("/api/feedback/export/ccl_keys").status_code == 401
    finally:
        client.__exit__(None, None, None)


def test_the_route_refuses_a_reviewer(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "PI")
        _approve_reviewer(client)
        _sign_in(client, "scholar")
        response = client.get("/api/feedback/export/ccl_keys")
        assert response.status_code == 403
        assert "Scholar" not in response.text
    finally:
        client.__exit__(None, None, None)


def test_the_route_serves_the_resolved_csv(tmp_path: Path) -> None:
    """End to end: two reviewers, one multi-rank pick, a dated attachment."""
    client = _client(tmp_path)
    try:
        state = tmp_path / ".deploy-cache" / "installed.state"
        state.parent.mkdir()
        state.write_text("data-20261101 " + "b" * 64 + "\n", encoding="utf-8")
        # build_store already ran; restart so the state file is read at startup.
        client.__exit__(None, None, None)
        client.__enter__()

        _register(client, "pi", "PI")
        _approve_reviewer(client)
        # Query 1: PI selects ranks 1 and 2 (candidate-a, candidate-b).
        saved = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "matched_rank",
                "correct_rank": 1,
                "selected_ranks": [1, 2],
                "notes": "",
            },
        )
        assert saved.status_code == 201, saved.text
        # Query 2: both reviewers pick rank 1.
        for username in ("pi", "scholar"):
            _sign_in(client, username)
            ok = client.post(
                "/api/feedback",
                json={"query_id": 2, "model_slug": "bowphs/LaTa", "correct_rank": 1},
            )
            assert ok.status_code == 201, ok.text

        _sign_in(client, "pi")
        response = client.get("/api/feedback/export/ccl_keys")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        disposition = response.headers["content-disposition"]
        rows = list(csv.DictReader(io.StringIO(response.text)))
        assert disposition == (
            f"attachment; filename=ccl_key_assignments_{rows[0]['exported_at'][:10]}.csv"
        )
        by_id = {row["xml_id"]: row for row in rows}
        assert set(by_id) == {"query-1", "query-2"}

        multi = by_id["query-1"]
        assert multi["status"] == "conflict"
        # candidate-b came from the served ranking, not from the stored row.
        assert multi["conflicting_keys"] == "candidate-a | candidate-b"
        assert multi["suggested_by"] == (
            "candidate-a=bowphs_LaTa:sif_abtt:1 | candidate-b=bowphs_LaTa:sif_abtt:2"
        )

        agreed = by_id["query-2"]
        assert agreed["status"] == "agreed"
        assert agreed["ccl_key"] == "candidate-a"
        assert agreed["reviewer_count"] == "2"
        assert agreed["reviewers"] == "PI | Scholar"
        assert agreed["data_release"] == "data-20261101"
        assert agreed["export_schema_version"] == "1"

        filtered = client.get("/api/feedback/export/ccl_keys", params={"status": "agreed"})
        assert [r["xml_id"] for r in csv.DictReader(io.StringIO(filtered.text))] == [
            "query-2"
        ]
        assert (
            client.get("/api/feedback/export/ccl_keys", params={"status": "bogus"}).status_code
            == 422
        )
        needs_check = client.get(
            "/api/feedback/export/ccl_keys", params={"status": "needs_check"}
        )
        assert needs_check.status_code == 200
        assert list(csv.DictReader(io.StringIO(needs_check.text))) == []
    finally:
        client.__exit__(None, None, None)


def test_the_route_reports_no_release_when_none_is_installed(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "PI")
        assert (
            client.post(
                "/api/feedback",
                json={"query_id": 1, "model_slug": "bowphs/LaTa", "correct_rank": 1},
            ).status_code
            == 201
        )
        rows = list(
            csv.DictReader(io.StringIO(client.get("/api/feedback/export/ccl_keys").text))
        )
        assert [r["data_release"] for r in rows] == [""]
    finally:
        client.__exit__(None, None, None)


def test_the_export_writes_nothing(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "PI")
        client.post(
            "/api/feedback",
            json={"query_id": 1, "model_slug": "bowphs/LaTa", "correct_rank": 1},
        )
        db_path = tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
        before = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM feedback").fetchone()
        assert client.get("/api/feedback/export/ccl_keys").status_code == 200
        after = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM feedback").fetchone()
        assert before == after == (1,)
    finally:
        client.__exit__(None, None, None)
