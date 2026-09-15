"""The CCL key an evaluator types beside "None of the top N" (issue #196).

Every branch the server can take, because each one writes something different
and none of them can be undone: the labelled corpus already holds the key, a
reviewer directory already holds it, nothing holds it, or this document already
seeds a directory of its own. Plus the two properties that matter more than any
individual branch -- the assessment and the directory action are one
transaction, and the migration that makes old directories reachable by key
touches nothing else.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.services.ccl_keys import (
    find_labelled_dir,
    is_key_shaped,
    match_form,
    normalize_ccl_key,
)
from web.services.feedback_db import FeedbackDB
from web.tests.test_reviewer_dirs import _create_dir, _signed_in, _write_fixture_data

KEY = "CTOU.567.16"


def _none(client: TestClient, query_id: int, **extra: object) -> dict:
    response = client.post(
        "/api/feedback",
        json={
            "query_id": query_id,
            "model_slug": "bowphs/LaTa",
            "outcome": "none_of_top_k",
            "correct_rank": 0,
            "notes": "",
            **extra,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


# --- normalisation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "stored"),
    [
        ("  CTOU.567.16  ", "CTOU.567.16"),
        ("CTOU.567.16\n", "CTOU.567.16"),
        ("DSIR.384.255   cap.  11", "DSIR.384.255 cap. 11"),
        ("ctou.567.16", "ctou.567.16"),
        ("  ", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_normalisation_keeps_the_reviewer_s_own_capitalisation(
    raw: str | None, stored: str
) -> None:
    assert normalize_ccl_key(raw) == stored


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("CTOU.567.16", "ctou.567.16"),
        ("CTOU.567.16", "  CTOU.567.16 "),
        ("Can.apost.49", "CAN.APOST.49"),
        ("DSIR.384.255 cap. 11", "dsir.384.255  cap.  11"),
    ],
)
def test_matching_folds_case_and_whitespace(left: str, right: str) -> None:
    assert match_form(left) == match_form(right)


def test_matching_does_not_fold_different_keys_together() -> None:
    assert match_form("CTOU.567.16") != match_form("CTOU.567.17")


@pytest.mark.parametrize(
    ("label", "shaped"),
    [
        ("CTOU.567.16", True),
        ("CARL.501?.18", True),
        ("DSIR.384.255 cap. 11", True),
        # Judged by content, not by stray whitespace.
        ("  CARL.501?.18  ", True),
        # Real CCL keys that the taxonomy rule deliberately does not call
        # key-*shaped*. A reviewer may type them, and the live lookup accepts
        # them; only the migration's backfill consults this predicate.
        ("Can.apost.49", False),
        ("Capit.Martini.7", False),
        ("New directory from query-2", False),
        ("", False),
    ],
)
def test_key_shaped_labels_are_the_taxonomy_rule(label: str, shaped: bool) -> None:
    assert is_key_shaped(label) is shaped


def test_labelled_lookup_is_case_insensitive_and_deterministic() -> None:
    names = {"Can.apost.49": [], "CTOU.567.16": [], "ctou.567.16": []}
    assert find_labelled_dir("can.apost.49", names) == "Can.apost.49"
    assert find_labelled_dir(" CTOU.567.16 ", names) == "CTOU.567.16"
    assert find_labelled_dir("CTOU.567.99", names) is None
    assert find_labelled_dir("", names) is None


# --- the four branches -----------------------------------------------------


def test_without_a_key_only_the_assessment_is_recorded(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        saved = _none(client, 1)
        assert saved["outcome"] == "none_of_top_k"
        assert saved["correct_rank"] == 0
        assert saved["ccl_key"] is None
        assert saved["ccl_key_action"] is None
        assert saved["ccl_key_dir"] is None
        assert client.get("/api/reviewer_dirs").json() == []
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize("typed", ["  ", "\t\n"])
def test_a_blank_key_is_no_key(tmp_path: Path, typed: str) -> None:
    client = _signed_in(tmp_path)
    try:
        saved = _none(client, 1, ccl_key=typed)
        assert saved["ccl_key"] is None
        assert saved["ccl_key_action"] is None
        assert client.get("/api/reviewer_dirs").json() == []
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize("typed", ["candidate-a", "  CANDIDATE-A  "])
def test_a_key_naming_a_labelled_directory_creates_nothing(
    tmp_path: Path, typed: str
) -> None:
    """Abigail's second case: the source IS in the labelled corpus, unranked here.

    A reviewer directory named after it would be a permanent duplicate of corpus
    data that nothing can merge away, so the assessment records the match and
    stops.
    """
    client = _signed_in(tmp_path)
    try:
        saved = _none(client, 1, ccl_key=typed)
        assert saved["ccl_key"] == typed.strip()
        assert saved["ccl_key_action"] == "matched_labelled_dir"
        assert saved["ccl_key_dir"] == "candidate-a"
        # Not a rank answer: the ranking's own candidates were all rejected.
        assert saved["correct_dir"] is None
        assert saved["correct_rank"] == 0
        assert client.get("/api/reviewer_dirs").json() == []
    finally:
        client.__exit__(None, None, None)


def test_an_unknown_key_creates_a_directory_named_by_it(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        saved = _none(client, 1, ccl_key=f"  {KEY} ")
        assert saved["ccl_key"] == KEY
        assert saved["ccl_key_action"] == "created_reviewer_dir"

        created = client.get(f"/api/reviewer_dirs/{saved['ccl_key_dir']}").json()
        # Named by the key, not by a siglum, and seeded by the document that
        # named it.
        assert created["label"] == KEY
        assert created["seed_query_id"] == 1
        assert created["member_query_ids"] == [1]
        assert created["status"] == "awaiting_match"
        assert created["created_by"] == "PI"
    finally:
        client.__exit__(None, None, None)


def test_a_known_key_joins_rather_than_duplicating(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        first = _none(client, 1, ccl_key=KEY)
        second = _none(client, 2, ccl_key=KEY.lower())
        assert second["ccl_key_action"] == "joined_reviewer_dir"
        assert second["ccl_key_dir"] == first["ccl_key_dir"]
        assert len(client.get("/api/reviewer_dirs").json()) == 1

        joined = client.get(f"/api/reviewer_dirs/{first['ccl_key_dir']}").json()
        assert joined["member_query_ids"] == [1, 2]
        # A human filed a second document into it: that, and only that, is what
        # the badge follows.
        assert joined["status"] == "matched"
    finally:
        client.__exit__(None, None, None)


def test_a_directory_created_by_the_old_route_is_joinable_by_its_key(
    tmp_path: Path,
) -> None:
    """The retired naming form could produce a key-named directory too."""
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 0, KEY)
        saved = _none(client, 1, ccl_key=KEY.lower())
        assert saved["ccl_key_action"] == "joined_reviewer_dir"
        assert saved["ccl_key_dir"] == created["dir_id"]
    finally:
        client.__exit__(None, None, None)


def test_a_non_key_label_is_a_second_handle_on_the_same_directory(
    tmp_path: Path,
) -> None:
    """A pre-#196 directory is joinable by the label the card shows.

    Its `ccl_key` is empty -- `New directory from query-0` is a filename, not a
    citation, and the migration refuses to invent a key from one. With the rank
    route gone, a key-only lookup would leave it permanently unjoinable while
    the unranked block kept showing its label, and typing that label would mint
    a SECOND directory for the same grouping. So the label joins; only the key
    column ever names a new one.
    """
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 0)
        assert created["label"] == "New directory from query-0"
        saved = _none(client, 1, ccl_key="  new directory FROM query-0 ")
        assert saved["ccl_key_action"] == "joined_reviewer_dir"
        assert saved["ccl_key_dir"] == created["dir_id"]
        assert len(client.get("/api/reviewer_dirs").json()) == 1
        members = client.get(f"/api/reviewer_dirs/{created['dir_id']}").json()
        assert members["member_query_ids"] == [0, 1]
        # The label is a handle, never a name: the directory keeps the label it
        # was created with, and gains no key.
        assert members["label"] == "New directory from query-0"
    finally:
        client.__exit__(None, None, None)


def test_the_key_column_wins_over_a_coincidental_label(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        by_key = _create_dir(client, 0, KEY)
        by_label = _create_dir(client, 2, f"About {KEY}")
        saved = _none(client, 1, ccl_key=KEY)
        assert saved["ccl_key_dir"] == by_key["dir_id"]
        assert saved["ccl_key_dir"] != by_label["dir_id"]
    finally:
        client.__exit__(None, None, None)


def test_a_repeat_join_reports_a_no_op_rather_than_a_join(tmp_path: Path) -> None:
    """The membership write is idempotent; the record says which it was.

    A second reviewer submitting the same key for a document that is already a
    member adds nothing, and a row reading "this document joined the group"
    would be a small untruth in a log nobody can correct.
    """
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 0, KEY)
        first = _none(client, 1, ccl_key=KEY, notes="first look")
        assert first["ccl_key_action"] == "joined_reviewer_dir"

        again = _none(client, 1, ccl_key=KEY, notes="second look, same answer")
        assert again["ccl_key_action"] == "already_joined"
        assert again["ccl_key_dir"] == created["dir_id"]
        assert again["id"] != first["id"]

        members = client.get(f"/api/reviewer_dirs/{created['dir_id']}").json()
        assert members["member_query_ids"] == [0, 1]
    finally:
        client.__exit__(None, None, None)


def test_the_seed_of_a_directory_re_naming_its_own_key_is_a_no_op(
    tmp_path: Path,
) -> None:
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 1, KEY)
        saved = _none(client, 1, ccl_key=KEY)
        assert saved["ccl_key_action"] == "already_joined"
        assert saved["ccl_key_dir"] == created["dir_id"]
    finally:
        client.__exit__(None, None, None)


def test_an_identical_repeat_returns_the_stored_row_and_appends_nothing(
    tmp_path: Path,
) -> None:
    """Pressing Record twice on an unchanged answer is one assertion.

    The panel shows a recorded answer with a "Change" action, so opening it,
    changing nothing and pressing again means "yes, that". The log is still
    append-only -- nothing is updated or removed -- this simply declines to add
    a row that would say exactly what the last one says.
    """
    client = _signed_in(tmp_path)
    try:
        first = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "none_of_top_k",
                "correct_rank": 0,
                "ccl_key": KEY,
                "notes": "not one of the ten",
            },
        )
        assert first.status_code == 201, first.text

        repeat = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "none_of_top_k",
                "correct_rank": 0,
                # Same assertion, typed differently.
                "ccl_key": f"  {KEY.lower()} ",
                "notes": "not one of the ten",
            },
        )
        assert repeat.status_code == 200, repeat.text
        assert repeat.json()["id"] == first.json()["id"]
        assert repeat.json()["ccl_key"] == KEY
        assert client.get("/api/stats").json()["feedback_count"] == 1
        assert len(client.get("/api/reviewer_dirs").json()) == 1

        # A revised note is a new assertion and is appended.
        revised = _none(client, 1, ccl_key=KEY, notes="on reflection, the same key")
        assert revised["id"] != first.json()["id"]
        assert revised["ccl_key_action"] == "already_joined"
        assert client.get("/api/stats").json()["feedback_count"] == 2
    finally:
        client.__exit__(None, None, None)


def test_a_repeat_without_a_key_is_also_a_single_assertion(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        first = _none(client, 1, notes="nothing here fits")
        repeat = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "none_of_top_k",
                "correct_rank": 0,
                "notes": "nothing here fits",
            },
        )
        assert repeat.status_code == 200, repeat.text
        assert repeat.json()["id"] == first["id"]
        assert client.get("/api/stats").json()["feedback_count"] == 1
    finally:
        client.__exit__(None, None, None)


def test_another_reviewer_saying_the_same_thing_is_a_second_assertion(
    tmp_path: Path,
) -> None:
    """Two reviewers agreeing is two facts, not a duplicate.

    The no-op rule is scoped to the caller's OWN newest row, exactly as the
    shared-note prefill is (issue #96): a colleague's identical answer is
    independent evidence and is appended.
    """
    client = _signed_in(tmp_path)
    try:
        first = _none(client, 1, ccl_key=KEY, notes="agreed")

        client.post(
            "/api/auth/register",
            json={
                "username": "scholar",
                "display_name": "Scholar",
                "password": "correct horse battery staple",
            },
        )
        pending = next(
            account
            for account in client.get("/api/auth/accounts?status=pending").json()
            if account["username"] == "scholar"
        )
        approved = client.post(
            f"/api/auth/accounts/{pending['id']}/approve", json={"note": ""}
        )
        assert approved.status_code == 200, approved.text
        signin = client.post(
            "/api/auth/signin",
            json={"username": "scholar", "password": "correct horse battery staple"},
        )
        assert signin.status_code == 200, signin.text

        second = _none(client, 1, ccl_key=KEY, notes="agreed")
        assert second["id"] != first["id"]
        assert second["reviewer"] == "Scholar"
        # Two rows, one directory, and the second reviewer's document was
        # already a member.
        assert second["ccl_key_action"] == "already_joined"
        connection = sqlite3.connect(
            tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
        )
        try:
            assert connection.execute(
                "SELECT COUNT(*) FROM feedback WHERE query_id = 1"
            ).fetchone()[0] == 2
        finally:
            connection.close()
    finally:
        client.__exit__(None, None, None)


def test_a_key_naming_a_shortlisted_directory_records_its_rank(
    tmp_path: Path,
) -> None:
    """The copy claims "not in the shortlist", so the record has to know.

    Resolved from the same candidate snapshot the save is already validated
    against, rather than re-read afterwards, so the rank stored is the one that
    was on screen.
    """
    client = _signed_in(tmp_path)
    try:
        # candidate-a is rank 1 for every query in the fixture.
        shortlisted = _none(client, 1, ccl_key="candidate-a")
        assert shortlisted["ccl_key_action"] == "matched_labelled_dir"
        assert shortlisted["ccl_key_rank"] == 1

        elsewhere = _none(client, 2, ccl_key="candidate-b")
        assert elsewhere["ccl_key_rank"] == 2
    finally:
        client.__exit__(None, None, None)


def test_a_key_outside_the_shortlist_records_no_rank(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        saved = _none(client, 1, ccl_key=KEY)
        assert saved["ccl_key_action"] == "created_reviewer_dir"
        assert saved["ccl_key_rank"] is None
        assert _none(client, 2)["ccl_key_rank"] is None
    finally:
        client.__exit__(None, None, None)


def test_a_group_is_never_seeded_from_an_unscorable_document(
    tmp_path: Path,
) -> None:
    """The one guard from #165's create route this path would otherwise skip.

    q3 is the fixture's blank-source query: no row in the q-q matrix, so a
    directory seeded there could never be offered and never leave
    `awaiting_match`. Unreachable through the UI today, because an excluded
    ranking already refuses an assessment, so the guard is asserted directly
    against the storage call rather than through a request that cannot get here.
    """
    import asyncio

    from web.exceptions import UnscorableSeedError
    from web.services.feedback_db import FeedbackDB

    client = _signed_in(tmp_path)
    db_path = tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
    client.__exit__(None, None, None)

    async def run() -> None:
        db = FeedbackDB(db_path)
        await db.connect()
        try:
            with pytest.raises(UnscorableSeedError):
                await db.insert_none_of_top_k(
                    query_id=3,
                    model_slug="bowphs_LaTa",
                    variant="sif_abtt",
                    notes="",
                    reviewer="PI",
                    reviewer_account_id=1,
                    ccl_key=KEY,
                    labelled_dir=None,
                    labelled_rank=None,
                    seed_scorable=False,
                    max_per_account=50,
                )
            # Nothing at all was written: not the directory, not the assessment.
            assert await db.list_reviewer_dirs() == []
            assert await db.get_feedback_for_query(3) == []

            # The same submission on a scorable document goes through.
            row, appended = await db.insert_none_of_top_k(
                query_id=1,
                model_slug="bowphs_LaTa",
                variant="sif_abtt",
                notes="",
                reviewer="PI",
                reviewer_account_id=1,
                ccl_key=KEY,
                labelled_dir=None,
                labelled_rank=None,
                seed_scorable=True,
                max_per_account=50,
            )
            assert appended is True
            assert row["ccl_key_action"] == "created_reviewer_dir"
        finally:
            await db.close()

    asyncio.run(run())


def test_a_colleagues_key_is_never_prefilled_as_your_own(tmp_path: Path) -> None:
    """The key is part of the ANSWER, so it follows the note/decision split.

    `/api/feedback/latest` merges the team's newest NOTE with the caller's own
    newest DECISION (issue #96). A key identifies the source this reviewer says
    the document belongs to, and the panel renders it back as "you recorded
    this", so inheriting a colleague's key would show one reviewer's
    identification as another's own record.
    """
    client = _signed_in(tmp_path)
    try:
        _none(client, 1, ccl_key=KEY, notes="I think this is the one")
        client.post(
            "/api/auth/register",
            json={
                "username": "scholar",
                "display_name": "Scholar",
                "password": "correct horse battery staple",
            },
        )
        pending = next(
            account
            for account in client.get("/api/auth/accounts?status=pending").json()
            if account["username"] == "scholar"
        )
        client.post(f"/api/auth/accounts/{pending['id']}/approve", json={"note": ""})
        client.post(
            "/api/auth/signin",
            json={"username": "scholar", "password": "correct horse battery staple"},
        )

        latest = client.get(
            "/api/feedback/latest",
            params={"query_id": 1, "model": "bowphs/LaTa"},
        ).json()
        # The colleague's prose arrives, attributed...
        assert latest["notes"] == "I think this is the one"
        assert latest["reviewer"] == "PI"
        # ...and none of their answer does.
        assert latest["outcome"] == "legacy_unresolved"
        assert latest["ccl_key"] is None
        assert latest["ccl_key_action"] is None
        assert latest["ccl_key_dir"] is None
        assert latest["ccl_key_rank"] is None
    finally:
        client.__exit__(None, None, None)


def test_your_own_key_comes_back_on_a_revisit(tmp_path: Path) -> None:
    """What the panel renders as "you recorded this" (issue #196 blocking fix)."""
    client = _signed_in(tmp_path)
    try:
        saved = _none(client, 1, ccl_key=KEY, notes="not in the ten")
        latest = client.get(
            "/api/feedback/latest",
            params={"query_id": 1, "model": "bowphs/LaTa"},
        ).json()
        assert latest["outcome"] == "none_of_top_k"
        assert latest["ccl_key"] == KEY
        assert latest["ccl_key_action"] == "created_reviewer_dir"
        assert latest["ccl_key_dir"] == saved["ccl_key_dir"]
    finally:
        client.__exit__(None, None, None)


def test_a_key_is_refused_on_a_rank_answer(tmp_path: Path) -> None:
    """A rank already names a directory; a key beside it is a second answer."""
    client = _signed_in(tmp_path)
    try:
        response = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "matched_rank",
                "correct_rank": 1,
                "ccl_key": KEY,
                "notes": "",
            },
        )
        assert response.status_code == 422, response.text
        assert client.get("/api/stats").json()["feedback_count"] == 0
        assert client.get("/api/reviewer_dirs").json() == []
    finally:
        client.__exit__(None, None, None)


def test_an_overlong_key_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    client = _signed_in(tmp_path)
    try:
        response = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "none_of_top_k",
                "correct_rank": 0,
                "ccl_key": "x" * 201,
                "notes": "",
            },
        )
        assert response.status_code == 422, response.text
        assert client.get("/api/stats").json()["feedback_count"] == 0
    finally:
        client.__exit__(None, None, None)


def test_the_per_account_cap_refuses_the_whole_submission(tmp_path: Path) -> None:
    """Nothing is written, so the reviewer can still record the answer alone.

    The cap exists because a directory is permanent. Refusing the assessment
    with it is the conservative half of the trade: a half-applied submission
    would leave an assessment citing a key no directory carries.
    """
    from web.services import reviewer_dirs as svc

    client = _signed_in(tmp_path, n_queries=8)
    original = svc.MAX_REVIEWER_DIRS_PER_ACCOUNT
    svc.MAX_REVIEWER_DIRS_PER_ACCOUNT = 1
    try:
        _none(client, 0, ccl_key="CTOU.567.1")
        response = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "none_of_top_k",
                "correct_rank": 0,
                "ccl_key": "CTOU.567.2",
                "notes": "",
            },
        )
        assert response.status_code == 429, response.text
        assert response.json()["error"]["code"] == "REVIEWER_DIR_LIMIT"
        assert client.get("/api/stats").json()["feedback_count"] == 1
        assert len(client.get("/api/reviewer_dirs").json()) == 1

        # Without the key the same answer is recorded normally.
        assert _none(client, 1)["ccl_key"] is None
    finally:
        svc.MAX_REVIEWER_DIRS_PER_ACCOUNT = original
        client.__exit__(None, None, None)


# --- atomicity -------------------------------------------------------------


@pytest.mark.parametrize("failing_table", ["reviewer_dirs", "reviewer_dir_members"])
def test_a_failed_directory_write_records_no_assessment(
    tmp_path: Path, failing_table: str
) -> None:
    """One transaction: an assessment citing a key must not outlive the write.

    Injected with a trigger rather than argued about, and on both halves of the
    directory write, because the feedback row is inserted last and a
    non-transactional version would already have committed it.
    """
    client = _signed_in(tmp_path)
    db_path = tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            f"""CREATE TRIGGER fail_dir BEFORE INSERT ON {failing_table}
                BEGIN SELECT RAISE(ABORT, 'injected directory failure'); END"""
        )
        connection.commit()
    finally:
        connection.close()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="injected directory failure"):
            client.post(
                "/api/feedback",
                json={
                    "query_id": 1,
                    "model_slug": "bowphs/LaTa",
                    "outcome": "none_of_top_k",
                    "correct_rank": 0,
                    "ccl_key": KEY,
                    "notes": "This must not survive on its own.",
                },
            )
        observer = sqlite3.connect(db_path)
        try:
            for table in ("feedback", "reviewer_dirs", "reviewer_dir_members"):
                assert observer.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[
                    0
                ] == 0, table
        finally:
            observer.close()
    finally:
        client.__exit__(None, None, None)


def test_the_export_carries_the_key_columns(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        _none(client, 1, ccl_key=KEY)
        _none(client, 2)
        body = client.get("/api/feedback/export").text
        rows = list(csv.DictReader(io.StringIO(body)))
        assert [row["ccl_key"] for row in rows] == [KEY, ""]
        assert [row["ccl_key_action"] for row in rows] == ["created_reviewer_dir", ""]
        assert rows[0]["ccl_key_dir"].startswith("reviewer-dir-")
        assert rows[1]["ccl_key_dir"] == ""
    finally:
        client.__exit__(None, None, None)


# --- migration -------------------------------------------------------------


def _pre_196_db(path: Path) -> None:
    """A database from before the key column, with directories and feedback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            model_slug TEXT NOT NULL,
            outcome TEXT,
            correct_rank INTEGER,
            correct_dir TEXT,
            notes TEXT NOT NULL DEFAULT '',
            reviewer TEXT NOT NULL
        );
        INSERT INTO feedback
            (query_id, timestamp, model_slug, outcome, correct_rank, correct_dir, notes, reviewer)
        VALUES
            (7, '2026-01-01 00:00:00', 'bowphs_LaTa', 'matched_rank', 11,
             'reviewer-dir-old', 'filed by rank', 'Abigail');
        CREATE TABLE reviewer_dirs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dir_id TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL,
            seed_query_id INTEGER NOT NULL,
            model_slug TEXT NOT NULL DEFAULT '',
            variant TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            created_by TEXT NOT NULL DEFAULT '',
            created_by_account_id INTEGER
        );
        INSERT INTO reviewer_dirs (dir_id, label, seed_query_id, created_by)
        VALUES ('reviewer-dir-old', 'New directory from BN2123.89r.5', 5, 'Abigail'),
               ('reviewer-dir-key', 'CTOU.567.16', 6, 'Abigail'),
               ('reviewer-dir-pad', '  CARL.501?.18  ', 7, 'Abigail');
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
        INSERT INTO reviewer_dir_members (dir_id, query_id, role)
        VALUES ('reviewer-dir-old', 5, 'seed'), ('reviewer-dir-key', 6, 'seed'),
               ('reviewer-dir-pad', 7, 'seed');
        """
    )
    connection.commit()
    connection.close()


def _dump(path: Path, table: str) -> list[tuple]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    finally:
        connection.close()


def test_migration_keeps_every_name_and_only_backfills_real_keys(
    tmp_path: Path,
) -> None:
    import asyncio

    path = tmp_path / "feedback.db"
    _pre_196_db(path)
    members_before = _dump(path, "reviewer_dir_members")

    async def run() -> None:
        db = FeedbackDB(path)
        await db.connect()
        try:
            records = {r["dir_id"]: r for r in await db.list_reviewer_dirs()}
            # Labels are untouched, including the padded one: both tables are
            # append-only and there is no rename.
            assert records["reviewer-dir-old"]["label"] == (
                "New directory from BN2123.89r.5"
            )
            assert records["reviewer-dir-pad"]["label"] == "  CARL.501?.18  "
            # Only a label that IS a key becomes one, normalised for storage.
            assert records["reviewer-dir-old"]["ccl_key"] == ""
            assert records["reviewer-dir-key"]["ccl_key"] == "CTOU.567.16"
            assert records["reviewer-dir-pad"]["ccl_key"] == "CARL.501?.18"

            # The backfilled keys are live: this is what stops an evaluator
            # making a second directory for a key somebody already has.
            found = await db.get_reviewer_dir_by_key("ctou.567.16")
            assert found is not None and found["dir_id"] == "reviewer-dir-key"
            assert await db.get_reviewer_dir_by_key("carl.501?.18") is not None
            # A non-key label is not a KEY, but it is still a handle: with the
            # rank route gone it is the only way these rows can gain a member.
            by_label = await db.get_reviewer_dir_by_key(
                "new directory from BN2123.89r.5"
            )
            assert by_label is not None
            assert by_label["dir_id"] == "reviewer-dir-old"
            assert by_label["ccl_key"] == ""
            assert await db.get_reviewer_dir_by_key("") is None

            # The reviewer's own history is preserved verbatim, including a
            # rank 11 that no ranking offers any more.
            rows = await db.get_feedback_for_query(7)
            assert len(rows) == 1
            assert rows[0]["correct_rank"] == 11
            assert rows[0]["correct_dir"] == "reviewer-dir-old"
            assert rows[0]["notes"] == "filed by rank"
            assert rows[0]["ccl_key"] is None
        finally:
            await db.close()

    asyncio.run(run())
    after_first = _dump(path, "reviewer_dirs")
    assert _dump(path, "reviewer_dir_members") == members_before

    async def reopen() -> None:
        db = FeedbackDB(path)
        await db.connect()
        await db.close()

    asyncio.run(reopen())
    # Idempotent: the second boot writes nothing at all.
    assert _dump(path, "reviewer_dirs") == after_first
    assert _dump(path, "reviewer_dir_members") == members_before


def test_a_fresh_database_gains_the_columns_without_any_row(tmp_path: Path) -> None:
    import asyncio

    path = tmp_path / "fresh.db"

    async def run() -> None:
        db = FeedbackDB(path)
        await db.connect()
        await db.close()

    asyncio.run(run())
    connection = sqlite3.connect(path)
    try:
        dirs = {
            row[1] for row in connection.execute("PRAGMA table_info(reviewer_dirs)")
        }
        feedback = {
            row[1] for row in connection.execute("PRAGMA table_info(feedback)")
        }
    finally:
        connection.close()
    assert "ccl_key" in dirs
    assert {"ccl_key", "ccl_key_action", "ccl_key_dir"} <= feedback


def test_the_fixture_app_boots_on_a_pre_196_database(tmp_path: Path) -> None:
    """The deployed path: an existing feedback.db is opened, not replaced."""
    from web.app import create_app

    config_path = _write_fixture_data(tmp_path)
    _pre_196_db(tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db")
    client = TestClient(create_app(str(config_path)))
    client.__enter__()
    try:
        client.post(
            "/api/auth/register",
            json={
                "username": "pi",
                "display_name": "PI",
                "password": "correct horse battery staple",
            },
        )
        listed = {d["dir_id"]: d for d in client.get("/api/reviewer_dirs").json()}
        assert set(listed) == {
            "reviewer-dir-old",
            "reviewer-dir-key",
            "reviewer-dir-pad",
        }
        saved = _none(client, 1, ccl_key="ctou.567.16")
        assert saved["ccl_key_action"] == "joined_reviewer_dir"
        assert saved["ccl_key_dir"] == "reviewer-dir-key"
    finally:
        client.__exit__(None, None, None)
