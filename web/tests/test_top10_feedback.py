from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

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
    (unlabelled / "query-0.txt").write_text("query text", encoding="utf-8")

    fieldnames = ["file_id", "filename", "model", "layer", "pooling"]
    for rank in range(1, 11):
        dir_name = f"candidate-{rank}"
        (labelled / dir_name).mkdir(parents=True)
        (labelled / dir_name / f"{rank}.txt").write_text(
            f"candidate text {rank}", encoding="utf-8"
        )
        fieldnames.extend([f"rank{rank}_dir", f"rank{rank}_score"])

    row = {
        "file_id": 0,
        "filename": "query-0.txt",
        "model": "bowphs/LaTa",
        "layer": 12,
        "pooling": "mean",
    }
    for rank in range(1, 11):
        row[f"rank{rank}_dir"] = f"candidate-{rank}"
        row[f"rank{rank}_score"] = 1 - rank / 100

    with (predictions / "unlabelled_predictions_sif_abtt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

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


def _signed_in_client(tmp_path: Path) -> TestClient:
    client = TestClient(create_app(str(_write_fixture_data(tmp_path))))
    client.__enter__()
    response = client.post(
        "/api/auth/register",
        json={
            "username": "pi",
            "display_name": "PI",
            "password": "correct horse battery staple",
        },
    )
    assert response.status_code == 201
    return client


def _install_append_only_triggers(root: Path) -> None:
    """Make the feedback table reject UPDATE and DELETE at the SQLite level.

    Triggers live in the schema, so the app's own connection is bound by them:
    any code path that tried to edit a stored review instead of appending a new
    one turns into a visible 500 rather than silent data loss.

    Test-only, and installed only after startup. These must NOT be promoted
    into `_SCHEMA`: two of `_migrate()`'s normalization UPDATEs are
    unconditional and re-fire on already-normalized rows at every boot, so an
    in-schema trigger would abort startup on any DB holding a `none_of_top_k`
    or `skipped` row. See the warning on `FeedbackDB._migrate`.
    """
    db_path = root / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TRIGGER feedback_is_append_only_update
            BEFORE UPDATE ON feedback
            BEGIN SELECT RAISE(ABORT, 'feedback rows are append-only'); END;
            CREATE TRIGGER feedback_is_append_only_delete
            BEFORE DELETE ON feedback
            BEGIN SELECT RAISE(ABORT, 'feedback rows are append-only'); END;
            """
        )
        connection.commit()
    finally:
        connection.close()


def test_top_10_rank_and_selected_directory_are_persisted(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        predictions = client.get(
            "/api/query/0/predictions", params={"model": "bowphs/LaTa"}
        )
        assert predictions.status_code == 200
        assert len(predictions.json()["predictions"]) == 10
        assert predictions.json()["predictions"][9]["dir_name"] == "candidate-10"

        feedback = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 10,
                "correct_dir": "candidate-10",
                "notes": "rank ten is correct",
            },
        )
        assert feedback.status_code == 201
        assert feedback.json()["outcome"] == "matched_rank"
        assert feedback.json()["correct_rank"] == 10
        assert feedback.json()["correct_dir"] == "candidate-10"

        rows = list(csv.DictReader(client.get("/api/feedback/export").text.splitlines()))
        assert rows[0]["outcome"] == "matched_rank"
        assert rows[0]["correct_rank"] == "10"
        assert rows[0]["correct_dir"] == "candidate-10"
    finally:
        client.__exit__(None, None, None)


def test_none_of_top_10_clears_candidate_directory(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        feedback = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "outcome": "none_of_top_k",
                "correct_rank": 0,
                "correct_dir": "candidate-1",
                "notes": "none of these are correct",
            },
        )
        assert feedback.status_code == 201
        assert feedback.json()["outcome"] == "none_of_top_k"
        assert feedback.json()["correct_rank"] == 0
        assert feedback.json()["correct_dir"] is None

        rows = list(csv.DictReader(client.get("/api/feedback/export").text.splitlines()))
        assert rows[0]["outcome"] == "none_of_top_k"
        assert rows[0]["correct_rank"] == "0"
        assert rows[0]["correct_dir"] == ""
    finally:
        client.__exit__(None, None, None)


def test_latest_feedback_returns_newest_row_without_collapsing_history(
    tmp_path: Path,
) -> None:
    client = _signed_in_client(tmp_path)
    try:
        first = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 1,
                "correct_dir": "candidate-1",
                "notes": "first note",
            },
        )
        assert first.status_code == 201

        second = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 2,
                "correct_dir": "candidate-2",
                "notes": "newer note",
            },
        )
        assert second.status_code == 201

        latest = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa"},
        )
        assert latest.status_code == 200
        assert latest.json()["id"] == second.json()["id"]
        assert latest.json()["notes"] == "newer note"
        assert latest.json()["correct_rank"] == 2

        rows = list(csv.DictReader(client.get("/api/feedback/export").text.splitlines()))
        assert [row["notes"] for row in rows] == ["first note", "newer note"]
    finally:
        client.__exit__(None, None, None)


def _add_reviewer(client: TestClient, username: str, display_name: str) -> None:
    created = client.post(
        "/api/auth/accounts",
        json={
            "username": username,
            "display_name": display_name,
            "role": "reviewer",
            "password": "correct horse battery staple",
        },
    )
    assert created.status_code == 201


def _sign_in_as(client: TestClient, username: str) -> None:
    assert client.post("/api/auth/signout").status_code == 204
    signed_in = client.post(
        "/api/auth/signin",
        json={"username": username, "password": "correct horse battery staple"},
    )
    assert signed_in.status_code == 200


def test_latest_feedback_is_shared_across_reviewers_and_normalized_model(
    tmp_path: Path,
) -> None:
    """Whoever opens a query sees the newest note on it, whoever wrote it (#96)."""
    client = _signed_in_client(tmp_path)
    try:
        _add_reviewer(client, "reviewer", "Reviewer")

        pi_feedback = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 1,
                "correct_dir": "candidate-1",
                "notes": "pi note",
            },
        )
        assert pi_feedback.status_code == 201

        _sign_in_as(client, "reviewer")

        # Reviewer B has written nothing here, and still sees reviewer A's note
        # with enough identity to attribute it.
        shared = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs_LaTa"},
        )
        assert shared.status_code == 200
        assert shared.json()["id"] == pi_feedback.json()["id"]
        assert shared.json()["notes"] == "pi note"
        assert shared.json()["reviewer"] == "PI"
        assert shared.json()["reviewer_username"] == "pi"
        assert shared.json()["timestamp"]

        reviewer_feedback = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs_LaTa",
                "correct_rank": 3,
                "correct_dir": "candidate-3",
                "notes": "reviewer note",
            },
        )
        assert reviewer_feedback.status_code == 201

        latest = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa"},
        )
        assert latest.status_code == 200
        assert latest.json()["id"] == reviewer_feedback.json()["id"]
        assert latest.json()["model_slug"] == "bowphs_LaTa"
        assert latest.json()["reviewer"] == "Reviewer"
        assert latest.json()["reviewer_username"] == "reviewer"
        assert latest.json()["notes"] == "reviewer note"

        # Sharing runs both ways: A now sees B's newer note, while keeping the
        # rank A themself chose.
        _sign_in_as(client, "pi")
        back_to_pi = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa"},
        )
        assert back_to_pi.json()["id"] == reviewer_feedback.json()["id"]
        assert back_to_pi.json()["reviewer_username"] == "reviewer"
        assert back_to_pi.json()["notes"] == "reviewer note"
        assert back_to_pi.json()["correct_rank"] == 1
    finally:
        client.__exit__(None, None, None)


def test_latest_feedback_shares_the_note_but_not_the_decision(tmp_path: Path) -> None:
    """Notes are the team's, answers are each reviewer's own (issue #96).

    A rank pressed by somebody else is an answer the caller never gave, and one
    they could submit as their own by reflex, so the prefill takes prose from
    the newest row by anyone and the selection from the caller's own newest row.
    """
    client = _signed_in_client(tmp_path)
    try:
        _add_reviewer(client, "reviewer", "Reviewer")

        pi_feedback = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "selected_ranks": [2, 5],
                "notes": "two plausible readings here",
            },
        )
        assert pi_feedback.status_code == 201
        assert pi_feedback.json()["selected_ranks"] == [2, 5]

        _sign_in_as(client, "reviewer")
        first_look = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa"},
        )
        assert first_look.status_code == 200
        body = first_look.json()
        # The note and its attribution come from the PI...
        assert body["notes"] == "two plausible readings here"
        assert body["reviewer_username"] == "pi"
        # ...but none of the PI's answer does.
        assert body["selected_ranks"] is None
        assert body["correct_rank"] is None
        assert body["correct_dir"] is None
        assert body["outcome"] == "legacy_unresolved"

        # Once reviewer B answers, their own answer is what comes back, still
        # paired with whichever note is newest.
        own = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 7,
                "correct_dir": "candidate-7",
                "notes": "",
            },
        )
        assert own.status_code == 201

        second_look = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa"},
        )
        merged = second_look.json()
        assert merged["correct_rank"] == 7
        assert merged["outcome"] == "matched_rank"
        # B's row is newer but says nothing, so it does not displace the PI's
        # note. The notes box exists to surface notes; an answer saved in
        # silence must not blank out a colleague's reasoning.
        assert merged["notes"] == "two plausible readings here"
        assert merged["reviewer_username"] == "pi"
    finally:
        client.__exit__(None, None, None)


def test_an_empty_note_never_displaces_an_older_real_one(tmp_path: Path) -> None:
    """The newest row wins the notes box only if it has something to say.

    Three reviewers: A writes a note, B answers in silence, C opens the query.
    C must read A's note under A's name, not an empty box under B's.
    """
    client = _signed_in_client(tmp_path)
    try:
        _add_reviewer(client, "silent", "Silent Reviewer")
        _add_reviewer(client, "carol", "Carol Codex")

        # A (the PI) leaves the substantive note.
        assert (
            client.post(
                "/api/feedback",
                json={
                    "query_id": 0,
                    "model_slug": "bowphs/LaTa",
                    "correct_rank": 2,
                    "correct_dir": "candidate-2",
                    "notes": "the hand matches the Verona scribe",
                },
            ).status_code
            == 201
        )

        # B answers later without explaining. Newest row, blank prose.
        _sign_in_as(client, "silent")
        silent = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 9,
                "correct_dir": "candidate-9",
                "notes": "   ",
            },
        )
        assert silent.status_code == 201

        # C sees A's note, attributed to A, with no decision of their own.
        _sign_in_as(client, "carol")
        body = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa"},
        ).json()
        assert body["notes"] == "the hand matches the Verona scribe"
        assert body["reviewer"] == "PI"
        assert body["reviewer_username"] == "pi"
        assert body["correct_rank"] is None
        assert body["outcome"] == "legacy_unresolved"

        # And B, whose own row is the blank one, still gets their rank back
        # alongside A's note: the filter costs nobody their own answer.
        _sign_in_as(client, "silent")
        for_silent = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa"},
        ).json()
        assert for_silent["notes"] == "the hand matches the Verona scribe"
        assert for_silent["reviewer_username"] == "pi"
        assert for_silent["correct_rank"] == 9
    finally:
        client.__exit__(None, None, None)


def test_a_decision_saved_with_no_note_anywhere_still_prefills(tmp_path: Path) -> None:
    """The note filter must not cost a reviewer their own answer.

    When nobody has written a note at all, the shared half finds nothing. The
    caller's own row has to carry the response on its own, or reopening a query
    would silently lose the rank they picked.
    """
    client = _signed_in_client(tmp_path)
    try:
        created = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 4,
                "correct_dir": "candidate-4",
                "notes": "",
            },
        )
        assert created.status_code == 201

        body = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa"},
        ).json()
        assert body is not None
        assert body["correct_rank"] == 4
        assert body["outcome"] == "matched_rank"
        assert body["notes"] == ""
        assert body["reviewer_username"] == "pi"
    finally:
        client.__exit__(None, None, None)


def test_saving_over_a_shared_note_appends_under_the_current_reviewer(
    tmp_path: Path,
) -> None:
    """The append-only guard, exercised across reviewers.

    A SQLite trigger makes any UPDATE or DELETE on `feedback` abort, so a save
    path that edited the prior reviewer's row instead of appending would fail
    here rather than quietly rewriting the pilot record.
    """
    client = _signed_in_client(tmp_path)
    try:
        _add_reviewer(client, "reviewer", "Reviewer")
        _install_append_only_triggers(tmp_path)

        pi_feedback = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 1,
                "correct_dir": "candidate-1",
                "notes": "pi note",
            },
        )
        assert pi_feedback.status_code == 201
        before = pi_feedback.json()

        _sign_in_as(client, "reviewer")
        reviewer_feedback = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 3,
                "correct_dir": "candidate-3",
                "notes": "reviewer disagrees",
            },
        )
        assert reviewer_feedback.status_code == 201
        assert reviewer_feedback.json()["id"] != before["id"]
        assert reviewer_feedback.json()["reviewer"] == "Reviewer"
        assert reviewer_feedback.json()["reviewer_username"] == "reviewer"

        _sign_in_as(client, "pi")
        rows = list(csv.DictReader(client.get("/api/feedback/export").text.splitlines()))
        assert [row["notes"] for row in rows] == ["pi note", "reviewer disagrees"]
        assert [row["reviewer"] for row in rows] == ["PI", "Reviewer"]

        # The earlier reviewer's row is untouched, field for field.
        original = next(row for row in rows if row["id"] == str(before["id"]))
        assert original["notes"] == before["notes"]
        assert original["reviewer"] == before["reviewer"]
        assert original["timestamp"] == before["timestamp"]
        assert original["correct_rank"] == str(before["correct_rank"])
        assert original["correct_dir"] == before["correct_dir"]
    finally:
        client.__exit__(None, None, None)


def test_multi_select_feedback_persists_all_ranks_and_exports_history(
    tmp_path: Path,
) -> None:
    client = _signed_in_client(tmp_path)
    try:
        single = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "correct_rank": 1,
                "correct_dir": "candidate-1",
                "notes": "legacy single rank",
            },
        )
        assert single.status_code == 201

        multi = client.post(
            "/api/feedback",
            json={
                "query_id": 0,
                "model_slug": "bowphs/LaTa",
                "selected_ranks": [3, 7, 10],
                "correct_dir": "client-supplied-value-is-ignored",
                "notes": "multiple plausible matches",
            },
        )
        assert multi.status_code == 201
        assert multi.json()["outcome"] == "matched_rank"
        assert multi.json()["correct_rank"] == 3
        assert multi.json()["correct_dir"] == "candidate-3"
        assert multi.json()["selected_ranks"] == [3, 7, 10]

        latest = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs_LaTa"},
        )
        assert latest.status_code == 200
        assert latest.json()["id"] == multi.json()["id"]
        assert latest.json()["selected_ranks"] == [3, 7, 10]

        rows = list(csv.DictReader(client.get("/api/feedback/export").text.splitlines()))
        assert [row["notes"] for row in rows] == [
            "legacy single rank",
            "multiple plausible matches",
        ]
        assert rows[0]["selected_ranks_json"] == ""
        assert rows[1]["correct_rank"] == "3"
        assert rows[1]["correct_dir"] == "candidate-3"
        assert rows[1]["selected_ranks_json"] == "[3, 7, 10]"
    finally:
        client.__exit__(None, None, None)
