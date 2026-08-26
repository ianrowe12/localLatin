"""Reviewer-created directories: creation, permissions, merge-scoring, lifecycle.

The fixture builds a four-query corpus with a hand-written q-q matrix, so every
expected score in this file is a literal the test states rather than something
read back out of the code under test:

    q0  q1  q2  q3
q0 1.00 .80 .30 .00
q1  .80 1.0 .20 .00
q2  .30 .20 1.0 .00
q3  .00 .00 .00 1.0   <- guard-excluded (blank source)

With a directory seeded by q0: q1 scores 0.80 (matched, over the 0.5 band),
q2 scores 0.30 (below the band), q3 is excluded and is never scored at all.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from web.app import create_app
from web.bands import REVIEWER_DIR_MATCH_BAND

QUERIES = ["query-0.txt", "query-1.txt", "query-2.txt", "query-3.txt"]

# Row/column i corresponds to file_id i. Symmetric, diagonal 1.
QQ = np.array(
    [
        [1.00, 0.80, 0.30, 0.00],
        [0.80, 1.00, 0.20, 0.00],
        [0.30, 0.20, 1.00, 0.00],
        [0.00, 0.00, 0.00, 1.00],
    ],
    dtype=np.float32,
)

# q3 stands in for a blank-source query: present in the corpus, never scorable.
EXCLUDED = np.array([False, False, False, True])

MODEL_SLUG = "bowphs_LaTa"

# The matrix ships as float16, so a score comes back as the nearest half-float
# to the fixture's value. Spelling those out keeps the expectations exact
# instead of hiding a real change behind a tolerance.
F16_080 = 0.7998046875
F16_030 = 0.300048828125


def _write_fixture_data(root: Path, *, with_matrix: bool = True) -> Path:
    unlabelled = root / "data" / "canon_unlabelled"
    labelled = root / "data" / "canon_labelled"
    predictions = root / "runs" / "active" / "resubmit" / "unlabelled"
    feedback = root / "runs" / "active" / "resubmit" / "webapp"

    unlabelled.mkdir(parents=True)
    predictions.mkdir(parents=True)
    feedback.mkdir(parents=True)
    for name in QUERIES:
        (unlabelled / name).write_text(f"text of {name}", encoding="utf-8")

    (labelled / "candidate-a").mkdir(parents=True)
    (labelled / "candidate-a" / "a.txt").write_text("candidate a", encoding="utf-8")
    (labelled / "candidate-b").mkdir(parents=True)
    (labelled / "candidate-b" / "b.txt").write_text("candidate b", encoding="utf-8")

    fieldnames = [
        "file_id",
        "filename",
        "model",
        "variant",
        "layer",
        "pooling",
        "rank1_dir",
        "rank1_score",
        "rank2_dir",
        "rank2_score",
    ]
    with (predictions / "unlabelled_predictions_sif_abtt.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for file_id, name in enumerate(QUERIES):
            writer.writerow(
                {
                    "file_id": file_id,
                    "filename": name,
                    "model": "bowphs/LaTa",
                    "variant": "sif_abtt",
                    "layer": 12,
                    "pooling": "sif",
                    "rank1_dir": "candidate-a",
                    "rank1_score": 0.41,
                    "rank2_dir": "candidate-b",
                    "rank2_score": 0.22,
                }
            )

    if with_matrix:
        np.savez(
            predictions / f"qq_sim_{MODEL_SLUG}.npz",
            sim=QQ.astype(np.float16),
            file_ids=np.arange(len(QUERIES), dtype=np.int32),
            excluded=EXCLUDED,
            meta=np.array(json.dumps({"model": "bowphs/LaTa", "layer": 12, "D": 3})),
        )

    config_path = root / "config.yaml"
    config_path.write_text(
        f"""
paths:
  data_root: "{root}"
  canon_unlabelled: "data/canon_unlabelled"
  canon_labelled: "data/canon_labelled"
  predictions_variant_pattern: "runs/active/resubmit/unlabelled/unlabelled_predictions_{{variant}}.csv"
  qq_matrix_pattern: "runs/active/resubmit/unlabelled/qq_sim_{{model_slug}}.npz"
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


def _client(tmp_path: Path, *, with_matrix: bool = True) -> TestClient:
    config_path = _write_fixture_data(tmp_path, with_matrix=with_matrix)
    client = TestClient(create_app(str(config_path)))
    client.__enter__()
    return client


def _register(client: TestClient, username: str, display_name: str) -> dict:
    """Register an account. The very first one is auto-approved as pi_admin."""
    response = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "display_name": display_name,
            "password": "correct horse battery staple",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _signed_in(tmp_path: Path, *, with_matrix: bool = True) -> TestClient:
    client = _client(tmp_path, with_matrix=with_matrix)
    _register(client, "pi", "PI")
    return client


def _create_dir(client: TestClient, query_file_id: int, label: str | None = None) -> dict:
    body: dict = {"query_file_id": query_file_id}
    if label is not None:
        body["label"] = label
    response = client.post("/api/reviewer_dirs", json=body)
    assert response.status_code == 201, response.text
    return response.json()


# --- creation --------------------------------------------------------------


def test_creation_returns_the_contract_shape(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        body = _create_dir(client, 2, "Unattested homily")
        assert body["dir_id"] == "reviewer-dir-1"
        assert body["label"] == "Unattested homily"
        assert body["seed_query_id"] == 2
        assert body["member_query_ids"] == [2]
        assert body["created_by"] == "PI"
        # q2's best neighbour is q0 at 0.30, below the 0.5 band.
        assert body["status"] == "awaiting_match"
        assert body["best_match_score"] == F16_030
    finally:
        client.__exit__(None, None, None)


def test_creation_defaults_the_label_to_the_seed_filename(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        body = _create_dir(client, 2)
        assert body["label"] == "New directory from query-2"
    finally:
        client.__exit__(None, None, None)


def test_creation_reports_matched_when_the_seed_already_has_a_neighbour(
    tmp_path: Path,
) -> None:
    """The 201 carries the computed status, not a hardcoded 'awaiting_match'.

    q0's best neighbour is q1 at 0.80, over the band, so the directory is
    matched from the moment it exists -- which is the discovery the feature is
    for, and reporting 'awaiting_match' there would be a lie.
    """
    client = _signed_in(tmp_path)
    try:
        body = _create_dir(client, 0)
        assert body["status"] == "matched"
        assert body["best_match_score"] > REVIEWER_DIR_MATCH_BAND
    finally:
        client.__exit__(None, None, None)


def test_creation_rejects_an_unknown_query(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        response = client.post("/api/reviewer_dirs", json={"query_file_id": 999})
        assert response.status_code == 404
    finally:
        client.__exit__(None, None, None)


# --- permissions -----------------------------------------------------------


def test_creation_requires_authentication(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "PI")
        client.post("/api/auth/signout")
        response = client.post("/api/reviewer_dirs", json={"query_file_id": 0})
        assert response.status_code == 401
        assert client.get("/api/reviewer_dirs").status_code == 401
    finally:
        client.__exit__(None, None, None)


def test_an_approved_reviewer_may_create_a_directory(tmp_path: Path) -> None:
    """Not a PI-only route: reviewers are exactly who this feature is for."""
    client = _client(tmp_path)
    try:
        _register(client, "pi", "PI")  # first account: pi_admin, approved
        registration = client.post(
            "/api/auth/register",
            json={
                "username": "scholar",
                "display_name": "Scholar",
                "password": "correct horse battery staple",
            },
        )
        assert registration.json()["status"] == "pending_approval"
        account_id = registration.json()["account"]["id"]
        assert client.post(f"/api/auth/accounts/{account_id}/approve").status_code == 200

        client.post("/api/auth/signout")
        signin = client.post(
            "/api/auth/signin",
            json={
                "username": "scholar",
                "password": "correct horse battery staple",
            },
        )
        assert signin.status_code == 200
        assert signin.json()["role"] == "reviewer"

        body = _create_dir(client, 2, "Reviewer's directory")
        assert body["created_by"] == "Scholar"
    finally:
        client.__exit__(None, None, None)


def test_a_pending_reviewer_may_not_create_a_directory(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        _register(client, "pi", "PI")
        client.post("/api/auth/signout")
        _register(client, "pending", "Pending")  # stays pending, no session
        response = client.post("/api/reviewer_dirs", json={"query_file_id": 0})
        assert response.status_code == 401
    finally:
        client.__exit__(None, None, None)


# --- merge scoring ---------------------------------------------------------


def test_reviewer_dirs_merge_into_predictions_as_extra_candidates(
    tmp_path: Path,
) -> None:
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 0, "Seeded by q0")
        body = client.get(
            "/api/query/1/predictions", params={"model": "bowphs/LaTa"}
        ).json()

        model_cards = [p for p in body["predictions"] if p["source"] == "model"]
        reviewer_cards = [p for p in body["predictions"] if p["source"] == "reviewer"]

        # The model's own ranks are untouched by the merge.
        assert [p["rank"] for p in model_cards] == [1, 2]
        assert [p["dir_name"] for p in model_cards] == ["candidate-a", "candidate-b"]

        assert len(reviewer_cards) == 1
        card = reviewer_cards[0]
        assert card["rank"] == 3  # appended after the model's candidates
        assert card["dir_name"] == created["dir_id"]
        assert card["label"] == "Seeded by q0"
        assert card["seed_query_id"] == 0
        assert card["score"] == F16_080  # QQ[1, 0] in float16
        assert card["dir_files"] == ["query-0.txt"]
        assert card["candidate_files"][0]["text"] == "text of query-0.txt"
    finally:
        client.__exit__(None, None, None)


def test_merge_score_is_the_max_over_member_documents(tmp_path: Path) -> None:
    """q2 scores 0.30 against {q0} and 0.30 against {q0, q1}: max, not mean."""
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 0)
        before = _reviewer_card(client, query_id=2, dir_id=created["dir_id"])
        assert before["score"] == F16_030  # QQ[2, 0] in float16

        # Confirming the directory from q1 appends q1 as a member.
        _submit_match(client, query_id=1, rank=before_rank(client, 1, created["dir_id"]))
        after = _reviewer_card(client, query_id=2, dir_id=created["dir_id"])
        # max(QQ[2,0], QQ[2,1]) = max(0.30, 0.20) = 0.30, unchanged. A mean
        # would have dropped to 0.25.
        assert after["score"] == before["score"]
        assert sorted(
            client.get(f"/api/reviewer_dirs/{created['dir_id']}").json()[
                "member_query_ids"
            ]
        ) == [0, 1]
    finally:
        client.__exit__(None, None, None)


def test_a_directory_is_not_offered_to_its_own_members(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 0)
        body = client.get(
            "/api/query/0/predictions", params={"model": "bowphs/LaTa"}
        ).json()
        assert [p for p in body["predictions"] if p["source"] == "reviewer"] == []
        # ... but the seed still learns about the directory it created.
        assert [d["dir_id"] for d in body["seeded_dirs"]] == [created["dir_id"]]
    finally:
        client.__exit__(None, None, None)


def test_guard_excluded_queries_are_never_scored(tmp_path: Path) -> None:
    """q3 is a blank-source query: it neither matches nor is matched.

    Without this, ABTT's mean-centring makes every empty document identical and
    two blank files score a spurious cosine of 1.0 (issue #66).
    """
    client = _signed_in(tmp_path)
    try:
        _create_dir(client, 0)
        body = client.get(
            "/api/query/3/predictions", params={"model": "bowphs/LaTa"}
        ).json()
        assert [p for p in body["predictions"] if p["source"] == "reviewer"] == []

        seeded = _create_dir(client, 3)
        assert seeded["status"] == "awaiting_match"
        assert seeded["best_match_score"] == 0.0
    finally:
        client.__exit__(None, None, None)


def test_a_model_without_a_matrix_serves_no_reviewer_candidates(
    tmp_path: Path,
) -> None:
    client = _signed_in(tmp_path, with_matrix=False)
    try:
        created = _create_dir(client, 0)
        assert created["status"] == "awaiting_match"
        assert created["best_match_score"] is None

        body = client.get(
            "/api/query/1/predictions", params={"model": "bowphs/LaTa"}
        ).json()
        assert [p for p in body["predictions"] if p["source"] == "reviewer"] == []

        model = client.get("/api/models").json()[0]
        assert model["supports_reviewer_dirs"] is False
    finally:
        client.__exit__(None, None, None)


def test_matrices_load_lazily(tmp_path: Path) -> None:
    from web.dependencies import get_store

    client = _signed_in(tmp_path)
    try:
        store = get_store()
        assert store.qq_paths and store.qq_matrices == {}
        _create_dir(client, 0)
        assert MODEL_SLUG in store.qq_matrices
    finally:
        client.__exit__(None, None, None)


# --- badge lifecycle -------------------------------------------------------


def test_badge_lifecycle_awaiting_then_matched(tmp_path: Path) -> None:
    """A directory seeded by q2 has no neighbour above the band until q1 joins.

    q2's best neighbour is q0 at 0.30, so it starts awaiting. Adding q0 as a
    member makes q1 (0.80 against q0) an external match, and the seed's badge
    flips.
    """
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 2)
        assert created["status"] == "awaiting_match"

        seed_view = client.get(
            "/api/query/2/predictions", params={"model": "bowphs/LaTa"}
        ).json()
        assert seed_view["seeded_dirs"][0]["status"] == "awaiting_match"

        # q0 confirms the directory, joining it.
        rank = before_rank(client, 0, created["dir_id"])
        _submit_match(client, query_id=0, rank=rank)

        after = client.get(f"/api/reviewer_dirs/{created['dir_id']}").json()
        assert after["status"] == "matched"
        assert after["best_match_score"] >= REVIEWER_DIR_MATCH_BAND

        seed_view = client.get(
            "/api/query/2/predictions", params={"model": "bowphs/LaTa"}
        ).json()
        assert seed_view["seeded_dirs"][0]["status"] == "matched"
    finally:
        client.__exit__(None, None, None)


def test_band_is_served_to_the_frontend(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        model = client.get("/api/models").json()[0]
        assert model["confidence_bands"] == {"no_match": 0.5, "verify": 0.7}
        assert model["supports_reviewer_dirs"] is True
        assert model["confidence_bands"]["no_match"] == REVIEWER_DIR_MATCH_BAND
    finally:
        client.__exit__(None, None, None)


# --- feedback integration --------------------------------------------------


def test_feedback_on_a_reviewer_dir_records_normally(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 0)
        rank = before_rank(client, 1, created["dir_id"])
        entry = _submit_match(client, query_id=1, rank=rank)
        assert entry["outcome"] == "matched_rank"
        assert entry["variant"] == "sif_abtt"
        assert entry["correct_dir"] == created["dir_id"]
        assert entry["correct_rank"] == rank
    finally:
        client.__exit__(None, None, None)


def test_joining_a_directory_is_idempotent(tmp_path: Path) -> None:
    """Re-submitting the same answer appends a feedback row and nothing else.

    Note the directory stops being offered to q1 once q1 joins it, so the
    second submission is posted by dir_id rather than re-read from the list --
    which is exactly the shape of a client replaying a stale form.
    """
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 0)
        rank = before_rank(client, 1, created["dir_id"])
        _submit_match(client, query_id=1, rank=rank)
        _post_feedback(
            client, query_id=1, rank=rank, dir_name=created["dir_id"]
        )
        members = client.get(f"/api/reviewer_dirs/{created['dir_id']}").json()
        assert sorted(members["member_query_ids"]) == [0, 1]
    finally:
        client.__exit__(None, None, None)


# --- migration additivity --------------------------------------------------


def _legacy_db(path: Path) -> None:
    """A pre-#95 database: v1 feedback schema, one row, no reviewer tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
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
        );
        INSERT INTO feedback (query_id, timestamp, model_slug, correct_rank, correct_dir, notes, reviewer)
        VALUES (7, '2026-01-01 00:00:00', 'bowphs_LaTa', 3, 'candidate-a', 'legacy note', 'Old Reviewer');
        """
    )
    connection.commit()
    connection.close()


def test_migration_is_additive_on_a_legacy_database(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    db_path = tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
    db_path.unlink(missing_ok=True)
    _legacy_db(db_path)

    client = TestClient(create_app(str(config_path)))
    client.__enter__()
    try:
        _register(client, "pi", "PI")
        created = _create_dir(client, 0, "New from a legacy DB")
        assert created["dir_id"] == "reviewer-dir-1"

        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            assert {"reviewer_dirs", "reviewer_dir_members"} <= tables

            # The pre-existing feedback row is byte-identical: the migration
            # added tables, it did not touch the append-only log.
            legacy = connection.execute("SELECT * FROM feedback WHERE id = 1").fetchone()
            assert legacy["query_id"] == 7
            assert legacy["timestamp"] == "2026-01-01 00:00:00"
            assert legacy["correct_rank"] == 3
            assert legacy["correct_dir"] == "candidate-a"
            assert legacy["notes"] == "legacy note"
            assert legacy["reviewer"] == "Old Reviewer"
            # ... and the columns earlier migrations added are still backfilled
            # the way they were, not restamped by this one.
            assert legacy["outcome"] == "matched_rank"
            assert legacy["variant"] is None
            assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
        finally:
            connection.close()
    finally:
        client.__exit__(None, None, None)


def test_reopening_a_migrated_database_changes_nothing(tmp_path: Path) -> None:
    config_path = _write_fixture_data(tmp_path)
    db_path = tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
    db_path.unlink(missing_ok=True)
    _legacy_db(db_path)

    client = TestClient(create_app(str(config_path)))
    client.__enter__()
    _register(client, "pi", "PI")
    created = _create_dir(client, 0)
    client.__exit__(None, None, None)

    # Second boot over the same file: migration must be a no-op.
    client = TestClient(create_app(str(config_path)))
    client.__enter__()
    try:
        signin = client.post(
            "/api/auth/signin",
            json={"username": "pi", "password": "correct horse battery staple"},
        )
        assert signin.status_code == 200
        again = client.get(f"/api/reviewer_dirs/{created['dir_id']}").json()
        assert again["dir_id"] == created["dir_id"]
        assert again["member_query_ids"] == [0]
        assert client.get("/api/reviewer_dirs").json().__len__() == 1
    finally:
        client.__exit__(None, None, None)


# --- helpers ---------------------------------------------------------------


def _reviewer_card(client: TestClient, *, query_id: int, dir_id: str) -> dict:
    body = client.get(
        f"/api/query/{query_id}/predictions", params={"model": "bowphs/LaTa"}
    ).json()
    card = next(p for p in body["predictions"] if p["dir_name"] == dir_id)
    assert card["source"] == "reviewer"
    return card


def before_rank(client: TestClient, query_id: int, dir_id: str) -> int:
    return _reviewer_card(client, query_id=query_id, dir_id=dir_id)["rank"]


def _post_feedback(client: TestClient, *, query_id: int, rank: int, dir_name: str) -> dict:
    response = client.post(
        "/api/feedback",
        json={
            "query_id": query_id,
            "model_slug": "bowphs/LaTa",
            "outcome": "matched_rank",
            "correct_rank": rank,
            "correct_dir": dir_name,
            "notes": "",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _submit_match(client: TestClient, *, query_id: int, rank: int) -> dict:
    body = client.get(
        f"/api/query/{query_id}/predictions", params={"model": "bowphs/LaTa"}
    ).json()
    card = next(p for p in body["predictions"] if p["rank"] == rank)
    return _post_feedback(
        client, query_id=query_id, rank=rank, dir_name=card["dir_name"]
    )
