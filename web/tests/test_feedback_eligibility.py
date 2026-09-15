"""New feedback writes need an assessment, usable evidence and stable choices."""
from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.dependencies import get_db, get_store
from web.tests.test_prediction_source_status import _signed_in_client
from web.tests.test_reviewer_dirs import _create_dir, _signed_in
from web.tests.test_top10_feedback import _install_append_only_triggers


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    client = _signed_in_client(tmp_path)
    try:
        yield client
    finally:
        client.__exit__(None, None, None)


@pytest.fixture
def ranked_client(tmp_path: Path) -> Iterator[TestClient]:
    client = _signed_in(tmp_path, n_queries=8)
    _install_append_only_triggers(tmp_path)
    try:
        yield client
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize(
    "assessment",
    [
        {},
        {"correct_rank": None},
        {"outcome": None},
        {"outcome": None, "correct_rank": 1},
        {"outcome": "legacy_unresolved"},
        {"outcome": "legacy_unresolved", "correct_rank": 1},
        {"outcome": "skipped"},
        {"outcome": "skipped", "notes": " \t\n "},
        {"selected_ranks": [1.5]},
        {"selected_ranks": [True]},
        {"selected_ranks": [1, 1]},
        {"selected_ranks": []},
        {"correct_rank": True},
    ],
)
def test_new_writes_reject_ambiguous_assessments(
    client: TestClient, assessment: dict,
) -> None:
    response = client.post(
        "/api/feedback",
        json={"query_id": 0, "model_slug": "bowphs_LaTa", **assessment},
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/stats").json()["feedback_count"] == 0


def test_deliberate_skip_trims_its_required_note(client: TestClient) -> None:
    response = client.post(
        "/api/feedback",
        json={
            "query_id": 0,
            "model_slug": "bowphs_LaTa",
            "outcome": "skipped",
            "notes": " \tNeed reference work.\n ",
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["notes"] == "Need reference work."
    assert response.json()["outcome"] == "skipped"
    assert response.json()["correct_rank"] is None


@pytest.mark.parametrize("query_id", [1, 2, 4])
@pytest.mark.parametrize("assessment", [{"correct_rank": 1}, {"correct_rank": 0}])
def test_excluded_or_empty_model_rows_cannot_be_evaluated(
    client: TestClient, query_id: int, assessment: dict,
) -> None:
    response = client.post(
        "/api/feedback",
        json={
            "query_id": query_id,
            "model_slug": "bowphs_LaTa",
            "hasUsableModelRanking": True,
            "correct_dir": "candidate-a",
            **assessment,
        },
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "RANKING_NOT_EVALUABLE"
    assert client.get("/api/stats").json()["feedback_count"] == 0


@pytest.mark.parametrize("query_id", [1, 2, 4])
def test_deliberate_skip_does_not_require_a_ranking(
    client: TestClient, query_id: int,
) -> None:
    response = client.post(
        "/api/feedback",
        json={
            "query_id": query_id,
            "model_slug": "bowphs_LaTa",
            "outcome": "skipped",
            "notes": "Need reference work.",
        },
    )
    assert response.status_code == 201, response.text
    stats = client.get("/api/stats").json()
    assert stats["skipped_count"] == 1
    assert stats["reviewed_count"] == 0


@pytest.mark.parametrize(
    ("identity", "status"),
    [
        ({"query_id": 999}, 404),
        ({"model_slug": "missing"}, 400),
        ({"variant": "raw"}, 400),
        ({"variant": "invented"}, 422),
    ],
)
def test_skip_still_requires_real_query_model_and_variant(
    client: TestClient, identity: dict, status: int,
) -> None:
    response = client.post(
        "/api/feedback",
        json={
            "query_id": 0,
            "model_slug": "bowphs_LaTa",
            "outcome": "skipped",
            "notes": "Need reference work.",
            **identity,
        },
    )
    assert response.status_code == status, response.text
    assert client.get("/api/stats").json()["feedback_count"] == 0


@pytest.mark.parametrize("rank", [0, 1, 11])
def test_reviewer_extras_do_not_replace_usable_model_evidence(
    ranked_client: TestClient, rank: int,
) -> None:
    _create_dir(ranked_client, 0)
    get_store().predictions[("bowphs_LaTa", "sif_abtt")][1]["predictions"] = []
    response = ranked_client.post(
        "/api/feedback",
        json={"query_id": 1, "model_slug": "bowphs_LaTa", "correct_rank": rank},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "RANKING_NOT_EVALUABLE"
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 0
    assert ranked_client.get("/api/reviewer_dirs").json()[0]["member_query_ids"] == [0]


@pytest.mark.parametrize("status", ["excluded_blank_source", "excluded_zero_norm", "excluded_future_guard"])
def test_exclusion_overrides_stale_candidates(client: TestClient, status: str) -> None:
    get_store().predictions[("bowphs_LaTa", "sif_abtt")][0]["status"] = status
    response = client.post(
        "/api/feedback",
        json={"query_id": 0, "model_slug": "bowphs_LaTa", "correct_rank": 1},
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/stats").json()["feedback_count"] == 0


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("unusable", ["blank", "missing", "nonfinite"])
def test_unusable_model_evidence_cannot_be_evaluated(
    client: TestClient, rank: int, unusable: str,
) -> None:
    store = get_store()
    if unusable == "blank":
        store.labelled_texts["candidate-a"]["candidate-a.txt"] = " \n "
    elif unusable == "missing":
        store.labelled_texts.pop("candidate-a")
    else:
        store.predictions[("bowphs_LaTa", "sif_abtt")][0]["predictions"][0]["score"] = float("nan")
    response = client.post(
        "/api/feedback",
        json={"query_id": 0, "model_slug": "bowphs_LaTa", "correct_rank": rank},
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/stats").json()["feedback_count"] == 0


@pytest.mark.parametrize("none_assessment", [{"correct_rank": 0}, {"outcome": "none_of_top_k"}])
@pytest.mark.parametrize("unusable", ["blank", "missing", "nonfinite"])
def test_partial_model_evidence_allows_readable_positive_but_not_none(
    ranked_client: TestClient, none_assessment: dict, unusable: str,
) -> None:
    store = get_store()
    if unusable == "blank":
        store.labelled_texts["candidate-a"]["a.txt"] = " \t "
    elif unusable == "missing":
        store.labelled_texts.pop("candidate-a")
    else:
        store.predictions[("bowphs_LaTa", "sif_abtt")][1]["predictions"][0]["score"] = float("nan")
    rejected = ranked_client.post(
        "/api/feedback",
        json={"query_id": 1, "model_slug": "bowphs_LaTa", "correct_rank": 1},
    )
    assert rejected.status_code == 422, rejected.text
    assert rejected.json()["error"]["code"] == "CANDIDATE_NOT_EVALUABLE"
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 0
    readable = ranked_client.post(
        "/api/feedback",
        json={"query_id": 1, "model_slug": "bowphs_LaTa", "correct_rank": 2},
    )
    assert readable.status_code == 201, readable.text
    assert readable.json()["correct_dir"] == "candidate-b"
    none = ranked_client.post(
        "/api/feedback",
        json={"query_id": 1, "model_slug": "bowphs_LaTa", **none_assessment},
    )
    assert none.status_code == 422, none.text
    assert none.json()["error"]["code"] == "RANKING_NOT_EVALUABLE"
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 1


@pytest.mark.parametrize("reviewer_text", ["Readable reviewer candidate.", " \t\n "])
def test_none_eligibility_ignores_reviewer_extra_text(
    ranked_client: TestClient, reviewer_text: str,
) -> None:
    _create_dir(ranked_client, 0)
    get_store().unlabelled_texts[0] = reviewer_text
    body = ranked_client.get(
        "/api/query/1/predictions", params={"model": "bowphs_LaTa"}
    ).json()
    # The ranked list is the model's ten (two, here); the reviewer directory is
    # beside it without a rank (issue #196).
    assert [candidate["rank"] for candidate in body["predictions"]] == [1, 2]
    assert len(body["reviewer_dir_candidates"]) == 1
    response = ranked_client.post(
        "/api/feedback",
        json={"query_id": 1, "model_slug": "bowphs_LaTa", "outcome": "none_of_top_k"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["outcome"] == "none_of_top_k"
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 1
    assert ranked_client.get("/api/reviewer_dirs").json()[0]["member_query_ids"] == [0]


@pytest.mark.parametrize(
    "expected",
    [
        None,
        {},
        {"1": "candidate-a"},
        {"2": "candidate-b"},
        {"1": "candidate-a", "2": "candidate-b", "3": "extra"},
        {"1": "candidate-a", "2": ""},
        {"1": "candidate-a", "2": " \t"},
        {"1": "candidate-a", "2": 2},
        {"1": "candidate-a", "16": "outside-rank-range"},
    ],
)
def test_precondition_requires_exactly_all_selected_identities(
    ranked_client: TestClient, expected: dict | None,
) -> None:
    response = ranked_client.post(
        "/api/feedback",
        json={
            "query_id": 1,
            "model_slug": "bowphs_LaTa",
            "selected_ranks": [2, 1],
            "expected_candidate_dirs": expected,
        },
    )
    assert response.status_code == 422, response.text
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 0


@pytest.mark.parametrize("outcome", ["none_of_top_k", "skipped"])
def test_precondition_cannot_accompany_a_nonpositive_outcome(
    client: TestClient, outcome: str,
) -> None:
    response = client.post(
        "/api/feedback",
        json={
            "query_id": 0,
            "model_slug": "bowphs_LaTa",
            "outcome": outcome,
            "notes": "Need reference work.",
            "expected_candidate_dirs": {"1": "candidate-a"},
        },
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/stats").json()["feedback_count"] == 0


@pytest.mark.parametrize("rank", range(11, 16))
def test_reviewer_directories_can_no_longer_be_selected_by_rank(
    ranked_client: TestClient, rank: int,
) -> None:
    """Issue #196 closes the rank route into a reviewer directory.

    Five directories are scorable for this query, so under the old anchoring
    every one of ranks 11-15 named one of them and a submission at that rank
    filed the query into it permanently. Now nothing but the model's own ranks
    exists: each of these is refused before anything is written, and no
    membership moves.

    Filing a query into a reviewer directory is done by naming its CCL key, and
    the last two assertions show the same query reaching the same directory that
    way -- the route is replaced, not merely removed.
    """
    by_seed = {
        seed: _create_dir(ranked_client, seed, f"CTOU.567.{seed}")
        for seed in (0, 2, 4, 5, 6)
    }
    get_store().predictions[("bowphs_LaTa", "sif_abtt")][1]["predictions"].pop()
    body = ranked_client.get(
        "/api/query/1/predictions", params={"model": "bowphs_LaTa", "top_k": 1}
    ).json()
    assert [candidate["rank"] for candidate in body["predictions"]] == [1]
    assert len(body["reviewer_dir_candidates"]) == 5

    response = ranked_client.post(
        "/api/feedback",
        json={
            "query_id": 1,
            "model_slug": "bowphs/LaTa",
            "correct_rank": rank,
            "selected_ranks": [rank, 1],
        },
    )
    assert response.status_code == 422, response.text
    assert "No candidate at rank" in response.json()["detail"]
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 0
    for record in ranked_client.get("/api/reviewer_dirs").json():
        assert 1 not in record["member_query_ids"]

    chosen = by_seed[0]["dir_id"]
    saved = ranked_client.post(
        "/api/feedback",
        json={
            "query_id": 1,
            "model_slug": "bowphs/LaTa",
            "outcome": "none_of_top_k",
            "correct_rank": 0,
            "ccl_key": "CTOU.567.0",
            "reviewer": "Untrusted attribution",
        },
    )
    assert saved.status_code == 201, saved.text
    assert saved.json()["ccl_key_dir"] == chosen
    assert saved.json()["reviewer"] == "PI"
    for record in ranked_client.get("/api/reviewer_dirs").json():
        assert (1 in record["member_query_ids"]) == (record["dir_id"] == chosen)


@pytest.mark.parametrize("changed_rank", [1, 2])
def test_changed_identity_at_any_selected_rank_writes_nothing(
    ranked_client: TestClient, changed_rank: int,
) -> None:
    expected = {"1": "candidate-a", "2": "candidate-b"}
    expected[str(changed_rank)] = "previous-directory"
    response = ranked_client.post(
        "/api/feedback",
        json={
            "query_id": 1,
            "model_slug": "bowphs_LaTa",
            "selected_ranks": [2, 1],
            "expected_candidate_dirs": expected,
        },
    )
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "CANDIDATE_IDENTITY_CHANGED"
    assert f"rank {changed_rank}" in error["message"]
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 0


@pytest.mark.parametrize("another_directory", [False, True])
def test_a_repeated_key_join_cannot_reassign_the_next_directory(
    ranked_client: TestClient, another_directory: bool,
) -> None:
    """The retry hazard the anchored ranks created is gone with them.

    A reviewer who retried a save at rank 11 used to hit a *different*
    directory, because their own first save had removed the original from the
    list and shifted everything up. A key names one directory whatever else has
    happened, so the replay reaches the same one, adds no membership and simply
    appends a second assessment to an append-only log.
    """
    first = _create_dir(ranked_client, 0, "CTOU.567.0")["dir_id"]
    if another_directory:
        _create_dir(ranked_client, 2, "CTOU.567.2")
    payload = {
        "query_id": 1,
        "model_slug": "bowphs_LaTa",
        "outcome": "none_of_top_k",
        "correct_rank": 0,
        "ccl_key": "CTOU.567.0",
    }
    assert ranked_client.post("/api/feedback", json=payload).status_code == 201
    before = ranked_client.get("/api/reviewer_dirs").json()
    retry = ranked_client.post("/api/feedback", json=payload)
    assert retry.status_code == 201, retry.text
    assert retry.json()["ccl_key_dir"] == first
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 2
    assert ranked_client.get("/api/reviewer_dirs").json() == before


def test_multi_select_resolves_one_snapshot_of_the_ranking(
    ranked_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every selected rank is resolved against ONE read of the candidate list.

    A reviewer directory can no longer be selected by rank, but the snapshot
    rule it was built for still holds and is still worth pinning: the router
    reads the candidates once per save, not once per rank.
    """
    _create_dir(ranked_client, 0, "CTOU.567.0")
    db = get_db()
    original_list = db.list_reviewer_dirs
    reads = 0

    async def counted() -> list[dict]:
        nonlocal reads
        reads += 1
        return await original_list()

    monkeypatch.setattr(db, "list_reviewer_dirs", counted)
    response = ranked_client.post(
        "/api/feedback",
        json={
            "query_id": 1,
            "model_slug": "bowphs_LaTa",
            "selected_ranks": [2, 1],
            "expected_candidate_dirs": {"2": "candidate-b", "1": "candidate-a"},
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["correct_dir"] == "candidate-b"
    assert response.json()["selected_ranks"] == [2, 1]
    assert reads == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_missing_query_row_is_not_evidence(client: TestClient, rank: int) -> None:
    get_store().predictions[("bowphs_LaTa", "sif_abtt")].pop(0)
    rejected = client.post(
        "/api/feedback",
        json={"query_id": 0, "model_slug": "bowphs_LaTa", "correct_rank": rank},
    )
    assert rejected.status_code == 404, rejected.text
    assert client.get("/api/stats").json()["feedback_count"] == 0
    skipped = client.post(
        "/api/feedback",
        json={
            "query_id": 0, "model_slug": "bowphs_LaTa", "outcome": "skipped",
            "notes": "Ranking unavailable.",
        },
    )
    assert skipped.status_code == 201, skipped.text


def test_a_blank_reviewer_directory_is_not_reachable_by_rank(
    ranked_client: TestClient,
) -> None:
    """It was rank 11 and unreadable; now it is not a rank at all."""
    _create_dir(ranked_client, 0)
    get_store().unlabelled_texts[0] = " \t\n "
    rejected = ranked_client.post(
        "/api/feedback",
        json={"query_id": 1, "model_slug": "bowphs_LaTa", "selected_ranks": [1, 11]},
    )
    assert rejected.status_code == 422, rejected.text
    assert "No candidate at rank 11" in rejected.json()["detail"]
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 0
    assert ranked_client.get("/api/reviewer_dirs").json()[0]["member_query_ids"] == [0]


@pytest.mark.parametrize("query_id", [0, 3])
def test_usable_ranking_accepts_none_with_or_without_source_status(
    client: TestClient, query_id: int,
) -> None:
    response = client.post(
        "/api/feedback",
        json={"query_id": query_id, "model_slug": "bowphs_LaTa", "outcome": "none_of_top_k"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["correct_rank"] == 0
    assert response.json()["outcome"] == "none_of_top_k"


def test_legacy_unresolved_history_remains_readable(client: TestClient) -> None:
    user = client.get("/api/auth/me").json()
    assert client.portal is not None
    historical = client.portal.call(
        partial(
            get_db().insert,
            query_id=0,
            model_slug="bowphs_LaTa",
            variant="sif_abtt",
            outcome="legacy_unresolved",
            correct_rank=None,
            correct_dir=None,
            notes="Historical fixture note.",
            reviewer=user["display_name"],
            reviewer_account_id=user["id"],
        )
    )
    latest = client.get(
        "/api/feedback/latest", params={"query_id": 0, "model": "bowphs_LaTa"}
    )
    assert latest.status_code == 200
    assert latest.json()["id"] == historical["id"]
    assert latest.json()["outcome"] == "legacy_unresolved"
    assert latest.json()["correct_rank"] is None
    assert latest.json()["selected_ranks"] is None
    assert "legacy_unresolved" in client.get("/api/feedback/export").text
