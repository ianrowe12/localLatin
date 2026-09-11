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
    predictions = ranked_client.get(
        "/api/query/1/predictions", params={"model": "bowphs_LaTa"}
    ).json()["predictions"]
    assert [candidate["rank"] for candidate in predictions] == [1, 2, 11]
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
def test_sparse_reviewer_ranks_keep_server_identity_and_first_selection(
    ranked_client: TestClient, rank: int,
) -> None:
    by_seed = {seed: _create_dir(ranked_client, seed) for seed in (0, 2, 4, 5, 6)}
    get_store().predictions[("bowphs_LaTa", "sif_abtt")][1]["predictions"].pop()
    predictions = ranked_client.get(
        "/api/query/1/predictions", params={"model": "bowphs_LaTa", "top_k": 1}
    ).json()["predictions"]
    assert [candidate["rank"] for candidate in predictions] == [1, 11, 12, 13, 14, 15]
    # These positions follow the fixture's literal q-q scores, not list length.
    seed_for_rank = {11: 0, 12: 4, 13: 5, 14: 6, 15: 2}
    chosen = by_seed[seed_for_rank[rank]]["dir_id"]
    other_rank = 15 if rank != 15 else 11
    other = by_seed[seed_for_rank[other_rank]]["dir_id"]
    selected = [rank, 1, other_rank]
    response = ranked_client.post(
        "/api/feedback",
        json={
            "query_id": 1,
            "model_slug": "bowphs/LaTa",
            "correct_rank": 1,
            "correct_dir": other,
            "selected_ranks": selected,
            "expected_candidate_dirs": {
                str(rank): chosen, "1": "candidate-a", str(other_rank): other,
            },
            "reviewer": "Untrusted attribution",
        },
    )
    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["correct_rank"] == rank
    assert saved["correct_dir"] == chosen
    assert saved["selected_ranks"] == selected
    assert saved["reviewer"] == "PI"
    latest = ranked_client.get(
        "/api/feedback/latest", params={"query_id": 1, "model": "bowphs_LaTa"}
    ).json()
    assert latest["selected_ranks"] == selected
    for record in ranked_client.get("/api/reviewer_dirs").json():
        assert (1 in record["member_query_ids"]) == (record["dir_id"] == chosen)


@pytest.mark.parametrize("changed_rank", [1, 11, 12])
def test_changed_identity_at_any_selected_rank_writes_nothing(
    ranked_client: TestClient, changed_rank: int,
) -> None:
    first = _create_dir(ranked_client, 0)["dir_id"]
    second = _create_dir(ranked_client, 2)["dir_id"]
    expected = {"11": first, "1": "candidate-a", "12": second}
    expected[str(changed_rank)] = "previous-directory"
    before = ranked_client.get("/api/reviewer_dirs").json()
    response = ranked_client.post(
        "/api/feedback",
        json={
            "query_id": 1,
            "model_slug": "bowphs_LaTa",
            "selected_ranks": [11, 1, 12],
            "expected_candidate_dirs": expected,
        },
    )
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "CANDIDATE_IDENTITY_CHANGED"
    assert f"rank {changed_rank}" in error["message"]
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 0
    assert ranked_client.get("/api/reviewer_dirs").json() == before


@pytest.mark.parametrize("another_directory", [False, True])
def test_retry_after_membership_changes_cannot_assign_the_next_directory(
    ranked_client: TestClient, another_directory: bool,
) -> None:
    first = _create_dir(ranked_client, 0)["dir_id"]
    if another_directory:
        _create_dir(ranked_client, 2)
    payload = {
        "query_id": 1,
        "model_slug": "bowphs_LaTa",
        "correct_rank": 11,
        "expected_candidate_dirs": {"11": first},
    }
    assert ranked_client.post("/api/feedback", json=payload).status_code == 201
    before = ranked_client.get("/api/reviewer_dirs").json()
    retry = ranked_client.post("/api/feedback", json=payload)
    assert retry.status_code == 409, retry.text
    assert retry.json()["error"]["code"] == "CANDIDATE_IDENTITY_CHANGED"
    assert ranked_client.get("/api/stats").json()["feedback_count"] == 1
    assert ranked_client.get("/api/reviewer_dirs").json() == before


def test_multi_select_resolves_one_snapshot_during_membership_change(
    ranked_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _create_dir(ranked_client, 0)["dir_id"]
    second = _create_dir(ranked_client, 2)["dir_id"]
    db = get_db()
    original_list = db.list_reviewer_dirs
    reads = 0

    async def list_then_confirm_elsewhere() -> list[dict]:
        nonlocal reads
        records = await original_list()
        reads += 1
        if reads == 1:
            # Another confirmation lands after the snapshot was read. It
            # removes the first directory from subsequent candidate rankings.
            await db.add_reviewer_dir_member(
                dir_id=first, query_id=1, added_by="Concurrent fixture reviewer",
                added_by_account_id=None,
            )
        return records

    monkeypatch.setattr(db, "list_reviewer_dirs", list_then_confirm_elsewhere)
    response = ranked_client.post(
        "/api/feedback",
        json={
            "query_id": 1,
            "model_slug": "bowphs_LaTa",
            "selected_ranks": [12, 11],
            "expected_candidate_dirs": {"12": second, "11": first},
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["correct_dir"] == second
    assert response.json()["selected_ranks"] == [12, 11]
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


def test_blank_reviewer_candidate_cannot_be_selected(ranked_client: TestClient) -> None:
    _create_dir(ranked_client, 0)
    get_store().unlabelled_texts[0] = " \t\n "
    rejected = ranked_client.post(
        "/api/feedback",
        json={"query_id": 1, "model_slug": "bowphs_LaTa", "selected_ranks": [1, 11]},
    )
    assert rejected.status_code == 422, rejected.text
    assert rejected.json()["error"]["code"] == "CANDIDATE_NOT_EVALUABLE"
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
