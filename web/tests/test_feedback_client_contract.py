"""The literal payloads the reviewer webapp sends, posted against the real API.

`web/tests/test_feedback_eligibility.py` pins the server's rules from the
server's side. This file pins the *client contract* from issue #157: the exact
JSON bodies `web/frontend/src/contexts/FeedbackContext.tsx` builds, so a change
on either side of the boundary fails here rather than in a reviewer's browser.

Two properties of those bodies are easy to lose and expensive to lose:

* ``correct_dir`` is always ``null``. The server resolves the assignment from
  the rank against its own snapshot, and the client no longer pretends to know
  it. A save must still land in the right directory.
* ``expected_candidate_dirs`` is keyed by rank as a JSON *string*, covers every
  selected rank, and travels in the order the reviewer clicked -- which is the
  order ``selected_ranks`` carries, first click first.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.dependencies import get_store
from web.tests.test_reviewer_dirs import _create_dir, _signed_in
from web.tests.test_top10_feedback import _install_append_only_triggers

MODEL_SLUG = "bowphs_LaTa"
VARIANT = "sif_abtt"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    client = _signed_in(tmp_path, n_queries=8)
    _install_append_only_triggers(tmp_path)
    try:
        yield client
    finally:
        client.__exit__(None, None, None)


def _base(**overrides: object) -> dict:
    """The fields every client save carries, whatever the outcome."""
    payload: dict = {
        "query_id": 1,
        "model_slug": MODEL_SLUG,
        "variant": VARIANT,
        "correct_dir": None,
        "notes": "",
    }
    payload.update(overrides)
    return payload


def _unranked_reviewer_dir(client: TestClient) -> str:
    """A reviewer directory served for query 1, in its own unranked block.

    Since issue #196 it has NO rank, which is what the two tests below are
    about: the client cannot address it by one, and no rank the client does
    send can reach it.
    """
    directory = _create_dir(client, 0, "Unattested homily")
    get_store().predictions[(MODEL_SLUG, VARIANT)][1]["predictions"].pop()
    body = client.get(
        "/api/query/1/predictions", params={"model": MODEL_SLUG, "top_k": 1}
    ).json()
    assert [c["rank"] for c in body["predictions"]] == [1]
    card = next(
        c
        for c in body["reviewer_dir_candidates"]
        if c["dir_id"] == directory["dir_id"]
    )
    assert "rank" not in card
    return directory["dir_id"]


def test_single_choice_payload_saves_and_resolves_its_own_directory(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/feedback",
        json=_base(
            outcome="matched_rank",
            correct_rank=1,
            selected_ranks=[1],
            expected_candidate_dirs={"1": "candidate-a"},
            notes="Shares the incipit.",
        ),
    )

    assert response.status_code == 201, response.text
    saved = response.json()
    # Sent as null; resolved by the server from rank 1 of its own snapshot.
    assert saved["correct_dir"] == "candidate-a"
    assert saved["correct_rank"] == 1
    assert saved["outcome"] == "matched_rank"
    assert saved["variant"] == VARIANT
    assert saved["notes"] == "Shares the incipit."


def test_multi_choice_payload_keeps_click_order_and_one_assignment(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/feedback",
        json=_base(
            outcome="matched_rank",
            # The reviewer clicked rank 2 first, so it is the canonical answer
            # even though rank 1 is numerically lower.
            correct_rank=2,
            selected_ranks=[2, 1],
            expected_candidate_dirs={"2": "candidate-b", "1": "candidate-a"},
        ),
    )

    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["correct_rank"] == 2
    assert saved["correct_dir"] == "candidate-b"
    assert saved["selected_ranks"] == [2, 1]


def test_no_client_rank_can_reach_a_reviewer_directory(client: TestClient) -> None:
    """Issue #196: the anchored ranks 11-15 name nothing at all now.

    The client used to be able to select a reviewer directory by its anchored
    rank, which is how one came to be filed against a query. That route is
    closed: every rank past the model's own is refused before anything is
    written, and the directory's membership is untouched.
    """
    dir_id = _unranked_reviewer_dir(client)
    before = client.get("/api/reviewer_dirs").json()

    for rank in (11, 12, 15):
        response = client.post(
            "/api/feedback",
            json=_base(
                outcome="matched_rank",
                correct_rank=rank,
                selected_ranks=[rank],
                expected_candidate_dirs={str(rank): dir_id},
            ),
        )
        assert response.status_code in (409, 422), (rank, response.text)

    assert client.get("/api/stats").json()["feedback_count"] == 0
    assert client.get("/api/reviewer_dirs").json() == before
    assert dir_id in {record["dir_id"] for record in before}


def test_stale_precondition_from_a_kept_draft_writes_nothing(
    client: TestClient,
) -> None:
    before = client.get("/api/reviewer_dirs").json()

    response = client.post(
        "/api/feedback",
        json=_base(
            outcome="matched_rank",
            correct_rank=1,
            selected_ranks=[1],
            # What the reviewer saw when they pressed the pill, which is no
            # longer what stands there.
            expected_candidate_dirs={"1": "candidate-from-an-older-view"},
        ),
    )

    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "CANDIDATE_IDENTITY_CHANGED"
    assert client.get("/api/stats").json()["feedback_count"] == 0
    assert client.get("/api/reviewer_dirs").json() == before


def test_none_payload_carries_no_identity_precondition(client: TestClient) -> None:
    response = client.post(
        "/api/feedback",
        json=_base(outcome="none_of_top_k", correct_rank=0, notes="New source."),
    )

    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["outcome"] == "none_of_top_k"
    assert saved["correct_rank"] == 0
    assert saved["correct_dir"] is None
    # No key typed: the non-match is recorded and nothing else happens.
    assert saved["ccl_key"] is None
    assert saved["ccl_key_action"] is None
    assert saved["ccl_key_dir"] is None


def test_none_payload_with_a_key_carries_it_through_unchanged(
    client: TestClient,
) -> None:
    """The literal body the blue None action sends (issue #196).

    One optional field beside the outcome. Everything else about the payload is
    what it always was, so a deployment that ignores `ccl_key` still records the
    assessment. Two submissions, because the interesting part is that the second
    JOINS what the first created rather than making a near-duplicate of it.
    """
    dir_id = _unranked_reviewer_dir(client)

    created = client.post(
        "/api/feedback",
        json=_base(
            query_id=2,
            outcome="none_of_top_k",
            correct_rank=0,
            ccl_key="CTOU.567.16",
            notes="Not in the ten; the CCL has this key.",
        ),
    )
    assert created.status_code == 201, created.text
    assert created.json()["ccl_key_action"] == "created_reviewer_dir"
    new_dir = created.json()["ccl_key_dir"]
    assert new_dir != dir_id

    response = client.post(
        "/api/feedback",
        json=_base(
            outcome="none_of_top_k",
            correct_rank=0,
            ccl_key="  ctou.567.16 ",
            notes="Same key, another witness.",
        ),
    )

    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["outcome"] == "none_of_top_k"
    assert saved["correct_rank"] == 0
    assert saved["correct_dir"] is None
    # Stored as typed; matched case- and whitespace-insensitively.
    assert saved["ccl_key"] == "ctou.567.16"
    assert saved["ccl_key_action"] == "joined_reviewer_dir"
    assert saved["ccl_key_dir"] == new_dir
    joined = client.get(f"/api/reviewer_dirs/{new_dir}").json()
    assert sorted(joined["member_query_ids"]) == [1, 2]
    assert joined["label"] == "CTOU.567.16"


def test_skip_payload_is_a_note_and_nothing_else(client: TestClient) -> None:
    response = client.post(
        "/api/feedback",
        json=_base(
            query_id=2,
            outcome="skipped",
            correct_rank=None,
            notes="Ranking would not load; needs another look.",
        ),
    )

    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["outcome"] == "skipped"
    assert saved["correct_rank"] is None
    assert saved["correct_dir"] is None
