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


def _sparse_reviewer_rank(client: TestClient) -> tuple[int, str]:
    """A reviewer directory anchored above the model candidates, as served."""
    directory = _create_dir(client, 0, "Unattested homily")
    get_store().predictions[(MODEL_SLUG, VARIANT)][1]["predictions"].pop()
    predictions = client.get(
        "/api/query/1/predictions", params={"model": MODEL_SLUG, "top_k": 1}
    ).json()["predictions"]
    reviewer = next(
        candidate
        for candidate in predictions
        if candidate["dir_name"] == directory["dir_id"]
    )
    assert reviewer["rank"] >= 11
    return reviewer["rank"], directory["dir_id"]


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
    rank, dir_id = _sparse_reviewer_rank(client)

    response = client.post(
        "/api/feedback",
        json=_base(
            outcome="matched_rank",
            # The reviewer clicked the anchored directory first, so it is the
            # canonical answer even though rank 1 is numerically lower.
            correct_rank=rank,
            selected_ranks=[rank, 1],
            expected_candidate_dirs={str(rank): dir_id, "1": "candidate-a"},
        ),
    )

    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["correct_rank"] == rank
    assert saved["correct_dir"] == dir_id
    assert saved["selected_ranks"] == [rank, 1]
    # Only the canonical choice gains membership; the second is recorded, not
    # assigned.
    for record in client.get("/api/reviewer_dirs").json():
        assert (1 in record["member_query_ids"]) == (record["dir_id"] == dir_id)


def test_stale_precondition_from_a_kept_draft_writes_nothing(
    client: TestClient,
) -> None:
    rank, dir_id = _sparse_reviewer_rank(client)
    before = client.get("/api/reviewer_dirs").json()

    response = client.post(
        "/api/feedback",
        json=_base(
            outcome="matched_rank",
            correct_rank=rank,
            selected_ranks=[rank],
            # What the reviewer saw when they pressed the pill, which is no
            # longer what stands there.
            expected_candidate_dirs={str(rank): "reviewer-dir-from-an-older-view"},
        ),
    )

    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "CANDIDATE_IDENTITY_CHANGED"
    assert client.get("/api/stats").json()["feedback_count"] == 0
    assert client.get("/api/reviewer_dirs").json() == before
    assert dir_id in {record["dir_id"] for record in before}


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
