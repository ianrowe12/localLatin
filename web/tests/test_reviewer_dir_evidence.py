"""Supporting witnesses for reviewer-directory max similarities."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from web.services.data_store import DataStore
from web.services.qq_matrix import QQMatrix
from web.services.reviewer_dirs import candidates_for_query, packet_dirs_for_query
from web.tests.test_reviewer_dirs import (
    F16_080,
    _build_matrix,
    _create_dir,
    _reviewer_card,
    _signed_in,
    _submit_match,
    before_rank,
)


def test_second_member_supports_the_score_without_reordering_files(
    tmp_path: Path,
) -> None:
    client = _signed_in(tmp_path)
    try:
        created = _create_dir(client, 2)
        dir_id = created["dir_id"]
        seed_only = _reviewer_card(client, query_id=1, dir_id=dir_id)
        assert seed_only["score"] == 0.199951171875
        assert seed_only["supporting_member"] == {
            "query_id": 2,
            "filename": "query-2.txt",
            "score": 0.199951171875,
        }
        _submit_match(client, query_id=0, rank=before_rank(client, 0, dir_id))
        before = client.get(f"/api/reviewer_dirs/{dir_id}").json()
        assert before["member_query_ids"] == [2, 0]

        card = _reviewer_card(client, query_id=1, dir_id=dir_id)
        assert card["score"] == F16_080
        assert card["rank"] == 11
        assert card["supporting_member"] == {
            "query_id": 0,
            "filename": "query-0.txt",
            "score": F16_080,
        }
        assert card["preview_text"] == "text of query-0.txt"
        assert card["dir_files"] == ["query-2.txt", "query-0.txt"]
        assert card["candidate_files"] == [
            {"filename": "query-2.txt", "text": "text of query-2.txt"},
            {"filename": "query-0.txt", "text": "text of query-0.txt"},
        ]
        for top_k in (1, 2, 10):
            response = client.get(
                "/api/query/1/predictions",
                params={"model": "bowphs/LaTa", "top_k": top_k},
            )
            assert response.status_code == 200, response.text
            predictions = response.json()["predictions"]
            assert predictions[-1] == card
            assert [
                (p["rank"], p["dir_name"], p["score"], p["supporting_member"])
                for p in predictions[:-1]
            ] == [(1, "candidate-a", 0.41, None), (2, "candidate-b", 0.22, None)][:top_k]

        files = client.get(
            "/api/query/1/predictions/11/candidates",
            params={"model": "bowphs/LaTa"},
        )
        assert files.status_code == 200, files.text
        assert files.json() == card["candidate_files"]
        assert client.get(f"/api/reviewer_dirs/{dir_id}").json() == before

        saved = client.post(
            "/api/feedback",
            json={
                "query_id": 1,
                "model_slug": "bowphs/LaTa",
                "outcome": "matched_rank",
                "correct_rank": card["rank"],
                "correct_dir": "candidate-a",
                "notes": "The second witness supports the group score.",
            },
        )
        assert saved.status_code == 201, saved.text
        assert saved.json()["correct_dir"] == dir_id
        assert saved.json()["correct_rank"] == 11
        assert client.get(f"/api/reviewer_dirs/{dir_id}").json()["member_query_ids"] == [
            2, 0, 1
        ]
    finally:
        client.__exit__(None, None, None)


@pytest.fixture
def qq() -> QQMatrix:
    sim, excluded = _build_matrix(4)
    return QQMatrix(
        sim=sim.astype(np.float16),
        file_ids=np.arange(4),
        excluded=excluded,
    )


@pytest.mark.parametrize(
    "query_id,members,expected_score,expected_support",
    [
        (1, [2, 0], F16_080, 0),
        (1, [2], 0.199951171875, 2),
        (1, [999, 3, 2], 0.199951171875, 2),
        (1, [1, 3, 2], 0.199951171875, 2),
        (1, [1, 3, 999], None, None),
        (3, [0, 2], None, None),
        (999, [0], None, None),
        (1, [], None, None),
    ],
)
def test_support_uses_only_the_existing_scorable_members(
    qq: QQMatrix,
    query_id: int,
    members: list[int],
    expected_score: float | None,
    expected_support: int | None,
) -> None:
    assert qq.score(query_id, members) == expected_score
    result = qq.score_with_support(query_id, members)
    if expected_score is None:
        assert result is None
    else:
        assert result is not None
        assert result.score == expected_score
        assert result.supporting_query_id == expected_support


@pytest.mark.parametrize("members", [[30, 20], [20, 30], [30, 20, 20]])
@pytest.mark.parametrize("order", [[0, 1, 2, 3], [2, 1, 0, 3]])
def test_equal_maxima_choose_lowest_query_id_not_member_or_matrix_order(
    qq: QQMatrix, members: list[int], order: list[int]
) -> None:
    qq.sim[1, 2] = qq.sim[1, 0]
    reordered = QQMatrix(
        sim=qq.sim[np.ix_(order, order)],
        file_ids=np.array([20, 10, 30, 40])[order],
        excluded=qq.excluded[order],
    )
    result = reordered.score_with_support(10, members)
    assert result is not None
    assert result.score == reordered.score(10, members) == F16_080
    assert result.supporting_query_id == 20


@pytest.mark.parametrize(
    "values,expected_score,expected_support",
    [
        ([0.0, -0.25], 0.0, 0),
        ([-0.5, -0.25], -0.25, 2),
        ([0.8, np.nan], np.nan, None),
        ([0.8, np.inf], np.inf, None),
        ([-np.inf, -np.inf], -np.inf, None),
        ([0.8, -np.inf], F16_080, 0),
    ],
)
def test_support_preserves_zero_negative_and_nonfinite_maxima(
    qq: QQMatrix,
    values: list[float],
    expected_score: float,
    expected_support: int | None,
) -> None:
    qq.sim[1, [0, 2]] = values
    result = qq.score_with_support(1, [2, 0])
    assert result is not None
    for score in (qq.score(1, [2, 0]), result.score):
        if np.isnan(expected_score):
            assert np.isnan(score)
        else:
            assert score == expected_score
    assert result.supporting_query_id == expected_support


def _record(dir_id: str, members: list[int]) -> dict:
    return {
        "dir_id": dir_id,
        "label": dir_id,
        "seed_query_id": members[0],
        "member_query_ids": members,
        "created_by": "Reviewer",
        "created_at": "2026-09-10T00:00:00Z",
    }


def _store() -> DataStore:
    return DataStore(
        file_id_to_filename={i: f"query-{i}.txt" for i in range(4)},
        unlabelled_texts={i: f"text of query-{i}.txt" for i in range(4)},
    )


def test_directory_scores_tie_order_cap_ranks_and_packet_shape_stay_unchanged(
    qq: QQMatrix,
) -> None:
    store = _store()
    records = [
        _record("reviewer-dir-a", [2, 0]),
        _record("reviewer-dir-b", [0]),
        *[_record(f"reviewer-dir-{letter}", [2]) for letter in "cdefg"],
        _record("reviewer-dir-self", [1, 0]),
        _record("reviewer-dir-unscorable", [3, 999]),
    ][::-1]
    before = deepcopy(records)
    cards = candidates_for_query(store=store, records=records, qq=qq, query_id=1)
    assert [(p.rank, p.dir_name, p.score) for p in cards] == [
        (11, "reviewer-dir-a", F16_080),
        (12, "reviewer-dir-b", F16_080),
        (13, "reviewer-dir-c", 0.199951171875),
        (14, "reviewer-dir-d", 0.199951171875),
        (15, "reviewer-dir-e", 0.199951171875),
    ]
    assert [p.supporting_member.query_id for p in cards] == [0, 0, 2, 2, 2]
    assert [p.supporting_member.score for p in cards] == [p.score for p in cards]
    assert cards[0].dir_files == ["query-2.txt", "query-0.txt"]

    packet = packet_dirs_for_query(store=store, records=records, qq=qq, query_id=1)
    assert len(packet) == 8
    assert packet[0]["dir_id"] == "reviewer-dir-self"
    assert packet[0]["group"] == "filed_into"
    assert packet[0]["score"] is None
    offered = next(p for p in packet if p["dir_id"] == "reviewer-dir-a")
    assert offered == {
        "dir_id": "reviewer-dir-a",
        "label": "reviewer-dir-a",
        "created_by": "Reviewer",
        "seed_query_id": 2,
        "group": "offered",
        "is_seed": False,
        "score": F16_080,
        "member_query_ids": [2, 0],
        "candidate_files": [
            {"filename": "query-2.txt", "text": "text of query-2.txt"},
            {"filename": "query-0.txt", "text": "text of query-0.txt"},
        ],
    }
    assert all("rank" not in p and "supporting_member" not in p for p in packet)
    assert records == before


@pytest.mark.parametrize("missing", ["filename", "text"])
def test_unavailable_support_keeps_its_identity_and_never_previews_the_seed(
    qq: QQMatrix, missing: str
) -> None:
    store = _store()
    if missing == "filename":
        del store.file_id_to_filename[0]
    else:
        del store.unlabelled_texts[0]
    record = _record("reviewer-dir-a", [2, 0])
    cards = candidates_for_query(store=store, records=[record], qq=qq, query_id=1)
    assert len(cards) == 1
    card = cards[0]
    assert card.rank == 11
    assert card.score == F16_080
    assert card.supporting_member.model_dump() == {
        "query_id": 0,
        "filename": None if missing == "filename" else "query-0.txt",
        "score": F16_080,
    }
    assert card.preview_text == ""
    assert card.dir_files == (
        ["query-2.txt"] if missing == "filename" else ["query-2.txt", "query-0.txt"]
    )
    assert record["member_query_ids"] == [2, 0]


def test_unscorable_members_remain_inspectable_without_becoming_support(
    qq: QQMatrix,
) -> None:
    record = _record("reviewer-dir-a", [3, 999, 2])
    card, = candidates_for_query(
        store=_store(), records=[record], qq=qq, query_id=1
    )
    assert card.supporting_member.query_id == 2
    assert card.score == card.supporting_member.score == 0.199951171875
    assert card.preview_text == "text of query-2.txt"
    assert card.dir_files == ["query-3.txt", "query-2.txt"]
    assert card.candidate_files[0].text == "text of query-3.txt"
    assert record["member_query_ids"] == [3, 999, 2]


def test_no_matrix_keeps_seeded_identity_and_files_without_ranked_evidence(
    tmp_path: Path,
) -> None:
    client = _signed_in(tmp_path, with_matrix=False)
    try:
        created = _create_dir(client, 0)
        for query_id in (0, 1):
            response = client.get(
                f"/api/query/{query_id}/predictions",
                params={"model": "bowphs/LaTa"},
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert [p["source"] for p in body["predictions"]] == ["model", "model"]
            assert all(p["supporting_member"] is None for p in body["predictions"])
            assert body["seeded_dirs"] == ([created] if query_id == 0 else [])
        assert created["best_match_score"] is None
        assert created["status"] == "awaiting_match"
        response = client.get(f"/api/candidate_dir/{created['dir_id']}/files")
        assert response.status_code == 200, response.text
        assert response.json() == [
            {"filename": "query-0.txt", "text": "text of query-0.txt"}
        ]
    finally:
        client.__exit__(None, None, None)
