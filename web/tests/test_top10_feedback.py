from __future__ import annotations

import csv
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

    with (predictions / "unlabelled_predictions.csv").open("w", newline="") as f:
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
  predictions_combined: "runs/active/resubmit/unlabelled/unlabelled_predictions.csv"
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


def test_latest_feedback_is_scoped_to_reviewer_and_normalized_model(
    tmp_path: Path,
) -> None:
    client = _signed_in_client(tmp_path)
    try:
        created = client.post(
            "/api/auth/accounts",
            json={
                "username": "reviewer",
                "display_name": "Reviewer",
                "role": "reviewer",
                "password": "correct horse battery staple",
            },
        )
        assert created.status_code == 201

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

        assert client.post("/api/auth/signout").status_code == 204
        signed_in = client.post(
            "/api/auth/signin",
            json={
                "username": "reviewer",
                "password": "correct horse battery staple",
            },
        )
        assert signed_in.status_code == 200

        no_reviewer_feedback = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs_LaTa"},
        )
        assert no_reviewer_feedback.status_code == 200
        assert no_reviewer_feedback.json() is None

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
        assert latest.json()["notes"] == "reviewer note"
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
