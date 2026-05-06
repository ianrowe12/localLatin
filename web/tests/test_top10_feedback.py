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
