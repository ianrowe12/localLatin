"""The retrieval run's per-row status, from the real CSV loader to the wire.

Issue #156. `scripts/resubmit/run_resubmit_unlabelled_retrieval.py` keeps a row
for every (model, query) pair even when its degenerate-source guard drops the
query: the row stays, every `rank*` cell is blank, and a `status` column records
which guard fired. Before this, `services/data_store.py` discarded that column,
so `GET /api/query/{id}/predictions` answered `predictions: []` for a
deliberately excluded manuscript and for a row that is empty for no known reason
alike -- and no frontend or diagnostic could honestly tell them apart.

These go through `build_store()` and the real route rather than a mocked
response, because a mock that invents the field would pass while the deployed
route never emitted it.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import create_app

MODEL = "bowphs/LaTa"
SLUG = "bowphs_LaTa"

# file_id -> (status cell, rank1 dir or None for a blank ranking)
_ROWS: dict[int, tuple[str | None, str | None]] = {
    0: ("ok", "candidate-a"),
    1: ("excluded_blank_source", None),
    2: ("excluded_zero_norm", None),
    # A scored row whose status the writer left empty: unknown, not excluded.
    3: ("", "candidate-b"),
    # An empty ranking with no exclusion recorded: unexplained, and it must stay
    # that way rather than being dressed up as one of the two guards.
    4: ("ok", None),
}

_FIELDNAMES = [
    "file_id",
    "filename",
    "model",
    "variant",
    "layer",
    "pooling",
    "rank1_dir",
    "rank1_score",
    "status",
]


def _write_fixture(root: Path, with_status_column: bool) -> Path:
    unlabelled = root / "data" / "canon_unlabelled"
    labelled = root / "data" / "canon_labelled"
    predictions = root / "runs" / "active" / "resubmit" / "unlabelled"
    feedback = root / "runs" / "active" / "resubmit" / "webapp"

    unlabelled.mkdir(parents=True)
    predictions.mkdir(parents=True)
    feedback.mkdir(parents=True)
    for name in ("candidate-a", "candidate-b"):
        (labelled / name).mkdir(parents=True)
        (labelled / name / f"{name}.txt").write_text(name, encoding="utf-8")

    for file_id in _ROWS:
        (unlabelled / f"query-{file_id}.txt").write_text("query text", encoding="utf-8")

    fieldnames = list(_FIELDNAMES)
    if not with_status_column:
        fieldnames.remove("status")

    path = predictions / "unlabelled_predictions_sif_abtt.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for file_id, (status, rank1) in _ROWS.items():
            row = {
                "file_id": file_id,
                "filename": f"query-{file_id}.txt",
                "model": MODEL,
                "variant": "sif_abtt",
                "layer": 12,
                "pooling": "mean",
                # An excluded query keeps its row with blank rank cells; that is
                # exactly how the retrieval script writes it.
                "rank1_dir": rank1 or "",
                "rank1_score": 0.91 if rank1 else "",
            }
            if with_status_column:
                row["status"] = status if status is not None else ""
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


def _signed_in_client(tmp_path: Path, with_status_column: bool = True) -> TestClient:
    config_path = _write_fixture(tmp_path, with_status_column)
    client = TestClient(create_app(str(config_path)))
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


@pytest.fixture
def client(tmp_path: Path):
    client = _signed_in_client(tmp_path)
    try:
        yield client
    finally:
        client.__exit__(None, None, None)


def _predictions(client: TestClient, file_id: int) -> dict:
    response = client.get(
        f"/api/query/{file_id}/predictions", params={"model": MODEL}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_scored_row_reports_ok_with_its_ranking(client: TestClient) -> None:
    body = _predictions(client, 0)
    assert body["status"] == "ok"
    assert [p["rank"] for p in body["predictions"]] == [1]


def test_blank_source_exclusion_survives_loader_and_router(client: TestClient) -> None:
    body = _predictions(client, 1)
    assert body["predictions"] == []
    assert body["status"] == "excluded_blank_source"


def test_zero_norm_exclusion_is_not_flattened_into_blank_source(
    client: TestClient,
) -> None:
    """Two different guards, two different explanations for the reviewer."""
    body = _predictions(client, 2)
    assert body["predictions"] == []
    assert body["status"] == "excluded_zero_norm"


def test_empty_status_cell_is_null_not_an_exclusion(client: TestClient) -> None:
    body = _predictions(client, 3)
    assert body["status"] is None
    assert [p["rank"] for p in body["predictions"]] == [1]


def test_empty_ranking_without_an_exclusion_stays_unexplained(
    client: TestClient,
) -> None:
    """`ok` plus no candidates is a failure to explain, not one to invent."""
    body = _predictions(client, 4)
    assert body["predictions"] == []
    assert body["status"] == "ok"


def test_legacy_csv_without_a_status_column_reports_null(tmp_path: Path) -> None:
    """Old artifacts keep working, and never gain an exclusion they never had."""
    client = _signed_in_client(tmp_path, with_status_column=False)
    try:
        for file_id in _ROWS:
            body = _predictions(client, file_id)
            assert body["status"] is None, file_id
    finally:
        client.__exit__(None, None, None)


def test_status_is_carried_per_row_not_per_model(client: TestClient) -> None:
    """One model's excluded query does not make its other queries excluded."""
    statuses = {file_id: _predictions(client, file_id)["status"] for file_id in _ROWS}
    assert statuses == {
        0: "ok",
        1: "excluded_blank_source",
        2: "excluded_zero_norm",
        3: None,
        4: "ok",
    }
