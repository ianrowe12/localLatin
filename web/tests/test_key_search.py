from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from web.app import create_app
from web.dependencies import get_current_user, get_db, get_store

SHORT = "CNIC.325.2"
LONG = "CNIC.325.10"
DOUBLE_SPACE = "DSIR.384.255  cap. 11"
COMMA = "DGEL492.636 cap. 7,8"
INFIX = f"Alias {SHORT}"
OTHER = "Ephesians.5:22"
FILES = ["z-witness.txt", "a-witness.txt", "unrelated.txt", "b-witness.txt"]
LATA = "bowphs/LaTa"
PHILTA = "bowphs/PhilTa"


def _write_predictions(
    root: Path,
    variant: str,
    model_rows: dict[str, list[tuple[str, dict[int, str]]]],
) -> None:
    path = root / f"predictions_{variant}.csv"
    ranks = [f"rank{i}_{suffix}" for i in range(1, 12) for suffix in ("dir", "score")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file_id", "filename", "model", "variant", "status", *ranks]
        )
        writer.writeheader()
        for model, rows in model_rows.items():
            for fid, (status, ranking) in enumerate(rows):
                row = dict(file_id=fid, filename=FILES[fid], model=model, variant=variant, status=status)
                for rank, name in ranking.items():
                    row[f"rank{rank}_dir"] = name
                    row[f"rank{rank}_score"] = 0.9
                writer.writerow(row)


@pytest.fixture
def client(tmp_path: Path):
    unlabelled = tmp_path / "unlabelled"
    unlabelled.mkdir()
    for name in FILES:
        (unlabelled / name).write_text("query text", encoding="utf-8")
    for key in (SHORT, LONG, DOUBLE_SPACE, COMMA, INFIX, OTHER):
        directory = tmp_path / "labelled" / key
        directory.mkdir(parents=True)
        (directory / "a.txt").write_text("first unit", encoding="utf-8")
        if key == SHORT:
            (directory / "b.txt").write_text("second unit", encoding="utf-8")

    _write_predictions(
        tmp_path, "abtt", {
            LATA: [
                ("ok", {1: LONG, 2: SHORT, 10: COMMA}),
                ("ok", {1: DOUBLE_SPACE, 2: SHORT}),
                ("ok", {1: LONG, 11: OTHER}),
                ("excluded_blank_source", {1: SHORT}),
            ],
            PHILTA: [
                ("ok", {1: SHORT, 2: DOUBLE_SPACE}),
                ("ok", {1: COMMA}),
                ("excluded_zero_norm", {1: SHORT}),
                ("excluded_zero_norm", {}),
            ],
        },
    )
    _write_predictions(
        tmp_path, "raw", {LATA: [("ok", {}), ("ok", {}), ("ok", {1: DOUBLE_SPACE}), ("ok", {})]}
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
paths:
  data_root: "{tmp_path}"
  canon_unlabelled: unlabelled
  canon_labelled: labelled
  predictions_variant_pattern: "predictions_{{variant}}.csv"
  variants: [abtt, raw]
  default_variant: abtt
  feedback_db: feedback.db
""",
        encoding="utf-8",
    )
    app = create_app(str(config))
    app.dependency_overrides[get_current_user] = lambda: None
    with TestClient(app) as test_client:
        yield test_client


def _get(client: TestClient, path: str, **params):
    response = client.get(path, params=params)
    assert response.status_code == 200
    return response.json()


def _ids(body: dict) -> list[int]:
    return [item["file_id"] for item in body["items"]]


def test_labelled_search_is_natural_and_prefix_first(client: TestClient) -> None:
    path = "/api/labelled_keys"
    assert _get(client, path, search="  cNic.325. ", page_size=2) == {
        "items": [{"key": SHORT, "unit_count": 2}, {"key": LONG, "unit_count": 1}],
        "total": 3, "page": 1, "page_size": 2, "has_more": True,
    }
    second = _get(client, path, search="cnic.325.", page_size=2, page=2)
    assert second["items"] == [{"key": INFIX, "unit_count": 1}]
    assert not second["has_more"]
    assert _get(client, path, search="dsir.384.255\tCAP.  11")["items"] == [
        {"key": DOUBLE_SPACE, "unit_count": 1}
    ]
    all_keys = _get(client, path, search="  ")
    assert [item["key"] for item in all_keys["items"]] == [
        INFIX, SHORT, LONG, COMMA, DOUBLE_SPACE, OTHER
    ]
    assert all_keys["page_size"] == 50
    assert _get(client, path, search="not in the corpus")["total"] == 0


def test_labelled_search_requires_auth_and_valid_pagination(client: TestClient) -> None:
    for params in ({"page": 0}, {"page_size": 0}, {"page_size": 1001}):
        assert client.get("/api/labelled_keys", params=params).status_code == 422
    del client.app.dependency_overrides[get_current_user]
    assert client.get("/api/labelled_keys").status_code == 401


@pytest.mark.parametrize("key", [COMMA, DOUBLE_SPACE])
def test_candidate_files_accept_url_encoded_keys(client: TestClient, key: str) -> None:
    assert _get(client, f"/api/candidate_dir/{quote(key, safe='')}/files") == [
        {"filename": "a.txt", "text": "first unit"}
    ]


def test_ranked_search_prefix_exact_and_any_model(client: TestClient) -> None:
    prefix = _get(client, "/api/queries", key=" cNiC.325. ")
    assert _ids(prefix) == [0, 1, 2]
    assert prefix["items"][0]["key_matches"] == [
        {"model": "bowphs_LaTa", "rank": 1, "dir_name": LONG},
        {"model": "bowphs_PhilTa", "rank": 1, "dir_name": SHORT},
        {"model": "bowphs_LaTa", "rank": 2, "dir_name": SHORT},
    ]
    assert _ids(_get(client, "/api/queries", key="CNIC.325.", key_match="exact")) == []
    assert _ids(_get(client, "/api/queries", key=SHORT.lower(), key_match="exact")) == [0, 1]
    assert _ids(_get(client, "/api/queries", key="325")) == []
    assert _ids(_get(client, "/api/queries", key=OTHER)) == []
    assert _get(client, "/api/queries", key=COMMA, key_model=LATA)["items"][0]["key_matches"] == [
        {"model": "bowphs_LaTa", "rank": 10, "dir_name": COMMA}
    ]
    unfiltered = _get(client, "/api/queries")
    assert _ids(unfiltered) == [0, 1, 2, 3]
    assert all(not item.get("key_matches") for item in unfiltered["items"])


def test_ranked_search_model_top1_and_variant(client: TestClient) -> None:
    params = {"key": " \t dsir.384.255  CAP.\t11 ", "key_match": "exact"}
    path = "/api/queries"
    combined = _get(client, path, **params)
    assert _ids(combined) == [0, 1]
    assert combined["items"][0]["key_matches"] == [
        {"model": "bowphs_PhilTa", "rank": 2, "dir_name": DOUBLE_SPACE}
    ]
    assert _ids(_get(client, path, **params, key_top1=True)) == [1]
    assert _ids(_get(client, path, **params, key_model="bowphs_PhilTa")) == [0]
    assert _ids(_get(client, path, **params, key_model="bowphs_PhilTa", key_top1=True)) == []
    assert _ids(_get(client, path, **params, variant="raw", key_top1=True, key_model=LATA)) == [2]
    assert _ids(_get(client, path, **params)) == [0, 1]


def test_ranked_search_composes_filters_before_sort_and_paging(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_db(), "get_query_statuses", AsyncMock(return_value={
        fid: {"review_status": "reviewed", "review_count": 1} for fid in (0, 1, 3)
    }))
    params = {
        "key": "cnic", "search": "WITNESS", "status": "reviewed",
        "sort": "filename", "page_size": 1,
    }
    first = _get(client, "/api/queries", **params)
    assert _ids(first) == [1]
    assert (first["total"], first["page"], first["page_size"], first["has_more"]) == (2, 1, 1, True)
    second = _get(client, "/api/queries", **params, page=2)
    assert _ids(second) == [0]
    assert not second["has_more"]
    assert _ids(_get(client, "/api/queries", key="cnic", search="unrelated", status="unreviewed")) == [2]
    assert _ids(_get(client, "/api/queries", key="cnic", search="unrelated", status="reviewed")) == []


@pytest.mark.parametrize(("variant", "model", "code"), [
    ("abtt", "unknown", "INVALID_MODEL"),
    ("raw", PHILTA, "INVALID_MODEL"),
    ("sif", LATA, "VARIANT_UNAVAILABLE"),
])
def test_ranked_search_uses_prediction_endpoint_errors(
    client: TestClient, variant: str, model: str, code: str,
) -> None:
    response = client.get("/api/queries", params={"key": SHORT, "variant": variant, "key_model": model})
    predictions = client.get("/api/query/0/predictions", params={"variant": variant, "model": model})
    assert response.status_code == predictions.status_code == 400
    assert response.json() == predictions.json()
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize("params", [{"variant": "unknown"}, {"key_match": "contains"}])
def test_ranked_search_rejects_invalid_options(client: TestClient, params: dict) -> None:
    assert client.get("/api/queries", params={"key": SHORT, **params}).status_code == 422


def test_ranked_search_cache_follows_variant_reloads(client: TestClient, tmp_path: Path) -> None:
    params = {"key": DOUBLE_SPACE, "variant": "raw"}
    assert _ids(_get(client, "/api/queries", **params)) == [2]
    _write_predictions(
        tmp_path, "raw", {LATA: [("ok", {1: DOUBLE_SPACE}), ("ok", {}), ("ok", {}), ("ok", {})]}
    )
    assert _ids(_get(client, "/api/queries", **params)) == [2]
    get_store()._discard_variant("raw")
    assert _ids(_get(client, "/api/queries", **params)) == [0]
