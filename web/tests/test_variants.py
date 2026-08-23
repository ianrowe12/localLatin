from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from web.app import create_app

VARIANTS = ["raw", "abtt", "sif", "sif_abtt"]

# Each variant ranks the same two candidate dirs in a different order, so a
# response can be attributed to the variant that produced it.
_RANK1_BY_VARIANT = {
    "raw": "candidate-a",
    "abtt": "candidate-b",
    "sif": "candidate-a",
    "sif_abtt": "candidate-b",
}


def _write_fixture_data(root: Path, variants: list[str] | None = None) -> Path:
    variants = variants if variants is not None else VARIANTS
    unlabelled = root / "data" / "canon_unlabelled"
    labelled = root / "data" / "canon_labelled"
    predictions = root / "runs" / "active" / "resubmit" / "unlabelled"
    feedback = root / "runs" / "active" / "resubmit" / "webapp"

    unlabelled.mkdir(parents=True)
    predictions.mkdir(parents=True)
    feedback.mkdir(parents=True)
    (labelled / "candidate-a").mkdir(parents=True)
    (labelled / "candidate-b").mkdir(parents=True)

    (unlabelled / "query-0.txt").write_text("query text", encoding="utf-8")
    (labelled / "candidate-a" / "a.txt").write_text("candidate a", encoding="utf-8")
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
    for variant in variants:
        first = _RANK1_BY_VARIANT[variant]
        second = "candidate-b" if first == "candidate-a" else "candidate-a"
        path = predictions / f"unlabelled_predictions_{variant}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "file_id": 0,
                    "filename": "query-0.txt",
                    "model": "bowphs/LaTa",
                    "variant": variant,
                    "layer": 12,
                    "pooling": "mean",
                    "rank1_dir": first,
                    "rank1_score": 0.91,
                    "rank2_dir": second,
                    "rank2_score": 0.72,
                }
            )

    config_path = root / "config.yaml"
    variant_list = ", ".join(f'"{variant}"' for variant in variants)
    config_path.write_text(
        f"""
paths:
  data_root: "{root}"
  canon_unlabelled: "data/canon_unlabelled"
  canon_labelled: "data/canon_labelled"
  predictions_variant_pattern: "runs/active/resubmit/unlabelled/unlabelled_predictions_{{variant}}.csv"
  variants: [{variant_list}]
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


def _signed_in_client(tmp_path: Path, variants: list[str] | None = None) -> TestClient:
    client = TestClient(create_app(str(_write_fixture_data(tmp_path, variants))))
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


# --- /api/models -----------------------------------------------------------


def test_models_endpoint_exposes_available_and_default_variants(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        response = client.get("/api/models")
        assert response.status_code == 200
        model = response.json()[0]
        assert model["available_variants"] == VARIANTS
        assert model["default_variant"] == "sif_abtt"
    finally:
        client.__exit__(None, None, None)


def test_models_endpoint_omits_variants_without_a_predictions_csv(
    tmp_path: Path,
) -> None:
    client = _signed_in_client(tmp_path, variants=["raw", "sif_abtt"])
    try:
        model = client.get("/api/models").json()[0]
        assert model["available_variants"] == ["raw", "sif_abtt"]
    finally:
        client.__exit__(None, None, None)


# --- predictions -----------------------------------------------------------


def test_predictions_default_to_sif_abtt_when_variant_is_omitted(
    tmp_path: Path,
) -> None:
    client = _signed_in_client(tmp_path)
    try:
        response = client.get(
            "/api/query/0/predictions", params={"model": "bowphs/LaTa"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["variant"] == "sif_abtt"
        assert body["predictions"][0]["dir_name"] == "candidate-b"
    finally:
        client.__exit__(None, None, None)


def test_predictions_are_served_per_variant(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        for variant in VARIANTS:
            response = client.get(
                "/api/query/0/predictions",
                params={"model": "bowphs/LaTa", "variant": variant},
            )
            assert response.status_code == 200, variant
            body = response.json()
            assert body["variant"] == variant
            assert body["predictions"][0]["dir_name"] == _RANK1_BY_VARIANT[variant]
    finally:
        client.__exit__(None, None, None)


def test_non_default_variants_are_loaded_lazily(tmp_path: Path) -> None:
    from web.dependencies import get_store

    client = _signed_in_client(tmp_path)
    try:
        store = get_store()
        assert store.loaded_variants == {"sif_abtt"}

        client.get(
            "/api/query/0/predictions",
            params={"model": "bowphs/LaTa", "variant": "raw"},
        )
        assert store.loaded_variants == {"sif_abtt", "raw"}
    finally:
        client.__exit__(None, None, None)


def test_unknown_variant_is_rejected_with_422(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        response = client.get(
            "/api/query/0/predictions",
            params={"model": "bowphs/LaTa", "variant": "whitening"},
        )
        assert response.status_code == 422
    finally:
        client.__exit__(None, None, None)


def test_known_but_unconfigured_variant_is_rejected_with_400(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path, variants=["sif_abtt"])
    try:
        response = client.get(
            "/api/query/0/predictions",
            params={"model": "bowphs/LaTa", "variant": "raw"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "VARIANT_UNAVAILABLE"
    finally:
        client.__exit__(None, None, None)


def test_candidates_endpoint_honours_the_variant(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        raw = client.get(
            "/api/query/0/predictions/1/candidates",
            params={"model": "bowphs/LaTa", "variant": "raw"},
        )
        assert raw.status_code == 200
        assert [f["filename"] for f in raw.json()] == ["a.txt"]

        abtt = client.get(
            "/api/query/0/predictions/1/candidates",
            params={"model": "bowphs/LaTa", "variant": "abtt"},
        )
        assert abtt.status_code == 200
        assert [f["filename"] for f in abtt.json()] == ["b.txt"]
    finally:
        client.__exit__(None, None, None)


# --- feedback --------------------------------------------------------------


def _submit(client: TestClient, variant: str | None, notes: str, rank: int = 1):
    payload = {
        "query_id": 0,
        "model_slug": "bowphs/LaTa",
        "outcome": "matched_rank",
        "correct_rank": rank,
        "notes": notes,
    }
    if variant is not None:
        payload["variant"] = variant
    return client.post("/api/feedback", json=payload)


def test_feedback_records_the_variant_and_defaults_to_sif_abtt(
    tmp_path: Path,
) -> None:
    client = _signed_in_client(tmp_path)
    try:
        explicit = _submit(client, "raw", "raw note")
        assert explicit.status_code == 201
        assert explicit.json()["variant"] == "raw"

        implicit = _submit(client, None, "default note")
        assert implicit.status_code == 201
        assert implicit.json()["variant"] == "sif_abtt"
    finally:
        client.__exit__(None, None, None)


def test_feedback_rejects_an_unknown_variant_with_422(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        assert _submit(client, "whitening", "nope").status_code == 422
    finally:
        client.__exit__(None, None, None)


def test_latest_feedback_is_keyed_by_variant(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        assert _submit(client, "sif_abtt", "sif_abtt note").status_code == 201

        same = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa", "variant": "sif_abtt"},
        )
        assert same.status_code == 200
        assert same.json()["notes"] == "sif_abtt note"

        # A note saved under sif_abtt must not prefill the raw form.
        other = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa", "variant": "raw"},
        )
        assert other.status_code == 200
        assert other.json() is None

        assert _submit(client, "raw", "raw note", rank=2).status_code == 201
        refetched = client.get(
            "/api/feedback/latest",
            params={"query_id": 0, "model": "bowphs/LaTa", "variant": "raw"},
        )
        assert refetched.json()["notes"] == "raw note"
        assert refetched.json()["variant"] == "raw"
    finally:
        client.__exit__(None, None, None)


def test_latest_feedback_without_a_variant_uses_the_default(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        assert _submit(client, "sif_abtt", "default variant note").status_code == 201
        response = client.get(
            "/api/feedback/latest", params={"query_id": 0, "model": "bowphs/LaTa"}
        )
        assert response.json()["notes"] == "default variant note"
    finally:
        client.__exit__(None, None, None)


def test_feedback_export_carries_and_filters_by_variant(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        assert _submit(client, "raw", "raw note").status_code == 201
        assert _submit(client, "sif_abtt", "sif_abtt note").status_code == 201

        rows = list(csv.DictReader(client.get("/api/feedback/export").text.splitlines()))
        assert "variant" in rows[0]
        assert sorted(row["variant"] for row in rows) == ["raw", "sif_abtt"]

        filtered = list(
            csv.DictReader(
                client.get(
                    "/api/feedback/export", params={"variant": "raw"}
                ).text.splitlines()
            )
        )
        assert [row["notes"] for row in filtered] == ["raw note"]
    finally:
        client.__exit__(None, None, None)


def test_variant_migration_is_additive_and_does_not_touch_legacy_rows(
    tmp_path: Path,
) -> None:
    """Pre-variant DBs gain a nullable column; existing rows are left alone."""
    config_path = _write_fixture_data(tmp_path)
    db_path = tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            model_slug TEXT NOT NULL,
            outcome TEXT,
            correct_rank INTEGER,
            correct_dir TEXT,
            notes TEXT NOT NULL DEFAULT '',
            reviewer TEXT NOT NULL
        );
        INSERT INTO feedback (query_id, model_slug, outcome, correct_rank, notes, reviewer)
        VALUES (0, 'bowphs_LaTa', 'matched_rank', 1, 'legacy note', 'Old Reviewer');
        """
    )
    legacy.commit()
    legacy.close()

    with TestClient(create_app(str(config_path))):
        pass

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM feedback").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["notes"] == "legacy note"
    # Not backfilled: the row predates the variant CSVs, so it is never
    # attributed to one retroactively.
    assert rows[0]["variant"] is None
