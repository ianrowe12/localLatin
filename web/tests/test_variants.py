from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import fitz
import pytest
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


def _write_fixture_data(
    root: Path,
    variants: list[str] | None = None,
    written: list[str] | None = None,
    default_variant: str = "sif_abtt",
) -> Path:
    """Write fixture data.

    `variants` is what the config lists; `written` is what actually exists on
    disk. They are allowed to differ so the discovery filter can be tested.
    """
    variants = variants if variants is not None else VARIANTS
    written = written if written is not None else variants
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
    for variant in written:
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
  default_variant: "{default_variant}"
  feedback_db: "runs/active/resubmit/webapp/feedback.db"
  ig_examples_csv: "missing/phase12f_examples.csv"
  ig_artifacts_dir: "missing/artifacts"
auth:
  secure_cookies: false
""",
        encoding="utf-8",
    )
    return config_path


def _signed_in_client(
    tmp_path: Path,
    variants: list[str] | None = None,
    written: list[str] | None = None,
    default_variant: str = "sif_abtt",
) -> TestClient:
    config_path = _write_fixture_data(tmp_path, variants, written, default_variant)
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
    """All four are configured; only two exist on disk, so only two are served."""
    client = _signed_in_client(tmp_path, variants=VARIANTS, written=["raw", "sif_abtt"])
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
    client = _signed_in_client(tmp_path, variants=VARIANTS, written=["sif_abtt"])
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


# --- configurable default variant ------------------------------------------


def test_configured_default_variant_drives_every_omitted_variant_route(
    tmp_path: Path,
) -> None:
    """A deployment whose default is not sif_abtt must still work end to end."""
    client = _signed_in_client(
        tmp_path, variants=["raw", "sif"], written=["raw", "sif"], default_variant="raw"
    )
    try:
        model = client.get("/api/models").json()[0]
        assert model["default_variant"] == "raw"
        assert model["available_variants"] == ["raw", "sif"]

        predictions = client.get(
            "/api/query/0/predictions", params={"model": "bowphs/LaTa"}
        )
        assert predictions.status_code == 200
        assert predictions.json()["variant"] == "raw"
        assert predictions.json()["predictions"][0]["dir_name"] == "candidate-a"

        created = _submit(client, None, "note under configured default")
        assert created.status_code == 201
        assert created.json()["variant"] == "raw"

        latest = client.get(
            "/api/feedback/latest", params={"query_id": 0, "model": "bowphs/LaTa"}
        )
        assert latest.json()["notes"] == "note under configured default"

        packet = client.get("/api/packets/review/0", params={"model": "bowphs/LaTa"})
        assert packet.status_code == 200
    finally:
        client.__exit__(None, None, None)


# --- startup safety ---------------------------------------------------------


def test_startup_raises_when_the_default_variant_csv_is_missing(
    tmp_path: Path,
) -> None:
    """No silent degradation, and explicitly no fallback to the stale CSV."""
    config_path = _write_fixture_data(tmp_path, variants=list(VARIANTS), written=["raw"])
    predictions = tmp_path / "runs" / "active" / "resubmit" / "unlabelled"
    # The pre-variant frozen file sitting right next to the variant CSVs must
    # not be picked up as a fallback.
    (predictions / "unlabelled_predictions.csv").write_text(
        (predictions / "unlabelled_predictions_raw.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        with TestClient(create_app(str(config_path))):
            pass

    message = str(excinfo.value)
    assert "sif_abtt" in message
    assert "unlabelled_predictions_sif_abtt.csv" in message


def test_a_corrupt_lazily_loaded_csv_is_reported_as_unavailable(
    tmp_path: Path,
) -> None:
    """A bad non-default CSV must not 500 or poison the store."""
    config_path = _write_fixture_data(tmp_path)
    predictions = tmp_path / "runs" / "active" / "resubmit" / "unlabelled"
    (predictions / "unlabelled_predictions_raw.csv").write_text(
        "file_id,filename\n0,too,many,fields,here\n", encoding="utf-8"
    )

    client = TestClient(create_app(str(config_path)))
    client.__enter__()
    try:
        client.post(
            "/api/auth/register",
            json={
                "username": "pi",
                "display_name": "PI",
                "password": "correct horse battery staple",
            },
        )
        response = client.get(
            "/api/query/0/predictions",
            params={"model": "bowphs/LaTa", "variant": "raw"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "VARIANT_UNAVAILABLE"

        # Nothing partial retained, and the default variant still serves.
        from web.dependencies import get_store

        store = get_store()
        assert not any(key[1] == "raw" for key in store.predictions)
        assert "raw" not in store.loaded_variants

        default = client.get(
            "/api/query/0/predictions", params={"model": "bowphs/LaTa"}
        )
        assert default.status_code == 200
    finally:
        client.__exit__(None, None, None)


# --- review packets ---------------------------------------------------------


def _legacy_feedback_row(tmp_path: Path, notes: str) -> None:
    """Insert a row the way the pre-variant schema did: variant IS NULL."""
    db_path = tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO feedback
            (query_id, model_slug, outcome, correct_rank, correct_dir, notes, reviewer)
        VALUES (0, 'bowphs_LaTa', 'matched_rank', 1, 'candidate-b', ?, 'Pilot Reviewer')
        """,
        (notes,),
    )
    conn.commit()
    conn.close()


def _packet_text(response) -> str:
    pdf = fitz.open(stream=response.content, filetype="pdf")
    return "\n".join(page.get_text() for page in pdf)


def test_review_packet_keeps_pre_variant_feedback(tmp_path: Path) -> None:
    """Regression: filtering the packet by variant dropped the whole pilot."""
    client = _signed_in_client(tmp_path)
    try:
        _legacy_feedback_row(tmp_path, "legacy pilot note")
        assert _submit(client, "sif_abtt", "new sif_abtt note").status_code == 201

        packet = client.get("/api/packets/review/0", params={"model": "bowphs/LaTa"})
        assert packet.status_code == 200
        text = _packet_text(packet)
        assert "legacy pilot note" in text
        assert "new sif_abtt note" in text
        # Provenance is visible rather than filtered away.
        assert "pre-variant" in text
        assert "variant sif_abtt" in text
    finally:
        client.__exit__(None, None, None)


def test_review_packet_feedback_can_be_filtered_by_variant_explicitly(
    tmp_path: Path,
) -> None:
    client = _signed_in_client(tmp_path)
    try:
        _legacy_feedback_row(tmp_path, "legacy pilot note")
        assert _submit(client, "raw", "raw only note").status_code == 201
        assert _submit(client, "sif_abtt", "sif_abtt only note").status_code == 201

        text = _packet_text(
            client.get(
                "/api/packets/review/0",
                params={"model": "bowphs/LaTa", "feedback_variant": "raw"},
            )
        )
        assert "raw only note" in text
        assert "sif_abtt only note" not in text
        assert "legacy pilot note" not in text
    finally:
        client.__exit__(None, None, None)


def test_review_packet_records_the_prediction_variant(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    try:
        text = _packet_text(
            client.get(
                "/api/packets/review/0",
                params={"model": "bowphs/LaTa", "variant": "abtt"},
            )
        )
        assert "Prediction variant: abtt" in text
    finally:
        client.__exit__(None, None, None)
