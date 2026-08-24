"""resolve_example_id: unlabelled queries vs labelled ones (issue #53).

``query_file_id`` is only unique within a corpus. The labelled canon examples
number their split from 0 and the unlabelled review queue numbers its own files
from 0, so a live query can collide with a labelled example on
(file_id, candidate_dir). ``GET /api/query/{file_id}/token_map`` always means
the unlabelled corpus, so it must resolve to the unlabelled row.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_token_map_variants import _write_npz  # noqa: E402

from web.app import create_app  # noqa: E402
from web.services import token_map_svc  # noqa: E402
from web.services.data_store import DataStore  # noqa: E402

ALL_VARIANTS = ("baseline", "abtt", "sif", "sif_abtt")


def _row(example_id: int, query_source: str | None, **overrides) -> dict:
    row = {
        "example_id": example_id,
        "model_name": "bowphs/LaTa",
        "bucket": "correct_similar",
        "query_path": "data/canon/X/a.txt",
        "candidate_path": "data/canon/X/b.txt",
        "query_file_id": 40,
        "query_folder_id": "X",
        "candidate_folder_id": "CANC.314.3",
        "candidate_label": "CANC.314.3",
        "gold_similar": 1,
        "baseline_pred": 0,
        "abtt_pred": 1,
    }
    if query_source is not None:
        row["query_source"] = query_source
    row.update(overrides)
    return row


def _store(rows: list[dict], artifacts: dict[int, Path]) -> DataStore:
    store = DataStore()
    store.ig_examples = pd.DataFrame(rows)
    store.ig_artifact_paths = dict(artifacts)
    return store


@pytest.fixture(autouse=True)
def _clear_npz_cache():
    token_map_svc._load_npz.cache_clear()
    yield
    token_map_svc._load_npz.cache_clear()


def test_unlabelled_query_wins_a_file_id_collision(tmp_path: Path) -> None:
    labelled_npz = tmp_path / "l.npz"
    unlabelled_npz = tmp_path / "u.npz"
    for p in (labelled_npz, unlabelled_npz):
        _write_npz(p, variants=ALL_VARIANTS, token_strings=True, sif_weights=True)

    store = _store(
        [_row(7, "labelled"), _row(128, "unlabelled")],
        {7: labelled_npz, 128: unlabelled_npz},
    )

    assert (
        token_map_svc.resolve_example_id(
            store, 40, "CANC.314.3", "bowphs_LaTa", query_source="unlabelled"
        )
        == 128
    )
    assert (
        token_map_svc.resolve_example_id(
            store, 40, "CANC.314.3", "bowphs_LaTa", query_source="labelled"
        )
        == 7
    )


def test_query_source_is_optional_on_the_call(tmp_path: Path) -> None:
    """Callers that do not pass query_source keep the old first-match behaviour."""
    npz = tmp_path / "a.npz"
    _write_npz(npz, variants=ALL_VARIANTS, token_strings=True, sif_weights=True)
    store = _store([_row(7, "labelled"), _row(128, "unlabelled")], {7: npz, 128: npz})

    assert token_map_svc.resolve_example_id(store, 40, "CANC.314.3") == 7


def test_query_source_is_optional_in_the_csv(tmp_path: Path) -> None:
    """CSVs written before unlabelled examples existed resolve exactly as before."""
    npz = tmp_path / "a.npz"
    _write_npz(npz, variants=ALL_VARIANTS, token_strings=True, sif_weights=True)
    store = _store([_row(7, None)], {7: npz})

    assert (
        token_map_svc.resolve_example_id(
            store, 40, "CANC.314.3", "bowphs_LaTa", query_source="unlabelled"
        )
        == 7
    )


def test_rows_without_an_artifact_do_not_mask_a_usable_row(tmp_path: Path) -> None:
    npz = tmp_path / "a.npz"
    _write_npz(npz, variants=ALL_VARIANTS, token_strings=True, sif_weights=True)
    # example 7 is listed first but has no NPZ on disk.
    store = _store([_row(7, "unlabelled"), _row(128, "unlabelled")], {128: npz})

    assert (
        token_map_svc.resolve_example_id(
            store, 40, "CANC.314.3", "bowphs_LaTa", query_source="unlabelled"
        )
        == 128
    )


def test_labelled_row_never_serves_an_unlabelled_query(tmp_path: Path) -> None:
    """A real collision from the deployed top-10 must resolve to nothing.

    Unlabelled query C1525.69r.5.txt has file_id 1189, and LaTa ranks
    Can.apost.7 into its top 10 under both raw and abtt. Labelled example 67
    happens to carry query_file_id 1189 with candidate_folder_id Can.apost.7
    as well, but its query is C1525.7v.2.txt, a different manuscript. One of
    13 such combinations. Before the strict filter this served ex67's token map
    as evidence for C1525.69r.5.txt.
    """
    npz = tmp_path / "a.npz"
    _write_npz(npz, variants=ALL_VARIANTS, token_strings=True, sif_weights=True)
    store = _store(
        [
            _row(
                67,
                "labelled",
                query_file_id=1189,
                candidate_folder_id="Can.apost.7",
                candidate_label="Can.apost.7",
                query_path="/u/irowerojas/localLatin/canon/Can.apost.45/C1525.7v.2.txt",
            )
        ],
        {67: npz},
    )

    assert (
        token_map_svc.resolve_example_id(
            store, 1189, "Can.apost.7", "bowphs_LaTa", query_source="unlabelled"
        )
        is None
    )
    # The labelled artifact is still reachable for what it actually describes.
    assert (
        token_map_svc.resolve_example_id(
            store, 1189, "Can.apost.7", "bowphs_LaTa", query_source="labelled"
        )
        == 67
    )


def test_no_match_still_returns_none(tmp_path: Path) -> None:
    npz = tmp_path / "a.npz"
    _write_npz(npz, variants=ALL_VARIANTS, token_strings=True, sif_weights=True)
    store = _store([_row(7, "unlabelled")], {7: npz})

    assert token_map_svc.resolve_example_id(store, 40, "OTHER.DIR") is None
    assert token_map_svc.resolve_example_id(store, 999, "CANC.314.3") is None


def _api_client_with_collision(tmp_path: Path) -> TestClient:
    """App fixture where a labelled and an unlabelled example collide on file_id 0."""
    unlabelled = tmp_path / "data" / "canon_unlabelled"
    labelled = tmp_path / "data" / "canon_labelled" / "candidate-a"
    predictions = tmp_path / "runs" / "active" / "resubmit" / "unlabelled"
    ig_root = tmp_path / "runs" / "active" / "ig_examples"
    for d in (unlabelled, labelled, predictions, ig_root):
        d.mkdir(parents=True)
    (tmp_path / "runs" / "active" / "resubmit" / "webapp").mkdir(parents=True)
    (unlabelled / "query-0.txt").write_text("query text", encoding="utf-8")
    (labelled / "a.txt").write_text("candidate a", encoding="utf-8")

    fieldnames = ["file_id", "filename", "model", "layer", "pooling", "rank1_dir", "rank1_score"]
    with (predictions / "unlabelled_predictions_sif_abtt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "file_id": 0,
            "filename": "query-0.txt",
            "model": "bowphs/LaTa",
            "layer": 1,
            "pooling": "mean",
            "rank1_dir": "candidate-a",
            "rank1_score": 0.9,
        })

    pd.DataFrame([
        _row(1, "labelled", query_file_id=0, candidate_folder_id="candidate-a",
             candidate_label="candidate-a"),
        _row(2, "unlabelled", query_file_id=0, candidate_folder_id="candidate-a",
             candidate_label="candidate-a"),
    ]).to_csv(ig_root / "phase12f_examples.csv", index=False)

    for eid in (1, 2):
        _write_npz(
            ig_root / "artifacts" / "bowphs_LaTa" / f"example{eid:03d}_pair_example.npz",
            variants=ALL_VARIANTS,
            token_strings=True,
            sif_weights=True,
        )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  data_root: "{tmp_path}"
  canon_unlabelled: "data/canon_unlabelled"
  canon_labelled: "data/canon_labelled"
  predictions_variant_pattern: "runs/active/resubmit/unlabelled/unlabelled_predictions_{{variant}}.csv"
  variants: ["sif_abtt"]
  default_variant: "sif_abtt"
  feedback_db: "runs/active/resubmit/webapp/feedback.db"
  ig_examples_csv: "runs/active/ig_examples/phase12f_examples.csv"
  ig_artifacts_dir: "runs/active/ig_examples/artifacts"
auth:
  secure_cookies: false
""",
        encoding="utf-8",
    )

    client = TestClient(create_app(str(config_path)))
    client.__enter__()
    response = client.post(
        "/api/auth/register",
        json={"username": "pi", "display_name": "PI", "password": "correct horse battery staple"},
    )
    assert response.status_code == 201
    return client


def test_api_query_token_map_resolves_the_unlabelled_example(tmp_path: Path) -> None:
    client = _api_client_with_collision(tmp_path)
    try:
        response = client.get(
            "/api/query/0/token_map",
            params={"candidate_dir": "candidate-a", "model": "bowphs_LaTa"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["example_id"] == 2
        assert payload["available_variants"] == list(ALL_VARIANTS)
        assert [t["text"] for t in payload["query_tokens"]] == ["de", "et", "in", "est"]
    finally:
        client.__exit__(None, None, None)
