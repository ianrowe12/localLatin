"""Token-map service: 4-variant plumbing and stored token strings (issues #46/#47)."""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from web.app import create_app
from web.services import token_map_svc
from web.services.data_store import DataStore

METHODS = ("ig", "bertscore")
Q_LEN = 4
C_LEN = 3
DIM = 5


def _write_npz(
    path: Path,
    *,
    variants: tuple[str, ...],
    token_strings: bool,
    sif_weights: bool,
) -> None:
    rng = np.random.default_rng(0)
    data: dict[str, np.ndarray] = {
        "example_id": np.array([1], dtype=np.int32),
        "layer": np.array([7], dtype=np.int32),
        "D": np.array([10], dtype=np.int32),
        "query_input_ids": np.arange(Q_LEN, dtype=np.int64)[None, :],
        "candidate_input_ids": np.arange(C_LEN, dtype=np.int64)[None, :],
        "query_attention_mask": np.ones((1, Q_LEN), dtype=np.int64),
        "candidate_attention_mask": np.ones((1, C_LEN), dtype=np.int64),
        "query_hidden": rng.normal(size=(Q_LEN, DIM)).astype(np.float32),
        "candidate_hidden": rng.normal(size=(C_LEN, DIM)).astype(np.float32),
        "query_ig_baseline": rng.normal(size=Q_LEN).astype(np.float32),
        "query_ig_abtt": rng.normal(size=Q_LEN).astype(np.float32),
        "candidate_ig_baseline": rng.normal(size=C_LEN).astype(np.float32),
        "candidate_ig_abtt": rng.normal(size=C_LEN).astype(np.float32),
    }
    for method in METHODS:
        for variant in variants:
            data[f"pair_matrix_{method}_{variant}"] = rng.normal(
                size=(Q_LEN, C_LEN)
            ).astype(np.float32)
            data[f"topk_{method}_{variant}_query"] = np.arange(2, dtype=np.int32)
            data[f"topk_{method}_{variant}_candidate"] = np.arange(2, dtype=np.int32)
    if token_strings:
        data["query_token_strings"] = np.asarray(["de", "et", "in", "est"], dtype=np.str_)
        data["candidate_token_strings"] = np.asarray(["qui", "non", "sponte"], dtype=np.str_)
    if sif_weights:
        data["query_sif_weights"] = np.asarray([0.2, 0.3, 0.4, 3.1], dtype=np.float32)
        data["candidate_sif_weights"] = np.asarray([0.5, 0.5, 2.0], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def _make_store(npz_path: Path) -> DataStore:
    store = DataStore()
    store.ig_examples = pd.DataFrame(
        [
            {
                "example_id": 1,
                "model_name": "bowphs/LaTa",
                "bucket": "correct_similar",
                "query_path": "data/canon/X/a.txt",
                "candidate_path": "data/canon/X/b.txt",
                "query_file_id": 11,
                "query_folder_id": "X",
                "candidate_folder_id": "X",
                "candidate_label": "X",
                "gold_similar": 1,
                "baseline_pred": 0,
                "abtt_pred": 1,
            }
        ]
    )
    store.ig_artifact_paths = {1: npz_path}
    return store


@pytest.fixture(autouse=True)
def _clear_npz_cache():
    token_map_svc._load_npz.cache_clear()
    yield
    token_map_svc._load_npz.cache_clear()


def test_all_four_variants_are_served(tmp_path: Path) -> None:
    npz = tmp_path / "bowphs_LaTa" / "example001_pair_example.npz"
    _write_npz(
        npz,
        variants=("baseline", "abtt", "sif", "sif_abtt"),
        token_strings=True,
        sif_weights=True,
    )
    resp = token_map_svc.load_token_map(_make_store(npz), 1)

    assert resp is not None
    assert resp.available_variants == ["baseline", "abtt", "sif", "sif_abtt"]
    for method in METHODS:
        assert set(resp.pair_matrices[method]) == {"baseline", "abtt", "sif", "sif_abtt"}
        for variant, matrix in resp.pair_matrices[method].items():
            assert len(matrix) == Q_LEN
            assert len(matrix[0]) == C_LEN
        assert set(resp.top_highlights[method]) == {"baseline", "abtt", "sif", "sif_abtt"}
    assert resp.query_sif_weights == pytest.approx([0.2, 0.3, 0.4, 3.1], rel=1e-5)
    assert resp.candidate_sif_weights == pytest.approx([0.5, 0.5, 2.0], rel=1e-5)


def test_stored_token_strings_are_preferred_over_placeholders(tmp_path: Path) -> None:
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write_npz(npz, variants=("baseline", "abtt"), token_strings=True, sif_weights=False)
    resp = token_map_svc.load_token_map(_make_store(npz), 1)

    assert resp is not None
    assert [t.text for t in resp.query_tokens] == ["de", "et", "in", "est"]
    assert [t.text for t in resp.candidate_tokens] == ["qui", "non", "sponte"]
    # "sponte" is long enough to count as content; "de" is not.
    assert resp.candidate_tokens[2].is_content is True
    assert resp.query_tokens[0].is_content is False


def test_legacy_two_variant_artifact_still_loads(tmp_path: Path, monkeypatch) -> None:
    """Artifacts predating #46/#47 keep working: no sif keys, no token strings."""
    monkeypatch.setattr(token_map_svc, "_try_decode_tokens", lambda ids, slug: None)
    npz = tmp_path / "bowphs_PhilTa" / "example001_pair_example.npz"
    _write_npz(npz, variants=("baseline", "abtt"), token_strings=False, sif_weights=False)
    resp = token_map_svc.load_token_map(_make_store(npz), 1)

    assert resp is not None
    assert resp.available_variants == ["baseline", "abtt"]
    assert resp.query_sif_weights is None
    assert [t.text for t in resp.query_tokens] == ["[0]", "[1]", "[2]", "[3]"]
    for method in METHODS:
        assert set(resp.pair_matrices[method]) == {"baseline", "abtt"}


def test_grouped_cards_report_available_variants(tmp_path: Path) -> None:
    npz = tmp_path / "bowphs_LaTa" / "example001_pair_example.npz"
    _write_npz(
        npz,
        variants=("baseline", "abtt", "sif", "sif_abtt"),
        token_strings=True,
        sif_weights=True,
    )
    grouped = token_map_svc.list_examples_grouped(_make_store(npz))

    card = grouped["by_model"]["bowphs_LaTa"][0]
    assert card["variants_available"] == ["baseline", "abtt", "sif", "sif_abtt"]
    assert card["methods_available"] == ["ig", "bertscore"]


def test_variant_filter_narrows_the_payload_but_not_availability(tmp_path: Path) -> None:
    """?variant= ships one variant's matrices; availability still lists all four."""
    npz = tmp_path / "bowphs_LaTa" / "example001_pair_example.npz"
    _write_npz(
        npz,
        variants=("baseline", "abtt", "sif", "sif_abtt"),
        token_strings=True,
        sif_weights=True,
    )
    resp = token_map_svc.load_token_map(_make_store(npz), 1, variant="sif")

    assert resp is not None
    assert resp.available_variants == ["baseline", "abtt", "sif", "sif_abtt"]
    assert resp.available_methods == list(METHODS)
    for method in METHODS:
        assert set(resp.pair_matrices[method]) == {"sif"}
        assert set(resp.top_highlights[method]) == {"sif"}


def test_method_filter_narrows_the_payload_but_not_availability(tmp_path: Path) -> None:
    npz = tmp_path / "bowphs_LaTa" / "example001_pair_example.npz"
    _write_npz(
        npz,
        variants=("baseline", "abtt", "sif", "sif_abtt"),
        token_strings=True,
        sif_weights=True,
    )
    resp = token_map_svc.load_token_map(_make_store(npz), 1, method="bertscore")

    assert resp is not None
    assert resp.available_methods == list(METHODS)
    assert set(resp.pair_matrices) == {"bertscore"}
    assert set(resp.top_highlights) == {"bertscore"}


def test_method_and_variant_filters_ship_exactly_one_matrix(tmp_path: Path) -> None:
    """The UI renders one cell of the method x variant grid; it should fetch one."""
    npz = tmp_path / "bowphs_LaTa" / "example001_pair_example.npz"
    _write_npz(
        npz,
        variants=("baseline", "abtt", "sif", "sif_abtt"),
        token_strings=True,
        sif_weights=True,
    )
    store = _make_store(npz)

    unfiltered = token_map_svc.load_token_map(store, 1)
    filtered = token_map_svc.load_token_map(store, 1, method="ig", variant="abtt")

    assert unfiltered is not None and filtered is not None
    assert sum(len(v) for v in unfiltered.pair_matrices.values()) == len(METHODS) * 4
    assert sum(len(v) for v in filtered.pair_matrices.values()) == 1
    assert filtered.pair_matrices["ig"]["abtt"] == unfiltered.pair_matrices["ig"]["abtt"]
    # Everything outside the grid is untouched by the filters.
    assert filtered.similarity_matrix == unfiltered.similarity_matrix
    assert filtered.query_tokens == unfiltered.query_tokens
    assert filtered.query_sif_weights == unfiltered.query_sif_weights


def test_filter_for_a_variant_the_artifact_lacks_yields_no_matrices(tmp_path: Path) -> None:
    """A legacy 2-variant artifact asked for 'sif' reports it as unavailable."""
    npz = tmp_path / "bowphs_PhilTa" / "example001_pair_example.npz"
    _write_npz(npz, variants=("baseline", "abtt"), token_strings=True, sif_weights=False)
    resp = token_map_svc.load_token_map(_make_store(npz), 1, variant="sif")

    assert resp is not None
    assert resp.pair_matrices == {}
    assert resp.available_variants == ["baseline", "abtt"]


# --- API surface for the filters (issue #72) --------------------------------


def _api_client(tmp_path: Path) -> TestClient:
    """A signed-in client over a fixture tree that has one IG artifact."""
    unlabelled = tmp_path / "data" / "canon_unlabelled"
    labelled = tmp_path / "data" / "canon_labelled" / "candidate-a"
    predictions = tmp_path / "runs" / "active" / "resubmit" / "unlabelled"
    ig_root = tmp_path / "runs" / "active" / "ig_examples"
    unlabelled.mkdir(parents=True)
    labelled.mkdir(parents=True)
    predictions.mkdir(parents=True)
    ig_root.mkdir(parents=True)
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
            "layer": 12,
            "pooling": "mean",
            "rank1_dir": "candidate-a",
            "rank1_score": 0.9,
        })

    pd.DataFrame([{
        "example_id": 1,
        "model_name": "bowphs/LaTa",
        "bucket": "correct_similar",
        "query_path": "data/canon/X/a.txt",
        "candidate_path": "data/canon/X/b.txt",
        "query_file_id": 0,
        "query_folder_id": "X",
        "candidate_folder_id": "candidate-a",
        "candidate_label": "candidate-a",
        "gold_similar": 1,
        "baseline_pred": 0,
        "abtt_pred": 1,
    }]).to_csv(ig_root / "phase12f_examples.csv", index=False)

    _write_npz(
        ig_root / "artifacts" / "bowphs_LaTa" / "example001_pair_example.npz",
        variants=("baseline", "abtt", "sif", "sif_abtt"),
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
        json={
            "username": "pi",
            "display_name": "PI",
            "password": "correct horse battery staple",
        },
    )
    assert response.status_code == 201
    return client


def test_api_forwards_method_and_variant_filters(tmp_path: Path) -> None:
    client = _api_client(tmp_path)
    try:
        full = client.get("/api/token_map/1").json()
        assert sum(len(v) for v in full["pair_matrices"].values()) == len(METHODS) * 4

        narrow = client.get(
            "/api/token_map/1", params={"method": "ig", "variant": "sif_abtt"}
        )
        assert narrow.status_code == 200
        body = narrow.json()
        assert list(body["pair_matrices"]) == ["ig"]
        assert list(body["pair_matrices"]["ig"]) == ["sif_abtt"]
        # Availability is still the artifact's full contents.
        assert body["available_variants"] == ["baseline", "abtt", "sif", "sif_abtt"]
        assert body["available_methods"] == list(METHODS)

        by_query = client.get(
            "/api/query/0/token_map",
            params={
                "candidate_dir": "candidate-a",
                "model": "bowphs/LaTa",
                "method": "bertscore",
                "variant": "baseline",
            },
        )
        assert by_query.status_code == 200
        assert list(by_query.json()["pair_matrices"]) == ["bertscore"]
        assert list(by_query.json()["pair_matrices"]["bertscore"]) == ["baseline"]
    finally:
        client.__exit__(None, None, None)


def test_api_rejects_unknown_method_or_variant(tmp_path: Path) -> None:
    """A typo must 422 rather than silently return an empty grid."""
    client = _api_client(tmp_path)
    try:
        assert client.get("/api/token_map/1", params={"variant": "raw"}).status_code == 422
        assert client.get("/api/token_map/1", params={"method": "nope"}).status_code == 422
    finally:
        client.__exit__(None, None, None)


def test_slug_to_hf_covers_every_model_with_artifacts() -> None:
    """The tokenizer fallback must cover all six models in the CLAUDE.md table."""
    expected = {
        "bowphs/LaTa",
        "bowphs/PhilTa",
        "sentence-transformers/LaBSE",
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-8B",
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "google/mt5-base",
    }
    assert expected <= set(token_map_svc.SLUG_TO_HF.values())
    for slug, hf_id in token_map_svc.SLUG_TO_HF.items():
        assert slug == hf_id.replace("/", "_")


def test_serving_path_never_executes_remote_tokenizer_code(monkeypatch) -> None:
    """The tokenizer fallback must not opt into remote code execution.

    Every model in SLUG_TO_HF resolves to a built-in fast tokenizer
    (T5TokenizerFast / BertTokenizerFast / Qwen2TokenizerFast), so the fallback
    decoder has no reason to run unpinned code fetched from the Hub. A stub
    ``transformers`` module records the kwargs the service actually passes,
    which keeps the check honest without installing transformers in CI.
    """
    seen: list[dict] = []

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(hf_id, **kwargs):
            seen.append(kwargs)
            return object()

    stub = types.ModuleType("transformers")
    stub.AutoTokenizer = _StubAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", stub)

    token_map_svc._get_tokenizer.cache_clear()
    try:
        for hf_id in token_map_svc.SLUG_TO_HF.values():
            token_map_svc._get_tokenizer(hf_id)
    finally:
        token_map_svc._get_tokenizer.cache_clear()

    assert len(seen) == len(token_map_svc.SLUG_TO_HF)
    for kwargs in seen:
        assert "trust_remote_code" not in kwargs
        assert kwargs == {}


def test_d_sif_is_served_when_the_artifact_carries_a_sif_pooled_cleaner(
    tmp_path: Path,
) -> None:
    """Issue #87: sif_abtt is cleaned in the SIF-pooled space, with its own D.

    LaTa layer 1 is the real case: the mean-pooled sweep picks D=10 and the
    SIF-pooled sweep picks D=3, so a single ``D`` in the payload cannot describe
    both ABTT panels.
    """
    npz = tmp_path / "bowphs_LaTa" / "example001_pair_example.npz"
    _write_npz(
        npz,
        variants=("baseline", "abtt", "sif", "sif_abtt"),
        token_strings=True,
        sif_weights=True,
    )
    with np.load(npz, allow_pickle=False) as handle:
        data = {k: handle[k] for k in handle.files}
    data["pcs_sif"] = np.eye(3, DIM, dtype=np.float32)
    data["mean_vec_sif"] = np.zeros(DIM, dtype=np.float32)
    data["D_sif"] = np.array([3], dtype=np.int32)
    np.savez(npz, **data)

    resp = token_map_svc.load_token_map(_make_store(npz), 1)

    assert resp is not None
    assert resp.D == 10
    assert resp.D_sif == 3


def test_d_sif_is_none_on_artifacts_without_a_sif_pooled_cleaner(tmp_path: Path) -> None:
    """Pre-#87 artifacts have no D_sif key and must still load."""
    npz = tmp_path / "bowphs_PhilTa" / "example001_pair_example.npz"
    _write_npz(
        npz,
        variants=("baseline", "abtt", "sif", "sif_abtt"),
        token_strings=True,
        sif_weights=True,
    )
    resp = token_map_svc.load_token_map(_make_store(npz), 1)

    assert resp is not None
    assert resp.D == 10
    assert resp.D_sif is None
