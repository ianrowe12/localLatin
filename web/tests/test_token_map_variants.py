"""Token-map service: 4-variant plumbing and stored token strings (issues #46/#47)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
