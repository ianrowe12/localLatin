"""Serving the slim full-corpus attribution artifacts (issue #84).

Three things change on the webapp side when the bulk run lands:

1. artifacts store ``similarity_matrix`` and drop the raw token hidden states;
2. one (query, candidate, model) has several artifacts, one per layer, and the
   requested variant is what picks between them;
3. the bulk rows are excluded from the Attribution gallery, which would
   otherwise list tens of thousands of cards and open every NPZ to do it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_token_map_variants import _write_npz  # noqa: E402

from web.services import token_map_svc  # noqa: E402
from web.services.data_store import DataStore  # noqa: E402

Q_LEN, C_LEN = 4, 3
ALL_VARIANTS = ("baseline", "abtt", "sif", "sif_abtt")


@pytest.fixture(autouse=True)
def _clear_npz_cache():
    token_map_svc._load_npz.cache_clear()
    yield
    token_map_svc._load_npz.cache_clear()


def _write_bulk_npz(path: Path, variants: tuple[str, ...], seed: int = 0) -> np.ndarray:
    """A slim bulk artifact: stored similarity, no hidden states, ig only."""
    rng = np.random.default_rng(seed)
    sim = rng.normal(size=(Q_LEN, C_LEN)).astype(np.float16)
    data: dict[str, np.ndarray] = {
        "example_id": np.array([1_000_000], dtype=np.int64),
        "layer": np.array([1], dtype=np.int32),
        "D": np.array([10], dtype=np.int32),
        "D_sif": np.array([3], dtype=np.int32),
        "similarity_matrix": sim,
        "query_input_ids": np.arange(Q_LEN, dtype=np.int32)[None, :],
        "candidate_input_ids": np.arange(C_LEN, dtype=np.int32)[None, :],
        "query_token_strings": np.asarray(["de", "et", "in", "est"], dtype=np.str_),
        "candidate_token_strings": np.asarray(["qui", "non", "sponte"], dtype=np.str_),
        "query_sif_weights": np.asarray([0.2, 0.3, 0.4, 3.1], dtype=np.float32),
        "candidate_sif_weights": np.asarray([0.5, 0.5, 2.0], dtype=np.float32),
        "artifact_variants": np.asarray(variants, dtype="<U16"),
    }
    for variant in variants:
        data[f"pair_matrix_ig_{variant}"] = rng.normal(size=(Q_LEN, C_LEN)).astype(np.float16)
        data[f"topk_ig_{variant}_query"] = np.arange(2, dtype=np.int32)
        data[f"topk_ig_{variant}_candidate"] = np.arange(2, dtype=np.int32)
        data[f"query_ig_{variant}"] = rng.normal(size=Q_LEN).astype(np.float32)
        data[f"candidate_ig_{variant}"] = rng.normal(size=C_LEN).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return sim


def _row(example_id: int, **overrides) -> dict:
    row = {
        "example_id": example_id,
        "model_name": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "bucket": "unlabelled_bulk",
        "query_path": "data/canon_unlabelled/C1525.56v.3.txt",
        "candidate_path": "data/canon_labelled/CANC.314.3/a.txt",
        "query_file_id": 40,
        "query_folder_id": "",
        "candidate_folder_id": "CANC.314.3",
        "candidate_label": "CANC.314.3",
        "gold_similar": 0,
        "baseline_pred": 0,
        "abtt_pred": 1,
        "query_source": "unlabelled",
        "variants_available": "baseline,abtt,sif,sif_abtt",
        "methods_available": "ig",
        "layer": 1,
    }
    row.update(overrides)
    return row


def _store(rows: list[dict], artifacts: dict[int, Path]) -> DataStore:
    store = DataStore()
    store.ig_examples = pd.DataFrame(rows)
    store.ig_artifact_paths = dict(artifacts)
    return store


# --- 1. stored similarity, no hidden states ---------------------------------


def test_stored_similarity_matrix_is_served_without_hidden_states(tmp_path: Path) -> None:
    npz = tmp_path / "b.npz"
    sim = _write_bulk_npz(npz, ALL_VARIANTS)
    store = _store([_row(1_000_000)], {1_000_000: npz})

    resp = token_map_svc.load_token_map(store, 1_000_000)

    assert resp is not None
    assert len(resp.similarity_matrix) == Q_LEN
    assert len(resp.similarity_matrix[0]) == C_LEN
    assert np.allclose(np.array(resp.similarity_matrix), sim.astype(np.float32), rtol=1e-3)
    assert resp.available_methods == ["ig"]
    assert resp.available_variants == list(ALL_VARIANTS)
    assert resp.D == 10
    assert resp.D_sif == 3
    assert [t.text for t in resp.query_tokens] == ["de", "et", "in", "est"]
    # top_matches and auto_highlights are derived from the stored matrix.
    assert set(resp.top_matches) == {str(i) for i in range(Q_LEN)}
    assert resp.auto_highlights


def test_artifact_with_neither_similarity_nor_hidden_states_is_not_served(tmp_path: Path) -> None:
    npz = tmp_path / "empty.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, layer=np.array([1], dtype=np.int32))
    store = _store([_row(1_000_000)], {1_000_000: npz})

    assert token_map_svc.load_token_map(store, 1_000_000) is None


def test_partial_variant_artifact_still_serves_highlights(tmp_path: Path) -> None:
    """LaBSE L12 carries `sif` alone -- there is no query_ig_abtt to fall back on."""
    npz = tmp_path / "sifonly.npz"
    _write_bulk_npz(npz, ("sif",))
    store = _store([_row(1_000_000, variants_available="sif")], {1_000_000: npz})

    resp = token_map_svc.load_token_map(store, 1_000_000, method="ig", variant="sif")

    assert resp is not None
    assert resp.available_variants == ["sif"]
    assert set(resp.pair_matrices["ig"]) == {"sif"}
    assert resp.auto_highlights


def test_hidden_state_artifacts_still_work(tmp_path: Path) -> None:
    """The 128 paper artifacts carry hidden states and no stored matrix."""
    npz = tmp_path / "legacy.npz"
    _write_npz(npz, variants=ALL_VARIANTS, token_strings=True, sif_weights=True)
    store = _store([_row(1)], {1: npz})

    resp = token_map_svc.load_token_map(store, 1)

    assert resp is not None
    assert len(resp.similarity_matrix) == 4
    assert len(resp.similarity_matrix[0]) == 3


# --- 2. variant picks the artifact ------------------------------------------


def test_variant_selects_the_artifact_built_at_its_layer(tmp_path: Path) -> None:
    """KaLM deploys raw at L22, abtt/sif_abtt at L1 and sif at L23.

    All three artifacts share (file_id, candidate_dir, model). Without the
    variant filter the first row wins and three of the four panels would be
    explained at a layer their ranking never used.
    """
    paths = {}
    for eid, variants in (
        (1_000_001, ("abtt", "sif_abtt")),
        (1_000_002, ("baseline",)),
        (1_000_003, ("sif",)),
    ):
        paths[eid] = tmp_path / f"{eid}.npz"
        _write_bulk_npz(paths[eid], variants, seed=eid)

    store = _store(
        [
            _row(1_000_001, variants_available="abtt,sif_abtt", layer=1),
            _row(1_000_002, variants_available="baseline", layer=22),
            _row(1_000_003, variants_available="sif", layer=23),
        ],
        paths,
    )

    def resolve(variant):
        return token_map_svc.resolve_example_id(
            store, 40, "CANC.314.3",
            "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
            query_source="unlabelled", variant=variant,
        )

    assert resolve("abtt") == 1_000_001
    assert resolve("sif_abtt") == 1_000_001
    assert resolve("baseline") == 1_000_002
    assert resolve("sif") == 1_000_003
    # No variant: unchanged first-match behaviour.
    assert resolve(None) == 1_000_001


def test_variant_with_no_artifact_resolves_to_none(tmp_path: Path) -> None:
    """Strict, like query_source: never serve a wrong-layer artifact instead."""
    npz = tmp_path / "a.npz"
    _write_bulk_npz(npz, ("sif",))
    store = _store([_row(1_000_003, variants_available="sif")], {1_000_003: npz})

    assert (
        token_map_svc.resolve_example_id(
            store, 40, "CANC.314.3", None, query_source="unlabelled", variant="baseline"
        )
        is None
    )


def test_blank_variants_available_means_every_variant(tmp_path: Path) -> None:
    """Paper rows predate the column; their artifacts carry all four."""
    npz = tmp_path / "a.npz"
    _write_bulk_npz(npz, ALL_VARIANTS)
    for cell in ("", np.nan):
        store = _store([_row(7, variants_available=cell)], {7: npz})
        assert (
            token_map_svc.resolve_example_id(
                store, 40, "CANC.314.3", None, query_source="unlabelled", variant="sif"
            )
            == 7
        )


def test_variants_available_column_may_be_absent(tmp_path: Path) -> None:
    npz = tmp_path / "a.npz"
    _write_bulk_npz(npz, ALL_VARIANTS)
    row = _row(7)
    del row["variants_available"]
    store = _store([row], {7: npz})

    assert (
        token_map_svc.resolve_example_id(
            store, 40, "CANC.314.3", None, query_source="unlabelled", variant="abtt"
        )
        == 7
    )


# --- 3. the gallery excludes the bulk rows ----------------------------------


def test_gallery_excludes_bulk_rows(tmp_path: Path) -> None:
    bulk = tmp_path / "bulk.npz"
    paper = tmp_path / "paper.npz"
    _write_bulk_npz(bulk, ALL_VARIANTS)
    _write_npz(paper, variants=ALL_VARIANTS, token_strings=True, sif_weights=True)

    store = _store(
        [
            _row(7, bucket="correct_similar", model_name="bowphs/LaTa",
                 variants_available="", methods_available=""),
            _row(1_000_000, bucket="unlabelled_bulk"),
        ],
        {7: paper, 1_000_000: bulk},
    )
    # The artifact paths' parent dirs stand in for the model slug.
    grouped = token_map_svc.list_examples_grouped(store)
    listed = [c["example_id"] for cards in grouped["by_model"].values() for c in cards]

    assert listed == [7]
    assert [e.example_id for e in token_map_svc.list_examples(store)] == [7]
    # ...but a bulk row is still reachable by resolution, which is the point.
    assert (
        token_map_svc.resolve_example_id(
            store, 40, "CANC.314.3",
            "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
            query_source="unlabelled", variant="sif_abtt",
        )
        == 1_000_000
    )


def test_grouped_listing_uses_csv_columns_without_opening_the_npz(tmp_path: Path) -> None:
    """The column is authoritative: a missing file must not blank the card."""
    missing = tmp_path / "not-written.npz"
    store = _store(
        [_row(7, bucket="correct_similar", methods_available="ig,bertscore",
              variants_available="baseline,abtt")],
        {7: missing},
    )

    grouped = token_map_svc.list_examples_grouped(store)
    cards = [c for cards in grouped["by_model"].values() for c in cards]

    assert len(cards) == 1
    assert cards[0]["methods_available"] == ["ig", "bertscore"]
    assert cards[0]["variants_available"] == ["baseline", "abtt"]
