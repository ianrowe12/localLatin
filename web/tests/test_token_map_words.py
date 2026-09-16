"""The token map follows the selected variant, and highlights whole words.

Issue #211 part 1, plus item 2 of issue #216. Two defects, one service:

1. the auto-highlights were taken from a fixed preference order with `abtt`
   first, whatever variant the reviewer had selected, and `abtt` and `sif_abtt`
   agree on only 44 to 78 percent of the top five slots per model;
2. the highlighted unit was a subword piece, so `Episcopus` reached the reviewer
   as `Epi` + `##scop` + `##us`.

Both numbers are from ``docs/research/prefix_attribution_analysis.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from web.services import token_map_svc  # noqa: E402
from web.services.data_store import DataStore  # noqa: E402

Q_TOKENS = ["[CLS]", "Epi", "##scop", "##us", "aut", "pres", "##biter", "[SEP]"]
C_TOKENS = ["[CLS]", "sacer", "##dotes", "et", "ministri", "[SEP]"]
Q_LEN, C_LEN = len(Q_TOKENS), len(C_TOKENS)

# One clear winner per variant, and they are different tokens: whichever token
# the highlights name says which vector the service actually read.
IG_BY_VARIANT = {
    # `##scop`, i.e. the middle of "Episcopus"
    "abtt": [0.0, 0.1, 0.9, 0.05, 0.0, 0.0, 0.0, 0.0],
    # `pres`, i.e. the start of "presbiter"
    "sif_abtt": [0.0, 0.0, 0.05, 0.0, 0.1, 0.9, 0.2, 0.0],
    "baseline": [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
}


@pytest.fixture(autouse=True)
def _clear_npz_cache():
    token_map_svc._load_npz.cache_clear()
    yield
    token_map_svc._load_npz.cache_clear()


def _write(
    path: Path,
    *,
    variants: tuple[str, ...] = ("abtt", "sif_abtt"),
    marc_variants: tuple[str, ...] = (),
    marc_nan_fraction: float = 0.0,
) -> None:
    rng = np.random.default_rng(3)
    data: dict[str, np.ndarray] = {
        "example_id": np.array([1], dtype=np.int64),
        "layer": np.array([11], dtype=np.int32),
        "D": np.array([10], dtype=np.int32),
        "similarity_matrix": rng.random((Q_LEN, C_LEN)).astype(np.float32),
        "query_token_strings": np.asarray(Q_TOKENS, dtype=np.str_),
        "candidate_token_strings": np.asarray(C_TOKENS, dtype=np.str_),
    }
    for variant in variants:
        data[f"query_ig_{variant}"] = np.asarray(IG_BY_VARIANT[variant], dtype=np.float32)
        data[f"candidate_ig_{variant}"] = rng.random(C_LEN).astype(np.float32)
        data[f"pair_matrix_ig_{variant}"] = rng.random((Q_LEN, C_LEN)).astype(np.float32)
    for variant in marc_variants:
        q = rng.random(Q_LEN).astype(np.float32)
        c = rng.random(C_LEN).astype(np.float32)
        if marc_nan_fraction:
            n_nan = int(round(marc_nan_fraction * Q_LEN))
            q[:n_nan] = np.nan
            c[: int(round(marc_nan_fraction * C_LEN))] = np.nan
        data[f"q_mask_retrieval_mark_{variant}"] = q
        data[f"c_mask_retrieval_mark_{variant}"] = c
        data[f"pair_matrix_retrieval_mark_{variant}"] = rng.random(
            (Q_LEN, C_LEN)
        ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def _store(npz: Path, *, with_texts: bool = False) -> DataStore:
    store = DataStore()
    store.ig_examples = pd.DataFrame([
        {
            "example_id": 1,
            "model_name": "sentence-transformers/LaBSE",
            "bucket": "correct_similar",
            "query_path": "data/canon_labelled/Can.apost.7/q.txt",
            "candidate_path": "data/canon_labelled/Can.apost.7/c.txt",
            "query_file_id": 11,
            "query_folder_id": "Can.apost.7",
            "candidate_folder_id": "Can.apost.7",
            "candidate_label": "Can.apost.7",
            "query_source": "labelled",
            "gold_similar": 1,
            "baseline_pred": 0,
            "abtt_pred": 1,
        }
    ])
    store.ig_artifact_paths = {1: npz}
    if with_texts:
        store.labelled_texts = {
            "Can.apost.7": {
                "q.txt": "Episcopus aut presbiter aut diaconus",
                "c.txt": "sacerdotes et ministri altaris",
            }
        }
    return store


def top_word(resp) -> str:
    """The word the first auto-highlight points at."""
    assert resp.auto_highlights
    idx = resp.auto_highlights[0].word_idx
    assert idx is not None
    return resp.query_words[idx].text


# --- 1. the highlight follows the selected variant (issue #216 item 2) -------


def test_requesting_sif_abtt_serves_the_sif_abtt_vector(tmp_path: Path) -> None:
    """The regression test for the fixed `abtt`-first preference order."""
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz, variants=("abtt", "sif_abtt"))

    resp = token_map_svc.load_token_map(_store(npz), 1, method="ig", variant="sif_abtt")

    assert resp is not None
    assert resp.variant_requested == "sif_abtt"
    assert resp.variant_served == "sif_abtt"
    assert resp.attribution_source == "ig"
    assert resp.query_attribution == pytest.approx(IG_BY_VARIANT["sif_abtt"], abs=1e-6)
    # The top slot is `pres`, from sif_abtt -- not `##scop`, from abtt.
    assert resp.auto_highlights[0].query_idx == Q_TOKENS.index("pres")


def test_requesting_abtt_on_the_same_artifact_serves_abtt(tmp_path: Path) -> None:
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz, variants=("abtt", "sif_abtt"))

    resp = token_map_svc.load_token_map(_store(npz), 1, method="ig", variant="abtt")

    assert resp.variant_served == "abtt"
    assert resp.auto_highlights[0].query_idx == Q_TOKENS.index("##scop")


def test_a_missing_variant_falls_back_but_says_so(tmp_path: Path) -> None:
    """An artifact built at another layer carries only its own variants."""
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz, variants=("abtt",))

    resp = token_map_svc.load_token_map(_store(npz), 1, method="ig", variant="sif_abtt")

    assert resp.variant_requested == "sif_abtt"
    assert resp.variant_served == "abtt"
    assert resp.variant_served != resp.variant_requested


def test_marc_is_used_where_there_is_no_ig_for_the_variant(tmp_path: Path) -> None:
    npz = tmp_path / "bowphs_LaTa" / "example001_pair_example.npz"
    _write(npz, variants=("abtt",), marc_variants=("baseline",))

    resp = token_map_svc.load_token_map(_store(npz), 1, variant="baseline")

    assert resp.variant_served == "baseline"
    assert resp.attribution_source == "retrieval_mark"


def test_a_mostly_nan_marc_mask_is_rejected_rather_than_ranked(tmp_path: Path) -> None:
    """The deployed Qwen gallery masks are NaN at 87 to 97 percent of positions.

    `np.argsort` would sort those NaNs to the top and outline padding.
    """
    npz = tmp_path / "Qwen_Qwen3-Embedding-0.6B" / "example016_pair_example.npz"
    _write(npz, variants=("abtt",), marc_variants=("baseline",), marc_nan_fraction=0.9)

    resp = token_map_svc.load_token_map(_store(npz), 1, variant="baseline")

    assert resp.attribution_source == "ig"
    assert resp.variant_served == "abtt"
    assert all(np.isfinite(v) for v in resp.query_attribution)


# --- 2. word-level display (issue #211) -------------------------------------


def test_word_spans_group_the_pieces_and_keep_them(tmp_path: Path) -> None:
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz)

    resp = token_map_svc.load_token_map(_store(npz), 1, method="ig", variant="abtt")

    assert [w.text for w in resp.query_words] == [
        "Episcopus", "aut", "presbiter",
    ]
    assert resp.query_words[0].piece_indices == [1, 2, 3]
    # The pieces are still there, so the frontend can toggle without refetching.
    assert [t.text for t in resp.query_tokens] == Q_TOKENS
    assert resp.word_segmentation == "markers"


def test_a_fragmented_word_is_one_highlight(tmp_path: Path) -> None:
    """`Epi` + `##scop` + `##us` reaches the reviewer as `Episcopus`."""
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz)

    resp = token_map_svc.load_token_map(_store(npz), 1, method="ig", variant="abtt")

    assert top_word(resp) == "Episcopus"
    # 0.1 + 0.9 + 0.05, the mass the piece view had scattered over three spans.
    assert resp.query_words[0].score == pytest.approx(1.05, abs=1e-5)
    assert resp.query_words[0].score_pos == pytest.approx(1.05, abs=1e-5)
    assert resp.query_words[0].score_neg == pytest.approx(0.0)


def test_word_aggregation_max_is_available(tmp_path: Path) -> None:
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz)

    resp = token_map_svc.load_token_map(
        _store(npz), 1, method="ig", variant="abtt", word_aggregation="max"
    )

    assert resp.word_aggregation == "max"
    assert resp.query_words[0].score == pytest.approx(0.9, abs=1e-5)


def test_word_matrices_have_word_shape(tmp_path: Path) -> None:
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz)

    resp = token_map_svc.load_token_map(_store(npz), 1, method="ig", variant="abtt")

    n_q, n_c = len(resp.query_words), len(resp.candidate_words)
    assert (len(resp.word_similarity_matrix), len(resp.word_similarity_matrix[0])) == (n_q, n_c)
    word_ig = resp.word_pair_matrices["ig"]["abtt"]
    assert (len(word_ig), len(word_ig[0])) == (n_q, n_c)
    # The piece grids are untouched: the aggregation is a view, not a rewrite.
    assert len(resp.similarity_matrix) == Q_LEN
    assert len(resp.pair_matrices["ig"]["abtt"]) == Q_LEN


def test_the_original_text_supplies_the_words_when_the_store_has_it(tmp_path: Path) -> None:
    """Words come from the file, so punctuation and case are the reader's."""
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz)

    resp = token_map_svc.load_token_map(
        _store(npz, with_texts=True), 1, method="ig", variant="abtt"
    )

    assert resp.word_segmentation == "text"
    assert [w.text for w in resp.query_words][:3] == ["Episcopus", "aut", "presbiter"]
    assert resp.query_words[0].piece_indices == [1, 2, 3]
    assert top_word(resp) == "Episcopus"


def test_the_word_grids_stop_where_the_model_stopped_reading(tmp_path: Path) -> None:
    """A long file keeps all its words; the matrices keep only the read ones.

    The frontend walks the whole word list against the text on screen, so the
    list must stay whole. The grids are a different matter: a model truncates
    at its maximum length, and sizing a word x word square by the file's word
    count fills it with zeros. On the deployed example 1002908 that was 1,151
    query words for 256 pieces, and 8.23 MB of response for 161 scored words.
    """
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz)
    store = _store(npz, with_texts=True)
    tail = " ".join(f"filler{i:03d}" for i in range(500))
    store.labelled_texts["Can.apost.7"] = {
        "q.txt": f"Episcopus aut presbiter {tail}",
        "c.txt": f"sacerdotes et ministri {tail}",
    }

    resp = token_map_svc.load_token_map(store, 1, method="ig", variant="abtt")

    # Whole, so `alignWordsToTokens` can still walk it.
    assert resp.word_segmentation == "text"
    assert len(resp.query_words) == 503
    assert len(resp.candidate_words) == 503
    # Cut at the last word carrying a piece.
    assert resp.query_words_scored == 3
    assert resp.candidate_words_scored == 3
    assert all(not w.piece_indices for w in resp.query_words[3:])

    grids = [resp.word_similarity_matrix, resp.word_pair_matrices["ig"]["abtt"]]
    for grid in grids:
        assert len(grid) == resp.query_words_scored
        assert all(len(row) == resp.candidate_words_scored for row in grid)

    # Nothing the highlights point at falls off the end of the grids. A
    # special token belongs to no word at all and says so with None.
    named = [h.word_idx for h in resp.auto_highlights or [] if h.word_idx is not None]
    assert named
    assert all(idx < resp.query_words_scored for idx in named)


def test_artifacts_are_never_written(tmp_path: Path) -> None:
    """The aggregation is a display step: the NPZ on disk must not change."""
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz)
    before = npz.read_bytes()

    token_map_svc.load_token_map(_store(npz), 1, method="ig", variant="abtt")

    assert npz.read_bytes() == before


# --- 3. through the endpoint the frontend calls -----------------------------


def _client(store: DataStore):
    from fastapi.testclient import TestClient

    from web.app import create_app
    from web.dependencies import get_current_user, get_store
    from web.models import UserPublic

    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_current_user] = lambda: UserPublic(
        id=1,
        username="smoke",
        display_name="Smoke",
        role="pi_admin",
        approval_status="approved",
        must_change_password=False,
    )
    return TestClient(app)


def test_endpoint_reports_the_served_variant_and_the_words(tmp_path: Path) -> None:
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz, variants=("abtt", "sif_abtt"))

    body = _client(_store(npz)).get("/api/token_map/1?method=ig&variant=sif_abtt").json()

    assert body["variant_requested"] == "sif_abtt"
    assert body["variant_served"] == "sif_abtt"
    assert [w["text"] for w in body["query_words"]] == ["Episcopus", "aut", "presbiter"]
    assert max(len(w["piece_indices"]) for w in body["query_words"]) > 1


def test_endpoint_rejects_an_unknown_aggregation(tmp_path: Path) -> None:
    """A typo must be a 422, not a silently different display."""
    npz = tmp_path / "sentence-transformers_LaBSE" / "example001_pair_example.npz"
    _write(npz)
    client = _client(_store(npz))

    assert client.get("/api/token_map/1?variant=abtt&word_aggregation=max").status_code == 200
    assert client.get("/api/token_map/1?variant=abtt&word_aggregation=mean").status_code == 422
