"""Row alignment in the fine-tuning ceiling script (issue #113, #123).

Two ways the ceiling run could quietly pair a vector with the wrong file:

* ``parity_check`` encodes texts in split-CSV order and diffs them against the
  paper's cached matrix, which is frozen in corpus-walk order. After a label
  correction the two differ by a permutation, so a positional diff reports a
  parity failure for embeddings that are in fact bit-identical.
* ``extract_and_save`` writes its matrices in split order. Without a manifest
  beside them, a resolver falls back to the ``row_order.csv`` at an ancestor
  bases root, which records a *different* order, and permutes correct rows into
  wrong ones.

Both tests build a cache whose order is a real permutation of the split's.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

pytest.importorskip("torch", reason="finetune_lata_ceiling imports torch")
pytest.importorskip("transformers", reason="finetune_lata_ceiling imports transformers")

from embedding_alignment import AlignmentResolver  # noqa: E402

import finetune_lata_ceiling as ceiling  # noqa: E402

# The cache was written before the relabelling; the split is sorted by
# (folder_id, filename) after it, so c and d swap places.
CACHE_ORDER = ["a.txt", "b.txt", "c.txt", "d.txt"]
SPLIT_ORDER = ["a.txt", "b.txt", "d.txt", "c.txt"]
DIM = 4


def vector_for(filename: str) -> np.ndarray:
    """One distinct constant row per file, so a mispairing is visible in values."""
    return np.full(DIM, float(CACHE_ORDER.index(filename) + 1), dtype=np.float32)


def matrix_in(order: Sequence[str]) -> np.ndarray:
    return np.stack([vector_for(name) for name in order])


def split_frame(order: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": range(len(order)),
            "folder_id": [f"Dir{i}" for i in range(len(order))],
            "filename": list(order),
            "path": [f"data/canon_labelled/Dir{i}/{n}" for i, n in enumerate(order)],
            "split": ["train"] * len(order),
        }
    )


class StubEncoder:
    """Encodes by filename, in the order the texts were handed over.

    The real ``Encoder`` needs LaTa weights; all ``parity_check`` and
    ``extract_and_save`` ask of it is ``encode_layers``, and the point of both
    tests is which *row* an embedding lands in, not what is in it.
    """

    def __init__(self, order: Sequence[str]) -> None:
        self.order = list(order)

    def encode_layers(
        self, texts: Sequence[str], layers: Sequence[int], batch_size: int,
        log_every: int = 0,
    ) -> Dict[int, np.ndarray]:
        assert len(texts) == len(self.order)
        return {int(layer): matrix_in(self.order) for layer in layers}


def write_manifest(run_dir: Path, order: Sequence[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "file_id": range(len(order)),
            "path": [f"canon_labelled/Dir{i}/{n}" for i, n in enumerate(order)],
        }
    ).to_csv(run_dir / "meta.csv", index=False)


def baseline_cache(tmp_path: Path, layers: Sequence[int]) -> Path:
    """The paper's LaTa cache, written in CACHE_ORDER with its own manifest."""
    root = tmp_path / "resubmit_bases"
    run_dir = ceiling.bases_dir(root, "bowphs/LaTa")
    write_manifest(run_dir, CACHE_ORDER)
    for layer in layers:
        np.save(run_dir / f"hidden_layer{layer}_embeddings.npy", matrix_in(CACHE_ORDER))
    return root


# --- parity_check ----------------------------------------------------------


def test_parity_check_is_clean_when_the_cache_is_merely_permuted(tmp_path):
    """A relabelled split must not read as an extraction mismatch."""
    root = baseline_cache(tmp_path, [1])
    split = split_frame(SPLIT_ORDER)

    report = ceiling.parity_check(
        StubEncoder(SPLIT_ORDER), ["text"] * 4, [1], root, 2, AlignmentResolver(split)
    )

    assert report["parity_layer1_max_abs_diff"] == pytest.approx(0.0)
    assert report["parity_layer1_mean_cosine"] == pytest.approx(1.0)


def test_parity_check_still_sees_a_genuine_extraction_difference(tmp_path):
    """The guard must not become a rubber stamp: real drift still shows up."""
    root = baseline_cache(tmp_path, [1])
    split = split_frame(SPLIT_ORDER)

    class DriftingEncoder(StubEncoder):
        def encode_layers(self, texts, layers, batch_size, log_every=0):
            embs = super().encode_layers(texts, layers, batch_size, log_every)
            return {layer: emb + 0.5 for layer, emb in embs.items()}

    report = ceiling.parity_check(
        DriftingEncoder(SPLIT_ORDER), ["text"] * 4, [1], root, 2,
        AlignmentResolver(split),
    )

    assert report["parity_layer1_max_abs_diff"] == pytest.approx(0.5)


# --- extract_and_save ------------------------------------------------------


def test_extracted_cache_carries_its_own_row_order(tmp_path):
    """The written manifest must describe the split order, not an ancestor's."""
    bases_root = tmp_path / "ft_bases"
    ft_dir = ceiling.bases_dir(bases_root, "bowphs/LaTa-ft")
    # A bases root left over from the paper's extraction, in the *other* order.
    (bases_root / "phase9_bases").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filename": CACHE_ORDER}).to_csv(
        bases_root / "phase9_bases" / "row_order.csv", index=False
    )
    split = split_frame(SPLIT_ORDER)

    ceiling.extract_and_save(StubEncoder(SPLIT_ORDER), ["text"] * 4, [1], ft_dir, 2, split)

    manifest = pd.read_csv(ft_dir / "meta.csv")
    assert [Path(p).name for p in manifest["path"]] == SPLIT_ORDER

    loaded = AlignmentResolver(split).load(ft_dir / "hidden_layer1_embeddings.npy")
    for row, name in enumerate(SPLIT_ORDER):
        assert list(loaded[row]) == list(vector_for(name))


def test_extract_refuses_a_split_that_does_not_describe_the_texts(tmp_path):
    ft_dir = ceiling.bases_dir(tmp_path / "ft_bases", "bowphs/LaTa-ft")
    with pytest.raises(ValueError, match="split rows"):
        ceiling.extract_and_save(
            StubEncoder(SPLIT_ORDER), ["text"] * 4, [1], ft_dir, 2,
            split_frame(SPLIT_ORDER[:3]),
        )


def test_row_manifest_needs_a_name_column(tmp_path):
    bare: List[str] = ["file_id"]
    with pytest.raises(ValueError, match="row manifest"):
        ceiling.write_row_manifest(tmp_path, pd.DataFrame({c: [0, 1] for c in bare}))
