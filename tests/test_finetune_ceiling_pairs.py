"""Guards for the two ways the fine-tuning ceiling (issue #123) could quietly lie.

Both are silent failures: the job still runs, still produces a number, and the
number is wrong in the direction that flatters the ceiling.

* **Leaked dev.** Model selection is only meaningful if the dev slice is held
  out by *directory*. Two files from the same directory are a positive pair, so
  a file-level split would train on exactly the supervision it then measures.
* **False in-batch negatives.** The InfoNCE loss labels every other pair in the
  batch a negative. Two pairs from the same directory in one batch make the
  objective push apart texts that belong together, which shows up as a worse
  ceiling rather than as an error.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

pytest.importorskip("torch", reason="finetune_lata_ceiling imports torch")
pytest.importorskip("transformers", reason="finetune_lata_ceiling imports transformers")

from finetune_lata_ceiling import build_pairs, batch_pairs_by_round  # noqa: E402


def make_split(sizes: dict[str, int], test_dirs: int = 2) -> pd.DataFrame:
    """A minimal split frame: `sizes` maps directory name to its train file count."""
    rows = []
    for folder, n in sizes.items():
        for i in range(n):
            rows.append({"folder_id": folder, "path": f"{folder}/{i}.txt", "split": "train"})
    for t in range(test_dirs):
        rows.append({"folder_id": f"T{t}", "path": f"T{t}/0.txt", "split": "test"})
    return pd.DataFrame(rows)


# --- the dev carve ---------------------------------------------------------


def test_dev_directories_contribute_no_training_pair() -> None:
    split = make_split({f"D{i}": 3 for i in range(20)})
    pairs = build_pairs(split, dev_dir_frac=0.15, seed=42)

    dev = set(pairs.dev_dirs)
    assert dev, "a 15% carve of 20 directories must hold at least one out"
    assert dev.isdisjoint(pairs.fit_dirs)
    assert dev.isdisjoint(pairs.train_pair_dirs)


def test_dev_rows_are_exactly_the_files_of_the_dev_directories() -> None:
    split = make_split({f"D{i}": 3 for i in range(20)})
    pairs = build_pairs(split, dev_dir_frac=0.15, seed=42)

    dev_rows_folders = sorted({split["folder_id"].values[r] for r in pairs.dev_rows})
    assert dev_rows_folders == sorted(pairs.dev_dirs)
    assert len(pairs.dev_rows) == 3 * len(pairs.dev_dirs)


def test_singletons_and_test_files_never_enter_training() -> None:
    split = make_split({"A": 4, "B": 2, "S1": 1, "S2": 1})
    pairs = build_pairs(split, dev_dir_frac=0.0, seed=7)

    assert pairs.n_train_multi_dirs == 2
    assert set(pairs.train_pair_dirs) <= {"A", "B"}
    train_rows = {r for pair in pairs.train_pairs for r in pair}
    assert all(split["split"].values[r] == "train" for r in train_rows)


def test_pair_count_matches_the_combinatorics() -> None:
    # 4 files -> 6 pairs, 2 files -> 1 pair; dev_dir_frac=0 keeps both.
    pairs = build_pairs(make_split({"A": 4, "B": 2}), dev_dir_frac=0.0, seed=1)
    assert pairs.n_all_train_pairs == 7
    assert len(pairs.train_pairs) == 7


def test_the_carve_is_reproducible_from_the_seed() -> None:
    split = make_split({f"D{i}": 2 for i in range(40)})
    a = build_pairs(split, dev_dir_frac=0.15, seed=42)
    b = build_pairs(split, dev_dir_frac=0.15, seed=42)
    c = build_pairs(split, dev_dir_frac=0.15, seed=43)
    assert a.dev_dirs == b.dev_dirs
    assert a.train_pairs == b.train_pairs
    assert a.dev_dirs != c.dev_dirs


# --- in-batch negatives ----------------------------------------------------


def test_no_batch_repeats_a_directory() -> None:
    """The invariant the loss depends on, on a corpus built to break it."""
    # D0 alone owns 21 pairs, so a naive shuffle would collide constantly.
    split = make_split({"D0": 7, **{f"D{i}": 3 for i in range(1, 15)}})
    pairs = build_pairs(split, dev_dir_frac=0.0, seed=42)

    for batch_size in (2, 4, 8, 16):
        batches = batch_pairs_by_round(
            pairs.train_pairs, pairs.train_pair_dirs, batch_size, random.Random(0)
        )
        assert batches, "expected at least one batch"
        for batch in batches:
            dirs = [pairs.train_pair_dirs[i] for i in batch]
            assert len(set(dirs)) == len(dirs)
            assert 2 <= len(batch) <= batch_size


def test_batches_cover_essentially_every_pair() -> None:
    """On a corpus shaped like the real one, at most one pair may be stranded.

    The real train split is 162 directories of 2 to 7 files each, none of them
    dominant, which is the regime this asserts. A directory holding more pairs
    than the rest of the corpus can supply as batch-mates is a different case,
    covered below.
    """
    split = make_split({f"D{i}": 2 + (i % 4) for i in range(60)})
    pairs = build_pairs(split, dev_dir_frac=0.0, seed=42)
    batches = batch_pairs_by_round(
        pairs.train_pairs, pairs.train_pair_dirs, 16, random.Random(0)
    )

    covered = {i for batch in batches for i in batch}
    # A final leftover has no negative to pair with and is dropped rather than
    # trained on a degenerate batch.
    assert len(pairs.train_pairs) - len(covered) <= 1


def test_a_dominant_directory_is_drained_as_far_as_negatives_allow() -> None:
    """One huge directory cannot be fully used, and must not be faked.

    Each batch spends one slot on the dominant directory and the rest on other
    directories, so the dominant directory's usable pairs are capped by what
    the rest of the corpus can supply. Dropping the remainder is correct; the
    alternative would be batches whose "negatives" are same-directory positives.
    """
    split = make_split({"D0": 7, **{f"D{i}": 3 for i in range(1, 15)}})
    pairs = build_pairs(split, dev_dir_frac=0.0, seed=42)
    batches = batch_pairs_by_round(
        pairs.train_pairs, pairs.train_pair_dirs, 8, random.Random(0)
    )

    used_per_dir: dict[str, int] = {}
    for batch in batches:
        for i in batch:
            d = pairs.train_pair_dirs[i]
            used_per_dir[d] = used_per_dir.get(d, 0) + 1
    # Every small directory is fully consumed; only the dominant one is capped.
    for d in {f"D{i}" for i in range(1, 15)} & set(pairs.fit_dirs):
        assert used_per_dir.get(d, 0) == 3
    assert used_per_dir.get("D0", 0) < 21


def test_batching_is_reproducible_from_its_rng() -> None:
    split = make_split({f"D{i}": 3 for i in range(10)})
    pairs = build_pairs(split, dev_dir_frac=0.0, seed=42)
    first = batch_pairs_by_round(pairs.train_pairs, pairs.train_pair_dirs, 4, random.Random(0))
    again = batch_pairs_by_round(pairs.train_pairs, pairs.train_pair_dirs, 4, random.Random(0))
    assert first == again


def test_a_single_pair_yields_no_batch() -> None:
    """One pair has no in-batch negative, so it must not become a training step."""
    pairs = build_pairs(make_split({"A": 2}), dev_dir_frac=0.0, seed=0)
    assert batch_pairs_by_round(
        pairs.train_pairs, pairs.train_pair_dirs, 16, random.Random(0)
    ) == []
