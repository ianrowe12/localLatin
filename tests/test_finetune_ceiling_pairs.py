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
* **A dev pool too small to choose with.** Model selection reads directory
  accuracy at rank 1 over 71 files, where one file is 1.4 points. Issue #194
  found the exact-equality tie-break silently returning the PRE-TRAINED encoder
  for a model that starts at 71 of 71, so the last group of tests replays both
  runs' measured dev curves.

Both functions live in ``src/finetune_pairs.py``, which imports nothing heavier
than pandas, so every test here runs in CI rather than being skipped for want of
torch.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# `src/finetune_pairs` is deliberately torch-free so these guards execute on a
# clean CI checkout. Importing them from the training CLI would drag in torch
# and transformers, and the whole module would be skipped -- which is exactly
# the failure these tests exist to prevent going unnoticed.
import finetune_pairs as fp  # noqa: E402
from finetune_pairs import build_pairs, batch_pairs_by_round  # noqa: E402


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


# --- checkpoint selection on a small dev pool (#194) ------------------------

# The three runs' measured dev curves, as (epoch, dir_acc@1, AUROC). They are
# literals because runs/ is gitignored, and they are the whole point of these
# tests: the selection rule has to be checked against what actually happened,
# not against a curve invented to suit it.
N_DEV_FILES = 71

LATA_DEV_CURVE = [
    (0, 0.9295774647887324, 0.9467598682150151),
    (1, 0.9295774647887324, 0.9629323411878187),
    (2, 0.9436619718309859, 0.9744948451025343),
    (3, 0.9718309859154930, 0.9794242549513322),
    (4, 0.9859154929577465, 0.9813596903303393),
    (5, 0.9859154929577465, 0.9825748180440202),
    (6, 0.9859154929577465, 0.9834391872424117),
    (7, 0.9859154929577465, 0.9836584113144675),
]

# Qwen3-0.6B starts at 71 of 71: the pool has no headroom left to measure in.
QWEN_DEV_CURVE = [
    (0, 1.0000000000000000, 0.9659952146516843),
    (1, 0.9859154929577465, 0.9991043130770291),
    (2, 0.9859154929577465, 0.9996241873050471),
    (3, 0.9859154929577465, 0.9997807759279441),
    (4, 0.9859154929577465, 0.9997619852931966),
]


# KaLM-mini is the third shape the window has to handle (#210): the pre-trained
# encoder misses two files, so epoch 1 wins on accuracy outright, and then
# epochs 2-4 sit ONE file below the incumbent. Neither curve above covers a
# candidate below the incumbent by exactly the pool's resolution.
KALM_DEV_CURVE = [
    (0, 0.9718309859154930, 0.9567001139965174),
    (1, 1.0000000000000000, 0.9996993498440377),
    (2, 0.9859154929577465, 0.9994049632329914),
    (3, 0.9859154929577465, 0.9995239705863931),
    (4, 0.9859154929577465, 0.9995114434965613),
]


def points(curve):
    return [fp.DevPoint(*row) for row in curve]


def test_dev_resolution_is_one_file():
    assert fp.dev_resolution(71) == pytest.approx(1 / 71)


def test_lata_selection_is_unchanged_by_the_wider_tie_window():
    """The published LaTa ceiling must not move when the rule is relaxed.

    Its curve climbs by two files at epoch 3, which is a real gain either way,
    and its last four epochs are exact ties that AUROC already decided.
    """
    assert fp.select_checkpoint(points(LATA_DEV_CURVE), N_DEV_FILES).epoch == 7


def test_a_saturated_dev_pool_no_longer_vetoes_a_trained_checkpoint():
    """Qwen3-0.6B's run selected epoch 0, the pre-trained encoder, under the
    exact-equality rule: one dev file out of 71 moved the wrong way, and that
    1.4-point difference outranked an AUROC gain from 0.966 to 0.9998. One file
    is the pool's whole resolution, so it is a tie, and AUROC decides."""
    assert fp.select_checkpoint(points(QWEN_DEV_CURVE), N_DEV_FILES).epoch == 3


def test_kalm_selection_is_the_trained_epoch_that_tops_the_pool():
    """KaLM-mini's published ceiling is extracted from epoch 1, and this pins it.

    The curve exercises the tie window from the side neither other run reaches.
    Epoch 1 gains two files over the pre-trained encoder (69/71 to 71/71), which
    is larger than the window and so wins on accuracy outright. Epochs 2 to 4
    then sit exactly one file *below* the incumbent, which the window makes a
    tie rather than a loss, so AUROC decides and epoch 1 keeps it on 0.99970
    against 0.99940 to 0.99952. Under the pre-#194 exact-equality rule the
    outcome would be the same here; what this curve guards is that widening the
    window did not hand the run to a later, worse epoch.
    """
    assert fp.select_checkpoint(points(KALM_DEV_CURVE), N_DEV_FILES).epoch == 1


def test_kalm_epochs_below_the_incumbent_are_ties_that_lose_on_auroc():
    """The three post-peak epochs must be reachable by the tie-break and lose it.

    If the window were narrower they would be outright losses, and if AUROC were
    not consulted they would be indistinguishable from epoch 1. Both halves are
    asserted, so a change to either rule fails here rather than silently moving
    which checkpoint the paper's KaLM-mini rows come from.
    """
    best = fp.DevPoint(*KALM_DEV_CURVE[1])
    for row in KALM_DEV_CURVE[2:]:
        candidate = fp.DevPoint(*row)
        # One file below the incumbent: inside the window, so not an outright loss.
        assert best.dir_acc_at_1 - candidate.dir_acc_at_1 == pytest.approx(1 / 71)
        # ... and it loses anyway, on the tie-break.
        assert candidate.aucroc < best.aucroc
        assert not fp.is_better_checkpoint(candidate, best, N_DEV_FILES)


def test_a_difference_larger_than_one_file_still_wins_outright():
    """The window must not turn accuracy into a tie-break of its own."""
    worse_acc_better_auroc = fp.DevPoint(1, 69 / 71, 0.999)
    best = fp.DevPoint(0, 71 / 71, 0.900)
    assert not fp.is_better_checkpoint(worse_acc_better_auroc, best, 71)
    better_acc_worse_auroc = fp.DevPoint(1, 71 / 71, 0.900)
    assert fp.is_better_checkpoint(better_acc_worse_auroc, fp.DevPoint(0, 69 / 71, 0.999), 71)


def test_epoch_zero_stays_selectable_when_training_helps_nothing():
    """If no trained epoch beats the pre-trained encoder, keeping it is the
    honest answer, and the generated caption has a branch that says so."""
    curve = points([
        (0, 0.90, 0.95),
        (1, 0.90, 0.94),
        (2, 0.88, 0.93),
    ])
    assert fp.select_checkpoint(curve, N_DEV_FILES).epoch == 0


def test_ties_go_to_the_earlier_epoch():
    curve = points([(0, 0.90, 0.95), (1, 0.90, 0.95)])
    assert fp.select_checkpoint(curve, N_DEV_FILES).epoch == 0


def test_select_checkpoint_refuses_an_empty_curve():
    with pytest.raises(ValueError, match="at least one dev point"):
        fp.select_checkpoint([], N_DEV_FILES)
