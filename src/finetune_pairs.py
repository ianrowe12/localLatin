"""Positive-pair construction and directory-safe batching for the fine-tuning ceiling.

Split out of ``scripts/resubmit/finetune_lata_ceiling.py`` (issue #123) so the
two functions that decide whether the ceiling is honest can be tested without
torch or transformers. Both are pure pandas/NumPy/stdlib:

* :func:`build_pairs` carves the dev slice by *directory* before enumerating
  positive pairs, so model selection never sees the supervision it measures.
* :func:`batch_pairs_by_round` guarantees no batch repeats a directory, so
  in-batch negatives are always true negatives.
* :func:`is_better_checkpoint` decides which epoch the run extracts from, and
  knows that the dev pool cannot measure accuracy differences smaller than one
  file (issue #194).

The training CLI imports both from here; ``tests/test_finetune_ceiling_pairs.py``
imports them too, and runs on a clean CI checkout that has no torch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PairData",
    "DevPoint",
    "build_pairs",
    "batch_pairs_by_round",
    "dev_resolution",
    "is_better_checkpoint",
    "select_checkpoint",
]


@dataclass
class PairData:
    train_pairs: List[Tuple[int, int]]          # row indices into the split CSV
    train_pair_dirs: List[str]
    dev_dirs: List[str]
    fit_dirs: List[str]
    dev_rows: List[int]
    n_train_multi_dirs: int
    n_all_train_pairs: int


def build_pairs(split_meta: pd.DataFrame, dev_dir_frac: float, seed: int) -> PairData:
    """Split train directories into fit/dev by DIRECTORY, then enumerate pairs.

    Held-out dev directories contribute no training pair, so the dev retrieval
    metric is measured on text the contrastive objective has never seen.
    """
    train_rows = np.flatnonzero(split_meta["split"].values == "train")
    folder_ids = split_meta["folder_id"].values

    by_dir: Dict[str, List[int]] = {}
    for row in train_rows:
        by_dir.setdefault(str(folder_ids[row]), []).append(int(row))

    multi_dirs = sorted(d for d, rows in by_dir.items() if len(rows) >= 2)
    n_all_pairs = sum(len(by_dir[d]) * (len(by_dir[d]) - 1) // 2 for d in multi_dirs)

    rng = np.random.default_rng(seed)
    # A positive fraction always carves at least one directory, so a small
    # corpus still gets a dev slice; an explicit 0.0 means "no dev carve".
    n_dev = int(round(dev_dir_frac * len(multi_dirs)))
    if dev_dir_frac > 0:
        n_dev = max(1, n_dev)
    dev_idx = rng.choice(len(multi_dirs), size=n_dev, replace=False)
    dev_dirs = sorted(multi_dirs[i] for i in dev_idx)
    dev_set = set(dev_dirs)
    fit_dirs = [d for d in multi_dirs if d not in dev_set]

    train_pairs: List[Tuple[int, int]] = []
    train_pair_dirs: List[str] = []
    for d in fit_dirs:
        rows = sorted(by_dir[d])
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                train_pairs.append((rows[i], rows[j]))
                train_pair_dirs.append(d)

    dev_rows = sorted(r for d in dev_dirs for r in by_dir[d])

    return PairData(
        train_pairs=train_pairs,
        train_pair_dirs=train_pair_dirs,
        dev_dirs=dev_dirs,
        fit_dirs=fit_dirs,
        dev_rows=dev_rows,
        n_train_multi_dirs=len(multi_dirs),
        n_all_train_pairs=n_all_pairs,
    )


def batch_pairs_by_round(
    pairs: Sequence[Tuple[int, int]],
    pair_dirs: Sequence[str],
    batch_pairs: int,
    rng: random.Random,
) -> List[List[int]]:
    """Group pair indices into batches that never repeat a directory.

    In-batch negatives assume every other pair in the batch is a true negative.
    Two pairs from the same directory would be labelled negative for each other,
    so pairs are dealt out in rounds (at most one pair per directory per round)
    and each round is chunked separately.
    """
    remaining: Dict[str, List[int]] = {}
    for idx, d in enumerate(pair_dirs):
        remaining.setdefault(d, []).append(idx)
    for d in remaining:
        rng.shuffle(remaining[d])

    batches: List[List[int]] = []
    while remaining:
        # Draw from the directories with the most pairs left, so that the last
        # batches are not all monopolised by one prolific directory.
        chosen = sorted(remaining, key=lambda d: (-len(remaining[d]), d))[:batch_pairs]
        if len(chosen) < 2:  # a batch of one has no negatives
            break
        batch = []
        for d in chosen:
            batch.append(remaining[d].pop())
            if not remaining[d]:
                del remaining[d]
        rng.shuffle(batch)
        batches.append(batch)
    rng.shuffle(batches)
    return batches


@dataclass
class DevPoint:
    """One epoch's dev measurement, as the checkpoint selector sees it."""

    epoch: int
    dir_acc_at_1: float
    aucroc: float


def dev_resolution(n_dev_files: int) -> float:
    """The smallest accuracy difference the dev pool can express: one file."""
    return 1.0 / max(1, int(n_dev_files))


# Floating point: 70/71 and 71/71 differ by exactly 1/71 in exact arithmetic
# and by 1/71 plus a few ulps in binary64.
_EPS = 1e-9


def is_better_checkpoint(
    candidate: DevPoint, best: DevPoint, n_dev_files: int
) -> bool:
    """Is ``candidate`` a better checkpoint than ``best`` on the dev pool?

    Directory accuracy at rank 1 leads and AUROC breaks ties, as in #123. What
    changed in #194 is what counts as a tie. The rule used to demand exact
    equality before consulting AUROC, and that makes the tie-break unreachable
    whenever the pre-trained encoder already tops the pool out: Qwen3-0.6B
    starts at 71 of 71, training moves one file the wrong way, and a 1.4-point
    accuracy difference then vetoed an AUROC gain from 0.966 to 0.9998, so the
    run selected epoch 0 and extracted the pre-trained weights.

    One file is the pool's entire resolution. Differences at or below it are
    not measurements, so they are ties and AUROC decides. Differences above it
    are real and accuracy still wins outright.

    This does not move LaTa's published selection: its dev curve climbs by two
    files at epoch 3 and its last four epochs are already exact ties, so epoch
    7 is selected either way. ``tests/test_finetune_ceiling_pairs.py`` replays
    both measured curves and asserts exactly that.
    """
    window = dev_resolution(n_dev_files) + _EPS
    delta = candidate.dir_acc_at_1 - best.dir_acc_at_1
    if delta > window:
        return True
    if delta < -window:
        return False
    return candidate.aucroc > best.aucroc


def select_checkpoint(curve: Sequence[DevPoint], n_dev_files: int) -> DevPoint:
    """Replay a dev curve and return the epoch the run would extract from.

    Epoch 0 is the pre-trained encoder and is selectable: if no training epoch
    beats it, keeping it is the honest answer. The curve is walked in order, so
    an earlier epoch wins a tie against a later one.
    """
    if not curve:
        raise ValueError("select_checkpoint needs at least one dev point")
    best = curve[0]
    for point in curve[1:]:
        if is_better_checkpoint(point, best, n_dev_files):
            best = point
    return best
