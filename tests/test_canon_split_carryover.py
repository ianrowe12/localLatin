"""Carry-over split rebuild after a label correction (issues #112, #113).

``build_meta_with_carried_over_split`` is what keeps a label correction from
reshuffling the whole benchmark: every file keeps the train/test and Task B
assignment it already had, and only the directory-derived bookkeeping is
recomputed. The failure mode worth testing is silent drift, so these check the
invariants directly rather than a golden file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from canon_split_v2 import (  # noqa: E402
    build_meta_with_carried_over_split,
    recompute_derived_columns,
)


# A corpus small enough to reason about by hand: one directory of three, one of
# two, and two singletons.
CORPUS = {
    "Dir.A": ["a1.txt", "a2.txt", "a3.txt"],
    "Dir.B": ["b1.txt", "b2.txt"],
    "Dir.C": ["c1.txt"],
    "Dir.D": ["d1.txt"],
}

ASSIGNMENT = {
    "a1.txt": ("train", "train"),
    "a2.txt": ("test", "query"),
    "a3.txt": ("test", "reference"),
    "b1.txt": ("train", "train"),
    "b2.txt": ("test", "query"),
    "c1.txt": ("test", "reference"),
    "d1.txt": ("train", "train"),
}


def write_corpus(root: Path, layout: dict) -> None:
    for folder, files in layout.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        for name in files:
            (root / folder / name).write_text(f"text of {name}\n", encoding="utf-8")


def write_prior_split(path: Path, layout: dict, root: Path) -> pd.DataFrame:
    rows = []
    for folder, files in sorted(layout.items()):
        for name in sorted(files):
            split, role = ASSIGNMENT[name]
            rows.append(
                {
                    "folder_id": folder,
                    "filename": name,
                    "path": str(root / folder / name),
                    "folder_size": len(files),
                    "is_singleton": len(files) == 1,
                    "is_winnable": len(files) >= 2,
                    "file_id": len(rows),
                    "split": split,
                    "is_test_query": False,
                    "has_test_partner": False,
                    "taskb_role": role,
                    "has_reference_dir": False,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "canon"
    write_corpus(root, CORPUS)
    prior = tmp_path / "prior_split.csv"
    write_prior_split(prior, CORPUS, root)
    return root, prior


def test_carry_over_preserves_every_assignment(corpus):
    root, prior = corpus
    meta = build_meta_with_carried_over_split(str(root), str(prior))

    assert len(meta) == 7
    for _, row in meta.iterrows():
        split, role = ASSIGNMENT[row["filename"]]
        assert row["split"] == split
        assert row["taskb_role"] == role


def test_carry_over_recomputes_directory_columns_after_a_move(corpus, tmp_path):
    """Moving one file between directories must move only its own bookkeeping."""
    root, prior = corpus
    before = build_meta_with_carried_over_split(str(root), str(prior))

    # The correction: a3 belonged in Dir.B all along.
    (root / "Dir.A" / "a3.txt").rename(root / "Dir.B" / "a3.txt")
    after = build_meta_with_carried_over_split(str(root), str(prior))

    assert list(after["split"]) == list(before["split"])
    moved = after[after["filename"] == "a3.txt"].iloc[0]
    assert moved["folder_id"] == "Dir.B"
    assert moved["folder_size"] == 3
    assert after[after["filename"] == "a1.txt"].iloc[0]["folder_size"] == 2
    # Dir.B now has two test files (a3 reference, b2 query), so b2 gains a partner.
    assert bool(after[after["filename"] == "b2.txt"].iloc[0]["has_test_partner"])
    assert bool(after[after["filename"] == "b2.txt"].iloc[0]["has_reference_dir"])


def test_file_id_is_the_row_index_in_folder_filename_order(corpus):
    root, prior = corpus
    meta = build_meta_with_carried_over_split(str(root), str(prior))

    assert list(meta["file_id"]) == list(range(len(meta)))
    ordered = meta.sort_values(["folder_id", "filename"]).reset_index(drop=True)
    assert list(ordered["filename"]) == list(meta["filename"])


def test_carry_over_refuses_an_added_file(corpus):
    root, prior = corpus
    (root / "Dir.C" / "c2.txt").write_text("new witness\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from"):
        build_meta_with_carried_over_split(str(root), str(prior))


def test_carry_over_refuses_a_removed_file(corpus):
    root, prior = corpus
    (root / "Dir.D" / "d1.txt").unlink()
    with pytest.raises(ValueError, match="no longer on disk"):
        build_meta_with_carried_over_split(str(root), str(prior))


def test_carry_over_refuses_a_prior_split_without_taskb_role(corpus, tmp_path):
    root, prior = corpus
    trimmed = pd.read_csv(prior).drop(columns=["taskb_role"])
    path = tmp_path / "no_role.csv"
    trimmed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="taskb_role"):
        build_meta_with_carried_over_split(str(root), str(path))


def base_meta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Two test files in one directory: partners, and one is a query with
            # a reference directory.
            {"folder_id": "X", "filename": "x1.txt", "split": "test", "taskb_role": "query"},
            {"folder_id": "X", "filename": "x2.txt", "split": "test", "taskb_role": "reference"},
            # One test file alone in its directory: no partner, no reference.
            {"folder_id": "Y", "filename": "y1.txt", "split": "test", "taskb_role": "query"},
            {"folder_id": "Y", "filename": "y2.txt", "split": "train", "taskb_role": "train"},
        ]
    )


def test_recompute_derived_columns_marks_partners_and_queries():
    out = recompute_derived_columns(base_meta())

    partner = dict(zip(out["filename"], out["has_test_partner"]))
    assert partner == {"x1.txt": True, "x2.txt": True, "y1.txt": False, "y2.txt": False}

    query = dict(zip(out["filename"], out["is_test_query"]))
    assert query == {"x1.txt": True, "x2.txt": True, "y1.txt": False, "y2.txt": False}

    has_ref = dict(zip(out["filename"], out["has_reference_dir"]))
    # Only a query whose own directory has a reference file counts.
    assert has_ref == {"x1.txt": True, "x2.txt": False, "y1.txt": False, "y2.txt": False}


def test_recompute_derived_columns_does_not_mutate_its_input():
    meta = base_meta()
    meta["has_test_partner"] = False
    recompute_derived_columns(meta)
    assert not meta["has_test_partner"].any()


def test_recompute_derived_columns_follows_a_moved_file():
    meta = base_meta()
    assert not recompute_derived_columns(meta)["has_test_partner"].iloc[2]

    # y1 is relabelled into X, which already holds a reference file.
    meta.loc[2, "folder_id"] = "X"
    out = recompute_derived_columns(meta)
    assert bool(out[out["filename"] == "y1.txt"]["has_test_partner"].iloc[0])
    assert bool(out[out["filename"] == "y1.txt"]["has_reference_dir"].iloc[0])
