"""Prove that a cached embedding cache still lines up with a relabelled split.

The label corrections frozen as benchmark v1 moved two files between
directories. That permuted seventeen split rows without touching a single
embedding, so a row-count check cannot tell a correct run from a misaligned one.
This script makes the claim checkable:

1. Every embedding cache under the bases root resolves to a row-order manifest.
2. The permutation the manifest implies matches the one implied by comparing the
   pre-correction split to the corrected split, row for row.
3. The vector a named file gets after alignment is bit-identical to the vector it
   got before the relabelling. That is the whole point: the text did not change,
   so the embedding must not change either.

Exits non-zero on any failure, so it can gate a re-run.

    python scripts/resubmit/verify_embedding_alignment.py \
        --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
        --prior_split_csv runs/active/resubmit/data/benchmark_v1/phase_resubmit_split.pre_correction_backup.csv \
        --runs_root runs/active/resubmit_bases/phase9_bases \
        --spot_check BN2123.89r.6.txt --spot_check BN2123.89r.5.txt
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from embedding_alignment import (  # noqa: E402
    STATUS_UNVERIFIED,
    AlignmentResolver,
    EmbeddingAligner,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split_csv", default="runs/active/resubmit/data/phase_resubmit_split.csv")
    p.add_argument(
        "--prior_split_csv",
        default="runs/active/resubmit/data/benchmark_v1/phase_resubmit_split.pre_correction_backup.csv",
        help="Split as it stood before the relabelling, for the before/after check.",
    )
    p.add_argument("--runs_root", default="runs/active/resubmit_bases/phase9_bases")
    p.add_argument(
        "--spot_check", action="append", default=None,
        help="Filename to check byte-for-byte. Repeatable.",
    )
    p.add_argument("--report_json", default="", help="Optional path for a JSON report.")
    return p.parse_args()


def row_sha256(vec: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vec).tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    spot_files = args.spot_check or ["BN2123.89r.6.txt", "BN2123.89r.5.txt"]

    split = pd.read_csv(args.split_csv)
    prior = pd.read_csv(args.prior_split_csv)
    runs_root = Path(args.runs_root)

    failures = []

    # --- 1. The permutation implied by the two splits -------------------------
    prior_pos = {f: i for i, f in enumerate(prior["filename"])}
    expected_perm = np.array(
        [prior_pos[f] for f in split["filename"]], dtype=np.int64
    )
    n_moved_expected = int((expected_perm != np.arange(len(expected_perm))).sum())
    print(
        f"split diff: {n_moved_expected} of {len(split)} rows change position "
        f"between {args.prior_split_csv} and {args.split_csv}"
    )
    moved = [
        (int(new), int(old), split["filename"].iloc[new], split["folder_id"].iloc[new])
        for new, old in enumerate(expected_perm)
        if new != old
    ]
    for new, old, fname, folder in sorted(moved, key=lambda r: r[1]):
        print(f"  file_id {old:5d} -> {new:5d}  {fname:<20s} {folder}")

    # --- 2. Every cache resolves to a manifest, and agrees with that permutation
    run_dirs = sorted({p.parent for p in runs_root.rglob("*_embeddings*.npy")})
    if not run_dirs:
        print(f"FAIL: no embedding caches under {runs_root}", file=sys.stderr)
        return 1
    print(f"\nchecking {len(run_dirs)} cache directories under {runs_root}")

    for run_dir in run_dirs:
        aligner = EmbeddingAligner.for_run_dir(run_dir, split)
        if aligner.status == STATUS_UNVERIFIED:
            failures.append(f"{run_dir}: no row-order manifest")
            print(f"  FAIL {run_dir}: {aligner.describe()}")
            continue
        actual = (
            np.arange(len(split)) if aligner.perm is None else aligner.perm
        )
        if not np.array_equal(actual, expected_perm):
            failures.append(f"{run_dir}: permutation disagrees with the split diff")
            print(f"  FAIL {run_dir}: permutation disagrees with the split diff")
        else:
            print(f"  ok   {run_dir.relative_to(runs_root)}: {aligner.describe()}")

    # --- 3. Named-file byte identity before and after the relabelling ---------
    print("\nspot checks (vector must be bit-identical before and after relabelling)")
    resolver = AlignmentResolver(split)
    prior_resolver = AlignmentResolver(prior)
    checked = 0
    for run_dir in run_dirs:
        emb_files = sorted(run_dir.glob("*_embeddings*.npy"))
        if not emb_files:
            continue
        emb_path = emb_files[0]
        raw = np.load(emb_path)
        after = resolver.aligner_for(emb_path).apply(raw)
        before = prior_resolver.aligner_for(emb_path).apply(raw)
        for fname in spot_files:
            new_row = int(np.flatnonzero(split["filename"].to_numpy() == fname)[0])
            old_row = int(np.flatnonzero(prior["filename"].to_numpy() == fname)[0])
            a, b = after[new_row], before[old_row]
            same = a.tobytes() == b.tobytes()
            checked += 1
            status = "ok  " if same else "FAIL"
            print(
                f"  {status} {emb_path.relative_to(runs_root)} {fname}: "
                f"split row {old_row} -> {new_row}, sha256 {row_sha256(a)[:16]} "
                f"(before {row_sha256(b)[:16]})"
            )
            if not same:
                failures.append(f"{emb_path} {fname}: vector changed under relabelling")

    print(f"\n{checked} spot checks over {len(run_dirs)} caches")
    if failures:
        print(f"FAILED: {len(failures)} problem(s)", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("PASS: every cache is manifest-verified and every spot check is byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
