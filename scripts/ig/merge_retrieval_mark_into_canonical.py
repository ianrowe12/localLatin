"""Merge retrieval-adapted MaRC sidecar NPZs into canonical per-pair artifacts.

For every row in the phase12f examples CSV, load the canonical NPZ at
``<canonical_dir>/<slug>/example<id>_pair_example.npz`` and the sidecar
at ``<sidecar_dir>/<slug>/example<id>_pair_example.npz``, then rewrite the
canonical file with the retrieval-adapted MaRC keys overlaid on top. ``hyperparams``
from the sidecar is renamed to ``hyperparams_retrieval_mark`` to avoid
clobbering any existing canonical key.

The merge is idempotent (safe to re-run) and atomic (writes to a .tmp.npz
first, then ``os.replace``s into place).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


NEW_KEYS = (
    "q_mask_retrieval_mark_baseline",
    "q_mask_retrieval_mark_abtt",
    "c_mask_retrieval_mark_baseline",
    "c_mask_retrieval_mark_abtt",
    "pair_matrix_retrieval_mark_baseline",
    "pair_matrix_retrieval_mark_abtt",
    "topk_retrieval_mark_baseline_query",
    "topk_retrieval_mark_baseline_candidate",
    "topk_retrieval_mark_abtt_query",
    "topk_retrieval_mark_abtt_candidate",
    "convergence_q_baseline",
    "convergence_q_abtt",
    "convergence_c_baseline",
    "convergence_c_abtt",
    "cos_orig_baseline",
    "cos_orig_abtt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples_csv",
        type=Path,
        default=Path("runs/active/ig_examples/phase12f_examples.csv"),
    )
    parser.add_argument(
        "--canonical_dir",
        type=Path,
        default=Path("runs/active/ig_examples/artifacts"),
    )
    parser.add_argument(
        "--sidecar_dir",
        type=Path,
        default=Path("runs/active/ig_examples/retrieval_mark/artifacts"),
    )
    parser.add_argument(
        "--update_methods_available",
        action="store_true",
        default=True,
        help="Add 'retrieval_mark' to the methods_available column (default: true).",
    )
    parser.add_argument(
        "--no_update_methods_available",
        dest="update_methods_available",
        action="store_false",
        help="Do not edit the CSV's methods_available column.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print plan without writing any files.",
    )
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def _scalar_int(arr) -> int:
    a = np.asarray(arr)
    if a.ndim == 0:
        return int(a)
    return int(a.reshape(-1)[0])


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)

    n_merged = 0
    n_missing_sidecar = 0
    n_failed = 0
    failures: list[tuple[int, str]] = []

    for row in examples.itertuples(index=False):
        example_id = int(row.example_id)
        slug = model_slug(str(row.model_name))
        canonical_path = (
            args.canonical_dir / slug / f"example{example_id:03d}_pair_example.npz"
        )
        sidecar_path = (
            args.sidecar_dir / slug / f"example{example_id:03d}_pair_example.npz"
        )

        if not sidecar_path.exists():
            print(f"[WARN] missing sidecar: {sidecar_path}")
            n_missing_sidecar += 1
            continue
        if not canonical_path.exists():
            print(f"[WARN] missing canonical: {canonical_path}")
            n_failed += 1
            failures.append((example_id, f"missing_canonical:{canonical_path}"))
            continue

        try:
            with np.load(canonical_path) as canon:
                canon_dict = {k: canon[k] for k in canon.files}
            with np.load(sidecar_path) as side:
                side_dict = {k: side[k] for k in side.files}

            canon_eid = _scalar_int(canon_dict["example_id"])
            side_eid = _scalar_int(side_dict["example_id"])
            if canon_eid != side_eid:
                raise ValueError(
                    f"example_id mismatch: canonical={canon_eid} sidecar={side_eid}"
                )

            merged = dict(canon_dict)
            for key in NEW_KEYS:
                if key in side_dict:
                    merged[key] = side_dict[key]

            if "hyperparams" in side_dict:
                merged["hyperparams_retrieval_mark"] = side_dict["hyperparams"]

            if args.dry_run:
                added = [k for k in NEW_KEYS if k in side_dict]
                print(
                    f"[DRY] example {example_id} ({slug}): "
                    f"would add {len(added)} keys -> {canonical_path}"
                )
            else:
                tmp_path = canonical_path.with_suffix(".tmp.npz")
                np.savez(tmp_path, **merged)
                os.replace(tmp_path, canonical_path)
                n_merged += 1
                print(f"[OK] merged example {example_id} ({slug}) -> {canonical_path}")
        except Exception as exc:
            n_failed += 1
            failures.append((example_id, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] example {example_id} ({slug}): {exc}")

    if args.update_methods_available:
        if "methods_available" in examples.columns:
            new_col = []
            changed = 0
            for val in examples["methods_available"].fillna(""):
                s = str(val).strip()
                parts = [p.strip() for p in s.split(";") if p.strip()] if ";" in s else (
                    [p.strip() for p in s.split(",") if p.strip()]
                )
                sep = ";" if ";" in s else ","
                if "retrieval_mark" not in parts:
                    parts.append("retrieval_mark")
                    changed += 1
                new_col.append(sep.join(parts))
            examples["methods_available"] = new_col
            if args.dry_run:
                print(
                    f"[DRY] would update methods_available on {changed} rows "
                    f"in {args.examples_csv}"
                )
            else:
                examples.to_csv(args.examples_csv, index=False)
                print(
                    f"[OK] updated methods_available on {changed} rows in "
                    f"{args.examples_csv}"
                )
        else:
            print(
                "[WARN] methods_available column not found in CSV; skipping update."
            )

    print("\n=== Summary ===")
    print(f"merged: {n_merged}")
    print(f"missing sidecar: {n_missing_sidecar}")
    print(f"failed: {n_failed}")
    for eid, reason in failures:
        print(f"  example {eid}: {reason}")


if __name__ == "__main__":
    main()
