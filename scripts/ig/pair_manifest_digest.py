"""Write and verify the attribution pair-sample digest.

The digest identifies *the sample*, not the file that carries it. It covers
`(model, query filename, candidate filename, folder)` only, so it survives the
`methods_available` column that the MaRC merge rewrites in place, and it does
not depend on column order or on the absolute paths baked into the examples CSV.

Rows are sorted by `(model_name, query_file_id, candidate_file_id)`, joined with
tabs, one row per line, `\\n`-separated, **without a trailing newline**, and
hashed with sha256. That is not `sha256sum pair_manifest.tsv`, which covers the
file including its trailing newline; this script is the definition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd


def manifest_rows(examples_csv: Path, split_csv: Path) -> str:
    """The canonical pair-manifest text for a sample."""
    examples = pd.read_csv(examples_csv)
    split = pd.read_csv(split_csv)
    filename = dict(zip(split["file_id"], split["filename"]))
    ordered = examples.sort_values(
        ["model_name", "query_file_id", "candidate_file_id"]
    )
    return "\n".join(
        "\t".join(
            (
                str(row.model_name),
                str(filename[row.query_file_id]),
                str(filename[row.candidate_file_id]),
                str(row.query_folder_id),
            )
        )
        for row in ordered.itertuples()
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def compute(examples_csv: Path, split_csv: Path) -> Tuple[str, str]:
    text = manifest_rows(examples_csv, split_csv)
    return text, digest(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples_csv", required=True, type=Path)
    parser.add_argument("--split_csv", required=True, type=Path)
    parser.add_argument(
        "--manifest_out", type=Path, default=None,
        help="Write the manifest text here (a trailing newline is added).",
    )
    parser.add_argument(
        "--expect", default=None,
        help="Fail if the computed digest differs from this one.",
    )
    args = parser.parse_args()

    text, sha = compute(args.examples_csv, args.split_csv)
    if args.manifest_out is not None:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(text + "\n")
        print(f"wrote {args.manifest_out}")
    print(json.dumps({"pairs": text.count("\n") + 1, "pair_manifest_sha256": sha}))
    if args.expect is not None and args.expect != sha:
        print(f"MISMATCH: expected {args.expect}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
