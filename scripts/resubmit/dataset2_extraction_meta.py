"""Write the meta CSVs the issue #289 Qwen3-8B extraction walks.

``unlabelled_meta.csv`` keeps the rows and order of
``runs/active/resubmit/data/unlabelled_meta.csv`` (the order every other
unlabelled cache was written in); only the path gains its ``data/`` prefix so it
resolves from the repo root. ``labelled_meta.csv`` is the benchmark v1 split in
split order. Both are checked for unique filenames and existing files, since
the extractor copies them verbatim into each run directory as the row-order
manifest the AlignmentResolver reads.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

N_UNLABELLED = 2238
N_LABELLED = 1705


def _check(df: pd.DataFrame, n: int, what: str) -> None:
    if len(df) != n:
        raise SystemExit(f"{what}: {len(df)} rows, expected {n}")
    if not df["filename"].is_unique:
        raise SystemExit(f"{what}: duplicate filenames")
    missing = [p for p in df["path"] if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"{what}: {len(missing)} path(s) missing, e.g. {missing[:3]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--unlabelled_meta", default="runs/active/resubmit/data/unlabelled_meta.csv")
    p.add_argument("--split_csv", default="runs/active/resubmit/data/phase_resubmit_split.csv")
    p.add_argument("--out_dir", default="runs/active/dataset2/extraction_meta")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    u = pd.read_csv(args.unlabelled_meta)
    u["path"] = [p if str(p).startswith("data/") else f"data/{p}" for p in u["path"]]
    _check(u, N_UNLABELLED, "unlabelled")
    u[["file_id", "filename", "path"]].to_csv(out / "unlabelled_meta.csv", index=False)

    s = pd.read_csv(args.split_csv)
    _check(s, N_LABELLED, "labelled")
    s[["file_id", "filename", "path"]].to_csv(out / "labelled_meta.csv", index=False)
    print(f"unlabelled {len(u)}, labelled {len(s)} -> {out}")


if __name__ == "__main__":
    main()
