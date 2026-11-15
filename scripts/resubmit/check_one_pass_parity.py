"""Check that ``extract_encoder_cli.py --one_pass`` reproduces a cached extraction.

The one-pass path pools every layer from one forward pass per batch instead of
re-running the model once per layer. It should give the same vectors, because
the batches are the same and the per-layer path already computed every hidden
state and kept one. This proves it on real data before a cache is written with
it: re-extract the first ``--n_files`` files of an existing cache, in that
cache's own row order and batch size (so the padding is identical), and compare
row by row against the cached vectors, matched by filename.

Two commands, because the GPU part is the extractor itself:

    # 1. write the subset meta (cache order, current on-disk paths)
    python scripts/resubmit/check_one_pass_parity.py make_meta \\
        --cache_dir <cached run dir> --split_csv <split> --n_files 16 --out_meta <csv>
    # 2. run the extractor with --one_pass on <csv> into <fresh dir>, then
    python scripts/resubmit/check_one_pass_parity.py compare \\
        --cache_dir <cached run dir> --fresh_dir <fresh dir> --suffix _lasttok

``compare`` exits non-zero if any row's cosine to its cached vector is below
``--min_cosine``, so an sbatch can stop before writing the real cache.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from embedding_alignment import load_row_order  # noqa: E402

LAYER_RE = re.compile(r"^hidden_layer(\d+)_embeddings(.*)\.npy$")


def make_meta(cache_dir: Path, split_csv: Path, n_files: int, out_meta: Path) -> None:
    """First ``n_files`` of the cache's row order, with today's on-disk paths.

    The cache's own ``meta.csv`` may carry paths from before a relabelling moved
    a file to another directory, so the path is looked up in the split by
    filename. The text is the same file either way.
    """
    order = load_row_order(cache_dir / "meta.csv")[:n_files]
    split = pd.read_csv(split_csv)
    by_name = dict(zip(split["filename"].astype(str), split["path"].astype(str)))
    missing = [name for name in order if name not in by_name]
    if missing:
        raise SystemExit(f"{len(missing)} cache file(s) not in the split, e.g. {missing[:3]}")
    meta = pd.DataFrame(
        {
            "file_id": range(len(order)),
            "filename": order,
            "path": [by_name[name] for name in order],
        }
    )
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_meta, index=False)
    print(f"Wrote {len(meta)} rows -> {out_meta}")


def compare(cache_dir: Path, fresh_dir: Path, suffix: str, min_cosine: float) -> int:
    cache_order = load_row_order(cache_dir / "meta.csv")
    fresh_order = load_row_order(fresh_dir / "meta.csv")
    cache_pos = {name: i for i, name in enumerate(cache_order)}
    rows = np.array([cache_pos[name] for name in fresh_order])

    worst = 1.0
    worst_abs = 0.0
    n_layers = 0
    for path in sorted(fresh_dir.glob(f"hidden_layer*_embeddings{suffix}.npy")):
        match = LAYER_RE.match(path.name)
        if not match or match.group(2) != suffix:
            continue
        fresh = np.load(path).astype(np.float64)
        cached = np.load(cache_dir / path.name)[rows].astype(np.float64)
        num = (fresh * cached).sum(axis=1)
        den = np.linalg.norm(fresh, axis=1) * np.linalg.norm(cached, axis=1)
        cos = np.where(den > 0, num / np.maximum(den, 1e-30), 1.0)
        abs_diff = float(np.abs(fresh - cached).max())
        worst = min(worst, float(cos.min()))
        worst_abs = max(worst_abs, abs_diff)
        n_layers += 1
        print(f"  layer {match.group(1):>2}: min cos {cos.min():.8f}  max |diff| {abs_diff:.3e}")

    if n_layers == 0:
        print("No layers compared.")
        return 1
    ok = worst >= min_cosine
    print(
        f"{'PASS' if ok else 'FAIL'}: {n_layers} layer(s), {len(rows)} row(s), "
        f"min cosine {worst:.8f} (threshold {min_cosine}), max |diff| {worst_abs:.3e}"
    )
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make_meta")
    m.add_argument("--cache_dir", type=Path, required=True)
    m.add_argument("--split_csv", type=Path, required=True)
    m.add_argument("--n_files", type=int, default=16)
    m.add_argument("--out_meta", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--cache_dir", type=Path, required=True)
    c.add_argument("--fresh_dir", type=Path, required=True)
    c.add_argument("--suffix", default="", help="'' (mean), '_lasttok' or '_sif'")
    c.add_argument("--min_cosine", type=float, default=0.9999)
    args = parser.parse_args()
    if args.cmd == "make_meta":
        make_meta(args.cache_dir, args.split_csv, args.n_files, args.out_meta)
    else:
        sys.exit(compare(args.cache_dir, args.fresh_dir, args.suffix, args.min_cosine))


if __name__ == "__main__":
    main()
