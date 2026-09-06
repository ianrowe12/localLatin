#!/usr/bin/env python3
"""Write the benchmark v1 freeze manifest: sha256 per labelled file, plus a digest.

The manifest lists every file in `data/canon_labelled/` as `<sha256>  <relpath>`,
sorted by path, and ends with a single digest over that listing. Re-running it on a
clean checkout must reproduce the digest recorded in docs/research/benchmark_v1.md.

    python scripts/data/benchmark_manifest.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DEFAULT_ROOT = "data/canon_labelled"
DEFAULT_OUT = "docs/research/benchmark_v1_manifest.txt"
DIGEST_PREFIX = "# digest "


def build_manifest(root: Path) -> tuple[str, str, int]:
    """Return (listing, digest, n_files) for every .txt under `root`.

    A handful of directory names contain newlines (they are edition citations), so
    the listing is not safely line-addressable; the digest and the file count are.
    """
    paths = sorted(p for p in root.rglob("*.txt"))
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.as_posix()}")
    listing = "\n".join(lines) + "\n"
    return listing, hashlib.sha256(listing.encode("utf-8")).hexdigest(), len(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--check", action="store_true",
        help="Verify the corpus still matches the manifest instead of writing it.",
    )
    args = parser.parse_args(argv)

    listing, digest, n_files = build_manifest(Path(args.root))

    if args.check:
        existing = Path(args.out).read_text(encoding="utf-8")
        recorded = existing.rstrip("\n").rsplit("\n", 1)[-1]
        expected = f"{DIGEST_PREFIX}{digest}"
        if recorded != expected:
            print(f"MANIFEST MISMATCH\n  on disk:   {expected}\n  manifest:  {recorded}")
            return 1
        print(f"manifest OK: {n_files} files, digest {digest}")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(listing + f"{DIGEST_PREFIX}{digest}\n", encoding="utf-8")
    print(f"Wrote {out}: {n_files} files")
    print(f"digest {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
