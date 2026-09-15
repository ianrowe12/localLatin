"""Count how many benchmark directories are named by a CCL source key.

Appendix A of the paper states the split of the 840 directory labels into CCL
source keys and everything else. The number depends on the counting rule, so
the rule lives here and the appendix sentence states it (issue #208):

    A directory is key-named when its label begins with a four-letter
    upper-case source code, a period, a year (digits, optionally followed by
    ``?`` for an uncertain date) and a period, whatever follows.

That admits keys with trailing annotations (``CTOU.567.16 (15)``,
``DSIR.384.255 cap. 11``, ``CARL.501?.18``) and excludes the Apostolic Canons
(``Can.apost.N``), biblical references, edition citations, other collections
(``Capit.Martini.N``, ``Stat.Eccl.Antiq.``) and the unidentified sources.

Directory names under ``data/canon_labelled/`` contain newlines, parentheses
and quotes (issue #203), so the walk uses ``Path.iterdir`` rather than a
line-based shell pipeline.

    python scripts/data/label_taxonomy.py            # counts on the frozen benchmark
    python scripts/data/label_taxonomy.py --list-other
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

KEY_PATTERN = re.compile(r"^[A-Z]{4}\.\d+\??\.")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "canon_labelled"


def is_key_named(label: str) -> bool:
    return KEY_PATTERN.match(label) is not None


def directory_labels(root: Path) -> list[str]:
    """Every directory name under ``root``, newline-safe, sorted."""
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def count_taxonomy(labels: Iterable[str]) -> tuple[int, int]:
    """(key-named, other) over the given labels."""
    labels = list(labels)
    keyed = sum(is_key_named(label) for label in labels)
    return keyed, len(labels) - keyed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--list-other", action="store_true",
                        help="Print the labels that are not key-named.")
    args = parser.parse_args()
    labels = directory_labels(args.root)
    keyed, other = count_taxonomy(labels)
    print(f"directories: {len(labels)}")
    print(f"key-named ({KEY_PATTERN.pattern}): {keyed}")
    print(f"other: {other}")
    if args.list_other:
        for label in labels:
            if not is_key_named(label):
                print(repr(label))


if __name__ == "__main__":
    main()
