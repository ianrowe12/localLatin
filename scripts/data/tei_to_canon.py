#!/usr/bin/env python3
"""Convert a CCL TEI-P5 manuscript export into the canon plain-text layout.

One `.txt` file per transcription unit, named after the unit's `xml:id`. The text
is the manuscript rubric (`msSource`, `number`, `rubric`) followed by the canon
text, with inline editorial apparatus rendered as the existing corpus renders it.
The CCL key in `<div type="scholSource">` is stripped from the text and becomes the
directory name; units without a key go to the unlabelled pool.

The rule is derived from and verified against the 279 existing BN2123 files; see
`docs/research/data_derivation.md` for the derivation and the reproduction figures.

Examples
--------
Reproduce an export into a fresh tree::

    python scripts/data/tei_to_canon.py data/tei_samples/BN2123.xml --out-dir /tmp/bn2123

Review an export against the frozen corpus without writing anything::

    python scripts/data/tei_to_canon.py data/tei_samples/BN2123.xml --dry-run \\
        --report /tmp/bn2123_diff.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

#: Content is dropped (a deletion is not part of the reading text) but the element
#: still contributes its boundary spaces, exactly as in the existing corpus files.
DROPPED_CONTENT_TAGS = frozenset({"del"})

#: Child divs whose paragraphs are never part of the unit text.
EXCLUDED_DIV_TYPES = frozenset({"scholSource"})

#: Elements that carry a block of text inside a unit.
BLOCK_TAGS = frozenset({"p", "ab"})

#: Whitespace conventions of the existing corpus files.
UNIT_PREFIX = "   "
BLOCK_SEPARATOR = "  "
INLINE_PAD = " "

LABELLED_SUBDIR = "canon_labelled"
UNLABELLED_SUBDIR = "canon_unlabelled"


@dataclass(frozen=True)
class Unit:
    """One transcription unit: a single `.txt` file in the corpus layout."""

    unit_id: str
    unit_type: Optional[str]
    key: Optional[str]
    text: str

    @property
    def filename(self) -> str:
        return f"{self.unit_id}.txt"

    @property
    def directory(self) -> Optional[str]:
        """Corpus directory for this unit, or None when it is unlabelled."""
        return self.key

    @property
    def siglum(self) -> str:
        """Manuscript siglum, i.e. the part of the id before the first dot."""
        return self.unit_id.split(".", 1)[0]


def local_name(element: ET.Element) -> str:
    tag = element.tag
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def render_inline(element: ET.Element) -> str:
    """Flatten an element's content (not its tail) to text.

    Every nested element contributes one space before and one space after its
    rendered content, and the content of `<del>` is dropped while its boundary
    spaces remain. Empty milestone elements such as `<pb/>` therefore render as
    whitespace, which is what keeps a word broken across a page break split the
    way the corpus has it.
    """
    parts: List[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        inner = "" if local_name(child) in DROPPED_CONTENT_TAGS else render_inline(child)
        parts.append(INLINE_PAD + inner + INLINE_PAD)
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def unit_blocks(unit: ET.Element) -> List[ET.Element]:
    """Text-bearing blocks of a unit, in document order, minus the CCL key."""
    blocks: List[ET.Element] = []
    for child in unit:
        name = local_name(child)
        if name == "div":
            if child.get("type") in EXCLUDED_DIV_TYPES:
                continue
            blocks.extend(p for p in child if local_name(p) in BLOCK_TAGS)
        elif name in BLOCK_TAGS or name == "note":
            blocks.append(child)
    return blocks


def render_unit_text(unit: ET.Element) -> str:
    """Render one unit div to the plain text stored in the corpus."""
    rendered = [
        re.sub(r"\s*\n\s*", " ", render_inline(block)) for block in unit_blocks(unit)
    ]
    return UNIT_PREFIX + BLOCK_SEPARATOR.join(rendered)


def unit_key(unit: ET.Element) -> Optional[str]:
    """CCL source key from `<div type="scholSource">`, or None when absent."""
    for child in unit:
        if local_name(child) == "div" and child.get("type") == "scholSource":
            key = " ".join("".join(child.itertext()).split())
            return key or None
    return None


def iter_unit_elements(root: ET.Element) -> Iterable[ET.Element]:
    for element in root.iter():
        if local_name(element) == "div" and element.get(XML_ID):
            yield element


def parse_units(
    source: str | Path,
    types: Optional[Sequence[str]] = None,
) -> Tuple[List[Unit], List[str]]:
    """Parse a TEI export into units.

    `source` is a path to a TEI file or a string of TEI markup. `types` optionally
    restricts the `div/@type` values that are converted. Returns the units and a
    list of human-readable warnings (duplicate ids, empty units, skipped types).
    """
    text = str(source)
    if "<" in text:
        root = ET.fromstring(text)
    else:
        root = ET.parse(str(source)).getroot()

    wanted = set(types) if types else None
    units: List[Unit] = []
    warnings: List[str] = []
    seen: Dict[str, str] = {}

    for element in iter_unit_elements(root):
        unit_id = element.get(XML_ID) or ""
        unit_type = element.get("type")
        if wanted is not None and unit_type not in wanted:
            continue
        if unit_id in seen:
            warnings.append(
                f"duplicate xml:id {unit_id!r} (type {unit_type!r}); keeping the first"
            )
            continue
        body = render_unit_text(element)
        if not body.strip():
            warnings.append(f"{unit_id!r} has no text content; skipped")
            continue
        seen[unit_id] = unit_type or ""
        units.append(
            Unit(unit_id=unit_id, unit_type=unit_type, key=unit_key(element), text=body)
        )
    return units, warnings


def normalise(text: str) -> str:
    """Whitespace normalisation used for comparing against existing files."""
    return re.sub(r"\s+", " ", text).strip()


def index_corpus(data_root: Path) -> Dict[str, Tuple[Optional[str], Path]]:
    """Map filename stem -> (directory or None for unlabelled, path)."""
    index: Dict[str, Tuple[Optional[str], Path]] = {}
    unlabelled = data_root / UNLABELLED_SUBDIR
    if unlabelled.is_dir():
        for path in sorted(unlabelled.glob("*.txt")):
            index[path.stem] = (None, path)
    labelled = data_root / LABELLED_SUBDIR
    if labelled.is_dir():
        for path in sorted(labelled.glob("*/*.txt")):
            index[path.stem] = (path.parent.name, path)
    return index


@dataclass
class DiffRow:
    unit_id: str
    unit_type: Optional[str]
    status: str
    existing_dir: str
    export_dir: str
    detail: str


def diff_units(
    units: Sequence[Unit],
    index: Dict[str, Tuple[Optional[str], Path]],
) -> List[DiffRow]:
    """Compare converted units against the corpus on disk.

    Statuses: `new` (no such file), `moved` (different directory), `changed`
    (different text), `moved+changed`, `identical` (byte-for-byte) and
    `whitespace` (identical after whitespace normalisation).
    """
    rows: List[DiffRow] = []
    export_sigla = {unit.siglum for unit in units}
    seen_stems = set()

    for unit in units:
        seen_stems.add(unit.unit_id)
        entry = index.get(unit.unit_id)
        export_dir = unit.directory or UNLABELLED_SUBDIR
        if entry is None:
            rows.append(
                DiffRow(unit.unit_id, unit.unit_type, "new", "", export_dir, "")
            )
            continue
        existing_dir_raw, path = entry
        existing_dir = existing_dir_raw or UNLABELLED_SUBDIR
        existing_text = path.read_text(encoding="utf-8")
        moved = existing_dir != export_dir
        if existing_text == unit.text:
            text_status, detail = "identical", ""
        elif normalise(existing_text) == normalise(unit.text):
            text_status, detail = "whitespace", "differs only in whitespace"
        else:
            text_status = "changed"
            detail = _first_difference(normalise(existing_text), normalise(unit.text))
        if moved and text_status == "changed":
            status = "moved+changed"
        elif moved:
            status = "moved"
        else:
            status = text_status
        rows.append(
            DiffRow(unit.unit_id, unit.unit_type, status, existing_dir, export_dir, detail)
        )

    for stem, (existing_dir_raw, _path) in index.items():
        if stem in seen_stems:
            continue
        if stem.split(".", 1)[0] in export_sigla:
            rows.append(
                DiffRow(
                    stem,
                    None,
                    "absent_from_export",
                    existing_dir_raw or UNLABELLED_SUBDIR,
                    "",
                    "",
                )
            )
    return rows


def _first_difference(a: str, b: str, window: int = 40) -> str:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    start = max(0, i - window // 2)
    return f"corpus ...{a[start:i + window]!r} vs export ...{b[start:i + window]!r}"


def write_units(units: Sequence[Unit], out_dir: Path) -> Tuple[int, int]:
    """Write units into `<out_dir>/canon_labelled/<key>/` and `canon_unlabelled/`."""
    n_labelled = 0
    n_unlabelled = 0
    for unit in units:
        if unit.directory:
            target = out_dir / LABELLED_SUBDIR / unit.directory
            n_labelled += 1
        else:
            target = out_dir / UNLABELLED_SUBDIR
            n_unlabelled += 1
        target.mkdir(parents=True, exist_ok=True)
        (target / unit.filename).write_text(unit.text, encoding="utf-8")
    return n_labelled, n_unlabelled


def summarise(rows: Sequence[DiffRow]) -> Counter:
    return Counter(row.status for row in rows)


def print_report(rows: Sequence[DiffRow], limit: int) -> None:
    counts = summarise(rows)
    reproduced = counts["identical"] + counts["whitespace"]
    compared = reproduced + counts["changed"] + counts["moved"] + counts["moved+changed"]

    print("Diff against the corpus on disk")
    for status in (
        "identical",
        "whitespace",
        "changed",
        "moved",
        "moved+changed",
        "new",
        "absent_from_export",
    ):
        if counts[status]:
            print(f"  {status:<19} {counts[status]:>5}")
    if compared:
        print(
            f"  reproduction        {reproduced}/{compared} "
            f"({100 * reproduced / compared:.1f}%) after whitespace normalisation; "
            f"{counts['identical']}/{compared} "
            f"({100 * counts['identical'] / compared:.1f}%) byte-identical"
        )

    for status in ("changed", "moved", "moved+changed", "new", "absent_from_export"):
        listed = [row for row in rows if row.status == status]
        if not listed:
            continue
        print(f"\n{status} ({len(listed)}):")
        for row in listed[:limit]:
            location = f"{row.existing_dir} -> {row.export_dir}".strip(" ->")
            print(f"  {row.unit_id:<20} {location}")
            if row.detail:
                print(f"      {row.detail}")
        if len(listed) > limit:
            print(f"  ... {len(listed) - limit} more (use --limit or --report)")


def write_report(rows: Sequence[DiffRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["unit_id", "unit_type", "status", "existing_dir", "export_dir", "detail"]
        )
        for row in rows:
            writer.writerow(
                [row.unit_id, row.unit_type or "", row.status, row.existing_dir,
                 row.export_dir, row.detail]
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CCL TEI-P5 exports to the canon plain-text layout.",
    )
    parser.add_argument("tei", nargs="+", help="TEI XML export file(s).")
    parser.add_argument(
        "--out-dir",
        help="Write the converted tree here (canon_labelled/ and canon_unlabelled/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write nothing; diff the export against --data-root instead.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Corpus root holding canon_labelled/ and canon_unlabelled/ (default: data).",
    )
    parser.add_argument(
        "--types",
        help="Comma-separated div/@type values to convert (default: all units).",
    )
    parser.add_argument("--report", help="Write the full diff to this CSV path.")
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Rows printed per diff category (default: 20).",
    )
    args = parser.parse_args(argv)
    if not args.out_dir and not args.dry_run:
        parser.error("pass --out-dir to write files or --dry-run to only diff")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    types = [t.strip() for t in args.types.split(",")] if args.types else None

    units: List[Unit] = []
    for tei_path in args.tei:
        parsed, warnings = parse_units(Path(tei_path), types=types)
        print(f"{tei_path}: {len(parsed)} units")
        by_type = Counter(unit.unit_type for unit in parsed)
        for unit_type, count in sorted(by_type.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            print(f"    {str(unit_type):<16} {count:>5}")
        n_labelled = sum(1 for unit in parsed if unit.directory)
        print(f"    labelled {n_labelled}, unlabelled {len(parsed) - n_labelled}")
        for warning in warnings:
            print(f"    warning: {warning}")
        units.extend(parsed)

    if args.out_dir:
        n_labelled, n_unlabelled = write_units(units, Path(args.out_dir))
        print(
            f"\nWrote {n_labelled} labelled and {n_unlabelled} unlabelled files "
            f"to {args.out_dir}"
        )

    if args.dry_run or args.report:
        rows = diff_units(units, index_corpus(Path(args.data_root)))
        print()
        print_report(rows, args.limit)
        if args.report:
            write_report(rows, Path(args.report))
            print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
