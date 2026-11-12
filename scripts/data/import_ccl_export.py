#!/usr/bin/env python3
"""Import a CCL plain-text export into a webapp corpus and report what changed.

CCL sends plain-text exports of the whole corpus (issue #282), laid out as::

    cclexport/keyed/<type>/<key>/<xml_id>.txt
    cclexport/unkeyed/<type>/<xml_id>.txt

This script turns one export, restricted to chosen unit types (``--types``,
default ``canon``), into a corpus in the layout the webapp and
``scripts/resubmit/index_unlabelled.py`` already read::

    <out-dir>/canon_labelled/<key>/<xml_id>.txt     one directory per key
    <out-dir>/canon_unlabelled/<xml_id>.txt         flat query set
    <out-dir>/witnesses.csv                         xml_id, type, status, directory, sha256
    <out-dir>/import_report.json                    the full change report
    <out-dir>/import_report.md                      a short summary of it

and compares it with a baseline corpus (``--baseline``, default the repo's
``data/``, i.e. ``data/canon_labelled`` + ``data/canon_unlabelled``).

It never writes into the baseline. ``data/canon_labelled`` is frozen benchmark
v1 (``docs/research/benchmark_v1.md``) and an import is webapp / v2 material
only, so ``--out-dir`` must lie outside every git working tree: CCL text is
never committed to this public repo.

Untrusted input
---------------
The export (a folder or the zip itself) is treated as untrusted:

* A zip is read through its own entry names, never through the filesystem.
  Names are decoded as UTF-8 (the 2026-11-12 zip does not set the UTF-8 flag,
  so Python would otherwise read them as cp437). Encrypted entries, symlink
  entries, absolute paths, ``.``/``..`` components, backslashes and entries
  larger than ``MAX_FILE_BYTES`` are refused.
* A folder is walked without following symlinks; any symlink or special file is
  refused, and files are opened with ``O_NOFOLLOW``.
* Every name is NFC-normalised (macOS unzipping, and some zip entries, give
  NFD), and two entries that collide after NFC, or after case folding (they
  would merge on a case-insensitive filesystem), are refused.
* The layout is checked strictly: nothing outside ``cclexport/``, only
  ``keyed``/``unkeyed`` below it, ``.txt`` files only at the expected depth,
  UTF-8 text, and no ``xml_id`` twice anywhere in the export (all types, not
  only the selected ones). ``.DS_Store`` files and a top-level ``__MACOSX/``
  are skipped and listed in the report; anything else unexpected is refused.
* Output is written to a fresh sibling directory with exclusive,
  non-following opens and renamed into place only when everything succeeded.

Directory names
---------------
A key is matched on its NFC form with every line-break character (CR, LF)
deleted and the remaining whitespace runs collapsed to one space. 19 long
citation keys in our corpus contain a line break; the macOS-unzipped 2026-11-12
export glues the two words together, so the two forms differ only by that
break. Any other space is significant: ``Ep. 1 23`` and ``Ep. 12 3`` are two
keys.

1. A key that matches a baseline directory this way is written under the
   baseline's directory name, verbatim. Every such case where the export's
   form differs is listed in the report (``key_name_cases``).
2. A key the baseline does not hold is written under the export's form, with
   every whitespace run that contains a character other than a plain space
   (line break, tab) replaced by one space, and leading and trailing
   whitespace removed. Each changed name is listed in the report.
3. A key is refused when it is empty, starts with ``.``, contains ``/``, a
   backslash, NUL or another control character besides whitespace, or is
   longer than 255 bytes in UTF-8, and when two different keys of the
   selected types collapse to the same match form. ``/`` can never reach a
   key: it is the path separator, so such a key shows up as extra nesting and
   fails the layout check.

The report holds xml_ids, keys, counts and sha256 digests only: no witness
text and no reviewer data (this script never opens the feedback database).

Examples
--------
::

    python scripts/data/import_ccl_export.py ccldocs_plaintext_2026_11nov_12.zip \\
        --out-dir /scratch/ccl_import_20261112

    python scripts/data/import_ccl_export.py path/to/cclexport \\
        --types canon,regCanon,decrExcerpt --out-dir /scratch/ccl_import_multi
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import unicodedata
import uuid
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPORT_SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = REPO_ROOT / "data"

EXPORT_ROOT = "cclexport"
KEYED = "keyed"
UNKEYED = "unkeyed"
STATUSES = (KEYED, UNKEYED)

LABELLED_SUBDIR = "canon_labelled"
UNLABELLED_SUBDIR = "canon_unlabelled"
WITNESSES_CSV = "witnesses.csv"
REPORT_JSON = "import_report.json"
REPORT_MD = "import_report.md"

DEFAULT_TYPES = ("canon",)

#: Per-file and whole-export bounds on what is read (zip bomb guard). The
#: 2026-11-12 export is 3.8 MB uncompressed over 8,852 files.
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024

#: Longest name component most filesystems accept, in bytes.
MAX_NAME_BYTES = 255

#: OS artefacts skipped (and listed in the report) rather than refused.
IGNORED_FILENAMES = frozenset({".DS_Store"})
IGNORED_TOP_DIRS = frozenset({"__MACOSX"})

TYPE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

#: How many ids each Markdown list shows before "... and N more".
MD_LIST_LIMIT = 25


class ExportError(ValueError):
    """The export (or the requested output) is not one this script accepts."""


# --------------------------------------------------------------------------
# Shared rules
# --------------------------------------------------------------------------


def nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name)


_LINE_BREAKS = re.compile(r"[\r\n]")


def key_match_form(key: str) -> str:
    """The form keys are matched on.

    NFC, then every line-break character (CR, LF) deleted, then the remaining
    whitespace runs collapsed to one space and the ends trimmed. Deleting the
    line break is what lets a key whose break was dropped without a space
    (``Gildas,82``) meet the form that kept it (``Gildas,\\n82``). Every other
    space is kept, so ``Ep. 1 23`` and ``Ep. 12 3`` stay different keys.
    """
    return " ".join(_LINE_BREAKS.sub("", nfc(key)).split())


_NON_SPACE_WHITESPACE_RUN = re.compile(r"\s*[^\S ]\s*")


def fresh_directory_name(key: str) -> str:
    """Directory name for a key the baseline does not hold (rule 2 above)."""
    return _NON_SPACE_WHITESPACE_RUN.sub(" ", key).strip()


def normalise_whitespace(text: str) -> str:
    """Whitespace normalisation for text comparison.

    The same rule as ``scripts/data/tei_to_canon.normalise`` and the
    ``" ".join(text.split())`` the extraction pipeline applies, so a
    whitespace-only change is one no model input sees.
    """
    return " ".join(text.split())


def corpus_digest(filename_by_query: Mapping[int, str]) -> str:
    """sha256 over ``"<file_id>\\t<filename>\\n"`` in file_id order.

    A copy of ``web.services.feedback_db.corpus_digest``, kept here so this
    script runs with the standard library only (no aiosqlite); the webapp test
    ``web/tests/test_ccl_import_digest.py`` proves the two agree.
    """
    text = "".join(
        f"{query_id}\t{filename_by_query[query_id]}\n"
        for query_id in sorted(filename_by_query)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def query_listing(filenames: Iterable[str]) -> Dict[int, str]:
    """file_id -> filename as ``index_unlabelled.py`` assigns it: sorted by name."""
    return dict(enumerate(sorted(filenames)))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _has_control(name: str, allow_whitespace: bool) -> bool:
    for char in name:
        if unicodedata.category(char) == "Cc":
            if allow_whitespace and char.isspace():
                continue
            return True
    return False


def _check_component(component: str, where: str) -> None:
    """Rules every path component obeys, whatever its role."""
    if component in ("", ".", ".."):
        raise ExportError(f"{where}: empty, '.' or '..' path component")
    if "\\" in component:
        raise ExportError(f"{where}: backslash in a name")
    if "\x00" in component:
        raise ExportError(f"{where}: NUL in a name")
    if len(component.encode("utf-8")) > MAX_NAME_BYTES:
        raise ExportError(f"{where}: name longer than {MAX_NAME_BYTES} bytes")


def _check_key(key: str, where: str) -> None:
    _check_component(key, where)
    if key.startswith("."):
        raise ExportError(f"{where}: key starts with '.'")
    if _has_control(key, allow_whitespace=True):
        raise ExportError(f"{where}: key contains a control character")
    if not key.strip():
        raise ExportError(f"{where}: key is blank")


def _xml_id_of(filename: str, where: str) -> str:
    if not filename.endswith(".txt"):
        raise ExportError(f"{where}: not a .txt file")
    xml_id = filename[: -len(".txt")]
    if not xml_id or xml_id.startswith("."):
        raise ExportError(f"{where}: empty or hidden xml_id")
    if any(char.isspace() for char in xml_id) or _has_control(xml_id, False):
        raise ExportError(f"{where}: whitespace or control character in xml_id")
    return xml_id


# --------------------------------------------------------------------------
# Reading the export
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Witness:
    xml_id: str
    unit_type: str
    status: str  # KEYED or UNKEYED
    key: Optional[str]
    data: bytes


@dataclass
class Export:
    kind: str  # "zip" or "folder"
    name: str
    sha256: Optional[str]
    witnesses: List[Witness]
    nfc_renamed: int = 0
    ignored: List[str] = field(default_factory=list)

    def types(self) -> Dict[str, Counter]:
        out: Dict[str, Counter] = defaultdict(Counter)
        for witness in self.witnesses:
            out[witness.unit_type][witness.status] += 1
        return out


#: One raw entry: (relative POSIX name, is_dir, loader of its bytes or None).
RawEntry = Tuple[str, bool, Optional[Callable[[], bytes]]]


def _zip_entry_name(info: zipfile.ZipInfo) -> str:
    """The entry's name as UTF-8, whatever the archiver's flag says."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        # zipfile decoded the raw bytes as cp437, which is lossless.
        return info.filename.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError) as exc:
        raise ExportError(f"zip entry {info.filename!r}: name is not UTF-8") from exc


def _read_bounded(handle: io.RawIOBase, where: str) -> bytes:
    data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ExportError(f"{where}: larger than {MAX_FILE_BYTES} bytes")
    return data


def _zip_entries(archive: zipfile.ZipFile) -> List[RawEntry]:
    entries: List[RawEntry] = []
    for info in archive.infolist():
        name = _zip_entry_name(info)
        where = f"zip entry {name!r}"
        if info.flag_bits & 0x1:
            raise ExportError(f"{where}: encrypted")
        mode = info.external_attr >> 16
        if info.create_system == 3 and stat.S_ISLNK(mode):
            raise ExportError(f"{where}: symlink entry")
        if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
            raise ExportError(f"{where}: absolute path")
        is_dir = name.endswith("/")
        if not is_dir and info.file_size > MAX_FILE_BYTES:
            raise ExportError(f"{where}: larger than {MAX_FILE_BYTES} bytes")

        def loader(info: zipfile.ZipInfo = info, where: str = where) -> bytes:
            with archive.open(info) as handle:
                return _read_bounded(handle, where)

        entries.append((name.rstrip("/") if is_dir else name, is_dir, None if is_dir else loader))
    return entries


def _folder_entries(root: Path) -> List[RawEntry]:
    """Every entry under ``root`` (the ``cclexport`` folder), never following links."""
    entries: List[RawEntry] = []

    def walk(directory: Path, rel: str) -> None:
        with os.scandir(directory) as scan:
            children = sorted(scan, key=lambda e: e.name)
        for entry in children:
            try:
                entry.name.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ExportError(f"{directory}/{entry.name!r}: name is not UTF-8") from exc
            child_rel = f"{rel}/{entry.name}"
            if entry.is_symlink():
                raise ExportError(f"{child_rel!r}: symlink (refused, never followed)")
            if entry.is_dir(follow_symlinks=False):
                entries.append((child_rel, True, None))
                walk(Path(entry.path), child_rel)
            elif entry.is_file(follow_symlinks=False):
                path = entry.path

                def loader(path: str = path, where: str = child_rel) -> bytes:
                    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(fd, "rb") as handle:
                        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                            raise ExportError(f"{where!r}: not a regular file")
                        return _read_bounded(handle, repr(where))

                entries.append((child_rel, False, loader))
            else:
                raise ExportError(f"{child_rel!r}: not a regular file or directory")

    walk(root, EXPORT_ROOT)
    return entries


def _parse_entries(entries: Sequence[RawEntry]) -> Tuple[List[Witness], int, List[str]]:
    """Validate the layout and load every witness. Returns (witnesses, nfc_renamed, ignored)."""
    witnesses: List[Witness] = []
    seen_names: Dict[str, str] = {}
    folded_names: Dict[str, str] = {}
    seen_ids: Dict[str, str] = {}
    folded_ids: Dict[str, str] = {}
    ignored: List[str] = []
    nfc_renamed = 0
    total = 0

    for raw_name, is_dir, loader in entries:
        name = nfc(raw_name)
        if name != raw_name:
            nfc_renamed += 1
        parts = name.split("/")
        where = repr(name)
        if parts[0] in IGNORED_TOP_DIRS:
            ignored.append(name + ("/" if is_dir else ""))
            continue
        if not is_dir and parts[-1] in IGNORED_FILENAMES:
            ignored.append(name)
            continue
        if parts[0] != EXPORT_ROOT:
            raise ExportError(f"{where}: outside '{EXPORT_ROOT}/'")
        for component in parts:
            _check_component(component, where)
        if name in seen_names:
            raise ExportError(f"{where}: appears twice (after NFC normalisation)")
        seen_names[name] = raw_name
        folded = name.casefold()
        if folded in folded_names:
            raise ExportError(
                f"{where} and {folded_names[folded]!r} differ only in case; they "
                "would merge on a case-insensitive filesystem"
            )
        folded_names[folded] = name

        if len(parts) >= 2 and parts[1] not in STATUSES:
            raise ExportError(f"{where}: expected '{KEYED}' or '{UNKEYED}' below '{EXPORT_ROOT}/'")
        if len(parts) >= 3 and not TYPE_PATTERN.match(parts[2]):
            raise ExportError(f"{where}: unexpected unit type name {parts[2]!r}")
        if len(parts) >= 4 and parts[1] == KEYED:
            _check_key(parts[3], where)

        file_depth = 5 if len(parts) >= 2 and parts[1] == KEYED else 4
        if is_dir:
            if len(parts) >= file_depth:
                raise ExportError(f"{where}: directory nested deeper than the layout allows")
            continue
        if len(parts) != file_depth:
            raise ExportError(
                f"{where}: file at an unexpected depth (expected "
                f"{EXPORT_ROOT}/{KEYED}/<type>/<key>/<xml_id>.txt or "
                f"{EXPORT_ROOT}/{UNKEYED}/<type>/<xml_id>.txt)"
            )
        xml_id = _xml_id_of(parts[-1], where)
        if xml_id in seen_ids:
            raise ExportError(
                f"xml_id {xml_id!r} appears twice in the export: {seen_ids[xml_id]!r} and {where}"
            )
        seen_ids[xml_id] = where
        if xml_id.casefold() in folded_ids:
            raise ExportError(
                f"xml_ids {folded_ids[xml_id.casefold()]!r} and {xml_id!r} differ only in case"
            )
        folded_ids[xml_id.casefold()] = xml_id

        assert loader is not None
        data = loader()
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ExportError(f"export is larger than {MAX_TOTAL_BYTES} bytes uncompressed")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExportError(f"{where}: text is not valid UTF-8 ({exc.reason})") from exc
        status = parts[1]
        witnesses.append(
            Witness(
                xml_id=xml_id,
                unit_type=parts[2],
                status=status,
                key=parts[3] if status == KEYED else None,
                data=data,
            )
        )
    if not witnesses:
        raise ExportError("the export holds no witness files")
    return witnesses, nfc_renamed, ignored


def read_export(path: Path) -> Export:
    """Read a CCL export from the zip itself or from the unzipped folder.

    A folder may be the ``cclexport`` folder or a folder whose only entry (OS
    artefacts aside) is ``cclexport``.
    """
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise ExportError(f"{path}: symlink (refused, never followed)")
    if stat.S_ISREG(st.st_mode):
        if not zipfile.is_zipfile(path):
            raise ExportError(f"{path}: a file that is not a zip archive")
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        with zipfile.ZipFile(path) as archive:
            witnesses, renamed, ignored = _parse_entries(_zip_entries(archive))
        return Export("zip", path.name, digest.hexdigest(), witnesses, renamed, ignored)
    if not stat.S_ISDIR(st.st_mode):
        raise ExportError(f"{path}: neither a zip file nor a folder")
    root = path
    if nfc(path.name) != EXPORT_ROOT:
        children = [c for c in os.listdir(path) if c not in IGNORED_FILENAMES]
        if children != [EXPORT_ROOT]:
            raise ExportError(
                f"{path}: expected the '{EXPORT_ROOT}' folder or a folder holding only it"
            )
        root = path / EXPORT_ROOT
        if os.path.islink(root) or not root.is_dir():
            raise ExportError(f"{root}: not a plain directory")
    witnesses, renamed, ignored = _parse_entries(_folder_entries(root))
    return Export("folder", path.name, None, witnesses, renamed, ignored)


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineWitness:
    xml_id: str
    directory: Optional[str]  # None for the unlabelled pool
    data: bytes

    @property
    def status(self) -> str:
        return KEYED if self.directory is not None else UNKEYED


def read_baseline(root: Path) -> Dict[str, BaselineWitness]:
    """``root/canon_labelled/<dir>/*.txt`` and ``root/canon_unlabelled/*.txt``, by xml_id."""
    labelled = root / LABELLED_SUBDIR
    unlabelled = root / UNLABELLED_SUBDIR
    if not labelled.is_dir() or not unlabelled.is_dir():
        raise ExportError(
            f"baseline {root} must hold {LABELLED_SUBDIR}/ and {UNLABELLED_SUBDIR}/"
        )
    out: Dict[str, BaselineWitness] = {}

    def add(path: Path, directory: Optional[str]) -> None:
        if path.is_symlink() or not path.is_file():
            raise ExportError(f"baseline {path}: not a regular file")
        xml_id = nfc(_xml_id_of(path.name, f"baseline {path}"))
        if xml_id in out:
            raise ExportError(f"baseline holds xml_id {xml_id!r} twice")
        out[xml_id] = BaselineWitness(xml_id, directory, path.read_bytes())

    for path in sorted(unlabelled.glob("*.txt")):
        add(path, None)
    for directory in sorted(labelled.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.txt")):
            add(path, nfc(directory.name))
    return out


# --------------------------------------------------------------------------
# Planning and comparing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyNameCase:
    export_key: str
    directory: str
    baseline_directory: Optional[str]
    reason: str


def plan_directories(
    keys: Iterable[str], baseline_dirs: Iterable[str]
) -> Tuple[Dict[str, str], List[KeyNameCase]]:
    """Directory name for every export key, and every case where it is not the key itself."""
    by_form: Dict[str, str] = {}
    for directory in sorted(set(baseline_dirs)):
        form = key_match_form(directory)
        if form in by_form:
            raise ExportError(
                f"baseline directories {by_form[form]!r} and {directory!r} differ only "
                "by line breaks or repeated whitespace, so a key cannot be matched to "
                "one of them"
            )
        by_form[form] = directory

    export_forms: Dict[str, str] = {}
    plan: Dict[str, str] = {}
    cases: List[KeyNameCase] = []
    for key in sorted(set(keys)):
        form = key_match_form(key)
        if form in export_forms:
            raise ExportError(
                f"export keys {export_forms[form]!r} and {key!r} differ only by "
                "line breaks or repeated whitespace; they cannot be kept apart"
            )
        export_forms[form] = key
        baseline = by_form.get(form)
        if baseline is not None:
            plan[key] = baseline
            if baseline != key:
                cases.append(KeyNameCase(key, baseline, baseline, "kept_baseline_whitespace"))
            continue
        directory = fresh_directory_name(key)
        _check_key(directory, f"key {key!r}")
        plan[key] = directory
        if directory != key:
            cases.append(KeyNameCase(key, directory, None, "whitespace_collapsed"))

    folded: Dict[str, str] = {}
    for directory in plan.values():
        other = folded.setdefault(directory.casefold(), directory)
        if other != directory:
            raise ExportError(
                f"directories {other!r} and {directory!r} differ only in case; they "
                "would merge on a case-insensitive filesystem"
            )
    return plan, cases


def parse_types(raw: str, export: Export) -> List[str]:
    present = sorted(export.types())
    if raw.strip() == "all":
        return present
    wanted = [t.strip() for t in raw.split(",") if t.strip()]
    if not wanted:
        raise ExportError("--types is empty")
    missing = [t for t in wanted if t not in present]
    if missing:
        raise ExportError(
            f"type(s) {', '.join(missing)} not in the export (it holds: {', '.join(present)})"
        )
    return sorted(set(wanted))


def _count_by(rows: Iterable[dict], *fields: str) -> dict:
    out: dict = {}
    for row in rows:
        node = out
        for name in fields[:-1]:
            node = node.setdefault(str(row[name]), {})
        leaf = str(row[fields[-1]])
        node[leaf] = node.get(leaf, 0) + 1
    return out


def build_report(
    export: Export,
    types: Sequence[str],
    baseline: Mapping[str, BaselineWitness],
    baseline_path: str,
) -> Tuple[dict, Dict[str, str]]:
    """The change report and the key -> directory plan."""
    selected = [w for w in export.witnesses if w.unit_type in types]
    by_id = {w.xml_id: w for w in selected}
    everywhere = {w.xml_id: w for w in export.witnesses}

    baseline_dirs = {b.directory for b in baseline.values() if b.directory is not None}
    plan, cases = plan_directories(
        (w.key for w in selected if w.key is not None), baseline_dirs
    )

    def directory_of(witness: Witness) -> Optional[str]:
        return plan[witness.key] if witness.key is not None else None

    added, removed, text_changed, key_changed = [], [], [], []
    became_keyed, became_unkeyed, empty_texts = [], [], []
    for witness in sorted(selected, key=lambda w: w.xml_id):
        if not witness.data.decode("utf-8").strip():
            empty_texts.append({"xml_id": witness.xml_id, "type": witness.unit_type})
        old = baseline.get(witness.xml_id)
        if old is None:
            added.append(
                {"xml_id": witness.xml_id, "type": witness.unit_type,
                 "status": witness.status, "directory": directory_of(witness)}
            )
            continue
        if old.data != witness.data:
            whitespace_only = normalise_whitespace(
                old.data.decode("utf-8")
            ) == normalise_whitespace(witness.data.decode("utf-8"))
            text_changed.append(
                {"xml_id": witness.xml_id, "type": witness.unit_type,
                 "whitespace_only": whitespace_only,
                 "baseline_sha256": sha256_bytes(old.data),
                 "export_sha256": sha256_bytes(witness.data)}
            )
        new_dir = directory_of(witness)
        if old.directory is None and new_dir is not None:
            became_keyed.append(
                {"xml_id": witness.xml_id, "type": witness.unit_type, "directory": new_dir}
            )
        elif old.directory is not None and new_dir is None:
            became_unkeyed.append(
                {"xml_id": witness.xml_id, "type": witness.unit_type,
                 "baseline_directory": old.directory}
            )
        elif old.directory is not None and old.directory != new_dir:
            key_changed.append(
                {"xml_id": witness.xml_id, "type": witness.unit_type,
                 "baseline_directory": old.directory, "directory": new_dir}
            )
    for xml_id in sorted(set(baseline) - set(by_id)):
        old = baseline[xml_id]
        elsewhere = everywhere.get(xml_id)
        removed.append(
            {"xml_id": xml_id, "status": old.status, "baseline_directory": old.directory,
             "in_export_as": (
                 {"type": elsewhere.unit_type, "status": elsewhere.status}
                 if elsewhere is not None else None
             )}
        )

    # --- the webapp's query corpus ---
    new_queries = sorted(f"{w.xml_id}.txt" for w in selected if w.key is None)
    old_queries = sorted(f"{b.xml_id}.txt" for b in baseline.values() if b.directory is None)
    new_digest = corpus_digest(query_listing(new_queries))
    old_digest = corpus_digest(query_listing(old_queries))
    became_keyed_ids = {row["xml_id"] for row in became_keyed}
    leaving = []
    for name in sorted(set(old_queries) - set(new_queries)):
        xml_id = name[: -len(".txt")]
        if xml_id in became_keyed_ids:
            reason = "became_keyed"
        elif xml_id in everywhere:
            reason = "type_not_selected"
        else:
            reason = "absent_from_export"
        new_dir = directory_of(by_id[xml_id]) if xml_id in by_id else None
        leaving.append({"xml_id": xml_id, "reason": reason, "directory": new_dir})
    joining = sorted(n[: -len(".txt")] for n in set(new_queries) - set(old_queries))

    new_dirs = {d for d in plan.values()}
    labelled_changed = (
        new_dirs != baseline_dirs
        or {w.xml_id: directory_of(w) for w in selected if w.key is not None}
        != {b.xml_id: b.directory for b in baseline.values() if b.directory is not None}
        or any(
            baseline[w.xml_id].data != w.data
            for w in selected if w.key is not None and w.xml_id in baseline
        )
    )
    shared = defaultdict(set)
    for witness in selected:
        if witness.key is not None:
            shared[plan[witness.key]].add(witness.unit_type)
    shared_across_types = sorted(d for d, t in shared.items() if len(t) > 1)

    summary = {
        "added": len(added),
        "removed": len(removed),
        "text_changed_bytes": len(text_changed),
        "text_changed_after_whitespace_normalisation": sum(
            not row["whitespace_only"] for row in text_changed
        ),
        "key_changed": len(key_changed),
        "became_keyed": len(became_keyed),
        "became_unkeyed": len(became_unkeyed),
        "key_name_cases": len(cases),
        "key_name_cases_kept_baseline_whitespace": sum(
            c.reason == "kept_baseline_whitespace" for c in cases
        ),
        "empty_texts": len(empty_texts),
        "added_by_type_status": _count_by(added, "type", "status"),
        "removed_by_status": _count_by(removed, "status"),
    }

    forecast = _startup_forecast(
        query_changed=new_digest != old_digest,
        labelled_changed=labelled_changed,
        leaving=leaving,
        joining=joining,
        old_size=len(old_queries),
    )

    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "scripts/data/import_ccl_export.py",
        "export": {
            "kind": export.kind,
            "name": export.name,
            "sha256": export.sha256,
            "witnesses": len(export.witnesses),
            "types": {t: dict(sorted(c.items())) for t, c in sorted(export.types().items())},
            "names_nfc_normalised": export.nfc_renamed,
            "ignored_entries": export.ignored,
        },
        "types_selected": list(types),
        "baseline": {
            "path": baseline_path,
            "keyed": sum(b.directory is not None for b in baseline.values()),
            "unkeyed": sum(b.directory is None for b in baseline.values()),
            "directories": len(baseline_dirs),
        },
        "output": {
            "keyed": sum(w.key is not None for w in selected),
            "unkeyed": sum(w.key is None for w in selected),
            "directories": len(new_dirs),
            "directories_shared_across_types": shared_across_types,
        },
        "summary": summary,
        "added": added,
        "removed": removed,
        "text_changed": text_changed,
        "key_changed": key_changed,
        "became_keyed": became_keyed,
        "became_unkeyed": became_unkeyed,
        "key_name_cases": [c.__dict__ for c in cases],
        "empty_texts": empty_texts,
        "webapp": {
            "query_corpus": {
                "baseline_sha256": old_digest,
                "baseline_size": len(old_queries),
                "sha256": new_digest,
                "size": len(new_queries),
                "changed": new_digest != old_digest,
            },
            "labelled_pool_changed": labelled_changed,
            "leaving_query_set": leaving,
            "joining_query_set": joining,
            "startup_forecast": forecast,
        },
    }
    return report, plan


def _startup_forecast(
    *, query_changed: bool, labelled_changed: bool, leaving: list, joining: list, old_size: int
) -> List[str]:
    """What #278's startup checks would do if the webapp served this corpus."""
    notes: List[str] = []
    if not query_changed:
        notes.append(
            "Query corpus digest unchanged: adopt_corpus() sees no corpus change and "
            "re-keys nothing; the current release's predictions and q-q matrices still "
            "address the same file ids."
        )
    else:
        notes.append(
            "Query corpus digest changed: verify_release_matches_corpus() refuses to start "
            "on any data release built from another listing, so predictions and q-q "
            "matrices for every model and variant must be rebuilt from this listing first."
        )
        notes.append(
            "adopt_corpus() will treat it as a corpus change: it refuses if any stored row "
            "lacks an xml_id (UnkeyedRowsError), otherwise backs up the feedback DB and "
            "records the re-keying by xml_id."
        )
        by_reason = Counter(row["reason"] for row in leaving)
        if leaving:
            detail = ", ".join(f"{n} {r}" for r, n in sorted(by_reason.items()))
            notes.append(
                f"{len(leaving)} of the {old_size} current queries leave the query set "
                f"({detail}). adopt_corpus() counts each reviewed one as missing and "
                "refuses to start (MissingWitnessThresholdError) when more than 5% of "
                "the reviewed witnesses are missing. This script cannot see the "
                "production DB: check leaving_query_set against the CCL key assignments "
                "export before deploying. Witnesses that became keyed are expected to "
                "leave; until #278's threshold counts them as resolved, an intended "
                "change above the limit needs corpus_change.accept_missing_reviewed_sha256 "
                "set to the new digest."
            )
        if joining:
            notes.append(f"{len(joining)} witnesses join the query set (unreviewed).")
    if labelled_changed:
        notes.append(
            "The labelled pool differs from the baseline (directories, membership or "
            "text), so predictions must be rebuilt even where the query listing is "
            "unchanged."
        )
    return notes


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _inside_git_worktree(path: Path) -> Optional[Path]:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def check_out_dir(out_dir: Path, *, forbidden: Sequence[Path]) -> Path:
    resolved = Path(os.path.abspath(out_dir))
    real = Path(os.path.realpath(resolved))
    for candidate in {resolved, real}:
        worktree = _inside_git_worktree(candidate)
        if worktree is not None:
            raise ExportError(
                f"--out-dir {out_dir} is inside the git working tree {worktree}; CCL "
                "text must never be committed, so write the corpus outside any repo"
            )
        for other in forbidden:
            other_real = Path(os.path.realpath(other))
            if candidate == other_real or other_real in candidate.parents or candidate in other_real.parents:
                raise ExportError(f"--out-dir {out_dir} overlaps {other}")
    if os.path.lexists(resolved):
        if os.path.islink(resolved) or not resolved.is_dir() or any(resolved.iterdir()):
            raise ExportError(f"--out-dir {out_dir} exists and is not an empty directory")
    if not resolved.parent.is_dir():
        raise ExportError(f"--out-dir parent {resolved.parent} does not exist")
    return resolved


def _write_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def _child(parent: Path, name: str) -> Path:
    """``parent / name`` for one validated component, never escaping ``parent``."""
    _check_component(name, f"output name {name!r}")
    if "/" in name:
        raise ExportError(f"output name {name!r} contains '/'")
    child = parent / name
    if child.parent != parent:
        raise ExportError(f"output name {name!r} escapes {parent}")
    return child


def write_corpus(
    staging: Path, witnesses: Sequence[Witness], plan: Mapping[str, str]
) -> List[dict]:
    labelled = staging / LABELLED_SUBDIR
    unlabelled = staging / UNLABELLED_SUBDIR
    labelled.mkdir()
    unlabelled.mkdir()
    rows = []
    for witness in sorted(witnesses, key=lambda w: w.xml_id):
        if witness.key is not None:
            directory = plan[witness.key]
            target_dir = _child(labelled, directory)
            if not target_dir.exists():
                target_dir.mkdir()
        else:
            directory = None
            target_dir = unlabelled
        _write_new(_child(target_dir, f"{witness.xml_id}.txt"), witness.data)
        rows.append(
            {"xml_id": witness.xml_id, "type": witness.unit_type, "status": witness.status,
             "directory": directory or "", "sha256": sha256_bytes(witness.data)}
        )
    return rows


def render_markdown(report: dict) -> str:
    s = report["summary"]
    q = report["webapp"]["query_corpus"]
    exp = report["export"]
    lines = [
        "# CCL export import report",
        "",
        f"Export `{exp['name']}` ({exp['kind']}, {exp['witnesses']} witnesses"
        + (f", sha256 `{exp['sha256']}`" if exp["sha256"] else "")
        + f"), types {', '.join(report['types_selected'])}, against baseline "
        f"`{report['baseline']['path']}`. Generated {report['generated_at']}.",
        "",
        "| | Count |",
        "|---|---|",
        f"| Output keyed / unkeyed / directories | {report['output']['keyed']} / "
        f"{report['output']['unkeyed']} / {report['output']['directories']} |",
        f"| Baseline keyed / unkeyed / directories | {report['baseline']['keyed']} / "
        f"{report['baseline']['unkeyed']} / {report['baseline']['directories']} |",
        f"| Added | {s['added']} |",
        f"| Removed | {s['removed']} |",
        f"| Text changed (bytes) | {s['text_changed_bytes']} |",
        f"| Text changed after whitespace normalisation | "
        f"{s['text_changed_after_whitespace_normalisation']} |",
        f"| Key changed | {s['key_changed']} |",
        f"| Unkeyed to keyed | {s['became_keyed']} |",
        f"| Keyed to unkeyed | {s['became_unkeyed']} |",
        f"| Key names kept in the baseline's whitespace form | "
        f"{s['key_name_cases_kept_baseline_whitespace']} |",
        f"| Other key-name normalisations | "
        f"{s['key_name_cases'] - s['key_name_cases_kept_baseline_whitespace']} |",
        f"| Directories holding more than one type | "
        f"{len(report['output']['directories_shared_across_types'])} |",
        f"| Names NFC-normalised on read | {exp['names_nfc_normalised']} |",
        f"| Empty texts (kept) | {s['empty_texts']} |",
        "",
        f"Query corpus digest: `{q['baseline_sha256']}` ({q['baseline_size']}) -> "
        f"`{q['sha256']}` ({q['size']}), {'changed' if q['changed'] else 'unchanged'}.",
        "",
        "## What the webapp's startup checks (#278) would do",
        "",
    ]
    lines += [f"- {note}" for note in report["webapp"]["startup_forecast"]]

    def listing(title: str, items: List[str]) -> None:
        if not items:
            return
        lines.extend(["", f"## {title} ({len(items)})", ""])
        lines.extend(f"- {item}" for item in items[:MD_LIST_LIMIT])
        if len(items) > MD_LIST_LIMIT:
            lines.append(f"- ... and {len(items) - MD_LIST_LIMIT} more (see the JSON report)")

    listing("Leaving the query set", [
        f"`{r['xml_id']}` ({r['reason']})" for r in report["webapp"]["leaving_query_set"]
    ])
    listing("Added", [f"`{r['xml_id']}` ({r['type']}, {r['status']})" for r in report["added"]])
    listing("Removed", [f"`{r['xml_id']}` ({r['status']})" for r in report["removed"]])
    listing("Text changed", [
        f"`{r['xml_id']}` ({'whitespace only' if r['whitespace_only'] else 'content'})"
        for r in report["text_changed"]
    ])
    listing("Key changed", [
        f"`{r['xml_id']}`: {json.dumps(r['baseline_directory'], ensure_ascii=False)} -> "
        f"{json.dumps(r['directory'], ensure_ascii=False)}" for r in report["key_changed"]
    ])
    listing("Key-name cases", [
        f"{json.dumps(c['export_key'], ensure_ascii=False)} written as "
        f"{json.dumps(c['directory'], ensure_ascii=False)} ({c['reason']})"
        for c in report["key_name_cases"]
    ])
    return "\n".join(lines) + "\n"


def run(
    export_path: Path,
    out_dir: Path,
    *,
    types: str = ",".join(DEFAULT_TYPES),
    baseline_root: Path = DEFAULT_BASELINE,
    dry_run: bool = False,
) -> dict:
    """Import, write and return the report. Raises ExportError on any refusal."""
    target = check_out_dir(out_dir, forbidden=[export_path, baseline_root])
    export = read_export(export_path)
    selected_types = parse_types(types, export)
    baseline = read_baseline(baseline_root)
    report, plan = build_report(export, selected_types, baseline, str(baseline_root))
    selected = [w for w in export.witnesses if w.unit_type in selected_types]

    staging = target.parent / f".{target.name}.partial-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        if not dry_run:
            rows = write_corpus(staging, selected, plan)
            buffer = io.StringIO()
            writer = csv.DictWriter(
                buffer, fieldnames=["xml_id", "type", "status", "directory", "sha256"]
            )
            writer.writeheader()
            writer.writerows(rows)
            _write_new(staging / WITNESSES_CSV, buffer.getvalue().encode("utf-8"))
        _write_new(
            staging / REPORT_JSON,
            (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        _write_new(staging / REPORT_MD, render_markdown(report).encode("utf-8"))
        if os.path.lexists(target):
            target.rmdir()  # checked empty above
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a CCL plain-text export into a webapp corpus and report changes.",
    )
    parser.add_argument("export", type=Path, help="The export zip, or the unzipped cclexport folder.")
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="New or empty directory outside any git working tree.",
    )
    parser.add_argument(
        "--types", default=",".join(DEFAULT_TYPES),
        help="Comma-separated unit types to import, or 'all' (default: canon).",
    )
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE,
        help="Corpus root holding canon_labelled/ and canon_unlabelled/ (default: repo data/).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write only the report into --out-dir, not the corpus.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run(
            args.export, args.out_dir, types=args.types,
            baseline_root=args.baseline, dry_run=args.dry_run,
        )
    except (ExportError, OSError) as exc:
        print(f"import_ccl_export: refused: {exc}", file=sys.stderr)
        return 2
    print(render_markdown(report), end="")
    print(f"\nWrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
