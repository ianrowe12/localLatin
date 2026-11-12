"""The CCL key assignments export (issue #270): one resolved row per query.

`/api/feedback/export` is the raw review log, one row per reviewer, model,
variant and save. CCL's import needs the opposite: one row per reviewed
unlabelled witness (`xml_id`), saying which source key the reviewers assigned
to it, or that they disagree, or that nobody named one. This module is that
resolution. It is pure: `FeedbackDB.ccl_key_assignments` reads the rows and
hands them here, so the rule can be tested without a database.

THE RULE, in the order it is applied:

1. Only decisions count: `matched_rank` and `none_of_top_k` rows, the same two
   outcomes that make a query `reviewed` everywhere else in the app
   (`FeedbackDB.get_query_statuses`, `get_reviewed_query_ids`, the stats). So
   the set of queries in this export is exactly the set the dashboard calls
   reviewed. `skipped` and `legacy_unresolved` rows are ignored, and a later
   skip does not retract an earlier decision, just as it does not turn a
   reviewed query back into a skipped one on the dashboard.
2. A reviewer's current decision on a query is their newest decision row on
   each (model, variant) they reviewed it under, newest by (timestamp, id).
   A query is its witness `xml_id`: the one stored on the row (issue #273),
   or for a row without one the name the served corpus gives its query id.
   That is the "own decision" half of `GET /api/feedback/latest`, which is also
   keyed by model and variant. A newer save on the same model and variant
   replaces an older one; a save under a different model does not, because the
   app itself still shows the reviewer both answers.
3. Each current decision gives an answer. A `matched_rank` row names the
   directory behind every selected rank (a labelled directory names itself; a
   reviewer directory names its `ccl_key`). A `none_of_top_k` row names its
   typed `ccl_key`, and one without a key is the answer "none of these", which
   rejects every candidate it was made against (`NONE_MARKER`).
4. A pick is trusted only when it can be shown to be what the reviewer saw
   (`resolve_picks`). A row saved since issue #273 stores the directory behind
   every selected rank (`selected_dirs_json`, resolved server-side from the
   snapshot the reviewer saw), and every pick is trusted as stored. An older
   row stores only the first selection's directory (`correct_dir`), so a
   single pick is trusted as stored, and later selections of a multi-rank pick
   are read from the ranking the server serves now, and only after the
   stored first pick is confirmed to still sit at its rank there. A pick that
   cannot be confirmed (the anchor moved, the rank or query is no longer
   served, the row has no variant, the row stores no directory at all, or the
   rank is a legacy reviewer-directory rank above `MAX_MODEL_RANK`) names
   nothing and is reported as unresolved (`UNRESOLVED_MARKER`). So is a
   trusted pick of a reviewer directory that has no key. No key is ever taken
   from a pick nobody can be shown to have seen.
5. Keys are compared by `ccl_keys.match_form` (case-folded). Status, first rule
   that applies:
   - `needs_check`: some counted decision holds an unresolved pick. A human
     must look before importing.
   - `conflict`: two or more different keys are named (by different reviewers,
     one reviewer under two models, or one multi-rank pick), or a key is named
     and another counted decision is a no-key "none of these".
   - `agreed`: exactly one key, and every counted decision names it.
   - `no_key`: nobody named a key (every decision is a no-key "none of these").
6. A reviewer counts once (`reviewer_count`, `reviewers`) however many models
   they decided under. `reviewers` lists everyone with a counted decision,
   whatever they answered. Only an `agreed` row carries a `ccl_key`, and in an
   `agreed` row every listed reviewer's decisions named that key, so nobody is
   ever credited with a key they rejected.

DISPLAY FORM. Reviewers who typed `ctou.567.16` and `CTOU.567.16` agree, and one
form has to be printed. It is the labelled directory's own name when the key
names a labelled directory; otherwise the earliest key-shaped form any counted
decision used (CCL writes the source code in capitals, so `CTOU.567.16` beats
an earlier `ctou.567.16`); otherwise the form on the earliest counted decision.
Every candidate is a stored form; nothing is re-cased here.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from web.models import MAX_MODEL_RANK
from web.services import ccl_keys
from web.services.reviewer_dirs import is_reviewer_dir_id

#: Bumped whenever a column is added, removed, renamed or changes meaning.
EXPORT_SCHEMA_VERSION = 1

COLUMNS = [
    "xml_id",
    "ccl_key",
    "status",
    "conflicting_keys",
    "key_source",
    "suggested_by",
    "key_shaped",
    "reviewer_count",
    "reviewers",
    "first_decided_at",
    "last_decided_at",
    "export_schema_version",
    "exported_at",
    "data_release",
]

#: The outcomes that are a decision. The same pair `get_query_statuses` reads
#: as `reviewed`; keep them together.
DECISION_OUTCOMES = ("matched_rank", "none_of_top_k")

STATUSES = ("agreed", "conflict", "no_key", "needs_check")

#: In `conflicting_keys`: somebody pressed "None of the top N" without a key.
#: Parenthesised and spelled like the stored outcome, so it cannot be mistaken
#: for a CCL key or for any of the 840 labelled directory names.
NONE_MARKER = "(none_of_top_k)"

#: In `conflicting_keys`, and as the key of a `suggested_by` entry: a pick
#: whose directory could not be confirmed (rule point 4).
UNRESOLVED_MARKER = "(unresolved)"

#: Separator inside a multi-valued cell. Not "; " or ", ": 18 labelled
#: directory names contain a semicolon and 103 a comma (edition citations such
#: as "... lines 1-10; trans. ..."), and no labelled name contains "|".
LIST_SEPARATOR = " | "

MODEL_SUGGESTION = "model_suggestion"
REVIEWER_TYPED = "reviewer_typed"
BOTH = "both"


@dataclass(frozen=True)
class KeyClaim:
    """One key named by one decision row."""

    key: str
    source: str  # MODEL_SUGGESTION or REVIEWER_TYPED
    decided_at: str
    row_id: int
    model_slug: str
    variant: str | None
    rank: int | None = None


@dataclass
class Decision:
    """One reviewer's current decision on one (query, model, variant)."""

    query_id: int
    xml_id: str
    reviewer_key: tuple
    reviewer_name: str
    decided_at: str
    row_id: int
    claims: list[KeyClaim] = field(default_factory=list)
    #: A `none_of_top_k` with no key: "none of these".
    rejects_all: bool = False
    #: `(model_slug, variant, rank)` of every pick that could not be confirmed.
    unresolved: list[tuple] = field(default_factory=list)


def reviewer_identity(row: dict) -> tuple:
    """Who decided. The account when there is one, the stored name otherwise.

    Rows written before accounts existed have `reviewer_account_id` NULL and are
    told apart by the display name they were saved under.
    """
    account_id = row.get("reviewer_account_id")
    if account_id is not None:
        return ("account", int(account_id))
    return ("name", str(row.get("reviewer") or ""))


def current_decisions(rows: Iterable[dict]) -> list[dict]:
    """The newest decision row per (query, reviewer, model, variant).

    Rule points 1 and 2 of the module docstring. Non-decision outcomes are
    dropped before "newest" is taken, so a skip never displaces a decision.
    """
    newest: dict[tuple, dict] = {}
    for row in rows:
        if row.get("outcome") not in DECISION_OUTCOMES:
            continue
        slot = (
            query_identity(row),
            reviewer_identity(row),
            row.get("model_slug"),
            row.get("variant"),
        )
        held = newest.get(slot)
        if held is None or _order(row) > _order(held):
            newest[slot] = row
    return list(newest.values())


def query_identity(row: dict) -> str | int:
    """Which witness a row is about: its `xml_id`, else its stored query id."""
    return row.get("xml_id") or int(row["query_id"])


def stored_dirs(row: dict) -> list[str] | None:
    """The directory behind every selected rank, as saved with the row.

    None unless the row stores exactly one non-empty name per selected rank,
    which every `matched_rank` row saved since issue #273 does.
    """
    names = row.get("selected_dirs")
    ranks = selected_ranks(row)
    if not names or len(names) != len(ranks) or not all(names):
        return None
    return [str(name) for name in names]


def needs_ranking(row: dict) -> bool:
    """Whether a `matched_rank` row can only be read with the served ranking.

    A row that stores every picked directory cannot, nor can a single pick
    with a stored `correct_dir`; every other pick can.
    """
    if row.get("outcome") != "matched_rank":
        return False
    if stored_dirs(row) is not None:
        return False
    ranks = selected_ranks(row)
    return not (len(ranks) == 1 and row.get("correct_dir"))


def resolve_picks(
    row: dict, served: dict[int, str | None] | None
) -> tuple[list[tuple[int, str]], list[int | None]]:
    """Which picks of a `matched_rank` row are trusted (rule point 4).

    `served` maps rank to the directory at that rank in the ranking the server
    serves now for this row's query, model and variant, or is None when there
    is no such ranking (no variant, or the variant or query is not served).
    Returns the trusted `(rank, directory)` picks and the ranks left unresolved.
    """
    ranks = selected_ranks(row)
    if not ranks:
        return [], [None]
    names = stored_dirs(row)
    if names is not None:
        # Saved with the row from the snapshot the reviewer saw (issue #273):
        # no ranking is consulted, today's or any other.
        return list(zip(ranks, names)), []
    stored = row.get("correct_dir") or None
    trusted: list[tuple[int, str]] = []
    unresolved: list[int | None] = []
    first, later = ranks[0], ranks[1:]
    if first > MAX_MODEL_RANK or not stored:
        # A legacy reviewer-directory rank, or a row with nothing stored:
        # nothing ties it to what the reviewer saw.
        unresolved.append(first)
    else:
        trusted.append((first, stored))
    # Later picks are read from today's ranking only when that ranking still
    # holds the stored first pick at its rank, i.e. it has not moved.
    anchored = bool(trusted) and served is not None and served.get(first) == stored
    for rank in later:
        directory = served.get(rank) if anchored and served is not None else None
        if rank <= MAX_MODEL_RANK and directory:
            trusted.append((rank, directory))
        else:
            unresolved.append(rank)
    return trusted, unresolved


def decision_from_row(
    row: dict,
    picked_dirs: list[tuple[int, str]],
    reviewer_dir_keys: dict[str, str],
    unresolved_ranks: Iterable[int | None] = (),
) -> Decision:
    """The answer one current decision row gives (rule point 3).

    `picked_dirs` is the trusted `(rank, directory)` picks of a `matched_rank`
    row and `unresolved_ranks` the rest, both from `resolve_picks`. Both are
    ignored for `none_of_top_k`, whose key is the one typed.
    """
    decision = Decision(
        query_id=int(row["query_id"]),
        xml_id=str(row.get("xml_id") or ""),
        reviewer_key=reviewer_identity(row),
        reviewer_name=str(row.get("reviewer") or ""),
        decided_at=str(row.get("timestamp") or ""),
        row_id=int(row["id"]),
    )
    common = {
        "decided_at": decision.decided_at,
        "row_id": decision.row_id,
        "model_slug": str(row.get("model_slug") or ""),
        "variant": row.get("variant"),
    }
    if row.get("outcome") == "matched_rank":
        where = (common["model_slug"], common["variant"])
        for rank, directory in picked_dirs:
            key = _key_of_dir(directory, reviewer_dir_keys)
            if key:
                decision.claims.append(
                    KeyClaim(key=key, source=MODEL_SUGGESTION, rank=rank, **common)
                )
            else:
                # A reviewer directory without a key: a human has to say
                # what it is before anything is imported.
                decision.unresolved.append((*where, rank))
        for rank in unresolved_ranks:
            decision.unresolved.append((*where, rank))
    elif row.get("outcome") == "none_of_top_k":
        key = ccl_keys.normalize_ccl_key(row.get("ccl_key"))
        if key:
            decision.claims.append(KeyClaim(key=key, source=REVIEWER_TYPED, **common))
        else:
            decision.rejects_all = True
    return decision


def selected_ranks(row: dict) -> list[int]:
    """Every rank a `matched_rank` row selected, first choice first."""
    ranks = row.get("selected_ranks")
    if ranks:
        return [int(rank) for rank in ranks]
    rank = row.get("correct_rank")
    return [int(rank)] if rank else []


def build_rows(
    decisions: list[Decision],
    *,
    labelled_dir_names: Iterable[str],
    data_release: str,
    exported_at: str,
    status: str | None = None,
) -> list[dict]:
    """One resolved row per witness, sorted by `xml_id`.

    Every decision must carry its `xml_id`. A row whose witness cannot be named
    (no stored xml_id, and an id the served corpus does not know) is left out
    by the caller before this point: inventing one (the raw export writes
    `unknown-<id>`) would hand CCL's import a witness that does not exist.
    """
    labelled = list(labelled_dir_names)
    by_witness: dict[str, list[Decision]] = {}
    for decision in decisions:
        by_witness.setdefault(decision.xml_id, []).append(decision)

    rows: list[dict] = []
    for xml_id, group in by_witness.items():
        resolved = resolve_query(group, labelled)
        if status is not None and resolved["status"] != status:
            continue
        rows.append(
            {
                "xml_id": xml_id,
                **resolved,
                "export_schema_version": EXPORT_SCHEMA_VERSION,
                "exported_at": exported_at,
                "data_release": data_release,
            }
        )
    rows.sort(key=lambda row: row["xml_id"])
    return rows


def to_csv(rows: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in COLUMNS})
    return output.getvalue()


def resolve_query(
    decisions: list[Decision], labelled_dir_names: Iterable[str]
) -> dict:
    """The resolved columns for one query, from its counted decisions.

    Returns every column except `xml_id` and the three file-level metadata
    columns, which the caller adds.
    """
    labelled = list(labelled_dir_names)
    claims = [claim for decision in decisions for claim in decision.claims]

    # Distinct keys by match form, each with its display form.
    by_form: dict[str, list[KeyClaim]] = {}
    for claim in claims:
        form = ccl_keys.match_form(claim.key)
        if form:
            by_form.setdefault(form, []).append(claim)
    display = {form: _display_form(group, labelled) for form, group in by_form.items()}

    rejected = any(decision.rejects_all for decision in decisions)
    unresolved = sorted(
        {entry for decision in decisions for entry in decision.unresolved},
        key=lambda entry: (entry[0], entry[1] or "", entry[2] or 0),
    )
    if unresolved:
        status = "needs_check"
    elif len(by_form) >= 2 or (by_form and rejected):
        status = "conflict"
    elif by_form:
        status = "agreed"
    else:
        status = "no_key"

    key = next(iter(display.values())) if status == "agreed" else ""
    answers = sorted(display.values(), key=_sort_text)
    if rejected:
        answers.append(NONE_MARKER)
    if unresolved:
        answers.append(UNRESOLVED_MARKER)
    conflicting = (
        LIST_SEPARATOR.join(answers) if status in ("conflict", "needs_check") else ""
    )

    sources = {claim.source for group in by_form.values() for claim in group}
    if not sources:
        key_source = ""
    elif sources == {MODEL_SUGGESTION}:
        key_source = MODEL_SUGGESTION
    elif sources == {REVIEWER_TYPED}:
        key_source = REVIEWER_TYPED
    else:
        key_source = BOTH

    suggestions: set[tuple] = set()
    for form, group in by_form.items():
        for claim in group:
            if claim.source != MODEL_SUGGESTION:
                continue
            suggestions.add(
                (display[form], claim.model_slug or "", claim.variant or "", claim.rank)
            )
    with_key = status in ("conflict", "needs_check")
    suggested_by = LIST_SEPARATOR.join(
        [
            _suggestion_text(entry, with_key=with_key)
            for entry in sorted(suggestions, key=_suggestion_order)
        ]
        + [
            _suggestion_text((UNRESOLVED_MARKER, model, variant or "", rank), with_key=True)
            for model, variant, rank in unresolved
        ]
    )

    names: dict[tuple, Decision] = {}
    for decision in decisions:
        held = names.get(decision.reviewer_key)
        if held is None or (decision.decided_at, decision.row_id) > (
            held.decided_at,
            held.row_id,
        ):
            names[decision.reviewer_key] = decision
    reviewers = sorted(
        (decision.reviewer_name for decision in names.values()), key=_sort_text
    )

    times = sorted(decision.decided_at for decision in decisions)
    return {
        "ccl_key": key,
        "status": status,
        "conflicting_keys": conflicting,
        "key_source": key_source,
        "suggested_by": suggested_by,
        "key_shaped": (
            ("true" if ccl_keys.KEY_SHAPED.match(key) else "false") if key else ""
        ),
        "reviewer_count": len(names),
        "reviewers": LIST_SEPARATOR.join(reviewers),
        "first_decided_at": iso_utc(times[0]) if times else "",
        "last_decided_at": iso_utc(times[-1]) if times else "",
    }


def iso_utc(timestamp: str | None) -> str:
    """`2026-11-11 14:03:22` (SQLite `datetime('now')`, UTC) as ISO 8601 with Z."""
    if not timestamp:
        return ""
    text = str(timestamp).strip()
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(text[:19], pattern)
        except ValueError:
            continue
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    return text


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def xml_id_of(filename: str) -> str:
    """A query's CCL `xml:id`: its filename without `.txt`.

    `scripts/data/tei_to_canon.py` writes one file per transcription unit,
    named after the unit's `xml:id`. Only the `.txt` suffix is removed, never
    the last dotted part: `BAV1341.13v.1.txt` is `BAV1341.13v.1`.
    """
    return filename[:-4] if filename.endswith(".txt") else filename


def _key_of_dir(directory: str | None, reviewer_dir_keys: dict[str, str]) -> str:
    """The key a picked directory names, or "" when it names none.

    A labelled directory is named by its CCL key in most cases, and by an
    edition citation or similar in the rest (151 of 840, see
    `scripts/data/label_taxonomy.py`); either way its name is what the reviewer
    confirmed, so it is the key, and `key_shaped` says which kind it is. A
    reviewer directory names its `ccl_key`. One created before issue #196 under
    a siglum label has none, and its label is not a key, so it names nothing.
    """
    if not directory:
        return ""
    if is_reviewer_dir_id(directory):
        return ccl_keys.normalize_ccl_key(reviewer_dir_keys.get(directory, ""))
    return ccl_keys.normalize_ccl_key(directory)


def _display_form(group: list[KeyClaim], labelled: list[str]) -> str:
    labelled_name = ccl_keys.find_labelled_dir(group[0].key, labelled)
    if labelled_name is not None:
        return labelled_name
    ordered = sorted(group, key=lambda claim: (claim.decided_at, claim.row_id))
    forms = [ccl_keys.normalize_ccl_key(claim.key) for claim in ordered]
    shaped = [form for form in forms if ccl_keys.KEY_SHAPED.match(form)]
    return shaped[0] if shaped else forms[0]


def _suggestion_text(entry: tuple, *, with_key: bool) -> str:
    key, model, variant, rank = entry
    text = f"{model}:{variant}:{rank if rank is not None else ''}"
    return f"{key}={text}" if with_key else text


def _suggestion_order(entry: tuple) -> tuple:
    key, model, variant, rank = entry
    return (_sort_text(key), model, variant, rank if rank is not None else 0)


def _sort_text(value: str) -> tuple:
    return (value.casefold(), value)


def _order(row: dict) -> tuple:
    return (str(row.get("timestamp") or ""), int(row.get("id") or 0))
