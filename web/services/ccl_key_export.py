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
   That is the "own decision" half of `GET /api/feedback/latest`, which is also
   keyed by model and variant. A newer save on the same model and variant
   replaces an older one; a save under a different model does not, because the
   app itself still shows the reviewer both answers.
3. Each current decision names zero or more keys. A `matched_rank` row names
   the directory behind every selected rank (a labelled directory names itself;
   a reviewer directory names its `ccl_key`, and one without a key names
   nothing). A `none_of_top_k` row names its typed `ccl_key`, if any.
4. Keys are compared by `ccl_keys.match_form` (case-folded). The query's
   status is `no_key` when no counted decision names a key, `agreed` when they
   all name the same key, and `conflict` when two or more different keys are
   named, whether by different reviewers, by one reviewer under two models, or
   by one multi-rank selection.
5. A reviewer counts once (`reviewer_count`, `reviewers`) however many models
   they decided under. A reviewer whose decisions name no key still counts: a
   `none_of_top_k` without a key says the ranking missed, not that there is no
   key, so it never conflicts with a key somebody else named.

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

STATUSES = ("agreed", "conflict", "no_key")

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
    reviewer_key: tuple
    reviewer_name: str
    decided_at: str
    row_id: int
    claims: list[KeyClaim] = field(default_factory=list)


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
            int(row["query_id"]),
            reviewer_identity(row),
            row.get("model_slug"),
            row.get("variant"),
        )
        held = newest.get(slot)
        if held is None or _order(row) > _order(held):
            newest[slot] = row
    return list(newest.values())


def decision_from_row(
    row: dict,
    picked_dirs: list[tuple[int, str | None]],
    reviewer_dir_keys: dict[str, str],
) -> Decision:
    """The keys one current decision row names (rule point 3).

    `picked_dirs` is `(rank, directory)` for every rank a `matched_rank` row
    selected, in selection order, with None where no directory could be
    resolved. It is ignored for `none_of_top_k`, whose key is the one typed.
    """
    decision = Decision(
        query_id=int(row["query_id"]),
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
        for rank, directory in picked_dirs:
            key = _key_of_dir(directory, reviewer_dir_keys)
            if key:
                decision.claims.append(
                    KeyClaim(key=key, source=MODEL_SUGGESTION, rank=rank, **common)
                )
    elif row.get("outcome") == "none_of_top_k":
        key = ccl_keys.normalize_ccl_key(row.get("ccl_key"))
        if key:
            decision.claims.append(KeyClaim(key=key, source=REVIEWER_TYPED, **common))
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
    filename_by_query: dict[int, str],
    labelled_dir_names: Iterable[str],
    data_release: str,
    exported_at: str,
    status: str | None = None,
) -> tuple[list[dict], int]:
    """One resolved row per query, sorted by `xml_id`.

    Returns the rows and how many queries were left out because the served
    corpus has no filename for them. Such a query has no `xml_id`, and inventing
    one (the raw export writes `unknown-<id>`) would hand CCL's import a witness
    that does not exist.
    """
    labelled = list(labelled_dir_names)
    by_query: dict[int, list[Decision]] = {}
    for decision in decisions:
        by_query.setdefault(decision.query_id, []).append(decision)

    rows: list[dict] = []
    unknown = 0
    for query_id, group in by_query.items():
        filename = filename_by_query.get(query_id)
        if not filename:
            unknown += 1
            continue
        resolved = resolve_query(group, labelled)
        if status is not None and resolved["status"] != status:
            continue
        rows.append(
            {
                "xml_id": xml_id_of(filename),
                **resolved,
                "export_schema_version": EXPORT_SCHEMA_VERSION,
                "exported_at": exported_at,
                "data_release": data_release,
            }
        )
    rows.sort(key=lambda row: row["xml_id"])
    return rows, unknown


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

    if not by_form:
        status = "no_key"
    elif len(by_form) == 1:
        status = "agreed"
    else:
        status = "conflict"

    key = next(iter(display.values())) if status == "agreed" else ""
    conflicting = (
        LIST_SEPARATOR.join(sorted(display.values(), key=_sort_text))
        if status == "conflict"
        else ""
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
    suggested_by = LIST_SEPARATOR.join(
        _suggestion_text(entry, with_key=status == "conflict")
        for entry in sorted(suggestions, key=_suggestion_order)
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
