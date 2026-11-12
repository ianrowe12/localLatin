"""Text a reviewer types that later lands in a CSV cell (issue #274).

The CCL key export (`services/ccl_key_export.py`) writes typed CCL keys,
reviewer-directory labels and display names verbatim, because CCL's import
reads the value itself and an escaped value would be a different key. So the
two things that make a verbatim cell dangerous are refused where the text comes
IN, never rewritten on the way out:

* A cell that starts with ``=``, ``+``, ``-`` or ``@`` is read as a formula by
  Excel, LibreOffice and Google Sheets (``=HYPERLINK(...)`` runs on open).
  Leading whitespace and control characters (tab, CR, LF, ...) are skipped
  before that first character is judged, because some spreadsheets skip them
  too and every field here is stored trimmed anyway, so ``"\\t=1+1"`` is refused
  exactly like ``"=1+1"``.
* ``|`` anywhere. The export joins lists with `` | `` (`LIST_SEPARATOR`). A
  value containing `` | `` splits in two, and a bare ``|`` is refused as well:
  a value that ends in `` |`` (or starts with ``| ``) is glued to its neighbour's
  separator, ``"Ann |" + " | " + "Bo"`` reads back as ``Ann`` and ``| Bo``. No
  labelled directory name contains ``|`` (asserted by
  `tests/test_input_safety.py` over `data/canon_labelled/`), so nothing
  legitimate is lost.

Stored values are never re-checked or rewritten: the feedback log is
append-only, and a name saved before this rule keeps serialising as it was.

The frontend mirrors this rule in `frontend/src/utils/inputSafety.ts`; both
sides are tested against the same cases in
`frontend/src/utils/inputSafetyCases.json`.
"""

from __future__ import annotations

import re

from pydantic_core import PydanticCustomError

#: Pydantic error type for a refused value. `web/app.py` turns a request
#: validation failure carrying it into ``{error: {code: "UNSAFE_TEXT",
#: message}}`` so the scholar sees the sentence, not a validation dump.
UNSAFE_TEXT_ERROR_TYPE = "unsafe_text"

#: First characters a spreadsheet reads as the start of a formula.
FORMULA_PREFIXES = ("=", "+", "-", "@")

#: The character the CCL export's ` | ` list separator is built on.
SEPARATOR_CHAR = "|"

# Whitespace (Python's Unicode \s), C0 and C1 control characters and the BOM,
# skipped before the first character is judged. The frontend's regex lists the
# same set so the two sides agree character for character.
_LEADING_FORMULA = re.compile(r"^[\s\x00-\x1f\x7f-\x9f﻿]*[=+\-@]")

#: Names for the message, by field.
CCL_KEY = "A CCL key"
DIRECTORY_NAME = "A directory name"
DISPLAY_NAME = "A display name"


def unsafe_text_message(subject: str) -> str:
    """The sentence a scholar is shown. Never includes the refused value."""
    return (
        f"{subject} cannot start with =, +, - or @ (a spreadsheet would run it "
        "as a formula) and cannot contain | (the export uses it to separate "
        "names)."
    )


def is_unsafe_text(value: str | None) -> bool:
    """Whether `value` would be refused. None and empty strings are allowed."""
    if not value:
        return False
    return SEPARATOR_CHAR in value or _LEADING_FORMULA.match(value) is not None


def check_safe_text(value: str | None, subject: str) -> str | None:
    """Pydantic validator body: return `value` unchanged or raise.

    Raises `PydanticCustomError` rather than `ValueError` so the error carries
    `UNSAFE_TEXT_ERROR_TYPE` and the message without pydantic's
    ``"Value error, "`` prefix.
    """
    if is_unsafe_text(value):
        raise PydanticCustomError(UNSAFE_TEXT_ERROR_TYPE, unsafe_text_message(subject))
    return value
