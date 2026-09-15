"""CCL source keys typed by a reviewer (issue #196).

Abigail's finding is the whole reason this module exists: what she goes and
checks in the CCL when the ten ranked candidates are wrong is usually **an
unlabelled witness of a source key that is already known** -- sometimes a key
the labelled corpus holds and the model simply did not rank, sometimes a key
that is not in the labelled corpus at all. Neither case is "a new directory
named after a siglum", which is what the retired red button produced.

So one optional field carries the key, and the server decides what it means.
Three outcomes, in this order:

1. the key names a LABELLED directory -> the assessment records
   "matches <key> (not in shortlist)" and nothing is created. A labelled
   directory is corpus data; a reviewer directory bearing the same name would
   be a duplicate of it that nothing can later merge or delete.
2. the key names an existing REVIEWER directory -> the query joins it.
3. otherwise -> a reviewer directory named by the key is created, and the query
   seeds it.

NORMALISATION is deliberately split in two. `normalize_ccl_key` is what gets
STORED: the reviewer's own capitalisation, with surrounding and repeated
whitespace removed, because a key is a citation and a stored citation should
read the way the person who typed it wrote it. `match_form` is what gets
COMPARED: the same string case-folded, so `ctou.567.16` and `CTOU.567.16` reach
the same directory. Storing the folded form instead would be lossy, and
comparing the stored form instead would scatter one key across several
directories -- which is exactly the accumulation of undeletable near-duplicates
this feature has to avoid.

`str.casefold()` rather than `str.lower()`: the corpus is Latin but the labels
carry editorial apparatus, and casefold is the Unicode-correct answer for the
comparison this makes.
"""

from __future__ import annotations

import re

#: A label that is a CCL source key, by the rule stated in
#: `scripts/data/label_taxonomy.py` and in the paper's appendix: a four-letter
#: upper-case source code, a period, a year (optionally `?` for an uncertain
#: date) and a period, whatever follows.
#:
#: Used for ONE thing only: deciding, during the migration, whether a reviewer
#: directory that predates this feature was already named by a key (issue #196
#: point 5). Live matching never consults it -- a reviewer may legitimately type
#: `Can.apost.49`, which is a real CCL key and not key-*shaped* by this rule --
#: so nothing a reviewer types is ever refused for failing it.
KEY_SHAPED = re.compile(r"^[A-Z]{4}\.\d+\??\.")

#: Longest key this accepts. Keys carry annotations (`DSIR.384.255 cap. 11`),
#: so this is a guard against a pasted paragraph, not a scholarly limit. Matches
#: the `label` bound on ReviewerDirCreate.
MAX_KEY_LENGTH = 200

_WHITESPACE = re.compile(r"\s+")


def normalize_ccl_key(raw: str | None) -> str:
    """The form that is STORED: trimmed, inner whitespace collapsed, case kept.

    An empty or whitespace-only field is "no key given", which is a valid
    submission (record the non-match and nothing else), so it normalises to the
    empty string rather than raising.
    """
    if raw is None:
        return ""
    return _WHITESPACE.sub(" ", raw).strip()


def match_form(key: str | None) -> str:
    """The form that is COMPARED. Never stored, never shown to a reviewer."""
    return normalize_ccl_key(key).casefold()


def is_key_shaped(label: str) -> bool:
    """Whether a pre-#196 directory label is already a CCL source key.

    Normalised first, so a label saved with stray leading whitespace is judged
    by its content rather than by how it was typed -- the pattern is anchored,
    and `"  CARL.501?.18"` is as much a key as `"CARL.501?.18"`.
    """
    return KEY_SHAPED.match(normalize_ccl_key(label)) is not None


def find_labelled_dir(key: str, labelled_dir_names: object) -> str | None:
    """The labelled directory this key names, or None.

    `labelled_dir_names` is any iterable of directory names -- in practice
    `DataStore.labelled_dir_files`, whose keys are the 840 benchmark directory
    names. Compared on `match_form`, so case and stray whitespace do not decide
    whether a reviewer's key finds the corpus directory it names.

    Ties (two directories differing only in case) resolve to the first in sorted
    order. That is arbitrary but deterministic, which is the property that
    matters: the same key must reach the same directory on every submission.
    """
    target = match_form(key)
    if not target:
        return None
    matches = [name for name in labelled_dir_names if match_form(name) == target]
    if not matches:
        return None
    return sorted(matches)[0]
