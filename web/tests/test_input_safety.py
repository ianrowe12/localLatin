"""Typed values that end up in a CSV cell are refused at input (issue #274).

Three things are pinned here:

1. The rule itself, over the case list the frontend test reads too, so the two
   sides cannot drift apart.
2. Every input path that stores such a value refuses it with a 422 carrying a
   sentence a scholar can read, and writes nothing: the typed CCL key, a
   reviewer-directory label, and a display name at signup and at admin account
   creation.
3. Every label the labelled corpus actually uses passes. If this ever fails, a
   real CCL key would be refused, which must be fixed in the rule, not here.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.services.ccl_key_export import LIST_SEPARATOR
from web.services.ccl_keys import normalize_ccl_key
from web.services.input_safety import (
    CCL_KEY,
    DIRECTORY_NAME,
    DISPLAY_NAME,
    SEPARATOR_CHAR,
    is_unsafe_text,
    unsafe_text_message,
)
from web.tests.test_reviewer_dirs import _signed_in

WEB_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = WEB_ROOT / "frontend" / "src" / "utils" / "inputSafetyCases.json"
# The research repo's corpus, one level above the web/ subtree. Absent in a
# standalone webapp checkout, where the real-label test skips.
LABELLED_ROOT = WEB_ROOT.parent / "data" / "canon_labelled"

_CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
REJECTED = [case["value"] for case in _CASES["rejected"]]
ACCEPTED = [case["value"] for case in _CASES["accepted"]]
REJECTED_IDS = [case["why"] for case in _CASES["rejected"]]
ACCEPTED_IDS = [case["why"] for case in _CASES["accepted"]]

#: One of each refused class, for the per-endpoint tests.
REJECTED_SAMPLE = ["=1+1", "+1", "-1", "@ann", "\t=1+1", "Ann | Bo", "Ann|Bo"]

PASSWORD = "correct horse battery staple"


def _db(tmp_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(
        tmp_path / "runs" / "active" / "resubmit" / "webapp" / "feedback.db"
    )


def _count(tmp_path: Path, table: str) -> int:
    with _db(tmp_path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _assert_refused(response, subject: str) -> None:
    assert response.status_code == 422, response.text
    body = response.json()
    assert body == {
        "error": {"code": "UNSAFE_TEXT", "message": unsafe_text_message(subject)}
    }


# --- the rule --------------------------------------------------------------


@pytest.mark.parametrize("value", REJECTED, ids=REJECTED_IDS)
def test_rejected_cases(value: str) -> None:
    assert is_unsafe_text(value)


@pytest.mark.parametrize("value", ACCEPTED, ids=ACCEPTED_IDS)
def test_accepted_cases(value: str) -> None:
    assert not is_unsafe_text(value)


def test_none_is_allowed() -> None:
    assert not is_unsafe_text(None)


def test_the_separator_rule_covers_the_export_separator() -> None:
    # If the export ever changes its separator, this rule has to follow it.
    assert SEPARATOR_CHAR in LIST_SEPARATOR


@pytest.mark.parametrize("subject", [CCL_KEY, DIRECTORY_NAME, DISPLAY_NAME])
def test_the_message_matches_the_frontends(subject: str) -> None:
    # The frontend shows the same sentence inline before submitting.
    assert unsafe_text_message(subject) == _CASES["messages"][subject]
    assert set(_CASES["messages"]) == {CCL_KEY, DIRECTORY_NAME, DISPLAY_NAME}


# --- real labels -----------------------------------------------------------


def _real_labels() -> list[str]:
    # Path.iterdir, not a shell pipeline: several names contain newlines.
    return sorted(p.name for p in LABELLED_ROOT.iterdir() if p.is_dir())


@pytest.mark.skipif(
    not LABELLED_ROOT.is_dir(),
    reason="data/canon_labelled is absent (standalone webapp checkout)",
)
def test_every_labelled_directory_name_passes() -> None:
    labels = _real_labels()
    # The frozen benchmark has 840 directories. A much smaller count means the
    # walk is not looking at the corpus, and the assertion below would be empty.
    assert len(labels) >= 800
    refused = [
        label
        for label in labels
        if is_unsafe_text(label) or is_unsafe_text(normalize_ccl_key(label))
    ]
    assert refused == []


# --- input paths -----------------------------------------------------------


@pytest.mark.parametrize("value", REJECTED_SAMPLE)
def test_a_typed_ccl_key_is_refused_and_nothing_is_written(
    tmp_path: Path, value: str
) -> None:
    client = _signed_in(tmp_path)
    response = client.post(
        "/api/feedback",
        json={
            "query_id": 0,
            "model_slug": "bowphs/LaTa",
            "outcome": "none_of_top_k",
            "correct_rank": 0,
            "notes": "",
            "ccl_key": value,
        },
    )
    _assert_refused(response, CCL_KEY)
    assert _count(tmp_path, "feedback") == 0
    assert _count(tmp_path, "reviewer_dirs") == 0


@pytest.mark.parametrize("key", ["CTOU.567.16", "Can.apost.49", "Capit.Martini.12"])
def test_a_legitimate_ccl_key_is_still_recorded(tmp_path: Path, key: str) -> None:
    client = _signed_in(tmp_path)
    response = client.post(
        "/api/feedback",
        json={
            "query_id": 0,
            "model_slug": "bowphs/LaTa",
            "outcome": "none_of_top_k",
            "correct_rank": 0,
            "notes": "",
            "ccl_key": key,
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["ccl_key"] == key


@pytest.mark.parametrize("value", REJECTED_SAMPLE)
def test_a_reviewer_directory_label_is_refused(tmp_path: Path, value: str) -> None:
    client = _signed_in(tmp_path)
    response = client.post(
        "/api/reviewer_dirs", json={"query_file_id": 0, "label": value}
    )
    _assert_refused(response, DIRECTORY_NAME)
    assert _count(tmp_path, "reviewer_dirs") == 0


def test_a_legitimate_reviewer_directory_label_is_accepted(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    response = client.post(
        "/api/reviewer_dirs", json={"query_file_id": 0, "label": "Capit.Martini.12"}
    )
    assert response.status_code == 201, response.text
    assert response.json()["label"] == "Capit.Martini.12"


@pytest.mark.parametrize("value", REJECTED_SAMPLE)
def test_a_display_name_is_refused_at_signup(tmp_path: Path, value: str) -> None:
    client = _signed_in(tmp_path)
    client.cookies.clear()
    before = _count(tmp_path, "accounts")
    response = client.post(
        "/api/auth/register",
        json={"username": "newcomer", "display_name": value, "password": PASSWORD},
    )
    _assert_refused(response, DISPLAY_NAME)
    assert _count(tmp_path, "accounts") == before


@pytest.mark.parametrize("value", REJECTED_SAMPLE)
def test_a_display_name_is_refused_at_admin_account_creation(
    tmp_path: Path, value: str
) -> None:
    client = _signed_in(tmp_path)
    before = _count(tmp_path, "accounts")
    response = client.post(
        "/api/auth/accounts",
        json={
            "username": "invitee",
            "display_name": value,
            "role": "reviewer",
            "password": PASSWORD,
        },
    )
    _assert_refused(response, DISPLAY_NAME)
    assert _count(tmp_path, "accounts") == before


def test_legitimate_display_names_are_accepted(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    created = client.post(
        "/api/auth/accounts",
        json={
            "username": "invitee",
            "display_name": "Ann-Marie Example",
            "role": "reviewer",
            "password": PASSWORD,
        },
    )
    assert created.status_code == 201, created.text
    client.cookies.clear()
    registered = client.post(
        "/api/auth/register",
        json={
            "username": "newcomer",
            "display_name": "A. N. Example",
            "password": PASSWORD,
        },
    )
    assert registered.status_code == 201, registered.text


def test_other_validation_failures_keep_fastapis_shape(tmp_path: Path) -> None:
    client = _signed_in(tmp_path)
    response = client.post(
        "/api/auth/accounts",
        json={"username": "x", "display_name": "Ann Example", "role": "reviewer"},
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_a_stored_unsafe_display_name_still_serialises(tmp_path: Path) -> None:
    # The rule applies to input only. A name stored before it (or written by a
    # maintenance script) is never rewritten and must still be readable.
    client = _signed_in(tmp_path)
    with _db(tmp_path) as connection:
        connection.execute("UPDATE accounts SET display_name = '=legacy' WHERE username = 'pi'")
    me = client.get("/api/auth/me")
    assert me.status_code == 200, me.text
    assert me.json()["display_name"] == "=legacy"
