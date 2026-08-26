"""Guard rails for scripts/webapp/provision_smoke_account.py (issue #98).

The script writes to the live reviewer database on the deploy host, which holds
human feedback that cannot be regenerated. These tests pin the three properties
that make that safe: it matches the real schema, it refuses ambiguous input, and
it leaves the append-only tables alone.

The schema here is copied from web/services/feedback_db.py rather than imported,
because tests/ runs on Python 3.10 in CI where the web package cannot import
(web/models.py uses enum.StrEnum). A drift test below keeps the copy honest.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "webapp" / "provision_smoke_account.py"
FEEDBACK_DB_SRC = REPO_ROOT / "web" / "services" / "feedback_db.py"

ACCOUNTS_SCHEMA = """
CREATE TABLE accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'reviewer' CHECK (role IN ('reviewer', 'pi_admin')),
    is_active INTEGER NOT NULL DEFAULT 1,
    approval_status TEXT NOT NULL DEFAULT 'approved'
        CHECK (approval_status IN ('pending', 'approved', 'rejected')),
    approved_at TEXT,
    approved_by_account_id INTEGER,
    rejected_at TEXT,
    approval_note TEXT NOT NULL DEFAULT '',
    must_change_password INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_login_at TEXT
);
CREATE TABLE account_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    last_seen_at TEXT
);
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE reviewer_dirs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dir_id TEXT NOT NULL UNIQUE
);
CREATE TABLE reviewer_dir_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dir_id TEXT NOT NULL,
    query_id INTEGER NOT NULL
);
"""


def make_hash(password: str) -> str:
    """Mirrors web/services/feedback_db.py::_hash_password."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return f"pbkdf2_sha256$200000${salt}${digest}"


def verify(password: str, stored: str) -> bool:
    """Mirrors web/services/feedback_db.py::_verify_password."""
    algorithm, rounds, salt, expected = stored.split("$", 3)
    assert algorithm == "pbkdf2_sha256"
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(rounds)).hex()
    return secrets.compare_digest(digest, expected)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "feedback.db"
    conn = sqlite3.connect(path)
    conn.executescript(ACCOUNTS_SCHEMA)
    conn.execute("INSERT INTO feedback (query_id, notes) VALUES (1, 'a reviewer wrote this')")
    conn.execute("INSERT INTO reviewer_dirs (dir_id) VALUES ('reviewer-dir-abc123')")
    conn.commit()
    conn.close()
    return path


def run(db: Path, **kwargs) -> subprocess.CompletedProcess:
    args = [
        sys.executable,
        str(SCRIPT),
        "--db",
        str(db),
        "--username",
        kwargs.pop("username", "locallatin-smoke"),
        "--display-name",
        kwargs.pop("display_name", "Deploy Smoke Check"),
        "--password-hash",
        kwargs.pop("password_hash", make_hash("hunter2-hunter2-hunter2")),
        "--role",
        kwargs.pop("role", "pi_admin"),
        "--mode",
        kwargs.pop("mode", "create"),
    ]
    assert not kwargs, f"unexpected kwargs {kwargs}"
    return subprocess.run(args, capture_output=True, text=True)


def account_row(db: Path, username: str = "locallatin-smoke") -> sqlite3.Row:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM accounts WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()


def test_creates_an_approved_account_that_can_sign_in_immediately(db: Path) -> None:
    """The whole point: the account must be usable by deploy.sh with no further steps."""
    password = "a-strong-generated-password"
    result = run(db, password_hash=make_hash(password))
    assert result.returncode == 0, result.stderr

    row = account_row(db)
    assert row is not None
    assert row["role"] == "pi_admin"
    assert row["approval_status"] == "approved"
    assert row["is_active"] == 1
    # The forced-change flag is what would otherwise 403 every smoke request
    # outside the five exempt auth paths.
    assert row["must_change_password"] == 0
    assert row["approved_at"] is not None
    assert verify(password, row["password_hash"])


def test_username_is_lowercased_to_match_the_apps_normalisation(db: Path) -> None:
    run(db, username="LocalLatin-Smoke")
    assert account_row(db, "locallatin-smoke") is not None


def test_create_refuses_to_clobber_an_existing_account(db: Path) -> None:
    first = "first-password-first-password"
    run(db, password_hash=make_hash(first))
    result = run(db, password_hash=make_hash("second-password-second"))
    assert result.returncode != 0
    assert "already exists" in result.stdout + result.stderr
    # The original password still works: nothing was overwritten.
    assert verify(first, account_row(db)["password_hash"])


def test_rotate_replaces_the_password_and_revokes_open_sessions(db: Path) -> None:
    run(db, password_hash=make_hash("old-password-old-password"))
    account_id = account_row(db)["id"]
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO account_sessions (account_id, token_hash, expires_at) VALUES (?, ?, ?)",
        (account_id, "deadbeef", "2099-01-01"),
    )
    conn.commit()
    conn.close()

    new = "new-password-new-password"
    result = run(db, password_hash=make_hash(new), mode="rotate")
    assert result.returncode == 0, result.stderr
    assert verify(new, account_row(db)["password_hash"])

    conn = sqlite3.connect(db)
    remaining = conn.execute("SELECT COUNT(*) FROM account_sessions").fetchone()[0]
    conn.close()
    assert remaining == 0, "a rotated password must not leave old sessions usable"


def test_rotate_clears_a_forced_password_change_and_reapproves(db: Path) -> None:
    run(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE accounts SET must_change_password = 1, approval_status = 'rejected',"
        " is_active = 0, rejected_at = datetime('now')"
    )
    conn.commit()
    conn.close()

    result = run(db, mode="rotate")
    assert result.returncode == 0, result.stderr
    row = account_row(db)
    assert row["must_change_password"] == 0
    assert row["approval_status"] == "approved"
    assert row["is_active"] == 1
    assert row["rejected_at"] is None


def test_rotate_refuses_when_the_account_does_not_exist(db: Path) -> None:
    result = run(db, mode="rotate")
    assert result.returncode != 0
    assert "does not exist" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-hash",
        "pbkdf2_sha256$200000$tooshort$" + "a" * 64,
        "bcrypt$200000$" + "a" * 32 + "$" + "b" * 64,
        "pbkdf2_sha256$200000$" + "a" * 32 + "$" + "b" * 63,
        "",
    ],
)
def test_a_malformed_password_hash_is_refused_before_any_write(db: Path, bad: str) -> None:
    """A bad hash must fail loudly, not create an account nobody can sign in to."""
    result = run(db, password_hash=bad)
    assert result.returncode != 0
    assert account_row(db) is None


def test_the_append_only_tables_are_left_untouched(db: Path) -> None:
    """Reviewer feedback and reviewer directories are irreplaceable human work."""
    run(db)
    run(db, mode="rotate")
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM reviewer_dirs").fetchone()[0] == 1
        assert (
            conn.execute("SELECT notes FROM feedback").fetchone()[0]
            == "a reviewer wrote this"
        )
    finally:
        conn.close()


def test_missing_accounts_table_is_a_readable_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    result = run(empty)
    assert result.returncode != 0
    assert "accounts table does not exist" in result.stdout + result.stderr


def test_a_schema_missing_must_change_password_is_refused(tmp_path: Path) -> None:
    """A pre-#97 database would yield an account the live app fails closed on."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'reviewer',
            is_active INTEGER NOT NULL DEFAULT 1,
            approval_status TEXT NOT NULL DEFAULT 'approved'
        );
        """
    )
    conn.commit()
    conn.close()
    result = run(path)
    assert result.returncode != 0
    assert "must_change_password" in result.stdout + result.stderr


def test_hash_format_still_matches_the_apps_hasher() -> None:
    """If feedback_db.py changes its hashing, this script's validator must follow.

    Read as source text rather than imported: web/ needs Python 3.12, and this
    file also runs under the 3.10 leg of the CI matrix.
    """
    source = FEEDBACK_DB_SRC.read_text(encoding="utf-8")
    assert 'f"pbkdf2_sha256$200000${salt}${digest}"' in source, (
        "web/services/feedback_db.py no longer produces pbkdf2_sha256$200000$... — "
        "update HASH_PATTERN in scripts/webapp/provision_smoke_account.py and this test"
    )
    script = SCRIPT.read_text(encoding="utf-8")
    pattern = re.search(r"HASH_PATTERN = re\.compile\((.+)\)", script)
    assert pattern is not None, "HASH_PATTERN disappeared from the provisioning script"


def test_required_columns_match_the_apps_fail_closed_set() -> None:
    """The script must demand at least what web/services/feedback_db.py demands."""
    source = FEEDBACK_DB_SRC.read_text(encoding="utf-8")
    block = re.search(r"_REQUIRED_ACCOUNT_COLUMNS\s*=\s*\{(.*?)\}", source, re.S)
    assert block is not None, "_REQUIRED_ACCOUNT_COLUMNS not found in feedback_db.py"
    app_columns = set(re.findall(r'"([a-z_]+)"', block.group(1)))

    script = SCRIPT.read_text(encoding="utf-8")
    script_block = re.search(r"REQUIRED_COLUMNS\s*=\s*\{(.*?)\}", script, re.S)
    assert script_block is not None
    script_columns = set(re.findall(r'"([a-z_]+)"', script_block.group(1)))

    missing = app_columns - script_columns
    assert not missing, f"provisioning script does not check column(s) {sorted(missing)}"
