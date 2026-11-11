"""Guard rails for scripts/webapp/set_account_role.py (issue #265).

The script changes roles in the live reviewer database on the deploy host. The
temp databases here are built from the webapp's own DDL, read out of
web/services/feedback_db.py as source text (`_SCHEMA` and
`ACCOUNT_ROLE_CHANGES_DDL`) rather than imported: tests/ also runs on the
Python 3.10 CI leg, where the web package cannot import (enum.StrEnum).
"""

from __future__ import annotations

import ast
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "webapp" / "set_account_role.py"
FEEDBACK_DB_SRC = REPO_ROOT / "web" / "services" / "feedback_db.py"
FAKE_HASH = "pbkdf2_sha256$200000$" + "a" * 32 + "$" + "b" * 64


def _module_string(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError(f"{name} not found in {path}")


APP_SCHEMA = _module_string(FEEDBACK_DB_SRC, "_SCHEMA")
APP_AUDIT_DDL = _module_string(FEEDBACK_DB_SRC, "ACCOUNT_ROLE_CHANGES_DDL")

# The reviewer-directory tables are created in _migrate(), not _SCHEMA; a
# minimal shape is enough to count their rows.
DIR_TABLES = """
CREATE TABLE reviewer_dirs (id INTEGER PRIMARY KEY AUTOINCREMENT, dir_id TEXT NOT NULL UNIQUE);
CREATE TABLE reviewer_dir_members (id INTEGER PRIMARY KEY AUTOINCREMENT, dir_id TEXT, query_id INTEGER);
"""


def _make_db(path: Path, *, with_audit: bool, schema: str = APP_SCHEMA) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(schema)
    conn.executescript(DIR_TABLES)
    if with_audit:
        conn.executescript(APP_AUDIT_DDL)
    conn.execute(
        "INSERT INTO feedback (query_id, model_slug, reviewer, notes) "
        "VALUES (1, 'm', 'PI', 'a reviewer wrote this')"
    )
    conn.execute("INSERT INTO reviewer_dirs (dir_id) VALUES ('reviewer-dir-abc')")
    conn.commit()
    conn.close()
    return path


def _add_account(
    db: Path,
    username: str,
    display_name: str,
    role: str,
    approval_status: str = "approved",
    is_active: int = 1,
) -> int:
    conn = sqlite3.connect(db)
    cur = conn.execute(
        "INSERT INTO accounts (username, display_name, password_hash, role, "
        "approval_status, is_active) VALUES (?, ?, ?, ?, ?, ?)",
        (username, display_name, FAKE_HASH, role, approval_status, is_active),
    )
    conn.commit()
    conn.close()
    return int(cur.lastrowid)


@pytest.fixture(params=[True, False], ids=["audit-table-present", "audit-table-absent"])
def db(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    path = _make_db(tmp_path / "feedback.db", with_audit=request.param)
    _add_account(path, "pi", "PI Scholar", "pi_admin")
    _add_account(path, "noah.adler@uky.edu", "Noah Adler", "reviewer", "pending", 1)
    _add_account(path, "reviewer", "External Reviewer", "reviewer")
    return path


def run(db: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db), *args],
        capture_output=True,
        text=True,
    )


def account(db: Path, username: str) -> sqlite3.Row:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM accounts WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()


def audit_rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'account_role_changes'"
        ).fetchone()
        if not exists:
            return []
        return conn.execute("SELECT * FROM account_role_changes ORDER BY id").fetchall()
    finally:
        conn.close()


def backups(db: Path) -> list[Path]:
    return sorted(db.parent.glob(f"{db.name}.pre-role-change-*"))


def test_audit_ddl_is_byte_identical_to_the_apps() -> None:
    script_ddl = _module_string(SCRIPT, "ACCOUNT_ROLE_CHANGES_DDL")
    assert script_ddl == APP_AUDIT_DDL, (
        "scripts/webapp/set_account_role.py and web/services/feedback_db.py disagree "
        "on ACCOUNT_ROLE_CHANGES_DDL; keep them identical"
    )


def test_find_matches_username_or_display_name_without_secrets(db: Path) -> None:
    result = run(db, "--find", "NOAH")
    assert result.returncode == 0, result.stderr
    assert "noah.adler@uky.edu" in result.stdout
    assert "Noah Adler" in result.stdout
    assert "pending" in result.stdout
    assert "pi_admin" not in result.stdout  # the PI's row is not a match
    for header in ("id", "username", "display_name", "role", "approval_status", "created_at"):
        assert header in result.stdout
    assert "pbkdf2" not in result.stdout and "password" not in result.stdout

    by_display = run(db, "--find", "adler")
    assert "noah.adler@uky.edu" in by_display.stdout

    # LIKE wildcards are literal, not "match everything".
    wildcard = run(db, "--find", "%")
    assert wildcard.returncode == 0
    assert "No account" in wildcard.stdout


def test_dry_run_writes_nothing(db: Path) -> None:
    before = db.read_bytes()
    result = run(db, "--username", "noah.adler@uky.edu", "--role", "pi_admin", "--approve-if-pending")
    assert result.returncode == 0, result.stderr
    assert "reviewer -> pi_admin" in result.stdout
    assert "Would approve" in result.stdout
    assert "Dry run" in result.stdout
    assert db.read_bytes() == before
    assert backups(db) == []
    assert account(db, "noah.adler@uky.edu")["role"] == "reviewer"


def test_apply_backs_up_then_promotes_approves_and_audits(db: Path) -> None:
    result = run(
        db,
        "--username", "Noah.Adler@uky.edu",  # case-insensitive, as sign-in is
        "--role", "pi_admin",
        "--approve-if-pending",
        "--note", "issue #265",
        "--apply",
    )
    assert result.returncode == 0, result.stderr

    made = backups(db)
    assert len(made) == 1
    # It holds every password hash, so it is private to the service user.
    assert made[0].stat().st_mode & 0o777 == 0o600
    assert str(made[0]) in result.stdout
    assert f"({made[0].stat().st_size} bytes" in result.stdout
    # The backup is a consistent pre-change snapshot.
    snap = sqlite3.connect(made[0])
    try:
        assert snap.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert snap.execute(
            "SELECT role, approval_status FROM accounts WHERE username = 'noah.adler@uky.edu'"
        ).fetchone() == ("reviewer", "pending")
    finally:
        snap.close()

    row = account(db, "noah.adler@uky.edu")
    assert row["role"] == "pi_admin"
    assert row["approval_status"] == "approved"
    assert row["is_active"] == 1
    assert row["approved_at"] is not None
    assert row["approved_by_account_id"] is None
    assert row["approval_note"] == "cli: issue #265"

    rows = audit_rows(db)
    assert len(rows) == 1
    assert rows[0]["account_id"] == row["id"]
    assert (rows[0]["old_role"], rows[0]["new_role"]) == ("reviewer", "pi_admin")
    assert rows[0]["changed_by_account_id"] is None
    assert rows[0]["note"] == "cli: issue #265"
    assert rows[0]["changed_at"]


def test_the_created_audit_table_matches_the_apps(tmp_path: Path) -> None:
    """A DB without the table gains exactly the shape _migrate() would create."""
    cli_db = _make_db(tmp_path / "cli.db", with_audit=False)
    _add_account(cli_db, "pi", "PI", "pi_admin")
    _add_account(cli_db, "rev", "Rev", "reviewer")
    assert run(cli_db, "--username", "rev", "--role", "pi_admin", "--apply").returncode == 0

    app_db = _make_db(tmp_path / "app.db", with_audit=True)

    def shape(path: Path) -> list[tuple]:
        conn = sqlite3.connect(path)
        try:
            return [
                tuple(r) for r in conn.execute("PRAGMA table_info(account_role_changes)")
            ] + [
                tuple(r) for r in conn.execute("PRAGMA index_list(account_role_changes)")
            ]
        finally:
            conn.close()

    assert shape(cli_db) == shape(app_db)


def test_pending_account_stays_pending_without_the_flag(db: Path) -> None:
    result = run(db, "--username", "noah.adler@uky.edu", "--role", "pi_admin", "--apply")
    assert result.returncode == 0, result.stderr
    assert "stays pending" in result.stdout
    row = account(db, "noah.adler@uky.edu")
    assert row["role"] == "pi_admin"
    assert row["approval_status"] == "pending"


def test_unknown_username_exits_non_zero(db: Path) -> None:
    result = run(db, "--username", "nobody", "--role", "pi_admin", "--apply")
    assert result.returncode == 1
    assert "no account with username 'nobody'" in result.stderr
    assert backups(db) == []


def test_ambiguous_username_exits_non_zero(tmp_path: Path) -> None:
    # The real column is COLLATE NOCASE UNIQUE; a DB that lost the constraint
    # could hold two spellings. The script must refuse rather than pick one.
    loose = APP_SCHEMA.replace("username TEXT NOT NULL COLLATE NOCASE UNIQUE", "username TEXT NOT NULL")
    assert loose != APP_SCHEMA
    path = _make_db(tmp_path / "loose.db", with_audit=False, schema=loose)
    _add_account(path, "pi", "PI", "pi_admin")
    _add_account(path, "noah", "Noah A", "reviewer")
    _add_account(path, "NOAH", "Noah B", "reviewer")
    result = run(path, "--username", "noah", "--role", "pi_admin", "--apply")
    assert result.returncode == 1
    assert "matches 2 accounts" in result.stderr
    assert backups(path) == []


def test_refuses_to_demote_the_last_approved_pi_admin(db: Path) -> None:
    # A pending and a deactivated pi_admin do not count: neither can sign in.
    _add_account(db, "pending-pi", "Pending PI", "pi_admin", "pending")
    _add_account(db, "inactive-pi", "Inactive PI", "pi_admin", "approved", is_active=0)
    for apply in ([], ["--apply"]):
        result = run(db, "--username", "pi", "--role", "reviewer", *apply)
        assert result.returncode == 1
        assert "last approved PI/admin" in result.stderr
    assert account(db, "pi")["role"] == "pi_admin"
    assert backups(db) == []
    assert audit_rows(db) == []

    # With a second approved admin the demotion goes through.
    assert run(db, "--username", "reviewer", "--role", "pi_admin", "--apply").returncode == 0
    assert run(db, "--username", "pi", "--role", "reviewer", "--apply").returncode == 0
    assert account(db, "pi")["role"] == "reviewer"


def test_unchanged_role_is_a_no_op(db: Path) -> None:
    before = db.read_bytes()
    result = run(db, "--username", "pi", "--role", "pi_admin", "--apply")
    assert result.returncode == 0, result.stderr
    assert "Nothing to do" in result.stdout
    assert db.read_bytes() == before
    assert backups(db) == []


def test_append_only_tables_are_untouched(db: Path) -> None:
    assert run(
        db, "--username", "noah.adler@uky.edu", "--role", "pi_admin",
        "--approve-if-pending", "--apply",
    ).returncode == 0
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*), MIN(notes) FROM feedback").fetchone() == (
            1,
            "a reviewer wrote this",
        )
        assert conn.execute("SELECT COUNT(*) FROM reviewer_dirs").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize(
    "args",
    [
        ["--find", "noah", "--apply"],
        ["--find", "noah", "--username", "pi"],
        ["--username", "pi"],
        ["--role", "pi_admin"],
        ["--username", "pi", "--role", "owner"],
    ],
)
def test_inconsistent_arguments_are_refused(db: Path, args: list[str]) -> None:
    result = run(db, *args)
    assert result.returncode == 2
    assert backups(db) == []


def test_missing_database_is_a_readable_error(tmp_path: Path) -> None:
    result = run(tmp_path / "absent.db", "--find", "x")
    assert result.returncode == 1
    assert "no database at" in result.stderr
    assert not (tmp_path / "absent.db").exists()
