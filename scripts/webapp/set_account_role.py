#!/usr/bin/env python3
"""Find an account in the reviewer webapp DB, or change its role (issue #265).

WHY THIS EXISTS
---------------
The review site's admin panel can promote and demote accounts
(`POST /api/auth/accounts/{id}/role`), but only for a PI/admin who is signed
in. The production database lives only on the deploy host, which is reachable
only through the self-hosted Actions runner, so an operator who is not a
PI/admin there (or who needs to promote the first co-PI) runs this script
through `.github/workflows/admin-set-account-role.yml`.

It is stdlib-only and runs on the host's system Python 3.12, without the
webapp's venv and without importing the `web` package.

WHAT IT DOES
------------
    --find TEXT
        Lists accounts whose username or display name contains TEXT,
        case-insensitively: id, username, display_name, role, approval_status,
        created_at. Never password hashes, never session data. Read-only.

    --username NAME --role reviewer|pi_admin [--approve-if-pending] [--note T]
        Dry run: prints what would change and writes nothing.

    ... --apply
        1. Writes a consistent backup with sqlite3's online backup API to
           `<db>.pre-role-change-<UTC timestamp>` next to the DB, checks it
           with `PRAGMA integrity_check`, and prints its path and size.
        2. In ONE `BEGIN IMMEDIATE` transaction: sets the role, approves the
           account if `--approve-if-pending` was given and it is pending, and
           writes the same `account_role_changes` audit row the endpoint
           writes (changed_by_account_id NULL, note "cli: <note>").

WHAT IT REFUSES
---------------
- an unknown username, or a lookup that matches more than one row;
- a change that would leave no approved, active PI/admin (the same guard as
  the endpoint: there is no email recovery);
- touching the append-only tables (`feedback`, `reviewer_dirs`,
  `reviewer_dir_members`): their row counts are compared before commit and the
  transaction rolls back if they moved.

Sessions are not revoked: the app re-reads the role from `accounts` on every
request, so the new role applies to the account's next request.

USAGE
-----
    python3 scripts/webapp/set_account_role.py --db data/feedback.db --find noah
    python3 scripts/webapp/set_account_role.py --db data/feedback.db \\
        --username noah --role pi_admin --approve-if-pending --note "issue #265"
    # ... then the same with --apply
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Byte-identical copy of web/services/feedback_db.py::ACCOUNT_ROLE_CHANGES_DDL,
# so a DB the app has not migrated since #265 gains the same table. Copied, not
# imported: this runs without the web package. tests/test_set_account_role.py
# fails if the two drift.
ACCOUNT_ROLE_CHANGES_DDL = """
CREATE TABLE IF NOT EXISTS account_role_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    old_role TEXT NOT NULL,
    new_role TEXT NOT NULL,
    changed_by_account_id INTEGER,
    note TEXT NOT NULL DEFAULT '',
    changed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_account_role_changes_account
    ON account_role_changes(account_id);
"""

VALID_ROLES = ("reviewer", "pi_admin")

# The app's registration accepts any 2-64 character string, so this only
# refuses what no login name can be (empty, too long, control characters). The
# lookup is a bound parameter either way. The workflow applies a stricter
# charset before it ever calls this script.
USERNAME_PATTERN = re.compile(r"[^\x00-\x1f\x7f]{1,128}")

# Columns printed by --find and the dry run. Deliberately excludes
# password_hash and anything from account_sessions.
LISTED_COLUMNS = ("id", "username", "display_name", "role", "approval_status", "created_at")

REQUIRED_COLUMNS = {
    "id",
    "username",
    "display_name",
    "role",
    "approval_status",
    "is_active",
    "created_at",
    "updated_at",
    "approved_at",
    "approved_by_account_id",
    "approval_note",
}

# Human work product: must be byte-for-byte untouched by this script.
PROTECTED_TABLES = ("feedback", "reviewer_dirs", "reviewer_dir_members")


class CliError(Exception):
    """A refusal with a message for the operator; exits non-zero."""


def connect(db: Path) -> sqlite3.Connection:
    if not db.is_file():
        raise CliError(f"no database at {db}")
    # isolation_level=None: transactions are explicit (BEGIN IMMEDIATE below).
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(accounts)")}
    if not columns:
        conn.close()
        raise CliError(f"{db}: accounts table does not exist")
    missing = REQUIRED_COLUMNS - columns
    if missing:
        conn.close()
        raise CliError(
            f"{db}: accounts table is missing column(s) {', '.join(sorted(missing))}; "
            "start the current webapp once so its migrations run"
        )
    return conn


def format_rows(rows: list[sqlite3.Row]) -> str:
    table = [LISTED_COLUMNS] + [
        tuple("" if row[c] is None else str(row[c]) for c in LISTED_COLUMNS) for row in rows
    ]
    widths = [max(len(line[i]) for line in table) for i in range(len(LISTED_COLUMNS))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip()
        for line in table
    )


def find_accounts(conn: sqlite3.Connection, text: str) -> list[sqlite3.Row]:
    # instr over lower() rather than LIKE, so % and _ in TEXT are literal.
    needle = text.strip().lower()
    return conn.execute(
        f"""
        SELECT {", ".join(LISTED_COLUMNS)} FROM accounts
         WHERE instr(lower(username), ?) > 0 OR instr(lower(display_name), ?) > 0
         ORDER BY id
        """,
        (needle, needle),
    ).fetchall()


def lookup(conn: sqlite3.Connection, username: str) -> sqlite3.Row:
    # The column is COLLATE NOCASE UNIQUE and the app stores it lowercased, so
    # more than one row is impossible on a healthy DB; refuse rather than pick.
    rows = conn.execute(
        "SELECT * FROM accounts WHERE lower(username) = ?", (username.strip().lower(),)
    ).fetchall()
    if not rows:
        raise CliError(
            f"no account with username {username!r}; use --find to search "
            "by part of the username or display name"
        )
    if len(rows) > 1:
        raise CliError(
            f"username {username!r} matches {len(rows)} accounts; refusing to guess\n"
            + format_rows(rows)
        )
    return rows[0]


def other_active_admins(conn: sqlite3.Connection, account_id: int) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) FROM accounts
         WHERE role = 'pi_admin' AND approval_status = 'approved' AND is_active = 1
           AND id != ?
        """,
        (account_id,),
    ).fetchone()
    return int(row[0])


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in PROTECTED_TABLES:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if exists:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return counts


def plan(row: sqlite3.Row, role: str, approve_if_pending: bool) -> tuple[bool, bool]:
    """(role changes, approval changes) for this account."""
    return row["role"] != role, approve_if_pending and row["approval_status"] == "pending"


def check_last_admin(conn: sqlite3.Connection, row: sqlite3.Row, role: str) -> None:
    if (
        row["role"] == "pi_admin"
        and role != "pi_admin"
        and row["approval_status"] == "approved"
        and int(row["is_active"]) == 1
        and other_active_admins(conn, row["id"]) == 0
    ):
        raise CliError(
            f"{row['username']!r} is the last approved PI/admin; make another "
            "account a PI/admin first"
        )


def backup(db: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = db.with_name(f"{db.name}.pre-role-change-{stamp}")
    suffix = 1
    while True:
        # O_EXCL: never overwrite an earlier backup (two runs in one second).
        # 0600: the copy holds every password hash in the database.
        try:
            os.close(os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            break
        except FileExistsError:
            suffix += 1
            dest = db.with_name(f"{db.name}.pre-role-change-{stamp}-{suffix}")
    # Online backup API, not a file copy: the service is running, and a copy can
    # miss committed state that still lives in the journal.
    source = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    snapshot = sqlite3.connect(str(dest))
    try:
        source.backup(snapshot)
        status = snapshot.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        source.close()
        snapshot.close()
    if status != "ok":
        raise CliError(f"backup {dest} failed integrity_check: {status}")
    return dest


def apply_change(
    conn: sqlite3.Connection, username: str, role: str, approve_if_pending: bool, note: str
) -> sqlite3.Row:
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Re-read under the write lock: the dry-run view may be stale.
        row = lookup(conn, username)
        role_changes, approval_changes = plan(row, role, approve_if_pending)
        check_last_admin(conn, row, role)
        before = table_counts(conn)
        for statement in ACCOUNT_ROLE_CHANGES_DDL.split(";"):
            if statement.strip():
                conn.execute(statement)
        if role_changes:
            conn.execute(
                "UPDATE accounts SET role = ?, updated_at = datetime('now') WHERE id = ?",
                (role, row["id"]),
            )
            conn.execute(
                """
                INSERT INTO account_role_changes
                    (account_id, old_role, new_role, changed_by_account_id, note)
                VALUES (?, ?, ?, NULL, ?)
                """,
                (row["id"], row["role"], role, f"cli: {note}".strip()),
            )
        if approval_changes:
            # Mirrors FeedbackDB.set_account_approval('approved'), with no
            # approver account because none exists outside the app.
            conn.execute(
                """
                UPDATE accounts
                   SET approval_status = 'approved',
                       is_active = 1,
                       approved_at = datetime('now'),
                       approved_by_account_id = NULL,
                       rejected_at = NULL,
                       approval_note = ?,
                       updated_at = datetime('now')
                 WHERE id = ?
                """,
                (f"cli: {note}".strip(), row["id"]),
            )
        if table_counts(conn) != before:
            raise CliError("append-only table row counts changed; rolled back")
        updated = conn.execute("SELECT * FROM accounts WHERE id = ?", (row["id"],)).fetchone()
        conn.execute("COMMIT")
        return updated
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def run(args: argparse.Namespace) -> int:
    db = Path(args.db)
    if args.find is not None:
        if not args.find.strip():
            raise CliError("--find needs some text")
        conn = connect(db)
        try:
            rows = find_accounts(conn, args.find)
        finally:
            conn.close()
        if not rows:
            print(f"No account's username or display name contains {args.find!r}.")
            return 0
        print(format_rows(rows))
        return 0

    if not USERNAME_PATTERN.fullmatch(args.username) or not args.username.strip():
        raise CliError(f"--username {args.username!r} is not a valid login name")
    conn = connect(db)
    try:
        row = lookup(conn, args.username)
        role_changes, approval_changes = plan(row, args.role, args.approve_if_pending)
        print("Current:")
        print(format_rows([row]))
        if not role_changes and not approval_changes:
            print(f"Nothing to do: {row['username']} is already {args.role}"
                  + ("" if row["approval_status"] != "pending" else " (still pending)")
                  + ".")
            return 0
        check_last_admin(conn, row, args.role)
        if role_changes:
            print(f"Would change role: {row['role']} -> {args.role}")
        else:
            print(f"Role already {args.role}; unchanged.")
        if approval_changes:
            print("Would approve the pending account.")
        elif row["approval_status"] == "pending":
            print("Account is pending and stays pending (pass --approve-if-pending to approve).")
        elif row["approval_status"] == "rejected":
            print("Note: account is rejected and cannot sign in; its role change will not "
                  "matter until a PI/admin approves it.")
        if not args.apply:
            print("Dry run: nothing written. Re-run with --apply to make the change.")
            return 0
    finally:
        conn.close()

    dest = backup(db)
    print(f"Backup: {dest} ({dest.stat().st_size} bytes, integrity_check ok)")
    conn = connect(db)
    try:
        updated = apply_change(
            conn, args.username, args.role, args.approve_if_pending, args.note
        )
    finally:
        conn.close()
    print("Applied:")
    print(format_rows([updated]))
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True, help="path to the webapp feedback.db")
    parser.add_argument("--find", help="list accounts whose username/display name contains TEXT")
    parser.add_argument("--username", help="exact login name of the account to change")
    parser.add_argument("--role", choices=VALID_ROLES)
    parser.add_argument(
        "--approve-if-pending",
        action="store_true",
        help="also approve the account when it is still pending",
    )
    parser.add_argument("--apply", action="store_true", help="write; default is a dry run")
    parser.add_argument("--note", default="", help="recorded in the audit row as 'cli: NOTE'")
    args = parser.parse_args(argv)
    if args.find is not None:
        if args.username or args.role or args.apply or args.approve_if_pending:
            parser.error("--find cannot be combined with --username/--role/--apply")
    elif not (args.username and args.role):
        parser.error("give --find TEXT, or --username NAME and --role ROLE")
    if len(args.note) > 500:
        parser.error("--note is limited to 500 characters")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
