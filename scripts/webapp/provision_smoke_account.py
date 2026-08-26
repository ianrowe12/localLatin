#!/usr/bin/env python3
"""Create (or rotate) the deploy smoke-check account directly in the webapp DB.

WHY THIS EXISTS
---------------
`deploy/deploy.sh` runs `smoke_reviewer_pilot.py` against the freshly restarted
service, but only when `LOCALLATIN_SMOKE_USERNAME` / `LOCALLATIN_SMOKE_PASSWORD`
are set. Those checks are the only thing standing between a green deploy and a
service that answers 200 on `/health` while serving no data. To set the secrets
we first need an account that exists, is approved, and is not stuck behind a
forced password change.

Self-service registration cannot produce one: `POST /api/auth/register` creates
a *pending* reviewer, and approving it needs a PI/admin who is already signed
in. The deploy host is reachable only through the self-hosted Actions runner, so
this script runs there, against the live SQLite file.

WHY A PASSWORD HASH, NOT A PASSWORD
-----------------------------------
The caller passes `--password-hash`, never the plaintext. `workflow_dispatch`
inputs are recorded in the workflow run's metadata and are visible to anyone who
can read the repository, so a plaintext password given that way would be
permanently exposed. A PBKDF2-SHA256 hash at 200,000 rounds over a 128-bit
random salt is not password-equivalent: the operator generates the password
locally, derives the hash locally, passes only the hash here, and sets the
plaintext straight into the repo secret with `gh secret set`. The plaintext
never touches a log, a run record, or this host.

ROLE
----
Defaults to `pi_admin` because `smoke_reviewer_pilot.py` requires it: the smoke
run exercises `/api/stats`, `/api/feedback/export`, the PDF packets and the
pending-account approve/reject loop, all of which are PI/admin gated. A plain
reviewer account would silently reduce the smoke run to its unauthenticated
subset, which is the opposite of why the account exists.

WHAT IT REFUSES TO DO
---------------------
The reviewer feedback log and the reviewer-directory tables are append-only and
carry irreplaceable human work. This script asserts their row counts are
unchanged before it commits, and rolls back if they are not. It never issues a
DELETE or UPDATE against them.

USAGE
-----
    python3 scripts/webapp/provision_smoke_account.py \
        --db /homes/ipro222/localLatin/data/feedback.db \
        --username locallatin-smoke \
        --display-name "Deploy Smoke Check" \
        --password-hash 'pbkdf2_sha256$200000$<salt>$<digest>' \
        --mode create

`--mode rotate` re-points an existing account at a new hash, re-approves it,
clears any forced-password-change flag, and revokes its open sessions.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys

# Mirrors web/services/feedback_db.py::_hash_password. Kept as a literal pattern
# rather than an import because this script runs on the deploy host with the
# system python and no dependency on the web package being importable.
HASH_PATTERN = re.compile(r"^pbkdf2_sha256\$\d+\$[0-9a-f]{32}\$[0-9a-f]{64}$")

# web/services/feedback_db.py::_REQUIRED_ACCOUNT_COLUMNS. The app fails closed at
# startup when one is missing; so does this script, because an INSERT that omits
# a column the live schema requires would create an account the app refuses to
# serve.
REQUIRED_COLUMNS = {
    "approval_status",
    "is_active",
    "must_change_password",
    "password_hash",
    "role",
    "username",
    "display_name",
}

# Tables whose contents are human work product and must survive this script
# untouched. Counted before and after; a change rolls the transaction back.
PROTECTED_TABLES = ("feedback", "reviewer_dirs", "reviewer_dir_members")

VALID_ROLES = ("reviewer", "pi_admin")


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts for the append-only tables, skipping any that do not exist yet."""
    counts: dict[str, int] = {}
    for table in PROTECTED_TABLES:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if row is None:
            continue
        counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return counts


def assert_schema(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='accounts'"
    ).fetchone()
    if row is None:
        raise SystemExit(
            "accounts table does not exist. Start the webapp once so its migrations "
            "run, then re-run this job."
        )
    columns = {r[1] for r in conn.execute("PRAGMA table_info(accounts)")}
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise SystemExit(
            f"accounts table is missing column(s): {', '.join(missing)}. The deployed "
            "webapp is older than this script expects; deploy first, then provision."
        )


def describe(conn: sqlite3.Connection, username: str) -> str:
    row = conn.execute(
        """
        SELECT id, username, display_name, role, is_active, approval_status,
               must_change_password
          FROM accounts WHERE username = ?
        """,
        (username,),
    ).fetchone()
    if row is None:
        return "<no such account>"
    return (
        f"id={row[0]} username={row[1]!r} display_name={row[2]!r} role={row[3]} "
        f"is_active={row[4]} approval_status={row[5]} must_change_password={row[6]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", required=True, help="Path to the live feedback.db")
    parser.add_argument("--username", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument(
        "--password-hash",
        required=True,
        help="pbkdf2_sha256$<rounds>$<salt>$<digest>, generated by the operator locally",
    )
    parser.add_argument("--role", default="pi_admin", choices=VALID_ROLES)
    parser.add_argument("--mode", default="create", choices=("create", "rotate"))
    args = parser.parse_args()

    username = args.username.strip().lower()
    if not username:
        raise SystemExit("--username is empty")
    if not HASH_PATTERN.match(args.password_hash):
        # Deliberately does not echo the value: it is not secret-equivalent, but
        # there is no reason to widen its exposure on a bad-input path.
        raise SystemExit(
            "--password-hash is not a pbkdf2_sha256$<rounds>$<salt>$<digest> string. "
            "Generate it with the snippet in .github/workflows/provision-smoke-account.yml."
        )

    conn = sqlite3.connect(args.db)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        assert_schema(conn)
        before = table_counts(conn)

        existing = conn.execute(
            "SELECT id FROM accounts WHERE username = ?", (username,)
        ).fetchone()

        if args.mode == "create":
            if existing is not None:
                raise SystemExit(
                    f"Account {username!r} already exists ({describe(conn, username)}). "
                    "Re-run with mode=rotate to re-point it at a new password."
                )
            conn.execute(
                """
                INSERT INTO accounts
                    (username, display_name, password_hash, role, is_active,
                     approval_status, approved_at, approval_note,
                     must_change_password)
                VALUES (?, ?, ?, ?, 1, 'approved', datetime('now'), ?, 0)
                """,
                (
                    username,
                    args.display_name.strip(),
                    args.password_hash,
                    args.role,
                    "Automated deploy smoke-check account (issue #98).",
                ),
            )
            action = "created"
        else:
            if existing is None:
                raise SystemExit(
                    f"Account {username!r} does not exist, so there is nothing to rotate. "
                    "Re-run with mode=create."
                )
            conn.execute(
                """
                UPDATE accounts
                   SET password_hash = ?,
                       display_name = ?,
                       role = ?,
                       is_active = 1,
                       approval_status = 'approved',
                       approved_at = COALESCE(approved_at, datetime('now')),
                       rejected_at = NULL,
                       must_change_password = 0,
                       updated_at = datetime('now')
                 WHERE username = ?
                """,
                (args.password_hash, args.display_name.strip(), args.role, username),
            )
            # A rotated password must not leave old sessions usable. This is the
            # same rule web/routers/auth.py applies on a PI/admin reset.
            revoked = conn.execute(
                """
                DELETE FROM account_sessions
                 WHERE account_id = (SELECT id FROM accounts WHERE username = ?)
                """,
                (username,),
            ).rowcount
            print(f"Revoked {max(revoked, 0)} open session(s) for {username!r}.")
            action = "rotated"

        after = table_counts(conn)
        if before != after:
            conn.rollback()
            raise SystemExit(
                f"Refusing to commit: append-only table counts changed {before} -> {after}. "
                "Nothing was written."
            )

        conn.commit()
        print(f"Account {action}: {describe(conn, username)}")
        print(
            "Set the repository secrets from the operator machine:\n"
            "  gh secret set LOCALLATIN_SMOKE_USERNAME --body '<username>'\n"
            "  gh secret set LOCALLATIN_SMOKE_PASSWORD --body '<password>'"
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
