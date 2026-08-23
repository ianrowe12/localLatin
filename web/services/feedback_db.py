from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac
import io
import json
import logging
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

from web.models import DEFAULT_VARIANT

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    model_slug TEXT NOT NULL,
    variant TEXT,
    outcome TEXT NOT NULL DEFAULT 'legacy_unresolved',
    correct_rank INTEGER,
    correct_dir TEXT,
    selected_ranks_json TEXT,
    notes TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL,
    reviewer_account_id INTEGER,
    schema_version INTEGER NOT NULL DEFAULT 2
);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_model ON feedback(model_slug);
-- idx_feedback_variant is created in _migrate(), after the additive ALTER that
-- gives pre-variant databases the column.
CREATE INDEX IF NOT EXISTS idx_feedback_reviewer ON feedback(reviewer);

CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'reviewer' CHECK (role IN ('reviewer', 'pi_admin')),
    is_active INTEGER NOT NULL DEFAULT 1,
    approval_status TEXT NOT NULL DEFAULT 'approved' CHECK (approval_status IN ('pending', 'approved', 'rejected')),
    approved_at TEXT,
    approved_by_account_id INTEGER,
    rejected_at TEXT,
    approval_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_login_at TEXT
);
CREATE TABLE IF NOT EXISTS account_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    last_seen_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_account_sessions_account ON account_sessions(account_id);
CREATE INDEX IF NOT EXISTS idx_account_sessions_expires ON account_sessions(expires_at);
"""

_EXPORT_COLUMNS = [
    "id",
    "query_id",
    "filename",
    "timestamp",
    "model_slug",
    "variant",
    "outcome",
    "correct_rank",
    "correct_dir",
    "notes",
    "reviewer",
    "reviewer_account_id",
    "schema_version",
    "selected_ranks_json",
]


class FeedbackDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._connection_lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._connection_lock:
            if self._db is not None:
                return
            await self._open_connection()

    async def _open_connection(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._migrate()
        await self._db.commit()
        logger.info("Feedback DB ready at %s", self.db_path)

    async def close(self) -> None:
        async with self._connection_lock:
            if self._db:
                await self._db.close()
                self._db = None

    async def _ensure_auth_connection(self) -> None:
        connection = self._db
        try:
            if connection is None:
                raise ValueError("no active connection")
            async with connection.execute(
                """
                SELECT accounts.approval_status,
                       accounts.is_active,
                       account_sessions.token_hash
                  FROM accounts
             LEFT JOIN account_sessions
                    ON account_sessions.account_id = accounts.id
                 LIMIT 0
                """
            ) as cursor:
                await cursor.fetchone()
            return
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            logger.warning(
                "Auth database connection failed health check; reconnecting (%s)",
                type(exc).__name__,
            )

        async with self._connection_lock:
            # Another request may already have repaired the shared connection.
            if self._db is not connection and self._db is not None:
                return
            self._db = None
            if connection is not None:
                try:
                    await connection.close()
                except (sqlite3.Error, ValueError, RuntimeError):
                    pass
            await self._open_connection()

    async def insert(
        self,
        query_id: int,
        model_slug: str,
        outcome: str,
        correct_rank: int | None,
        correct_dir: str | None,
        notes: str,
        reviewer: str,
        reviewer_account_id: int | None = None,
        selected_ranks: list[int] | None = None,
        variant: str = DEFAULT_VARIANT,
    ) -> dict:
        assert self._db is not None
        selected_ranks_json = json.dumps(selected_ranks) if selected_ranks else None
        cursor = await self._db.execute(
            """INSERT INTO feedback
                   (query_id, model_slug, variant, outcome, correct_rank, correct_dir, selected_ranks_json, notes, reviewer, reviewer_account_id, schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 2)""",
            (
                query_id,
                model_slug,
                variant,
                outcome,
                correct_rank,
                correct_dir,
                selected_ranks_json,
                notes,
                reviewer,
                reviewer_account_id,
            ),
        )
        await self._db.commit()
        row = await (
            await self._db.execute("SELECT * FROM feedback WHERE id = ?", (cursor.lastrowid,))
        ).fetchone()
        return _feedback_row(row)

    async def _migrate(self) -> None:
        assert self._db is not None
        rows = await (await self._db.execute("PRAGMA table_info(feedback)")).fetchall()
        columns = {r["name"] for r in rows}

        if "outcome" not in columns:
            await self._db.execute("ALTER TABLE feedback ADD COLUMN outcome TEXT")
        if "schema_version" not in columns:
            await self._db.execute(
                "ALTER TABLE feedback ADD COLUMN schema_version INTEGER DEFAULT 1"
            )
        if "selected_ranks_json" not in columns:
            await self._db.execute(
                "ALTER TABLE feedback ADD COLUMN selected_ranks_json TEXT"
            )
        if "variant" not in columns:
            # Additive only, and deliberately NOT backfilled: rows written before
            # this column existed were reviewed against the pre-variant frozen
            # predictions CSV, so stamping them 'sif_abtt' would misattribute
            # them. They keep variant NULL and simply never prefill a variant.
            await self._db.execute("ALTER TABLE feedback ADD COLUMN variant TEXT")

        await self._db.execute(
            """
            UPDATE feedback
               SET outcome = CASE
                   WHEN correct_rank BETWEEN 1 AND 10 THEN 'matched_rank'
                   WHEN correct_rank = 0 THEN 'none_of_top_k'
                   ELSE 'legacy_unresolved'
               END
             WHERE outcome IS NULL OR outcome = ''
            """
        )
        await self._db.execute(
            """
            UPDATE feedback
               SET correct_rank = 0,
                   correct_dir = NULL
             WHERE outcome = 'none_of_top_k'
            """
        )
        await self._db.execute(
            """
            UPDATE feedback
               SET correct_rank = NULL,
                   correct_dir = NULL
             WHERE outcome = 'skipped'
            """
        )
        await self._db.execute(
            """
            UPDATE feedback
               SET schema_version = 2
             WHERE schema_version IS NULL OR schema_version < 2
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_outcome ON feedback(outcome)"
        )
        if "reviewer_account_id" not in columns:
            await self._db.execute(
                "ALTER TABLE feedback ADD COLUMN reviewer_account_id INTEGER"
            )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_reviewer_account ON feedback(reviewer_account_id)"
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_latest
                ON feedback(query_id, model_slug, reviewer_account_id, timestamp, id)
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_variant ON feedback(variant)"
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_latest_variant
                ON feedback(query_id, model_slug, variant, reviewer_account_id, timestamp, id)
            """
        )

        account_rows = await (
            await self._db.execute("PRAGMA table_info(accounts)")
        ).fetchall()
        account_columns = {r["name"] for r in account_rows}
        if "approval_status" not in account_columns:
            await self._db.execute(
                """
                ALTER TABLE accounts
                ADD COLUMN approval_status TEXT NOT NULL DEFAULT 'approved'
                CHECK (approval_status IN ('pending', 'approved', 'rejected'))
                """
            )
        if "approved_at" not in account_columns:
            await self._db.execute("ALTER TABLE accounts ADD COLUMN approved_at TEXT")
        if "approved_by_account_id" not in account_columns:
            await self._db.execute(
                "ALTER TABLE accounts ADD COLUMN approved_by_account_id INTEGER"
            )
        if "rejected_at" not in account_columns:
            await self._db.execute("ALTER TABLE accounts ADD COLUMN rejected_at TEXT")
        if "approval_note" not in account_columns:
            await self._db.execute(
                "ALTER TABLE accounts ADD COLUMN approval_note TEXT NOT NULL DEFAULT ''"
            )
        await self._db.execute(
            """
            UPDATE accounts
               SET approval_status = 'approved',
                   updated_at = datetime('now')
             WHERE approval_status IS NULL OR approval_status = ''
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_accounts_approval_status ON accounts(approval_status)"
        )

    async def account_count(self) -> int:
        await self._ensure_auth_connection()
        assert self._db is not None
        row = await (await self._db.execute("SELECT COUNT(*) FROM accounts")).fetchone()
        return int(row[0])

    async def create_account(
        self,
        username: str,
        display_name: str,
        password: str,
        role: str,
        approval_status: str = "approved",
        approved_by_account_id: int | None = None,
        approval_note: str = "",
    ) -> dict:
        await self._ensure_auth_connection()
        assert self._db is not None
        password_hash = _hash_password(password)
        approved_at = "datetime('now')" if approval_status == "approved" else "NULL"
        cursor = await self._db.execute(
            f"""
            INSERT INTO accounts
                (username, display_name, password_hash, role, approval_status,
                 approved_at, approved_by_account_id, approval_note)
            VALUES (?, ?, ?, ?, ?, {approved_at}, ?, ?)
            """,
            (
                username.strip().lower(),
                display_name.strip(),
                password_hash,
                role,
                approval_status,
                approved_by_account_id,
                approval_note.strip(),
            ),
        )
        await self._db.commit()
        row = await (
            await self._db.execute("SELECT * FROM accounts WHERE id = ?", (cursor.lastrowid,))
        ).fetchone()
        return _public_account(row)

    async def verify_account(self, username: str, password: str) -> tuple[dict | None, str | None]:
        await self._ensure_auth_connection()
        assert self._db is not None
        row = await (
            await self._db.execute(
                "SELECT * FROM accounts WHERE username = ?",
                (username.strip().lower(),),
            )
        ).fetchone()
        if row is None or not _verify_password(password, row["password_hash"]):
            return None, "invalid"
        if row["approval_status"] == "pending":
            return None, "pending"
        if row["approval_status"] == "rejected":
            return None, "rejected"
        if int(row["is_active"]) != 1:
            return None, "inactive"
        if row["approval_status"] != "approved":
            return None, "inactive"
        await self._db.execute(
            "UPDATE accounts SET last_login_at = datetime('now') WHERE id = ?",
            (row["id"],),
        )
        await self._db.commit()
        return _public_account(row), None

    async def list_accounts(self, approval_status: str | None = None) -> list[dict]:
        await self._ensure_auth_connection()
        assert self._db is not None
        query = "SELECT * FROM accounts"
        params: list[str] = []
        if approval_status:
            query += " WHERE approval_status = ?"
            params.append(approval_status)
        query += " ORDER BY created_at DESC, id DESC"
        rows = await (await self._db.execute(query, params)).fetchall()
        return [_account_public(row) for row in rows]

    async def set_account_approval(
        self,
        account_id: int,
        approval_status: str,
        approver_account_id: int,
        note: str = "",
    ) -> dict | None:
        await self._ensure_auth_connection()
        assert self._db is not None
        if approval_status == "approved":
            await self._db.execute(
                """
                UPDATE accounts
                   SET approval_status = 'approved',
                       is_active = 1,
                       approved_at = datetime('now'),
                       approved_by_account_id = ?,
                       rejected_at = NULL,
                       approval_note = ?,
                       updated_at = datetime('now')
                 WHERE id = ?
                """,
                (approver_account_id, note.strip(), account_id),
            )
        elif approval_status == "rejected":
            await self._db.execute(
                """
                UPDATE accounts
                   SET approval_status = 'rejected',
                       is_active = 0,
                       rejected_at = datetime('now'),
                       approval_note = ?,
                       updated_at = datetime('now')
                 WHERE id = ?
                """,
                (note.strip(), account_id),
            )
            await self._db.execute(
                """
                UPDATE account_sessions
                   SET revoked_at = datetime('now')
                 WHERE account_id = ? AND revoked_at IS NULL
                """,
                (account_id,),
            )
        else:
            raise ValueError(f"Unsupported approval status: {approval_status}")

        await self._db.commit()
        row = await (
            await self._db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,))
        ).fetchone()
        return _account_public(row) if row is not None else None

    async def create_session(self, account_id: int, session_days: int) -> str:
        await self._ensure_auth_connection()
        assert self._db is not None
        token = secrets.token_urlsafe(32)
        expires_at = _utc_now() + timedelta(days=session_days)
        await self._db.execute(
            """
            INSERT INTO account_sessions (account_id, token_hash, expires_at)
            VALUES (?, ?, ?)
            """,
            (account_id, _hash_token(token), _format_time(expires_at)),
        )
        await self._db.commit()
        return token

    async def get_account_by_session(self, token: str | None) -> dict | None:
        if not token:
            return None
        await self._ensure_auth_connection()
        assert self._db is not None
        row = await (
            await self._db.execute(
                """
                SELECT accounts.*
                FROM account_sessions
                JOIN accounts ON accounts.id = account_sessions.account_id
                WHERE account_sessions.token_hash = ?
                  AND account_sessions.revoked_at IS NULL
                  AND account_sessions.expires_at > datetime('now')
                  AND accounts.is_active = 1
                  AND accounts.approval_status = 'approved'
                """,
                (_hash_token(token),),
            )
        ).fetchone()
        if row is None:
            return None
        await self._db.execute(
            "UPDATE account_sessions SET last_seen_at = datetime('now') WHERE token_hash = ?",
            (_hash_token(token),),
        )
        await self._db.commit()
        return _public_account(row)

    async def revoke_session(self, token: str | None) -> None:
        if not token:
            return
        await self._ensure_auth_connection()
        assert self._db is not None
        await self._db.execute(
            """
            UPDATE account_sessions
               SET revoked_at = datetime('now')
             WHERE token_hash = ? AND revoked_at IS NULL
            """,
            (_hash_token(token),),
        )
        await self._db.commit()

    async def get_reviewed_query_ids(self) -> set[int]:
        assert self._db is not None
        rows = await (
            await self._db.execute(
                """SELECT DISTINCT query_id FROM feedback
                   WHERE outcome IN ('matched_rank', 'none_of_top_k')"""
            )
        ).fetchall()
        return {r["query_id"] for r in rows}

    async def get_review_counts(self) -> dict[int, int]:
        assert self._db is not None
        rows = await (
            await self._db.execute(
                "SELECT query_id, COUNT(*) as cnt FROM feedback GROUP BY query_id"
            )
        ).fetchall()
        return {r["query_id"]: r["cnt"] for r in rows}

    async def get_query_statuses(self) -> dict[int, dict]:
        assert self._db is not None
        rows = await (
            await self._db.execute(
                """
                SELECT
                    query_id,
                    COUNT(*) AS cnt,
                    MAX(CASE WHEN outcome IN ('matched_rank', 'none_of_top_k') THEN 1 ELSE 0 END) AS has_review,
                    MAX(CASE WHEN outcome = 'skipped' THEN 1 ELSE 0 END) AS has_skip
                FROM feedback
                GROUP BY query_id
                """
            )
        ).fetchall()

        statuses: dict[int, dict] = {}
        for row in rows:
            if row["has_review"]:
                status = "reviewed"
            elif row["has_skip"]:
                status = "skipped"
            else:
                status = "unreviewed"
            statuses[row["query_id"]] = {
                "review_status": status,
                "review_count": row["cnt"],
            }
        return statuses

    async def get_stats(self) -> dict:
        assert self._db is not None
        total = (await (await self._db.execute("SELECT COUNT(*) FROM feedback")).fetchone())[0]

        status_row = await (
            await self._db.execute(
                """
                WITH per_query AS (
                    SELECT
                        query_id,
                        MAX(CASE WHEN outcome IN ('matched_rank', 'none_of_top_k') THEN 1 ELSE 0 END) AS has_review,
                        MAX(CASE WHEN outcome = 'skipped' THEN 1 ELSE 0 END) AS has_skip
                    FROM feedback
                    GROUP BY query_id
                )
                SELECT
                    COALESCE(SUM(CASE WHEN has_review = 1 THEN 1 ELSE 0 END), 0) AS reviewed_count,
                    COALESCE(SUM(CASE WHEN has_review = 0 AND has_skip = 1 THEN 1 ELSE 0 END), 0) AS skipped_count
                FROM per_query
                """
            )
        ).fetchone()
        reviewed = status_row["reviewed_count"]
        skipped = status_row["skipped_count"]

        unresolved = (
            await (
                await self._db.execute(
                    "SELECT COUNT(*) FROM feedback WHERE outcome = 'legacy_unresolved'"
                )
            ).fetchone()
        )[0]

        by_model_rows = await (
            await self._db.execute(
                "SELECT model_slug, COUNT(*) as cnt FROM feedback GROUP BY model_slug"
            )
        ).fetchall()
        by_model = {r["model_slug"]: r["cnt"] for r in by_model_rows}

        by_reviewer_rows = await (
            await self._db.execute(
                "SELECT reviewer, COUNT(*) as cnt FROM feedback GROUP BY reviewer"
            )
        ).fetchall()
        by_reviewer = {r["reviewer"]: r["cnt"] for r in by_reviewer_rows}

        outcome_rows = await (
            await self._db.execute(
                "SELECT outcome, COUNT(*) as cnt FROM feedback GROUP BY outcome"
            )
        ).fetchall()
        outcome_dist = {r["outcome"]: r["cnt"] for r in outcome_rows}

        rank_rows = await (
            await self._db.execute(
                """
                SELECT
                    CASE
                        WHEN outcome = 'matched_rank' THEN CAST(correct_rank AS TEXT)
                        ELSE outcome
                    END AS rank_val,
                    COUNT(*) as cnt
                FROM feedback
                GROUP BY rank_val
                """
            )
        ).fetchall()
        rank_dist = {str(r["rank_val"]): r["cnt"] for r in rank_rows}

        return {
            "feedback_count": total,
            "reviewed_count": reviewed,
            "skipped_count": skipped,
            "unresolved_count": unresolved,
            "reviews_by_model": by_model,
            "reviews_by_reviewer": by_reviewer,
            "outcome_distribution": outcome_dist,
            "rank_distribution": rank_dist,
        }

    async def get_recent_reviews(self, limit: int = 10) -> list[dict]:
        assert self._db is not None
        rows = await (
            await self._db.execute(
                """
                SELECT query_id, timestamp, model_slug, outcome, reviewer, correct_rank
                FROM feedback
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        ).fetchall()
        return [
            {
                "file_id": r["query_id"],
                "timestamp": r["timestamp"],
                "model_slug": r["model_slug"],
                "outcome": r["outcome"],
                "reviewer": r["reviewer"],
                "correct_rank": r["correct_rank"],
            }
            for r in rows
        ]

    async def get_needs_attention(self, limit: int = 10) -> list[dict]:
        assert self._db is not None
        rows = await (
            await self._db.execute(
                """
                SELECT query_id, timestamp, model_slug, outcome, notes, reviewer
                FROM feedback
                WHERE outcome IN ('skipped', 'none_of_top_k')
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        ).fetchall()
        return [
            {
                "file_id": r["query_id"],
                "timestamp": r["timestamp"],
                "model_slug": r["model_slug"],
                "outcome": r["outcome"],
                "notes": r["notes"],
                "reviewer": r["reviewer"],
            }
            for r in rows
        ]

    async def get_feedback_for_query(
        self,
        query_id: int,
        model: str | None = None,
        limit: int = 10,
        variant: str | None = None,
    ) -> list[dict]:
        assert self._db is not None
        query = "SELECT * FROM feedback WHERE query_id = ?"
        params: list = [query_id]
        if model:
            query += " AND model_slug = ?"
            params.append(model)
        if variant:
            query += " AND variant = ?"
            params.append(variant)
        query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = await (await self._db.execute(query, params)).fetchall()
        return [_feedback_row(row) for row in rows]

    async def get_latest_feedback(
        self,
        query_id: int,
        model_slug: str,
        reviewer_account_id: int,
        variant: str = DEFAULT_VARIANT,
    ) -> dict | None:
        """Latest row for this reviewer on (query, model, variant).

        Keyed by variant so a note saved while reviewing sif_abtt never prefills
        the form for raw -- different rankings, different answers.
        """
        assert self._db is not None
        row = await (
            await self._db.execute(
                """
                SELECT *
                FROM feedback
                WHERE query_id = ?
                  AND model_slug = ?
                  AND variant = ?
                  AND reviewer_account_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (query_id, model_slug, variant, reviewer_account_id),
            )
        ).fetchone()
        return _feedback_row(row) if row is not None else None

    async def get_next_unreviewed(self, all_file_ids: list[int], limit: int = 5) -> list[int]:
        statuses = await self.get_query_statuses()
        result: list[int] = []
        for fid in all_file_ids:
            if statuses.get(fid, {}).get("review_status", "unreviewed") == "unreviewed":
                result.append(fid)
                if len(result) >= limit:
                    break
        return result

    async def export_csv(
        self,
        model: str | None = None,
        variant: str | None = None,
        reviewer: str | None = None,
        outcome: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        filename_by_query: dict[int, str] | None = None,
    ) -> str:
        assert self._db is not None
        query = "SELECT * FROM feedback WHERE 1=1"
        params: list = []
        if model:
            query += " AND model_slug = ?"
            params.append(model)
        if variant:
            query += " AND variant = ?"
            params.append(variant)
        if reviewer:
            query += " AND reviewer = ?"
            params.append(reviewer)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        if status:
            if status == "reviewed":
                query += " AND outcome IN ('matched_rank', 'none_of_top_k')"
            elif status == "skipped":
                query += " AND outcome = 'skipped'"
            elif status == "needs_attention":
                query += " AND outcome IN ('skipped', 'none_of_top_k')"
        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from)
        if date_to:
            query += " AND timestamp <= ?"
            params.append(date_to)
        query += " ORDER BY timestamp, id"

        rows = await (await self._db.execute(query, params)).fetchall()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=_EXPORT_COLUMNS)
        writer.writeheader()
        filename_lookup = filename_by_query or {}
        for r in rows:
            row = dict(r)
            row["filename"] = filename_lookup.get(
                row["query_id"], f"unknown-{row['query_id']}"
            )
            writer.writerow({column: row.get(column) for column in _EXPORT_COLUMNS})
        return output.getvalue()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    ).hex()
    return f"pbkdf2_sha256$200000${salt}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, rounds, salt, expected = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(rounds),
        ).hex()
        return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError):
        return False


def _feedback_row(row: aiosqlite.Row) -> dict:
    values = dict(row)
    selected_ranks_json = values.get("selected_ranks_json")
    values["selected_ranks"] = None
    if selected_ranks_json:
        try:
            selected_ranks = json.loads(selected_ranks_json)
            if isinstance(selected_ranks, list):
                values["selected_ranks"] = [int(rank) for rank in selected_ranks]
        except (TypeError, ValueError, json.JSONDecodeError):
            values["selected_ranks"] = None
    return values


def _public_account(row: aiosqlite.Row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
        "approval_status": row["approval_status"],
    }


def _account_public(row: aiosqlite.Row) -> dict:
    account = _public_account(row)
    account.update(
        {
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_login_at": row["last_login_at"],
            "approved_at": row["approved_at"],
            "approved_by_account_id": row["approved_by_account_id"],
            "rejected_at": row["rejected_at"],
            "approval_note": row["approval_note"] or "",
        }
    )
    return account
