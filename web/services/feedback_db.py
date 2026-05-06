from __future__ import annotations

import csv
import hashlib
import hmac
import io
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    model_slug TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'legacy_unresolved',
    correct_rank INTEGER,
    correct_dir TEXT,
    notes TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL,
    reviewer_account_id INTEGER,
    schema_version INTEGER NOT NULL DEFAULT 2
);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_model ON feedback(model_slug);
CREATE INDEX IF NOT EXISTS idx_feedback_reviewer ON feedback(reviewer);

CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'reviewer' CHECK (role IN ('reviewer', 'pi_admin')),
    is_active INTEGER NOT NULL DEFAULT 1,
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
    "outcome",
    "correct_rank",
    "correct_dir",
    "notes",
    "reviewer",
    "reviewer_account_id",
    "schema_version",
]


class FeedbackDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._db is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._migrate()
        await self._db.commit()
        logger.info("Feedback DB ready at %s", self.db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

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
    ) -> dict:
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO feedback
                   (query_id, model_slug, outcome, correct_rank, correct_dir, notes, reviewer, reviewer_account_id, schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 2)""",
            (
                query_id,
                model_slug,
                outcome,
                correct_rank,
                correct_dir,
                notes,
                reviewer,
                reviewer_account_id,
            ),
        )
        await self._db.commit()
        row = await (
            await self._db.execute("SELECT * FROM feedback WHERE id = ?", (cursor.lastrowid,))
        ).fetchone()
        return dict(row)

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

    async def account_count(self) -> int:
        assert self._db is not None
        row = await (await self._db.execute("SELECT COUNT(*) FROM accounts")).fetchone()
        return int(row[0])

    async def create_account(
        self,
        username: str,
        display_name: str,
        password: str,
        role: str,
    ) -> dict:
        assert self._db is not None
        password_hash = _hash_password(password)
        cursor = await self._db.execute(
            """
            INSERT INTO accounts (username, display_name, password_hash, role)
            VALUES (?, ?, ?, ?)
            """,
            (username.strip().lower(), display_name.strip(), password_hash, role),
        )
        await self._db.commit()
        row = await (
            await self._db.execute("SELECT * FROM accounts WHERE id = ?", (cursor.lastrowid,))
        ).fetchone()
        return _public_account(row)

    async def verify_account(self, username: str, password: str) -> dict | None:
        assert self._db is not None
        row = await (
            await self._db.execute(
                "SELECT * FROM accounts WHERE username = ? AND is_active = 1",
                (username.strip().lower(),),
            )
        ).fetchone()
        if row is None or not _verify_password(password, row["password_hash"]):
            return None
        await self._db.execute(
            "UPDATE accounts SET last_login_at = datetime('now') WHERE id = ?",
            (row["id"],),
        )
        await self._db.commit()
        return _public_account(row)

    async def create_session(self, account_id: int, session_days: int) -> str:
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
        assert self._db is not None
        if not token:
            return None
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
        assert self._db is not None
        if not token:
            return
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


def _public_account(row: aiosqlite.Row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
    }
