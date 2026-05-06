from __future__ import annotations

import csv
import io
import logging
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
    schema_version INTEGER NOT NULL DEFAULT 2
);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_model ON feedback(model_slug);
CREATE INDEX IF NOT EXISTS idx_feedback_reviewer ON feedback(reviewer);
"""

_EXPORT_COLUMNS = [
    "id",
    "query_id",
    "timestamp",
    "model_slug",
    "outcome",
    "correct_rank",
    "correct_dir",
    "notes",
    "reviewer",
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
    ) -> dict:
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO feedback
                   (query_id, model_slug, outcome, correct_rank, correct_dir, notes, reviewer, schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, 2)""",
            (query_id, model_slug, outcome, correct_rank, correct_dir, notes, reviewer),
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
                "SELECT DISTINCT query_id, timestamp, model_slug FROM feedback ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        ).fetchall()
        return [
            {
                "file_id": r["query_id"],
                "timestamp": r["timestamp"],
                "model_slug": r["model_slug"],
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
        self, model: str | None = None, reviewer: str | None = None
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
        query += " ORDER BY timestamp"

        rows = await (await self._db.execute(query, params)).fetchall()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=_EXPORT_COLUMNS)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            writer.writerow({column: row.get(column) for column in _EXPORT_COLUMNS})
        return output.getvalue()
