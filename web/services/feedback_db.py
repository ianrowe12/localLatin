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
    correct_rank INTEGER,
    correct_dir TEXT,
    notes TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_model ON feedback(model_slug);
CREATE INDEX IF NOT EXISTS idx_feedback_reviewer ON feedback(reviewer);
"""


class FeedbackDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info("Feedback DB ready at %s", self.db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def insert(
        self,
        query_id: int,
        model_slug: str,
        correct_rank: int | None,
        correct_dir: str | None,
        notes: str,
        reviewer: str,
    ) -> dict:
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO feedback (query_id, model_slug, correct_rank, correct_dir, notes, reviewer)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (query_id, model_slug, correct_rank, correct_dir, notes, reviewer),
        )
        await self._db.commit()
        row = await (
            await self._db.execute("SELECT * FROM feedback WHERE id = ?", (cursor.lastrowid,))
        ).fetchone()
        return dict(row)

    async def get_reviewed_query_ids(self) -> set[int]:
        assert self._db is not None
        rows = await (
            await self._db.execute("SELECT DISTINCT query_id FROM feedback")
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

    async def get_stats(self) -> dict:
        assert self._db is not None
        total = (await (await self._db.execute("SELECT COUNT(*) FROM feedback")).fetchone())[0]

        reviewed = (
            await (
                await self._db.execute("SELECT COUNT(DISTINCT query_id) FROM feedback")
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

        rank_rows = await (
            await self._db.execute(
                """SELECT COALESCE(correct_rank, 0) as rank_val, COUNT(*) as cnt
                   FROM feedback GROUP BY rank_val"""
            )
        ).fetchall()
        rank_dist = {str(r["rank_val"]): r["cnt"] for r in rank_rows}

        return {
            "feedback_count": total,
            "reviewed_count": reviewed,
            "reviews_by_model": by_model,
            "reviews_by_reviewer": by_reviewer,
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
        assert self._db is not None
        rows = await (
            await self._db.execute("SELECT DISTINCT query_id FROM feedback")
        ).fetchall()
        reviewed = {r["query_id"] for r in rows}
        result: list[int] = []
        for fid in all_file_ids:
            if fid not in reviewed:
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
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            for r in rows:
                writer.writerow(dict(r))
        return output.getvalue()
