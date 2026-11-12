from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Mapping

import aiosqlite

from web.exceptions import (
    LastPiAdminError,
    ReviewerDirLimitError,
    ReviewerDirSeedExistsError,
    UnscorableSeedError,
)
from web.services import ccl_key_export, ccl_keys
from web.variants import DEFAULT_VARIANT

logger = logging.getLogger(__name__)

#: `(query_id, model_slug, variant, rank) -> directory name or None`.
RankDirResolver = Callable[[int, str, "str | None", int], Awaitable["str | None"]]

#: Digest (`corpus_digest`) of the query corpus every row written before issue
#: #273 was saved against: the 2,238 files of `data/canon_unlabelled/`, file_id
#: i being the i-th name in sorted order. That listing has not changed since it
#: was committed on 2026-03-28, and every prediction CSV in every data release
#: (data-20260823 through data-20260906-v1) maps file_id to it with no
#: exception. `tests/test_xml_id_keys.py` pins this constant to the repo corpus.
PRE_XML_ID_CORPUS_SHA256 = (
    "23e913f50d96be43e8b412d393302f4b7515586cd9f6b40aecbd0269f81188af"
)


class CorpusMismatchError(RuntimeError):
    """The served query corpus disagrees with the witness ids stored in the DB.

    Raised at startup, so the app refuses to serve rather than show reviews
    against the wrong witness (issue #273).
    """


def corpus_digest(filename_by_query: Mapping[int, str]) -> str:
    """sha256 over `"<file_id>\\t<filename>\\n"` in file_id order.

    Covers the id-to-name mapping, not only the listing, because a query id is
    exactly what an older feedback row stored.
    """
    text = "".join(
        f"{query_id}\t{filename_by_query[query_id]}\n"
        for query_id in sorted(filename_by_query)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
    must_change_password INTEGER NOT NULL DEFAULT 0,
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

#: Audit trail for role changes (issue #265). Accounts are mutable and are not
#: part of the append-only feedback log, but who promoted or demoted whom must
#: still be recoverable. Created in _migrate() with IF NOT EXISTS, so an older
#: database gains it on next boot without any existing row being touched.
#: `changed_by_account_id` is NULL for a change made outside the app
#: (scripts/webapp/set_account_role.py), whose note starts with "cli: ".
#: scripts/webapp/set_account_role.py carries a byte-identical copy of this DDL
#: (it must run without the web package); tests/test_set_account_role.py pins
#: the two together.
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

#: Roles an account may hold; mirrors the CHECK constraint on accounts.role.
ACCOUNT_ROLES = ("reviewer", "pi_admin")

#: Every read of a feedback row carries the author's login name alongside the
#: stored display name, so shared notes and review packets can attribute a note
#: to a person. LEFT JOIN, never INNER: rows written before accounts existed
#: (reviewer_account_id NULL) and rows whose account was later removed must
#: still come back, just with reviewer_username NULL.
_FEEDBACK_SELECT = """
    SELECT feedback.*, accounts.username AS reviewer_username
    FROM feedback
    LEFT JOIN accounts ON accounts.id = feedback.reviewer_account_id
"""

# Columns the auth layer reads unconditionally; verified after every migration.
_REQUIRED_ACCOUNT_COLUMNS = {
    "approval_status",
    "is_active",
    "must_change_password",
    "password_hash",
    "role",
}

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
    "selected_ranks_json",
    # Appended, not inserted: anything parsing the export positionally keeps
    # working. Empty for rows written before the variant column existed.
    "variant",
    # Also appended, for the same reason. Empty on every row whose reviewer
    # typed no CCL key, which is every row written before issue #196.
    "ccl_key",
    "ccl_key_action",
    "ccl_key_dir",
    "ccl_key_rank",
    # Appended (issue #273). Empty only on a row whose query id the corpus it
    # was saved against never knew; every older row is backfilled at startup.
    "xml_id",
    # The directory behind every selected rank, as the reviewer saw it at save
    # time. Empty on every row written before issue #273.
    "selected_dirs_json",
]

#: Tables that reference an unlabelled query, as (table, query id column,
#: xml_id column). Every one gains the xml_id column in `_migrate()`, is
#: backfilled by `adopt_corpus()` and is checked by its startup guard.
_QUERY_REFERENCES = (
    ("feedback", "query_id", "xml_id"),
    ("reviewer_dirs", "seed_query_id", "seed_xml_id"),
    ("reviewer_dir_members", "query_id", "xml_id"),
)

#: Audit trail of the one-time xml_id backfill (issue #273): which corpus the
#: ids were read from and how many rows got one. Additive like every table
#: here; a row is written only by a boot that actually filled something.
XML_ID_BACKFILLS_DDL = """
CREATE TABLE IF NOT EXISTS xml_id_backfills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backfilled_at TEXT NOT NULL DEFAULT (datetime('now')),
    corpus_sha256 TEXT NOT NULL,
    corpus_size INTEGER NOT NULL,
    data_release TEXT NOT NULL DEFAULT '',
    feedback_rows INTEGER NOT NULL DEFAULT 0,
    reviewer_dir_rows INTEGER NOT NULL DEFAULT 0,
    reviewer_dir_member_rows INTEGER NOT NULL DEFAULT 0
)
"""


class FeedbackDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._connection_lock = asyncio.Lock()
        # The served corpus, bound by `adopt_corpus()` at startup (issue #273).
        # None until then: a bare FeedbackDB (as in unit tests) keys reads by
        # query id alone and writes rows without an xml_id.
        self._qid_to_xml: dict[int, str] | None = None
        self._xml_to_qid: dict[str, int] | None = None

    async def connect(self) -> None:
        async with self._connection_lock:
            if self._db is not None:
                return
            await self._open_connection()

    async def _open_connection(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Before the first write of the xml_id migration, and only then.
        await asyncio.to_thread(_backup_before_xml_id_migration, self.db_path)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._migrate()
        await self._assert_account_schema()
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
        selected_dirs: list[str] | None = None,
    ) -> dict:
        """Append one feedback row.

        `selected_dirs` is the directory behind each selected rank, in the
        same order, resolved by the caller from the snapshot the reviewer saw
        (issue #273). The witness `xml_id` comes from the bound corpus.
        """
        assert self._db is not None
        selected_ranks_json = json.dumps(selected_ranks) if selected_ranks else None
        selected_dirs_json = json.dumps(selected_dirs) if selected_dirs else None
        cursor = await self._db.execute(
            """INSERT INTO feedback
                   (query_id, xml_id, model_slug, variant, outcome, correct_rank, correct_dir, selected_ranks_json, selected_dirs_json, notes, reviewer, reviewer_account_id, schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 2)""",
            (
                query_id,
                self._xml_id_for(query_id),
                model_slug,
                variant,
                outcome,
                correct_rank,
                correct_dir,
                selected_ranks_json,
                selected_dirs_json,
                notes,
                reviewer,
                reviewer_account_id,
            ),
        )
        await self._db.commit()
        row = await (
            await self._db.execute(
                _FEEDBACK_SELECT + " WHERE feedback.id = ?", (cursor.lastrowid,)
            )
        ).fetchone()
        return _feedback_row(row)

    async def _migrate(self) -> None:
        """Additive schema repair, run on every boot.

        Warning for anyone tempted to enforce the append-only rule with a real
        `BEFORE UPDATE` trigger on `feedback` (the tests install exactly such a
        trigger, but only after startup): two of the normalization UPDATEs
        below are unconditional. `WHERE outcome = 'none_of_top_k'` and
        `WHERE outcome = 'skipped'` re-fire on already-normalized rows every
        time the app starts, so a schema-level trigger would abort startup as
        soon as one such row exists. Guard them first -- e.g. add
        `AND (correct_rank IS NOT 0 OR correct_dir IS NOT NULL)` -- so they
        become genuine no-ops once the data is clean.
        """
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

        # --- CCL key carried by a None-of-top-N assessment (issue #196) ------
        # Three columns, all NULL on every row written before this release and
        # never backfilled: a reviewer who did not type a key did not type one,
        # and no rule can recover a key from a row that has none.
        #
        # `correct_dir` is deliberately untouched by this feature. It means
        # "the directory this document was filed under, resolved from the rank
        # the reviewer pressed", and every consumer -- the export, the packets,
        # the dashboard, the reviewer-directory membership rule -- reads it that
        # way. A key names a directory the ranking did NOT offer, so writing it
        # there would silently change the meaning of a column that already has
        # one, for every row ever written.
        for column, ddl in (
            # Exactly as typed, whitespace-normalised. See services/ccl_keys.py.
            ("ccl_key", "ALTER TABLE feedback ADD COLUMN ccl_key TEXT"),
            # What the server did with it: matched_labelled_dir /
            # joined_reviewer_dir / created_reviewer_dir / seed_taken.
            ("ccl_key_action", "ALTER TABLE feedback ADD COLUMN ccl_key_action TEXT"),
            # The directory that action resolved to: a labelled directory name
            # or a reviewer dir_id.
            ("ccl_key_dir", "ALTER TABLE feedback ADD COLUMN ccl_key_dir TEXT"),
            # The rank that directory held in the ranking this assessment was
            # made against, when the key named one of its candidates. NULL
            # otherwise, which is the ordinary case: the whole point of the key
            # is a source the ten did NOT offer.
            ("ccl_key_rank", "ALTER TABLE feedback ADD COLUMN ccl_key_rank INTEGER"),
        ):
            if column not in columns:
                await self._db.execute(ddl)

        # --- witness identity and the names behind each pick (issue #273) ---
        # Two nullable columns. `query_id` is a position in the sorted corpus,
        # so a release that adds or removes one query file would shift every
        # later row onto another witness; `xml_id` is the filename stem, which
        # does not move. It is filled for older rows by `adopt_corpus()` (the
        # only thing that knows the corpus) and is the ONLY column that
        # backfill writes. `selected_dirs_json` is never backfilled: what a
        # reviewer saw at a rank is known only at save time.
        for column, ddl in (
            ("xml_id", "ALTER TABLE feedback ADD COLUMN xml_id TEXT"),
            (
                "selected_dirs_json",
                "ALTER TABLE feedback ADD COLUMN selected_dirs_json TEXT",
            ),
        ):
            if column not in columns:
                await self._db.execute(ddl)
        # idx_feedback_xml_id is created below, once reviewer_account_id exists.
        await self._db.execute(XML_ID_BACKFILLS_DDL)

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
        # The same, led by the witness id (issue #273): serves a plain witness
        # lookup and the scoped "latest decision" read.
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_xml_id
                ON feedback(xml_id, model_slug, variant, reviewer_account_id, timestamp, id)
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
        if "must_change_password" not in account_columns:
            # Additive: existing accounts keep their password and are never
            # retroactively forced through a change. Only an admin reset sets
            # this flag.
            await self._db.execute(
                "ALTER TABLE accounts ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
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

        # --- account role audit (issue #265) --------------------------------
        # Additive: one new table, no existing row touched. executescript()
        # would COMMIT the migration transaction early, so run each statement.
        for statement in ACCOUNT_ROLE_CHANGES_DDL.split(";"):
            if statement.strip():
                await self._db.execute(statement)

        # --- reviewer directories (issue #95) -------------------------------
        # Purely additive: two new tables, no column added to and no row
        # touched in `feedback`. A database written before this release gains
        # them on next open and its existing rows are read back byte-identical.
        # Both tables are append-only like `feedback` itself -- a directory is
        # never renamed or deleted and a membership is never withdrawn, so the
        # log stays a faithful record of what each reviewer actually asserted.
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS reviewer_dirs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dir_id TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                seed_query_id INTEGER NOT NULL,
                model_slug TEXT NOT NULL DEFAULT '',
                variant TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                created_by TEXT NOT NULL DEFAULT '',
                created_by_account_id INTEGER
            )
            """
        )
        # Historical seeds may have multiple directories. A UNIQUE index would
        # fail on those rows; deleting or merging them would erase assertions.
        # Keep this non-unique and prevent additions in create_reviewer_dir's
        # serialized transaction, including additions to already-duplicate seeds.
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviewer_dirs_seed ON reviewer_dirs(seed_query_id)"
        )
        await self._migrate_reviewer_dir_keys()
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS reviewer_dir_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dir_id TEXT NOT NULL,
                query_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                added_at TEXT NOT NULL DEFAULT (datetime('now')),
                added_by TEXT NOT NULL DEFAULT '',
                added_by_account_id INTEGER,
                UNIQUE (dir_id, query_id)
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviewer_dir_members_dir ON reviewer_dir_members(dir_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviewer_dir_members_query ON reviewer_dir_members(query_id)"
        )

        # --- witness identity on the reviewer-directory tables (issue #273) -
        # Same additive rule as feedback.xml_id: nullable, filled for older
        # rows by `adopt_corpus()` and nothing else touched.
        for table, column in (
            ("reviewer_dirs", "seed_xml_id"),
            ("reviewer_dir_members", "xml_id"),
        ):
            table_rows = await (
                await self._db.execute(f"PRAGMA table_info({table})")
            ).fetchall()
            if column not in {r["name"] for r in table_rows}:
                await self._db.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} TEXT"
                )
            await self._db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table}({column})"
            )

    async def _migrate_reviewer_dir_keys(self) -> None:
        """Give reviewer directories a `ccl_key`, keeping every existing name.

        Issue #196 point 5. A directory created before this release is named by
        whatever the reviewer typed into the retired naming form -- usually the
        seed's filename, occasionally a real source key. Both tables are
        append-only and there is no rename, so this migration does NOT touch a
        single `label`: what a reviewer called a directory is what it is still
        called.

        It adds one column and fills it only where the answer is already
        written down: a label that IS a CCL source key by the
        `scripts/data/label_taxonomy.py` rule becomes that directory's key, so
        an evaluator typing `CTOU.567.16` joins the directory somebody already
        made for it instead of creating a second one. A label that is not
        key-shaped keeps `ccl_key = ''`, which means "no key recorded" and never
        matches anything -- inventing a key from `New directory from BN2123.89r.5`
        would attach a citation nobody made.

        Idempotent twice over: the ALTER is guarded by the column list, and the
        backfill only ever writes rows whose `ccl_key` is still empty, so the
        second boot updates nothing. Preserving `feedback.db` in place is the
        point -- see deploy/deploy.sh, which never writes into `data/`.
        """
        assert self._db is not None
        rows = await (
            await self._db.execute("PRAGMA table_info(reviewer_dirs)")
        ).fetchall()
        columns = {r["name"] for r in rows}
        if "ccl_key" not in columns:
            await self._db.execute(
                "ALTER TABLE reviewer_dirs ADD COLUMN ccl_key TEXT NOT NULL DEFAULT ''"
            )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviewer_dirs_key ON reviewer_dirs(ccl_key)"
        )
        # Key-shapedness is a Python rule (a regex over Unicode), so the
        # candidates are read out and only the matching ones written back.
        # `ccl_key = ''` is the only state this will overwrite.
        candidates = await (
            await self._db.execute(
                "SELECT dir_id, label FROM reviewer_dirs "
                "WHERE ccl_key IS NULL OR ccl_key = ''"
            )
        ).fetchall()
        for row in candidates:
            label = str(row["label"] or "")
            if not ccl_keys.is_key_shaped(label):
                continue
            await self._db.execute(
                "UPDATE reviewer_dirs SET ccl_key = ? WHERE dir_id = ?",
                (ccl_keys.normalize_ccl_key(label), row["dir_id"]),
            )

    async def _assert_account_schema(self) -> None:
        """Refuse to serve if the accounts table lost a security-relevant column.

        `must_change_password` gates every authenticated route, so a database
        that somehow skipped the migration must fail loudly at startup rather
        than answer requests with the flag silently absent.
        """
        assert self._db is not None
        rows = await (
            await self._db.execute("PRAGMA table_info(accounts)")
        ).fetchall()
        columns = {r["name"] for r in rows}
        missing = _REQUIRED_ACCOUNT_COLUMNS - columns
        if missing:
            raise RuntimeError(
                "accounts table is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )

    # --- witness identity (issue #273) ----------------------------------------

    async def adopt_corpus(
        self, filename_by_query: Mapping[int, str], *, data_release: str = ""
    ) -> dict:
        """Check the served corpus against the stored witness ids, then bind it.

        Called once at startup, after `connect()`, with the corpus the app is
        about to serve (`DataStore.file_id_to_filename`). In order:

        1. GUARD. Every stored `(query id, xml_id)` pair must agree with the
           served corpus: the id must name that witness, and the witness must
           sit at that id. Any disagreement raises `CorpusMismatchError`, so
           the app refuses to start instead of serving reviews against the
           wrong text. A pair whose id and witness are both absent from the
           corpus is only logged: nothing is shown against another witness.
        2. BACKFILL. Rows written before xml_id existed get it from the served
           corpus. Only the NULL xml_id column is written; nothing else on any
           row changes, so the append-only log keeps every original value. On a
           deployment (`data_release` set) the served corpus must be the one
           those rows were saved against, `PRE_XML_ID_CORPUS_SHA256`, or the
           app refuses to start rather than guess. Once every row carries an
           id, later boots backfill nothing and this check never runs again.
        3. BIND. Later writes record the xml_id of the query they name, and
           reads translate a stored xml_id to the id the corpus serves it at.

        Returns counts for the startup log. No reviewer name is read here.
        """
        assert self._db is not None
        qid_to_xml = {
            int(query_id): ccl_key_export.xml_id_of(str(filename))
            for query_id, filename in filename_by_query.items()
        }
        xml_to_qid = {xml_id: query_id for query_id, xml_id in qid_to_xml.items()}

        mismatches, unserved = await self._stored_pairs_against(qid_to_xml, xml_to_qid)
        if mismatches:
            shown = "; ".join(
                f"{table} query_id {query_id} stores {stored!r} but the corpus serves "
                f"{served!r} there"
                for table, query_id, stored, served in mismatches[:5]
            )
            raise CorpusMismatchError(
                f"The served query corpus disagrees with {len(mismatches)} stored "
                f"witness id(s) in {self.db_path.name}: {shown}. Refusing to start, "
                "because every review would be shown against the wrong witness. "
                "Serve the data release these reviews were saved against, or re-key "
                "the stored query ids by xml_id before importing a corpus that adds "
                "or removes query files (issue #273)."
            )
        if unserved:
            logger.warning(
                "%d stored witness id(s) are not in the served corpus; their reviews "
                "are kept but not shown",
                unserved,
            )

        pending = await self._rows_awaiting_xml_id(qid_to_xml)
        pending_rows = sum(count for count, _ in pending.values())
        digest = corpus_digest(filename_by_query)
        filled = {table: 0 for table, _, _ in _QUERY_REFERENCES}
        if pending_rows:
            if data_release and digest != PRE_XML_ID_CORPUS_SHA256:
                raise CorpusMismatchError(
                    f"{pending_rows} stored row(s) predate witness ids and "
                    f"the served corpus (release {data_release!r}, sha256 {digest}) "
                    "is not the one they were saved against "
                    f"(sha256 {PRE_XML_ID_CORPUS_SHA256}). Refusing to start rather "
                    "than guess their witnesses: boot once on the earlier release so "
                    "the backfill runs, then deploy this one (issue #273)."
                )
            for table, id_column, xml_column in _QUERY_REFERENCES:
                for query_id in sorted(pending[table][1]):
                    # Writes the NULL witness id and nothing else; the
                    # `IS NULL` makes a rerun a no-op.
                    cursor = await self._db.execute(
                        f"UPDATE {table} SET {xml_column} = ? "
                        f"WHERE {id_column} = ? AND {xml_column} IS NULL",
                        (qid_to_xml[query_id], query_id),
                    )
                    filled[table] += max(cursor.rowcount, 0)
            await self._db.execute(
                """
                INSERT INTO xml_id_backfills
                    (corpus_sha256, corpus_size, data_release, feedback_rows,
                     reviewer_dir_rows, reviewer_dir_member_rows)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    digest,
                    len(qid_to_xml),
                    data_release,
                    filled["feedback"],
                    filled["reviewer_dirs"],
                    filled["reviewer_dir_members"],
                ),
            )
            await self._db.commit()
            logger.info(
                "Backfilled witness ids from corpus sha256 %s: %d feedback, %d "
                "reviewer directory and %d membership row(s)",
                digest,
                filled["feedback"],
                filled["reviewer_dirs"],
                filled["reviewer_dir_members"],
            )

        self._qid_to_xml = qid_to_xml
        self._xml_to_qid = xml_to_qid
        return {"backfilled": filled, "unserved": unserved, "corpus_sha256": digest}

    async def _stored_pairs_against(
        self, qid_to_xml: dict[int, str], xml_to_qid: dict[str, int]
    ) -> tuple[list[tuple[str, int, str, str | None]], int]:
        """Stored pairs the served corpus contradicts, and how many it lacks."""
        assert self._db is not None
        mismatches: list[tuple[str, int, str, str | None]] = []
        unserved = 0
        for table, id_column, xml_column in _QUERY_REFERENCES:
            rows = await (
                await self._db.execute(
                    f"SELECT DISTINCT {id_column} AS query_id, {xml_column} AS xml_id "
                    f"FROM {table} WHERE {xml_column} IS NOT NULL "
                    f"ORDER BY {id_column}"
                )
            ).fetchall()
            for row in rows:
                query_id, xml_id = int(row["query_id"]), str(row["xml_id"])
                served_xml = qid_to_xml.get(query_id)
                served_qid = xml_to_qid.get(xml_id)
                if served_xml is None and served_qid is None:
                    unserved += 1
                elif served_xml != xml_id or served_qid != query_id:
                    mismatches.append((table, query_id, xml_id, served_xml))
        return mismatches, unserved

    async def _rows_awaiting_xml_id(
        self, qid_to_xml: dict[int, str]
    ) -> dict[str, tuple[int, set[int]]]:
        """Rows with no xml_id whose query id the corpus can name, per table.

        Returns `{table: (row count, query ids)}`. A row whose id the corpus
        does not know stays NULL and is not counted: there is nothing to fill
        it from, and reads keep using its stored id.
        """
        assert self._db is not None
        pending: dict[str, tuple[int, set[int]]] = {}
        for table, id_column, xml_column in _QUERY_REFERENCES:
            rows = await (
                await self._db.execute(
                    f"SELECT {id_column} AS query_id, COUNT(*) AS n FROM {table} "
                    f"WHERE {xml_column} IS NULL GROUP BY {id_column}"
                )
            ).fetchall()
            known = [r for r in rows if int(r["query_id"]) in qid_to_xml]
            pending[table] = (
                sum(int(r["n"]) for r in known),
                {int(r["query_id"]) for r in known},
            )
        return pending

    def _xml_id_for(self, query_id: int) -> str | None:
        """The witness the bound corpus serves at this query id, if any."""
        if self._qid_to_xml is None:
            return None
        return self._qid_to_xml.get(int(query_id))

    def _served_query_id(self, query_id: int, xml_id: str | None) -> int | None:
        """Where a stored reference is served now: by xml_id when it has one.

        None when the witness is not in the served corpus. A row without an
        xml_id (or a DB with no bound corpus) keeps its stored id.
        """
        if xml_id and self._xml_to_qid is not None:
            return self._xml_to_qid.get(xml_id)
        return int(query_id)

    def _shown_query_id(self, query_id: int, xml_id: str | None) -> int:
        """`_served_query_id`, or the stored id for a witness no longer served.

        For lists that must name some id. The startup guard refuses a corpus
        that serves another witness at that stored id, so it names nothing
        else; the stats page shows it as `unknown-<id>`.
        """
        served = self._served_query_id(query_id, xml_id)
        return int(query_id) if served is None else served

    def _match_query(
        self, query_id: int, *, id_column: str, xml_column: str
    ) -> tuple[str, list]:
        """SQL predicate for "this row is about this query", xml_id first."""
        xml_id = self._xml_id_for(query_id)
        if xml_id is None:
            return f"{id_column} = ?", [query_id]
        return (
            f"({xml_column} = ? OR ({xml_column} IS NULL AND {id_column} = ?))",
            [xml_id, query_id],
        )

    # --- reviewer directories -------------------------------------------------

    async def create_reviewer_dir(
        self,
        *,
        label: str,
        seed_query_id: int,
        model_slug: str,
        variant: str,
        created_by: str,
        created_by_account_id: int | None,
        max_per_account: int,
    ) -> dict:
        """Check global seed uniqueness and account quota, then insert both rows.

        BEGIN IMMEDIATE serializes decisions across connections and processes.
        This connection belongs only to this operation: auth and feedback commit
        on the shared connection and must not commit a half-created directory.
        No migrations or matrix/scoring work run inside this transaction.

        Closing the dedicated connection rolls back any uncommitted writes on
        failure or cancellation. Cancellation during commit may leave both rows
        saved; callers recover the outcome through the seed-filtered GET.
        """
        assert self._db is not None
        # Generate the opaque id up front, without placeholders or UPDATEs.
        dir_id = f"reviewer-dir-{uuid.uuid4().hex[:12]}"
        async with aiosqlite.connect(str(self.db_path)) as transaction:
            transaction.row_factory = aiosqlite.Row
            await transaction.execute("BEGIN IMMEDIATE")
            seed_match, seed_params = self._match_query(
                seed_query_id, id_column="seed_query_id", xml_column="seed_xml_id"
            )
            existing = await (
                await transaction.execute(
                    "SELECT dir_id, label FROM reviewer_dirs "
                    f"WHERE {seed_match} ORDER BY id LIMIT 1",
                    seed_params,
                )
            ).fetchone()
            if existing is not None:
                raise ReviewerDirSeedExistsError(
                    seed_query_id, existing["dir_id"], existing["label"]
                )
            row = await (
                await transaction.execute(
                    "SELECT COUNT(*) FROM reviewer_dirs WHERE created_by_account_id IS ?",
                    (created_by_account_id,),
                )
            ).fetchone()
            created_count = int(row[0])
            if created_count >= max_per_account:
                raise ReviewerDirLimitError(created_count, max_per_account)

            # A label that IS a CCL source key becomes this directory's key,
            # by the same rule the migration applies to historical rows: a
            # directory named `CTOU.567.16` must be reachable by an evaluator
            # who types `CTOU.567.16`, whichever route created it. Anything else
            # keeps `ccl_key = ''`, which matches nothing.
            label = label.strip()
            dir_key = (
                ccl_keys.normalize_ccl_key(label)
                if ccl_keys.is_key_shaped(label)
                else ""
            )
            seed_xml_id = self._xml_id_for(seed_query_id)
            await transaction.execute(
                """
                INSERT INTO reviewer_dirs
                    (dir_id, label, ccl_key, seed_query_id, seed_xml_id, model_slug,
                     variant, created_by, created_by_account_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dir_id,
                    label,
                    dir_key,
                    seed_query_id,
                    seed_xml_id,
                    model_slug,
                    variant,
                    created_by,
                    created_by_account_id,
                ),
            )
            await transaction.execute(
                """
                INSERT INTO reviewer_dir_members
                    (dir_id, query_id, xml_id, role, added_by, added_by_account_id)
                VALUES (?, ?, ?, 'seed', ?, ?)
                """,
                (dir_id, seed_query_id, seed_xml_id, created_by, created_by_account_id),
            )
            row = await (
                await transaction.execute(
                    "SELECT * FROM reviewer_dirs WHERE dir_id = ?", (dir_id,)
                )
            ).fetchone()
            await transaction.commit()
            return dict(row)

    async def count_reviewer_dirs_by_account(self, account_id: int | None) -> int:
        """How many directories this account has created. Feeds the per-reviewer cap."""
        assert self._db is not None
        row = await (
            await self._db.execute(
                "SELECT COUNT(*) FROM reviewer_dirs WHERE created_by_account_id IS ?",
                (account_id,),
            )
        ).fetchone()
        return int(row[0])

    async def get_reviewer_dir_by_seed(self, seed_query_id: int) -> dict | None:
        """The oldest directory for this seed, ordered by insertion id.

        Historical duplicate seeds remain intact. This deterministic recovery
        choice does not hide them: list_reviewer_dirs returns every record.
        """
        assert self._db is not None
        seed_match, seed_params = self._match_query(
            seed_query_id, id_column="seed_query_id", xml_column="seed_xml_id"
        )
        row = await (
            await self._db.execute(
                f"SELECT dir_id FROM reviewer_dirs WHERE {seed_match} ORDER BY id LIMIT 1",
                seed_params,
            )
        ).fetchone()
        if row is None:
            return None
        return await self.get_reviewer_dir(row["dir_id"])

    async def list_reviewer_dirs(self) -> list[dict]:
        """Every reviewer directory, each with its member query ids.

        The whole table is read at once: it is reviewer-authored and grows by
        hand, so it stays small enough that per-request filtering in Python
        beats a query per candidate list.
        """
        assert self._db is not None
        dir_rows = await (
            await self._db.execute(
                "SELECT * FROM reviewer_dirs ORDER BY id"
            )
        ).fetchall()
        member_rows = await (
            await self._db.execute(
                "SELECT dir_id, query_id, xml_id, role FROM reviewer_dir_members ORDER BY id"
            )
        ).fetchall()
        members: dict[str, list[int]] = {}
        for row in member_rows:
            served = self._served_query_id(row["query_id"], row["xml_id"])
            if served is not None:
                members.setdefault(row["dir_id"], []).append(served)
        return [
            {
                **self._served_dir_record(row),
                "member_query_ids": members.get(row["dir_id"], []),
            }
            for row in dir_rows
        ]

    def _served_dir_record(self, row: aiosqlite.Row) -> dict:
        """A reviewer_dirs row with its seed id read through its xml_id.

        A seed the served corpus lacks keeps its stored id, which the corpus
        does not serve either (the startup guard refuses any other case), so
        nothing is ever shown against another witness.
        """
        record = dict(row)
        served = self._served_query_id(record["seed_query_id"], record.get("seed_xml_id"))
        if served is not None:
            record["seed_query_id"] = served
        return record

    async def get_reviewer_dir(self, dir_id: str) -> dict | None:
        assert self._db is not None
        row = await (
            await self._db.execute(
                "SELECT * FROM reviewer_dirs WHERE dir_id = ?", (dir_id,)
            )
        ).fetchone()
        if row is None:
            return None
        member_rows = await (
            await self._db.execute(
                "SELECT query_id, xml_id FROM reviewer_dir_members WHERE dir_id = ? ORDER BY id",
                (dir_id,),
            )
        ).fetchall()
        served = (self._served_query_id(r["query_id"], r["xml_id"]) for r in member_rows)
        return {
            **self._served_dir_record(row),
            "member_query_ids": [query_id for query_id in served if query_id is not None],
        }

    async def add_reviewer_dir_member(
        self,
        *,
        dir_id: str,
        query_id: int,
        added_by: str,
        added_by_account_id: int | None,
    ) -> bool:
        """Append a query to a reviewer directory. Idempotent, never updates.

        Returns True when a new membership row was written. The UNIQUE
        constraint plus INSERT OR IGNORE means re-submitting the same feedback
        adds nothing rather than duplicating or overwriting.

        The directory must already exist. Without this check an unknown
        ``dir_id`` wrote an orphan membership row that nothing could ever read
        or remove -- both tables are append-only with no DELETE and no admin
        removal route, so a typo or a stale client value was permanent.
        """
        assert self._db is not None
        exists = await (
            await self._db.execute(
                "SELECT 1 FROM reviewer_dirs WHERE dir_id = ?", (dir_id,)
            )
        ).fetchone()
        if exists is None:
            raise KeyError(dir_id)
        cursor = await self._db.execute(
            """
            INSERT OR IGNORE INTO reviewer_dir_members
                (dir_id, query_id, xml_id, role, added_by, added_by_account_id)
            VALUES (?, ?, ?, 'member', ?, ?)
            """,
            (dir_id, query_id, self._xml_id_for(query_id), added_by, added_by_account_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_reviewer_dir_by_key(self, key: str) -> dict | None:
        """The oldest reviewer directory this key reaches, or None.

        Key first, then label, by exactly the rule `insert_none_of_top_k` uses
        inside its transaction -- see `_reviewer_dir_for_key` for why a label is
        a second handle on the same directory.
        """
        assert self._db is not None
        dir_id = await _reviewer_dir_for_key(self._db, key)
        return await self.get_reviewer_dir(dir_id) if dir_id is not None else None

    async def insert_none_of_top_k(
        self,
        *,
        query_id: int,
        model_slug: str,
        variant: str,
        notes: str,
        reviewer: str,
        reviewer_account_id: int | None,
        ccl_key: str,
        labelled_dir: str | None,
        labelled_rank: int | None,
        seed_scorable: bool,
        max_per_account: int,
    ) -> tuple[dict, bool]:
        """Record "none of the ranked candidates" and act on the key. ONE write.

        Issue #196 point 2. The assessment and whatever the key implies are a
        single transaction on a dedicated connection, opened with
        BEGIN IMMEDIATE exactly like `create_reviewer_dir` (issue #165): either
        the reviewer's answer and the directory action are both stored, or
        neither is. The alternative -- the pattern the rank path still uses,
        where the feedback row commits and the membership row follows on a
        shared connection -- can leave an assessment claiming a key whose
        directory was never written.

        Returns ``(row, appended)``. ``appended`` is False for an IDENTICAL
        REPEAT -- same document, model, variant, account, key and note as this
        reviewer's own newest row -- in which case that row comes back unchanged
        and nothing at all is written. The feedback log stays append-only and
        gains no correction path; this only declines to append a row that would
        say exactly what the previous one says. It exists because the panel now
        shows a recorded answer with a "Change" action, and a reviewer who opens
        that form, changes nothing and presses Record again means "yes, that",
        not "record it twice". A changed note, a changed key or another
        reviewer's submission is a new assertion and is appended.

        `labelled_dir` and `labelled_rank` are resolved by the caller: the
        corpus lookup is a pure in-memory one (`ccl_keys.find_labelled_dir`) and
        the rank comes from the same candidate snapshot the route already
        fetched, so the record says whether the key the reviewer typed was in
        front of them all along.

        The six recorded actions:

        ``matched_labelled_dir``  the key names a labelled directory. NOTHING is
                                  created: that directory is corpus data, and a
                                  reviewer directory bearing its name would be a
                                  duplicate nothing can merge away.
        ``joined_reviewer_dir``   the key names a reviewer directory; the query joins it
        ``already_joined``        ... and was already one of its members, so the
                                  membership write is a no-op. Reported as its
                                  own branch rather than as a join, because
                                  telling a reviewer their document "joined" a
                                  group it has been in since last week is a
                                  small untruth in a record they are asked to
                                  trust.
        ``created_reviewer_dir``  no such directory; one is created, named by the key
        ``seed_taken``            no such directory, but this query already seeds
                                  another one. One directory per seed is a
                                  standing invariant (`create_reviewer_dir`
                                  refuses a second with 409), so the key is
                                  recorded against the assessment and nothing is
                                  created. `ccl_key_dir` stays NULL here: that
                                  column means "the directory this key resolved
                                  to", and the directory that blocked the write
                                  is not it.

        Without a key, `ccl_key_action` stays NULL and only the assessment is
        written -- which is the ordinary None-of-top-N save.

        Raises `ReviewerDirLimitError` when a creation would exceed the caller's
        quota and `UnscorableSeedError` when it would seed a directory from a
        query the degenerate-source guard excluded. Both are raised BEFORE
        anything is written, so the reviewer keeps their answer and can record
        it without the key.
        """
        assert self._db is not None
        key = ccl_keys.normalize_ccl_key(ccl_key)
        action: str | None = None
        key_dir: str | None = None

        async with aiosqlite.connect(str(self.db_path)) as transaction:
            transaction.row_factory = aiosqlite.Row
            await transaction.execute("BEGIN IMMEDIATE")

            xml_id = self._xml_id_for(query_id)
            query_match, query_params = self._match_query(
                query_id, id_column="query_id", xml_column="xml_id"
            )
            previous = await (
                await transaction.execute(
                    f"""SELECT * FROM feedback
                        WHERE {query_match} AND model_slug = ? AND variant IS ?
                          AND reviewer_account_id IS ?
                     ORDER BY id DESC LIMIT 1""",
                    (*query_params, model_slug, variant, reviewer_account_id),
                )
            ).fetchone()
            if _is_identical_none_repeat(previous, key, notes):
                row = await (
                    await transaction.execute(
                        _FEEDBACK_SELECT + " WHERE feedback.id = ?",
                        (previous["id"],),
                    )
                ).fetchone()
                # Read-only transaction: commit releases the write lock it took
                # without having written a thing.
                await transaction.commit()
                return _feedback_row(row), False

            if key and labelled_dir is not None:
                action, key_dir = "matched_labelled_dir", labelled_dir
            elif key:
                existing = await _reviewer_dir_for_key(transaction, key)
                if existing is not None:
                    # Idempotent by the UNIQUE (dir_id, query_id) constraint,
                    # and reported as what it was: a join or a no-op.
                    cursor = await transaction.execute(
                        """
                        INSERT OR IGNORE INTO reviewer_dir_members
                            (dir_id, query_id, xml_id, role, added_by, added_by_account_id)
                        VALUES (?, ?, ?, 'member', ?, ?)
                        """,
                        (existing, query_id, xml_id, reviewer, reviewer_account_id),
                    )
                    action = (
                        "joined_reviewer_dir"
                        if cursor.rowcount > 0
                        else "already_joined"
                    )
                    key_dir = existing
                else:
                    seed_match, seed_params = self._match_query(
                        query_id, id_column="seed_query_id", xml_column="seed_xml_id"
                    )
                    seeded = await (
                        await transaction.execute(
                            f"SELECT dir_id FROM reviewer_dirs WHERE {seed_match} "
                            "ORDER BY id LIMIT 1",
                            seed_params,
                        )
                    ).fetchone()
                    if seeded is not None:
                        # Named, but not resolved: see the docstring.
                        action, key_dir = "seed_taken", None
                    else:
                        if not seed_scorable:
                            raise UnscorableSeedError(query_id)
                        quota = await (
                            await transaction.execute(
                                "SELECT COUNT(*) FROM reviewer_dirs "
                                "WHERE created_by_account_id IS ?",
                                (reviewer_account_id,),
                            )
                        ).fetchone()
                        created_count = int(quota[0])
                        if created_count >= max_per_account:
                            raise ReviewerDirLimitError(created_count, max_per_account)
                        dir_id = f"reviewer-dir-{uuid.uuid4().hex[:12]}"
                        await transaction.execute(
                            """
                            INSERT INTO reviewer_dirs
                                (dir_id, label, ccl_key, seed_query_id, seed_xml_id,
                                 model_slug, variant, created_by, created_by_account_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                dir_id,
                                # Named by the key, never by a siglum: that is
                                # the correction Abigail asked for.
                                key,
                                key,
                                query_id,
                                xml_id,
                                model_slug,
                                variant,
                                reviewer,
                                reviewer_account_id,
                            ),
                        )
                        await transaction.execute(
                            """
                            INSERT INTO reviewer_dir_members
                                (dir_id, query_id, xml_id, role, added_by, added_by_account_id)
                            VALUES (?, ?, ?, 'seed', ?, ?)
                            """,
                            (dir_id, query_id, xml_id, reviewer, reviewer_account_id),
                        )
                        action, key_dir = "created_reviewer_dir", dir_id

            cursor = await transaction.execute(
                """INSERT INTO feedback
                       (query_id, xml_id, model_slug, variant, outcome, correct_rank,
                        correct_dir, selected_ranks_json, notes, reviewer,
                        reviewer_account_id, ccl_key, ccl_key_action, ccl_key_dir,
                        ccl_key_rank, schema_version)
                   VALUES (?, ?, ?, ?, 'none_of_top_k', 0, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, 2)""",
                (
                    query_id,
                    xml_id,
                    model_slug,
                    variant,
                    notes,
                    reviewer,
                    reviewer_account_id,
                    key or None,
                    action,
                    key_dir,
                    labelled_rank if action == "matched_labelled_dir" else None,
                ),
            )
            row = await (
                await transaction.execute(
                    _FEEDBACK_SELECT + " WHERE feedback.id = ?", (cursor.lastrowid,)
                )
            ).fetchone()
            await transaction.commit()
            return _feedback_row(row), True

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

    async def verify_account_password(self, account_id: int, password: str) -> bool:
        """True when ``password`` matches the stored hash for that account."""
        await self._ensure_auth_connection()
        assert self._db is not None
        row = await (
            await self._db.execute(
                "SELECT password_hash FROM accounts WHERE id = ?", (account_id,)
            )
        ).fetchone()
        if row is None:
            return False
        return _verify_password(password, row["password_hash"])

    async def set_account_password(
        self,
        account_id: int,
        new_password: str,
        must_change_password: bool = False,
        keep_session_token: str | None = None,
    ) -> dict | None:
        """Rehash the account password and drop its other live sessions.

        ``keep_session_token`` is the caller's own session, spared so a
        self-serve change does not sign the user out of the tab they are in.
        An admin reset passes None, which revokes every session.
        """
        await self._ensure_auth_connection()
        assert self._db is not None
        cursor = await self._db.execute(
            """
            UPDATE accounts
               SET password_hash = ?,
                   must_change_password = ?,
                   updated_at = datetime('now')
             WHERE id = ?
            """,
            (_hash_password(new_password), 1 if must_change_password else 0, account_id),
        )
        if cursor.rowcount == 0:
            await self._db.commit()
            return None
        if keep_session_token:
            await self._db.execute(
                """
                UPDATE account_sessions
                   SET revoked_at = datetime('now')
                 WHERE account_id = ?
                   AND revoked_at IS NULL
                   AND token_hash != ?
                """,
                (account_id, _hash_token(keep_session_token)),
            )
        else:
            await self._db.execute(
                """
                UPDATE account_sessions
                   SET revoked_at = datetime('now')
                 WHERE account_id = ? AND revoked_at IS NULL
                """,
                (account_id,),
            )
        await self._db.commit()
        row = await (
            await self._db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,))
        ).fetchone()
        return _account_public(row) if row is not None else None

    async def set_account_role(
        self,
        account_id: int,
        role: str,
        changed_by_account_id: int | None,
        note: str = "",
    ) -> tuple[dict | None, bool]:
        """Change an account's role and record who did it.

        Returns ``(account, changed)``: ``(None, False)`` for an unknown
        account, ``(account, False)`` when it already holds ``role`` (nothing
        is written, no audit row), ``(account, True)`` after a change.

        Raises :class:`LastPiAdminError` when the change would leave no
        approved, active PI/admin: there is no email recovery, so somebody must
        always be able to approve accounts and reset passwords.

        Sessions are NOT revoked. ``get_account_by_session`` re-reads the
        account row (role included) on every request, so the new role applies
        to the account's very next request without signing anybody out.

        Runs on its own connection under BEGIN IMMEDIATE, like
        create_reviewer_dir, so the last-admin count and the UPDATE cannot
        interleave with a concurrent change (the app's or the CLI's).
        """
        if role not in ACCOUNT_ROLES:
            raise ValueError(f"Unsupported role: {role}")
        await self._ensure_auth_connection()
        async with aiosqlite.connect(str(self.db_path)) as transaction:
            transaction.row_factory = aiosqlite.Row
            await transaction.execute("BEGIN IMMEDIATE")
            row = await (
                await transaction.execute(
                    "SELECT * FROM accounts WHERE id = ?", (account_id,)
                )
            ).fetchone()
            if row is None:
                return None, False
            old_role = row["role"]
            if old_role == role:
                return _account_public(row), False
            # Only an account that can currently sign in as PI/admin counts:
            # demoting a pending or rejected one cannot lock anybody out.
            if (
                old_role == "pi_admin"
                and row["approval_status"] == "approved"
                and int(row["is_active"]) == 1
            ):
                remaining = await (
                    await transaction.execute(
                        """
                        SELECT COUNT(*) FROM accounts
                         WHERE role = 'pi_admin'
                           AND approval_status = 'approved'
                           AND is_active = 1
                           AND id != ?
                        """,
                        (account_id,),
                    )
                ).fetchone()
                if int(remaining[0]) == 0:
                    raise LastPiAdminError(account_id)
            await transaction.execute(
                """
                UPDATE accounts
                   SET role = ?,
                       updated_at = datetime('now')
                 WHERE id = ?
                """,
                (role, account_id),
            )
            await transaction.execute(
                """
                INSERT INTO account_role_changes
                    (account_id, old_role, new_role, changed_by_account_id, note)
                VALUES (?, ?, ?, ?, ?)
                """,
                (account_id, old_role, role, changed_by_account_id, note.strip()),
            )
            updated = await (
                await transaction.execute(
                    "SELECT * FROM accounts WHERE id = ?", (account_id,)
                )
            ).fetchone()
            await transaction.commit()
            return _account_public(updated), True

    async def list_account_role_changes(
        self, account_id: int | None = None
    ) -> list[dict]:
        """The role audit trail, oldest first, optionally for one account."""
        await self._ensure_auth_connection()
        assert self._db is not None
        query = "SELECT * FROM account_role_changes"
        params: list[int] = []
        if account_id is not None:
            query += " WHERE account_id = ?"
            params.append(account_id)
        query += " ORDER BY id"
        rows = await (await self._db.execute(query, params)).fetchall()
        return [dict(row) for row in rows]

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
        return {
            query_id
            for query_id, status in (await self.get_query_statuses()).items()
            if status["review_status"] == "reviewed"
        }

    async def get_review_counts(self) -> dict[int, int]:
        return {
            query_id: status["review_count"]
            for query_id, status in (await self.get_query_statuses()).items()
        }

    async def get_query_statuses(self) -> dict[int, dict]:
        """Review status per served query id, read by witness (issue #273).

        Grouped by `(query_id, xml_id)` and then folded onto the id the served
        corpus gives each witness, so a stored id never decides which query a
        review belongs to when the row also names its witness. A witness the
        corpus does not serve is left out.
        """
        assert self._db is not None
        rows = await (
            await self._db.execute(
                """
                SELECT
                    query_id,
                    xml_id,
                    COUNT(*) AS cnt,
                    MAX(CASE WHEN outcome IN ('matched_rank', 'none_of_top_k') THEN 1 ELSE 0 END) AS has_review,
                    MAX(CASE WHEN outcome = 'skipped' THEN 1 ELSE 0 END) AS has_skip
                FROM feedback
                GROUP BY query_id, xml_id
                """
            )
        ).fetchall()

        folded: dict[int, list[int]] = {}
        for row in rows:
            served = self._served_query_id(row["query_id"], row["xml_id"])
            if served is None:
                continue
            held = folded.setdefault(served, [0, 0, 0])
            held[0] += int(row["cnt"])
            held[1] = max(held[1], int(row["has_review"]))
            held[2] = max(held[2], int(row["has_skip"]))

        statuses: dict[int, dict] = {}
        for query_id, (count, has_review, has_skip) in folded.items():
            if has_review:
                status = "reviewed"
            elif has_skip:
                status = "skipped"
            else:
                status = "unreviewed"
            statuses[query_id] = {
                "review_status": status,
                "review_count": count,
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
                        MAX(CASE WHEN outcome IN ('matched_rank', 'none_of_top_k') THEN 1 ELSE 0 END) AS has_review,
                        MAX(CASE WHEN outcome = 'skipped' THEN 1 ELSE 0 END) AS has_skip
                    FROM feedback
                    -- One group per witness; a row without one by its id.
                    GROUP BY COALESCE(xml_id, '#' || query_id)
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
                SELECT query_id, xml_id, timestamp, model_slug, outcome, reviewer, correct_rank
                FROM feedback
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        ).fetchall()
        return [
            {
                "file_id": self._shown_query_id(r["query_id"], r["xml_id"]),
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
                SELECT query_id, xml_id, timestamp, model_slug, outcome, notes, reviewer
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
                "file_id": self._shown_query_id(r["query_id"], r["xml_id"]),
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
        match, params = self._match_query(
            query_id, id_column="feedback.query_id", xml_column="feedback.xml_id"
        )
        query = _FEEDBACK_SELECT + f" WHERE {match}"
        if model:
            query += " AND feedback.model_slug = ?"
            params.append(model)
        if variant:
            query += " AND feedback.variant = ?"
            params.append(variant)
        query += " ORDER BY feedback.timestamp DESC, feedback.id DESC LIMIT ?"
        params.append(limit)
        rows = await (await self._db.execute(query, params)).fetchall()
        return [_feedback_row(row) for row in rows]

    async def get_latest_feedback(
        self,
        query_id: int,
        model_slug: str,
        variant: str = DEFAULT_VARIANT,
        reviewer_account_id: int | None = None,
        require_note: bool = False,
    ) -> dict | None:
        """Latest row on (query, model, variant), team-wide or for one reviewer.

        Both scopes exist because issue #96 split them (see the router): the
        NOTE a reviewer sees is the newest one from ANY reviewer, so two people
        working the same query read each other's reasoning instead of silently
        duplicating it, while the DECISION stays each reviewer's own. Pass
        `reviewer_account_id` for the second; `require_note` for the first.

        `require_note` skips rows whose notes are blank. The notes box exists to
        surface notes, so an answer saved without prose must not displace a
        colleague's substantive note -- "latest" there means latest row that
        actually says something. The decision lookup never sets it: an answer
        counts whether or not the reviewer explained it.

        Still keyed by variant either way, so a note saved while reviewing
        sif_abtt never prefills the form for raw -- different rankings,
        different answers.

        `idx_feedback_latest_variant` covers the scoped call outright. The
        team-wide call uses only its first three columns, leaving the ORDER BY
        to a sort: a handful of rows per query/model/variant in the pilot, so
        not worth a second index.
        """
        assert self._db is not None
        match, params = self._match_query(
            query_id, id_column="feedback.query_id", xml_column="feedback.xml_id"
        )
        query = (
            _FEEDBACK_SELECT
            + f"""
            WHERE {match}
              AND feedback.model_slug = ?
              AND feedback.variant = ?
            """
        )
        params += [model_slug, variant]
        if reviewer_account_id is not None:
            query += " AND feedback.reviewer_account_id = ?"
            params.append(reviewer_account_id)
        if require_note:
            # COALESCE for safety: the column is NOT NULL today, but pre-schema
            # rows have surprised this table before.
            query += " AND TRIM(COALESCE(feedback.notes, '')) != ''"
        query += " ORDER BY feedback.timestamp DESC, feedback.id DESC LIMIT 1"
        row = await (await self._db.execute(query, params)).fetchone()
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
            # The stored witness first: it names the text the row was saved
            # against whatever the served corpus does (issue #273).
            row["filename"] = (
                f"{row['xml_id']}.txt"
                if row.get("xml_id")
                else filename_lookup.get(row["query_id"], f"unknown-{row['query_id']}")
            )
            writer.writerow({column: row.get(column) for column in _EXPORT_COLUMNS})
        return output.getvalue()

    async def ccl_key_assignments(
        self,
        *,
        filename_by_query: dict[int, str],
        labelled_dir_names: Iterable[str],
        rank_dir: RankDirResolver | None = None,
        data_release: str = "",
        exported_at: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """One resolved row per reviewed query, for CCL's import (issue #270).

        The rule is stated in `services/ccl_key_export.py`; this method only
        reads what it needs. `rank_dir(query_id, model_slug, variant, rank)`
        names the directory at a rank in the ranking the app serves. It is
        consulted only for a row that cannot be read from what it stored: a
        row saved since issue #273 stores every picked directory
        (`selected_dirs_json`) and never needs it. On an older row
        `correct_dir` holds only the FIRST selection of a multi-rank pick, so
        later selections come from the served ranking, and only once it is
        shown to still hold the stored first pick at its rank
        (`ccl_key_export.resolve_picks`). Without a resolver those picks stay
        unresolved and the row is `needs_check`.

        Reads only. Reviewer names reach the returned rows and nothing else:
        the one log line here carries a count.
        """
        assert self._db is not None
        # Every row, unfiltered: which outcomes count is the export module's
        # rule (`current_decisions`), stated and tested in one place.
        rows = await (
            await self._db.execute("SELECT * FROM feedback ORDER BY timestamp, id")
        ).fetchall()
        dir_rows = await (
            await self._db.execute("SELECT dir_id, ccl_key FROM reviewer_dirs")
        ).fetchall()
        reviewer_dir_keys = {r["dir_id"]: r["ccl_key"] or "" for r in dir_rows}

        # Each row is about the witness it stored (issue #273). Only a row
        # without one is named through the served corpus, and a row neither
        # names is left out rather than given an invented witness.
        query_of_witness = {
            ccl_key_export.xml_id_of(filename): query_id
            for query_id, filename in filename_by_query.items()
        }
        named: list[dict] = []
        unknown_ids: set[int] = set()
        for r in rows:
            row = _feedback_row(r)
            if not row.get("xml_id"):
                filename = filename_by_query.get(int(row["query_id"]))
                if not filename:
                    if row.get("outcome") in ccl_key_export.DECISION_OUTCOMES:
                        unknown_ids.add(int(row["query_id"]))
                    continue
                row["xml_id"] = ccl_key_export.xml_id_of(filename)
            named.append(row)

        decisions = []
        for row in ccl_key_export.current_decisions(named):
            picked: list[tuple[int, str]] = []
            unresolved: list[int | None] = []
            if row["outcome"] == "matched_rank":
                served: dict[int, str | None] | None = None
                # Only an older row that did not store its picks reaches the
                # served ranking, and it asks under the id the corpus serves
                # its witness at now.
                served_id = query_of_witness.get(row["xml_id"])
                if (
                    ccl_key_export.needs_ranking(row)
                    and row.get("variant")
                    and rank_dir is not None
                    and served_id is not None
                ):
                    served = {}
                    for rank in ccl_key_export.selected_ranks(row):
                        served[rank] = await rank_dir(
                            served_id, row["model_slug"], row["variant"], rank
                        )
                picked, unresolved = ccl_key_export.resolve_picks(row, served)
            decisions.append(
                ccl_key_export.decision_from_row(
                    row, picked, reviewer_dir_keys, unresolved
                )
            )

        unknown = len(unknown_ids)
        resolved = ccl_key_export.build_rows(
            decisions,
            labelled_dir_names=labelled_dir_names,
            data_release=data_release,
            exported_at=exported_at or ccl_key_export.now_iso_utc(),
            status=status,
        )
        if unknown:
            logger.warning(
                "CCL key export left out %d reviewed query id(s) the served corpus "
                "does not know",
                unknown,
            )
        return resolved


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


async def _reviewer_dir_for_key(
    connection: aiosqlite.Connection, key: str
) -> str | None:
    """The oldest reviewer directory this key reaches, by key OR by label.

    Matching the LABEL as well as the key is what keeps a pre-#196 directory
    reachable at all (review finding 2). Those rows were named through the
    retired form -- often `New directory from BN2123.89r.5` -- so the migration
    leaves their `ccl_key` empty, and with the rank route gone a key-only lookup
    would make them permanently unjoinable while the unranked card kept showing
    their label to the evaluator. Typing that label verbatim then created a
    SECOND directory for the same grouping, which nothing can merge away.

    So the label is a second handle on the same directory, compared the same
    way (`ccl_keys.match_form`: case-folded, whitespace-collapsed). It can only
    ever join an existing directory; it never decides a name, and the key column
    remains the only thing a new directory is created under.

    Read in insertion order and scanned in Python rather than in SQL: `lower()`
    in SQLite is ASCII-only, the table is reviewer-authored and small, and the
    oldest match is the deterministic answer when history holds duplicates.
    """
    target = ccl_keys.match_form(key)
    if not target:
        return None
    rows = await (
        await connection.execute(
            "SELECT dir_id, ccl_key, label FROM reviewer_dirs ORDER BY id"
        )
    ).fetchall()
    for row in rows:
        if ccl_keys.match_form(row["ccl_key"]) == target:
            return row["dir_id"]
    for row in rows:
        if ccl_keys.match_form(row["label"]) == target:
            return row["dir_id"]
    return None


def _is_identical_none_repeat(
    previous: aiosqlite.Row | None, key: str, notes: str
) -> bool:
    """Would this submission say exactly what the caller's own last row says?

    Compared on the assertion, not on the timestamp: the outcome, the key (by
    `match_form`, so re-typing `ctou.567.16` for `CTOU.567.16` is still the same
    assertion) and the note. Anything else -- a revised note, a different key, a
    first submission, another reviewer's row -- is a new assertion and is
    appended, because the log is append-only and two reviewers agreeing is two
    facts.
    """
    if previous is None:
        return False
    if previous["outcome"] != "none_of_top_k":
        return False
    stored_key = ""
    if "ccl_key" in previous.keys():
        stored_key = previous["ccl_key"] or ""
    return (
        ccl_keys.match_form(stored_key) == ccl_keys.match_form(key)
        and (previous["notes"] or "") == notes
    )


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
    # The directory behind each selected rank, saved with the row (issue #273).
    # None on older rows and on anything that does not parse as a list of names.
    selected_dirs_json = values.get("selected_dirs_json")
    values["selected_dirs"] = None
    if selected_dirs_json:
        try:
            selected_dirs = json.loads(selected_dirs_json)
            if isinstance(selected_dirs, list) and all(
                isinstance(name, str) for name in selected_dirs
            ):
                values["selected_dirs"] = selected_dirs
        except (TypeError, ValueError, json.JSONDecodeError):
            values["selected_dirs"] = None
    return values


def _backup_before_xml_id_migration(db_path: Path) -> Path | None:
    """Snapshot a populated pre-#273 database before its first xml_id write.

    Runs before the connection that migrates is opened, and only when the
    `feedback` table exists, has rows and has no `xml_id` column yet, so it
    fires exactly once per database. The copy is written beside it as
    `<name>.pre-xml-id-<UTC stamp>` with SQLite's online backup API (a file
    copy can miss committed state still in the journal), mode 0600 (it holds
    every password hash) and O_EXCL (never overwrites an earlier one), then
    integrity-checked. A failed backup is removed and startup stops, the same
    rule as `scripts/webapp/set_account_role.py`.
    """
    if not db_path.exists():
        return None
    probe = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        columns = {
            row[1] for row in probe.execute("PRAGMA table_info(feedback)").fetchall()
        }
        if not columns or "xml_id" in columns:
            return None
        if probe.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0:
            return None
    finally:
        probe.close()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = db_path.with_name(f"{db_path.name}.pre-xml-id-{stamp}")
    suffix = 1
    while True:
        try:
            os.close(os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            break
        except FileExistsError:
            suffix += 1
            dest = db_path.with_name(f"{db_path.name}.pre-xml-id-{stamp}-{suffix}")
    try:
        source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        try:
            snapshot = sqlite3.connect(str(dest), timeout=30)
            try:
                source.backup(snapshot)
                status = snapshot.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                snapshot.close()
        finally:
            source.close()
        if status != "ok":
            raise RuntimeError(f"pre-migration backup failed integrity_check: {status}")
    except BaseException:
        try:
            dest.unlink()
        except OSError:
            pass
        raise
    logger.info(
        "Backed up %s to %s before adding witness ids", db_path.name, dest.name
    )
    return dest


def _public_account(row: aiosqlite.Row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
        "approval_status": row["approval_status"],
        # Read straight from the row rather than through a tolerant getter: this
        # is a security flag, and _open_connection() asserts the column exists
        # after _migrate(), so a missing column must surface as an error instead
        # of silently defaulting to "no forced change".
        "must_change_password": bool(row["must_change_password"]),
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
