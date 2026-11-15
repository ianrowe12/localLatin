"""Synthetic, local-only privacy, API parity and encryption checks for #285."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sqlite3
import subprocess
import sys
import threading
import warnings
from contextlib import closing
from pathlib import Path

import pytest

from scripts.webapp import review_snapshot as snapshot
from web.config import load_settings
from web.services import ccl_key_export
from web.services.data_store import build_store
from web.services.feedback_db import FeedbackDB, corpus_digest
from web.tests.test_reviewer_dirs import _write_fixture_data

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_STRINGS = (
    "Alice Scholar", "Alice Renamed", "alice.private@example.invalid",
    "Bruno Scholar", "bruno.private@example.invalid", "Former Scholar",
    "Directory Only Scholar", "Automated Runner", "custom-robot-login",
    "PRIVATE reviewer note", "PRIVATE directory label", "PRIVATE password hash",
    "PRIVATE session token", "731041", "731042", "731043", "731044",
)


@pytest.fixture
def production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "production"
    config = _write_fixture_data(root, with_matrix=False)
    text = config.read_text().replace(
        'feedback_db: "runs/active/resubmit/webapp/feedback.db"',
        'feedback_db: "data/feedback.db"',
    )
    (root / "web").mkdir()
    config = root / "web/config.production.yaml"
    config.write_text(text)
    settings = load_settings(config)
    db_path = settings.paths.resolve(settings.paths.feedback_db)

    async def create() -> None:
        db = FeedbackDB(db_path)
        await db.connect()
        await db.close()

    asyncio.run(create())
    with sqlite3.connect(db_path) as db:
        for account, username, name, role in (
            (731041, "alice.private@example.invalid", "Alice Scholar", "reviewer"),
            (731042, "custom-robot-login", "Automated Runner", "pi_admin"),
            (731043, "bruno.private@example.invalid", "Bruno Scholar", "reviewer"),
            (731044, "directory-only", "Directory Only Scholar", "reviewer"),
        ):
            db.execute(
                "INSERT INTO accounts (id, username, display_name, password_hash, role) "
                "VALUES (?, ?, ?, 'PRIVATE password hash', ?)",
                (account, username, name, role),
            )
        db.execute(
            "INSERT INTO account_sessions (account_id, token_hash, expires_at) "
            "VALUES (731041, 'PRIVATE session token', '2099-01-01')"
        )
        rows = [
            (0, None, "matched_rank", 1, "candidate-a", "[1]", '["candidate-a"]',
             None, "Alice Scholar", 731041, "sif_abtt"),
            (0, "query-0", "none_of_top_k", 0, None, None, None,
             "candidate-b", "Automated Runner", 731042, "sif_abtt"),
            (0, "query-0", "matched_rank", 1, "candidate-a", "[1]", '["candidate-a"]',
             None, "Alice Renamed", 731041, "sif_abtt"),
            (1, "query-1", "none_of_top_k", 0, None, None, None,
             "CTOU.567.16", "Former Scholar", None, "sif_abtt"),
            # Legacy multi-rank: the API must resolve the second pick from
            # the served ranking, not silently mark it unresolved.
            (2, "query-2", "matched_rank", 1, "candidate-a", "[1, 2]", None,
             None, "Alice Renamed", 731041, "sif_abtt"),
            (2, "query-2", "skipped", None, None, None, None,
             None, "Former Scholar", None, "sif_abtt"),
            # Stored ids and disappeared witnesses must survive unchanged.
            (88, "retired-witness", "matched_rank", 1, "candidate-a", "[1]",
             '["candidate-a"]', None, "Alice Renamed", 731041, "sif_abtt"),
            (1, "query-1", "none_of_top_k", 0, None, None, None,
             None, "Alice Scholar", 731041, "raw"),
            (3, "query-3", "legacy_unresolved", None, None, None, None,
             None, "Bruno Scholar", 731043, None),
        ]
        for index, row in enumerate(rows):
            db.execute(
                """INSERT INTO feedback
                   (query_id, xml_id, outcome, correct_rank, correct_dir,
                    selected_ranks_json, selected_dirs_json, ccl_key, reviewer,
                    reviewer_account_id, variant, model_slug, notes, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'bowphs_LaTa',
                           'PRIVATE reviewer note', ?)""",
                # Deliberately nonchronological, so insertion order is tested.
                (*row, f"2026-11-{20 - index:02d} 10:00:00"),
            )
        db.execute(
            """INSERT INTO reviewer_dirs
               (dir_id, label, ccl_key, seed_query_id, seed_xml_id,
                created_by, created_by_account_id)
               VALUES ('reviewer-dir-example', 'PRIVATE directory label',
                       'CTOU.567.16', 1, 'query-1', 'Directory Only Scholar', 731044)"""
        )
        db.executemany(
            """INSERT INTO reviewer_dir_members
               (dir_id, query_id, xml_id, role, added_by, added_by_account_id)
               VALUES ('reviewer-dir-example', ?, ?, ?, ?, ?)""",
            [
                (1, "query-1", "seed", "Directory Only Scholar", 731044),
                (2, "query-2", "member", "Alice Scholar", 731041),
            ],
        )
    (root / ".deploy-cache").mkdir()
    (root / ".deploy-cache/installed.state").write_text("data-20261115 abc123\n")
    monkeypatch.setenv("LOCALLATIN_SMOKE_USERNAME", "custom-robot-login")
    return root, db_path, settings


def build(production, tmp_path: Path):
    _, live, settings = production
    work = tmp_path / "work"
    work.mkdir()
    copied = work / "feedback.db"
    snapshot.backup_database(live, copied)
    bundle = work / "bundle"
    manifest = asyncio.run(snapshot.build_bundle(copied, bundle, settings, "a" * 40))
    return work, bundle, manifest


def csv_rows(bundle: Path, filename: str) -> list[dict[str, str]]:
    with (bundle / filename).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_bundle_redaction_pseudonyms_and_manifest(production, tmp_path: Path) -> None:
    work, bundle, manifest = build(production, tmp_path)
    for private in PRIVATE_STRINGS:
        grep = subprocess.run(
            ["grep", "-r", "-F", "--", private, str(bundle)], capture_output=True, check=False
        )
        assert grep.returncode == 1, private
    assert {p.name for p in bundle.iterdir()} == {*snapshot.FILES, "manifest.json"}
    reviews = csv_rows(bundle, "reviews.csv")
    assert [int(r["id"]) for r in reviews] == list(range(1, 10))
    assert [r["reviewer"] for r in reviews] == [
        "R01", "R02", "R01", "R03", "R01", "R03", "R01", "R01", "R04",
    ]
    assert reviews[1]["reviewer_role"] == "pi_admin"
    assert reviews[3]["reviewer_role"] == "unknown"
    assert [r["is_smoke_or_test"] for r in reviews] == ["0", "1", "0", "0", "0", "0", "0", "0", "0"]
    assert reviews[6]["query_id"] == "88"
    assert reviews[6]["xml_id"] == "retired-witness"
    assert reviews[0]["selected_dirs_json"] == '["candidate-a"]'
    assert csv_rows(bundle, "reviewer_dirs.csv")[0]["reviewer"] == "R05"
    assert [r["reviewer"] for r in csv_rows(bundle, "reviewer_dir_members.csv")] == ["R05", "R01"]
    store = build_store(production[2])
    assert manifest["served_corpus_sha256"] == corpus_digest(store.file_id_to_filename)
    assert manifest["data_release"] == "data-20261115"
    assert manifest["git_sha"] == "a" * 40
    assert manifest["created_at"].endswith("Z")
    assert manifest["rows_per_outcome"] == {
        "matched_rank": 4, "none_of_top_k": 3, "skipped": 1, "legacy_unresolved": 1,
    }
    assert manifest["distinct_queries_reviewed"] == 4
    assert manifest["reviewers"] == 4
    assert manifest["row_counts"] == {
        "reviews.csv": 9, "ccl_key_assignments.csv": 4,
        "reviewer_dirs.csv": 1, "reviewer_dir_members.csv": 2,
    }
    snapshot.verify_bundle(bundle)
    # The API's input copy cannot contain identities either.
    redacted_bytes = (work / "pseudonymised.db").read_bytes()
    assert all(value.encode() not in redacted_bytes for value in PRIVATE_STRINGS)


def test_ccl_table_is_the_api_export_with_pseudonyms(production, tmp_path: Path) -> None:
    _, bundle, manifest = build(production, tmp_path)
    store = build_store(production[2])

    async def api_export() -> list[dict]:
        db = FeedbackDB(production[1])
        await db.connect(read_only=True)

        async def ranking(query, slug, variant, rank):
            if not variant or not await store.ensure_variant_async(variant):
                return None
            row = next(
                (r for r in store.predictions.get((slug, variant), []) if r["file_id"] == query),
                None,
            )
            if row:
                return next((p["dir_name"] for p in row["predictions"] if p["rank"] == rank), None)
            return None

        try:
            return await db.ccl_key_assignments(
                filename_by_query=store.file_id_to_filename,
                labelled_dir_names=store.labelled_dir_files,
                rank_dir=ranking, data_release=store.data_release,
                exported_at=manifest["created_at"],
            )
        finally:
            await db.close()

    expected = asyncio.run(api_export())
    aliases = {
        "Alice Scholar": "R01", "Alice Renamed": "R01",
        "Automated Runner": "R02", "Former Scholar": "R03", "Bruno Scholar": "R04",
    }
    for row in expected:
        row["reviewers"] = ccl_key_export.LIST_SEPARATOR.join(sorted(
            aliases[name] for name in row["reviewers"].split(ccl_key_export.LIST_SEPARATOR)
        ))
    actual = csv_rows(bundle, "ccl_key_assignments.csv")
    assert actual == [{k: str(v) for k, v in row.items()} for row in expected]
    multi = next(row for row in actual if row["xml_id"] == "query-2")
    assert multi["status"] == "conflict"
    assert multi["conflicting_keys"] == "candidate-a | candidate-b"


def test_backup_includes_wal_without_changing_live_database(production, tmp_path: Path) -> None:
    _, live, _ = production
    writer = sqlite3.connect(live)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("UPDATE accounts SET display_name = 'Private WAL Name' WHERE id = 731041")
        writer.commit()
        before = live.read_bytes()
        copied = tmp_path / "wal-copy.db"
        snapshot.backup_database(live, copied)
        assert live.read_bytes() == before
        with sqlite3.connect(copied) as db:
            assert db.execute("SELECT display_name FROM accounts WHERE id=731041").fetchone()[0] == "Private WAL Name"
    finally:
        writer.close()


@pytest.fixture(scope="module")
def keypair(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("snapshot-keys")
    private, public = root / "private.pem", root / "public.pem"
    snapshot.openssl("genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:4096",
                     "-out", str(private))
    snapshot.openssl("pkey", "-in", str(private), "-pubout", "-out", str(public))
    return private, public


def test_export_cli_counts_only_cleanup_and_no_live_writes(
    production, tmp_path: Path, keypair
) -> None:
    root, live, _ = production
    private, public = keypair
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Fixture", "-c",
         "user.email=fixture@example.invalid", "commit", "-q", "--allow-empty", "-m", "test: fixture"],
        check=True,
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    artifact = tmp_path / "artifact"
    env = {
        **os.environ, "ENABLE_PRODUCTION_DEPLOY": "true", "TMPDIR": str(staging),
        "REVIEW_SNAPSHOT_PUBKEY_PEM": public.read_text(),
    }
    before = live.read_bytes()
    command = [
        sys.executable, str(ROOT / "scripts/webapp/review_snapshot.py"), "export",
        f"--deploy-path={root}", f"--output={artifact}",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "Feedback rows: 9", "Reviewed queries: 4", "Reviewers: 4", "Smoke/test rows: 1",
    ]
    assert result.stderr == ""
    assert live.read_bytes() == before
    assert list(staging.iterdir()) == []
    snapshot.decrypt_snapshot(artifact, private, tmp_path / "out")
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["git_sha"] == snapshot.repository_sha(root)
    assert manifest["exporter_git_sha"] == snapshot.repository_sha(ROOT)
    assert manifest["git_sha"] != manifest["exporter_git_sha"]
    # Invalid key failures must clean staging and never log the supplied text.
    env["REVIEW_SNAPSHOT_PUBKEY_PEM"] = "PRIVATE malformed key"
    command[-1] = f"--output={tmp_path / 'failed'}"
    failed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert failed.returncode != 0
    assert "PRIVATE" not in failed.stdout + failed.stderr
    assert list(staging.iterdir()) == []
    assert not (tmp_path / "failed").exists()
    env["ENABLE_PRODUCTION_DEPLOY"] = "false"
    refused = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert refused.returncode != 0
    assert (
        "stage=configure exception=SnapshotError: Production export is disabled."
        in refused.stderr
    )


def test_read_only_feedback_connection_does_not_create_or_migrate(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"

    async def run() -> None:
        db = FeedbackDB(missing)
        with pytest.raises(sqlite3.OperationalError):
            await db.connect(read_only=True)
        assert not missing.exists()
        with sqlite3.connect(missing) as conn:
            conn.execute("CREATE TABLE marker (id INTEGER)")
        before = missing.read_bytes()
        await db.connect(read_only=True)
        try:
            assert db._db is not None
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                await db._db.execute("INSERT INTO marker VALUES (1)")
        finally:
            await db.close()
        assert missing.read_bytes() == before

    asyncio.run(run())


@pytest.fixture
def export_cli(production, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(snapshot, "repository_sha", lambda root: "a" * 40)
    monkeypatch.setenv("ENABLE_PRODUCTION_DEPLOY", "true")
    monkeypatch.setenv("REVIEW_SNAPSHOT_PUBKEY_PEM", "unused public key")
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(snapshot.logging.root.manager, "disable", 0)
    artifact = tmp_path / "artifact"
    monkeypatch.setattr(sys, "argv", [
        "review_snapshot.py", "export", f"--deploy-path={production[0]}",
        f"--output={artifact}",
    ])
    return artifact


@pytest.mark.parametrize(("stage", "error"), [
    ("deploy_sha", subprocess.CalledProcessError(1, ["PRIVATE deployed path"])),
    ("backup", OSError("PRIVATE path")),
    ("pseudonymise", sqlite3.IntegrityError("UNIQUE constraint failed: feedback.id")),
    ("load_store", ValueError("PRIVATE loader value")),
    ("load_store", RuntimeError("PRIVATE corpus mismatch")),
    ("load_store", KeyError("PRIVATE loader key")),
    ("load_store", TypeError("PRIVATE loader type")),
    ("load_store", ImportError("PRIVATE loader import")),
    ("ccl_assignments", sqlite3.OperationalError("no such column: missing_column")),
    ("write_bundle", ValueError("PRIVATE row value")),
    ("exporter_sha", subprocess.CalledProcessError(1, ["PRIVATE runner path"])),
    ("encrypt", subprocess.CalledProcessError(1, ["PRIVATE command"])),
])
def test_export_failure_diagnostics_are_private(
    production, export_cli: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    stage: str, error: Exception,
) -> None:
    from web.services import data_store

    def fail(*args, **kwargs):
        warnings.warn("PRIVATE warning value", UserWarning)
        raise error

    async def fail_async(*args, **kwargs):
        fail()

    def fail_exporter_sha(root):
        if root == production[0]:
            return "a" * 40
        fail()

    targets = {
        "deploy_sha": (snapshot, "repository_sha", fail),
        "backup": (snapshot, "backup_database", fail),
        "pseudonymise": (snapshot, "safe_tables", fail),
        "load_store": (data_store, "build_store", fail),
        "ccl_assignments": (FeedbackDB, "ccl_key_assignments", fail_async),
        "write_bundle": (snapshot, "write_csv", fail),
        "exporter_sha": (snapshot, "repository_sha", fail_exporter_sha),
        "encrypt": (snapshot, "encrypt_bundle", fail),
    }
    monkeypatch.setattr(*targets[stage])
    with warnings.catch_warnings(record=True) as emitted:
        if isinstance(error, (KeyError, TypeError, ImportError)):
            with pytest.raises(type(error)) as raised:
                snapshot.main()
            assert raised.value is error
            sys.excepthook(type(error), error, raised.tb)
        else:
            assert snapshot.main() == 1
    captured = capsys.readouterr()
    assert "PRIVATE" not in captured.out + captured.err
    assert f"stage={stage}" in captured.err
    assert f"exception={type(error).__name__}" in captured.err
    assert not emitted
    assert "Traceback" not in captured.err
    assert not export_cli.exists()
    if str(error).startswith("no such column:"):
        assert str(error) in captured.err


@pytest.mark.parametrize(("message", "allowed"), [
    ("no such table: feedback", True),
    ("no such column: t.ccl_key", True),
    ("database is locked", True),
    ("disk I/O error", True),
    ("unable to open database file", True),
    ("database or disk is full", True),
    ("attempt to write a readonly database", True),
    ("database disk image is malformed", True),
    ("UNIQUE constraint failed: feedback.id", False),
    ("no such column: ccl_key PRIVATE", False),
    ("no such table: /PRIVATE/path", False),
    ("database is locked\nPRIVATE", False),
    ("PRIVATE unknown sqlite error", False),
])
def test_sqlite_diagnostic_allowlist(message: str, allowed: bool, capsys) -> None:
    snapshot.report_failure(snapshot.SnapshotProgress("backup"), sqlite3.OperationalError(message))
    expected = "error: stage=backup exception=OperationalError"
    assert capsys.readouterr().err == expected + (f": {message}" if allowed else "") + "\n"


def test_invalid_utf8_cell_is_not_logged(
    production, export_cli: Path, capsys,
) -> None:
    with closing(sqlite3.connect(production[1])) as db, db:
        db.execute(
            "UPDATE feedback SET ccl_key = CAST(? AS TEXT) WHERE id = 1",
            (b"PRIVATE typed key\xff",),
        )
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        with pytest.raises(sqlite3.OperationalError, match="PRIVATE typed key"):
            db.execute("SELECT ccl_key FROM feedback WHERE id = 1").fetchone()
    assert snapshot.main() == 1
    captured = capsys.readouterr()
    assert captured.err == "error: stage=pseudonymise exception=OperationalError\n"
    assert "PRIVATE" not in captured.out + captured.err
    assert not export_cli.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux /proc/self/fd")
@pytest.mark.parametrize("failure_stage", [None, "load_store", "ccl_assignments"])
def test_export_closes_temporary_files_before_cleanup(
    export_cli: Path, keypair, monkeypatch: pytest.MonkeyPatch, failure_stage: str | None,
) -> None:
    from web.services import data_store

    connect = sqlite3.connect
    connections = []
    checked = []

    def retain_connection(*args, **kwargs):
        connection = connect(*args, **kwargs)
        if threading.current_thread() is threading.main_thread():
            connections.append(connection)
        return connection

    class AuditedTemporaryDirectory(snapshot.tempfile.TemporaryDirectory):
        def cleanup(self):
            opened = []
            for fd in Path("/proc/self/fd").iterdir():
                try:
                    target = fd.readlink()
                except FileNotFoundError:
                    continue
                if target.is_relative_to(self.name):
                    opened.append(target)
            checked.append(self.name)
            try:
                assert not opened, f"Temporary files still open at cleanup: {opened}"
            finally:
                super().cleanup()

    def fail(*args, **kwargs):
        raise ValueError("PRIVATE loader value")

    async def fail_async(*args, **kwargs):
        fail()

    if failure_stage == "load_store":
        monkeypatch.setattr(data_store, "build_store", fail)
    elif failure_stage == "ccl_assignments":
        monkeypatch.setattr(FeedbackDB, "ccl_key_assignments", fail_async)
    monkeypatch.setenv("REVIEW_SNAPSHOT_PUBKEY_PEM", keypair[1].read_text())
    # Keep connections reachable so garbage collection cannot hide a missing close.
    monkeypatch.setattr(sqlite3, "connect", retain_connection)
    monkeypatch.setattr(snapshot.tempfile, "TemporaryDirectory", AuditedTemporaryDirectory)
    try:
        assert snapshot.main() == (1 if failure_stage else 0)
        assert len(checked) == 1
    finally:
        for connection in connections:
            connection.close()


@pytest.mark.parametrize("loader_failure", [False, True])
def test_cleanup_failure_is_fatal_and_reports_cleanup_stage(
    export_cli: Path, monkeypatch: pytest.MonkeyPatch, capsys, loader_failure: bool,
) -> None:
    from web.services import data_store

    class FailedCleanup(snapshot.tempfile.TemporaryDirectory):
        def cleanup(self):
            super().cleanup()
            raise OSError("PRIVATE cleanup path")

    def fail(*args, **kwargs):
        raise ValueError("PRIVATE loader value")

    monkeypatch.setattr(snapshot.tempfile, "TemporaryDirectory", FailedCleanup)
    monkeypatch.setattr(snapshot, "encrypt_bundle", lambda *args: None)
    if loader_failure:
        monkeypatch.setattr(data_store, "build_store", fail)
    assert snapshot.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: stage=cleanup exception=OSError\n"
