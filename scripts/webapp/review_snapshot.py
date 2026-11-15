#!/usr/bin/env python3
"""Export an encrypted review snapshot, or decrypt and verify one locally.

Export uses the production venv and the API's CCL resolver. Decrypt needs only
Python 3's standard library and OpenSSL 1.1.1+ (including OpenSSL 3 on macOS).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from web.config import Settings

SCHEMA_VERSION = 1
ITERATIONS = 200_000
MAGIC = b"LLRS001\n"
CIPHERTEXT = "review-snapshot.tar.gz.enc"
WRAPPED_KEY = "review-snapshot.key.enc"
FILES = ("reviews.csv", "ccl_key_assignments.csv", "reviewer_dirs.csv",
         "reviewer_dir_members.csv")
REVIEW_COLUMNS = (
    "id", "xml_id", "query_id", "model_slug", "variant", "outcome",
    "correct_rank", "correct_dir", "selected_ranks_json", "selected_dirs_json",
    "ccl_key", "ccl_key_action", "ccl_key_dir", "ccl_key_rank", "timestamp",
    "schema_version",
)
DIR_COLUMNS = (
    "id", "dir_id", "seed_query_id", "seed_xml_id", "model_slug", "variant",
    "ccl_key", "created_at",
)
MEMBER_COLUMNS = ("id", "dir_id", "query_id", "xml_id", "role", "added_at")
TEST_TOKEN = re.compile(r"(^|[^a-z0-9])(smoke|test|testing)([^a-z0-9]|$)", re.IGNORECASE)

# Computed afresh for each SELECT, never stored as a table or Python lookup.
# Account identity takes precedence over the historical display name, exactly
# as in ccl_key_export.reviewer_identity. Directory-only authors follow reviews.
PSEUDONYMS = """
WITH actors(source, row_id, account_id, name) AS (
    SELECT 0, id, reviewer_account_id, reviewer FROM feedback
    UNION ALL
    SELECT 1, id, created_by_account_id, created_by FROM reviewer_dirs
    UNION ALL
    SELECT 2, id, added_by_account_id, added_by FROM reviewer_dir_members
), first_events AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY account_id, CASE WHEN account_id IS NULL THEN name END
        ORDER BY source, row_id
    ) AS occurrence FROM actors
), pseudonyms AS (
    SELECT account_id, name,
           printf('R%02d', ROW_NUMBER() OVER (ORDER BY source, row_id)) AS reviewer
    FROM first_events WHERE occurrence = 1
)
"""


class SnapshotError(Exception):
    """An operator-facing error containing no database values."""


def openssl(*args: str) -> bytes:
    command = [os.environ.get("OPENSSL", "openssl"), *args]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        # Neither stderr nor the command is safe to echo in a public log.
        raise SnapshotError("OpenSSL operation failed; check the key and OpenSSL version.")
    return result.stdout


def check_openssl() -> None:
    result = subprocess.run(
        [os.environ.get("OPENSSL", "openssl"), "enc", "-help"],
        capture_output=True, check=False,
    )
    if b"-pbkdf2" not in result.stdout + result.stderr:
        raise SnapshotError("PBKDF2 requires OpenSSL 1.1.1+; on macOS use OpenSSL 3.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_sha(root: Path) -> str:
    value = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise SnapshotError("Invalid repository git SHA.")
    return value


def backup_database(source: Path, destination: Path) -> None:
    os.close(os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    src = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        dest = sqlite3.connect(destination)
        try:
            src.backup(dest)
            if dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise SnapshotError("Database backup failed its integrity check.")
        finally:
            dest.close()
    finally:
        src.close()


def safe_tables(snapshot: Path) -> dict[str, list[dict]]:
    smoke_username = os.environ.get("LOCALLATIN_SMOKE_USERNAME", "").casefold()

    def is_test(*values: str | None) -> int:
        return int(any(
            value and (
                (smoke_username and value.casefold() == smoke_username)
                or TEST_TOKEN.search(value)
            ) for value in values
        ))

    conn = sqlite3.connect(snapshot.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.create_function("is_smoke_or_test", 3, is_test)
    try:
        result = {}
        for table, columns, name, account in (
            ("feedback", REVIEW_COLUMNS, "reviewer", "reviewer_account_id"),
            ("reviewer_dirs", DIR_COLUMNS, "created_by", "created_by_account_id"),
            ("reviewer_dir_members", MEMBER_COLUMNS, "added_by", "added_by_account_id"),
        ):
            selected = ", ".join(f"t.{column}" for column in columns)
            result[table] = [dict(row) for row in conn.execute(
                PSEUDONYMS + f"""
                SELECT {selected}, p.reviewer,
                       COALESCE(a.role, 'unknown') AS reviewer_role,
                       is_smoke_or_test(a.username, a.display_name, t.{name})
                           AS is_smoke_or_test
                FROM {table} AS t
                JOIN pseudonyms AS p ON (
                    (t.{account} IS NOT NULL AND p.account_id = t.{account})
                    OR (t.{account} IS NULL AND p.account_id IS NULL
                        AND p.name = t.{name})
                )
                LEFT JOIN accounts AS a ON a.id = t.{account}
                ORDER BY t.id
                """
            )]
        return result
    finally:
        conn.close()


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


async def build_bundle(snapshot: Path, bundle: Path, settings: Settings, git_sha: str) -> dict:
    from web.services import ccl_key_export
    from web.services.data_store import build_store
    from web.services.feedback_db import FeedbackDB, corpus_digest

    tables = safe_tables(snapshot)
    reviews = tables["feedback"]
    # Feed only pseudonyms into the API code. This database has no accounts,
    # notes, sessions or identity map, and is never part of the archive.
    redacted = snapshot.with_name("pseudonymised.db")
    with sqlite3.connect(redacted) as conn:
        for table, columns in (
            ("feedback", REVIEW_COLUMNS),
            ("reviewer_dirs", DIR_COLUMNS),
            ("reviewer_dir_members", MEMBER_COLUMNS),
        ):
            fields = (*columns, "reviewer", "reviewer_role", "is_smoke_or_test")
            conn.execute(f"CREATE TABLE {table} ({', '.join(fields)})")
            conn.executemany(
                f"INSERT INTO {table} VALUES ({', '.join('?' for _ in fields)})",
                [tuple(row[field] for field in fields) for row in tables[table]],
            )
        conn.execute("ALTER TABLE feedback ADD COLUMN reviewer_account_id INTEGER")

    store = build_store(settings)

    async def rank_dir(query_id: int, slug: str, variant: str | None, rank: int) -> str | None:
        if not variant or not await store.ensure_variant_async(variant):
            return None
        rows = store.predictions.get((slug, variant), [])
        row = next((row for row in rows if row["file_id"] == query_id), None)
        if row is None:
            return None
        match = next((p for p in row["predictions"] if p["rank"] == rank), None)
        return match["dir_name"] if match else None

    db = FeedbackDB(redacted)
    await db.connect(read_only=True)
    created_at = ccl_key_export.now_iso_utc()
    try:
        assignments = await db.ccl_key_assignments(
            filename_by_query=store.file_id_to_filename,
            labelled_dir_names=store.labelled_dir_files,
            rank_dir=rank_dir,
            data_release=store.data_release,
            exported_at=created_at,
        )
    finally:
        await db.close()

    bundle.mkdir(mode=0o700)
    extras = ("reviewer", "reviewer_role", "is_smoke_or_test")
    write_csv(bundle / FILES[0], (*REVIEW_COLUMNS, *extras), reviews)
    (bundle / FILES[1]).write_text(ccl_key_export.to_csv(assignments), encoding="utf-8")
    write_csv(bundle / FILES[2], (*DIR_COLUMNS, *extras), tables["reviewer_dirs"])
    write_csv(bundle / FILES[3], (*MEMBER_COLUMNS, *extras), tables["reviewer_dir_members"])
    reviewed = [r for r in reviews if r["outcome"] in ccl_key_export.DECISION_OUTCOMES]
    reviewed_queries = set()
    for row in reviewed:
        filename = store.file_id_to_filename.get(row["query_id"])
        witness = row["xml_id"] or (ccl_key_export.xml_id_of(filename) if filename else None)
        reviewed_queries.add(("xml_id", witness) if witness else ("query_id", row["query_id"]))
    manifest = {
        "snapshot_schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "git_sha": git_sha,
        "exporter_git_sha": repository_sha(Path(__file__).resolve().parents[2]),
        "data_release": store.data_release,
        "served_corpus_sha256": corpus_digest(store.file_id_to_filename),
        "row_counts": {filename: len(rows) for filename, rows in zip(FILES, (
            reviews, assignments, tables["reviewer_dirs"], tables["reviewer_dir_members"],
        ))},
        "rows_per_outcome": dict(sorted(Counter(r["outcome"] for r in reviews).items())),
        "distinct_queries_reviewed": len(reviewed_queries),
        "reviewers": len({r["reviewer"] for r in reviews}),
        "is_smoke_or_test_rows": sum(r["is_smoke_or_test"] for r in reviews),
        "files": {name: sha256(bundle / name) for name in FILES},
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def mac_for(path: Path, secret: bytes) -> bytes:
    key = hashlib.sha256(b"localLatin review snapshot HMAC v1\0" + secret).digest()
    mac = hmac.new(key, MAGIC, hashlib.sha256)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            mac.update(chunk)
    return mac.digest()


def encrypt_bundle(bundle: Path, public_key: Path, output: Path, temp: Path) -> None:
    check_openssl()
    key_text = openssl("rsa", "-pubin", "-in", str(public_key), "-text", "-noout")
    if not re.search(rb"Public-Key: \(4096 bit\)", key_text):
        raise SnapshotError("The snapshot public key must be RSA-4096.")
    secret = secrets.token_bytes(32)
    password = temp / "password"
    password.write_text(secret.hex() + "\n", encoding="ascii")
    archive = temp / "snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in (*FILES, "manifest.json"):
            info = tar.gettarinfo(str(bundle / name), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o600
            with (bundle / name).open("rb") as stream:
                tar.addfile(info, stream)
    encrypted = temp / "ciphertext"
    openssl(
        "enc", "-aes-256-cbc", "-pbkdf2", "-iter", str(ITERATIONS), "-md", "sha256",
        "-salt", "-in", str(archive), "-out", str(encrypted), "-pass", f"file:{password}",
    )
    wrapped = temp / WRAPPED_KEY
    openssl(
        "pkeyutl", "-encrypt", "-pubin", "-inkey", str(public_key),
        "-pkeyopt", "rsa_padding_mode:oaep", "-pkeyopt", "rsa_oaep_md:sha256",
        "-pkeyopt", "rsa_mgf1_md:sha256", "-in", str(password), "-out", str(wrapped),
    )
    envelope = temp / CIPHERTEXT
    with envelope.open("wb") as stream, encrypted.open("rb") as body:
        stream.write(MAGIC + mac_for(encrypted, secret))
        shutil.copyfileobj(body, stream)
    output.mkdir(mode=0o700)
    shutil.copyfile(envelope, output / CIPHERTEXT)
    shutil.copyfile(wrapped, output / WRAPPED_KEY)


def verify_bundle(bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("snapshot_schema_version") != SCHEMA_VERSION:
        raise SnapshotError("Unsupported snapshot schema version.")
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or set(hashes) != set(FILES):
        raise SnapshotError("Manifest file inventory is invalid.")
    if {p.name for p in bundle.iterdir()} != {*FILES, "manifest.json"}:
        raise SnapshotError("Bundle file inventory is invalid.")
    for name in FILES:
        if not hmac.compare_digest(sha256(bundle / name), hashes[name]):
            raise SnapshotError("Manifest SHA256 verification failed.")


def decrypt_snapshot(artifact: Path, private_key: Path, output: Path) -> None:
    check_openssl()
    if output.exists():
        raise SnapshotError("Output directory must not already exist.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination, then publish only a completely verified bundle.
    with tempfile.TemporaryDirectory(prefix=".review-decrypt-", dir=output.parent) as work:
        temp = Path(work)
        password = temp / "password"
        openssl(
            "pkeyutl", "-decrypt", "-inkey", str(private_key),
            "-pkeyopt", "rsa_padding_mode:oaep", "-pkeyopt", "rsa_oaep_md:sha256",
            "-pkeyopt", "rsa_mgf1_md:sha256",
            "-in", str(artifact / WRAPPED_KEY), "-out", str(password),
        )
        password_text = password.read_text(encoding="ascii")
        if not re.fullmatch(r"[0-9a-f]{64}\n", password_text):
            raise SnapshotError("Invalid snapshot key.")
        encrypted = temp / "ciphertext"
        with (artifact / CIPHERTEXT).open("rb") as stream, encrypted.open("wb") as body:
            if stream.read(len(MAGIC)) != MAGIC:
                raise SnapshotError("Unsupported encrypted snapshot format.")
            expected_mac = stream.read(32)
            shutil.copyfileobj(stream, body)
        if not hmac.compare_digest(mac_for(encrypted, bytes.fromhex(password_text)), expected_mac):
            raise SnapshotError("Ciphertext authentication failed.")
        archive = temp / "snapshot.tar.gz"
        openssl(
            "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", str(ITERATIONS),
            "-md", "sha256", "-in", str(encrypted), "-out", str(archive),
            "-pass", f"file:{password}",
        )
        bundle = temp / "bundle"
        bundle.mkdir(mode=0o700)
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            wanted = {*FILES, "manifest.json"}
            if (
                len(members) != len(wanted)
                or {member.name for member in members} != wanted
                or any(not member.isfile() for member in members)
            ):
                raise SnapshotError("Unsafe or unexpected archive members.")
            for member in members:
                source = tar.extractfile(member)
                if source is None:
                    raise SnapshotError("Unreadable archive member.")
                with source, (bundle / member.name).open("wb") as dest:
                    shutil.copyfileobj(source, dest)
        verify_bundle(bundle)
        bundle.rename(output)
    print(f"Verified files: {len(FILES)}")


def export_snapshot(deploy: Path, output: Path) -> None:
    from web.config import load_settings

    if os.environ.get("ENABLE_PRODUCTION_DEPLOY") != "true":
        raise SnapshotError("Production export is disabled.")
    config = deploy / "web/config.production.yaml"
    if not config.is_file():
        raise SnapshotError("Production configuration is missing.")
    settings = load_settings(config)
    if (
        Path(settings.paths.data_root).resolve() != deploy.resolve()
        or settings.paths.feedback_db != "data/feedback.db"
    ):
        raise SnapshotError("Production database configuration changed; update the workflow.")
    if output.exists():
        raise SnapshotError("Artifact directory must not already exist.")
    public_pem = os.environ.get("REVIEW_SNAPSHOT_PUBKEY_PEM", "")
    if not public_pem:
        raise SnapshotError("REVIEW_SNAPSHOT_PUBKEY_PEM is required.")
    git_sha = repository_sha(deploy)
    with tempfile.TemporaryDirectory(prefix="review-snapshot-") as work:
        temp = Path(work)
        public_key = temp / "public.pem"
        public_key.write_text(public_pem, encoding="ascii")
        snapshot = temp / "feedback.db"
        backup_database(deploy / "data/feedback.db", snapshot)
        manifest = asyncio.run(build_bundle(snapshot, temp / "bundle", settings, git_sha))
        encrypt_bundle(temp / "bundle", public_key, output, temp)
    print(f"Feedback rows: {manifest['row_counts'][FILES[0]]}")
    print(f"Reviewed queries: {manifest['distinct_queries_reviewed']}")
    print(f"Reviewers: {manifest['reviewers']}")
    print(f"Smoke/test rows: {manifest['is_smoke_or_test_rows']}")


def main() -> int:
    os.umask(0o077)
    # Third-party loaders can log paths and corpus values. The public workflow
    # deliberately emits only the counts above, and static failure messages.
    logging.disable(logging.CRITICAL)
    # Even an unexpected loader failure must not put database values in a
    # traceback in public Actions logs. The process still exits nonzero.
    sys.excepthook = lambda *_: print(
        "error: Unexpected snapshot failure; details withheld from public logs.", file=sys.stderr
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--deploy-path", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    decrypt = commands.add_parser("decrypt")
    decrypt.add_argument("artifact_dir", type=Path)
    decrypt.add_argument("private_key", type=Path)
    decrypt.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "export":
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            export_snapshot(args.deploy_path, args.output)
        else:
            decrypt_snapshot(args.artifact_dir, args.private_key, args.output_dir)
    except SnapshotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError, tarfile.TarError):
        print("error: Snapshot operation failed; no verified bundle was published.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
