#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_ENV_FILE = Path("~/.config/locallatin/feedback-backup.env").expanduser()
DEFAULT_DEST = Path("~/Backups/locallatin-feedback").expanduser()
DEFAULT_REMOTE_DB = "/homes/ipro222/localLatin/data/feedback.db"

REMOTE_BACKUP_CODE = r"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile

db_path = sys.argv[1]
fd, snapshot_path = tempfile.mkstemp(prefix="locallatin-feedback-", suffix=".db")
os.close(fd)

try:
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    snapshot = sqlite3.connect(snapshot_path)
    source.backup(snapshot)
    snapshot.commit()
    snapshot.close()
    source.close()

    check_conn = sqlite3.connect(snapshot_path)
    integrity = check_conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"Snapshot failed integrity_check: {integrity}")

    feedback_digest = hashlib.sha256()
    columns = [
        "id",
        "query_id",
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
    query = f"SELECT {', '.join(columns)} FROM feedback ORDER BY id"
    count = 0
    for row in check_conn.execute(query):
        count += 1
        feedback_digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        )
        feedback_digest.update(b"\n")
    check_conn.close()

    db_digest = hashlib.sha256()
    with open(snapshot_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            db_digest.update(chunk)

    print(
        json.dumps(
            {
                "snapshot_path": snapshot_path,
                "db_sha256": db_digest.hexdigest(),
                "feedback_fingerprint": feedback_digest.hexdigest(),
                "feedback_rows": count,
                "size": os.path.getsize(snapshot_path),
            }
        )
    )
except Exception:
    try:
        os.unlink(snapshot_path)
    except OSError:
        pass
    raise
"""


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def env_default(values: dict[str, str], key: str, fallback: str) -> str:
    return os.environ.get(key) or values.get(key) or fallback


def build_ssh_base(args: argparse.Namespace) -> list[str]:
    target = args.host
    if args.user:
        target = f"{args.user}@{target}"
    command = ["ssh", "-o", "BatchMode=yes"]
    if args.port:
        command.extend(["-p", str(args.port)])
    if args.identity_file:
        command.extend(["-i", str(args.identity_file)])
    command.append(target)
    return command


def remote_quote(path: str) -> str:
    return shlex.quote(path)


def run_remote_snapshot(args: argparse.Namespace) -> dict[str, object]:
    command = build_ssh_base(args) + ["python3", "-", args.remote_db]
    result = subprocess.run(
        command,
        input=REMOTE_BACKUP_CODE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Remote SQLite snapshot failed")
    return json.loads(result.stdout)


def remove_remote_snapshot(args: argparse.Namespace, snapshot_path: str) -> None:
    command = build_ssh_base(args) + ["rm", "-f", remote_quote(snapshot_path)]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def scp_from_remote(args: argparse.Namespace, remote_path: str, local_path: Path) -> None:
    target = args.host
    if args.user:
        target = f"{args.user}@{target}"
    remote_spec = f"{target}:{remote_path}"
    command = ["scp", "-q"]
    if args.port:
        command.extend(["-P", str(args.port)])
    if args.identity_file:
        command.extend(["-i", str(args.identity_file)])
    command.extend([remote_spec, str(local_path)])
    subprocess.run(command, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_latest_link(dest: Path, backup_path: Path) -> None:
    latest = dest / "latest.db"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(backup_path.name)
    except OSError:
        shutil.copy2(backup_path, latest)


def prune_old_backups(dest: Path, retention_days: int) -> None:
    if retention_days <= 0:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for path in dest.glob("feedback-*.db"):
        if path.is_symlink():
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            path.unlink()


def build_parser(defaults: dict[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pull a consistent LocalLatin feedback DB snapshot over SSH, "
            "copying only when feedback rows changed."
        )
    )
    parser.add_argument("--host", default=env_default(defaults, "LOCALLATIN_BACKUP_HOST", ""))
    parser.add_argument("--user", default=env_default(defaults, "LOCALLATIN_BACKUP_USER", ""))
    parser.add_argument("--port", type=int, default=int(env_default(defaults, "LOCALLATIN_BACKUP_PORT", "22")))
    parser.add_argument(
        "--identity-file",
        default=env_default(defaults, "LOCALLATIN_BACKUP_IDENTITY_FILE", ""),
    )
    parser.add_argument(
        "--remote-db",
        default=env_default(defaults, "LOCALLATIN_REMOTE_DB", DEFAULT_REMOTE_DB),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(env_default(defaults, "LOCALLATIN_BACKUP_DEST", str(DEFAULT_DEST))).expanduser(),
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(env_default(defaults, "LOCALLATIN_BACKUP_RETENTION_DAYS", "30")),
    )
    parser.add_argument("--force", action="store_true", help="Download even when feedback is unchanged.")
    return parser


def main() -> int:
    defaults = load_env_file(DEFAULT_ENV_FILE)
    parser = build_parser(defaults)
    args = parser.parse_args()

    if not args.host:
        parser.error(
            "Set --host or LOCALLATIN_BACKUP_HOST in "
            f"{DEFAULT_ENV_FILE}"
        )

    args.identity_file = str(Path(args.identity_file).expanduser()) if args.identity_file else ""
    args.dest.mkdir(parents=True, exist_ok=True)
    args.dest.chmod(stat.S_IRWXU)

    state_path = args.dest / ".last_feedback_fingerprint"
    metadata_path = args.dest / "latest.json"
    previous_fingerprint = state_path.read_text(encoding="utf-8").strip() if state_path.exists() else ""

    snapshot: dict[str, object] | None = None
    local_tmp = args.dest / ".incoming-feedback.db"
    try:
        snapshot = run_remote_snapshot(args)
        fingerprint = str(snapshot["feedback_fingerprint"])
        if fingerprint == previous_fingerprint and not args.force:
            print("No feedback changes detected; backup unchanged.")
            return 0

        scp_from_remote(args, str(snapshot["snapshot_path"]), local_tmp)
        actual_sha = sha256_file(local_tmp)
        expected_sha = str(snapshot["db_sha256"])
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Downloaded snapshot hash mismatch: got {actual_sha}, expected {expected_sha}"
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = args.dest / f"feedback-{timestamp}.db"
        local_tmp.replace(backup_path)
        backup_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        write_latest_link(args.dest, backup_path)
        state_path.write_text(f"{fingerprint}\n", encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "backup_path": str(backup_path),
                    "db_sha256": expected_sha,
                    "feedback_fingerprint": fingerprint,
                    "feedback_rows": snapshot["feedback_rows"],
                    "remote_db": args.remote_db,
                    "remote_host": args.host,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        prune_old_backups(args.dest, args.retention_days)
        print(f"Copied updated feedback DB to {backup_path}")
        return 0
    finally:
        if local_tmp.exists():
            local_tmp.unlink()
        if snapshot and snapshot.get("snapshot_path"):
            remove_remote_snapshot(args, str(snapshot["snapshot_path"]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Backup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
