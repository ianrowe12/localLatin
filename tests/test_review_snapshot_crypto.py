"""Stdlib-only snapshot format checks, also run with macOS OpenSSL 3."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.webapp import review_snapshot as snapshot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def keypair(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("snapshot-crypto-keys")
    private, public = root / "private.pem", root / "public.pem"
    snapshot.openssl("genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:4096",
                     "-out", str(private))
    snapshot.openssl("pkey", "-in", str(private), "-pubout", "-out", str(public))
    return private, public


def test_encrypt_decrypt_helper_and_tamper_checks(tmp_path: Path, keypair) -> None:
    work = tmp_path / "work"
    bundle = work / "bundle"
    bundle.mkdir(parents=True)
    for name in snapshot.FILES:
        (bundle / name).write_text("xml_id,ccl_key\nwitness,CTOU.567.16\n", encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps({
        "snapshot_schema_version": 1,
        "files": {
            name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            for name in snapshot.FILES
        },
    }), encoding="utf-8")
    private, public = keypair
    artifact = tmp_path / "artifact"
    snapshot.encrypt_bundle(bundle, public, artifact, work)
    assert {p.name for p in artifact.iterdir()} == {snapshot.CIPHERTEXT, snapshot.WRAPPED_KEY}
    with tarfile.open(work / "snapshot.tar.gz") as tar:
        assert all(m.uid == m.gid == 0 and not m.uname and not m.gname for m in tar.getmembers())
    out = tmp_path / "decrypted"
    helper = ROOT / "scripts/webapp/decrypt_review_snapshot.sh"
    # Set this to a second binary to exercise 1.1.1 <-> 3 interoperability.
    decrypt_env = {
        **os.environ,
        "OPENSSL": os.environ.get("SNAPSHOT_DECRYPT_OPENSSL", os.environ.get("OPENSSL", "openssl")),
    }
    result = subprocess.run([str(helper), str(artifact), str(private), str(out)],
                            capture_output=True, check=False, env=decrypt_env)
    assert result.returncode == 0, result.stderr
    assert all((out / name).read_bytes() == (bundle / name).read_bytes()
               for name in (*snapshot.FILES, "manifest.json"))
    # Authentication succeeds for this new envelope, but its stale manifest
    # must still prevent publication of the modified payload.
    (bundle / "reviews.csv").write_text((bundle / "reviews.csv").read_text() + "tampered\n")
    tampered = tmp_path / "tampered-artifact"
    snapshot.encrypt_bundle(bundle, public, tampered, work)
    refused = tmp_path / "refused"
    result = subprocess.run([str(helper), str(tampered), str(private), str(refused)],
                            capture_output=True, check=False, env=decrypt_env)
    assert result.returncode != 0
    assert b"stage=decrypt exception=SnapshotError" in result.stderr
    assert not refused.exists()
    with pytest.raises(snapshot.SnapshotError, match="Manifest SHA256 verification failed"):
        snapshot.decrypt_snapshot(tampered, private, refused)
    ciphertext = artifact / snapshot.CIPHERTEXT
    data = bytearray(ciphertext.read_bytes())
    data[-1] ^= 1
    ciphertext.write_bytes(data)
    with pytest.raises(snapshot.SnapshotError, match="authentication failed"):
        snapshot.decrypt_snapshot(artifact, private, refused)
    assert not refused.exists()
    assert not list(tmp_path.glob(".review-decrypt-*"))


def test_export_rejects_smaller_rsa_key(tmp_path: Path) -> None:
    private, public = tmp_path / "private.pem", tmp_path / "public.pem"
    snapshot.openssl("genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048",
                     "-out", str(private))
    snapshot.openssl("pkey", "-in", str(private), "-pubout", "-out", str(public))
    with pytest.raises(snapshot.SnapshotError, match="RSA-4096"):
        snapshot.encrypt_bundle(tmp_path / "unused-bundle", public, tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()
