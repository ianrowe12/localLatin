"""deploy/deploy.sh installs a sharded data release (issue #84).

The full-corpus attribution payload is several times the 2 GB GitHub Release
asset limit, so `make_data_release.sh --shard-bytes` splits it into parts. The
deploy host has to install the set, not one asset, and the two scripts have to
agree on the layout -- so these tests build a real sharded release with the real
packaging script and then install it with the real deploy function, over
``file://``. No network, no repo checkout, no systemd: ``DEPLOY_LIB_ONLY=1``
sources the definitions and stops before anything touches a host.

What each test pins down:

* a 3-part release installs every file, byte-for-byte;
* re-running is a no-op (the state hash covers the whole set, not one part);
* a corrupted part is refused *before* anything is staged;
* a parts list naming something outside the release is refused;
* the single-tarball layout still works, because most releases are still that.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SH = REPO_ROOT / "deploy" / "deploy.sh"
MAKE_RELEASE_SH = REPO_ROOT / "scripts" / "webapp" / "make_data_release.sh"

# The payload members make_data_release.sh requires.
PAYLOAD = [
    "runs/active/resubmit/unlabelled/unlabelled_predictions_raw.csv",
    "runs/active/resubmit/unlabelled/unlabelled_predictions_abtt.csv",
    "runs/active/resubmit/unlabelled/unlabelled_predictions_sif.csv",
    "runs/active/resubmit/unlabelled/unlabelled_predictions_sif_abtt.csv",
    "runs/active/ig_examples/phase12f_examples.csv",
    # Query-query matrices (issue #95). Picked up by glob rather than by name,
    # so having one here is what proves the glob actually reaches the archive.
    "runs/active/resubmit/unlabelled/qq_sim_bowphs_LaTa.npz",
    "runs/active/resubmit/unlabelled/qq_sim_google_mt5-base.npz",
]
N_ARTIFACTS = 24
ARTIFACT_BYTES = 40_000
SHARD_BYTES = 200_000  # forces several parts out of ~1 MB of artifacts


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("sha256sum") is None,
    reason="needs bash and sha256sum",
)


def _make_source_tree(root: Path) -> dict[str, bytes]:
    """A miniature repo with the payload make_data_release.sh expects."""
    contents: dict[str, bytes] = {}
    for rel in PAYLOAD:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        body = f"file_id,filename\n0,{Path(rel).name}\n".encode()
        path.write_bytes(body)
        contents[rel] = body
    for i in range(N_ARTIFACTS):
        rel = f"runs/active/ig_examples/artifacts/model_x/example{1_000_000 + i}_pair_example.npz"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        # Incompressible, so the packer cannot collapse the parts into one.
        body = hashlib.sha256(str(i).encode()).digest() * (ARTIFACT_BYTES // 32)
        path.write_bytes(body)
        contents[rel] = body
    return contents


def _build_release(src: Path, out: Path, tag: str, shard_bytes: int | None) -> None:
    cmd = [
        "bash", str(MAKE_RELEASE_SH),
        "--tag", tag, "--repo-root", str(src), "--out-dir", str(out),
    ]
    if shard_bytes is not None:
        cmd += ["--shard-bytes", str(shard_bytes)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _sync(cache: Path, dest: Path, out: Path, tag: str) -> subprocess.CompletedProcess:
    """Run deploy.sh's sync_data_release against the release in ``out``."""
    script = (
        f'DEPLOY_LIB_ONLY=1 source "{DEPLOY_SH}"\n'
        "sync_data_release\n"
    )
    env = dict(os.environ)
    env.update(
        DEPLOY_PATH=str(dest),
        DATA_CACHE_DIR=str(cache),
        DATA_RELEASE_TAG=tag,
        DATA_RELEASE_BASE_URL=out.as_uri(),
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env
    )


@pytest.fixture
def sharded(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "dist"
    dest = tmp_path / "host"
    cache = tmp_path / "cache"
    dest.mkdir()
    contents = _make_source_tree(src)
    _build_release(src, out, "test-sharded", SHARD_BYTES)
    return contents, out, dest, cache


def test_release_actually_sharded(sharded) -> None:
    """Guard the guard: a one-part release would make these tests vacuous."""
    _contents, out, _dest, _cache = sharded
    parts = sorted(out.glob("locallatin-test-sharded.part*.tar.gz"))
    assert len(parts) >= 3
    listed = (out / "locallatin-test-sharded.parts.txt").read_text().split()
    assert listed == [p.name for p in parts]
    for part in parts:
        assert (part.with_suffix(".gz.sha256")).exists()


def test_sharded_release_installs_every_file(sharded) -> None:
    contents, out, dest, cache = sharded

    proc = _sync(cache, dest, out, "test-sharded")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Sharded release: " in proc.stdout
    for rel, body in contents.items():
        assert (dest / rel).read_bytes() == body, rel
    installed = {
        str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()
    }
    assert installed == set(contents)


def test_second_sync_is_a_noop(sharded) -> None:
    """The state hash has to cover the whole set, not one part."""
    _contents, out, dest, cache = sharded
    assert _sync(cache, dest, out, "test-sharded").returncode == 0

    proc = _sync(cache, dest, out, "test-sharded")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "already installed" in proc.stdout


def test_a_corrupt_part_is_refused_before_anything_is_staged(sharded) -> None:
    _contents, out, dest, cache = sharded
    parts = sorted(out.glob("locallatin-test-sharded.part*.tar.gz"))
    # Corrupt the LAST part: an implementation that verified and unpacked part
    # by part would already have installed the first ones by the time it noticed.
    parts[-1].write_bytes(b"not a tarball")

    proc = _sync(cache, dest, out, "test-sharded")

    assert proc.returncode != 0
    assert "sha256 verification failed" in proc.stdout + proc.stderr
    assert list(dest.rglob("*")) == []
    assert not (cache / "installed.state").exists()


def test_a_parts_list_naming_a_foreign_file_is_refused(sharded) -> None:
    _contents, out, dest, cache = sharded
    listing = out / "locallatin-test-sharded.parts.txt"
    listing.write_text(listing.read_text() + "../../etc/passwd\n")

    proc = _sync(cache, dest, out, "test-sharded")

    assert proc.returncode != 0
    assert "Refusing parts list" in proc.stdout + proc.stderr
    assert list(dest.rglob("*")) == []


def test_unsharded_release_still_installs(tmp_path: Path) -> None:
    """Most releases are still a single asset; that path must not regress."""
    src = tmp_path / "src"
    out = tmp_path / "dist"
    dest = tmp_path / "host"
    cache = tmp_path / "cache"
    dest.mkdir()
    contents = _make_source_tree(src)
    _build_release(src, out, "test-single", None)
    assert (out / "locallatin-test-single.tar.gz").exists()
    assert not list(out.glob("*.parts.txt"))

    proc = _sync(cache, dest, out, "test-single")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Sharded release: " not in proc.stdout
    for rel, body in contents.items():
        assert (dest / rel).read_bytes() == body, rel
