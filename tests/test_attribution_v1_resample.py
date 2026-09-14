"""Guards for the issue #141 re-sample of the attribution pair set.

Three things went wrong, or could have gone wrong, when the 200-positive-pair
set was rebuilt on benchmark v1, and each has a test here.

1. The published sample came from the legacy phase-9 split over ``data/canon``,
   whose ``path`` column is absolute and points at a directory that no longer
   holds the corpus. The benchmark v1 split records paths *relative* to the repo
   root instead. The sampler has to turn both into absolute paths, because the
   NPZ generator opens the strings it writes.
2. ``refit_pcs_for_attribution.py`` selected its fit rows with
   ``np.where(split["split"] == "train")``, which indexes the cached matrix by
   position. The cache is frozen in corpus-walk order and the split is re-sorted
   by ``(folder_id, filename)``, so a label correction feeds test vectors into a
   train-only fit while every row count stays equal.
3. The GPU job cannot use ``build_attribution_run_manifest.py
   --require_complete`` (it demands a metrics summary this run never produces),
   so artifact completeness needs its own check that actually fails.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

from sample_positive_test_pairs import resolve_corpus_path  # noqa: E402
from verify_attribution_artifacts import verify  # noqa: E402
from compare_attribution_runs import win  # noqa: E402

SLUG = "bowphs_LaTa"
LAYER = 7
DIM = 5

# Four files. The relabelling moves ``d`` ahead of ``b`` and ``c`` in the split,
# which is the shape of the benchmark v1 correction: same files, same counts,
# different order.
CACHE_ORDER = ["a.txt", "b.txt", "c.txt", "d.txt"]
SPLIT_ORDER = ["a.txt", "d.txt", "b.txt", "c.txt"]
SPLIT_LABELS = ["train", "train", "test", "test"]


class TestResolveCorpusPath:
    def test_relative_benchmark_v1_path_becomes_absolute(self):
        resolved = resolve_corpus_path(
            "data/canon_labelled/Can.apost.50/BN2123.89r.6.txt", "/repo"
        )
        assert resolved == "/repo/data/canon_labelled/Can.apost.50/BN2123.89r.6.txt"

    def test_legacy_phase9_path_is_rewritten_under_data(self):
        resolved = resolve_corpus_path(
            "/u/irowerojas/localLatin/canon/Can.apost.19/C1525.5v.3.txt", "/repo"
        )
        assert resolved == "/repo/data/canon/Can.apost.19/C1525.5v.3.txt"

    def test_an_already_absolute_current_path_is_left_alone(self):
        path = "/repo/data/canon_labelled/X/y.txt"
        assert resolve_corpus_path(path, "/other-root") == path


@pytest.fixture
def permuted_cache(tmp_path: Path):
    """A bases root whose cache order is a real permutation of the split order."""
    bases_root = tmp_path / "bases"
    run_dir = bases_root / SLUG / "hidden_mean_tokempty"
    run_dir.mkdir(parents=True)

    # Row i of the cache is the constant vector i, so a row is identifiable by
    # its value and a wrong pairing is visible rather than merely different.
    emb = np.tile(np.arange(len(CACHE_ORDER), dtype=np.float32)[:, None], (1, DIM))
    np.save(run_dir / f"hidden_layer{LAYER}_embeddings.npy", emb)
    pd.DataFrame({"path": [f"canon_labelled/dir/{n}" for n in CACHE_ORDER]}).to_csv(
        run_dir / "meta.csv", index=False
    )

    split = pd.DataFrame(
        {
            "folder_id": ["dir"] * len(SPLIT_ORDER),
            "filename": SPLIT_ORDER,
            "path": [f"data/canon_labelled/dir/{n}" for n in SPLIT_ORDER],
            "split": SPLIT_LABELS,
        }
    )
    split_csv = tmp_path / "split.csv"
    split.to_csv(split_csv, index=False)
    return bases_root, split_csv, tmp_path / "pcs"


class TestRefitPcsAlignment:
    def _run(self, bases_root: Path, split_csv: Path, pc_root: Path):
        return subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/ig/refit_pcs_for_attribution.py"),
                "--slugs", SLUG,
                "--bases_root", str(bases_root),
                "--pooling", "hidden_mean_tokempty",
                "--split_csv", str(split_csv),
                "--pc_root", str(pc_root),
                "--d", "1",
                "--layer_overrides", f"{SLUG}={LAYER}",
            ],
            capture_output=True,
            text=True,
        )

    def test_fit_uses_filename_aligned_train_rows(self, permuted_cache):
        bases_root, split_csv, pc_root = permuted_cache
        result = self._run(bases_root, split_csv, pc_root)
        assert result.returncode == 0, result.stderr

        mean_vec = np.load(pc_root / SLUG / f"layer{LAYER}_pcs.npz")["mean_vec"]
        # Split rows 0 and 1 are train: a.txt and d.txt, cache rows 0 and 3.
        # Their mean is 1.5. Positional selection would take cache rows 0 and 1
        # and report 0.5, so the two answers cannot be confused.
        assert mean_vec == pytest.approx(np.full(DIM, 1.5))

    def test_the_permutation_is_reported(self, permuted_cache):
        bases_root, split_csv, pc_root = permuted_cache
        result = self._run(bases_root, split_csv, pc_root)
        assert "verified-permuted" in result.stdout
        assert "3 moved" in result.stdout

    def test_a_cache_describing_other_files_is_rejected(self, permuted_cache):
        bases_root, split_csv, pc_root = permuted_cache
        split = pd.read_csv(split_csv)
        split.loc[0, "filename"] = "not-in-the-cache.txt"
        split.to_csv(split_csv, index=False)
        result = self._run(bases_root, split_csv, pc_root)
        assert result.returncode != 0
        assert "different files" in (result.stdout + result.stderr)


def _write_artifact(path: Path, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.zeros((2, 2)) for key in keys})


@pytest.fixture
def artifact_run(tmp_path: Path):
    examples = pd.DataFrame(
        {
            "example_id": [1, 2],
            "model_name": ["bowphs/LaTa", "bowphs/LaTa"],
        }
    )
    examples_csv = tmp_path / "examples.csv"
    examples.to_csv(examples_csv, index=False)
    artifacts = tmp_path / "artifacts"
    full = [
        "pair_matrix_ig_baseline",
        "pair_matrix_ig_abtt",
        "pair_matrix_retrieval_mark_baseline",
        "pair_matrix_retrieval_mark_abtt",
    ]
    for example_id in (1, 2):
        _write_artifact(
            artifacts / "bowphs_LaTa" / f"example{example_id:03d}_pair_example.npz",
            full,
        )
    return examples_csv, artifacts, full


class TestVerifyArtifacts:
    def test_complete_run_has_no_problems(self, artifact_run):
        examples_csv, artifacts, _ = artifact_run
        assert verify(examples_csv, artifacts, ["ig", "retrieval_mark"]) == []

    def test_missing_npz_is_reported(self, artifact_run):
        examples_csv, artifacts, _ = artifact_run
        (artifacts / "bowphs_LaTa" / "example002_pair_example.npz").unlink()
        problems = verify(examples_csv, artifacts, ["ig", "retrieval_mark"])
        assert len(problems) == 1
        assert "missing artifact" in problems[0]

    def test_unmerged_marc_sidecar_is_reported(self, artifact_run):
        examples_csv, artifacts, full = artifact_run
        _write_artifact(
            artifacts / "bowphs_LaTa" / "example001_pair_example.npz",
            [k for k in full if "retrieval_mark" not in k],
        )
        problems = verify(examples_csv, artifacts, ["ig", "retrieval_mark"])
        assert len(problems) == 2
        assert all("retrieval_mark" in p for p in problems)


class TestWinDirection:
    def test_higher_is_better_metric(self):
        assert win(0.013, 0.298, higher_is_better=True) == "A"
        assert win(0.298, 0.013, higher_is_better=True) == "b"

    def test_lower_is_better_metric(self):
        # DelAUC and MinFrac are read the other way round; a metric read in the
        # wrong direction silently inverts a win count.
        assert win(0.291, 0.614, higher_is_better=False) == "b"
        assert win(0.534, 0.124, higher_is_better=False) == "A"
