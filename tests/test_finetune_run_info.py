"""``run_info.json`` must survive the job that did not write most of it (#194).

The ceiling runs in two jobs. The GPU job trains, extracts and parity-checks;
the CPU job scores. Both write ``run_info.json``, and both used to dump it
wholesale, so the scoring job deleted the GPU job's ``parity`` report, its
``selection``, its ``train_seconds`` and its ``grad_checkpointing`` flag. The
loss was silent and the file still looked complete: it simply said
``grad_checkpointing: false`` and ``parity_check: false``, which is what a
scoring job's own arguments say. Appendix G quotes that record, so a reviewer
checking the claim found nothing behind it.

These tests pin the merge. They are deliberately about key survival, not about
values: the failure mode was a key disappearing, not a number changing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

pytest.importorskip("torch", reason="finetune_ceiling imports torch")
pytest.importorskip("transformers", reason="finetune_ceiling imports transformers")

import finetune_ceiling as ceiling  # noqa: E402


GPU_RECORD = {
    "config": {"stages": "train,extract", "grad_checkpointing": True,
               "parity_check": True, "cpu": False},
    "device": "cuda",
    "model_family": "single_stack",
    "n_blocks": 28,
    "grad_checkpointing": True,
    "parity": {"parity_layers": 2, "parity_layer1_max_abs_diff": 2.325e-06},
    "selection": {"selected_epoch": 3, "epochs_run": 6},
    "train_seconds": 123.4,
    "total_seconds": 200.0,
}

CPU_RECORD = {
    "config": {"stages": "evaluate,report,mseed", "grad_checkpointing": False,
               "parity_check": False, "cpu": True},
    "device": "cpu",
    "caption_facts": {"n_fit_pairs": 499, "selected_epoch": 3},
    "total_seconds": 300.0,
}


def merged_after_scoring() -> dict:
    return ceiling.merge_run_info(GPU_RECORD, CPU_RECORD, touched_model=False)


def test_scoring_keeps_every_key_the_gpu_job_wrote():
    merged = merged_after_scoring()
    for key in ("parity", "selection", "train_seconds", "model_family", "n_blocks"):
        assert merged[key] == GPU_RECORD[key], key


def test_scoring_does_not_flip_grad_checkpointing_to_its_own_default():
    """The exact symptom: a CPU job's argparse default overwrote the GPU flag."""
    assert merged_after_scoring()["grad_checkpointing"] is True


def test_the_training_config_stays_the_record_of_how_the_weights_were_made():
    merged = merged_after_scoring()
    assert merged["config"] == GPU_RECORD["config"]
    assert merged["config"]["parity_check"] is True
    # The scoring job's own arguments are kept, just not under `config`.
    assert merged["report_config"] == CPU_RECORD["config"]
    assert merged["report_total_seconds"] == 300.0
    assert merged["total_seconds"] == 200.0


def test_scoring_still_contributes_what_it_produced():
    assert merged_after_scoring()["caption_facts"] == CPU_RECORD["caption_facts"]


def test_scoring_does_not_rewrite_the_device_the_weights_were_trained_on():
    """Issue #210 review: `device` used to be written by both jobs, later wins.

    That is not harmless. Scoring runs on the CPU partition by the repo's budget
    rule, so every finished run's record claimed its weights were trained on
    `cpu`, contradicting the `parity` and `train_seconds` sitting beside it. It
    is job-scoped like `config`, so the scoring job's value is namespaced.
    """
    merged = merged_after_scoring()
    assert merged["device"] == "cuda"
    assert merged["report_device"] == "cpu"


def test_a_job_that_loaded_the_model_owns_config_and_total_seconds():
    """Re-running the GPU job must replace the training record, not shelve it."""
    merged = ceiling.merge_run_info(
        merged_after_scoring(), GPU_RECORD, touched_model=True
    )
    assert merged["config"] == GPU_RECORD["config"]
    assert merged["total_seconds"] == 200.0
    # The previous scoring job's namespaced keys survive untouched.
    assert merged["report_config"] == CPU_RECORD["config"]


def test_the_first_job_writes_an_ordinary_record():
    merged = ceiling.merge_run_info({}, CPU_RECORD, touched_model=False)
    assert merged["config"] == CPU_RECORD["config"]
    assert merged["total_seconds"] == 300.0
    assert "report_config" not in merged


def test_write_run_info_round_trips_through_disk(tmp_path):
    path = tmp_path / "run_info.json"
    ceiling.write_run_info(path, GPU_RECORD, touched_model=True)
    ceiling.write_run_info(path, CPU_RECORD, touched_model=False)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["grad_checkpointing"] is True
    assert on_disk["selection"]["selected_epoch"] == 3
    assert on_disk["caption_facts"]["n_fit_pairs"] == 499
    assert on_disk["device"] == "cuda"


def test_repeated_scoring_runs_do_not_erode_the_training_record():
    """A re-verification pass is a scoring job like any other, and there may be
    several. Every job-scoped key must still name the run that made the weights
    after the second and third of them."""
    merged = merged_after_scoring()
    for _ in range(2):
        merged = ceiling.merge_run_info(merged, CPU_RECORD, touched_model=False)
    assert merged["device"] == "cuda"
    assert merged["config"] == GPU_RECORD["config"]
    assert merged["total_seconds"] == 200.0
    assert merged["train_seconds"] == GPU_RECORD["train_seconds"]


def test_a_corrupt_record_is_replaced_rather_than_crashing(tmp_path):
    path = tmp_path / "run_info.json"
    path.write_text("{not json", encoding="utf-8")
    ceiling.write_run_info(path, CPU_RECORD, touched_model=False)
    # Nothing to preserve, so the scoring job's own device is the record.
    assert json.loads(path.read_text(encoding="utf-8"))["device"] == "cpu"
