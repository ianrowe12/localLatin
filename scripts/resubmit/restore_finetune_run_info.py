"""Put back the GPU-job provenance that the scoring job used to delete (#194).

``finetune_ceiling.py`` now merges into ``run_info.json`` instead of dumping it
(``merge_run_info``), but the two records already on disk were flattened before
that fix landed: both read ``grad_checkpointing: false`` and
``parity_check: false``, and neither carried the ``parity`` report, the
``selection`` or ``train_seconds`` the GPU job produced. Appendix G quotes that
record, so it has to be real.

**Every value here has a named source, and nothing is reconstructed by
inference.** ``selection`` is copied from the ``selection.json`` the training
stage wrote beside the checkpoint; ``config`` is the sbatch's own arguments,
which are committed; the parity numbers come from the GPU job's SLURM log
(Qwen3-0.6B) or, for LaTa, from the parity table in
``docs/research/finetune_ceiling.md``, whose job log predates the current
``slurm/logs`` contents. ``train_seconds`` was recorded nowhere that survived,
so it is **omitted rather than guessed**, and the ``restored`` block in each
file says so.

Re-running the GPU job would regenerate all of this first-hand and is the right
fix if the checkpoints are ever retrained; this script exists so that the
records match the artifacts without spending an A100 to recover a JSON file.

    python scripts/resubmit/restore_finetune_run_info.py            # dry run
    python scripts/resubmit/restore_finetune_run_info.py --write

``--runs_root`` points at the ``runs/`` tree holding the artifacts, which is the
main checkout's even when the script is invoked from a worktree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

from finetune_ceiling import merge_run_info  # noqa: E402

DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"

# Arguments as the committed sbatch files pass them. Only the keys that the
# scoring job's own defaults overwrote are listed; the rest of `config` is
# identical between the two jobs and is already correct on disk.
LATA = {
    "out_dir": "active/resubmit/finetune",
    "record": {
        "device": "cuda",
        "model_family": "seq2seq_encoder",
        "n_blocks": 12,
        "grad_checkpointing": False,
        "config_overrides": {
            "stages": "train,extract",
            "parity_check": True,
            "cpu": False,
            "grad_checkpointing": False,
            "trust_remote_code": False,
            "layers": "1-12",
        },
        # docs/research/finetune_ceiling.md, "Extraction parity" table.
        "parity": {
            "parity_layers": 2,
            "parity_layer1_max_abs_diff": 5.7e-05,
            "parity_layer1_mean_cosine": 1.000000,
            "parity_layer12_max_abs_diff": 1.4e-06,
            "parity_layer12_mean_cosine": 1.000000,
        },
        "restored": {
            "issue": 194,
            "why": (
                "the scoring stage overwrote run_info.json before merge_run_info "
                "existed, dropping the GPU job's keys"
            ),
            "selection_from": "selection.json beside the checkpoint",
            "config_from": "slurm/resubmit/finetune_lata_ceiling.sbatch",
            "parity_from": (
                "docs/research/finetune_ceiling.md, Extraction parity table "
                "(job 21847379, whose log is no longer under slurm/logs)"
            ),
            "not_recovered": ["train_seconds"],
        },
    },
}

QWEN = {
    "out_dir": "active/resubmit/finetune/qwen3_0.6b",
    "record": {
        "device": "cuda",
        "model_family": "single_stack",
        "n_blocks": 28,
        "grad_checkpointing": True,
        "config_overrides": {
            "stages": "train,extract",
            "parity_check": True,
            "cpu": False,
            "grad_checkpointing": True,
            "trust_remote_code": True,
            "layers": "1-28",
        },
        # slurm/logs/finetune_qwen_ceiling_22080571.out
        "parity": {
            "parity_layers": 2,
            "parity_layer1_max_abs_diff": 2.325e-06,
            "parity_layer1_mean_cosine": 1.000000,
            "parity_layer28_max_abs_diff": 4.625e-05,
            "parity_layer28_mean_cosine": 1.000000,
        },
        "restored": {
            "issue": 194,
            "why": (
                "the scoring stage overwrote run_info.json before merge_run_info "
                "existed, dropping the GPU job's keys"
            ),
            "selection_from": "selection.json beside the checkpoint",
            "config_from": "slurm/resubmit/finetune_qwen_ceiling.sbatch",
            "parity_from": "SLURM job 22080571 stdout",
            "not_recovered": ["train_seconds"],
        },
    },
}


def build(out_dir: Path, record: Dict[str, object]) -> Dict[str, object]:
    """The GPU job's record, assembled from the sources named above."""
    info_path = out_dir / "run_info.json"
    if not info_path.exists():
        raise SystemExit(f"no run_info.json at {info_path}")
    existing = json.loads(info_path.read_text(encoding="utf-8"))

    selection_path = out_dir / "selection.json"
    if not selection_path.exists():
        raise SystemExit(f"no selection.json at {selection_path}")

    gpu: Dict[str, object] = {
        k: v for k, v in record.items() if k not in ("config_overrides",)
    }
    gpu["selection"] = json.loads(selection_path.read_text(encoding="utf-8"))
    # `config` is the scoring job's on disk; correct only the keys whose values
    # differ between the two jobs, and leave the shared hyperparameters alone.
    config = dict(existing.get("config") or {})
    config.update(record["config_overrides"])  # type: ignore[arg-type]
    gpu["config"] = config
    return merge_run_info(existing, gpu, touched_model=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--write", action="store_true",
                   help="Write the files; without it the merged records are printed.")
    p.add_argument("--runs_root", default=str(DEFAULT_RUNS_ROOT),
                   help="The runs/ tree holding the artifacts. In a worktree this "
                        "is the main checkout's runs/, not the worktree's.")
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    changed: List[str] = []
    for name, spec in (("LaTa", LATA), ("Qwen3-0.6B", QWEN)):
        out_dir = runs_root / str(spec["out_dir"])
        merged = build(out_dir, dict(spec["record"]))  # type: ignore[arg-type]
        path = out_dir / "run_info.json"
        print(f"=== {name}: {path}")
        for key in ("grad_checkpointing", "model_family", "n_blocks"):
            print(f"    {key}: {merged[key]}")
        print(f"    selection: epoch {merged['selection']['selected_epoch']}")
        print(f"    parity: {merged['parity']}")
        print(f"    config.parity_check: {merged['config']['parity_check']}")
        if args.write:
            path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
            changed.append(str(path))
    print("wrote: " + ", ".join(changed) if changed else "dry run; pass --write")


if __name__ == "__main__":
    main()
