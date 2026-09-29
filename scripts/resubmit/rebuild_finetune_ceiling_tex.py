"""Rebuild ``tables/finetune_ceiling.tex`` from finished ceiling runs, on CPU.

``finetune_ceiling.py`` writes the table at the end of its ``report`` stage,
which needs the split, the model and write access to the run directory. A
caption or layout change to the table needs none of that: every number and
every caption fact is already on disk in each run's comparison CSV, five-seed
CSV and ``run_info.json``. This rebuilds the table from those files through the
same ``write_tex`` the scoring job uses, and reads the run directories only.

    python scripts/resubmit/rebuild_finetune_ceiling_tex.py
    python scripts/resubmit/rebuild_finetune_ceiling_tex.py --tex_out /tmp/check.tex

The default runs are the three committed ceilings, in table order, so a bare
run reproduces the committed table (issue #208).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

import finetune_ceiling as ceiling  # noqa: E402

DEFAULT_RUNS = [
    "LaTa:finetune_lata:runs/active/resubmit/finetune",
    "Qwen3-0.6B:finetune_qwen3_0.6b:runs/active/resubmit/finetune/qwen3_0.6b",
    "KaLM-mini:finetune_kalm_mini:runs/active/resubmit/finetune/kalm_mini",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="SPEC",
        help="'<display_name>:<results_prefix>:<out_dir>', repeatable, in table "
             "order. Default: the three committed ceilings.",
    )
    p.add_argument("--results_dir", default="runs/active/resubmit/results/finetune")
    p.add_argument("--tex_out", default="overleaf_drafts/tables/finetune_ceiling.tex")
    p.add_argument(
        "--notes_out", default=None,
        help="Optional file for the run notes and five-seed readout, which the "
             "table itself no longer carries.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    sections = [
        ceiling.load_extra_section(spec, results_dir) for spec in (args.run or DEFAULT_RUNS)
    ]
    ceiling.write_tex(sections, Path(args.tex_out),
                      notes_path=Path(args.notes_out) if args.notes_out else None)


if __name__ == "__main__":
    main()
