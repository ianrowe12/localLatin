"""The attribution run the paper's tables are built from, and the stamp that pins them to it.

Every attribution table in ``overleaf_drafts/tables/`` is generated from one
``summary_v2.csv``. Which run that summary comes from used to be a default
buried in each generator, and after the benchmark v1 re-sample (issue #187) the
defaults still named the earlier run 3 sample while the committed tables came
from v1 (issue #201). A bare regeneration then silently rewrote the paper's
numbers from the old sample.

Two things stop that here:

* ``RUN_OF_RECORD`` is the single place the run name lives; the generators
  build their defaults from it.
* Each generated table carries a ``% source run: <name>`` line, and a
  generator refuses to overwrite a table stamped with a different run unless
  ``--allow_run_change`` is passed. Regenerating an older run for comparison
  still works: point ``--summary_csv`` at it and write the outputs somewhere
  else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_OF_RECORD = "ig_examples_200pos_v1"
ATTRIBUTION_METRICS_DIR = REPO_ROOT / "runs/active" / RUN_OF_RECORD / "attribution_metrics"
DEFAULT_SUMMARY_CSV = ATTRIBUTION_METRICS_DIR / "summary_v2.csv"

STAMP_PREFIX = "% source run: "
_STAMP_SEARCH_LINES = 8

PathLike = Union[str, Path]


def run_name(summary_csv: PathLike) -> str:
    """Run directory name a summary belongs to.

    Summaries live at ``runs/active/<run>/attribution_metrics/<summary>.csv``,
    so the run is the grandparent. The path is taken as given (not resolved),
    so a symlinked checkout reports the name the operator typed.
    """
    return Path(summary_csv).absolute().parent.parent.name


def stamp_line(run: str) -> str:
    return f"{STAMP_PREFIX}{run}"


def stamped_run(path: PathLike) -> Optional[str]:
    """The run a generated file was built from, or ``None`` if it carries no stamp."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        for _, line in zip(range(_STAMP_SEARCH_LINES), handle):
            if line.startswith(STAMP_PREFIX):
                return line[len(STAMP_PREFIX):].strip()
    return None


def refuse_run_change(out_path: PathLike, run: str, *, allow: bool = False) -> None:
    """Fail before writing if ``out_path`` was built from a different run.

    An unstamped target is treated as writable: that is the one-time case of
    stamping a table for the first time, and the case of a scratch output path.
    """
    existing = stamped_run(out_path)
    if existing is None or existing == run or allow:
        return
    raise SystemExit(
        f"{out_path} was built from run {existing}; refusing to rewrite it from "
        f"run {run}. If the run of record is changing on purpose, update "
        f"RUN_OF_RECORD in scripts/ig/attribution_run_of_record.py and pass "
        f"--allow_run_change; to compare an older run, write to another path."
    )
