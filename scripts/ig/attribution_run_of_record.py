"""The attribution run the paper's tables are built from, and the stamp that pins them to it.

Every attribution table in ``overleaf_drafts/tables/`` is generated from one
``summary_v2.csv``. Which run that summary comes from used to be a default
buried in each generator, and after the benchmark v1 re-sample (issue #187) the
defaults still named the earlier run 3 sample while the committed tables came
from v1 (issue #201). A bare regeneration then silently rewrote the paper's
numbers from the old sample.

Two things stop that here:

* ``RUN_OF_RECORD`` and ``METRICS_DIR_OF_RECORD`` are the single place the run
  name and the metrics directory live; the generators build their defaults
  from them.
* Each generated table carries a ``% source run: <run>/<metrics dir>`` line,
  and a generator refuses to overwrite a table stamped with a different source
  unless ``--allow_run_change`` is passed. Regenerating an older run for
  comparison still works: point ``--summary_csv`` at it and write the outputs
  somewhere else.

The metrics directory is part of the stamp because one run can carry more than
one metrics pass over the same artifacts. Issue #206 re-ran the benchmark v1
metrics with the chance-corrected deletion reference averaged over 20 random
orderings instead of 5 (``--random_order_draws 20``, as A8 of
``docs/research/attribution_metrics_decision.md`` advised) and wrote it beside
the 5-draw pass rather than over it. The two summaries differ in every DelAUC
gap cell, so a stamp that named only the run would let a bare regeneration
swap one for the other without a trace.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_OF_RECORD = "ig_examples_200pos_v1"
# 20 random-order draws for the deletion and insertion references (issue #206).
# ``attribution_metrics/`` in the same run is the earlier 5-draw pass, kept for
# the record and still the cache the DelAUC sensitivity sweep verifies against.
METRICS_DIR_OF_RECORD = "attribution_metrics_draws20"
ATTRIBUTION_METRICS_DIR = (
    REPO_ROOT / "runs/active" / RUN_OF_RECORD / METRICS_DIR_OF_RECORD
)
DEFAULT_SUMMARY_CSV = ATTRIBUTION_METRICS_DIR / "summary_v2.csv"

# Permutations of each attribution vector averaged by the shuffled-attribution
# control (``rand_*_gap`` columns of the summary). The sbatch of record,
# ``slurm/ig/attribution_metrics_200pos_v1_draws20.sbatch``, raises only
# ``--random_order_draws`` (the deletion and insertion reference) to 20 and
# leaves ``--shuffle_draws`` at ``DEFAULT_SHUFFLE_DRAWS`` in
# ``src/attribution_metrics.py``. The summary does not record the count, so
# the paper's control table takes it from here; a test pins it to both the
# code default and the sbatch.
SHUFFLE_DRAWS_OF_RECORD = 5

STAMP_PREFIX = "% source run: "
_STAMP_SEARCH_LINES = 8

PathLike = Union[str, Path]


def run_name(summary_csv: PathLike) -> str:
    """``<run>/<metrics dir>`` a summary belongs to.

    Summaries live at ``runs/active/<run>/<metrics dir>/<summary>.csv``, so the
    run is the grandparent and the metrics directory the parent. The path is
    taken as given (not resolved), so a symlinked checkout reports the names
    the operator typed.
    """
    metrics_dir = Path(summary_csv).absolute().parent
    return f"{metrics_dir.parent.name}/{metrics_dir.name}"


def stamp_line(run: str) -> str:
    return f"{STAMP_PREFIX}{run}"


def stamped_run(path: PathLike) -> Optional[str]:
    """The source a generated file was built from, or ``None`` if it carries no stamp."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        for _, line in zip(range(_STAMP_SEARCH_LINES), handle):
            if line.startswith(STAMP_PREFIX):
                return line[len(STAMP_PREFIX):].strip()
    return None


def refuse_run_change(out_path: PathLike, run: str, *, allow: bool = False) -> None:
    """Fail before writing if ``out_path`` was built from a different source.

    An unstamped target is treated as writable: that is the one-time case of
    stamping a table for the first time, and the case of a scratch output path.
    """
    existing = stamped_run(out_path)
    if existing is None or existing == run or allow:
        return
    raise SystemExit(
        f"{out_path} was built from run {existing}; refusing to rewrite it from "
        f"run {run}. If the run of record is changing on purpose, update "
        f"RUN_OF_RECORD or METRICS_DIR_OF_RECORD in "
        f"scripts/ig/attribution_run_of_record.py and pass --allow_run_change; "
        f"to compare an older run, write to another path."
    )
