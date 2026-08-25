"""Merge one chunk's bulk registry into ``phase12f_examples.csv`` (issue #84).

The canonical examples CSV is what the webapp resolves against, so it is updated
once per chunk completion, atomically, and never in place:

1. back it up to ``phase12f_examples.csv.PRE_<slug>_<timestamp>.bak`` (kept, one
   per chunk, so any chunk's registration can be rolled back individually);
2. drop every existing row for this chunk's example_id block, so a rerun
   replaces its own rows instead of duplicating them;
3. concatenate the chunk's registry rows;
4. write to a temp file in the same directory and rename over the original.

The rows carry two columns the paper set does not have. Both are backfilled on
the existing rows so the CSV stays rectangular:

``variants_available``
    Which attribution variants the artifact can serve. Load-bearing: on four of
    the six models the four variants are deployed at different layers, so one
    ``(query, dir, model)`` has several artifacts and only this column says which
    one answers a given variant request. Backfilled to the full four-variant list
    on the paper rows, whose artifacts do carry all four.

``methods_available``
    Already present on the paper rows. Bulk artifacts carry ``ig`` only.

Usage::

    python scripts/ig/register_bulk_examples.py \\
        --registry runs/active/ig_examples/bulk_registry/bowphs_LaTa.csv \\
        --examples_csv runs/active/ig_examples/phase12f_examples.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bulk_attribution import (  # noqa: E402
    ARTIFACT_VARIANT_ORDER,
    EXAMPLE_ID_STRIDE,
    MODEL_PRIORITY,
    example_id_block,
)

LEGACY_VARIANTS = ",".join(ARTIFACT_VARIANT_ORDER)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", required=True, type=Path,
                   help="Per-chunk registry CSV from run_bulk_attribution.py.")
    p.add_argument("--examples_csv", type=Path,
                   default=REPO_ROOT / "runs/active/ig_examples/phase12f_examples.csv")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def backfill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "query_source" not in out.columns:
        out["query_source"] = "labelled"
    else:
        out["query_source"] = out["query_source"].fillna("labelled")
    if "top1_variants" not in out.columns:
        out["top1_variants"] = ""
    if "variants_available" not in out.columns:
        out["variants_available"] = LEGACY_VARIANTS
    else:
        blank = out["variants_available"].isna() | (
            out["variants_available"].astype(str).str.strip() == ""
        )
        out.loc[blank, "variants_available"] = LEGACY_VARIANTS
    return out


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.registry)
    if registry.empty:
        print(f"{args.registry} is empty; nothing to register.")
        return

    models = sorted(set(registry["model_name"].astype(str)))
    if len(models) != 1 or models[0] not in MODEL_PRIORITY:
        raise SystemExit(f"Registry must hold exactly one known model; got {models}")
    model_name = models[0]
    block = example_id_block(model_name)
    lo, hi = block, block + EXAMPLE_ID_STRIDE

    out_of_block = registry[(registry["example_id"] < lo) | (registry["example_id"] >= hi)]
    if not out_of_block.empty:
        raise SystemExit(
            f"{len(out_of_block)} registry rows fall outside {model_name}'s "
            f"example_id block [{lo}, {hi}); refusing to register."
        )
    dupes = registry["example_id"].duplicated().sum()
    if dupes:
        raise SystemExit(f"Registry has {dupes} duplicate example_id values.")

    existing = backfill(pd.read_csv(args.examples_csv))
    in_block = existing["example_id"].between(lo, hi - 1)
    print(f"{args.examples_csv}: {len(existing)} rows "
          f"({int(in_block.sum())} already in {model_name}'s block)")

    kept = existing[~in_block]
    new = registry.copy()
    for col in kept.columns:
        if col not in new.columns:
            new[col] = ""
    for col in new.columns:
        if col not in kept.columns:
            kept = kept.copy()
            kept[col] = LEGACY_VARIANTS if col == "variants_available" else ""
    new = new[list(kept.columns)]
    combined = pd.concat([kept, new], ignore_index=True)

    collisions = combined["example_id"].duplicated().sum()
    if collisions:
        raise SystemExit(f"Merged CSV would have {collisions} duplicate example_ids.")

    print(f"  removing {int(in_block.sum())}, adding {len(new)} -> {len(combined)} rows")
    if args.dry_run:
        print("[dry-run] nothing written.")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = model_name.replace("/", "_")
    backup = args.examples_csv.with_name(
        f"{args.examples_csv.name}.PRE_{slug}_{stamp}.bak"
    )
    backup.write_bytes(args.examples_csv.read_bytes())
    tmp = args.examples_csv.with_suffix(".tmp.csv")
    combined.to_csv(tmp, index=False)
    tmp.replace(args.examples_csv)
    print(f"  backup: {backup}")
    print(f"  wrote:  {args.examples_csv}")


if __name__ == "__main__":
    main()
