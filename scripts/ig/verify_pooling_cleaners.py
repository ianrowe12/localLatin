"""Verify that every artifact's sif_abtt panel uses the deployed SIF-pooled fit.

Three checks, per ``(model, layer)`` and per artifact (issue #87):

1. **Subspace.** Refit the SIF-pooled cleaner from the labelled TRAIN split the
   way ``run_resubmit_unlabelled_retrieval.py`` does, then compare it to the
   ``pcs_sif`` stored in each artifact by principal angles. All cosines must be
   1.0 to ``--atol``; anything less means the panel explains a different
   subspace from the ranking. The same comparison against the artifact's
   *mean*-pooled ``pcs`` is printed alongside, because that is what the panel
   used before the fix and it is the number the issue quotes.
2. **Provenance.** ``sif_abtt_cleaner_pooling`` must read ``"sif"``, and
   ``D_sif`` must equal the swept D.
3. **Reproduction.** Rebuild ``pair_matrix_<method>_sif_abtt`` from the raw
   hidden states with the stored SIF cleaner and the stored SIF token weights,
   and check it matches what is on disk. This catches an artifact that carries
   the right cleaner but was written by an older run.

Usage::

    python scripts/ig/verify_pooling_cleaners.py \\
        --examples_csv runs/active/ig_examples/phase12f_examples.csv \\
        --artifacts_dir runs/active/ig_examples/artifacts \\
        --labelled_bases runs/active/resubmit_bases/phase9_bases \\
        --split_csv runs/active/resubmit/data/phase_resubmit_split.csv

Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

from pooling_cleaners import (  # noqa: E402
    SIF_ABTT_CLEANER_KEY,
    fit_cleaner,
    principal_angle_cosines,
    read_cleaner,
)

from persist_attribution_methods import MAIN_METHODS, compute_cleaned_matrices  # noqa: E402
from persist_sif_attribution import reweight_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--examples_csv", type=Path,
                   default=REPO_ROOT / "runs/active/ig_examples/phase12f_examples.csv")
    p.add_argument("--artifacts_dir", type=Path,
                   default=REPO_ROOT / "runs/active/ig_examples/artifacts")
    p.add_argument("--labelled_bases", type=Path,
                   default=REPO_ROOT / "runs/active/resubmit_bases/phase9_bases")
    p.add_argument("--split_csv", type=Path,
                   default=REPO_ROOT / "runs/active/resubmit/data/phase_resubmit_split.csv")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--atol", type=float, default=1e-5,
                   help="Tolerance on 1 - cos(principal angle).")
    p.add_argument("--matrix_atol", type=float, default=1e-5,
                   help="Tolerance on the rebuilt sif_abtt matrices.")
    p.add_argument("--skip_matrix_check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)
    if args.models:
        examples = examples[examples["model_name"].isin(args.models)]
    split = pd.read_csv(args.split_csv)

    failures: list[str] = []
    checked = 0

    for (model_name, layer), group in examples.groupby(["model_name", "layer"], sort=True):
        slug = str(model_name).replace("/", "_")
        layer = int(layer)
        print(f"\n=== {slug} layer {layer} ({len(group)} rows) ===")
        deployed = fit_cleaner(args.labelled_bases, slug, "sif", layer, split, verbose=False)
        print(f"  deployed SIF-pooled fit: D={deployed.D}")

        for row in group.itertuples(index=False):
            path = (
                args.artifacts_dir / slug
                / f"example{int(row.example_id):03d}_{row.candidate_role}.npz"
            )
            if not path.exists():
                failures.append(f"{path}: missing")
                continue
            with np.load(path, allow_pickle=False) as handle:
                data = {k: handle[k] for k in handle.files}
            checked += 1
            tag = f"ex{int(row.example_id)} {slug}"

            stored = read_cleaner(data, "sif")
            if stored is None:
                failures.append(f"{tag}: no pcs_sif in the artifact")
                continue
            if stored.D != deployed.D:
                failures.append(f"{tag}: D_sif={stored.D}, deployed sweep says {deployed.D}")
            cos = principal_angle_cosines(stored.pcs, deployed.pcs)
            worst = float(cos.min()) if cos.size else 0.0
            if cos.size != deployed.D or worst < 1.0 - args.atol:
                failures.append(f"{tag}: principal-angle cosines vs deployed fit {cos}")

            marker = data.get(SIF_ABTT_CLEANER_KEY)
            if marker is None or str(np.asarray(marker).reshape(-1)[0]) != "sif":
                failures.append(f"{tag}: {SIF_ABTT_CLEANER_KEY} is {marker!r}, expected 'sif'")

            if args.skip_matrix_check:
                continue
            w_q = np.asarray(data["query_sif_weights"], dtype=np.float32)
            w_c = np.asarray(data["candidate_sif_weights"], dtype=np.float32)
            rebuilt = compute_cleaned_matrices(data, stored.pcs, stored.mean_vec)
            for method in MAIN_METHODS:
                key = f"pair_matrix_{method}_sif_abtt"
                if method not in rebuilt or key not in data:
                    continue
                expect = reweight_matrix(rebuilt[method], w_q, w_c)
                got = np.asarray(data[key], dtype=np.float32)
                if got.shape != expect.shape or not np.allclose(
                    got, expect, atol=args.matrix_atol, equal_nan=True
                ):
                    delta = float(np.nanmax(np.abs(got - expect))) if got.shape == expect.shape else float("nan")
                    failures.append(f"{tag}: {key} does not reproduce (max |delta| {delta:.3g})")

        # One representative artifact per group carries the before/after story.
        first = next(iter(group.itertuples(index=False)))
        path = (
            args.artifacts_dir / slug
            / f"example{int(first.example_id):03d}_{first.candidate_role}.npz"
        )
        if path.exists():
            with np.load(path, allow_pickle=False) as handle:
                rep = {k: handle[k] for k in handle.files}
            mean_cleaner = read_cleaner(rep, "mean")
            sif_cleaner = read_cleaner(rep, "sif")
            if mean_cleaner is not None:
                old = principal_angle_cosines(mean_cleaner.pcs, deployed.pcs)
                print(
                    f"  before the fix (mean D={mean_cleaner.D} vs deployed sif D={deployed.D}): "
                    + np.array2string(old, precision=3, floatmode="fixed")
                )
            if sif_cleaner is not None:
                new = principal_angle_cosines(sif_cleaner.pcs, deployed.pcs)
                print(
                    "  after  the fix: "
                    + np.array2string(new, precision=6, floatmode="fixed")
                )

    print(f"\n=== {checked} artifacts checked, {len(failures)} failures ===")
    for f in failures[:40]:
        print("  FAIL " + f)
    if failures:
        raise SystemExit(1)
    print("All artifacts carry the deployed SIF-pooled cleaner and reproduce their "
          "sif_abtt matrices from it.")


if __name__ == "__main__":
    main()
