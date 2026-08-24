"""Fit and persist the per-pooling ABTT cleaners the artifacts need (issue #87).

For every ``(model, layer)`` an attribution artifact uses, this fits the
SIF-pooled ``EmbeddingCleaner`` exactly the way the deployed ``sif_abtt``
retrieval fits it -- labelled TRAIN split only, D swept by
``find_optimal_D`` -- and merges it into the shared PC file at
``<pc_root>/<slug>/layer{N}_pcs.npz`` under the ``pcs_sif`` / ``mean_vec_sif``
/ ``D_sif`` keys (see ``scripts/ig/pooling_cleaners.py`` for the format).

The mean-pooled keys (``pcs`` / ``mean_vec``) are **never** overwritten. They
back the ``abtt`` panels of 128 shipped artifacts, several of which were fit by
a pipeline that no longer exists (LaTa L4 at D=2, LaBSE L12 at D=1, Qwen L23 at
D=3), and re-deriving them would silently change panels this change has no
business touching. When a PC file is missing the mean keys entirely, pass
``--seed_mean_from_artifacts`` to copy them out of the artifacts themselves --
every artifact of a given ``(model, layer)`` carries the same pair, and the
script verifies that before copying.

``--report_only`` prints the principal-angle cosines between the two poolings'
subspaces without writing anything, which is the diagnostic that motivated the
issue.

Usage::

    python scripts/ig/fit_pooling_cleaners.py \\
        --examples_csv runs/active/ig_examples/phase12f_examples.csv \\
        --artifacts_dir runs/active/ig_examples/artifacts \\
        --labelled_bases runs/active/resubmit_bases/phase9_bases \\
        --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \\
        --pc_root runs/phase12_release/pcs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pooling_cleaners import (  # noqa: E402
    CLEANER_KEYS,
    Cleaner,
    cleaners_match,
    fit_cleaner,
    pc_file_path,
    principal_angle_cosines,
    read_cleaner,
    write_cleaner,
)


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
    p.add_argument("--pc_root", type=Path, default=REPO_ROOT / "runs/phase12_release/pcs")
    p.add_argument("--models", nargs="*", default=None,
                   help="Restrict to these model_name strings.")
    p.add_argument("--fixed_d_sif", type=int, default=None,
                   help="Skip the D sweep and use this D for the SIF fit. Only for "
                        "reproducing an older run; the default sweep is what the "
                        "deployed sif_abtt variant does.")
    p.add_argument("--seed_mean_from_artifacts", action="store_true",
                   help="When the PC file has no mean-pooled keys, copy them from the "
                        "artifacts so the file describes both poolings.")
    p.add_argument("--report_only", action="store_true",
                   help="Fit and compare, but write nothing.")
    p.add_argument("--dry_run", action="store_true", help="Alias for --report_only.")
    return p.parse_args()


def artifact_paths(artifacts_dir: Path, group: pd.DataFrame) -> list[Path]:
    paths = []
    for row in group.itertuples(index=False):
        slug = str(row.model_name).replace("/", "_")
        p = artifacts_dir / slug / f"example{int(row.example_id):03d}_{row.candidate_role}.npz"
        if p.exists():
            paths.append(p)
    return paths


def mean_cleaner_from_artifacts(paths: list[Path]) -> Cleaner | None:
    """The mean-pooled cleaner baked into a set of artifacts, if they agree."""
    found: Cleaner | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            cleaner = read_cleaner(data, "mean")
        if cleaner is None:
            return None
        if found is None:
            found = cleaner
        elif not cleaners_match(found, cleaner):
            raise SystemExit(
                f"Artifacts disagree on the mean-pooled cleaner at {path.parent.name}; "
                "refusing to seed the PC file from them."
            )
    return found


def merge_pc_file(
    path: Path,
    sif: Cleaner,
    mean_seed: Cleaner | None,
    write: bool,
) -> str:
    """Merge the SIF cleaner into a PC file, preserving every existing key."""
    existing: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            existing = {k: data[k] for k in data.files}

    prior_sif = read_cleaner(existing, "sif")
    if cleaners_match(prior_sif, sif):
        return "unchanged"

    merged = dict(existing)
    write_cleaner(merged, sif)

    seeded = ""
    if read_cleaner(existing, "mean") is None and mean_seed is not None:
        write_cleaner(merged, mean_seed)
        seeded = " (+ mean seeded from artifacts)"

    if not write:
        keys = CLEANER_KEYS["sif"]
        return f"[DRY] would write {keys.pcs}/{keys.mean_vec}/{keys.d}{seeded}"

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **merged)
    tmp.replace(path)
    return f"written{seeded}"


def main() -> None:
    args = parse_args()
    write = not (args.report_only or args.dry_run)

    examples = pd.read_csv(args.examples_csv)
    if args.models:
        examples = examples[examples["model_name"].isin(args.models)]
    if examples.empty:
        raise SystemExit("No example rows selected.")

    split = pd.read_csv(args.split_csv)

    summary: list[dict] = []
    for (model_name, layer), group in examples.groupby(["model_name", "layer"], sort=True):
        slug = str(model_name).replace("/", "_")
        layer = int(layer)
        print(f"\n=== {slug} layer {layer} ({len(group)} artifacts) ===")

        sif = fit_cleaner(
            args.labelled_bases, slug, "sif", layer, split, fixed_d=args.fixed_d_sif
        )

        paths = artifact_paths(args.artifacts_dir, group)
        artifact_mean = mean_cleaner_from_artifacts(paths) if paths else None

        # The diagnostic from the issue: how far the panel's subspace was from
        # the one the deployed sif_abtt ranking removes.
        if artifact_mean is not None:
            cos = principal_angle_cosines(artifact_mean.pcs, sif.pcs)
            print(
                f"  artifact mean cleaner: D={artifact_mean.D}   "
                f"SIF-pooled fit: D={sif.D}"
            )
            print(
                "  principal-angle cosines (artifact mean subspace vs SIF-pooled fit): "
                + np.array2string(cos, precision=3, floatmode="fixed")
            )

        pc_path = pc_file_path(args.pc_root, slug, layer)
        status = merge_pc_file(
            pc_path,
            sif,
            artifact_mean if args.seed_mean_from_artifacts else None,
            write,
        )
        print(f"  {pc_path}: {status}")

        summary.append(
            {
                "model": model_name,
                "layer": layer,
                "artifacts": len(paths),
                "D_mean_artifact": artifact_mean.D if artifact_mean else None,
                "D_sif": sif.D,
                "pc_file": str(pc_path),
                "status": status,
            }
        )

    print("\n=== Summary ===")
    print(pd.DataFrame(summary).to_string(index=False))
    if not write:
        print("\nReport only -- nothing written.")


if __name__ == "__main__":
    main()
