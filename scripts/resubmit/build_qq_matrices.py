"""Query-query cosine matrices for the unlabelled corpus (issue #95).

The webapp lets a reviewer create a *new* directory seeded by an unlabelled
query that matches nothing in ``data/canon_labelled/``. For that directory to
behave like any other candidate, every later query has to be scored against it,
and its members are unlabelled queries -- not labelled files. The predictions
CSVs only hold query -> labelled-directory scores, so the webapp needs the
missing half: query -> query cosine.

This script builds it, once per model, at exactly the configuration the
deployed retrieval uses.

Same-pipeline guarantee
-----------------------
Everything that decides the numbers is imported from
``run_resubmit_unlabelled_retrieval.py`` rather than re-implemented:

* the (layer, repr) pair, read from the results CSV by
  :func:`find_best_config_from_results` restricted to the variant's methods;
* the SIF-pooled embedding file the variant implies;
* the degenerate-file guard (blank source text + zero-norm backstop);
* ``find_optimal_D``'s train-only sweep, and the ``EmbeddingCleaner`` fitted on
  the train split only.

Only after that shared prefix does this script diverge: instead of
``unlab_norm @ lab_norm.T`` it computes ``unlab_norm @ unlab_norm.T``.

``--verify_n`` (default 3) then closes the loop empirically. For N queries per
model it runs the *labelled* half of the pipeline too and compares the
resulting top-1 directory and score against
``unlabelled_predictions_{variant}.csv``. A mismatch beyond ``--verify_tol``
(1e-5, i.e. 5 decimals) is fatal: it means the matrix was built from different
bases, a different layer, or a differently fitted cleaner than the predictions
the webapp serves, and mixing the two would put two incomparable similarity
scales on the same screen.

Output
------
``{out_dir}/qq_sim_{model_slug}.npz`` with

===============  ============================  =====================================
key              dtype / shape                 meaning
===============  ============================  =====================================
``sim``          float16 (n, n)                cosine, row/col i <-> ``file_ids[i]``
``file_ids``     int32 (n,)                    query file_id per row/column
``excluded``     bool (n,)                     degenerate query (guard); row/col 0
``meta``         0-d unicode (JSON)            model, layer, D, variant, ...
===============  ============================  =====================================

float16 costs ~1e-3 of absolute precision on a [-1, 1] cosine, which is far
below the 0.5/0.7 confidence bands the UI draws, and keeps a matrix at ~10 MB
so all six ship in a data release.

Rows and columns of guard-excluded queries are zeroed rather than left at their
computed value. After ABTT with ``center=True`` every zero-norm row becomes
``-mean_vec``, so two empty files score a spurious cosine of exactly 1.0
(issue #66) -- the same trap the labelled path avoids by dropping them from the
directory max. Zeroing makes them inert instead.

Usage
-----
    python scripts/resubmit/build_qq_matrices.py            # all six models
    python scripts/resubmit/build_qq_matrices.py --models bowphs/LaTa
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

from canon_retrieval import l2_normalize, zero_norm_mask  # noqa: E402
from embedding_alignment import AlignmentResolver  # noqa: E402
from sif_abtt import EmbeddingCleaner  # noqa: E402


def _load_retrieval_module():
    """Import the deployed retrieval script as a module.

    ``scripts/`` is not a package, so this goes through importlib rather than a
    plain import. Reusing the module is the point: the layer choice, the guard
    and the D sweep must be *the same code*, not a copy that can drift.
    """
    path = Path(__file__).resolve().parent / "run_resubmit_unlabelled_retrieval.py"
    spec = importlib.util.spec_from_file_location("_resubmit_unlabelled_retrieval", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load retrieval module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RETRIEVAL = _load_retrieval_module()

VARIANTS = RETRIEVAL.VARIANTS
FALLBACK_CONFIGS = RETRIEVAL.FALLBACK_CONFIGS
D_VALUES = RETRIEVAL.D_VALUES
model_slug = RETRIEVAL.model_slug

#: Absolute tolerance for the against-the-deployed-CSV check. The CSV rounds
#: scores to 6 decimals, so 1e-5 is "matches to 5 decimals" with room to spare.
VERIFY_TOL = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build query-query cosine matrices for the unlabelled corpus."
    )
    parser.add_argument("--labelled_bases", default="runs/active/resubmit_bases/phase9_bases")
    parser.add_argument("--unlabelled_bases", default="runs/active/resubmit/unlabelled/bases")
    parser.add_argument("--split_csv", default="runs/active/resubmit/data/phase_resubmit_split.csv")
    parser.add_argument("--unlabelled_meta", default="runs/active/resubmit/data/unlabelled_meta.csv")
    parser.add_argument(
        "--results_csv", default="runs/active/resubmit/results/phase_resubmit_results.csv"
    )
    parser.add_argument("--out_dir", default="runs/active/resubmit/unlabelled")
    parser.add_argument(
        "--predictions_dir",
        default="runs/active/resubmit/unlabelled",
        help="Where unlabelled_predictions_{variant}_{slug}.csv lives (for --verify_n).",
    )
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'.")
    parser.add_argument("--variant", default="sif_abtt", choices=sorted(VARIANTS))
    parser.add_argument("--layer_overrides", default="")
    parser.add_argument("--data_root", default="data")
    parser.add_argument(
        "--verify_n",
        type=int,
        default=3,
        help="Queries per model to re-score against the deployed predictions CSV. 0 disables.",
    )
    parser.add_argument("--verify_tol", type=float, default=VERIFY_TOL)
    parser.add_argument("--no_degenerate_guard", action="store_true")
    return parser.parse_args()


def resolve_model_configs(args: argparse.Namespace, sel_methods) -> list[tuple[str, int, str]]:
    """(model, layer, repr) per model, exactly as the retrieval script picks it."""
    results_path = Path(args.results_csv)
    if results_path.exists():
        print(f"Reading best layers from {results_path}")
        best_configs = RETRIEVAL.find_best_config_from_results(args.results_csv, sel_methods)
    else:
        print("Results CSV not found, using fallback configs.")
        best_configs = {}

    layer_overrides = RETRIEVAL.parse_layer_overrides(args.layer_overrides)

    configs: list[tuple[str, int, str]] = []
    for model_name, fallback_layer, fallback_repr in FALLBACK_CONFIGS:
        if model_name in best_configs:
            layer, repr_name = best_configs[model_name]
            source = "from results"
        else:
            layer, repr_name = fallback_layer, fallback_repr
            source = "fallback"
        if model_name in layer_overrides:
            layer = layer_overrides[model_name]
            source = "override"
        print(f"  {model_name}: layer={layer}, repr={repr_name} ({source})")
        configs.append((model_name, layer, repr_name))

    if args.models != "all":
        selected = {m.strip() for m in args.models.split(",")}
        configs = [c for c in configs if c[0] in selected]
    return configs


def verify_against_predictions(
    *,
    model_name: str,
    slug: str,
    variant: str,
    lab_norm: np.ndarray,
    unlab_norm: np.ndarray,
    split_meta: pd.DataFrame,
    unlabelled_meta: pd.DataFrame,
    lab_exclude: np.ndarray,
    unlab_exclude: np.ndarray,
    predictions_dir: Path,
    verify_n: int,
    tol: float,
) -> list[dict]:
    """Re-score N queries against the labelled dirs and diff the deployed CSV.

    Returns one record per checked query. Raises if any differs by more than
    ``tol`` -- a silent mismatch here is the whole failure mode this guards.
    """
    csv_path = predictions_dir / f"unlabelled_predictions_{variant}_{slug}.csv"
    if not csv_path.exists():
        csv_path = predictions_dir / f"unlabelled_predictions_{variant}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No deployed predictions CSV for {model_name} under {predictions_dir}; "
            "cannot verify. Pass --verify_n 0 only if you know why."
        )

    deployed = pd.read_csv(csv_path)
    if "model" in deployed.columns:
        deployed = deployed[deployed["model"] == model_name]
    deployed = deployed[deployed["status"] == RETRIEVAL.STATUS_OK]
    if deployed.empty:
        raise ValueError(f"No 'ok' prediction rows for {model_name} in {csv_path}")

    # Deterministic, spread across the corpus rather than the first N rows.
    positions = np.linspace(0, len(deployed) - 1, num=verify_n).astype(int)
    sample = deployed.iloc[np.unique(positions)]

    fid_to_index = {int(fid): i for i, fid in enumerate(unlabelled_meta["file_id"].tolist())}

    records: list[dict] = []
    failures: list[str] = []
    for _, row in sample.iterrows():
        file_id = int(row["file_id"])
        idx = fid_to_index[file_id]
        preds = RETRIEVAL.predict_directories(
            lab_norm,
            unlab_norm[idx : idx + 1],
            split_meta["folder_id"].values,
            top_k=1,
            exclude_labelled=lab_exclude,
            exclude_unlabelled=unlab_exclude[idx : idx + 1],
        )[0]
        if not preds:
            failures.append(f"file_id={file_id}: recomputed no prediction, CSV has one")
            continue
        got_dir, got_score = preds[0]
        want_dir = str(row["rank1_dir"])
        want_score = float(row["rank1_score"])
        delta = abs(got_score - want_score)
        ok = got_dir == want_dir and delta <= tol
        records.append(
            {
                "model": model_name,
                "file_id": file_id,
                "filename": str(row["filename"]),
                "csv_rank1_dir": want_dir,
                "csv_rank1_score": want_score,
                "recomputed_rank1_dir": got_dir,
                "recomputed_rank1_score": round(float(got_score), 8),
                "abs_delta": float(delta),
                "match": bool(ok),
            }
        )
        status = "OK " if ok else "FAIL"
        print(
            f"    [{status}] file_id={file_id:<5d} dir={want_dir!r}"
            f" csv={want_score:.6f} recomputed={got_score:.6f} |delta|={delta:.2e}"
        )
        if not ok:
            failures.append(
                f"file_id={file_id}: CSV ({want_dir}, {want_score:.6f}) vs "
                f"recomputed ({got_dir}, {got_score:.6f}), |delta|={delta:.2e}"
            )

    if failures:
        raise SystemExit(
            f"Verification FAILED for {model_name} against {csv_path}:\n  "
            + "\n  ".join(failures)
        )
    return records


def main() -> None:
    args = parse_args()
    variant = args.variant
    pooling, apply_abtt, sel_methods = VARIANTS[variant]
    print(f"Variant: {variant} (pooling={pooling}, abtt={apply_abtt})")

    split_meta = pd.read_csv(args.split_csv)
    lab_aligner = AlignmentResolver(split_meta)
    unlabelled_meta = pd.read_csv(args.unlabelled_meta)
    file_ids = unlabelled_meta["file_id"].to_numpy(dtype=np.int32)

    data_root = Path(args.data_root)
    if args.no_degenerate_guard:
        print("\nDegenerate-file guard DISABLED.")
        lab_blank = np.zeros(len(split_meta), dtype=bool)
        unlab_blank = np.zeros(len(unlabelled_meta), dtype=bool)
    else:
        lab_blank = RETRIEVAL.read_blank_source_mask(split_meta["path"].tolist(), data_root)
        unlab_blank = RETRIEVAL.read_blank_source_mask(
            unlabelled_meta["path"].tolist(), data_root
        )
        print(
            f"\nDegenerate-file guard: {int(lab_blank.sum())} labelled, "
            f"{int(unlab_blank.sum())} unlabelled blank source file(s)."
        )

    model_configs = resolve_model_configs(args, sel_methods)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = Path(args.predictions_dir)

    all_verifications: list[dict] = []
    summary: list[dict] = []

    for model_name, layer, repr_name in model_configs:
        slug = model_slug(model_name)
        print(f"\n{'=' * 60}\nModel: {model_name} (layer {layer}, {pooling}, {variant})")

        suffix = "_sif" if pooling == "sif" else ""
        subdir = f"{repr_name}_{pooling}_tokempty"
        fname = f"{repr_name}_layer{layer}_embeddings{suffix}.npy"
        lab_path = Path(args.labelled_bases) / slug / subdir / fname
        unlab_path = Path(args.unlabelled_bases) / slug / subdir / fname

        if not lab_path.exists() or not unlab_path.exists():
            print(f"  Embeddings missing ({lab_path} / {unlab_path}); skipping.")
            continue

        lab_emb = lab_aligner.load(lab_path)
        unlab_emb = np.load(unlab_path)
        print(f"  Labelled: {lab_emb.shape}, Unlabelled: {unlab_emb.shape}")
        if lab_emb.shape[0] != len(split_meta) or unlab_emb.shape[0] != len(unlabelled_meta):
            print("  Shape mismatch against meta CSVs; skipping.")
            continue

        if args.no_degenerate_guard:
            lab_zero = np.zeros(lab_emb.shape[0], dtype=bool)
            unlab_zero = np.zeros(unlab_emb.shape[0], dtype=bool)
        else:
            lab_zero = zero_norm_mask(lab_emb)
            unlab_zero = zero_norm_mask(unlab_emb)
        lab_exclude = lab_blank | lab_zero
        unlab_exclude = unlab_blank | unlab_zero
        print(
            f"  Excluded: {int(lab_exclude.sum())} labelled reference(s), "
            f"{int(unlab_exclude.sum())} quer(y/ies)."
        )

        best_D: int | None = None
        if apply_abtt:
            train_mask = split_meta["split"].values == "train"
            train_emb = lab_emb[train_mask]
            train_folder_ids = split_meta.loc[train_mask, "folder_id"].values
            best_D = RETRIEVAL.find_optimal_D(train_emb, train_folder_ids, D_VALUES)
            print(f"  Optimal D: {best_D}")
            cleaner = EmbeddingCleaner(num_components=best_D, center=True)
            cleaner.fit(train_emb)
            lab_emb = cleaner.transform(lab_emb)
            unlab_emb = cleaner.transform(unlab_emb)

        lab_norm = l2_normalize(lab_emb)
        unlab_norm = l2_normalize(unlab_emb)

        if args.verify_n > 0:
            print(f"  Verifying {args.verify_n} quer(y/ies) against the deployed CSV:")
            all_verifications.extend(
                verify_against_predictions(
                    model_name=model_name,
                    slug=slug,
                    variant=variant,
                    lab_norm=lab_norm,
                    unlab_norm=unlab_norm,
                    split_meta=split_meta,
                    unlabelled_meta=unlabelled_meta,
                    lab_exclude=lab_exclude,
                    unlab_exclude=unlab_exclude,
                    predictions_dir=predictions_dir,
                    verify_n=args.verify_n,
                    tol=args.verify_tol,
                )
            )

        # --- the query-query half -------------------------------------------
        qq = (unlab_norm @ unlab_norm.T).astype(np.float32)
        # Guard-excluded queries are inert, not "identical to every other empty
        # file" (issue #66). Zero the row AND the column: a degenerate query
        # must neither seed a reviewer directory nor match one.
        if unlab_exclude.any():
            qq[unlab_exclude, :] = 0.0
            qq[:, unlab_exclude] = 0.0
        np.clip(qq, -1.0, 1.0, out=qq)
        qq16 = qq.astype(np.float16)

        meta = {
            "model": model_name,
            "model_slug": slug,
            "variant": variant,
            "layer": int(layer),
            "repr": repr_name,
            "pooling": pooling,
            "abtt": bool(apply_abtt),
            "D": best_D,
            "dim": int(unlab_norm.shape[1]),
            "n_queries": int(qq16.shape[0]),
            "n_excluded": int(unlab_exclude.sum()),
            "dtype": "float16",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labelled_bases": str(lab_path),
            "unlabelled_bases": str(unlab_path),
        }

        out_path = out_dir / f"qq_sim_{slug}.npz"
        np.savez(
            out_path,
            sim=qq16,
            file_ids=file_ids,
            excluded=unlab_exclude,
            meta=np.array(json.dumps(meta)),
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved {qq16.shape} float16 -> {out_path} ({size_mb:.1f} MB)")
        summary.append({**meta, "path": str(out_path), "size_mb": round(size_mb, 2)})

    if all_verifications:
        report = out_dir / f"qq_sim_verification_{variant}.csv"
        pd.DataFrame(all_verifications).to_csv(report, index=False)
        print(f"\nVerification report ({len(all_verifications)} checks) -> {report}")
        print("All checks matched the deployed predictions CSV within "
              f"{args.verify_tol:g}.")

    if summary:
        index_path = out_dir / "qq_sim_index.json"
        index_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Matrix index -> {index_path}")


if __name__ == "__main__":
    main()
