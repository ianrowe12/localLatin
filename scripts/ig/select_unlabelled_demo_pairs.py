"""Select attribution pair examples for *unlabelled* review-queue queries.

Every artifact under ``runs/active/ig_examples/artifacts/`` so far describes a
labelled canon pair from the paper's test split, so the webapp's Attribution
view almost never lines up with a live query (see
``docs/research/abigail_demo_script.md`` section 5). This script closes that gap
for a named set of demo queries by emitting rows in the existing
``phase12f_examples.csv`` schema that point at

    query      data/canon_unlabelled/<filename>
    candidate  data/canon_labelled/<dir>/<best file in dir>

``scripts/_archive/run_phase12e_pair_explanations.py`` reads ``query_path`` and
``candidate_path`` straight off disk, so it needs no change to consume these:
the whole unlabelled extension lives in this selection step.

Fidelity to the deployed ranking is the point, so every knob is taken from the
variant prediction CSVs the webapp actually serves rather than from the paper's
attribution layer contract:

* **Layer** comes from the ``layer`` column of the variant CSVs (asserted equal
  across all four variants for a given model).
* **Candidate file** is the argmax-cosine file inside the predicted directory,
  which is exactly the file ``predict_directories`` scored the directory by.
  The variant used for that argmax is the highest-priority variant under which
  the directory is top-1 (``sif_abtt`` > ``sif`` > ``abtt`` > ``raw``), and the
  resulting score is checked against ``rank1_score`` from the CSV.
* **D and the ABTT principal components** are refit here with the same
  ``find_optimal_D`` sweep and the same train-only ``EmbeddingCleaner`` fit that
  ``run_resubmit_unlabelled_retrieval.py`` performs, on the mean-pooled
  labelled train split. The token-level ``abtt`` variant in the artifact
  therefore removes the same directions the deployed ranking removed.

``gold_similar`` is 0 for every row: an unlabelled query has no gold directory.
``baseline_pred`` / ``abtt_pred`` are set truthfully -- 1 when the directory is
top-1 under ``raw`` / ``abtt`` respectively.

Two known gaps between the artifact and the deployed configuration:

1. **One cleaner per artifact.** The NPZ format holds a single
   ``(pcs, mean_vec)`` pair, so ``write_pcs`` persists the mean-pooled fit and
   the ``sif_abtt`` panel reweights those same cleaned states. Where the two
   poolings disagree on D -- LaTa layer 1 is mean D=10 against SIF D=3 -- the
   ``sif_abtt`` panel is a close but not identical subspace to the ranking it
   explains (principal-angle cosines 0.98/0.96/0.91). ``raw`` and ``abtt`` are
   exact. Fixing this needs per-pooling PCs in the artifact format.
2. **Token budget.** ``run_phase12e_pair_explanations.py`` truncates at
   ``--max_length`` 256 while the retrieval embeddings were pooled at 512, so a
   query longer than 256 tokens has an invisible tail in the attribution panel.
   ``C1525.56v.3.txt`` is 294 tokens, about 13% of it unseen; the other three
   demo queries are 51/144/147 and unaffected. Pass ``--max_length 512`` to the
   generator to close this at the cost of a few GPU-seconds.

Usage::

    python scripts/ig/select_unlabelled_demo_pairs.py \\
        --examples_csv runs/active/ig_examples/phase12f_examples.csv \\
        --pc_root runs/phase12_release/pcs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

from canon_retrieval import l2_normalize  # noqa: E402
from sif_abtt import EmbeddingCleaner  # noqa: E402

from run_resubmit_unlabelled_retrieval import (  # noqa: E402
    D_VALUES,
    VARIANTS,
    find_optimal_D,
    model_slug,
)

# Variant priority when a directory is top-1 under more than one variant.
# sif_abtt is the webapp default, so it wins.
VARIANT_PRIORITY = ["sif_abtt", "sif", "abtt", "raw"]

POOLING_SUBDIR = {"mean": "hidden_mean_tokempty", "sif": "hidden_sif_tokempty"}
# SIF-pooled caches carry a "_sif" suffix on the array filename; mean-pooled do not.
POOLING_SUFFIX = {"mean": "", "sif": "_sif"}

# The four queries in docs/research/abigail_demo_script.md, with the model each
# is demoed under.
DEFAULT_DEMO = [
    ("C1525.56v.3.txt", "bowphs/LaTa"),
    ("C1525.35r.9.txt", "bowphs/PhilTa"),
    ("BAV1341.16r.7.txt", "bowphs/PhilTa"),
    ("C1525.54r.2.txt", "bowphs/LaTa"),
]

BUCKET = "unlabelled_demo"
QUERY_SOURCE = "unlabelled"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Override the demo set as FILENAME=MODEL entries, e.g. "
        "'C1525.56v.3.txt=bowphs/LaTa'.",
    )
    p.add_argument(
        "--examples_csv",
        type=Path,
        default=REPO_ROOT / "runs/active/ig_examples/phase12f_examples.csv",
    )
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument(
        "--subset_csv",
        type=Path,
        default=None,
        help="Also write a CSV holding only the unlabelled-source rows. Feed this "
        "to run_phase12e_pair_explanations.py so it does not choke on the legacy "
        "rows, whose query_path predates the data/ move and no longer resolves.",
    )
    p.add_argument(
        "--unlabelled_root",
        type=Path,
        default=REPO_ROOT / "runs/active/resubmit/unlabelled",
    )
    p.add_argument(
        "--labelled_bases",
        type=Path,
        default=REPO_ROOT / "runs/active/resubmit_bases/phase9_bases",
    )
    p.add_argument(
        "--split_csv",
        type=Path,
        default=REPO_ROOT / "runs/active/resubmit/data/phase_resubmit_split.csv",
    )
    p.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--pc_root", type=Path, default=REPO_ROOT / "runs/phase12_release/pcs")
    p.add_argument(
        "--overwrite_pcs",
        action="store_true",
        help="Replace an existing PC file that holds a different fit, keeping a "
        "'.PRE_ISSUE53.npz' backup. Without this the run refuses, because "
        "--pc_root is shared with artifacts this run did not build.",
    )
    p.add_argument(
        "--results_csv",
        type=Path,
        default=REPO_ROOT / "runs/active/resubmit/results/phase_resubmit_results.csv",
    )
    p.add_argument(
        "--score_tolerance",
        type=float,
        default=5e-5,
        help="Max allowed |recomputed score - rank1_score| before failing. The "
        "default is the 5 decimal places the acceptance criterion claims. A "
        "looser bound hides exactly the class of bug this guard exists for: "
        "cleaning with the wrong pooling's D moved one pair by only 1.8e-3.",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def parse_queries(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_DEMO)
    out = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"--queries entry must be FILENAME=MODEL, got {raw!r}")
        fname, model = raw.split("=", 1)
        out.append((fname.strip(), model.strip()))
    return out


def load_variant_frames(unlabelled_root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for variant in VARIANTS:
        path = unlabelled_root / f"unlabelled_predictions_{variant}.csv"
        if not path.exists():
            raise SystemExit(f"Missing variant predictions CSV: {path}")
        frames[variant] = pd.read_csv(path)
    return frames


def embeddings(root: Path, slug: str, pooling: str, layer: int) -> np.ndarray:
    fname = f"hidden_layer{layer}_embeddings{POOLING_SUFFIX[pooling]}.npy"
    path = root / slug / POOLING_SUBDIR[pooling] / fname
    if not path.exists():
        raise SystemExit(f"Embedding cache missing: {path}")
    return np.load(path)


def pcs_match(path: Path, pcs: np.ndarray, mean_vec: np.ndarray) -> bool:
    """True when an existing PC file already holds this exact fit."""
    try:
        existing = np.load(path, allow_pickle=False)
        return (
            existing["pcs"].shape == pcs.shape
            and existing["mean_vec"].shape == mean_vec.shape
            and np.allclose(existing["pcs"], pcs, atol=1e-6)
            and np.allclose(existing["mean_vec"], mean_vec, atol=1e-6)
        )
    except Exception:  # noqa: BLE001  -- unreadable counts as "does not match"
        return False


def tau_for(results: pd.DataFrame, model: str, layer: int, method: str) -> float:
    sub = results[
        (results["model"] == model)
        & (results["layer"] == layer)
        & (results["method"] == method)
        & (results["repr"] == "hidden")
        & (results["pooling"] == "mean")
    ]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0]["tau"])


class ModelContext:
    """Per-model embeddings, ABTT cleaner and PC file, built once and reused."""

    def __init__(self, model: str, layer: int, args: argparse.Namespace, split: pd.DataFrame):
        self.model = model
        self.slug = model_slug(model)
        self.layer = layer
        self.split = split

        self.lab: dict[str, np.ndarray] = {}
        self.unlab: dict[str, np.ndarray] = {}
        for pooling in ("mean", "sif"):
            self.lab[pooling] = embeddings(args.labelled_bases, self.slug, pooling, layer)
            self.unlab[pooling] = embeddings(
                args.unlabelled_root / "bases", self.slug, pooling, layer
            )
            if self.lab[pooling].shape[0] != len(split):
                raise SystemExit(
                    f"[{self.slug}] labelled cache rows {self.lab[pooling].shape[0]} "
                    f"!= split rows {len(split)}"
                )

        # Same fit as run_resubmit_unlabelled_retrieval.main(): train split only,
        # and -- crucially -- once per pooling, because that script reloads
        # lab_emb with the variant's own pooling before sweeping D and fitting
        # the cleaner. abtt therefore cleans mean-pooled space and sif_abtt
        # cleans SIF-pooled space, with independently chosen D.
        train_mask = split["split"].to_numpy() == "train"
        train_folders = split.loc[train_mask, "folder_id"].to_numpy()

        self.D: dict[str, int] = {}
        self.cleaners: dict[str, EmbeddingCleaner] = {}
        self.lab_abtt: dict[str, np.ndarray] = {}
        self.unlab_abtt: dict[str, np.ndarray] = {}
        for pooling in ("mean", "sif"):
            train_emb = self.lab[pooling][train_mask]
            print(
                f"[{self.slug}] layer {layer} {pooling}-pooled: sweeping D on "
                f"{train_emb.shape[0]} train docs"
            )
            d = find_optimal_D(train_emb, train_folders, D_VALUES)
            print(f"[{self.slug}] optimal D ({pooling}) = {d}")
            cleaner = EmbeddingCleaner(num_components=d, center=True)
            cleaner.fit(train_emb)
            self.D[pooling] = d
            self.cleaners[pooling] = cleaner
            self.lab_abtt[pooling] = cleaner.transform(self.lab[pooling])
            self.unlab_abtt[pooling] = cleaner.transform(self.unlab[pooling])

    def write_pcs(self, pc_root: Path, dry_run: bool, overwrite: bool) -> Path:
        """Write the mean-pooled PCs, which is what the NPZ's abtt variant uses.

        The artifact stores one (pcs, mean_vec) pair and applies it to token
        vectors. Its ``abtt`` variant is the unweighted-aggregation one, so it
        takes the mean-pooled cleaner; ``sif_abtt`` layers SIF token weights on
        top of those same cleaned states, per the issue #46 convention.

        ``pc_root`` is shared with the paper pipeline and already holds files
        that back shipped artifacts (mT5-base and KaLM-mini at layer 1, LaTa L4,
        PhilTa L6). Since ``--queries`` accepts any model, a silent overwrite
        there would invalidate artifacts this run never looked at. So: write
        when absent, no-op when the existing file already matches, and refuse
        otherwise unless ``--overwrite_pcs``, which first takes a backup in the
        repo's own ``.PRE_*`` style.
        """
        out = pc_root / self.slug / f"layer{self.layer}_pcs.npz"
        cleaner = self.cleaners["mean"]
        pcs = np.asarray(cleaner.pcs, dtype=np.float32)
        mean_vec = np.asarray(cleaner.mean_vec, dtype=np.float32)

        if out.exists():
            if pcs_match(out, pcs, mean_vec):
                print(f"[{self.slug}] {out} already matches this fit; leaving it alone")
                return out
            if not overwrite:
                raise SystemExit(
                    f"[{self.slug}] refusing to overwrite {out}: it exists and holds a "
                    f"different fit. Other artifacts may depend on it. Re-run with "
                    f"--overwrite_pcs to replace it (a .PRE_ISSUE53 backup is kept), "
                    f"or point --pc_root somewhere else."
                )
            backup = out.with_suffix(".PRE_ISSUE53.npz")
            if dry_run:
                print(f"[{self.slug}] [DRY] would back up {out} -> {backup} and overwrite")
                return out
            if not backup.exists():
                backup.write_bytes(out.read_bytes())
                print(f"[{self.slug}] backed up {out} -> {backup}")

        if dry_run:
            print(f"[{self.slug}] [DRY] would write {out} pcs={pcs.shape}")
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, pcs=pcs, mean_vec=mean_vec)
        print(f"[{self.slug}] wrote {out} pcs={pcs.shape} mean={mean_vec.shape}")
        return out

    def vectors(self, variant: str) -> tuple[np.ndarray, np.ndarray]:
        pooling, apply_abtt, _ = VARIANTS[variant]
        lab = self.lab_abtt[pooling] if apply_abtt else self.lab[pooling]
        unlab = self.unlab_abtt[pooling] if apply_abtt else self.unlab[pooling]
        return l2_normalize(lab), l2_normalize(unlab)


def write_subset(df: pd.DataFrame, path: Path) -> None:
    """Write only the unlabelled-source rows, for the NPZ generator to consume."""
    if "query_source" not in df.columns:
        subset = df.iloc[0:0]
    else:
        subset = df[df["query_source"].astype(str) == QUERY_SOURCE]
    path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(path, index=False)
    print(f"Wrote {len(subset)} unlabelled-source rows to {path}")


def main() -> None:
    args = parse_args()
    demo = parse_queries(args.queries)

    existing = pd.read_csv(args.examples_csv)
    next_id = int(existing["example_id"].max()) + 1
    known = {
        (str(r["query_path"]), str(r["candidate_folder_id"]), str(r["model_name"]))
        for _, r in existing.iterrows()
    }

    frames = load_variant_frames(args.unlabelled_root)
    split = pd.read_csv(args.split_csv)
    results = pd.read_csv(args.results_csv)
    unlab_meta = pd.read_csv(args.unlabelled_root / "meta_unlabelled.csv")
    filename_to_row = {str(r["filename"]): i for i, r in unlab_meta.iterrows()}

    contexts: dict[str, ModelContext] = {}
    rows: list[dict] = []
    failures: list[str] = []

    for fname, model in demo:
        print(f"\n=== {fname} [{model}] ===")
        if fname not in filename_to_row:
            failures.append(f"{fname}: not in meta_unlabelled.csv")
            continue
        u_idx = filename_to_row[fname]
        file_id = int(unlab_meta.iloc[u_idx]["file_id"])

        # Collect the top-1 directory per variant, and the layer they all agree on.
        top1: dict[str, tuple[str, float]] = {}
        layers = set()
        for variant, df in frames.items():
            sub = df[(df["filename"] == fname) & (df["model"] == model)]
            if sub.empty:
                failures.append(f"{fname}/{model}: no row in {variant} predictions")
                continue
            r = sub.iloc[0]
            top1[variant] = (str(r["rank1_dir"]), float(r["rank1_score"]))
            layers.add(int(r["layer"]))
        if not top1:
            continue
        if len(layers) != 1:
            failures.append(f"{fname}/{model}: variants disagree on layer: {sorted(layers)}")
            continue
        layer = layers.pop()

        key = f"{model}@{layer}"
        if key not in contexts:
            contexts[key] = ModelContext(model, layer, args, split)
        ctx = contexts[key]

        dir_to_variants: dict[str, list[str]] = {}
        for variant, (d, _) in top1.items():
            dir_to_variants.setdefault(d, []).append(variant)

        for cand_dir, variants in dir_to_variants.items():
            pick = next(v for v in VARIANT_PRIORITY if v in variants)
            lab_norm, unlab_norm = ctx.vectors(pick)
            member_idx = np.where(split["folder_id"].to_numpy() == cand_dir)[0]
            if member_idx.size == 0:
                failures.append(f"{fname}/{model}: directory {cand_dir} not in split")
                continue
            sims = lab_norm[member_idx] @ unlab_norm[u_idx]
            best = int(member_idx[int(np.argmax(sims))])
            score = float(np.max(sims))
            expected = top1[pick][1]
            delta = abs(score - expected)
            flag = "OK " if delta <= args.score_tolerance else "MISMATCH"
            print(
                f"  {cand_dir:16s} top1 under {','.join(variants):22s} "
                f"file={split.iloc[best]['filename']:22s} "
                f"score={score:.4f} vs csv[{pick}]={expected:.4f} d={delta:.5f} {flag}"
            )
            if delta > args.score_tolerance:
                failures.append(
                    f"{fname}/{model}/{cand_dir}: recomputed {score:.6f} != "
                    f"csv {expected:.6f} (variant {pick})"
                )
                continue

            cand_path = args.data_root / "canon_labelled" / cand_dir / str(split.iloc[best]["filename"])
            query_path = args.data_root / "canon_unlabelled" / fname
            for p in (cand_path, query_path):
                if not p.is_file():
                    failures.append(f"{fname}/{model}: missing text file {p}")
            if (str(query_path), cand_dir, model) in known:
                print(f"    [skip] already present in {args.examples_csv}")
                continue

            rows.append(
                {
                    "model_name": model,
                    "model_short": "LaTa" if "LaTa" in model else "PhilTa",
                    "model_type": "t5",
                    "layer": layer,
                    "method": "abtt_optimal",
                    "repr": "hidden",
                    "pooling": "mean",
                    "D": ctx.D["mean"],
                    "tau": tau_for(results, model, layer, "abtt_optimal"),
                    "baseline_tau": tau_for(results, model, layer, "baseline"),
                    "abtt_tau": tau_for(results, model, layer, "abtt_optimal"),
                    "query_index": -1,
                    "candidate_index": best,
                    "query_file_id": file_id,
                    "candidate_file_id": int(split.iloc[best]["file_id"]),
                    "query_path": str(query_path),
                    "candidate_path": str(cand_path),
                    "query_folder_id": "",
                    "candidate_folder_id": cand_dir,
                    "candidate_label": cand_dir,
                    "candidate_role": "pair_example",
                    "gold_similar": 0,
                    "baseline_score": top1.get("raw", ("", float("nan")))[1],
                    "abtt_score": top1.get("abtt", ("", float("nan")))[1],
                    "baseline_pred": int(top1.get("raw", ("", 0))[0] == cand_dir),
                    "abtt_pred": int(top1.get("abtt", ("", 0))[0] == cand_dir),
                    "bucket": BUCKET,
                    "query_source": QUERY_SOURCE,
                    "top1_variants": ",".join(sorted(variants)),
                }
            )

    if failures:
        print("\n=== FAILURES ===", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        raise SystemExit(1)

    if not rows:
        print("\nNothing to add.")
        if args.subset_csv and not args.dry_run:
            write_subset(existing, args.subset_csv)
        return

    new_df = pd.DataFrame(rows)
    new_df.insert(0, "example_id", np.arange(next_id, next_id + len(new_df), dtype=np.int32))

    # Existing rows predate query_source/top1_variants; backfill them as labelled.
    combined_existing = existing.copy()
    if "query_source" not in combined_existing.columns:
        combined_existing["query_source"] = "labelled"
    else:
        combined_existing["query_source"] = combined_existing["query_source"].fillna("labelled")
    if "top1_variants" not in combined_existing.columns:
        combined_existing["top1_variants"] = ""

    for col in combined_existing.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[list(combined_existing.columns)]
    combined = pd.concat([combined_existing, new_df], ignore_index=True)

    print("\n=== NEW ROWS ===")
    print(
        new_df[
            ["example_id", "model_short", "layer", "D", "query_path", "candidate_folder_id", "top1_variants"]
        ].to_string(index=False)
    )

    for ctx in contexts.values():
        ctx.write_pcs(args.pc_root, args.dry_run, args.overwrite_pcs)

    if args.dry_run:
        print(f"\n[dry-run] would write {len(combined)} rows ({len(new_df)} new)")
        return

    out_csv = args.out_csv or args.examples_csv
    combined.to_csv(out_csv, index=False)
    print(f"\nWrote {len(combined)} rows ({len(new_df)} new) to {out_csv}")
    if args.subset_csv:
        write_subset(combined, args.subset_csv)


if __name__ == "__main__":
    main()
