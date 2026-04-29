"""Sample N random query/candidate pairs from the phase9 test split per model.

This is the pair-list generator for the larger-sample attribution metrics rerun
the professor asked for (~200 pairs/model instead of the curated 20/model used
in phase12f).

Strategy: balanced random. For each featured model, draw N/2 random
gold-similar pairs (same folder, both winnable test files) and N/2 random
gold-dissimilar pairs (different folders, both in winnable test split). No
bucket stratification (that relied on a pre-scored pair frame we no longer
reproduce). The downstream NPZ generator, MaRC driver and metrics script all
work off this CSV schema.

Output schema mirrors phase12f_examples.csv closely enough for downstream
scripts. Score/bucket columns are left NaN because we don't need them for
NPZ generation — they would require re-running the similarity matrix.
"""
from __future__ import annotations

import argparse
from pathlib import Path  # noqa: F401 - used below via Path(p).exists()

import numpy as np
import pandas as pd

from attribution_model_config import (  # noqa: E402
    DEFAULT_MODELS,
    FEATURED_MODELS,
    methods_available_string,
    model_config,
    parse_layer_overrides,
)


def sample_pairs_for_model(
    test_meta: pd.DataFrame,
    n_positive: int,
    n_negative: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a dataframe with n_positive gold-similar + n_negative gold-dissimilar
    pairs, each row a query/candidate draw."""
    positives = _sample_positive(test_meta, n_positive, rng)
    negatives = _sample_negative(test_meta, n_negative, rng)
    pairs = pd.concat([positives, negatives], ignore_index=True)
    return pairs.sample(frac=1.0, random_state=rng.integers(0, 2**31)).reset_index(drop=True)


def _sample_positive(test_meta: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    groups = test_meta.groupby("folder_id")
    eligible_folders = [fid for fid, g in groups if len(g) >= 2]
    rows = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(rows) < n and attempts < n * 50:
        attempts += 1
        fid = eligible_folders[rng.integers(0, len(eligible_folders))]
        folder_rows = groups.get_group(fid).reset_index(drop=True)
        i, j = rng.choice(len(folder_rows), size=2, replace=False)
        q = folder_rows.iloc[int(i)]
        c = folder_rows.iloc[int(j)]
        key = (int(q["file_id"]), int(c["file_id"]))
        if key in seen or key[::-1] in seen:
            continue
        seen.add(key)
        rows.append((q, c))
    if len(rows) < n:
        raise SystemExit(f"Could not sample {n} positives; got {len(rows)} after {attempts} tries")
    return _rows_to_frame(rows, gold_similar=1)


def _sample_negative(test_meta: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    m = len(test_meta)
    while len(rows) < n and attempts < n * 50:
        attempts += 1
        i, j = rng.choice(m, size=2, replace=False)
        q = test_meta.iloc[int(i)]
        c = test_meta.iloc[int(j)]
        if q["folder_id"] == c["folder_id"]:
            continue
        key = (int(q["file_id"]), int(c["file_id"]))
        if key in seen or key[::-1] in seen:
            continue
        seen.add(key)
        rows.append((q, c))
    if len(rows) < n:
        raise SystemExit(f"Could not sample {n} negatives; got {len(rows)} after {attempts} tries")
    return _rows_to_frame(rows, gold_similar=0)


def _rows_to_frame(rows, gold_similar: int) -> pd.DataFrame:
    records = []
    for q, c in rows:
        records.append({
            "query_file_id": int(q["file_id"]),
            "candidate_file_id": int(c["file_id"]),
            "query_path": str(q["path"]),
            "candidate_path": str(c["path"]),
            "query_folder_id": str(q["folder_id"]),
            "candidate_folder_id": str(c["folder_id"]),
            "gold_similar": gold_similar,
        })
    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="runs/phase9/phase9_split.csv")
    ap.add_argument(
        "--out_csv",
        default="runs/active/ig_examples_200pair/random200_examples.csv",
    )
    ap.add_argument("--n_per_model", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument(
        "--models", nargs="*", default=DEFAULT_MODELS,
        help="Subset of FEATURED_MODELS to sample for.",
    )
    ap.add_argument(
        "--layer_overrides",
        nargs="*",
        default=None,
        help="Optional MODEL=LAYER overrides, e.g. google/mt5-base=4.",
    )
    args = ap.parse_args()
    try:
        layer_overrides = parse_layer_overrides(args.layer_overrides)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    split = pd.read_csv(args.split_csv)
    test = split[(split["split"] == "test") & split["is_winnable"]].reset_index(drop=True)
    # Rewrite legacy /u/irowerojas/localLatin/canon/ paths to the current
    # /projects/beto/irowerojas/localLatin/data/canon/ layout. The phase9 split
    # still records the old paths but the files themselves moved under data/.
    test["path"] = test["path"].str.replace(
        "/u/irowerojas/localLatin/canon/",
        "/projects/beto/irowerojas/localLatin/data/canon/",
        regex=False,
    )
    missing = sum(1 for p in test["path"] if not Path(p).exists())
    if missing:
        raise SystemExit(f"{missing}/{len(test)} test paths do not exist after rewrite; check layout")
    print(f"Test winnable pool: {len(test)} files across {test.folder_id.nunique()} folders")

    rng = np.random.default_rng(args.seed)

    frames = []
    example_id_offset = 0
    for model_name in args.models:
        if model_name not in FEATURED_MODELS:
            raise SystemExit(f"Unknown model: {model_name}")
        cfg = model_config(model_name, layer_overrides)
        pairs = sample_pairs_for_model(
            test,
            n_positive=args.n_per_model // 2,
            n_negative=args.n_per_model - args.n_per_model // 2,
            rng=rng,
        )
        pairs["model_name"] = model_name
        pairs["model_short"] = cfg["model_short"]
        pairs["model_type"] = cfg["model_type"]
        pairs["layer"] = cfg["layer"]
        pairs["method"] = "abtt_optimal"
        pairs["repr"] = "hidden"
        pairs["pooling"] = "mean"
        pairs["D"] = cfg["D"]
        pairs["tau"] = cfg["tau"]
        pairs["baseline_tau"] = cfg["baseline_tau"]
        pairs["abtt_tau"] = cfg["abtt_tau"]
        pairs["candidate_label"] = pairs["candidate_folder_id"]
        pairs["candidate_role"] = "pair_example"
        # Filler columns for schema compatibility with downstream scripts
        for col in [
            "query_index", "candidate_index", "baseline_score", "abtt_score",
            "baseline_pred", "abtt_pred", "baseline_disagrees", "baseline_margin",
            "abtt_margin", "baseline_margin_abs", "abtt_margin_abs", "bucket_slot",
        ]:
            pairs[col] = np.nan
        pairs["bucket"] = np.where(pairs["gold_similar"] == 1, "rand_similar", "rand_not_similar")
        pairs["methods_available"] = methods_available_string()
        pairs.insert(0, "example_id", np.arange(1, len(pairs) + 1) + example_id_offset)
        example_id_offset += len(pairs)
        frames.append(pairs)

    out = pd.concat(frames, ignore_index=True)
    # Reorder columns to match phase12f_examples.csv ordering as closely as reasonable
    preferred_order = [
        "example_id", "model_name", "model_short", "model_type", "layer",
        "method", "repr", "pooling", "D", "tau", "baseline_tau", "abtt_tau",
        "query_index", "candidate_index", "query_file_id", "candidate_file_id",
        "query_path", "candidate_path", "query_folder_id", "candidate_folder_id",
        "candidate_label", "candidate_role", "gold_similar", "baseline_score",
        "abtt_score", "baseline_pred", "abtt_pred", "bucket", "baseline_disagrees",
        "baseline_margin", "abtt_margin", "baseline_margin_abs", "abtt_margin_abs",
        "bucket_slot", "methods_available",
    ]
    out = out[[c for c in preferred_order if c in out.columns]]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} pairs across {out.model_name.nunique()} models -> {out_path}")
    print(out.groupby(["model_short", "bucket"]).size().rename("n").reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
