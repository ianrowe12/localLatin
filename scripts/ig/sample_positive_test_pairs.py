"""Sample N gold-similar (positive) test pairs per model for the attribution rerun.

Sibling of ``scripts/ig/sample_random_test_pairs.py`` but draws **only**
positive (same-folder, both winnable) pairs. The 100 positive + 100 negative
mix produced by the random sampler hides per-bucket attribution behavior; this
script gives a clean 200-positive sample so attribution-quality metrics are
computed exclusively on the regime where attribution should be most informative.

Default models are scoped to LaTa + PhilTa, matching Run 1 of the resubmit
pipeline. ``FEATURED_MODELS`` (layer / D / tau / baseline_tau / abtt_tau) is
copied verbatim from ``sample_random_test_pairs.py`` so the two scripts stay in
lockstep; if Agent 1.1's cosine investigation prescribes a different layer,
update both files in the same commit.

Output schema mirrors ``runs/active/ig_examples_200pair/random200_examples.csv``
exactly so the downstream pipeline (IG NPZ generation, MaRC, persist methods,
attribution metrics) reads it without changes. Every row has
``gold_similar=1`` and ``bucket="rand_similar"``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Copied verbatim from scripts/ig/sample_random_test_pairs.py. Keep in sync.
FEATURED_MODELS = {
    "bowphs/LaTa": {
        "model_short": "LaTa",
        "model_type": "t5",
        "layer": 4,
        "D": 2,
        # tau/baseline_tau/abtt_tau from runs/active/ig_examples/phase12f_examples.csv
        "tau": 0.5628140703517588,
        "baseline_tau": 0.9296482412060302,
        "abtt_tau": 0.5628140703517588,
    },
    "bowphs/PhilTa": {
        "model_short": "PhilTa",
        "model_type": "t5",
        "layer": 6,
        "D": 10,
        "tau": 0.4623115577889447,
        "baseline_tau": 0.9748743718592964,
        "abtt_tau": 0.4623115577889447,
    },
    "sentence-transformers/LaBSE": {
        "model_short": "LaBSE",
        "model_type": "bert",
        "layer": 12,
        "D": 1,
        "tau": 0.5829145728643216,
        "baseline_tau": 0.9195979899497488,
        "abtt_tau": 0.5829145728643216,
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "model_short": "Qwen3-0.6B",
        "model_type": "decoder",
        "layer": 23,
        "D": 3,
        "tau": 0.5226130653266332,
        "baseline_tau": 0.984924623115578,
        "abtt_tau": 0.5226130653266332,
    },
}


DEFAULT_MODELS = ["bowphs/LaTa", "bowphs/PhilTa"]


def _sample_positive(test_meta: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n unique gold-similar (same-folder) pairs from the test pool."""
    groups = test_meta.groupby("folder_id")
    eligible_folders = [fid for fid, g in groups if len(g) >= 2]
    if not eligible_folders:
        raise SystemExit("No folders with >=2 winnable test files; nothing to sample")
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
    return _rows_to_frame(rows)


def _rows_to_frame(rows) -> pd.DataFrame:
    records = []
    for q, c in rows:
        records.append({
            "query_file_id": int(q["file_id"]),
            "candidate_file_id": int(c["file_id"]),
            "query_path": str(q["path"]),
            "candidate_path": str(c["path"]),
            "query_folder_id": str(q["folder_id"]),
            "candidate_folder_id": str(c["folder_id"]),
            "gold_similar": 1,
        })
    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="runs/phase9/phase9_split.csv")
    ap.add_argument(
        "--out_csv",
        default="runs/active/ig_examples_200pos/positive200_examples.csv",
    )
    ap.add_argument("--n_per_model", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument(
        "--models", nargs="*", default=DEFAULT_MODELS,
        help="Subset of FEATURED_MODELS to sample for. Default: LaTa + PhilTa.",
    )
    args = ap.parse_args()

    split = pd.read_csv(args.split_csv)
    test = split[(split["split"] == "test") & split["is_winnable"]].reset_index(drop=True)
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
        cfg = FEATURED_MODELS[model_name]
        pairs = _sample_positive(test, args.n_per_model, rng)
        # Deterministic shuffle so artifact filenames don't preserve the sample order.
        pairs = pairs.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
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
        for col in [
            "query_index", "candidate_index", "baseline_score", "abtt_score",
            "baseline_pred", "abtt_pred", "baseline_disagrees", "baseline_margin",
            "abtt_margin", "baseline_margin_abs", "abtt_margin_abs", "bucket_slot",
        ]:
            pairs[col] = np.nan
        pairs["bucket"] = "rand_similar"
        pairs["methods_available"] = "ig,bertscore,ot,attention_weighted,dla,attention_standalone,retrieval_mark"
        pairs.insert(0, "example_id", np.arange(1, len(pairs) + 1) + example_id_offset)
        example_id_offset += len(pairs)
        frames.append(pairs)

    out = pd.concat(frames, ignore_index=True)
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
