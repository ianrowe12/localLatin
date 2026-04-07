"""Persist 6 attribution methods x {baseline, abtt} per IG example pair.

Augments existing NPZ artifacts under
``runs/active/ig_examples/artifacts/<model_slug>/example*_pair_example.npz``
with 12 full-sequence pair matrices and 24 top-K index arrays per pair, all
computed from data already present in those NPZs (hidden states, attention
matrices, IG scores, PCs, mean vector). The 17 pre-existing keys are preserved
byte-identical.

Method x variant matrix:

    method                | baseline (raw hidden)         | abtt (cleaned hidden)
    ----------------------|-------------------------------|----------------------
    ig                    | IG scores: query_ig_baseline  | IG scores: query_ig_abtt
    bertscore             | -                             | -
    ot                    | IG scores: query_ig_baseline  | IG scores: query_ig_abtt
    attention_weighted    | (uses query_attention)        | (uses query_attention)
    dla                   | -                             | -
    attention_standalone  | (uses query_attention)        | (uses query_attention)

The "baseline" variant uses the raw (uncleaned) hidden states; the "abtt"
variant uses ``clean_tokens(...)``. Attention matrices come from the same NPZ
in both variants since they are not affected by ABTT cleaning.

Pair matrices are stored at the masked length ``(q_len, c_len)`` -- not the
padded length -- so downstream consumers do not need to re-mask.

Also rewrites ``phase12f_examples.csv`` with a new ``methods_available`` column
listing the comma-separated method names successfully computed for each row,
in canonical order. The webapp uses this column to render its method dropdown
without inspecting every NPZ.

Usage:
    python scripts/resubmit/persist_attribution_methods.py \\
        --examples_csv runs/active/ig_examples/phase12f_examples.csv \\
        --artifacts_dir runs/active/ig_examples/artifacts
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Import the 6 build_* functions and helpers from the existing comparison
# script. Its `if __name__ == "__main__": main()` guard at line 736 makes this
# safe -- nothing runs at import time.
sys.path.insert(0, str(Path(__file__).parent))
from run_resubmit_ig_comparison import (  # noqa: E402
    build_attention_crosssim,
    build_attention_standalone,
    build_bertscore_matrix,
    build_dla_pair_matrix,
    build_ig_pair_matrix,
    build_ot_transport_plan,
    clean_tokens,
    model_slug,
)


# Canonical method order. Must match the keys the webapp expects in
# `methods_available` and the `pair_matrix_<method>_<variant>` NPZ keys.
MAIN_METHODS: list[str] = [
    "ig",
    "bertscore",
    "ot",
    "attention_weighted",
    "dla",
    "attention_standalone",
]

VARIANTS: list[str] = ["baseline", "abtt"]

# The 17 keys that must remain present and byte-identical after rewriting.
ORIGINAL_KEYS: tuple[str, ...] = (
    "query_hidden",
    "candidate_hidden",
    "query_attention",
    "candidate_attention",
    "query_attention_mask",
    "candidate_attention_mask",
    "query_input_ids",
    "candidate_input_ids",
    "query_ig_baseline",
    "query_ig_abtt",
    "candidate_ig_baseline",
    "candidate_ig_abtt",
    "pcs",
    "mean_vec",
    "D",
    "layer",
    "example_id",
)


def topk_indices(M: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k query and candidate indices by row/col marginal of |M|.

    Returns ``(query_top, candidate_top)`` as int32 arrays in descending order.
    Lengths are ``min(k, n_q)`` and ``min(k, n_c)`` respectively.
    """
    abs_M = np.abs(M)
    q_marginal = abs_M.sum(axis=1)  # (n_q,)
    c_marginal = abs_M.sum(axis=0)  # (n_c,)
    k_q = min(k, q_marginal.shape[0])
    k_c = min(k, c_marginal.shape[0])
    q_top = np.argsort(q_marginal)[-k_q:][::-1]
    c_top = np.argsort(c_marginal)[-k_c:][::-1]
    return q_top.astype(np.int32), c_top.astype(np.int32)


def compute_pair_matrices(
    data: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Build 6 methods x 2 variants of pair matrices from NPZ data.

    Returns a dict ``{method_name: {variant: matrix}}``. Methods that fail are
    omitted from the returned dict; the caller decides whether to surface the
    failure.
    """
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())

    # Truncate everything to masked sequence length so the persisted matrices
    # contain no padding tokens.
    q_h = data["query_hidden"][:q_len].astype(np.float32, copy=False)
    c_h = data["candidate_hidden"][:c_len].astype(np.float32, copy=False)
    q_attn = data["query_attention"][:q_len, :q_len].astype(np.float32, copy=False)
    c_attn = data["candidate_attention"][:c_len, :c_len].astype(np.float32, copy=False)
    q_ig_b = data["query_ig_baseline"][:q_len].astype(np.float32, copy=False)
    q_ig_a = data["query_ig_abtt"][:q_len].astype(np.float32, copy=False)
    c_ig_b = data["candidate_ig_baseline"][:c_len].astype(np.float32, copy=False)
    c_ig_a = data["candidate_ig_abtt"][:c_len].astype(np.float32, copy=False)

    pcs = data["pcs"]
    mean_vec = data["mean_vec"]
    q_hc = clean_tokens(q_h, pcs, mean_vec).astype(np.float32, copy=False)
    c_hc = clean_tokens(c_h, pcs, mean_vec).astype(np.float32, copy=False)

    out: dict[str, dict[str, np.ndarray]] = {}

    # Each builder is wrapped individually so a single failure (e.g., POT
    # missing for OT, NaN cost) does not poison the rest.
    method_builders: list[tuple[str, callable, callable]] = [
        (
            "ig",
            lambda: build_ig_pair_matrix(q_h, c_h, q_ig_b, c_ig_b),
            lambda: build_ig_pair_matrix(q_hc, c_hc, q_ig_a, c_ig_a),
        ),
        (
            "bertscore",
            lambda: build_bertscore_matrix(q_h, c_h)[1],
            lambda: build_bertscore_matrix(q_hc, c_hc)[1],
        ),
        (
            "ot",
            lambda: build_ot_transport_plan(q_h, c_h, q_ig_b, c_ig_b),
            lambda: build_ot_transport_plan(q_hc, c_hc, q_ig_a, c_ig_a),
        ),
        (
            "attention_weighted",
            lambda: build_attention_crosssim(q_h, c_h, q_attn, c_attn),
            lambda: build_attention_crosssim(q_hc, c_hc, q_attn, c_attn),
        ),
        (
            "dla",
            lambda: build_dla_pair_matrix(q_h, c_h),
            lambda: build_dla_pair_matrix(q_hc, c_hc),
        ),
        (
            "attention_standalone",
            lambda: build_attention_standalone(q_h, c_h, q_attn, c_attn),
            lambda: build_attention_standalone(q_hc, c_hc, q_attn, c_attn),
        ),
    ]

    for method, build_baseline, build_abtt in method_builders:
        try:
            mat_b = np.asarray(build_baseline(), dtype=np.float32)
            mat_a = np.asarray(build_abtt(), dtype=np.float32)
            if mat_b.shape != (q_len, c_len) or mat_a.shape != (q_len, c_len):
                raise ValueError(
                    f"method {method}: expected ({q_len},{c_len}), got "
                    f"baseline={mat_b.shape} abtt={mat_a.shape}"
                )
            out[method] = {"baseline": mat_b, "abtt": mat_a}
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] method {method} failed: {exc}", file=sys.stderr)
    return out


def build_new_npz_keys(
    matrices: dict[str, dict[str, np.ndarray]],
    topk: int,
) -> dict[str, np.ndarray]:
    """Convert nested {method: {variant: M}} dict into flat NPZ key dict."""
    new_keys: dict[str, np.ndarray] = {}
    for method, variants in matrices.items():
        for variant, mat in variants.items():
            new_keys[f"pair_matrix_{method}_{variant}"] = mat
            q_top, c_top = topk_indices(mat, topk)
            new_keys[f"topk_{method}_{variant}_query"] = q_top
            new_keys[f"topk_{method}_{variant}_candidate"] = c_top
    return new_keys


def atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write NPZ atomically: savez to a sibling tmp file, then rename.

    The tmp filename ends in ``.npz`` so ``np.savez`` does not append another
    ``.npz`` extension (it only appends if the suffix is missing). Without
    this, ``path.with_suffix(".npz.tmp")`` would produce ``foo.npz.tmp`` but
    ``np.savez`` would write to ``foo.npz.tmp.npz``, breaking the rename.
    """
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def artifact_path_for_row(artifacts_dir: Path, row: pd.Series) -> Path:
    return (
        artifacts_dir
        / model_slug(row["model_name"])
        / f"example{int(row['example_id']):03d}_{row['candidate_role']}.npz"
    )


def process_row(
    row: pd.Series,
    artifacts_dir: Path,
    topk: int,
    dry_run: bool,
) -> tuple[list[str], str]:
    """Process a single CSV row.

    Returns ``(methods_succeeded, status)`` where status is one of
    ``"ok"``, ``"missing"``, ``"error"``.
    """
    path = artifact_path_for_row(artifacts_dir, row)
    if not path.exists():
        return [], "missing"

    try:
        existing = dict(np.load(path, allow_pickle=False))
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] failed to load {path}: {exc}", file=sys.stderr)
        return [], "error"

    # Sanity-check the originals before we touch anything.
    missing_originals = [k for k in ORIGINAL_KEYS if k not in existing]
    if missing_originals:
        print(
            f"  [ERROR] {path} missing original keys: {missing_originals}",
            file=sys.stderr,
        )
        return [], "error"

    matrices = compute_pair_matrices(existing)
    succeeded = [m for m in MAIN_METHODS if m in matrices]
    new_keys = build_new_npz_keys(matrices, topk)

    if dry_run:
        print(
            f"  [DRY] {path.name}: would write {len(new_keys)} new keys "
            f"({len(succeeded)}/{len(MAIN_METHODS)} methods OK)"
        )
        for k, v in new_keys.items():
            print(f"    {k}: shape={v.shape} dtype={v.dtype}")
        return succeeded, "ok"

    # Merge new keys into existing dict and atomically rewrite.
    merged = dict(existing)
    merged.update(new_keys)
    atomic_write_npz(path, merged)
    return succeeded, "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Persist 6 attribution methods x {baseline, abtt} per pair."
    )
    p.add_argument("--examples_csv", required=True, type=Path)
    p.add_argument("--artifacts_dir", required=True, type=Path)
    p.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-K query/candidate indices per method (default: 5).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute matrices but do not modify NPZs or CSV. "
        "If --max_rows is unset, only processes the first row.",
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Process at most N rows (for smoke testing).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    examples = pd.read_csv(args.examples_csv)
    print(f"Loaded {len(examples)} rows from {args.examples_csv}")

    if args.dry_run and args.max_rows is None:
        args.max_rows = 1
        print("Dry run: limiting to first row")

    methods_available_per_row: list[str] = []
    counts = {"ok": 0, "missing": 0, "error": 0}
    method_success_counts = {m: 0 for m in MAIN_METHODS}

    for idx, row in examples.iterrows():
        if args.max_rows is not None and counts["ok"] + counts["missing"] + counts["error"] >= args.max_rows:
            methods_available_per_row.append("")
            continue

        slug = model_slug(row["model_name"])
        tag = f"example{int(row['example_id']):03d}_{row['candidate_role']}"
        print(f"[{idx + 1}/{len(examples)}] {slug}/{tag}")

        try:
            succeeded, status = process_row(row, args.artifacts_dir, args.topk, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"  [ERROR] uncaught: {exc}", file=sys.stderr)
            succeeded, status = [], "error"

        counts[status] += 1
        for m in succeeded:
            method_success_counts[m] += 1
        methods_available_per_row.append(",".join(succeeded))

    print()
    print("=== Summary ===")
    print(f"  rows processed (ok):      {counts['ok']}")
    print(f"  rows skipped (missing):   {counts['missing']}")
    print(f"  rows failed (error):      {counts['error']}")
    print("  per-method success counts:")
    for m in MAIN_METHODS:
        print(f"    {m:22s}: {method_success_counts[m]}")

    if args.dry_run:
        print("\nDry run -- CSV not modified.")
        return

    # Pad the per-row list if --max_rows truncated processing.
    while len(methods_available_per_row) < len(examples):
        methods_available_per_row.append("")

    examples["methods_available"] = methods_available_per_row
    examples.to_csv(args.examples_csv, index=False)
    print(f"\nUpdated {args.examples_csv} with methods_available column")


if __name__ == "__main__":
    main()
