"""Select 5 hardcoded examples from Phase 12e for the IG comparison study.

Reads the full Phase 12e examples CSV (40 rows) and filters to 5 specific
(example_id, candidate_role) pairs chosen for the Integrated Gradients
comparison in the resubmission.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# (example_id, candidate_role) -> short rationale
SELECTED = [
    (2, "correct_option", "True match, Can.apost.44 (LaTa)"),
    (8, "correct_option", "True match, CANT.328.2 (LaTa)"),
    (22, "correct_option", "True match, CNEO.315.11 (PhilTa, featured)"),
    (1, "predicted_directory", "Error case, wrong prediction (LaTa)"),
    (25, "predicted_directory", "Error case, wrong prediction (PhilTa)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Phase 12e examples for IG comparison study."
    )
    parser.add_argument(
        "--input_csv",
        default="runs/phase12e_release/phase12e_examples.csv",
        help="Phase 12e examples CSV (default: runs/phase12e_release/phase12e_examples.csv)",
    )
    parser.add_argument(
        "--out_dir",
        default="runs/phase_resubmit/ig_comparison",
        help="Output directory (default: runs/phase_resubmit/ig_comparison)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    out_dir = Path(args.out_dir)

    print(f"Reading {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total rows: {len(df)}")

    # Build lookup keys for filtering
    selected_keys = {(eid, role) for eid, role, _ in SELECTED}

    mask = df.apply(
        lambda row: (int(row["example_id"]), row["candidate_role"]) in selected_keys,
        axis=1,
    )
    filtered = df[mask].copy()

    # Verify all 5 were found
    found_keys = set(
        zip(filtered["example_id"].astype(int), filtered["candidate_role"])
    )
    for eid, role, rationale in SELECTED:
        if (eid, role) not in found_keys:
            print(f"  WARNING: missing example_id={eid}, candidate_role={role} ({rationale})")

    if len(filtered) < len(SELECTED):
        print(
            f"  WARNING: found {len(filtered)}/{len(SELECTED)} expected examples"
        )

    # Write output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selected_examples.csv"
    filtered.to_csv(out_path, index=False)
    print(f"\nWrote {len(filtered)} rows to {out_path}")

    # Summary
    print("\nSelected examples:")
    for _, row in filtered.iterrows():
        rationale = next(
            r for eid, role, r in SELECTED
            if eid == int(row["example_id"]) and role == row["candidate_role"]
        )
        print(
            f"  id={int(row['example_id']):>2}  role={row['candidate_role']:<22s}  "
            f"model={row['model_name']:<16s}  query={row['query_path']}  "
            f"[{rationale}]"
        )


if __name__ == "__main__":
    main()
