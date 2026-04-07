"""Index canon_unlabelled and generate meta CSVs for embedding extraction.

Outputs:
  runs/active/resubmit/unlabelled/meta_unlabelled.csv  — unlabelled file index
  runs/active/resubmit/data/meta.csv                    — labelled file index (from split CSV)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Index unlabelled canon files for extraction.")
    parser.add_argument("--unlabelled_root", default="data/canon_unlabelled", help="Root of unlabelled files.")
    parser.add_argument("--split_csv", default="runs/active/resubmit/data/phase_resubmit_split.csv",
                        help="Existing labelled split CSV.")
    parser.add_argument("--out_unlabelled", default="runs/active/resubmit/unlabelled/meta_unlabelled.csv")
    parser.add_argument("--out_labelled_meta", default="runs/active/resubmit/data/meta.csv")
    args = parser.parse_args()

    # --- Unlabelled ---
    root = Path(args.unlabelled_root)
    txt_files = sorted(root.glob("*.txt"), key=lambda p: p.name)
    rows = []
    for i, fp in enumerate(txt_files):
        rows.append({"file_id": i, "filename": fp.name, "path": str(fp)})
    df_unlab = pd.DataFrame(rows)

    out_path = Path(args.out_unlabelled)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_unlab.to_csv(out_path, index=False)
    print(f"Unlabelled meta: {len(df_unlab)} files -> {out_path}")

    # --- Labelled (from split CSV) ---
    split = pd.read_csv(args.split_csv)
    meta_labelled = split[["file_id", "path"]].copy()
    out_lab = Path(args.out_labelled_meta)
    out_lab.parent.mkdir(parents=True, exist_ok=True)
    meta_labelled.to_csv(out_lab, index=False)
    print(f"Labelled meta: {len(meta_labelled)} files -> {out_lab}")


if __name__ == "__main__":
    main()
