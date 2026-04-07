"""Select paper-facing Phase 12e examples from Task B human-simulation outputs.

For each featured T5 model, choose query cases where the correct directory is
not the model's first choice but is still in the top-k list. Emit two pair rows
per selected query:

1. the model's highest-ranked explainable directory representative
2. the representative from the correct directory option
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURED_MODELS = {
    "bowphs/LaTa": {
        "model_type": "t5",
        "featured_layer": 4,
    },
    "bowphs/PhilTa": {
        "model_type": "t5",
        "featured_layer": 6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Phase 12e pair-explanation examples.")
    parser.add_argument("--taskb_csv", required=True, help="taskb_top5_predictions.csv")
    parser.add_argument("--out_csv", required=True, help="Output example-pairs CSV")
    parser.add_argument(
        "--examples_per_model",
        type=int,
        default=10,
        help="Number of query cases to keep per featured model.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Expected top-k width from the Task B CSV.",
    )
    return parser.parse_args()


def _correct_option_fields(row: pd.Series, top_k: int) -> tuple[str, float, str, str]:
    correct_rank = int(row["correct_rank"])
    if correct_rank < 1 or correct_rank > top_k:
        raise ValueError("Correct option is not inside the saved top-k list.")
    prefix = f"rank{correct_rank}"
    return (
        str(row[f"{prefix}_label"]),
        float(row[f"{prefix}_score"]),
        str(row[f"{prefix}_rep_path"]),
        str(row[f"{prefix}_rep_folder_id"]),
    )


def _predicted_directory_fields(
    row: pd.Series,
    top_k: int,
) -> tuple[int, str, float, str, str] | None:
    for rank in range(1, top_k + 1):
        prefix = f"rank{rank}"
        label = str(row[f"{prefix}_label"])
        rep_path = row[f"{prefix}_rep_path"]
        if label == "__NEW__":
            continue
        if label == str(row["correct_label"]):
            continue
        if pd.isna(rep_path) or not str(rep_path).strip():
            continue
        return (
            rank,
            label,
            float(row[f"{prefix}_score"]),
            str(rep_path),
            str(row[f"{prefix}_rep_folder_id"]),
        )
    return None


def main() -> None:
    args = parse_args()
    taskb_df = pd.read_csv(args.taskb_csv)
    rows = []

    for model_name, cfg in FEATURED_MODELS.items():
        sub = taskb_df[
            (taskb_df["model"] == model_name)
            & (taskb_df["correct_rank"] >= 2)
            & (taskb_df["correct_rank"] <= args.top_k)
            & (taskb_df["correct_label"] != "__NEW__")
            & (taskb_df["rank1_label"] != taskb_df["correct_label"])
        ].copy()
        if sub.empty:
            continue

        sub["predicted_directory_option"] = [
            _predicted_directory_fields(row, args.top_k)
            for _, row in sub.iterrows()
        ]
        sub = sub[sub["predicted_directory_option"].notna()].copy()
        if sub.empty:
            continue

        sub["correct_score"] = [
            _correct_option_fields(row, args.top_k)[1]
            for _, row in sub.iterrows()
        ]
        sub = sub.sort_values(
            by=["correct_rank", "correct_score", "query_path"],
            ascending=[True, False, True],
        ).head(args.examples_per_model)

        for _, row in sub.iterrows():
            (
                predicted_rank,
                predicted_label,
                predicted_score,
                predicted_path,
                predicted_folder_id,
            ) = row["predicted_directory_option"]
            top1 = {
                "candidate_label": predicted_label,
                "candidate_score": predicted_score,
                "candidate_path": predicted_path,
                "candidate_folder_id": predicted_folder_id,
                "candidate_rank": int(predicted_rank),
                "candidate_role": "predicted_directory",
            }
            correct_label, correct_score, correct_path, correct_folder_id = _correct_option_fields(
                row, args.top_k
            )
            correct = {
                "candidate_label": correct_label,
                "candidate_score": correct_score,
                "candidate_path": correct_path,
                "candidate_folder_id": correct_folder_id,
                "candidate_rank": int(row["correct_rank"]),
                "candidate_role": "correct_option",
            }
            for item in (top1, correct):
                rows.append(
                    {
                        "model_name": model_name,
                        "model_type": cfg["model_type"],
                        "layer": int(cfg["featured_layer"]),
                        "method": str(row["method"]),
                        "repr": str(row["repr"]),
                        "pooling": str(row["pooling"]),
                        "D": int(row["D"]),
                        "tau": float(row["tau"]),
                        "query_index": int(row["query_index"]),
                        "query_path": str(row["query_path"]),
                        "query_folder_id": str(row["query_folder_id"]),
                        "correct_rank": int(row["correct_rank"]),
                        "correct_bucket": str(row["correct_bucket"]),
                        **item,
                    }
                )

    if not rows:
        raise SystemExit("No eligible Phase 12e examples were found.")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.insert(0, "example_id", range(1, len(out_df) + 1))
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} pair rows to {out_path}")


if __name__ == "__main__":
    main()
