"""Phase 9 Experiment 2: ROUGE-L + BLEU-1 lexical diagnostics.

Computes word-level ROUGE-L (F1) and BLEU-1 (unigram precision) from scratch
for all test pairs across 5 languages. No external NLP dependencies.

Output: runs/phase9/experiment2/diagnostics.csv
  columns: language, sentence_1, sentence_2, similarity, rouge_l, bleu_1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


LANGUAGES = ["english", "french", "sinhala", "tamil", "serbian"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 Exp 2 lexical diagnostics.")
    parser.add_argument("--data_dir", required=True, help="Dir with {lang}_split.csv files.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--languages", default=",".join(LANGUAGES),
        help="Comma-separated language names.",
    )
    return parser.parse_args()


def tokenize_words(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenization."""
    return text.lower().split()


def lcs_length(a: List[str], b: List[str]) -> int:
    """Compute length of the longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(hyp: str, ref: str) -> float:
    """Word-level ROUGE-L F1 score."""
    hyp_tokens = tokenize_words(hyp)
    ref_tokens = tokenize_words(ref)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_1(hyp: str, ref: str) -> float:
    """Unigram precision (BLEU-1 without brevity penalty)."""
    hyp_tokens = tokenize_words(hyp)
    ref_tokens = tokenize_words(ref)
    if not hyp_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    clipped = 0
    for t in hyp_tokens:
        if ref_counts.get(t, 0) > 0:
            clipped += 1
            ref_counts[t] -= 1
    return clipped / len(hyp_tokens)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    all_rows = []

    for language in languages:
        print(f"Processing {language}...")
        csv_path = data_dir / f"{language}_split.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping.")
            continue
        df = pd.read_csv(csv_path)
        test = df[df["split"] == "test"].copy()
        print(f"  Test pairs: {len(test)}")

        for _, row in test.iterrows():
            s1 = str(row["sentence_1"])
            s2 = str(row["sentence_2"])
            all_rows.append({
                "language": language,
                "sentence_1": s1,
                "sentence_2": s2,
                "similarity": row["similarity"],
                "rouge_l": rouge_l(s1, s2),
                "bleu_1": bleu_1(s1, s2),
            })

    result = pd.DataFrame(all_rows)
    out_path = out_dir / "diagnostics.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} rows to {out_path}")

    # Quick summary
    for lang in languages:
        subset = result[result["language"] == lang]
        if subset.empty:
            continue
        print(f"  {lang}: ROUGE-L mean={subset['rouge_l'].mean():.3f}, "
              f"BLEU-1 mean={subset['bleu_1'].mean():.3f}")


if __name__ == "__main__":
    main()
