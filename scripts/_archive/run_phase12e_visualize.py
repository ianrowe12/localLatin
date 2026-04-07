"""Render Phase 12e derived pair matrices and companion attention figures."""
from __future__ import annotations

import argparse
from pathlib import Path
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


FEATURE_MODEL = "bowphs/PhilTa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Phase 12e artifacts.")
    parser.add_argument("--examples_csv", required=True, help="CSV from run_phase12e_select_examples.py")
    parser.add_argument("--artifacts_dir", required=True, help="artifacts/ root from run_phase12e_pair_explanations.py")
    parser.add_argument("--fig_dir", required=True, help="Output figure directory")
    parser.add_argument("--max_tokens", type=int, default=18, help="Maximum query/candidate tokens to display")
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def load_artifact(artifacts_dir: Path, row: pd.Series) -> dict[str, np.ndarray]:
    path = artifacts_dir / model_slug(row["model_name"]) / f"example{int(row['example_id']):03d}_{row['candidate_role']}.npz"
    return dict(np.load(path))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a_norm @ b_norm.T


def clean_tokens(hidden: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    centered = hidden - mean_vec
    proj = centered @ pcs.T @ pcs
    return centered - proj


def select_positions(ig_base: np.ndarray, ig_abtt: np.ndarray, seq_len: int, max_tokens: int) -> np.ndarray:
    importance = np.maximum(np.abs(ig_base[:seq_len]), np.abs(ig_abtt[:seq_len]))
    top_idx = np.argsort(importance)[-max_tokens:]
    return np.sort(top_idx)


def decode_tokens(tokenizer, input_ids: np.ndarray, positions: np.ndarray) -> list[str]:
    ids = input_ids[0, positions].tolist()
    return [tokenizer.decode([token_id]) for token_id in ids]


def normalize_token(token: str) -> str:
    return token.strip().replace("▁", "").replace("Ġ", "").replace("</s>", "")


def is_punctuation_token(token: str) -> bool:
    text = normalize_token(token)
    return bool(text) and all(ch in string.punctuation for ch in text)


def is_empty_token(token: str) -> bool:
    return normalize_token(token) == ""


def pretty_role(role: str) -> str:
    if role == "correct_option":
        return "Correct directory"
    if role == "predicted_directory":
        return "Predicted directory"
    return role.replace("_", " ")


def build_pair_matrix(
    query_hidden: np.ndarray,
    candidate_hidden: np.ndarray,
    query_ig: np.ndarray,
    candidate_ig: np.ndarray,
) -> np.ndarray:
    cos = cosine_matrix(query_hidden, candidate_hidden)
    weight = np.sqrt(
        np.abs(query_ig)[:, None] * np.abs(candidate_ig)[None, :]
    )
    sign = np.sign(query_ig)[:, None] * np.sign(candidate_ig)[None, :]
    return cos * weight * sign


def compute_display_metrics(
    data: dict[str, np.ndarray],
    tokenizer,
    max_tokens: int,
) -> dict[str, float]:
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    q_pos = select_positions(data["query_ig_baseline"], data["query_ig_abtt"], q_len, max_tokens)
    c_pos = select_positions(
        data["candidate_ig_baseline"],
        data["candidate_ig_abtt"],
        c_len,
        max_tokens,
    )
    q_tokens = decode_tokens(tokenizer, data["query_input_ids"], q_pos)
    c_tokens = decode_tokens(tokenizer, data["candidate_input_ids"], c_pos)
    q_abtt = np.abs(data["query_ig_abtt"][q_pos])
    c_abtt = np.abs(data["candidate_ig_abtt"][c_pos])
    weights = np.concatenate([q_abtt, c_abtt])
    tokens = q_tokens + c_tokens
    punct_mask = np.array([is_punctuation_token(token) for token in tokens], dtype=bool)
    empty_mask = np.array([is_empty_token(token) for token in tokens], dtype=bool)
    total_weight = float(weights.sum())
    punct_mass_frac = float(weights[punct_mask].sum() / total_weight) if total_weight else 0.0
    empty_mass_frac = float(weights[empty_mask].sum() / total_weight) if total_weight else 0.0
    return {
        "punct_mass_frac": punct_mass_frac,
        "empty_mass_frac": empty_mass_frac,
        "punct_count": int(punct_mask.sum()),
        "empty_count": int(empty_mask.sum()),
    }


def render_pair_heatmaps(
    row: pd.Series,
    data: dict[str, np.ndarray],
    tokenizer,
    out_path: Path,
    max_tokens: int,
    paper_mode: bool = False,
) -> float:
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    q_pos = select_positions(data["query_ig_baseline"], data["query_ig_abtt"], q_len, max_tokens)
    c_pos = select_positions(
        data["candidate_ig_baseline"],
        data["candidate_ig_abtt"],
        c_len,
        max_tokens,
    )

    q_tokens = decode_tokens(tokenizer, data["query_input_ids"], q_pos)
    c_tokens = decode_tokens(tokenizer, data["candidate_input_ids"], c_pos)

    q_hidden = data["query_hidden"][q_pos]
    c_hidden = data["candidate_hidden"][c_pos]
    q_hidden_abtt = clean_tokens(data["query_hidden"], data["pcs"], data["mean_vec"])[q_pos]
    c_hidden_abtt = clean_tokens(data["candidate_hidden"], data["pcs"], data["mean_vec"])[c_pos]

    q_ig_base = data["query_ig_baseline"][q_pos]
    q_ig_abtt = data["query_ig_abtt"][q_pos]
    c_ig_base = data["candidate_ig_baseline"][c_pos]
    c_ig_abtt = data["candidate_ig_abtt"][c_pos]

    baseline_matrix = build_pair_matrix(q_hidden, c_hidden, q_ig_base, c_ig_base)
    abtt_matrix = build_pair_matrix(q_hidden_abtt, c_hidden_abtt, q_ig_abtt, c_ig_abtt)
    vmax = np.percentile(np.abs(np.concatenate([baseline_matrix.ravel(), abtt_matrix.ravel()])), 95)
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, matrix, title in zip(
        axes,
        [baseline_matrix, abtt_matrix],
        ["Baseline", "ABTT"],
    ):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(c_tokens)))
        ax.set_xticklabels(c_tokens, rotation=90, fontsize=7, fontfamily="monospace")
        ax.set_yticks(range(len(q_tokens)))
        ax.set_yticklabels(q_tokens, fontsize=7, fontfamily="monospace")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Candidate")
        ax.set_ylabel("Query")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    if not paper_mode:
        fig.suptitle(
            f"{row['model_name']} L{int(row['layer'])} | example {int(row['example_id'])} | "
            f"{pretty_role(row['candidate_role'])} | correct rank {int(row['correct_rank'])}",
            fontweight="bold",
            fontsize=12,
        )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return float(np.mean(np.abs(q_ig_abtt)) + np.mean(np.abs(c_ig_abtt)) - np.mean(np.abs(q_ig_base)) - np.mean(np.abs(c_ig_base)))


def render_attention(
    row: pd.Series,
    data: dict[str, np.ndarray],
    tokenizer,
    out_path: Path,
    max_tokens: int,
    paper_mode: bool = False,
) -> None:
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    q_pos = select_positions(data["query_ig_baseline"], data["query_ig_abtt"], q_len, max_tokens)
    c_pos = select_positions(
        data["candidate_ig_baseline"],
        data["candidate_ig_abtt"],
        c_len,
        max_tokens,
    )
    q_tokens = decode_tokens(tokenizer, data["query_input_ids"], q_pos)
    c_tokens = decode_tokens(tokenizer, data["candidate_input_ids"], c_pos)
    q_attn = data["query_attention"][np.ix_(q_pos, q_pos)]
    c_attn = data["candidate_attention"][np.ix_(c_pos, c_pos)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, matrix, labels, title in zip(
        axes,
        [q_attn, c_attn],
        [q_tokens, c_tokens],
        ["Query Self-Attention", "Candidate Self-Attention"],
    ):
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7, fontfamily="monospace")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
        ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    if not paper_mode:
        fig.suptitle(
            f"{row['model_name']} L{int(row['layer'])} | example {int(row['example_id'])} | "
            f"{pretty_role(row['candidate_role'])}",
            fontweight="bold",
            fontsize=12,
        )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)
    artifacts_dir = Path(args.artifacts_dir)
    fig_dir = Path(args.fig_dir)
    review_dir = fig_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    tokenizers = {}
    manifest_rows = []
    for _, row in examples.iterrows():
        model_name = row["model_name"]
        if model_name not in tokenizers:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        data = load_artifact(artifacts_dir, row)
        pair_path = review_dir / f"example{int(row['example_id']):03d}_{row['candidate_role']}_pair.png"
        attn_path = review_dir / f"example{int(row['example_id']):03d}_{row['candidate_role']}_attn.png"
        lift = render_pair_heatmaps(row, data, tokenizers[model_name], pair_path, args.max_tokens)
        render_attention(row, data, tokenizers[model_name], attn_path, args.max_tokens)
        display_metrics = compute_display_metrics(data, tokenizers[model_name], args.max_tokens)
        manifest_rows.append(
            {
                "example_id": int(row["example_id"]),
                "model_name": model_name,
                "candidate_role": row["candidate_role"],
                "correct_rank": int(row["correct_rank"]),
                "pair_lift": lift,
                "punct_mass_frac": display_metrics["punct_mass_frac"],
                "empty_mass_frac": display_metrics["empty_mass_frac"],
                "display_quality": (
                    lift
                    - 0.2 * display_metrics["punct_mass_frac"]
                    - 0.2 * display_metrics["empty_mass_frac"]
                ),
                "pair_path": str(pair_path),
                "attention_path": str(attn_path),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(fig_dir / "phase12e_review_manifest.csv", index=False)

    feature = manifest_df[
        (manifest_df["model_name"] == FEATURE_MODEL)
        & (manifest_df["candidate_role"] == "correct_option")
    ].sort_values(["display_quality", "pair_lift"], ascending=False)
    if feature.empty:
        raise SystemExit("No PhilTa correct-option example was available for the final paper figures.")

    best = feature.iloc[0]
    best_row = examples[
        (examples["example_id"] == best["example_id"])
        & (examples["candidate_role"] == best["candidate_role"])
    ].iloc[0]
    best_data = load_artifact(artifacts_dir, best_row)
    render_pair_heatmaps(
        best_row,
        best_data,
        tokenizers[best_row["model_name"]],
        fig_dir / "fig_pair_matrix_philta.png",
        args.max_tokens,
        paper_mode=True,
    )
    render_attention(
        best_row,
        best_data,
        tokenizers[best_row["model_name"]],
        fig_dir / "fig_attention_philta.png",
        args.max_tokens,
        paper_mode=True,
    )
    print(f"Saved review figures to {review_dir}")
    print(f"Saved final figures to {fig_dir}")


if __name__ == "__main__":
    main()
